#!/usr/bin/env python3
"""Fixed-protocol GPU pass measurement for VulkanEngine.

The renderer does parse a command line (see ``src/core/CommandLine.cpp``), but
none of its flags reach the toggles an A/B varies -- there is no general
``--set`` for a runtime settings key -- so feature configuration still comes
from ``config/runtime_settings.json``. GPU timings arrive as a once-per-second
``GPU timings:`` block on stdout, the only machine-readable source of per-pass
GPU time. This harness wraps that into a repeatable protocol so a pass timing
can be quoted as evidence:

- patch only named settings keys, then restore the user's file afterwards;
- discard a warm-up window instead of trusting the first frames;
- report medians over many samples, never a single frame;
- interleave A/B/A/B and refuse the comparison when the repeated control drifts.

Scene and camera default to the renderer's launch defaults, which is what makes
separate runs comparable. Scene presets are not persisted settings, so ``--set``
cannot reach them; ``--args`` passes the renderer's own flags through instead,
identically to both sides of an A/B so the scene is never the variable.

Reach for it when the control keeps drifting: the drift gate assumes the machine
holds a steady clock, and the default scene is light enough on a fast discrete
GPU that it does not -- the part idles between frames and the boost clock wanders
more than the effect being measured.

A heavier preset is NOT the remedy on a laptop, whatever this docstring used to
say. Measured on an RTX 3080 Ti Laptop: drift grew with load (1.4% on ``stress``,
9.6% at 4K), because the card throttles rather than settling at a steady boost.
Pin the clocks instead -- ``tools/dev/gpu_clock.ps1 lock`` -- which took the same
comparison to 0.30%. A pin is a ceiling, not a floor, so a heavier scene needs a
*lower* pin; see docs/profiling.md. ``CONTROL_DRIFT_LIMIT`` is itself calibrated
on hardware, not universal.

Usage:
    tools/dev/measure_gpu.py run   [--set k=v ...] [--label NAME] [--args ...]
    tools/dev/measure_gpu.py ab    [--a-set k=v ...] --b-set k=v [...] [--args ...]
    tools/dev/measure_gpu.py parse LOG [LOG ...] [--warmup-blocks N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO_ROOT / "config" / "runtime_settings.json"
# The suffix is not cosmetic: ensure_binary() gates every run on
# RELEASE_BINARY.exists(), so a bare name on Windows fails as "missing binary,
# rerun with --build" no matter how recently it was built.
RELEASE_BINARY = REPO_ROOT / "build" / "release" / ("VulkanEngine.exe" if os.name == "nt" else "VulkanEngine")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "measurements"

# Timings are printed once per second, so these are also sample counts.
DEFAULT_WARMUP_SECONDS = 10
DEFAULT_DURATION_SECONDS = 30

# Minimum samples worth reporting a median over.
MIN_SAMPLES = 8

# Idle time after a build, before the first run. A parallel build leaves the
# machine hot and the first control run would absorb all of it.
DEFAULT_SETTLE_SECONDS = 90

# Control drift above this fraction invalidates an A/B comparison: the machine
# moved more than the effect being measured.
CONTROL_DRIFT_LIMIT = 0.01

# docs/profiling.md: scopes recorded between vkCmdBeginRendering and
# vkCmdEndRendering read near zero on tile-based hardware regardless of the work
# they contain. They are parsed so the log stays faithful, but never compared.
UNRELIABLE_SCOPES = frozenset({"Skybox", "RenderObjects", "SkinnedMesh"})

# A pass that runs every frame appears in every block. Anything below this is
# conditional -- the depth pyramid, for instance, is built in 1-2 blocks out of
# 29 while occlusion culling is suspended. Its median is the cost of a rare
# frame, not of the configuration, and whether it appears at all is luck.
INTERMITTENT_COVERAGE = 0.9

# A pass present in only one configuration has no A/B delta to test its control
# drift against, so the drift is compared with the pass's own median instead.
# Above this fraction the value is not stable enough to quote: SSRTrace once
# reported 0.158 ms as an "A only" row while moving 0.137 ms between the two
# control runs, and it was read as a real cost for an hour.
UNSTABLE_DRIFT_FRACTION = 0.25

FRAME_TOTAL = "Frame total"

BLOCK_START_RE = re.compile(r"^(?:\[\w+\s*\]\s+)?GPU timings:\s*$")
SCOPE_RE = re.compile(r"^ {2,}(.+?):\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*$")
QUERY_LIMIT_RE = re.compile(r"^ {2,}warning: timestamp query capacity was exceeded\s*$")


class MeasureError(RuntimeError):
    """A protocol violation that must abort rather than produce a number."""


def rel(path: Path) -> str:
    """Repo-relative path for messages, falling back to the absolute path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Log parsing
# --------------------------------------------------------------------------


@dataclass
class Samples:
    """Per-scope timing samples collected from one or more runs."""

    label: str
    scopes: dict[str, list[float]] = field(default_factory=dict)
    block_count: int = 0
    query_limit_exceeded: bool = False
    validation_errors: list[str] = field(default_factory=list)
    effective_settings: dict = field(default_factory=dict)

    def add(self, other: "Samples") -> None:
        for name, values in other.scopes.items():
            self.scopes.setdefault(name, []).extend(values)
        self.block_count += other.block_count
        self.query_limit_exceeded |= other.query_limit_exceeded
        self.validation_errors.extend(other.validation_errors)
        if not self.effective_settings:
            self.effective_settings = other.effective_settings

    def median(self, scope: str) -> float | None:
        values = self.scopes.get(scope)
        return statistics.median(values) if values else None

    def sample_count(self, scope: str) -> int:
        return len(self.scopes.get(scope, ()))


def parse_log(text: str, label: str, warmup_blocks: int) -> Samples:
    """Extract ``GPU timings:`` blocks, dropping the first ``warmup_blocks``."""
    result = Samples(label=label)
    blocks: list[dict[str, float]] = []
    current: dict[str, float] | None = None

    for line in text.splitlines():
        if BLOCK_START_RE.match(line):
            if current is not None:
                blocks.append(current)
            current = {}
            continue

        if current is None:
            if "error" in line.lower() and "validation" in line.lower():
                result.validation_errors.append(line.strip())
            continue

        if QUERY_LIMIT_RE.match(line):
            result.query_limit_exceeded = True
            continue

        scope_match = SCOPE_RE.match(line)
        if scope_match:
            current[scope_match.group(1).strip()] = float(scope_match.group(2))
            continue

        # "timestamp queries: 42/256" and similar indented non-timing lines stay
        # inside the block; anything unindented ends it.
        if line.startswith("  "):
            continue

        blocks.append(current)
        current = None
        if "error" in line.lower() and "validation" in line.lower():
            result.validation_errors.append(line.strip())

    if current is not None:
        blocks.append(current)

    kept = blocks[warmup_blocks:]
    result.block_count = len(kept)
    for block in kept:
        for name, value in block.items():
            result.scopes.setdefault(name, []).append(value)

    return result


# --------------------------------------------------------------------------
# Settings patching
# --------------------------------------------------------------------------


def parse_assignment(assignment: str) -> tuple[str, str]:
    if "=" not in assignment:
        raise MeasureError(f"--set expects dotted.key=value, got: {assignment!r}")
    key, value = assignment.split("=", 1)
    key = key.strip()
    if not key:
        raise MeasureError(f"--set has an empty key: {assignment!r}")
    return key, value.strip()


def coerce_like(existing: object, raw: str, key: str) -> object:
    """Coerce a text value to the type already stored at ``key``.

    Typing is taken from the settings file rather than guessed, so a bool key
    can never silently become the string "true".
    """
    if isinstance(existing, bool):
        lowered = raw.lower()
        if lowered in ("true", "1", "on", "yes"):
            return True
        if lowered in ("false", "0", "off", "no"):
            return False
        raise MeasureError(f"{key} is a boolean; cannot use {raw!r}")
    if isinstance(existing, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise MeasureError(f"{key} is an integer; cannot use {raw!r}") from exc
    if isinstance(existing, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise MeasureError(f"{key} is a number; cannot use {raw!r}") from exc
    return raw


def apply_settings(settings: dict, assignments: list[str]) -> dict[str, object]:
    """Apply dotted assignments in place; return the effective values.

    An unknown key aborts. A typo that silently created a new key would measure
    the unchanged configuration twice and read as "no effect".
    """
    applied: dict[str, object] = {}
    for assignment in assignments:
        key, raw = parse_assignment(assignment)
        parts = key.split(".")
        node = settings
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise MeasureError(f"unknown settings key: {key} (no {part!r} section)")
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise MeasureError(f"unknown settings key: {key} (check config/runtime_settings.json)")
        value = coerce_like(node[leaf], raw, key)
        node[leaf] = value
        applied[key] = value
    return applied


# --------------------------------------------------------------------------
# Running the renderer
# --------------------------------------------------------------------------


def newer_sources() -> list[Path]:
    """Source files modified after the Release binary was linked."""
    if not RELEASE_BINARY.exists():
        return []
    binary_mtime = RELEASE_BINARY.stat().st_mtime
    stale: list[Path] = []
    for pattern in ("*.cpp", "*.h", "*.vert", "*.frag", "*.comp", "*.glsl"):
        for path in (REPO_ROOT / "src").rglob(pattern):
            if path.stat().st_mtime > binary_mtime:
                stale.append(path)
    cmake_lists = REPO_ROOT / "CMakeLists.txt"
    if cmake_lists.exists() and cmake_lists.stat().st_mtime > binary_mtime:
        stale.append(cmake_lists)
    return stale


def ensure_binary(build: bool, settle: int) -> None:
    if build:
        print("[measure] build Release renderer", flush=True)
        subprocess.run(
            ["cmake", "--build", "--preset", "release", "--parallel"],
            cwd=REPO_ROOT,
            check=True,
        )
        if settle > 0:
            # A parallel build heats the machine, and the first control run would
            # absorb all of it. Measured: an identical series drifted 0.41% with a
            # cold start and 28.5% when it began right after a build.
            print(f"[measure] settle {settle}s after the build", flush=True)
            time.sleep(settle)

    if not RELEASE_BINARY.exists():
        raise MeasureError(
            f"missing {rel(RELEASE_BINARY)}; run with --build "
            "or 'cmake --build --preset release --parallel'. Debug timings are not evidence."
        )

    # A stale binary produces a clean number for the wrong code. This one already
    # cost a series: SSRTrace read 0.805 ms on a binary 17 commits behind and
    # 0.158 ms once rebuilt, so the stale answer was over three times too large.
    stale = newer_sources()
    if stale:
        listed = "\n".join(f"  {rel(path)}" for path in sorted(stale)[:8])
        more = f"\n  ... and {len(stale) - 8} more" if len(stale) > 8 else ""
        raise MeasureError(
            f"{len(stale)} source files are newer than "
            f"{rel(RELEASE_BINARY)}:\n{listed}{more}\n"
            "Rerun with --build. Measuring a stale binary answers the wrong question."
        )


def run_once(
    label: str,
    assignments: list[str],
    warmup: int,
    duration: int,
    output_dir: Path,
    binary_args: list[str] | None = None,
) -> Samples:
    """Launch the renderer once with patched settings and collect samples."""
    if duration <= warmup + MIN_SAMPLES:
        raise MeasureError(
            f"--duration {duration}s leaves fewer than {MIN_SAMPLES} samples after a "
            f"{warmup}s warm-up; raise --duration."
        )

    original = SETTINGS_PATH.read_bytes() if SETTINGS_PATH.exists() else None
    settings = json.loads(original) if original else {}
    applied = apply_settings(settings, assignments)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{label}.log"

    try:
        if assignments:
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")
        described = ", ".join(f"{k}={v}" for k, v in applied.items()) or "persisted settings"
        print(f"[measure] run {label}: {described}", flush=True)
        print(f"[measure]   warm-up {warmup}s, sample {duration - warmup}s", flush=True)

        with log_path.open("wb") as log_file:
            # Both flags mean the same thing -- put the renderer in its own
            # process group so terminate() can take down anything it spawned --
            # but each is rejected outright on the other platform, so they
            # cannot be passed unconditionally.
            if os.name == "nt":
                group_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            else:
                group_kwargs = {"start_new_session": True}
            process = subprocess.Popen(
                [str(RELEASE_BINARY), *(binary_args or [])],
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                **group_kwargs,
            )
            try:
                exited = wait_or_timeout(process, duration)
            finally:
                if process.poll() is None:
                    terminate(process)

        if exited is not None and exited != 0:
            raise MeasureError(
                f"renderer exited early with code {exited} after less than {duration}s; "
                f"see {rel(log_path)}"
            )
    finally:
        if original is not None:
            SETTINGS_PATH.write_bytes(original)

    samples = parse_log(log_path.read_text(errors="replace"), label, warmup)
    # Recorded so the run can be reproduced later: configuration A is "whatever
    # was persisted that day", and that file is per-user state outside git.
    samples.effective_settings = settings
    if samples.block_count < MIN_SAMPLES:
        raise MeasureError(
            f"{label}: only {samples.block_count} timing blocks survived the warm-up "
            f"(need {MIN_SAMPLES}). Is the GPU profiler available? "
            f"See {rel(log_path)}"
        )
    print(f"[measure]   {samples.block_count} samples", flush=True)
    return samples


def wait_or_timeout(process: subprocess.Popen, duration: int) -> int | None:
    """Wait ``duration`` seconds; return the exit code if it exited early."""
    try:
        return process.wait(timeout=duration)
    except subprocess.TimeoutExpired:
        return None


def terminate(process: subprocess.Popen) -> None:
    # Windows has no SIGTERM delivery and no process groups to signal: both
    # send_signal(SIGTERM) and kill() land on the same TerminateProcess, and
    # os.killpg does not exist. So the escalation below is POSIX-only, and the
    # Windows path is the single hard kill that is all the OS offers here.
    if os.name == "nt":
        process.kill()
        process.wait(timeout=10)
        return

    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait(timeout=10)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def is_intermittent(count: int, total: int) -> bool:
    return total > 0 and count < total * INTERMITTENT_COVERAGE


def format_single(samples: Samples, release_run: bool = True) -> str:
    lines = [
        f"## {samples.label}",
        "",
        f"{samples.block_count} samples, medians in ms.",
        "",
        "| Pass | Median | Min | Max | Blocks |",
        "| --- | --- | --- | --- | --- |",
    ]
    intermittent: list[str] = []
    for name in ordered_scopes(samples.scopes):
        values = samples.scopes[name]
        note = "  *(nested: reads ~0, not a breakdown)*" if name in UNRELIABLE_SCOPES else ""
        coverage = f"{len(values)}/{samples.block_count}"
        if is_intermittent(len(values), samples.block_count):
            coverage = f"**{coverage}**"
            intermittent.append(name)
        lines.append(
            f"| {name}{note} | {statistics.median(values):.3f} | "
            f"{min(values):.3f} | {max(values):.3f} | {coverage} |"
        )
    if intermittent:
        lines.extend(
            [
                "",
                f"**Intermittent passes:** {', '.join(intermittent)}. These did not run "
                "in every sampled frame, so their median is the cost of the frames that "
                "did run them, not a per-frame cost of this configuration.",
            ]
        )
    lines.extend(caveats(samples, release_run))
    return "\n".join(lines)


def scope_control_drift(first_a: Samples, last_a: Samples) -> dict[str, float]:
    """Absolute median movement of each scope between the two control runs.

    A frame-level control that returns says nothing about a sub-millisecond
    pass: the composite sharpen filter once read 0.416 vs 0.424 ms at frame
    level while the pass itself tripled. Each scope needs its own noise floor.
    """
    drift: dict[str, float] = {}
    for name in first_a.scopes:
        baseline = first_a.median(name)
        repeated = last_a.median(name)
        if baseline is not None and repeated is not None:
            drift[name] = abs(repeated - baseline)
    return drift


def format_comparison(
    a: Samples,
    b: Samples,
    drift: float | None,
    scope_drift: dict[str, float] | None = None,
) -> str:
    scope_drift = scope_drift or {}
    lines = [
        f"## {a.label} vs {b.label}",
        "",
        f"Medians in ms over {a.block_count} and {b.block_count} samples.",
        "",
        "| Pass | A | B | Delta | Delta % | Control drift | Attributable | Blocks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    names = ordered_scopes({**a.scopes, **b.scopes})
    buried: list[str] = []
    intermittent: list[str] = []
    unstable: list[str] = []
    for name in names:
        median_a = a.median(name)
        median_b = b.median(name)
        count_a = a.sample_count(name)
        count_b = b.sample_count(name)
        coverage = f"{count_a}/{a.block_count} · {count_b}/{b.block_count}"
        # Judge coverage only on the sides where the pass runs at all. A pass that
        # is genuinely absent from one configuration -- SSRTrace with SSR off --
        # scores 0 there, and counting that as low coverage would label a real
        # presence difference as sampling luck, the opposite of the mistake this
        # check exists to prevent.
        coverages = [
            (count, total)
            for count, total in ((count_a, a.block_count), (count_b, b.block_count))
            if count > 0
        ]
        rare = any(is_intermittent(count, total) for count, total in coverages)
        if rare:
            coverage = f"**{coverage}**"
            intermittent.append(name)

        if name in UNRELIABLE_SCOPES:
            lines.append(
                f"| {name} *(nested: reads ~0, not a breakdown)* | "
                f"{fmt(median_a)} | {fmt(median_b)} | - | - | - | no | {coverage} |"
            )
            continue
        if median_a is None or median_b is None:
            # "A only" reads as a real presence difference, but a pass that runs
            # in 2 of 29 frames on both sides lands here purely by sampling luck.
            appeared = "B only" if median_a is None else "A only"
            if rare:
                appeared += ", intermittent"
            # There is no delta to test the drift against, so test it against the
            # pass's own median. Hiding the drift here is what let a 0.158 ms row
            # be quoted while its control moved 0.137 ms between runs.
            own_drift = scope_drift.get(name)
            present = median_a if median_b is None else median_b
            if own_drift is None:
                drift_text, verdict = "no control", "-"
            else:
                drift_text = f"{own_drift:.3f}"
                if present and own_drift > present * UNSTABLE_DRIFT_FRACTION:
                    verdict = "**unstable**"
                    unstable.append(name)
                else:
                    verdict = "-"
            lines.append(
                f"| {name} | {fmt(median_a)} | {fmt(median_b)} | {appeared} | - "
                f"| {drift_text} | {verdict} | {coverage} |"
            )
            continue
        delta = median_b - median_a
        percent = (delta / median_a * 100.0) if median_a else 0.0
        own_drift = scope_drift.get(name)
        if own_drift is None:
            drift_text, verdict = "-", "unchecked"
        elif abs(delta) <= own_drift:
            drift_text, verdict = f"{own_drift:.3f}", "**no**"
            buried.append(name)
        else:
            drift_text, verdict = f"{own_drift:.3f}", "yes"
        if rare:
            verdict = "**no**"
        lines.append(
            f"| {name} | {median_a:.3f} | {median_b:.3f} | {delta:+.3f} | "
            f"{percent:+.1f}% | {drift_text} | {verdict} | {coverage} |"
        )

    lines.append("")
    if unstable:
        lines.append(
            f"**Unstable values:** {', '.join(unstable)}. These run in only one "
            "configuration, so there is no delta to check; instead their own median "
            "moved more than a quarter of itself between the two control runs. The "
            "number is not reproducible from one run to the next -- do not quote it "
            "as the pass's cost."
        )
        lines.append("")
    if intermittent:
        lines.append(
            f"**Intermittent passes:** {', '.join(intermittent)}. These did not run in "
            "every sampled frame, so an 'A only' label is sampling luck rather than a "
            "presence difference, and the median is the cost of the frames that did run "
            "them. Not comparable between configurations."
        )
        lines.append("")
    if buried:
        lines.append(
            f"**Inside the noise floor:** {', '.join(buried)}. The control moved at "
            "least as much as the change did, so these rows are not evidence of an "
            "effect in either direction. Report them as not measured, not as no cost."
        )
        lines.append("")
    if drift is None:
        lines.append(
            "**Control not repeated.** Run with `--repeat 2` or higher to prove the "
            "machine held still; a single A/B pair is not evidence."
        )
    elif drift > CONTROL_DRIFT_LIMIT:
        lines.append(
            f"**Control drifted {drift * 100:.1f}%** between the first and last "
            f"{a.label} run, above the {CONTROL_DRIFT_LIMIT * 100:.0f}% limit. The machine "
            "moved during the series -- do not quote this comparison. Let it settle and rerun."
        )
    else:
        lines.append(
            f"Control drift {drift * 100:.2f}% (limit {CONTROL_DRIFT_LIMIT * 100:.0f}%): "
            "the repeated control returned, so the deltas above are attributable to the change."
        )

    merged = Samples(label="combined")
    merged.add(a)
    merged.add(b)
    lines.extend(caveats(merged, release_run=True))
    return "\n".join(lines)


def caveats(samples: Samples, release_run: bool) -> list[str]:
    lines: list[str] = []
    if release_run:
        # CMakeLists.txt compiles VULKAN_ENGINE_ENABLE_VALIDATION=0 outside Debug,
        # and this harness only runs Release. Silence here means the layers were
        # absent, not that the frame was clean -- say so rather than let a reader
        # infer a validation result the run could not produce.
        lines.extend(
            [
                "",
                "Validation layers are compiled out of Release, so this run cannot "
                "report validation errors. It is not evidence of a clean frame.",
            ]
        )
    if samples.query_limit_exceeded:
        lines.extend(
            [
                "",
                "**Timestamp query capacity was exceeded during this run**, so some "
                "scopes are missing. Raise the profiler query budget before trusting "
                "the pass list.",
            ]
        )
    if samples.validation_errors:
        unique = list(dict.fromkeys(samples.validation_errors))[:5]
        lines.extend(["", "**Validation errors in the log:**"])
        lines.extend(f"- `{line}`" for line in unique)
    lines.extend(
        [
            "",
            "Scene and camera are at launch defaults. Absolute numbers late in a "
            "session run thermally inflated -- compare within a series, not across "
            "sessions.",
        ]
    )
    return lines


def ordered_scopes(scopes: dict[str, list[float]]) -> list[str]:
    """Frame total first, then descending cost, so the expensive passes lead."""

    def sort_key(name: str) -> tuple[int, float]:
        if name == FRAME_TOTAL:
            return (0, 0.0)
        values = scopes.get(name) or [0.0]
        return (1, -statistics.median(values))

    return sorted(scopes, key=sort_key)


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def write_summary(output_dir: Path, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    return summary_path


def samples_to_json(samples: Samples) -> dict:
    return {
        "label": samples.label,
        "sample_count": samples.block_count,
        "query_limit_exceeded": samples.query_limit_exceeded,
        "medians_ms": {
            name: round(statistics.median(values), 4) for name, values in samples.scopes.items()
        },
    }


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    ensure_binary(args.build, args.settle)
    output_dir = Path(args.out)
    samples = run_once(args.label, args.set, args.warmup, args.duration, output_dir, args.args)
    print()
    print(format_single(samples))
    summary = write_summary(
        output_dir,
        {
            "mode": "run",
            "set": args.set,
            "effective_settings": samples.effective_settings,
            "runs": [samples_to_json(samples)],
        },
    )
    print(f"\nLogs: {output_dir}\nSummary: {summary}")
    return 0


def cmd_ab(args: argparse.Namespace) -> int:
    if not args.b_set:
        raise MeasureError("ab requires at least one --b-set; A is the unchanged control.")
    ensure_binary(args.build, args.settle)
    output_dir = Path(args.out)

    pooled_a = Samples(label=args.a_label)
    pooled_b = Samples(label=args.b_label)
    first_a: Samples | None = None
    last_a: Samples | None = None

    # A/B/A/B, not AA/BB: interleaving keeps a slow thermal ramp from landing
    # entirely on one configuration.
    for index in range(1, args.repeat + 1):
        run_a = run_once(
            f"{args.a_label}-{index}", args.a_set, args.warmup, args.duration, output_dir, args.args
        )
        pooled_a.add(run_a)
        first_a = first_a or run_a
        last_a = run_a

        run_b = run_once(
            f"{args.b_label}-{index}", args.b_set, args.warmup, args.duration, output_dir, args.args
        )
        pooled_b.add(run_b)

    drift = None
    scope_drift: dict[str, float] = {}
    if args.repeat >= 2 and first_a is not None and last_a is not None:
        baseline = first_a.median(FRAME_TOTAL)
        repeated = last_a.median(FRAME_TOTAL)
        if baseline and repeated:
            drift = abs(repeated - baseline) / baseline
        scope_drift = scope_control_drift(first_a, last_a)

    print()
    print(format_comparison(pooled_a, pooled_b, drift, scope_drift))
    summary = write_summary(
        output_dir,
        {
            "mode": "ab",
            "repeat": args.repeat,
            "control_drift": drift,
            "control_drift_limit": CONTROL_DRIFT_LIMIT,
            "scope_control_drift_ms": {k: round(v, 4) for k, v in sorted(scope_drift.items())},
            "a_set": args.a_set,
            "b_set": args.b_set,
            "effective_settings_a": pooled_a.effective_settings,
            "effective_settings_b": pooled_b.effective_settings,
            "runs": [samples_to_json(pooled_a), samples_to_json(pooled_b)],
        },
    )
    print(f"\nLogs: {output_dir}\nSummary: {summary}")
    return 1 if drift is not None and drift > CONTROL_DRIFT_LIMIT else 0


def cmd_parse(args: argparse.Namespace) -> int:
    pooled = Samples(label=args.label)
    for log in args.logs:
        path = Path(log)
        if not path.exists():
            raise MeasureError(f"no such log: {log}")
        pooled.add(parse_log(path.read_text(errors="replace"), args.label, args.warmup_blocks))
    if pooled.block_count == 0:
        raise MeasureError("no 'GPU timings:' blocks found; was the log captured from stdout?")
    if pooled.block_count < MIN_SAMPLES:
        print(
            f"[measure] warning: only {pooled.block_count} samples "
            f"(want at least {MIN_SAMPLES})",
            file=sys.stderr,
        )
    # A hand-captured log may come from a Debug build, where validation layers
    # are compiled in, so the Release-only caveat would be wrong here.
    print(format_single(pooled, release_run=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="measure_gpu.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--warmup",
            type=int,
            default=DEFAULT_WARMUP_SECONDS,
            help=f"seconds discarded before sampling (default {DEFAULT_WARMUP_SECONDS})",
        )
        target.add_argument(
            "--duration",
            type=int,
            default=DEFAULT_DURATION_SECONDS,
            help=f"total seconds per launch (default {DEFAULT_DURATION_SECONDS})",
        )
        target.add_argument(
            "--out",
            default=str(DEFAULT_OUTPUT_DIR),
            help=f"directory for logs and summary (default {DEFAULT_OUTPUT_DIR})",
        )
        target.add_argument(
            "--build",
            action="store_true",
            help="build the Release preset before measuring",
        )
        # Scene is a launch flag, not a persisted setting, so --set cannot reach
        # it. Without this the harness can only ever measure the default scene --
        # and on a fast GPU that scene is too light to hold the clocks steady,
        # which shows up as control drift rather than as a signal.
        # Applied identically to A and B, so it never becomes the variable.
        target.add_argument(
            "--args",
            nargs=argparse.REMAINDER,
            default=[],
            help="arguments passed through to the renderer (e.g. --args --scene stress)",
        )
        target.add_argument(
            "--settle",
            type=int,
            default=DEFAULT_SETTLE_SECONDS,
            help=(
                "seconds to idle after --build before the first run "
                f"(default {DEFAULT_SETTLE_SECONDS}); build heat otherwise lands "
                "entirely on the first control run"
            ),
        )

    run = sub.add_parser("run", help="measure one configuration")
    run.add_argument("--label", default="run", help="name used for the log file")
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted runtime_settings.json key to override (repeatable)",
    )
    add_common(run)
    run.set_defaults(func=cmd_run)

    ab = sub.add_parser("ab", help="interleaved A/B/A/B comparison with a control-drift gate")
    ab.add_argument(
        "--a-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="overrides for configuration A (default: persisted settings)",
    )
    ab.add_argument(
        "--b-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="overrides for configuration B (required)",
    )
    ab.add_argument("--a-label", default="A", help="label for configuration A")
    ab.add_argument("--b-label", default="B", help="label for configuration B")
    ab.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="A/B pairs to run; 2 or more is required to check control drift (default 2)",
    )
    add_common(ab)
    ab.set_defaults(func=cmd_ab)

    parse_cmd = sub.add_parser(
        "parse", help="summarize logs captured by hand (needed for ImGui-loaded scenes)"
    )
    parse_cmd.add_argument("logs", nargs="+", help="log files containing 'GPU timings:' blocks")
    parse_cmd.add_argument("--label", default="captured", help="label for the report")
    parse_cmd.add_argument(
        "--warmup-blocks",
        type=int,
        default=DEFAULT_WARMUP_SECONDS,
        help=f"leading blocks to discard (default {DEFAULT_WARMUP_SECONDS})",
    )
    parse_cmd.set_defaults(func=cmd_parse)

    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except MeasureError as error:
        print(f"[measure] error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"[measure] build failed with code {error.returncode}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[measure] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    if shutil.which("cmake") is None:
        print("[measure] warning: cmake not found; --build is unavailable", file=sys.stderr)
    sys.exit(main(sys.argv[1:]))
