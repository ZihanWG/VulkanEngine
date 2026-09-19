#!/usr/bin/env python3
"""Assert that the headless configuration sweep still covers what it claims to.

`.github/workflows/headless-render.yml` renders one leg per configuration, and
`config/ci/README.md` says why: a pass that is declared and never recorded leaves
the render graph's barriers, resource lifetimes and pass culling all describing
work that did not happen, and no validation error is raised. The sweep is what
catches that -- but only in the configurations it actually runs.

Nothing measured which configurations those were. The leg list grew one entry at
a time as features shipped, so its coverage was whatever had accumulated, and the
gap was invisible: a sweep of twenty-nine green legs looks the same whether it
exercises every code path or half of them.

It was less than half. Measured 2026-09-19 against the tracked schema: twenty-one
leg files moved twenty-one of 137 settings keys off their defaults, and of the 49
settings that select a code path, 30 were only ever rendered one way. Two legs
set a key to the value it already had, so they rendered the default under
another name.

One of the thirty was a live defect. `renderer.useGpuCulling = false` -- the CPU
culling fallback -- tripped the backstop on every frame, with
`MainGpuCullingPass` declared and never recorded, and had done for as long as
the graph had sequenced the frame.

So this counts instead of trusting. Three failures, each one a way the sweep can
quietly stop covering something:

1. A boolean or enum setting that no leg ever moves off its default. Both sides
   of a toggle select different code; a sweep that only ever renders one of them
   is not testing the other.
2. A leg whose every key already equals the default. It renders the default
   configuration under a name that promises something else -- `taa-off.json` did
   exactly this, so the TAA resolve pass had never run in CI at all.
3. A leg file and the workflow disagreeing about which legs exist.

Scalars are out of scope on purpose: `csm.shadowDistance` takes a range, not two
sides, and a leg per value would be a sweep of arbitrary numbers. What this
guards is the settings that select a path.

Run from anywhere; paths resolve relative to the repository root.
Exit status is 0 when the sweep covers everything, 1 otherwise.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The example file, not config/runtime_settings.json: that one is git-ignored,
# is whatever the last person to run the engine happened to save, and on the
# machine this check was written on it was twenty keys behind. The example is
# tracked and structurally pinned to RuntimeSettings' own defaults by
# test_runtime_settings.cpp, which writes a fresh document and diffs the
# flattened key paths against it -- so it is the schema, and tools/dev
# /measure_gpu.py already treats it as one.
DEFAULTS_PATH = REPO_ROOT / "config" / "runtime_settings.example.json"
LEGS_DIR = REPO_ROOT / "config" / "ci"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "headless-render.yml"

# Settings a leg would not usefully cover, with the reason kept here rather than
# in a reviewer's head. An entry is a dotted prefix; everything under it is
# exempt. Keep this list short and specific -- it is the one place coverage can
# be given away, so each line has to say what it gives away and why.
EXEMPT_PREFIXES: dict[str, str] = {
    # ImGui panel visibility. These decide whether a window is drawn in the
    # debug UI, which the headless sweep does not render at all (the renderer
    # runs with the overlay off unless --capture-include-ui asks for it), so a
    # leg would render a byte-identical frame under a different name.
    "debugUi.": "ImGui panel visibility; the headless sweep renders no debug UI",
}

# Settings a leg reaches through a command-line flag rather than a settings file,
# because the renderer applies them before it reads one. Listed rather than
# exempted: these keys ARE covered, and a table says by what. The left-hand side
# is the leg's argument string exactly as the sweep writes it, so a renamed or
# deleted leg stops matching and the keys go back to being uncovered.
FLAG_COVERAGE: dict[str, list[str]] = {
    # --vsm expands one cumulative stage name into the three booleans in
    # Application::run; see src/core/Application.cpp.
    "--vsm mark": ["vsm.enableMarking"],
    "--vsm render": ["vsm.enableMarking", "vsm.enablePageRendering"],
    "--vsm shadows": ["vsm.enableMarking", "vsm.enablePageRendering", "vsm.enableShadows"],
}


def leaf_settings(node: dict, prefix: str = "") -> dict[str, object]:
    """Flatten a settings document into dotted key -> value."""
    leaves: dict[str, object] = {}
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            leaves.update(leaf_settings(value, dotted))
        else:
            leaves[dotted] = value
    return leaves


def load_settings(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    # Written by the engine when it saves settings, never read back, and the
    # leg files omit it. It is not a setting and has no default to deviate from.
    document.pop("schemaVersion", None)
    return leaf_settings(document)


def is_path_selecting(value: object) -> bool:
    """Whether a setting picks between code paths rather than tuning one.

    bool is the obvious case. str covers the enums (`toneMapper`,
    `exposureMode`), which name a branch just as much as a bool does. Numbers
    are excluded -- see the module docstring.
    """
    return isinstance(value, (bool, str))


def workflow_leg_files() -> tuple[list[str], list[str]]:
    """The `--settings config/ci/<name>.json` paths the sweep actually runs.

    Read out of the workflow rather than taken from the directory listing,
    because a leg file nothing runs and a leg line with no file are both ways
    the two can drift apart, and only comparing them finds either.
    """
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    block = re.search(r"^\s*cat > artifacts/sweep/legs\.txt <<'LEGS'\n(.*?)^\s*LEGS$", text, re.S | re.M)
    if block is None:
        raise SystemExit(f"{WORKFLOW_PATH}: could not find the sweep's legs.txt heredoc")

    referenced = re.findall(r"--settings\s+config/ci/([A-Za-z0-9._-]+\.json)", block.group(1))
    leg_lines = [line.strip() for line in block.group(1).splitlines() if line.strip()]
    return referenced, leg_lines


def main() -> int:
    defaults = load_settings(DEFAULTS_PATH)
    referenced, leg_lines = workflow_leg_files()
    on_disk = sorted(path.name for path in LEGS_DIR.glob("*.json"))

    failures: list[str] = []

    for name in referenced:
        if name not in on_disk:
            failures.append(f"the sweep runs config/ci/{name}, which does not exist")
    for name in on_disk:
        if name not in referenced:
            failures.append(f"config/ci/{name} exists but no leg in {WORKFLOW_PATH.name} runs it")

    # key -> the legs that move it off its default.
    moved_by: dict[str, list[str]] = {}
    for line in leg_lines:
        _, _, arguments = line.partition("|")
        for flag, keys in FLAG_COVERAGE.items():
            # Prefix rather than equality: a leg may combine the flag with a
            # settings file, the way vsm-debug-views does.
            if arguments.strip().startswith(flag):
                for key in keys:
                    moved_by.setdefault(key, []).append(line)
    for name in sorted(set(referenced) & set(on_disk)):
        leg = load_settings(LEGS_DIR / name)
        if not leg:
            failures.append(f"config/ci/{name} sets nothing")
            continue

        deviations = 0
        for key, value in leg.items():
            if key not in defaults:
                failures.append(f"config/ci/{name} sets {key}, which is not a key in config/runtime_settings.example.json")
                continue
            if defaults[key] != value:
                deviations += 1
                moved_by.setdefault(key, []).append(name)

        if deviations == 0:
            failures.append(
                f"config/ci/{name} is a no-op: every key it sets already holds that value in "
                f"config/runtime_settings.example.json, so this leg renders the default configuration"
            )

    exempt_count = 0
    uncovered: list[str] = []
    for key, value in sorted(defaults.items()):
        if not is_path_selecting(value):
            continue
        if any(key.startswith(prefix) for prefix in EXEMPT_PREFIXES):
            exempt_count += 1
            continue
        if key not in moved_by:
            uncovered.append(f"{key} is only ever rendered as {value!r}; no leg selects the other path")

    failures.extend(uncovered)

    if failures:
        print("CI settings coverage check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print(
            "\nAdd a leg to config/ci/ and a line to the sweep in "
            f"{WORKFLOW_PATH.name}, or exempt the key in {Path(__file__).name} with a reason.",
            file=sys.stderr,
        )
        return 1

    covered = sum(1 for value in defaults.values() if is_path_selecting(value)) - exempt_count
    print(
        f"CI settings coverage check passed ({covered} path-selecting settings covered by "
        f"{len(leg_lines)} sweep legs, {exempt_count} exempt)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
