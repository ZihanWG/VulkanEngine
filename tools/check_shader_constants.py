#!/usr/bin/env python3
"""Assert that constants duplicated between C++ and GLSL still agree.

The engine mirrors a number of layout constants by hand -- the cluster grid, the
fog froxel grid, the per-cluster light cap, the cull-stats counter offsets. Every
one is commented at both ends ("Mirrors ...", "Must match ..."), and the C++ side
is unit-tested, but nothing verifies that the two copies are actually equal. That
is the whole risk: a mismatch does not fail to build and does not trip validation.
It silently addresses the wrong froxel, or drops lights past a cap the shader
thinks is larger, and shows up as subtly wrong lighting much later.

This is deliberately a dumb textual comparison rather than anything clever. It
parses `inline constexpr uint32_t NAME = VALUE;` on the C++ side and
`const uint NAME = VALUEu;` on the GLSL side, and compares the pairs it is told
to compare. Expressions (kClusterCount = X * Y * Z) are evaluated only in terms
of constants already collected, so the products are checked too.

Run from anywhere; paths resolve relative to the repository root.
Exit status is 0 when every pair agrees, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (C++ header, GLSL file, [constant names]). A name must appear in both files.
PAIRS: list[tuple[str, str, list[str]]] = [
    (
        "src/renderer/ClusterGrid.h",
        "src/shaders/cluster_grid.glsl",
        ["kClusterGridX", "kClusterGridY", "kClusterGridZ", "kClusterCount", "kMaxLightsPerCluster"],
    ),
    (
        "src/renderer/VolumetricFog.h",
        "src/shaders/fog_inject.comp",
        ["kFogGridX", "kFogGridY", "kFogGridZ"],
    ),
    (
        "src/renderer/VolumetricFog.h",
        "src/shaders/fog_integrate.comp",
        ["kFogGridX", "kFogGridY", "kFogGridZ"],
    ),
]

CPP_PATTERN = re.compile(
    r"^\s*inline\s+constexpr\s+(?:uint32_t|int32_t|uint|int|size_t)\s+(\w+)\s*=\s*([^;]+);",
    re.MULTILINE,
)
GLSL_PATTERN = re.compile(r"^\s*const\s+(?:uint|int)\s+(\w+)\s*=\s*([^;]+);", re.MULTILINE)


def evaluate(expression: str, known: dict[str, int]) -> int | None:
    """Evaluate an integer constant expression over already-collected constants.

    Returns None for anything that is not a plain product/sum of integers and
    known names -- those are reported as unresolved rather than guessed at.
    """
    cleaned = re.sub(r"\bu\b|[uU](?=\s*(?:[*+\-/)]|$))", "", expression)
    cleaned = cleaned.replace("u;", "").strip()
    tokens = re.findall(r"[A-Za-z_]\w*|\d+|[*+\-/()]", cleaned)
    if not tokens:
        return None

    rebuilt = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z_]\w*", token):
            if token not in known:
                return None
            rebuilt.append(str(known[token]))
        else:
            rebuilt.append(token)

    try:
        # Only digits and arithmetic remain by construction.
        return int(eval("".join(rebuilt), {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return None


def collect(path: Path, pattern: re.Pattern[str]) -> dict[str, int]:
    text = path.read_text()
    values: dict[str, int] = {}
    for name, expression in pattern.findall(text):
        value = evaluate(expression, values)
        if value is not None:
            values[name] = value
    return values


def main() -> int:
    failures: list[str] = []
    checked = 0

    for cpp_relative, glsl_relative, names in PAIRS:
        cpp_path = REPO_ROOT / cpp_relative
        glsl_path = REPO_ROOT / glsl_relative
        for path in (cpp_path, glsl_path):
            if not path.is_file():
                failures.append(f"missing file: {path.relative_to(REPO_ROOT)}")
        if failures:
            continue

        cpp_values = collect(cpp_path, CPP_PATTERN)
        glsl_values = collect(glsl_path, GLSL_PATTERN)

        for name in names:
            if name not in cpp_values:
                failures.append(f"{cpp_relative}: could not read a value for {name}")
                continue
            if name not in glsl_values:
                failures.append(f"{glsl_relative}: could not read a value for {name}")
                continue
            checked += 1
            if cpp_values[name] != glsl_values[name]:
                failures.append(
                    f"{name} disagrees: {cpp_relative} says {cpp_values[name]}, "
                    f"{glsl_relative} says {glsl_values[name]}"
                )

    if failures:
        print("Shader constant parity check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"Shader constant parity check passed ({checked} constants).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
