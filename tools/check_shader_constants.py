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
#
# Keep this table complete: check_table_completeness below fails when a shader
# carrying a "Must match ve::" comment has no entry here, because a table that
# quietly falls behind is indistinguishable from one that passes. That is how
# this table came to guard three groups while its own docstring claimed more.
PAIRS: list[tuple[str, str, list[str]]] = [
    (
        "src/renderer/ClusterGrid.h",
        "src/shaders/cluster_grid.glsl",
        ["kClusterGridX", "kClusterGridY", "kClusterGridZ", "kClusterCount", "kMaxLightsPerCluster"],
    ),
    # One shared GLSL header for the froxel volume, so this is one pair rather
    # than three -- the third copy was a function-local kFogNearPlane in
    # simple_bindless.frag that no name-based lookup was ever going to find.
    (
        "src/renderer/VolumetricFog.h",
        "src/shaders/volumetric_fog.glsl",
        ["kFogGridX", "kFogGridY", "kFogGridZ", "kFogNearPlane"],
    ),
    # The clipmap layout. PR #13 raised kVsmPagesPerLevelAxis from 16 to 32 by
    # hand on both sides -- exactly the edit this script exists to guard, and it
    # was not guarded at the time.
    (
        "src/renderer/VirtualShadowMap.h",
        "src/shaders/virtual_shadow_map.glsl",
        [
            "kVsmPageSize",
            "kVsmPagesPerLevelAxis",
            "kVsmPagesPerLevel",
            "kVsmLevelResolution",
            "kVsmMaxClipmapLevels",
            "kVsmMaxVirtualPages",
            "kVsmPagePoolSize",
            "kVsmPagePoolPagesPerAxis",
            "kVsmInvalidPhysicalPage",
        ],
    ),
    # One shared GLSL header for every probe shader, so this is one pair rather
    # than five. It used to be five copies with nothing comparing any of them.
    (
        "src/renderer/IrradianceProbes.h",
        "src/shaders/irradiance_probes.glsl",
        [
            "kProbeGridX",
            "kProbeGridY",
            "kProbeGridZ",
            "kProbeIrradianceResolution",
            "kProbeDepthResolution",
            "kProbeBorderTexels",
            "kProbeAtlasTilesX",
            "kProbeAtlasTilesY",
            "kProbeCaptureFaceCount",
            "kProbeCaptureFaceResolution",
            "kProbeMaxDistance",
            "kProbeDepthLobeExponent",
            "kProbeMinVisibility",
            "kProbeBackfaceFloor",
        ],
    ),
    # The palette buffer's two halves. A mismatched offset would not fail to
    # build: the main pass would read its previous pose from the wrong matrices
    # and the velocity buffer would carry motion that never happened.
    (
        "src/renderer/SkeletalAnimation.h",
        "src/shaders/joint_palette.glsl",
        ["kMaxSkinJoints", "kSkinPreviousPaletteOffset"],
    ),
]

# Constants whose two copies deliberately carry different names, as
# (C++ header, GLSL file, {glsl_name: cpp_name}).
#
# Kept apart from PAIRS because a rename is a claim about intent, not a lookup
# failure: listing one here says "these two are the same number under two
# names", which a reader should have to opt into rather than have inferred.
RENAMED_PAIRS: list[tuple[str, str, dict[str, str]]] = [
    (
        "src/renderer/RendererInternal.h",
        "src/shaders/cull.comp",
        {"kStatsLodCounterOffset": "kGpuCullStatsLodCounterOffset"},
    ),
]

# Shaders that carry a "Must match ve::" comment but deliberately have no pair.
# Empty, and the completeness check exists to keep it that way.
COMPLETENESS_EXEMPT: set[str] = set()

CPP_PATTERN = re.compile(
    r"^\s*(?:inline\s+)?constexpr\s+(?:uint32_t|int32_t|uint|int|size_t|float|double)\s+(\w+)\s*=\s*([^;]+);",
    re.MULTILINE,
)
GLSL_PATTERN = re.compile(r"^\s*const\s+(?:uint|int|float)\s+(\w+)\s*=\s*([^;]+);", re.MULTILINE)

# The shader-side half of a mirror is always announced in a comment. This finds
# those announcements so check_table_completeness can compare them against PAIRS.
CONSTANT_MIRROR_COMMENT = re.compile(r"//\s*(?:Must match|Mirrors)\s+(ve::[\w:]*\*?)", re.MULTILINE)


Number = int | float


def evaluate(expression: str, known: dict[str, Number]) -> Number | None:
    """Evaluate a constant expression over already-collected constants.

    Returns None for anything that is not a plain product/sum of numbers and
    known names -- those are reported as unresolved rather than guessed at.

    Integers stay integers. A value is float only when it is written as one on
    either side, which keeps the comparison exact: both languages parse "64.0"
    and "64.0f" to the same double, so no tolerance is needed and none is used.
    A tolerance here would hide the small divergences this exists to catch.
    """
    # Strip the integer suffix GLSL writes (2u) and the float suffix C++ writes
    # (0.2f); neither language accepts the other's.
    cleaned = expression.strip().rstrip(";").strip()
    # One tokenizer, ordered longest-first so 0xFF is not read as 0 then xFF.
    # Numeric literals carry their language's suffix -- GLSL writes 2u, C++
    # writes 0.2f and 0xFFFFFFFFu -- and the suffix is stripped per token rather
    # than by a global regex, which would also eat the trailing letter of an
    # identifier that happens to end in u or f.
    tokens = re.findall(
        r"0[xX][0-9a-fA-F]+[uUlL]*|\d+\.\d*[fFlL]?|\.\d+[fFlL]?|\d+[uUlLfF]*|[A-Za-z_]\w*|[*+\-/()]",
        cleaned,
    )
    if not tokens:
        return None

    rebuilt = []
    for token in tokens:
        if re.fullmatch(r"[A-Za-z_]\w*", token):
            if token not in known:
                return None
            rebuilt.append(repr(known[token]))
        elif re.match(r"0[xX]", token):
            rebuilt.append(token.rstrip("uUlL"))
        elif token[0].isdigit() or token[0] == ".":
            rebuilt.append(token.rstrip("uUlLfF"))
        else:
            rebuilt.append(token)

    try:
        # Only numbers and arithmetic remain by construction.
        value = eval("".join(rebuilt), {"__builtins__": {}}, {})  # noqa: S307
    except Exception:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # An integer expression Python widened through "/" is still an integer
    # value; keep it one so int and float spellings of a number compare equal.
    if isinstance(value, float) and value.is_integer() and "." not in cleaned:
        return int(value)
    return value
def collect(path: Path, pattern: re.Pattern[str]) -> dict[str, Number]:
    text = path.read_text(encoding="utf-8")
    values: dict[str, Number] = {}
    for name, expression in pattern.findall(text):
        value = evaluate(expression, values)
        if value is not None:
            values[name] = value
    return values


def cpp_constant_names() -> dict[str, str]:
    """Every C++ constant a shader could be mirroring, name -> where it lives."""
    names: dict[str, str] = {}
    for root in ("src/renderer", "src/rhi"):
        for path in sorted((REPO_ROOT / root).rglob("*.h")):
            for name, _ in CPP_PATTERN.findall(path.read_text(encoding="utf-8")):
                names.setdefault(name, path.relative_to(REPO_ROOT).as_posix())
    return names


def check_table_completeness() -> list[str]:
    """Fail when a shader declares a constant that also exists in C++, unchecked.

    This is structural on purpose. The first version of this check read the
    "Must match ve::..." comments, on the theory that the comments are a second
    list of what ought to be in PAIRS. They are not a reliable one: the same
    relationship is written "Must match", "Mirrors", and "Keep the two in sync"
    in different files, so that version caught two of the five groups and
    reported success -- which is worse than not checking, because it reads as
    coverage.

    So the signal is the declaration, not the prose. A GLSL constant whose name
    is also a C++ constant is a mirror by construction, whatever the comment
    above it says. A name that collides by accident rather than by mirroring is
    the one false positive this can produce, and the answer to it is
    COMPLETENESS_EXEMPT -- which forces someone to look, which is the point.
    """
    covered = {glsl for _, glsl, _ in PAIRS} | {glsl for _, glsl, _ in RENAMED_PAIRS}
    covered |= COMPLETENESS_EXEMPT
    renamed_glsl_names = {name for _, _, mapping in RENAMED_PAIRS for name in mapping}

    cpp_names = cpp_constant_names()
    problems: list[str] = []
    shader_root = REPO_ROOT / "src" / "shaders"
    for path in sorted(shader_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".glsl", ".comp", ".frag", ".vert"}:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in covered:
            continue

        mirrored = sorted(
            name
            for name, _ in GLSL_PATTERN.findall(path.read_text(encoding="utf-8"))
            if name in cpp_names and name not in renamed_glsl_names
        )
        if mirrored:
            shown = ", ".join(mirrored[:4]) + (", ..." if len(mirrored) > 4 else "")
            problems.append(
                f"{relative} declares {len(mirrored)} constant(s) that also exist in C++ "
                f"({shown}) but the file is in neither PAIRS nor RENAMED_PAIRS -- add it, "
                f"or list it in COMPLETENESS_EXEMPT"
            )

    # A file can be in PAIRS and still leave constants unchecked, which is the
    # same hole one level down: the entry looks like coverage while some of the
    # file's mirrored names are absent from its list.
    for cpp_relative, glsl_relative, names in PAIRS:
        glsl_path = REPO_ROOT / glsl_relative
        cpp_path = REPO_ROOT / cpp_relative
        if not glsl_path.is_file() or not cpp_path.is_file():
            continue
        cpp_local = {name for name, _ in CPP_PATTERN.findall(cpp_path.read_text(encoding="utf-8"))}
        declared = {name for name, _ in GLSL_PATTERN.findall(glsl_path.read_text(encoding="utf-8"))}
        missing = sorted((declared & cpp_local) - set(names))
        if missing:
            problems.append(
                f"{glsl_relative} declares {', '.join(missing)}, which {cpp_relative} also "
                f"defines, but the PAIRS entry does not list them"
            )

    return problems


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

    for cpp_relative, glsl_relative, mapping in RENAMED_PAIRS:
        cpp_path = REPO_ROOT / cpp_relative
        glsl_path = REPO_ROOT / glsl_relative
        if not cpp_path.is_file() or not glsl_path.is_file():
            failures.append(f"missing file for renamed pair {cpp_relative} / {glsl_relative}")
            continue

        cpp_values = collect(cpp_path, CPP_PATTERN)
        glsl_values = collect(glsl_path, GLSL_PATTERN)
        for glsl_name, cpp_name in mapping.items():
            if cpp_name not in cpp_values:
                failures.append(f"{cpp_relative}: could not read a value for {cpp_name}")
                continue
            if glsl_name not in glsl_values:
                failures.append(f"{glsl_relative}: could not read a value for {glsl_name}")
                continue
            checked += 1
            if cpp_values[cpp_name] != glsl_values[glsl_name]:
                failures.append(
                    f"{glsl_name} disagrees with {cpp_name}: {cpp_relative} says "
                    f"{cpp_values[cpp_name]}, {glsl_relative} says {glsl_values[glsl_name]}"
                )

    failures.extend(check_table_completeness())

    if failures:
        print("Shader constant parity check FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(f"Shader constant parity check passed ({checked} constants).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
