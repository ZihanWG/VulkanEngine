#!/usr/bin/env python3
"""Pin every shader's descriptor and push-constant interface against a golden.

The engine hand-mirrors 134 `layout(set = , binding = )` declarations in GLSL
against descriptor writes in C++, plus push-constant blocks whose member offsets
have to agree across the language boundary where no static_assert can reach. A
mismatch there does not fail to build. Some of it the validation layer catches at
pipeline creation, and CI does run validation over every configuration in the
sweep -- but only for the code paths those configurations actually execute, and
never for "the
right slot, filled with the wrong resource".

What this adds is narrower than "verify C++ against GLSL" and worth stating
plainly, because a guard that overstates its coverage is worse than no guard:
it makes a shader's interface impossible to change SILENTLY. The reflected
interface is committed, so renumbering a binding, adding a descriptor, or moving
a push-constant member shows up as a diff a reviewer has to consciously accept,
next to the C++ change that should accompany it.

Two designs were measured and rejected before this one, both on the real shaders:

  * "a resource name must sit at the same (set, binding) everywhere" -- 12 of 67
    names legitimately differ, because unrelated pipelines bind different inputs
    at the same slot (uBrdfLut is binding 5 in one pipeline and 6 in another).
  * "no (set, binding) may hold two different resources" -- 7 of 18 slots hold
    many names, for the same reason.

Both would have needed a hand-maintained exception list, which is the stale
mirror this is supposed to replace.

This does NOT replace tools/check_shader_constants.py. Reflection sees the
interface, not `const uint` values -- those are folded away by the compiler and
never reach the SPIR-V -- so the two checkers cover different halves and both
are needed.

Usage:
    python tools/check_shader_interface.py [--build-dir DIR] [--update]

Exit status is 0 when the interface matches the golden, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "shader_interface.txt"

# Build directories searched, in order, when --build-dir is not given.
DEFAULT_BUILD_DIRS = ("build/ci-debug", "build/debug", "build/release")

# Reflection keys that carry descriptor bindings. Listed explicitly rather than
# "every key with a binding" so that a spirv-cross version introducing a new
# resource class shows up as a missing section rather than being silently
# dropped from the guard.
DESCRIPTOR_KINDS = (
    "textures",
    "separate_images",
    "separate_samplers",
    "images",
    "ubos",
    "ssbos",
    "acceleration_structures",
    "subpass_inputs",
)


def find_spirv_cross() -> str:
    """Locate spirv-cross, preferring the SDK the shaders were built with."""
    sdk = os.environ.get("VULKAN_SDK")
    if sdk:
        for name in ("spirv-cross", "spirv-cross.exe"):
            candidate = Path(sdk) / "Bin" / name
            if candidate.is_file():
                return str(candidate)
    found = shutil.which("spirv-cross")
    if found:
        return found
    sys.exit(
        "spirv-cross was not found. It ships with the Vulkan SDK; set VULKAN_SDK "
        "or put it on PATH.\nThis check fails rather than skipping: a guard that "
        "quietly does nothing is indistinguishable from one that passes."
    )


def find_build_dir(explicit: str | None) -> Path:
    if explicit:
        directory = Path(explicit)
        if not (directory / "shaders").is_dir():
            sys.exit(f"No shaders/ directory under {directory}.")
        return directory / "shaders"
    for candidate in DEFAULT_BUILD_DIRS:
        directory = REPO_ROOT / candidate / "shaders"
        if directory.is_dir():
            return directory
    sys.exit(
        "No compiled shaders found. Looked for shaders/ under: "
        + ", ".join(DEFAULT_BUILD_DIRS)
        + ".\nBuild the VulkanEngineShaders target first, or pass --build-dir."
    )


def resolve_type(type_name: str, types: dict, depth: int = 0) -> str:
    """Render a spirv-cross type id as a stable signature.

    Block types come back as ids like `_1906`, which are SPIR-V ids and move
    whenever anything earlier in the module changes. Left raw they would make
    the golden churn on edits that changed no interface at all, so they are
    resolved to the struct's own name and members.
    """
    entry = types.get(type_name)
    if entry is None:
        return type_name
    if depth >= 4:
        # Deep buffer-reference chains bottom out rather than recursing forever;
        # the outer levels are what the C++ side mirrors.
        return entry.get("name", type_name)
    members = []
    for member in entry.get("members", []):
        member_type = resolve_type(member.get("type", "?"), types, depth + 1)
        if member.get("physical_pointer"):
            member_type += "*"
        offset = member.get("offset")
        array = member.get("array")
        suffix = f"[{'x'.join(str(a) for a in array)}]" if array else ""
        offset_text = f"@{offset}" if offset is not None else ""
        members.append(f"{member.get('name', '?')}{suffix}:{member_type}{offset_text}")
    return f"{entry.get('name', type_name)}{{{', '.join(members)}}}"


def reflect(spirv_cross: str, spv: Path) -> list[str]:
    result = subprocess.run(
        [spirv_cross, str(spv), "--reflect"], capture_output=True, text=True
    )
    if result.returncode != 0:
        sys.exit(f"spirv-cross failed on {spv.name}:\n{result.stderr.strip()}")
    reflection = json.loads(result.stdout)
    types = reflection.get("types", {})

    lines: list[str] = []
    for kind in DESCRIPTOR_KINDS:
        for entry in reflection.get(kind, []):
            if entry.get("set") is None or entry.get("binding") is None:
                continue
            array = entry.get("array")
            # An unsized bindless array reflects as array [0]; record the shape
            # either way, because the count is part of the contract with the
            # descriptor pool.
            shape = f"[{'x'.join(str(a) for a in array)}]" if array else ""
            declared = entry.get("type", "?")
            # Texture/image types reflect as readable names already; only block
            # types need resolving through the type table.
            rendered = resolve_type(declared, types) if declared.startswith("_") else declared
            lines.append(
                f"  set={entry['set']} binding={entry['binding']:<3} {kind:<12}"
                f" {entry.get('name', '?')}{shape} : {rendered}"
            )

    for entry in reflection.get("push_constants", []):
        rendered = resolve_type(entry.get("type", "?"), types)
        lines.append(f"  push_constant {entry.get('name', '?')} : {rendered}")

    for entry in reflection.get("specialization_constants", []):
        lines.append(
            f"  spec_constant id={entry.get('id')} {entry.get('name', '?')}"
            f" : {entry.get('type', '?')} = {entry.get('default_value')}"
        )

    # Sorted so the golden depends on the interface, not on the order
    # spirv-cross happened to walk the module in.
    return sorted(lines)


def build_report(spirv_cross: str, shader_dir: Path) -> str:
    modules = sorted(shader_dir.glob("*.spv"), key=lambda p: p.name)
    if not modules:
        sys.exit(f"No .spv modules under {shader_dir}.")
    out: list[str] = [
        "# Reflected descriptor and push-constant interface of every shader.",
        "# Generated by tools/check_shader_interface.py -- do not hand-edit.",
        "#",
        "# A diff here means a shader's interface moved. That is allowed, but it",
        "# has to be accompanied by the matching C++ descriptor or push-constant",
        "# change, which is the pairing this file exists to make visible.",
        "#",
        f"# {len(modules)} modules.",
        "",
    ]
    for spv in modules:
        out.append(spv.name[: -len(".spv")])
        lines = reflect(spirv_cross, spv)
        out.extend(lines if lines else ["  (no descriptors or push constants)"])
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", help="Build directory holding shaders/*.spv.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite the golden instead of comparing against it.",
    )
    args = parser.parse_args()

    shader_dir = find_build_dir(args.build_dir)
    report = build_report(find_spirv_cross(), shader_dir)

    if args.update:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(report, encoding="utf-8", newline="\n")
        print(f"Wrote {GOLDEN_PATH.relative_to(REPO_ROOT)} from {shader_dir}.")
        return 0

    if not GOLDEN_PATH.is_file():
        sys.exit(
            f"{GOLDEN_PATH.relative_to(REPO_ROOT)} is missing. Create it with "
            "--update and commit it."
        )

    expected = GOLDEN_PATH.read_text(encoding="utf-8")
    if expected == report:
        module_count = report.count("\n\n")
        print(f"Shader interface matches the golden ({module_count} modules).")
        return 0

    import difflib

    diff = difflib.unified_diff(
        expected.splitlines(),
        report.splitlines(),
        fromfile=str(GOLDEN_PATH.relative_to(REPO_ROOT)),
        tofile=f"reflected from {shader_dir}",
        lineterm="",
    )
    print("\n".join(diff))
    print()
    print(
        "A shader's descriptor or push-constant interface changed. If that was "
        "intended, make the matching C++ change, then re-run with --update and "
        "commit the golden alongside it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
