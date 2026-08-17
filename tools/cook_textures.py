#!/usr/bin/env python3
"""Cook every texture a glTF scene references into block-compressed KTX2.

    tools/cook_textures.py <scene.gltf> [--vecook PATH] [--force] [--dry-run]

Runs `vecook` once per (image, material slot) and writes the sidecar beside the
source image, which is where the runtime looks (see docs/asset_system.md).

It reads a glTF rather than walking a directory on purpose. The cooked format is
decided by the material slot, not by the file -- base colour and emissive are
sRGB BC7, metallic-roughness and occlusion are linear BC7, normal maps are BC5 --
and a directory listing carries no slot information. Guessing the slot from a
file name is exactly the silent-corruption case the pipeline refuses to risk, so
the slots come from the materials that actually use each image.

One image used in two slots is cooked twice, to two sidecars. That is correct:
the formats and the mip filtering differ.

Exit codes: 0 all cooked (or already up to date), 1 at least one cook failed,
2 usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

# glTF spec locations for each texture reference, mapped to the slot names
# assets/TextureCook.cpp uses. Kept as data so an unhandled extension shows up as
# a skipped texture rather than a silently mis-slotted one.
PBR_SLOTS = {
    "baseColorTexture": "base-color",
    "metallicRoughnessTexture": "metallic-roughness",
}
MATERIAL_SLOTS = {
    "normalTexture": "normal",
    "occlusionTexture": "occlusion",
    "emissiveTexture": "emissive",
}


def find_vecook(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            sys.exit(f"cook_textures: no vecook at {path}")
        return path

    repo_root = Path(__file__).resolve().parent.parent
    # Release first: the cook is compute bound, and a Debug vecook is roughly an
    # order of magnitude slower for identical output.
    for candidate in ("build/release/vecook", "build/debug/vecook", "build/vecook"):
        path = repo_root / candidate
        if path.is_file():
            return path

    found = shutil.which("vecook")
    if found:
        return Path(found)

    sys.exit("cook_textures: could not find vecook; build it or pass --vecook PATH")


def collect_slots(scene_path: Path) -> tuple[dict[int, set[str]], list[str]]:
    """Map image index -> set of slot names, plus any warnings worth printing."""
    with scene_path.open("r", encoding="utf-8") as handle:
        gltf = json.load(handle)

    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    warnings: list[str] = []
    slots: dict[int, set[str]] = {}

    def assign(texture_reference: object, slot: str) -> None:
        if not isinstance(texture_reference, dict):
            return
        texture_index = texture_reference.get("index")
        if not isinstance(texture_index, int) or texture_index >= len(textures):
            return
        source = textures[texture_index].get("source")
        if not isinstance(source, int) or source >= len(images):
            return
        slots.setdefault(source, set()).add(slot)

    for material in gltf.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        for key, slot in PBR_SLOTS.items():
            assign(pbr.get(key), slot)
        for key, slot in MATERIAL_SLOTS.items():
            assign(material.get(key), slot)

    for index, image in enumerate(images):
        uri = image.get("uri")
        if not uri:
            # Embedded in a buffer view, or a data: URI. There is no file to cook
            # beside, and the runtime only looks for a sidecar next to a real path.
            warnings.append(f"image {index} has no uri (embedded); skipped")
            slots.pop(index, None)
        elif index not in slots:
            warnings.append(f"image {index} ({uri}) is not used by any material slot; skipped")

    return slots, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path, help="path to a .gltf file")
    parser.add_argument("--vecook", default=None, help="path to the vecook binary")
    parser.add_argument("--force", action="store_true", help="re-encode even when a sidecar is up to date")
    parser.add_argument("--dry-run", action="store_true", help="list the cooks without running them")
    args = parser.parse_args()

    if not args.scene.is_file():
        sys.exit(f"cook_textures: no such scene: {args.scene}")

    vecook = find_vecook(args.vecook)
    scene_path = args.scene.resolve()
    slots, warnings = collect_slots(scene_path)

    with scene_path.open("r", encoding="utf-8") as handle:
        images = json.load(handle).get("images", [])

    for warning in warnings:
        print(f"cook_textures: {warning}")

    jobs: list[tuple[Path, str]] = []
    for image_index in sorted(slots):
        uri = images[image_index].get("uri")
        source = (scene_path.parent / unquote(uri)).resolve()
        if not source.is_file():
            print(f"cook_textures: missing image, skipped: {source}")
            continue
        for slot in sorted(slots[image_index]):
            jobs.append((source, slot))

    print(f"cook_textures: {len(jobs)} cook(s) from {len(slots)} image(s) using {vecook}")

    failures = 0
    for source, slot in jobs:
        output = source.with_suffix("").with_name(f"{source.stem}.{slot}.ktx2")
        command = [str(vecook), str(source), "-o", str(output), "--usage", slot]
        if args.force:
            command.append("--force")

        if args.dry_run:
            print("  " + " ".join(command))
            continue

        # vecook skips a sidecar that is newer than its source itself, so a repeat
        # run costs a stat per texture rather than a re-encode.
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            failures += 1
            print(f"cook_textures: FAILED {source.name} [{slot}]", file=sys.stderr)
            print(result.stdout + result.stderr, file=sys.stderr)
        else:
            print(f"  {source.name} [{slot}] -> {output.name}")

    if failures:
        print(f"cook_textures: {failures} of {len(jobs)} cook(s) failed", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
