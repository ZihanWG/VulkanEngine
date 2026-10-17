#!/usr/bin/env python3
"""Error of a ray-traced shadow mask against the reference mode's mask.

Reads the linear PFMs ``--rt-shadow-dump`` writes -- one float of sun visibility
per pixel, NaN where nothing was traced -- and compares each candidate with a
reference captured from the same frame, scene and camera. The debug views go
through exposure and tone mapping, so they can show where a mask is wrong but
not by how much; these numbers are what a filter's change is judged by.

The reference is ``debugReferenceSamples`` rays per pixel on a golden-angle
spiral over the sun's disc. At 64 rays it is the closest thing to ground truth
the renderer can make, not ground truth: its own error is a fraction of a
sample's 1/64 step wherever the disc is partly covered.

Reported per candidate, over pixels both masks traced:

- ``penumbra MAE``: mean absolute error where the reference lies strictly
  between 0.05 and 0.95. The number a soft-shadow filter is for; the umbra and
  the lit floor are easy, and averaging them in hides a bad penumbra.
- ``MAE`` and ``p95``: over every pixel, so a filter that blurs shadow into lit
  ground shows up even where the reference has no penumbra.
- ``leaks``: pixels the reference calls umbra (<= 0.05) that the candidate
  lights to 0.5 or more -- light bleeding through a contact -- and the reverse,
  lit (>= 0.95) pixels darkened below 0.5.
- ``coverage``: pixels the candidate traced, as a share of those the reference
  traced. A filter or an upsample that drops pixels shows here, not as error.

Usage:
    tools/dev/compare_shadow_visibility.py REFERENCE.pfm CANDIDATE.pfm [CANDIDATE.pfm ...]
    tools/dev/compare_shadow_visibility.py REFERENCE.pfm CANDIDATE.pfm --error-map error.pfm
"""

from __future__ import annotations

import argparse
import array
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PENUMBRA_LOW = 0.05
PENUMBRA_HIGH = 0.95
LEAK_THRESHOLD = 0.5


@dataclass
class Pfm:
    width: int
    height: int
    # Row-major, top row first: the file's bottom-up order is undone on read.
    values: list[float]


def read_pfm(path: Path) -> Pfm:
    data = path.read_bytes()
    # Three whitespace-separated header tokens, then one whitespace byte.
    tokens: list[bytes] = []
    cursor = 0
    while len(tokens) < 4:
        while data[cursor : cursor + 1].isspace():
            cursor += 1
        start = cursor
        while not data[cursor : cursor + 1].isspace():
            cursor += 1
        tokens.append(data[start:cursor])
    cursor += 1
    magic, width_token, height_token, scale_token = tokens
    if magic != b"Pf":
        raise ValueError(f"{path}: not a single-channel PFM (magic {magic!r}).")
    width, height, scale = int(width_token), int(height_token), float(scale_token)
    floats = array.array("f")
    floats.frombytes(data[cursor : cursor + width * height * 4])
    if len(floats) != width * height:
        raise ValueError(f"{path}: {len(floats)} values for a {width}x{height} image.")
    if (scale < 0) != (sys.byteorder == "little"):
        floats.byteswap()
    rows = [floats[row * width : (row + 1) * width] for row in range(height)]
    values: list[float] = []
    for row in reversed(rows):
        values.extend(row)
    return Pfm(width, height, values)


def write_pfm(path: Path, width: int, height: int, values: list[float]) -> None:
    floats = array.array("f", values)
    if sys.byteorder != "little":
        floats.byteswap()
    with path.open("wb") as file:
        file.write(f"Pf\n{width} {height}\n-1.0\n".encode())
        for row in reversed(range(height)):
            file.write(floats[row * width : (row + 1) * width].tobytes())


@dataclass
class Comparison:
    penumbra_mae: float
    penumbra_pixels: int
    mae: float
    p95: float
    light_leaks: int
    false_shadow: int
    coverage: float
    compared: int


def compare(reference: Pfm, candidate: Pfm, error_map: list[float] | None) -> Comparison:
    errors: list[float] = []
    penumbra_sum = 0.0
    penumbra_pixels = 0
    light_leaks = 0
    false_shadow = 0
    reference_traced = 0
    for index, (expected, actual) in enumerate(zip(reference.values, candidate.values)):
        if math.isnan(expected):
            continue
        reference_traced += 1
        if math.isnan(actual):
            continue
        error = abs(actual - expected)
        errors.append(error)
        if error_map is not None:
            error_map[index] = error
        if PENUMBRA_LOW < expected < PENUMBRA_HIGH:
            penumbra_sum += error
            penumbra_pixels += 1
        if expected <= PENUMBRA_LOW and actual >= LEAK_THRESHOLD:
            light_leaks += 1
        elif expected >= PENUMBRA_HIGH and actual < LEAK_THRESHOLD:
            false_shadow += 1
    errors.sort()
    compared = len(errors)
    return Comparison(
        penumbra_mae=penumbra_sum / penumbra_pixels if penumbra_pixels else float("nan"),
        penumbra_pixels=penumbra_pixels,
        mae=sum(errors) / compared if compared else float("nan"),
        p95=errors[min(compared - 1, int(0.95 * compared))] if compared else float("nan"),
        light_leaks=light_leaks,
        false_shadow=false_shadow,
        coverage=compared / reference_traced if reference_traced else float("nan"),
        compared=compared,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument(
        "--error-map",
        type=Path,
        help="Write |candidate - reference| as a PFM (NaN where either is missing). One candidate only.",
    )
    args = parser.parse_args()
    if args.error_map is not None and len(args.candidates) != 1:
        parser.error("--error-map takes exactly one candidate.")

    reference = read_pfm(args.reference)
    print(f"Reference: {args.reference} ({reference.width}x{reference.height})")
    print()
    print("| Candidate | Penumbra MAE | Penumbra px | MAE | p95 | Light leaks | False shadow | Coverage |")
    print("|---|---|---|---|---|---|---|---|")
    for path in args.candidates:
        candidate = read_pfm(path)
        if (candidate.width, candidate.height) != (reference.width, reference.height):
            print(f"{path}: {candidate.width}x{candidate.height} does not match the reference.", file=sys.stderr)
            return 1
        error_map = [float("nan")] * len(reference.values) if args.error_map is not None else None
        result = compare(reference, candidate, error_map)
        print(
            f"| {path.name} | {result.penumbra_mae:.4f} | {result.penumbra_pixels} | {result.mae:.4f} | "
            f"{result.p95:.4f} | {result.light_leaks} | {result.false_shadow} | {100.0 * result.coverage:.2f}% |"
        )
        if error_map is not None:
            write_pfm(args.error_map, reference.width, reference.height, error_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
