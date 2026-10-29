#!/usr/bin/env python3
"""Assemble a `--capture-sequence` frame directory into a looping GIF.

The renderer writes `frame_0000.png` onwards, one per captured frame, with the
fixed-step clock, so the frames are evenly spaced in scene time whatever the
readbacks cost in wall time. This script only encodes them:

- every frame is mapped onto ONE shared 256-colour palette, built from a sample
  of frames spread across the whole clip. Per-frame palettes are what GIF
  encoders do by default, and on a slowly moving camera they make flat regions
  -- sky, a lit wall -- visibly shimmer as each frame picks slightly different
  colours;
- the frame delay is rounded to the GIF format's 10 ms unit, so `--fps` is the
  nearest rate the format can actually play (30 -> 33.3, 50 -> 50).

Example, from the repository root:

    ./build/release/VulkanEngine --scene sponza --deterministic \\
        --window-size 960x540 --camera-orbit 0.01 \\
        --capture-sequence 60 180 --capture-output-dir /tmp/frames
    python tools/make_gif.py /tmp/frames screenshots/demo.gif --fps 30

Requires Pillow. Exit status is 0 on success, 1 on bad input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on the environment
    sys.exit("make_gif.py needs Pillow: python -m pip install Pillow")

# How many frames the shared palette is built from. More is slower and rarely
# changes the result; fewer can miss a colour that only appears mid-clip.
PALETTE_SAMPLES = 12


def build_palette(frames: list[Image.Image], colours: int) -> Image.Image:
    """Quantize a strip of evenly spaced frames into one palette image."""
    step = max(1, len(frames) // PALETTE_SAMPLES)
    samples = frames[::step][:PALETTE_SAMPLES]
    width, height = samples[0].size
    strip = Image.new("RGB", (width, height * len(samples)))
    for index, frame in enumerate(samples):
        strip.paste(frame, (0, index * height))
    return strip.quantize(colors=colours, method=Image.Quantize.MEDIANCUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=Path, help="directory holding frame_0000.png onwards")
    parser.add_argument("output", type=Path, help="GIF to write")
    parser.add_argument("--fps", type=float, default=30.0, help="playback rate (default 30)")
    parser.add_argument("--width", type=int, default=0, help="scale to this width, keeping aspect (default: as captured)")
    parser.add_argument("--colours", type=int, default=256, help="palette size, 2..256 (default 256)")
    parser.add_argument("--no-dither", action="store_true", help="map to the palette without Floyd-Steinberg dithering")
    args = parser.parse_args()

    paths = sorted(args.frames.glob("frame_*.png"))
    if not paths:
        print(f"No frame_*.png files in {args.frames}", file=sys.stderr)
        return 1
    if args.fps <= 0.0:
        print("--fps must be positive", file=sys.stderr)
        return 1
    if not 2 <= args.colours <= 256:
        print("--colours must be in [2, 256]", file=sys.stderr)
        return 1

    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            frame = image.convert("RGB")
        if args.width > 0 and frame.width != args.width:
            height = max(1, round(frame.height * args.width / frame.width))
            frame = frame.resize((args.width, height), Image.Resampling.LANCZOS)
        if frames and frame.size != frames[0].size:
            print(f"{path.name} is {frame.size}, the first frame is {frames[0].size}", file=sys.stderr)
            return 1
        frames.append(frame)

    palette = build_palette(frames, args.colours)
    dither = Image.Dither.NONE if args.no_dither else Image.Dither.FLOYDSTEINBERG
    indexed = [frame.quantize(palette=palette, dither=dither) for frame in frames]

    # GIF delays are in hundredths of a second.
    delay_ms = max(10, round(1000.0 / args.fps / 10.0) * 10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        args.output,
        save_all=True,
        append_images=indexed[1:],
        duration=delay_ms,
        loop=0,
        optimize=False,
        disposal=1,
    )

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(
        f"Wrote {args.output}: {len(indexed)} frames, {indexed[0].width}x{indexed[0].height}, "
        f"{1000.0 / delay_ms:.1f} fps, {size_mb:.1f} MiB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
