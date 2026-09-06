---
name: temporal-upsampling-initiative
description: "TAAU shipped: the TAA resolve reconstructs at presentation resolution from jittered samples (merged 408a009, pushed). Not FSR2 -- no locks/reactive masks/disocclusion."
metadata:
  type: project
---

**Merged `408a009` and pushed** 2026-08-12, branch deleted. Built on the render
scale + sub-rect work ([[render-scale-initiative]]), which is what made it
tractable: "different sources written over different fractions" was already a
first-class concept.

## What it is

The TAA resolve runs at **presentation** resolution and reconstructs each output
pixel from the nine nearest jittered low-resolution samples. With TAA on the
composite does no stretching at all, and everything after the resolve -- bloom,
luminance, histogram -- is already full size.

**The idea in one line**: the projection is offset by `+jitter`, so a source
texel centred at `c` holds what an unjittered frame would have at `c - jitter`.
Undoing that gives every sample a real sub-pixel position. Kernel is
`exp(-2.29 d^2)` in source pixels.

**Why bilinear accumulation cannot work**: it converges on an upscale of itself
however long it runs, because where each sample really was is gone before the
accumulation sees it. This is the sentence to reach for if anyone asks why TAA
alone is not upsampling.

## Structure worth not re-deriving

- **`PostProcessStack::activePostProcessSource()`** returns allocated extent,
  written extent and uv scale together. Five consumers switch source when TAA is
  on (bloom extract, bloom downsample L0, composite, luminance, histogram);
  landing this *first*, as a no-op, is what made the resolution change one branch
  instead of five hunts.
- **The history stops being a low-resolution source.** Scene colour / velocity /
  depth take `sourceUvScale`; history is written in full and sampled directly.
- **Neighbourhood offsets are one SOURCE texel in output space** = texelSize /
  sourceUvScale. Measuring the box in output texels clamps history against an
  upscaled blur of itself and it stops rejecting anything.
- **The sharpen gate moved** from "frame is not native" to "the composite is
  actually stretching", so it is off when TAAU is on. Sharpening a temporally
  upsampled image is a separate decision.

## Ghosting rejection: both OFF by default

Variance clipping (YCoCg mean +/- gamma sigma, clipped toward the centre) and
rejection feedback (feedback scaled down by how far the history had to travel).

**They were built on a user report of ghosting that was then retracted.** Every
form of stricter rejection costs accumulated history, which is what upsampling
exists to gather, so with nothing to reject they are cost without benefit. With
both off the resolve is bit-identical to the version that was actually judged
good. Kept because ghosting is real and this scene (11 static objects, orbiting
lights) has almost no object motion to produce it.

**Lesson**: for a reported *visual* problem I cannot see, get a reproducible
condition (scene, motion, where to look) before building the fix.

## Not done

- No disocclusion detection -- needs a previous-frame depth the engine does not
  keep.
- No locks, no reactive masks, no shading-change detection. Not FSR2.
- No confidence weighting: an output pixel no sample landed near blends the same
  as one a sample landed on.
- **Perf numbers are all thermally inflated.** Measured hot; 0.5 with TAA off
  shows the same MainHDRPass as with it on, so the resolve's ~1.0-1.6 ms is the
  only defensible figure. Re-measure cold before quoting anything.
