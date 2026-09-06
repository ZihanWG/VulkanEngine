---
name: render-graph-transient-allocator
description: "Render graph transient allocator — COMPLETE and merged (f0b2038, 5139b4c, e9378f7). Bloom aliasing MEASURED AND REJECTED as a default: 17.48 MiB for +1.2% frame time. Do not extend it without tightening the handoff barrier first."
metadata: 
  node_type: memory
  type: project
  originSessionId: 901cbd1c-8080-4c6f-9d21-4f11902d4869
  modified: 2026-08-15T14:07:54.387Z
---

Making `RenderGraph` own transient *memory* (not images — descriptors stay put).
Plan: `/Users/zihanw/.claude/plans/fancy-wandering-zebra.md`.

**Phases 1-3 MERGED to main** (`f0b2038` via PR #6, `5139b4c` via PR #7), branches
deleted, all three workflows green on main.

**Phase 3 SHIPPED: bloom chain aliases into one pool, 41.12 MiB of images in a
23.64 MiB pool = 17.48 MiB saved.** OFF by default (`enableTransientAliasing`).
Phase 3 split into 3a (RHI mechanism) + 3b (wiring) because bloom lifetimes are
NOT locally derivable: BloomMipDownsample0 lives passes 12-18 only because
BloomMipUpsample0 reads it at 18 for the two-source blend.
Key pieces: `rhi::VulkanTransientMemoryPool`, `VulkanImage::createAliased`
(reset() must NOT free pool memory — keyed on an `aliased_` flag that also has to
travel through `moveFrom`, since bloom images live in a std::vector).
Alias-handoff barrier is ONE place in `transitionTexture`, keyed off
`usedThisFrame`: first use transitions from UNDEFINED with a conservative
ALL_COMMANDS/MEMORY source scope.
**Known wart:** the plan needs lifetimes, so it applies after frame 1, and that
recreate resets the auto-exposure accumulator → 1-LSB differences vs the
non-aliased run at startup. Two aliased runs are byte-identical. Fixing it needs
the plan cached rather than measured.

**Codex review caught two real defects worth remembering as patterns:** binding
was not actually all-or-nothing though a comment claimed it was (partial binding
leaves images sharing memory with no barrier), and the plan was never invalidated
on resize (new-sized images at old offsets). Both fixed in `f43cf51`.
Scope was deliberately memory-only: subsystems keep their VkImage/VkImageView and
every descriptor write; only the backing memory would change.

**The numbers (2560x1440, default scene, 23 passes):**
- transient pool was **144.87 MiB / 15 textures**; after the SSR change **123.74 MiB**
- greedy packing → 103.75 MiB originally, **82.62 MiB** now (41.12 MiB, 33.2% reuse)
- **greedy already hits the theoretical floor** (= peak concurrent live bytes), so
  DON'T go looking for a better packing algorithm — there is nothing left.
- cost estimate: ~10 alias handoffs, but each merges into a layout transition the
  resource *already* has, so it's widened barrier scopes, NOT 10 new barriers.

**MoltenVK supports image memory aliasing** — probed and confirmed (two images,
different formats/extents, one VmaAllocation, validation clean). Gotcha found:
`vmaAllocateMemory` with `VMA_MEMORY_USAGE_AUTO` **asserts and aborts**
(vk_mem_alloc.h:4053) — AUTO only works via vmaCreateBuffer/vmaCreateImage. Use
`USAGE_UNKNOWN` + `requiredFlags = DEVICE_LOCAL` for raw pool allocations.

**Shipped along the way: half-resolution SSR scene-colour copy** (`dc01e60`).
28.38 → 7.25 MiB, saves 21.13 MiB with no aliasing machinery, and lowers the
floor. The trace takes exactly ONE point sample from that copy, so full size was
buying nothing. Uses `vkCmdBlitImage` (copy can't rescale) and carries its own
uvScale in `SsrParams::subRect.zw` because halving rounds each side
independently. Measured visual cost: 0.23% of pixels, worst delta 15/255,
confined to reflection edges. Golden re-baselined deliberately.

**PHASE 4 DONE — MEASURED AND REJECTED** (`e9378f7`). A/B with repeated control
(drift 0.48%): aliasing off 15.236 ms, on 15.412 ms, **+0.176 ms (+1.2%)** for
17.48 MiB against a 13 GiB budget. Bad trade → stays OFF by default, and the
remaining ~23.6 MiB is **deliberately NOT wired** (more resources = more
handoffs = more cost, memory stays unscarce). **Don't re-propose extending it as
if it were free.**
Only Frame total is quotable from that series: `PunctualShadowAtlas` and
`ImGuiPass` cleared their noise floor although bloom aliasing cannot touch them,
and `Bloom Downsample Chain` moved OPPOSITE to the frame total = work shifted
across a pass boundary on this TBDR. `Attributable` is a noise-floor check, not
a causality check.
**The one thing that could flip the verdict:** the handoff barrier waits on
`ALL_COMMANDS`/`MEMORY` instead of the union of the predecessors that actually
own the overlapping bytes. Tighten and re-measure BEFORE aliasing anything else.
Harness gotcha: `measure_gpu.py` validates `--b-set` keys against the persisted
`config/runtime_settings.json`, so a setting absent from that file is rejected as
unknown even when the code reads it.

**Superseded original Phase 4 plan**: `VulkanTransientMemoryPool`,
`VulkanImage::createAliased`, the alias-handoff barrier contract (new image
transitions from UNDEFINED + waits on the union of predecessors' last accesses),
bloom chain first, then the rest, then `/measure` decides the default.
`--probe-aliasing` and `VulkanAliasingProbe` were planned as throwaway but are
**deliberately KEPT** (user decision, 2026-08-16): the pool's capability check
only proves the driver accepted a binding, while the probe proves the two images
genuinely share bytes by writing through one and reading back through the other.
Don't "clean them up".

**How to apply:** aliasing bugs corrupt pixels without tripping validation, so
check every phase against the golden job and treat a golden diff as a real bug
until proven otherwise. See [[lavapipe-headless-ci]].
