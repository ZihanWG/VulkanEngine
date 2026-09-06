---
name: frame-capacity-overflow
description: "Frame-cap overflow counting SHIPPED (47f24b7): over-cap geometry is counted, not silently dropped; BOTH halves runtime-verified by forcing the caps down — the ImGui half via --capture-include-ui (2026-08-23)"
metadata:
  node_type: memory
  type: project
---

**MERGED to main via PR #16 as `47f24b7` on 2026-08-19, pushed; branch and its
worktree deleted.** It reached main as `dbdd2ce`, rebased onto `b13f19e`.
CI step lists were read rather than trusting the one-line verdict: the lavapipe
job really ran `Render and gate on validation` **and** `Compare against the golden
image`, so the change is confirmed **pixel-neutral** on the CI scene, which is the
right expectation for a pure counting change.

What it does: `kMaxFrameObjects` / `kMaxDrawItems` (both 8192) used to drop
geometry **silently** — five scattered `std::min` clamps and a bare
`return false`, so an over-cap scene lost meshes with no log, counter, or
indicator. `renderer::FrameCapacityBudget` (GPU-free, in `VulkanEngineCore`)
now counts the losses, driven by the same calls that make the decision —
`admitDrawItem()` *is* the capacity check, so the counts cannot drift from the
loop. The caps are unchanged; only the reporting is new. 445 lines, 11 tests,
`fast` gate 389/389 exit 0 after the rebase (378 on main + 11 new). Two counters, not one: over-cap objects are dropped
whole and are invisible to culling/shadows/GPU-cull input, so their draw items
are never counted as dropped draw items.

The caps clamp rather than grow because several device buffers are sized from
them at init and are already bound into descriptor sets and referenced by
in-flight command buffers — growing them needs a device-idle wait (banned on the
steady-state frame path) or a deferred destroy plus descriptor rewrite that does
not exist. Ceiling is 65535 from `cull.comp`'s `firstInstance` packing.

**`claude/relaxed-goodall-fa2449` is GONE** — diff discarded, branch and worktree
deleted on the user's instruction, 2026-08-19. It had held an uncommitted change
adding `normalBias` / `cascadeBlend` to the hand-written CSM copy list in
`Renderer::applyRuntimeSettings`. main had already fixed this **better** in
`5b9deb9`: the policy moved into `applyCsmSettings` in `RuntimeSettings.h` as a
whole-struct assignment with a defaulted `operator==` regression test, so a
future forgotten field fails the suite instead of silently not loading. Recorded
because the lesson generalises — **an agent worktree left open across a merge can
hold the losing half of a decision that main already settled**; check a stale
worktree's diff against main before assuming it is unfinished work worth keeping.

**RUNTIME-VERIFIED on the M3, 2026-08-19, by forcing the caps down** — the same
technique that cracked [[mesh-cook]]. Debug build, `--scene stress`
(2322 render objects), `--exit-after-frames 90`, A/B/A:

| caps | result |
| --- | --- |
| 8192 / 8192 (control) | 2322 draw items, **0 warnings** |
| **objects 1000** | "1322 render object(s) past the 1000-object cap", **no** draw-item clause |
| **draw items 1000** | "1322 draw item(s) past the 1000-draw-item cap", **no** object clause |
| 8192 / 8192 (restored) | 2322 draw items, **0 warnings** |

Every number is exactly 2322 - 1000, and **0 validation errors in every run**.
Two things this proves that the unit tests cannot:

1. **The "keep offering after the cap fills" loop works.** The draw-item run
   reported **1322, not 1** — the old early `return false` could only ever have
   produced 1. That was the single most review-worthy line in the change.
2. **The two counters really are independent in the real frame.** Each forced run
   printed only its own clause, confirming that objects dropped whole do not also
   get counted as dropped draw items.

The warning fired **once across 90 frames**, so the log-on-change latching holds.

**The ImGui amber line is now VERIFIED (2026-08-23), and the "cannot get one from
a session" premise is gone with it.** `--capture-include-ui` (PR #22) takes the
capture copy AFTER the ImGui pass instead of before it, so the debug panel finally
exists in an image a scripted run can produce. Off by default; the default capture
path is byte-for-byte unchanged, and the portfolio screenshot keeps its own copy
point unconditionally.

Verified the same way the log half was — caps forced to 1000 against the
2322-object stress scene, deterministic capture, 0 validation errors. The amber
line reads **"1322 object(s) and 0 draw item(s) dropped"** directly under
"Total objects: 1000": the log's number exactly, and the independence property
(objects dropped whole are not also counted as dropped draw items) now visible in
the UI rather than only in the log.

**Trap this cost a first attempt:** after the overlay the swapchain image is past
the graph's present transition, so restoring it to COLOR_ATTACHMENT (what the
pre-overlay copy does) makes vkQueuePresentKHR reject it. `recordCopy` takes the
restore layout as a parameter now. The thing that caught it was running with
`--fail-on-validation-error`, not re-reading the code.

**Found by looking:** the warning text is clipped at the default window width
rather than wrapping. Recorded, not fixed — and nothing but a capture of the
overlay would ever have shown it.

Related: [[cpu-frame-prep-cost]], [[gpu-cpu-struct-layout]],
[[runtime-settings-persistence]], [[repo-hygiene-conventions]].
