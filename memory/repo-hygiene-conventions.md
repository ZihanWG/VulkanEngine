---
name: repo-hygiene-conventions
description: "Screenshots: only _latest.png is tracked, timestamped F12 captures are gitignored. Plus the 2026-08-06 cleanup outcome (worktrees, stale branches)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 963494d3-081e-45ac-82ef-ea80c9cd19d0
  modified: 2026-08-06T13:19:47.969Z
---

Settled 2026-08-06 (merged `377d292`, pushed).

**Screenshots.** F12 writes `screenshots/vulkan_engine_portfolio_YYYYMMDD_HHMMSS.png`
plus an overwritten `_latest.png`. Only `_latest.png` is tracked; the timestamped
ones are gitignored via `/screenshots/vulkan_engine_portfolio_[0-9]*.png` (the
`[0-9]` is what spares `_latest`). They are 3-15 MB each and nothing references
them — README embeds the demo GIF, `docs/portfolio_capture.md` points at
`_latest`. **Do not offer to `git rm --cached` the eleven already in history**:
the blobs stay in history either way, so it would not shrink the 51 MB repo, only
remove the files from fresh clones.

**Don't commit stray material JSON without diffing it against the code default.**
`assets/materials/portfolio_ground.material.json` showed up untracked looking like
the five tracked portfolio-material siblings, but it was metallic 1.0 / roughness
0.0 while `RendererScene.cpp` creates `Portfolio_Ground` as metallic 0.0 /
roughness 0.86. Since those five DO load from disk, committing it would have
silently turned the default scene's ground into a chrome mirror. It was an
accidental material-inspector save; deleted at the user's direction.

**Worktrees under `.claude/worktrees/` can hide real uncommitted work.** One held
an uncommitted `Renderer.cpp` fix that was NOT in main and fixed a live bug (see
below). Always `git -C <worktree> diff` before removing one, and check whether the
change exists on main — do not assume a merged branch means the worktree is
redundant.

Cleanup done: both worktrees removed, `claude/zealous-greider-e81db1` deleted,
`origin/feature/async-compute-clustered` deleted from the remote. Only `main`
remains, local and remote.

## The bug that was hiding in the worktree

`df527e7`: the `Renderer` constructor called `recreatePostProcessResources()`
before `createPipeline()`, but SSR/GTAO resource creation binds descriptor sets
against pipelines that must already exist and throws "pipeline resources are
missing". So **every startup logged two warnings and both subsystems failed their
first init**, recovering only because the swapchain path re-creates the resources
later. Swapping the calls fixes it; resize keeps the reverse order because its
pipeline-recreate decision reads availability flags that resource creation sets.

Worth remembering as a diagnostic habit: those warnings sat in every startup log
this project ever captured, including logs read earlier the same day for other
reasons. Grep the startup log for `Warn`/`unavailable` occasionally — see
[[instrument-before-guessing-runtime-bugs]].
