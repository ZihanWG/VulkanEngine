---
name: gpu-ab-needs-matched-inputs
description: A cross-tree GPU A/B is only valid if assets, SPIR-V and the actual branch match — two real confounds caught in one session
metadata:
  type: feedback
---

Comparing two builds on the GPU needs the *inputs* proven equal, not assumed.
Two separate confounds hit the same A/B run (2026-08-30):

1. **Untracked cooked assets.** `assets/textures/*.ktx2` (BC7, from the texture
   cook) are untracked, so a fresh `git worktree` has only the tracked `.png`s.
   BC7 is lossy, so the same commit rendered **10 pixels different at delta 1**
   with GI probes on. It looked exactly like a real regression, and the
   same-binary control was clean, which made it look *more* real.
2. **The shared checkout moved.** A background-task session switched
   `/Users/zihanw/Projects/VulkanEngine` to its own branch and merged a PR, so the
   "after" binary was another agent's code. The tell was a log line my change adds
   being **absent** from the run.

**Why:** a control that runs the same binary twice only rules out nondeterminism.
It cannot see a difference in assets, shaders, or which commit is checked out.

**How to apply:** before trusting a cross-tree GPU A/B —
- copy untracked assets into both trees, then `diff` a `shasum` listing of
  `assets/` and `cmp` every `build/*/shaders/*.spv`;
- run **both** sides from worktrees you created, never from the shared checkout,
  which another session may move under you (see [[no-agent-traces-on-github]] for
  why worktrees are the house rule);
- assert a log line that only the new code emits, so "the right binary ran" is
  evidence rather than assumption — the same trick as
  [[instrument-before-guessing-runtime-bugs]];
- `--fail-on-validation-error` turns validation into an exit code, so a matrix
  can be a table of exit codes instead of greps.

**The shared checkout is not yours alone.** A background task can switch its
branch and merge a PR while you work. Do branch surgery (rebase included) in a
worktree, never in `/Users/zihanw/Projects/VulkanEngine`, and check
`git branch --show-current` before trusting anything built there.

Related: [[back-to-back-or-dont-claim]], and the stale-binary A/Bs in
[[csm-cascade-cache]].
