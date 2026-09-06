---
name: user-prefers-phased-checkpoints
description: User likes large work broken into phases with a commit + checkpoint between each
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 23f06752-0feb-47dc-97c2-9bdca3b36d09
---

For large multi-part work on VulkanEngine, the user engaged well with a phased approach: each phase committed separately, verified (build + tests), then a short checkpoint via AskUserQuestion to pick the next phase.

**Why:** Keeps risky/large changes reviewable and lets the user steer ordering (they chose phase order at each step).

**How to apply:** Default to per-phase commits with verification, and check in between large phases rather than doing everything in one shot. Don't over-ask for trivial steps. See [[vulkanengine-improvement-roadmap]].
