---
name: proceed-with-recommendation
description: "When I have already measured and formed a clear recommendation, act on it instead of stopping to ask which option to take."
metadata:
  node_type: memory
  type: feedback
---

The user rejected an `AskUserQuestion` that offered three ways to attack a
measured bottleneck (each with a stated recommendation) and replied simply
"继续推进" — keep going.

**Why:** by that point the measurement was done and the recommendation was
already argued in the response. Re-asking added a round trip without adding
information. Across this session every such question was answered by picking the
option I had recommended.

**How to apply:** present the measurement and the recommendation, then *do* it.
Reserve a question for cases where the options differ in a way I genuinely cannot
settle from evidence — scope that changes what ships, a destructive or
outward-facing action, or a preference about the repo that is the user's to make
(e.g. where cooked artifacts live). Announce the choice and proceed; the user
redirects if they disagree, which is cheaper than blocking.

Note this coexists with [[user-prefers-phased-checkpoints]]: keep committing per
phase and reporting, just do not gate each phase on a multiple-choice question.
