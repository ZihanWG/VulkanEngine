---
name: ssr-measured-cost
description: SSR costs 0.33 ms of a 15.6 ms default-scene frame (2.1%) as of 2026-08-14; the per-pass decomposition is not trustworthy on this TBDR
metadata: 
  node_type: memory
  type: project
  originSessionId: f80a1a05-7e41-4d54-985f-d03ef6f409e7
  modified: 2026-08-15T01:31:01.295Z
---

**SSR's measured cost is 0.33 ms of a 15.6 ms frame (2.1%)** on the default
scene at default settings, measured 2026-08-14 at main `f4d2c6f` with
`/measure ab --a-set ssr.enabled=true --b-set ssr.enabled=false --repeat 2`.
Control drift 0.20%, `Frame total` delta 0.332 ms against its own control drift
of 0.031 ms — a 10x margin, 38/38 block coverage on both sides.

**Quote only the frame-level number.** The per-pass decomposition contradicts
itself: `MainHDRPass` +0.790 ms and `Transparent` −0.728 ms when SSR is turned
*off*, both clearing their own control drift, nearly cancelling. SSR cannot make
the main HDR pass more expensive by disappearing. The mechanism is the one
[[gpu-profiler-nested-scopes]] describes — `SSRCopy` inserts a pass boundary, and
on this TBDR the deferred fragment work re-attributes across it. SSR's own passes
read `SSRTrace` 0.158 + `SSRCopy` 0.004 = 0.162 ms, only half the frame delta;
the rest is not cleanly attributable to any row.

**Why:** two earlier attempts at this same measurement were void — one on a
binary 17 commits stale (reported `SSRTrace` 0.805 ms, over 3x too high), one at
28.5% control drift from measuring straight after a build. Both failure modes are
now gated in the harness; see [[back-to-back-or-dont-claim]].

**`docs/ssr.md:168` was re-measured on 2026-08-14 and CONFIRMED — do not
"correct" it.** Building the two commits and running A/B/A (`4d4bb46` view-space
vs `62c1893` screen-space march) reproduced the doc's own absolutes: 0.625 vs
0.496/0.522 ms against a documented 0.627 → 0.502. `Frame total` control drift
was 0.001 ms. The later commits touching `ssr_trace.frag` (`617954a`, `6c30a41`)
are comments plus a weight clamp and do not touch the march.

**`SSRTrace` has no quotable cost — settled, do not re-derive.** An A/B/A across
`f4d2c6f` and `62c1893` found 0.439 / 0.494 / 0.458 ms: the two builds differ by
0.046 ms against a control drift of 0.019, too thin to claim a change. Across
four runs of *identical* build and settings today it read 0.106, 0.243, 0.439,
0.458 — a 4x spread — with single frames ranging 0.003 to 1.186 ms. The earlier
0.158 ms figure was that instability, not a real per-commit change, and the
suspicion that SSR got 3x cheaper after `62c1893` is refuted. Measuring this pass
at all needs far more than 30 samples.

**How to apply:** quote `Frame total` for SSR, never `SSRTrace`. The harness now
marks such rows `unstable` (one-sided row whose own control moved more than a
quarter of its median) — see [[back-to-back-or-dont-claim]].
