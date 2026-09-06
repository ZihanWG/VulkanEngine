---
name: back-to-back-or-dont-claim
description: "A sub-millisecond GPU pass measured once per configuration will lie; two such claims were wrong this way, both caught later"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f80a1a05-7e41-4d54-985f-d03ef6f409e7
  modified: 2026-08-14T12:01:31.352Z
---

**One run per configuration is not a measurement.** Twice now a cost claim made
that way was wrong:

- The composite sharpen filter, measured once with and once without, read
  0.416 vs 0.424 ms — reported as "not measurable". Back-to-back A/B/A/B on the
  same build gave 0.362 / 1.460 / 0.474 / 1.227: it roughly **triples** the pass.
- Earlier, an ablation series drifted 10.885 → 11.879 ms on the *control* over
  four runs (see [[mainhdrpass-attribution]]).

**Why:** GPU timestamp medians for a sub-millisecond pass on this TBDR are wide
enough that two unpaired runs routinely overlap, and the machine degrades over a
session — after a long run of GPU launches the scale-0.5 control read 13-14 ms
against 6.2 ms measured cold, so absolute numbers from late in a session are not
comparable to early ones.

**How to apply:** this protocol is now scripted — use the `/measure` slash
command (`.claude/commands/measure.md`, driving `tools/agent/measure_gpu.py`)
instead of hand-rolling a series. `ab --repeat 2` interleaves A/B/A/B, discards a
warm-up, reports medians with min/max, and **exits non-zero when the repeated
control drifts over 1%**. Crucially it also computes drift **per pass** and
marks a row not attributable when its own control moved as much as the A/B
delta — the frame-level gate alone would have passed the sharpen case above.
Release compiles validation layers out, so a measurement run is never evidence
about validation errors; the report says so rather than staying silent. It patches `config/runtime_settings.json` (and restores
it), so ImGui-only scene presets still need a hand-captured log fed to `parse`.
**The 1% frame-level gate can be unreachable on this machine, and that is not a
reason to keep rerunning.** During the CSM cascade-cache work two consecutive
series failed it (1.3%, then 1.6% with `--repeat 3` -- more repeats made it
*worse*, since a longer series is more thermal exposure). The cause was visible
in `ps -Ao %cpu,comm -r`: the **Claude desktop app's own WindowServer/renderer
processes were taking ~65% CPU and contending for the GPU**, which does not
settle while the session is open. Check the load *before* burning six 30s runs;
when the contention is structural, stop and report honestly instead of
rerunning. Two useful things survive a voided series: a row whose **own** control
drift is orders of magnitude below its delta and that **reproduces across two
independent series** is still real evidence (say so, and say the frame total is
not quotable), and **presence/absence in the `Blocks` column is immune to
timing noise entirely** -- `ShadowGpuCulling` at 0/57 vs 57/57 frames proved the
cascade pass was skipped without any timing claim at all.

For any pass under ~2 ms, run A/B/A/B back to back in one
command and report the ratio; quote absolutes only from a quiet machine, and say
so when the machine was not quiet. If only one pair exists, say "not measured"
rather than "no cost" — a wrong free-lunch claim survives into docs and commit
messages and has to be retracted in public. Related: [[gpu-profiler-nested-scopes]],
[[exposure-reduce-serial-bottleneck]].
