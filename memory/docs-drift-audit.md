---
name: docs-drift-audit
description: "Docs drift into saying false things as features ship. Two audits done (02f531a, 5e875b5). Audit by README-reachability, not file age; distinguish a false claim from a misleading-but-true one."
metadata: 
  node_type: memory
  type: project
  originSessionId: 963494d3-081e-45ac-82ef-ea80c9cd19d0
  modified: 2026-08-08T06:49:07.117Z
---

This project's docs are unusually good and unusually honest, which is exactly why
stale claims hurt: **a reader takes every sentence literally, so a "limitation"
the code has overtaken reads worse than a missing doc.** It has now happened
twice — the milestone history stopped for 101 commits (fixed 2a4f3f0), and five
Limitations entries had gone false (fixed `02f531a`, pushed).

## The method that works

Extract every Limitations bullet across all docs at once and check each against
the code, rather than reading docs one by one:

```
for f in docs/*.md; do awk -v F="$(basename $f)" \
  '/^## (Known )?Limitations|^## Limitations and Future Work/{p=1;next} /^## /{p=0} p&&/^- /{print F": "$0}' "$f"; done
```

Then verify the suspicious ones against the shipped feature, grep that the same
claim does not survive elsewhere (intros repeat them), and check cross-doc links
still resolve.

## The distinction that matters — do not just delete

**Three were genuinely false** and were rewritten to say what is still true:
- `clustered_lighting.md` "no shadow-casting punctual lights" — the atlas shipped;
  what is true is that atlas capacity caps how many cast.
- `post_processing.md` "no TAA reprojection" — motion-vector TAA reprojects along
  velocity; what is missing is depth-based *disocclusion classification*.
- `gpu_culling.md` "occlusion uses previous-frame depth" — phase 1 does, phase 2
  re-tests against a pyramid built from this frame's phase-1 depth. Neither uses
  a depth prepass, which is the part worth keeping.

**Two were true but misleading, and deleting them would have been the opposite
error.** `render_graph.md`'s "no async compute scheduling" / "no multi-queue
scheduling" are about **the graph**, which really does not schedule across
queues — the renderer owns that submission and its semaphores. Read cold they
deny the engine has async compute, which it does. Same shape in `profiling.md`.
They were clarified to name *whose* limitation it is.

**So: check whether a stale-looking claim is false or merely scoped to a
subsystem before touching it.**

## Second pass, 2026-08-10 (`5e875b5`) — five more, and the worst one was unlisted

Audited the four longest-untouched docs. All four had a false claim:
`design_decisions.md` (Hi-Z "off by default" / "opt-in" — two-phase fixed the
exact reason it was opt-in), `scene_editing.md` ("no free-camera controller yet"
— EditorCamera exists), `portfolio_capture.md` ("does not implement SSR"),
`asset_system.md` ("no alpha blend/mask pipelines").

**The one that mattered most was not on the list.** `engine_upgrade_audit.md` is
linked from the README **three times**, once as the full list of current
limitations, and still claimed no SSR, no motion vectors, no skinning and no
alpha pipelines — four features the README's own highlight table advertises.
Anyone following the front page landed on a document contradicting it.

**Lesson: audit by *reachability from the README*, not just by file mtime.** A
prominently-linked status doc drifting is far worse than a quiet subsystem doc
drifting, and mtime does not rank them.

## Docs most at risk

Ones whose subsystem kept moving after they were written. Check
`git log -1 --format=%ad --date=short -- docs/X.md` against features shipped
since. Those four (`asset_system.md`, `portfolio_capture.md`, `scene_editing.md`,
`design_decisions.md`) were audited on 2026-08-10 and **every one of them had a
false claim** — the guess that they were "plausibly fine since their subsystems
have been quiet" was wrong. Age alone predicted drift better than subsystem
activity did, because the claims that went false were about *other* subsystems
shipping.

See [[repo-hygiene-conventions]].
