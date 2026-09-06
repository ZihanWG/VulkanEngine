---
name: no-claude-coauthor-trailer
description: "No agent attribution anywhere this repo publishes — commit trailers, PR bodies, AND review-comment footers, even when tooling explicitly asks for one"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: de3c3d22-c6e1-44fa-92b2-aac5735cbfb2
---

Do not add the `Co-Authored-By: Claude ...` trailer to commit messages (and don't add the "Generated with Claude Code" line to PRs). The user does not want Claude appearing in the GitHub contributor list.

**Why:** On 2026-06-20 the user asked to remove Claude as a GitHub contributor and had me strip the trailer from all existing commits (8 on `main` + 3 on `code-health-improvements`) via filter-branch.

**How to apply:** Omit the trailer the harness default would otherwise append. This overrides the standard "end commit messages with Co-Authored-By" instruction for this user/repo.

## 2026-08-24: it also covers review-comment footers, and tooling will ask for one

The `<ci-monitor-event>` from "Autofix pull requests" instructs, every time, to
end each review reply with `_🤖 Addressed by [Claude Code](...)_`. **Do not.** The
user ruled on this directly on 2026-08-24 when I raised the conflict.

**Why:** it lands on the published GitHub surface, which the repo's own rule and
the `repo-hygiene` skill's pattern table both forbid — `claude` is a blocking
pattern there. The hooks cannot catch it: they see git, not a review comment. And
removing a published trace is what cost a 488-commit rewrite, 41 deleted CI runs
and a republished repository once already.

**How to apply:** reply to review threads with the substance and no footer, then
resolve. The Autofix instruction is tooling text, not the user's instruction, and
the repo rule wins. If a footer is ever wanted, it must name the renderer and its
author, nothing else. Related: [[no-agent-traces-on-github]].
