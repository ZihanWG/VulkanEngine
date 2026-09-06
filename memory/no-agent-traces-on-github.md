---
name: no-agent-traces-on-github
description: "The repo must carry no agent/AI traces on GitHub; AGENTS.md, CLAUDE.md, .claude/, .agents/, .githooks/ are deliberately untracked and history was rewritten on 2026-08-23."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fff8c1f9-efdc-40ee-982c-2a8de65dc358
  modified: 2026-08-23T07:16:44.218Z
---

The GitHub repository must show **no trace of agent or AI tooling** — not in
tracked files, not in commit messages, not in paths. On 2026-08-23 the user
asked for this directly ("不要在github留agent，ai痕迹") and chose a full history
rewrite.

**Why:** the repo is a graphics-programming portfolio. A reader should see the
renderer and its author, not the tooling used to write it.

**How to apply:**

- `AGENTS.md`, `CLAUDE.md`, `.claude/`, `.agents/`, `.githooks/`, and
  `.github/workflows/attribution.yml` still exist **on disk and still govern the
  work** — they are untracked via `.git/info/exclude`, deliberately *not*
  `.gitignore` (a `.gitignore` naming `.claude/` is itself the trace). Never
  `git add` them, and never re-add them to `.gitignore`.
- The verification/measurement scripts live at **`tools/dev/`**, not
  `tools/agent/` — the directory was renamed throughout history. See
  [[back-to-back-or-dont-claim]].
- Do not cite `AGENTS.md` by name in tracked docs, code comments, or commit
  messages; state the rule directly instead.
- `.githooks/commit-msg` still strips attribution trailers and `core.hooksPath`
  still points at it, so the local guard survives untracking. See
  [[no-claude-coauthor-trailer]].
- History was rewritten with `git filter-repo` (488 → 480 commits, all SHAs
  changed) and force-pushed. Backup bundle:
  `/Users/zihanw/Projects/VulkanEngine-backup-93556f4.bundle`.

**Also scrubbed the GitHub surface:** 41 CI runs naming the tooling in workflow
name, branch, or title were deleted (496 → 455), which took the Attribution
workflow out of the Actions sidebar; 22 PR titles/bodies were rewritten to drop
the `🤖 Generated with [Claude Code]` footers, the `[Codex Task]` link, and the
`tools/agent/` path; 19 of the user's own review comments had the same footers
stripped. Repo metadata, issues, releases, and tags were already clean.

**Then the repo was republished, which closed the last gap.** The PR branch
names (`claude/…`, `codex/…`) and the `chatgpt-codex-connector` reviews on 18 of
24 PRs could not be deleted — GitHub does not allow deleting a pull request — so
the old repo was renamed `VulkanEngine-archive` and set **private**, and a fresh
public `ZihanWG/VulkanEngine` was created at the same URL and pushed. The new
repo has 0 PRs, 0 issues, one branch, and all three CI pipelines green. Cost: the
one star and the PR/issue record. The user still has to delete the archive by
hand — the `gh` token has `repo` and `workflow` but **no `delete_repo` scope**.

**The rule is now enforced mechanically, so stop relying on memory for it.**
`.githooks/hygiene.py` holds ONE pattern table read by `pre-commit`,
`commit-msg` (strips trailers, then refuses prose it cannot repair), and
`pre-push` (scans the whole outgoing range, catching `--no-verify` commits).
`.githooks/test-hygiene.sh` is 21 cases and passes. Modes: `staged`, `message`,
`range HEAD [BASE]`, `tree`, `history` (~3s). Escape hatch `HYGIENE_SKIP=1`. `--no-verify` is refused outright by a
PreToolUse hook (`.claude/hooks/block-verify-bypass.sh`) because permission
rules match by prefix and cannot catch a flag in the middle of a line.
The `repo-hygiene` skill covers the GitHub surface hooks cannot see.
**A fresh clone has NO hooks** — `.githooks/` is untracked, so `core.hooksPath`
names a missing directory and git silently runs nothing (verified: a `// by
Claude` commit walked straight in). `.claude/hooks/session-start.sh` now
mirrors the hooks to `~/.claude/vulkanengine-hooks/` and restores them when a
clone arrives without them; a terminal-only clone is still unguarded. **Do not
add a second copy of the pattern list anywhere** — that is exactly what caused
the PR #20 drift. Known false positives, both covered by tests: `magenta`
contains `agent`, assimp's `aiScene` contains `ai`.

**Bulk GitHub mutations get blocked by the auto-mode classifier** (deleting many
runs, PATCHing many comments). Hand the user a one-liner or a script instead of
retrying — that worked twice here.
