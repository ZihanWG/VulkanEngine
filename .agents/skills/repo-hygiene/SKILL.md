---
name: repo-hygiene
description: Keep tooling out of everything this repository publishes. Use before a commit, a push, a pull request, or a release, and whenever asked to audit what the repository shows on GitHub.
---

# Repository Hygiene

The repository is a graphics-programming portfolio. A reader should see the
renderer and its author, nothing about what was used to write it. That rule
covers tracked files, paths, commit messages, and the whole GitHub surface
around them: pull request titles and bodies, review comments, and CI run names.

Removing a trace after it is published is expensive. On 2026-08-23 it took a
`git filter-repo` rewrite of 488 commits, a force-push, deleting 41 CI runs,
editing 22 pull requests and 19 review comments, and finally republishing the
repository under a new one because GitHub does not allow deleting a pull
request. Every one of those would have been a two-second fix before the push.

## Two Layers

**The hooks are the enforcement.** `.githooks/hygiene.py` holds the single
pattern table; `pre-commit`, `commit-msg`, and `pre-push` all read it. They run
without an agent and cannot be forgotten. Do not add a second copy of the
pattern list anywhere — the previous version of this guard kept one list in a
hook and another in a CI workflow behind a "keep in sync" comment, they drifted,
and the drift surfaced on a pull request.

**This skill is the reach.** Hooks see git. They cannot see a pull request body,
a CI run title, or a review comment footer, and those are exactly where the
loudest traces ended up last time.

## Scope and Safety

1. Resolve the repository root with `git rev-parse --show-toplevel` and work
   from there.
2. Inspect `git status --short` and preserve every pre-existing change.
3. Treat an audit request as read-only. Editing published content — a pull
   request title, a body, a comment — needs the user's explicit go-ahead each
   time, and deleting anything on GitHub needs it separately.
4. Never propose adding the tooling files back to tracking, and never move the
   exclusions from `.git/info/exclude` into `.gitignore`. A `.gitignore` naming
   `.claude/` is itself the trace.

## Local Check

Run the layer that matches what is about to happen. All four exit non-zero on a
finding and print the file, line, match, and reason.

| Situation | Command |
| --- | --- |
| Before a commit | `.githooks/hygiene.py staged` |
| A message being written | `.githooks/hygiene.py message <file>` |
| Before a push or a pull request | `.githooks/hygiene.py range HEAD origin/main` |
| Whole working set | `.githooks/hygiene.py tree` |
| Everything ever committed (~3s) | `.githooks/hygiene.py history` |

`.githooks/test-hygiene.sh` exercises the hooks in a throwaway repository — 21
cases covering each pattern class, the false positives that matter, and the
escape hatch. Run it after touching the pattern table; a guard nobody tested is
worse than no guard, because it is trusted.

The two false positives worth knowing: `magenta` contains `agent`, and assimp's
`aiScene` contains `ai`. Both are covered by word boundaries and both are in the
test. If a new pattern is added, add its false-positive case too.

Vendored code under `external/` is checked for the strong markers only. Upstream
may legitimately say "OpenAI"; it never ships our `.claude` directory.

## GitHub Check

Hooks cannot reach any of this. Run it before opening a pull request and again
before a release.

```bash
R=ZihanWG/VulkanEngine
P='(?i)\bagent|codex|claude|copilot|anthropic|openai|chatgpt|tools/agent'

# Pull request titles and bodies — where the "Generated with" footers land.
gh pr list -R $R --state all --limit 100 --json number,title,body \
  --jq ".[] | select((.title + \" \" + (.body // \"\")) | test(\"$P\")) | \"#\(.number) \(.title)\""

# Your own review comments — the "Addressed by" footer is easy to miss.
for n in $(gh pr list -R $R --state all --limit 100 --json number --jq '.[].number'); do
  gh api /repos/$R/pulls/$n/comments \
    --jq ".[] | select(.user.login==\"ZihanWG\") | select((.body // \"\") | test(\"$P\")) | \"#$n \(.id)\""
done

# CI runs — the workflow name, the branch, and the title are all displayed.
gh run list -R $R --limit 1000 --json workflowName,displayTitle,headBranch \
  --jq ".[] | select((.workflowName + \" \" + .displayTitle + \" \" + .headBranch) | test(\"$P\"))"

# Metadata.
gh api /repos/$R --jq '"desc: \(.description // "-")  topics: \(.topics)"'
```

A branch name is the one thing to get right the first time. It shows on the pull
request forever: GitHub will not rename the branch of a merged pull request, and
will not delete a pull request at all. Name branches `feature/`, `perf/`,
`refactor/`, `docs/`, `ci/`.

## Reporting a Finding

State where it is, whether it is reachable by a stranger, and what removing it
costs — those three decide whether it is worth acting on:

| Location | Cost to remove |
| --- | --- |
| Staged, uncommitted | Edit the line |
| Committed, unpushed | Amend or rebase |
| Pushed, in the last commits | History rewrite and force-push, all SHAs change |
| A pull request title, body, or comment | Editable through the API |
| A CI run title | Delete the run |
| A merged pull request's branch name | **Not removable.** Republish or accept |

Do not quietly widen the scope. Finding one footer in one body is not licence to
rewrite history; say what you found, say what it would take, and let the user
choose.

## When the Pattern Table Changes

The table is `STRONG` and `WORDS` in `.githooks/hygiene.py`. `STRONG` is checked
everywhere including vendored code; `WORDS` is skipped under `external/`.

1. Add the pattern with a short reason string — the reason is printed to whoever
   gets blocked, so it should say what to do instead.
2. Add both a blocking case and a false-positive case to
   `.githooks/test-hygiene.sh`.
3. Run `.githooks/test-hygiene.sh`, then `.githooks/hygiene.py tree` and
   `history` to confirm the new pattern does not fire on work already committed.

## What Stays Untracked

`AGENTS.md`, `CLAUDE.md`, `.claude/`, `.agents/`, `.githooks/`. They live on
disk, they still govern the work, and they are excluded through
`.git/info/exclude`. `core.hooksPath` must point at `.githooks` for any of the
hooks to run:

```bash
git config core.hooksPath .githooks
```

That setting is per-clone and cannot be committed. Worse, `.githooks/` is itself
untracked, so **a fresh clone has no hooks at all** — `core.hooksPath` then names
a directory that does not exist, git runs nothing, and says nothing about it.
Verified: a commit reading `// by Claude` goes straight into a fresh clone.

`.claude/hooks/session-start.sh` closes that on the paths it can reach. It
mirrors the hooks to `~/.claude/vulkanengine-hooks/` and copies them back when a
clone arrives without them, so a session in a fresh clone is guarded from its
first commit. It cannot help a clone that is only ever used from a terminal, so
when a trace gets through, check in this order:

```bash
git config core.hooksPath        # expect .githooks
ls .githooks/hygiene.py          # expect it to exist and be executable
.githooks/test-hygiene.sh        # expect 21 passed, 0 failed
```
