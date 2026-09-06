#!/bin/bash
#
# Points a session at this repository's identity and its hooks.
#
# Remote and web containers ship a git identity of their own, so a commit made
# in one is authored by the container rather than by this repository's author.
# That is invisible locally and shows up on GitHub as an extra contributor, so
# set the identity per clone instead of trusting whatever the environment has
# in its global config.
#
# The hooks in .githooks/ are deliberately untracked -- a guard whose job is to
# keep tooling out of the published repository cannot itself be published. That
# has a consequence worth naming: a fresh clone has no .githooks/ directory at
# all, core.hooksPath then points at nothing, and git runs no hooks and says
# nothing about it. The guard would be gone and the first symptom would be a
# trace on GitHub.
#
# So this mirrors the hooks to a copy outside the repository and restores them
# when a clone arrives without them. The in-repo copy is the source of truth
# whenever it exists; the mirror only fills a gap.
set -euo pipefail

readonly MIRROR="${HOME}/.claude/vulkanengine-hooks"
readonly HOOKS=".githooks"

git config user.name "Zihan Wang"
git config user.email "zihanwang7@outlook.com"
git config core.hooksPath "$HOOKS"

if [ -f "$HOOKS/hygiene.py" ]; then
    # Normal case: refresh the mirror from the working copy.
    mkdir -p "$MIRROR"
    cp -p "$HOOKS"/* "$MIRROR"/ 2>/dev/null || true
elif [ -f "$MIRROR/hygiene.py" ]; then
    # Fresh clone: put the guard back before anything can be committed without it.
    mkdir -p "$HOOKS"
    cp -p "$MIRROR"/* "$HOOKS"/ 2>/dev/null || true
    chmod +x "$HOOKS"/* 2>/dev/null || true
    echo "session-start: restored $HOOKS/ from $MIRROR (fresh clone)" >&2
else
    echo "session-start: WARNING -- no $HOOKS/hygiene.py and no mirror at $MIRROR." >&2
    echo "session-start: commits are UNGUARDED. Nothing checks for tooling traces." >&2
fi
