#!/bin/bash
# Exercises the hygiene hooks in a throwaway repository.
set -uo pipefail

HOOKS="/Users/zihanw/Projects/VulkanEngine/.githooks"
T="$(mktemp -d)"
pass=0; fail=0

check() { # check <expect: pass|block> <label> <command...>
    local expect="$1" label="$2"; shift 2
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    local got=pass; [ $rc -ne 0 ] && got=block
    if [ "$got" = "$expect" ]; then
        pass=$((pass+1)); printf '  ok    %-52s (%s)\n' "$label" "$got"
    else
        fail=$((fail+1)); printf '  FAIL  %-52s expected %s got %s\n%s\n' \
            "$label" "$expect" "$got" "$out"
    fi
}

cd "$T"
git init -q .
git config user.name "Test"; git config user.email "t@example.com"
git config core.hooksPath "$HOOKS"
mkdir -p external/vendor src docs

# ---- a clean baseline commit, so later tests have a HEAD -------------------
echo "float shade() { return 1.0; }" > src/shade.cpp
git add -A
check pass "clean first commit" git commit -q -m "Add a shading helper"

# ---- staged content --------------------------------------------------------
echo "// reviewed by Claude" > src/dirty.cpp; git add src/dirty.cpp
check block "content: tool identity in a staged line" "$HOOKS/pre-commit"
git reset -q; rm src/dirty.cpp

echo "see AGENTS.md for the rule" > docs/x.md; git add docs/x.md
check block "content: AGENTS.md reference" "$HOOKS/pre-commit"
git reset -q; rm docs/x.md

printf 'run tools/agent/verify_renderer.sh\n' > docs/y.md; git add docs/y.md
check block "content: old tools/agent path" "$HOOKS/pre-commit"
git reset -q; rm docs/y.md

printf 'done\n\n\xf0\x9f\xa4\x96 Generated with [Claude Code](https://claude.com/claude-code)\n' > docs/z.md
git add docs/z.md
check block "content: generated-with footer" "$HOOKS/pre-commit"
git reset -q; rm docs/z.md

# ---- paths -----------------------------------------------------------------
mkdir -p tools/agent; echo hi > tools/agent/run.sh; git add tools/agent/run.sh
check block "path: a directory component named agent" "$HOOKS/pre-commit"
git reset -q; rm -rf tools/agent

echo "policy" > AGENTS.md; git add -f AGENTS.md
check block "path: AGENTS.md staged" "$HOOKS/pre-commit"
git reset -q; rm AGENTS.md

mkdir -p .claude; echo "{}" > .claude/settings.json; git add -f .claude/settings.json
check block "path: .claude/ staged" "$HOOKS/pre-commit"
git reset -q; rm -rf .claude

# ---- false positives that must NOT block ----------------------------------
echo "magenta means nothing resident" > docs/colour.md; git add docs/colour.md
check pass "no false positive: 'magenta' contains 'agent'" "$HOOKS/pre-commit"
git reset -q; rm docs/colour.md

echo "const aiScene* scene = importer.ReadFile(path);" > src/import.cpp
git add src/import.cpp
check pass "no false positive: assimp aiScene" "$HOOKS/pre-commit"
git reset -q; rm src/import.cpp

echo "// Portions contributed by OpenAI staff, MIT" > external/vendor/lib.h
git add external/vendor/lib.h
check pass "vendored external/ exempt from identity words" "$HOOKS/pre-commit"
git reset -q

echo ".claude" > external/vendor/.gitignore; git add external/vendor/.gitignore
check block "vendored external/ still caught for .claude" "$HOOKS/pre-commit"
git reset -q; rm -f external/vendor/.gitignore external/vendor/lib.h

# ---- commit messages -------------------------------------------------------
echo "ok" > src/a.cpp; git add src/a.cpp
check block "message: subject names the tooling" \
    git commit -q -m "Add Codex and OpenAI to the agent list"
check block "message: body names a session link" \
    git commit -q -m "Do a thing" -m "See https://claude.ai/code/session_01AB"
check pass "message: clean subject and body" \
    git commit -q -m "Add a helper" -m "Because the caller needed one."

# the trailer is stripped rather than rejected
echo "ok" > src/b.cpp; git add src/b.cpp
git commit -q -m "$(printf 'Add another helper\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n')" 2>/dev/null
if git log -1 --format='%B' | grep -qi 'co-authored-by'; then
    fail=$((fail+1)); echo "  FAIL  trailer should have been stripped"
else
    pass=$((pass+1)); printf '  ok    %-52s (stripped)\n' "message: attribution trailer"
fi

# ---- escape hatch ----------------------------------------------------------
echo "// Claude" > src/c.cpp; git add src/c.cpp
check pass "HYGIENE_SKIP=1 bypasses" env HYGIENE_SKIP=1 "$HOOKS/pre-commit"
git reset -q; rm src/c.cpp

# ---- range mode ------------------------------------------------------------
base="$(git rev-list --max-parents=0 HEAD)"
check pass "range: clean history against a base" "$HOOKS/hygiene.py" range HEAD "$base"
check pass "range: clean history with no base" "$HOOKS/hygiene.py" range HEAD

echo "// by Claude" > src/d.cpp; git add src/d.cpp
git commit -q --no-verify -m "Sneak one in"
check block "range: a --no-verify commit is caught later" \
    "$HOOKS/hygiene.py" range HEAD "HEAD~1"
check block "range: same commit caught with no base" "$HOOKS/hygiene.py" range HEAD
git reset -q --hard HEAD~1

echo
echo "passed=$pass failed=$fail"
cd /; rm -rf "$T"
[ "$fail" -eq 0 ]
