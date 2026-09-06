#!/bin/bash
#
# One-time setup for the Windows clone. Run from the repository root in Git Bash:
#
#   bash setup-windows.sh
#
# Git for Windows ships bash and awk, so the shell scripts in .githooks/ and
# tools/dev/ run as they do on macOS. What it does not ship is a `python3`, and
# four scripts here resolve their interpreter through `#!/usr/bin/env python3`:
# the three git hooks (via hygiene.py) and .claude/hooks/block-verify-bypass.sh.
#
# The two fail in opposite directions, which is why this matters. A git hook
# that cannot start exits non-zero and git refuses the commit -- loud, safe. But
# a PreToolUse hook only blocks on exit code 2; a missing interpreter exits 127,
# which Claude Code treats as a non-blocking error and runs the tool anyway. So
# block-verify-bypass.sh stops denying and says nothing about it.
#
# Rather than teach each caller a second interpreter name, put the name they
# already ask for on PATH.

set -euo pipefail

if [ ! -f .githooks/hygiene.py ]; then
    echo "error: run this from the repository root (no .githooks/hygiene.py here)" >&2
    exit 1
fi

# 1. python3 shim ------------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    echo "ok: python3 already on PATH -- $(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    mkdir -p "$HOME/bin"
    printf '#!/bin/sh\nexec python "$@"\n' > "$HOME/bin/python3"
    chmod +x "$HOME/bin/python3"
    echo "created: $HOME/bin/python3 -> $(command -v python)"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "note: ~/bin is not on PATH in this shell. Git Bash picks it up on the"
        echo "      next login shell; open a new terminal and re-run to confirm."
        export PATH="$HOME/bin:$PATH"
    fi
else
    echo "error: no python or python3 found. Install Python, then re-run." >&2
    exit 1
fi

# 2. exec bits ---------------------------------------------------------------
# A zip transfer drops the executable bit, and the hooks are exec'd by path.
chmod +x .githooks/* .claude/hooks/* tools/dev/*.sh tools/dev/*.py 2>/dev/null || true
echo "ok: exec bits set on .githooks/, .claude/hooks/, tools/dev/"

# 3. hooks wired -------------------------------------------------------------
git config core.hooksPath .githooks
echo "ok: core.hooksPath = $(git config --get core.hooksPath)"

# 4. prove both guards actually run ------------------------------------------
if .githooks/hygiene.py message /dev/null >/dev/null 2>&1; then
    echo "ok: hygiene.py runs"
else
    echo "FAIL: hygiene.py did not run -- commits will be refused" >&2
    exit 1
fi

probe='{"tool_name":"Bash","tool_input":{"command":"git commit -m x --no-verify"}}'
if printf '%s' "$probe" | .claude/hooks/block-verify-bypass.sh 2>/dev/null | grep -q deny; then
    echo "ok: block-verify-bypass.sh denies --no-verify"
else
    echo "FAIL: block-verify-bypass.sh did not deny. The guard is OFF and will" >&2
    echo "      stay off silently -- do not commit until this passes." >&2
    exit 1
fi

echo
echo "Setup complete. Both guards verified live, not just installed."
