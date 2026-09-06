#!/usr/bin/env python3
"""Refuse git commands that switch the repository's guards off.

Permission rules match by prefix, so a deny rule on "git commit --no-verify"
does not catch `git commit -m "x" --no-verify` -- the flag can sit anywhere on
the line. This sees the whole command string, so flag position does not matter.

--no-verify skips pre-commit, commit-msg, AND pre-push in one go, which is
every layer of the hygiene guard at once. HYGIENE_SKIP=1 is the escape hatch
that exists for a considered exception; it prompts rather than being blocked.

Reads the PreToolUse payload on stdin and prints a deny decision, or nothing.
"""

import json
import re
import sys

BLOCKED = [
    (re.compile(r"(?:^|\s)--no-verify(?:\s|$)"),
     "--no-verify turns off pre-commit, commit-msg and pre-push together, "
     "which is the whole hygiene guard."),
    (re.compile(r"(?:^|\s)git\s+commit\s+(?:[^|;&]*\s)?-n(?:\s|$)"),
     "git commit -n is --no-verify, which turns off the hygiene guard."),
]

ADVICE = (" Fix what the hook reported instead. If the finding is genuinely "
          "fine, run the same command with HYGIENE_SKIP=1 in front, which "
          "prompts rather than silently bypassing.")


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    command = (payload.get("tool_input") or {}).get("command") or ""
    for rx, why in BLOCKED:
        if rx.search(command):
            json.dump({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": why + ADVICE,
            }}, sys.stdout)
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
