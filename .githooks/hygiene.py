#!/usr/bin/env python3
"""Repository hygiene scan: keep tooling out of what gets published.

The repository is a portfolio. A reader should see the renderer and its author,
nothing about what was used to write it. That is easy to hold for one commit and
impossible to hold by memory across hundreds, so it is checked mechanically.

One pattern table, used by every caller. The previous version of this guard kept
two copies of the list in two files with a "keep in sync" comment, they drifted,
and the drift was only discovered on a pull request. There is exactly one table
below and every mode reads it.

Modes:

    hygiene.py staged            what `git commit` is about to record
    hygiene.py message FILE      a commit message file
    hygiene.py range HEAD [BASE] commit messages and added lines HEAD adds
    hygiene.py tree              every tracked file at HEAD
    hygiene.py history           every commit message and every reachable blob

`range` with no BASE means every commit reachable from HEAD, compared against
the empty tree. That is the shape a brand-new branch has, and it is why BASE is
optional rather than folded into an A..B string: `git log` cannot take a tree.

Exit 0 clean, 1 dirty, 2 on a usage or git error. Set HYGIENE_SKIP=1 for a
deliberate one-off exception; `git commit --no-verify` bypasses it entirely.
"""

import os
import re
import subprocess
import sys

# --------------------------------------------------------------------------
# The pattern table. Single source of truth.
# --------------------------------------------------------------------------

# Paths that exist on disk, govern the work, and must never be tracked.
PATH_DENY = re.compile(
    r"(^|/)(\.claude|\.agents|\.githooks)(/|$)"
    r"|(^|/)(AGENTS|CLAUDE)\.md$"
    r"|(^|/)attribution\.ya?ml$"
)

# A path component that reads as tooling regardless of what is inside it.
PATH_WORD = re.compile(
    r"(^|/)(agent|agents|codex|claude|copilot)([/._-]|$)",
    re.IGNORECASE,
)

# Unambiguous markers. Checked everywhere, vendored code included: upstream may
# legitimately say "OpenAI", it never ships our .claude directory.
STRONG = [
    (r"\.claude\b", "local tooling directory"),
    (r"\.agents\b", "local tooling directory"),
    (r"\bAGENTS\.md\b", "untracked policy file — state the rule directly instead"),
    (r"\bCLAUDE\.md\b", "untracked policy file"),
    (r"\btools/agent/", "renamed to tools/dev/"),
    (r"claude\.ai", "session or product link"),
    (r"claude\.com/claude-code", "product link"),
    (r"chatgpt\.com", "product link"),
    (r"\U0001f916", "robot emoji, used by generated-by footers"),
    (r"Generated (?:with|by) \[", "generated-by footer"),
    (r"^Co-Authored-By:.*(?:claude|codex|copilot|anthropic|openai)", "attribution trailer"),
    (r"^Claude-Session:", "session trailer"),
]

# Identity words. Skipped inside external/, which is vendored upstream code.
WORDS = [
    (r"\b(?:claude|codex|copilot|anthropic|openai|chatgpt)\b", "tool identity"),
    (r"\bagents?\b", "reads as coding-agent tooling"),
    (r"\bAI\b", "reads as AI tooling"),
]

STRONG_RX = [(re.compile(p, re.IGNORECASE | re.MULTILINE), why) for p, why in STRONG]
WORDS_RX = [(re.compile(p, re.IGNORECASE), why) for p, why in WORDS]

VENDORED = re.compile(r"(^|/)external/")


def rules_for(path):
    """Vendored code gets the strong markers only; ours gets everything."""
    if path and VENDORED.search(path):
        return STRONG_RX
    return STRONG_RX + WORDS_RX


# --------------------------------------------------------------------------

def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True,
                       errors="replace")
    if r.returncode != 0:
        sys.stderr.write("hygiene: git %s failed:\n%s\n" % (" ".join(args), r.stderr))
        sys.exit(2)
    return r.stdout


class Report:
    def __init__(self):
        self.hits = []

    def add(self, where, text, match, why):
        snippet = text.strip()
        if len(snippet) > 100:
            start = max(0, match.start() - 40)
            snippet = "…" + snippet[start:match.end() + 40].strip() + "…"
        self.hits.append((where, match.group(0), why, snippet))

    def scan(self, where, text, path=None):
        for rx, why in rules_for(path if path is not None else where):
            m = rx.search(text)
            if m:
                self.add(where, text, m, why)
                return True
        return False

    def ok(self):
        return not self.hits

    def print(self, what):
        if self.ok():
            return
        sys.stderr.write("\nhygiene: %s\n\n" % what)
        for where, hit, why, snippet in self.hits[:40]:
            sys.stderr.write("  %s\n      matched %r — %s\n      %s\n"
                             % (where, hit, why, snippet))
        extra = len(self.hits) - 40
        if extra > 0:
            sys.stderr.write("  … and %d more\n" % extra)
        sys.stderr.write(
            "\n  Fix the lines above, or if this one is genuinely fine:\n"
            "      HYGIENE_SKIP=1 git commit …\n\n")


def added_lines(diff):
    """Yield (path, lineno, text) for added lines in a unified diff."""
    path, lineno = None, 0
    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            path, lineno = line[6:], 0
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            lineno = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++"):
            yield path, lineno, line[1:]
            lineno += 1
        elif not line.startswith("-") and not line.startswith("\\"):
            lineno += 1


def check_paths(rep, paths):
    for p in paths:
        if not p:
            continue
        if PATH_DENY.search(p):
            rep.hits.append((p, p, "deliberately untracked — do not commit it",
                             "staged for commit"))
        elif PATH_WORD.search(p):
            rep.hits.append((p, p, "path component reads as tooling",
                             "rename the path"))


def check_diff(rep, diff):
    for path, lineno, text in added_lines(diff):
        for rx, why in rules_for(path):
            m = rx.search(text)
            if m:
                rep.add("%s:%d" % (path or "?", lineno), text, m, why)
                break


def strip_comments(msg):
    return "\n".join(l for l in msg.split("\n") if not l.startswith("#"))


def mode_staged(rep):
    check_paths(rep, git("diff", "--cached", "--name-only",
                         "--diff-filter=ACMR").split("\n"))
    check_diff(rep, git("diff", "--cached", "--unified=0", "--diff-filter=ACMR"))
    return "staged changes would publish tooling traces"


def mode_message(rep, path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        msg = strip_comments(fh.read())
    for i, line in enumerate(msg.split("\n"), 1):
        for rx, why in STRONG_RX + WORDS_RX:
            m = rx.search(line)
            if m:
                rep.add("commit message:%d" % i, line, m, why)
                break
    return "the commit message names the tooling"


EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


def mode_range(rep, head, base=None):
    revs = ["%s..%s" % (base, head)] if base else [head]
    for sha in git("rev-list", *revs).split():
        msg = git("log", "-1", "--format=%s%n%b", sha)
        for i, line in enumerate(msg.split("\n"), 1):
            for rx, why in STRONG_RX + WORDS_RX:
                m = rx.search(line)
                if m:
                    rep.add("%s message:%d" % (sha[:8], i), line, m, why)
                    break
    # No base means a brand-new branch: everything it carries is new, so the
    # comparison point is the empty tree. Naming the working tree here instead
    # would compare HEAD against itself and see nothing.
    against = base if base else EMPTY_TREE
    check_paths(rep, git("diff", "--name-only", "--diff-filter=ACMR",
                         against, head).split("\n"))
    check_diff(rep, git("diff", "--unified=0", "--diff-filter=ACMR",
                        against, head))
    return "commits %s carry tooling traces" % (
        "in %s..%s" % (base, head) if base else "reachable from %s" % head[:8])


def mode_tree(rep):
    paths = [p for p in git("ls-files").split("\n") if p]
    check_paths(rep, paths)
    for p in paths:
        try:
            blob = git("show", "HEAD:" + p)
        except SystemExit:
            continue
        if "\0" in blob[:8000]:
            continue
        for i, line in enumerate(blob.split("\n"), 1):
            for rx, why in rules_for(p):
                m = rx.search(line)
                if m:
                    rep.add("%s:%d" % (p, i), line, m, why)
                    break
    return "tracked files at HEAD carry tooling traces"


def mode_history(rep):
    for sha in git("rev-list", "--all").split():
        msg = git("log", "-1", "--format=%s%n%b", sha)
        for rx, why in STRONG_RX + WORDS_RX:
            m = rx.search(msg)
            if m:
                rep.add("%s message" % sha[:8], msg, m, why)
                break
    listing = git("rev-list", "--objects", "--all")
    names = {}
    for line in listing.split("\n"):
        parts = line.split(" ", 1)
        if len(parts) == 2:
            names[parts[0]] = parts[1]
    batch = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input=listing, capture_output=True, text=True)
    for line in batch.stdout.split("\n"):
        f = line.split()
        if len(f) != 3 or f[1] != "blob" or int(f[2]) > 400000:
            continue
        name = names.get(f[0], "")
        blob = subprocess.run(["git", "cat-file", "blob", f[0]],
                              capture_output=True, text=True, errors="replace").stdout
        for rx, why in rules_for(name):
            m = rx.search(blob)
            if m:
                rep.add("blob %s (%s)" % (f[0][:8], name or "?"), blob, m, why)
                break
    return "history carries tooling traces"


def main(argv):
    if os.environ.get("HYGIENE_SKIP"):
        return 0
    if not argv:
        sys.stderr.write(__doc__)
        return 2
    mode, rest = argv[0], argv[1:]
    rep = Report()
    if mode == "staged":
        what = mode_staged(rep)
    elif mode == "message" and rest:
        what = mode_message(rep, rest[0])
    elif mode == "range" and rest:
        what = mode_range(rep, rest[0], rest[1] if len(rest) > 1 else None)
    elif mode == "tree":
        what = mode_tree(rep)
    elif mode == "history":
        what = mode_history(rep)
    else:
        sys.stderr.write(__doc__)
        return 2
    rep.print(what)
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
