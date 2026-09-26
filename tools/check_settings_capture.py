#!/usr/bin/env python3
"""Assert that every runtime setting survives a Save Settings round trip.

`Renderer::applyRuntimeSettings` copies a loaded `RuntimeSettings` into the
renderer's live state, and `Renderer::captureRuntimeSettings` copies the live
state back out for the ImGui Save Settings button. Both are hand-written lists,
and a field missing from either fails silently:

- missing from apply, a saved value is read, clamped and then ignored;
- missing from capture, Save writes the field's default over whatever the file
  said, because capture starts from a default-constructed `RuntimeSettings`.

The second happened. `framesInFlight` and `enableShaderHotReload` were applied,
written and re-read, but never captured, so every Save reset them to 2 and
false. The first happened too, inside the cascade settings -- which is why
`applyCsmSettings` became a free function with its own test. Neither function
can run in a headless test (both are `Renderer` members), so this reads their
source instead.

The rule, for each top-level field of `RuntimeSettings`:

- a whole-struct or scalar use -- `settings.bloom = ...` in capture,
  `settings.bloom` in apply -- covers the field;
- a field that is only ever used member by member -- `punctualShadows` -- has
  to have every member of its struct type used, in each function separately.

Run from anywhere; paths resolve relative to the repository root.
Exit status is 0 when both functions cover every field, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_HEADER = REPO_ROOT / "src" / "renderer" / "RuntimeSettings.h"
RENDERER_SOURCE = REPO_ROOT / "src" / "renderer" / "Renderer.cpp"

# One member declaration per line at the struct's own indentation: a type, a
# name, an optional initialiser, a semicolon. Anything with a parenthesis is a
# function or an operator and is skipped.
MEMBER = re.compile(r"^    (?!//)(?P<type>[A-Za-z_][\w:<>, ]*?)\s+(?P<name>[A-Za-z_]\w*)\s*(?:=[^;]*|\{[^;]*\})?;")
STRUCT_OPEN = re.compile(r"^struct (?P<name>\w+) \{$")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def parse_structs(header: str) -> dict[str, list[tuple[str, str]]]:
    structs: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for line in header.splitlines():
        if current is None:
            opened = STRUCT_OPEN.match(line)
            if opened:
                current = opened.group("name")
                structs[current] = []
            continue
        if line.startswith("};"):
            current = None
            continue
        member = MEMBER.match(line)
        if member and "(" not in line:
            structs[current].append((member.group("type").strip(), member.group("name")))
    return structs


def function_body(source: str, qualified_name: str) -> str:
    start = re.search(rf"^[^\n;]*\b{re.escape(qualified_name)}\([^)]*\)[^{{;]*\{{", source, flags=re.MULTILINE)
    if start is None:
        raise SystemExit(f"check_settings_capture: {qualified_name} not found in {RENDERER_SOURCE}")
    depth = 0
    for index in range(start.end() - 1, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return strip_comments(source[start.end() : index])
    raise SystemExit(f"check_settings_capture: unbalanced braces in {qualified_name}")


def missing_fields(body: str, structs: dict[str, list[tuple[str, str]]], whole_use: str, member_use: str) -> list[str]:
    missing: list[str] = []
    for field_type, field in structs["RuntimeSettings"]:
        if re.search(whole_use.format(field=re.escape(field)), body):
            continue
        members = structs.get(field_type.split("::")[-1])
        if not members:
            missing.append(field)
            continue
        missing.extend(
            f"{field}.{member}"
            for _, member in members
            if not re.search(member_use.format(field=re.escape(field), member=re.escape(member)), body)
        )
    return missing


def main() -> int:
    header = strip_comments(SETTINGS_HEADER.read_text(encoding="utf-8"))
    structs = parse_structs(header)
    if not structs.get("RuntimeSettings"):
        raise SystemExit(f"check_settings_capture: no RuntimeSettings members parsed from {SETTINGS_HEADER}")

    source = RENDERER_SOURCE.read_text(encoding="utf-8")
    capture = function_body(source, "Renderer::captureRuntimeSettings")
    apply = function_body(source, "Renderer::applyRuntimeSettings")

    # Capture has to WRITE the field; apply has to READ it. A member use in
    # capture is `settings.x.y =`; in apply it is any `settings.x.y`.
    uncaptured = missing_fields(
        capture, structs, r"\bsettings\.{field}\s*=[^=]", r"\bsettings\.{field}\.{member}\s*=[^=]"
    )
    unapplied = missing_fields(apply, structs, r"\bsettings\.{field}\b(?!\.)", r"\bsettings\.{field}\.{member}\b")

    field_count = len(structs["RuntimeSettings"])
    if not uncaptured and not unapplied:
        print(f"Settings capture check passed ({field_count} RuntimeSettings fields, applied and captured).")
        return 0
    for name in uncaptured:
        print(f"captureRuntimeSettings never writes settings.{name}: Save Settings would reset it to its default.")
    for name in unapplied:
        print(f"applyRuntimeSettings never reads settings.{name}: a saved value would be loaded and ignored.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
