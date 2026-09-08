# Contributing

This is a single-maintainer project, so nothing here is process for its own
sake. These are the conventions the existing history already follows, written
down so they survive a gap between sessions and so a tool can check the ones
worth checking.

## Branching

`main` is the only long-lived branch. Everything else is short-lived: branch
off `main`, do one thing, merge back, delete.

Branch names are `type/short-description` in kebab-case, where `type` comes
from the same vocabulary the commit conventions use:

| Type | For |
| --- | --- |
| `feat` | A new capability |
| `fix` | A defect in existing behaviour |
| `perf` | A change whose point is measured cost |
| `refactor` | Restructuring that neither fixes nor adds |
| `docs` | Documentation only |
| `test` | Tests only |
| `chore` | Build, tooling, dependencies |
| `style` | Formatting only, no semantic change |

Names already in the history: `feat/png-writer`,
`fix/gpu-frame-time-narrowing`, `perf/gpu-clock-pinning`,
`refactor/pipeline-key-store`, `docs/pipeline-store-numbers`,
`chore/strip-utf8-boms`.

### Why not Git Flow

The usual enterprise model — long-lived `develop`, `test` and `release`
branches mapped onto DEV/FAT/UAT/PRO environments — is deliberately not used.
It solves problems this project does not have: several teams landing work at
once, a separate QA group that needs a stable branch pointed at a test
deployment, and a release train with staged sign-off. There are no deployment
environments here and no second contributor, so `develop` would be a shadow of
`main` that costs one extra merge per change, and `test` and `release` would be
dead branches.

If the project ever ships versioned binaries, the increment to add is a
`release/x.y` branch cut from `main` for backporting fixes — not the rest of
the model.

## Commits

One logical change per commit. "Logical" is about the change, not the file
count: an implementation and the tests that cover it are one commit, while
three unrelated fixes are three commits even though they are small.

Write the subject in the imperative mood, describing the change itself:

```
Rewrite the PNG writer and cover it with round-trip tests
Feed the fresh GPU frame time through the same guard as the history
Hold compute pipelines in the store, chosen by lifetime not by type
```

Subjects carry **no `feat:` / `fix:` type prefix**. The branch name already
carries the type, and nothing in this repository consumes Conventional Commits
— there is no generated changelog and no semantic-version release. Adding the
prefix would spend part of a short subject line on machine-readability that
nothing reads. Revisit this if changelog generation ever lands.

Use the body for why, not what — the diff already says what. Worth including:
the reasoning a future reader would otherwise have to reconstruct, the numbers
behind a `perf` change, and what you actually verified.

### Fixing a commit you have not pushed

```sh
git commit --amend            # reword or extend the last commit
git reset --soft HEAD~1       # undo the last commit, keep the changes staged
git reset --hard HEAD~1       # undo the last commit AND discard its changes
```

`git reset --hard HEAD` is **not** on that list, and it is worth naming because
it circulates as advice for redoing a commit. It does not undo anything: `HEAD`
is the commit you just made, so the commit stays exactly where it is, and the
`--hard` throws away every uncommitted change in the working tree. It is the
one command in this area that can lose work you cannot get back.

Only rewrite history that has not been pushed, or that lives on a branch nobody
else has.

## Formatting

`.clang-format` is enforced. The `clang-format` job in
`.github/workflows/build.yml` is blocking, so a formatting slip fails CI rather
than accumulating — which is how the tree previously drifted to 157
non-conforming files with the config sitting right there.

Run it before pushing:

```sh
find src tests tools \
    \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) \
    -print0 | xargs -0 clang-format-18 -i
```

Use clang-format 18 specifically. Its output changes between major versions, so
a different major will fight the CI job even on code you did not touch. CI
pins `clang-format-18`, the major the `ubuntu-24.04` runner image ships.

Two things the config does on purpose:

- `SortIncludes: false`. Include order in this tree carries implicit
  dependencies, so the formatter leaves it alone. Order includes yourself.
- `SortUsingDeclarations` is left at the LLVM default, so `using` declarations
  do get sorted. This is safe and is why the initial reformat touched
  `tests/` beyond whitespace.

GLSL under `src/shaders` is not formatted by any tool — clang-format does not
understand it. Match the surrounding style by hand.

### Blame across the reformat

`fa2f4a5` reformatted the whole tree and is listed in `.git-blame-ignore-revs`.
GitHub's web blame reads that file automatically. Locally it is one-time setup:

```sh
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

Without it, `git blame` credits the reformat for most lines in the repository.

## Static analysis

`.clang-tidy` runs in CI but is **report-only** — `WarningsAsErrors` is empty
and the step sets `continue-on-error`. A green build therefore does not mean
clang-tidy is clean; read the step's log. The check set is deliberately narrow
(`bugprone-*`, `performance-*`, a few `modernize-*`) so that what it does print
is worth reading. Tighten it once the tree is clean under the current set.

## What CI checks

| Check | Workflow | Blocking |
| --- | --- | --- |
| clang-format | `build.yml` | Yes |
| Ubuntu build + unit tests | `build.yml` | Yes |
| Shader constant parity | `build.yml`, `windows-ci.yml` | Yes |
| ASan/UBSan build + tests | `build.yml` | Yes |
| clang-tidy | `build.yml` | No — report only |
| MSVC build + tests | `windows-ci.yml` | Yes |
| lavapipe render, validation + golden image | `headless-render.yml` | Yes |

## Before pushing

- The change is one logical unit, with a subject that says what it does and a
  body that says why.
- `clang-format-18` leaves the tree unchanged.
- The tests build and pass. Use the `ci-debug` preset rather than `debug`: the
  `debug` build preset builds only the `VulkanEngine` target, into a different
  binary directory than `ctest` reads.

  ```sh
  cmake --preset ci-debug
  cmake --build --preset ci-debug --parallel
  ctest --preset ci-debug
  ```

- Anything a reader would have to take on trust — a measurement, a claim about
  behaviour — is either verified in the commit body or not claimed.

See [docs/build.md](docs/build.md) for toolchain setup and the full preset list.
