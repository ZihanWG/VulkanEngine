#!/usr/bin/env bash
set -euo pipefail

readonly VERIFY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly VERIFY_MODE="${1:-fast}"

usage() {
    printf '%s\n' \
        "Usage: tools/dev/verify_renderer.sh [shaders|tests|fast|sync|full]" \
        "" \
        "  shaders  Configure ci-debug if needed and compile GLSL." \
        "  tests    Configure ci-debug if needed, build tests, and run CTest." \
        "  fast     Build shaders, renderer, and tests, then run CTest." \
        "  sync     Run the renderer under synchronization validation. Needs a GPU." \
        "  full     Run fast, sync, ASan/UBSan tests, and a Release renderer build."
}

configure_if_needed() {
    local preset="$1"
    local build_dir="$VERIFY_ROOT/build/$preset"

    if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
        printf '[verify] configure preset: %s\n' "$preset"
        cmake --preset "$preset"
    fi
}

run_shaders() {
    configure_if_needed ci-debug
    printf '%s\n' '[verify] build shaders'
    cmake --build --preset ci-debug-shaders --parallel
}

run_tests() {
    configure_if_needed ci-debug
    printf '%s\n' '[verify] build headless tests'
    cmake --build "$VERIFY_ROOT/build/ci-debug" --target VulkanEngineTests --parallel
    printf '%s\n' '[verify] run headless tests'
    ctest --preset ci-debug
}

run_fast() {
    configure_if_needed ci-debug
    printf '%s\n' '[verify] build renderer, shaders, and headless tests'
    cmake --build --preset ci-debug --parallel
    printf '%s\n' '[verify] run headless tests'
    ctest --preset ci-debug
}

# The one gate this machine has that CI does not.
#
# CI runs synchronization validation over the whole configuration sweep, but on
# lavapipe, which exposes no TRANSFER-only and no compute-only queue family. The
# queue family ownership transfers in the upload path and everything the async
# compute queue does therefore never execute there -- and an unordered ownership
# transfer in exactly that path survived CI for as long as it existed, because no
# machine that could see it ever ran it.
#
# --scene sponza is what drives the batched upload path; the default scene loads
# too few textures to reach it. It is skipped when the asset has not been fetched
# rather than failing, because it is optional by design.
run_sync() {
    # Either Debug build will do. What this mode needs is validation layers
    # and the machine's real queue families, not a particular preset, and a
    # developer box often has build/debug from an IDE rather than the ci-debug
    # preset CI uses.
    local binary=""
    local candidate
    for candidate in \
        "$VERIFY_ROOT/build/ci-debug/VulkanEngine" \
        "$VERIFY_ROOT/build/ci-debug/VulkanEngine.exe" \
        "$VERIFY_ROOT/build/debug/VulkanEngine" \
        "$VERIFY_ROOT/build/debug/VulkanEngine.exe"; do
        if [[ -x "$candidate" ]]; then
            binary="$candidate"
            break
        fi
    done
    if [[ -z "$binary" ]]; then
        printf '%s\n' '[verify] sync: no Debug renderer binary; run "fast" first' >&2
        return 1
    fi
    printf '[verify] sync: using %s\n' "${binary#"$VERIFY_ROOT/"}"

    printf '%s\n' '[verify] renderer under synchronization validation: default scene'
    "$binary" --deterministic --exit-after-frames 40 --sync-validation --fail-on-validation-error

    if [[ -f "$VERIFY_ROOT/build/fetched-assets/sponza/Sponza.gltf" ]]; then
        printf '%s\n' '[verify] renderer under synchronization validation: --scene sponza'
        "$binary" --deterministic --exit-after-frames 40 --sync-validation --fail-on-validation-error \
            --scene sponza
    else
        printf '%s\n' '[verify] sync: skipping --scene sponza, asset not fetched'
    fi
}

run_full() {
    run_fast
    run_sync

    configure_if_needed ci-asan
    printf '%s\n' '[verify] build ASan/UBSan headless tests'
    cmake --build --preset ci-asan --parallel
    printf '%s\n' '[verify] run ASan/UBSan headless tests'
    ctest --preset ci-asan

    configure_if_needed release
    printf '%s\n' '[verify] build Release renderer'
    cmake --build --preset release --parallel
}

cd "$VERIFY_ROOT"

case "$VERIFY_MODE" in
    shaders)
        run_shaders
        ;;
    tests)
        run_tests
        ;;
    fast)
        run_fast
        ;;
    sync)
        run_sync
        ;;
    full)
        run_full
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        printf 'Unknown mode: %s\n\n' "$VERIFY_MODE" >&2
        usage >&2
        exit 2
        ;;
esac
