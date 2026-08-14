#!/usr/bin/env bash
set -euo pipefail

readonly VERIFY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly VERIFY_MODE="${1:-fast}"

usage() {
    printf '%s\n' \
        "Usage: tools/dev/verify_renderer.sh [shaders|tests|fast|full]" \
        "" \
        "  shaders  Configure ci-debug if needed and compile GLSL." \
        "  tests    Configure ci-debug if needed, build tests, and run CTest." \
        "  fast     Build shaders, renderer, and tests, then run CTest." \
        "  full     Run fast, ASan/UBSan tests, and a Release renderer build."
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

run_full() {
    run_fast

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
