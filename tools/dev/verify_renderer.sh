#!/usr/bin/env bash
set -euo pipefail

readonly VERIFY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly VERIFY_MODE="${1:-fast}"
readonly VERIFY_PLATFORM="$(uname -s)"

usage() {
    printf '%s\n' \
        "Usage: tools/dev/verify_renderer.sh [shaders|tests|fast|sync|full]" \
        "" \
        "  shaders  Configure ci-debug if needed and compile GLSL." \
        "  tests    Configure ci-debug if needed, build tests, and run CTest." \
        "  fast     Build shaders, renderer, and tests, then run CTest." \
        "  sync     Run the renderer under synchronization validation. Needs a GPU." \
        "  full     Run fast, sync, ASan/UBSan tests (not on MSVC), and a Release renderer build." \
        "" \
        "On Windows, run it from Git Bash. Without cl.exe on PATH it imports the" \
        "Visual Studio x64 build environment itself."
}

is_windows() {
    case "$VERIFY_PLATFORM" in
        MINGW* | MSYS* | CYGWIN*) return 0 ;;
        *) return 1 ;;
    esac
}

# Git Bash does not start inside a Visual Studio developer environment, so on
# Windows cl.exe, its INCLUDE and LIB paths, and the CMake and Ninja that ship
# with Visual Studio are all missing unless the shell was opened from a
# developer prompt. Import them from vcvars64.bat rather than require that, so
# the same command works from any terminal on either machine. A shell that
# already has cl.exe is left exactly as it is.
import_msvc_environment() {
    is_windows || return 0
    if command -v cl.exe >/dev/null 2>&1; then
        return 0
    fi

    local vswhere
    vswhere="$(cygpath -u "${SYSTEMDRIVE:-C:}\\")Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    if [[ ! -x "$vswhere" ]]; then
        printf '%s\n' '[verify] no cl.exe on PATH and no vswhere.exe to find one; run from a developer prompt' >&2
        return 1
    fi

    local install
    install="$("$vswhere" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
        -property installationPath | tr -d '\r')"
    if [[ -z "$install" ]]; then
        printf '%s\n' '[verify] vswhere found no Visual Studio with the x64 C++ tools' >&2
        return 1
    fi

    # The paths are %s arguments and never part of the format string: a Visual
    # Studio 18 install path contains "\18" and "\v", which printf would read
    # as escapes, and vcvars would then silently never run. vcvars calls
    # vswhere itself -- it is how the Visual Studio CMake and Ninja reach PATH --
    # and the installer directory is not on PATH by default.
    local script_dir
    script_dir="$(mktemp -d)"
    printf '@set "PATH=%s;%%PATH%%"\r\n@call "%s" >nul\r\n@set\r\n' \
        "$(cygpath -w "$(dirname "$vswhere")")" "$install\\VC\\Auxiliary\\Build\\vcvars64.bat" \
        >"$script_dir/msvc_env.bat"
    # `//c`, not `/c`: MSYS rewrites a lone `/c` into a path, and cmd then starts
    # interactively and waits for input instead of running the script.
    local environment
    environment="$(cmd //c "$(cygpath -w "$script_dir/msvc_env.bat")" | tr -d '\r')"
    rm -rf "$script_dir"

    # Only what vcvars sets. The rest of cmd's `set` output is this shell's own
    # environment in Windows form, and importing it back would overwrite HOME,
    # TEMP and the like with values Git Bash does not use.
    local line name value
    while IFS= read -r line; do
        name="${line%%=*}"
        value="${line#*=}"
        case "$name" in
            Path | PATH)
                PATH="$(cygpath -u -p "$value")"
                export PATH
                ;;
            INCLUDE | LIB | LIBPATH | EXTERNAL_INCLUDE | VCINSTALLDIR | VCToolsInstallDir | VCToolsVersion | \
                VSINSTALLDIR | WindowsSdkDir | WindowsSDKVersion | WindowsSdkBinPath | UniversalCRTSdkDir | \
                UCRTVersion | VSCMD_ARG_HOST_ARCH | VSCMD_ARG_TGT_ARCH | VSCMD_VER | DevEnvDir | Platform)
                export "$name=$value"
                ;;
        esac
    done <<<"$environment"

    if ! command -v cl.exe >/dev/null 2>&1; then
        printf '[verify] vcvars64.bat under %s did not put cl.exe on PATH\n' "$install" >&2
        return 1
    fi
    printf '[verify] MSVC x64 environment imported from %s\n' "$install"
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
#
# What it cannot see: anything read through a buffer device address. The layer
# tracks descriptor-bound accesses, so the cluster grid and light index list --
# written by descriptor on the async queue, read by address on the graphics one
# -- are invisible to it. Fog injection read them unordered for as long as both
# existed and this mode, run with fog on, reported 0 hazards. There is no fog
# run here for that reason: it would pass whether or not the ordering is right.
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

    # MoltenVK exposes no compute-only or transfer-only queue family unless
    # asked to, which would leave a Mac running exactly the inline fallbacks
    # this mode exists to get past. A value already set, 0 included, is kept.
    if [[ "$VERIFY_PLATFORM" == Darwin && -z "${MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES+set}" ]]; then
        export MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1
        printf '%s\n' '[verify] sync: MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1, so MoltenVK exposes its dedicated queues'
    fi

    printf '%s\n' '[verify] renderer under synchronization validation: default scene'
    "$binary" --deterministic --exit-after-frames 40 --sync-validation --fail-on-validation-error

    # The asset is fetched once into a directory every preset shares, but only a
    # build configured with the flag compiles the scene in; any other exits 4 on
    # --scene sponza. Ask the chosen binary's own cache rather than fall back to
    # a different build, which may be stale and would gate the wrong code.
    local cache
    cache="$(dirname "$binary")/CMakeCache.txt"
    if [[ ! -f "$VERIFY_ROOT/build/fetched-assets/sponza/Sponza.gltf" ]]; then
        printf '%s\n' '[verify] sync: skipping --scene sponza, asset not fetched'
    elif ! grep -q '^VULKAN_ENGINE_FETCH_SAMPLE_SCENE:BOOL=ON$' "$cache" 2>/dev/null; then
        printf '[verify] sync: skipping --scene sponza, %s was configured without VULKAN_ENGINE_FETCH_SAMPLE_SCENE\n' \
            "${cache#"$VERIFY_ROOT/"}"
    else
        printf '%s\n' '[verify] renderer under synchronization validation: --scene sponza'
        "$binary" --deterministic --exit-after-frames 40 --sync-validation --fail-on-validation-error \
            --scene sponza
    fi
}

run_full() {
    run_fast
    run_sync

    # MSVC ignores VULKAN_ENGINE_ENABLE_SANITIZERS with a configure warning, so
    # on Windows this leg would build the plain tests, run them a second time,
    # and report an ASan/UBSan pass that checked nothing. The Mac and Linux CI
    # run it for real.
    if is_windows; then
        printf '%s\n' '[verify] skip ASan/UBSan: MSVC builds without the sanitizers; run it on the Mac or read Linux CI'
    else
        configure_if_needed ci-asan
        printf '%s\n' '[verify] build ASan/UBSan headless tests'
        cmake --build --preset ci-asan --parallel
        printf '%s\n' '[verify] run ASan/UBSan headless tests'
        ctest --preset ci-asan
    fi

    configure_if_needed release
    printf '%s\n' '[verify] build Release renderer'
    cmake --build --preset release --parallel
}

cd "$VERIFY_ROOT"

case "$VERIFY_MODE" in
    shaders | tests | fast | full)
        import_msvc_environment
        ;;
esac

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
