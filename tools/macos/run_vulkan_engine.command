#!/bin/zsh

set -u

SCRIPT_PATH="${0:A}"
SCRIPT_DIR="${SCRIPT_PATH:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
# Search order matches the CMake presets first, then the older build-mac layout
# the macOS docs used to describe. Pinning a single directory is what made this
# script fail for anyone who followed docs/build.md.
BUILD_DIR_CANDIDATES=(
    "${REPO_ROOT}/build/debug"
    "${REPO_ROOT}/build/release"
    "${REPO_ROOT}/build-mac"
    "${REPO_ROOT}/build"
)

function wait_on_error() {
    local status="$1"
    if (( status != 0 )); then
        echo
        echo "VulkanEngine exited with status ${status}."
        echo "Press Return to close this Terminal window."
        read -r
    fi
    exit "${status}"
}

function find_setup_env() {
    if [[ -n "${VULKAN_SDK_ROOT:-}" ]]; then
        if [[ -f "${VULKAN_SDK_ROOT}/setup-env.sh" ]]; then
            echo "${VULKAN_SDK_ROOT}/setup-env.sh"
            return 0
        fi
        if [[ -f "${VULKAN_SDK_ROOT}/macOS/setup-env.sh" ]]; then
            echo "${VULKAN_SDK_ROOT}/macOS/setup-env.sh"
            return 0
        fi
    fi

    if [[ -f "${HOME}/VulkanSDK/1.4.350.1/setup-env.sh" ]]; then
        echo "${HOME}/VulkanSDK/1.4.350.1/setup-env.sh"
        return 0
    fi

    local candidates=("${HOME}"/VulkanSDK/*/setup-env.sh(N))
    if (( ${#candidates[@]} > 0 )); then
        candidates=("${(@on)candidates}")
        echo "${candidates[-1]}"
        return 0
    fi

    return 1
}

if SETUP_ENV="$(find_setup_env)"; then
    echo "Using Vulkan SDK setup: ${SETUP_ENV}"
    source "${SETUP_ENV}"
else
    echo "Warning: Vulkan SDK setup-env.sh was not found."
    echo "Set VULKAN_SDK_ROOT or install the LunarG Vulkan SDK under ~/VulkanSDK."
fi

export SDL_VIDEODRIVER=cocoa

EXECUTABLE=""
BUILD_DIR=""
for candidate in "${BUILD_DIR_CANDIDATES[@]}"; do
    if [[ -x "${candidate}/VulkanEngine" ]]; then
        BUILD_DIR="${candidate}"
        EXECUTABLE="${candidate}/VulkanEngine"
        break
    fi
    if [[ -x "${candidate}/VulkanEngine.app/Contents/MacOS/VulkanEngine" ]]; then
        BUILD_DIR="${candidate}"
        EXECUTABLE="${candidate}/VulkanEngine.app/Contents/MacOS/VulkanEngine"
        break
    fi
done

if [[ -z "${EXECUTABLE}" ]]; then
    echo "Could not find a built VulkanEngine executable."
    echo "Looked under:"
    for candidate in "${BUILD_DIR_CANDIDATES[@]}"; do
        echo "  ${candidate}"
    done
    echo
    echo "Build first:"
    echo "  cmake --preset debug"
    echo "  cmake --build build/debug"
    wait_on_error 127
fi

echo "Running ${EXECUTABLE}"
cd "${BUILD_DIR}" || wait_on_error $?
"${EXECUTABLE}"
wait_on_error $?
