#!/bin/zsh

set -u

SCRIPT_PATH="${0:A}"
SCRIPT_DIR="${SCRIPT_PATH:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
BUILD_DIR="${REPO_ROOT}/build-mac"
PLAIN_EXECUTABLE="${BUILD_DIR}/VulkanEngine"
BUNDLE_EXECUTABLE="${BUILD_DIR}/VulkanEngine.app/Contents/MacOS/VulkanEngine"

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

if [[ -x "${PLAIN_EXECUTABLE}" ]]; then
    EXECUTABLE="${PLAIN_EXECUTABLE}"
elif [[ -x "${BUNDLE_EXECUTABLE}" ]]; then
    EXECUTABLE="${BUNDLE_EXECUTABLE}"
else
    echo "Could not find a built VulkanEngine executable."
    echo "Expected one of:"
    echo "  ${PLAIN_EXECUTABLE}"
    echo "  ${BUNDLE_EXECUTABLE}"
    echo
    echo "Build first, for example:"
    echo "  mkdir -p build-mac"
    echo "  cd build-mac"
    echo "  cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug"
    echo "  ninja"
    wait_on_error 127
fi

cd "${BUILD_DIR}" || wait_on_error $?
"${EXECUTABLE}"
wait_on_error $?
