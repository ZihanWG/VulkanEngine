# macOS / MoltenVK Build Guide

This project can run on macOS through MoltenVK for portability and basic
renderer validation. The primary/full showcase platform remains RTX/NVIDIA;
some advanced GPU paths may be disabled on macOS depending on MoltenVK and Apple
GPU feature support.

## Prerequisites

- macOS on Apple Silicon or Intel.
- LunarG Vulkan SDK with MoltenVK installed.
- CMake 3.25 or newer.
- Ninja.
- SDL3.

The SDK installs under your home directory. The verified version is:

```sh
$HOME/VulkanSDK/1.4.350.1
```

## Environment

Set up the LunarG SDK before configuring or running:

```sh
source "$HOME/VulkanSDK/1.4.350.1/setup-env.sh"
export SDL_VIDEODRIVER=cocoa
```

After sourcing, `VULKAN_SDK` should point at the SDK's macOS directory:

```sh
echo "$VULKAN_SDK"
# $HOME/VulkanSDK/1.4.350.1/macOS
```

## Build

```sh
cmake --preset debug
cmake --build build/debug
```

The presets in `CMakePresets.json` are the single source of truth for build
directories: `debug` builds into `build/debug`, `release` into `build/release`.
Earlier revisions of this document used a hand-rolled `build-mac`; the launcher
still accepts it, but nothing else does.

## Run

From `build/debug`:

```sh
./VulkanEngine
```

Expected startup logs include SDL using the Cocoa video driver, SDL loading the
Vulkan SDK loader, Vulkan API version 1.3, Apple M3 or the local Apple GPU, and
`VK_KHR_portability_subset` when MoltenVK exposes it.

## Running from Terminal

The direct developer workflow is:

```sh
cd build/debug
source "$HOME/VulkanSDK/1.4.350.1/setup-env.sh"
export SDL_VIDEODRIVER=cocoa
./VulkanEngine
```

This remains the simplest way to run a non-bundle Debug build.

## Running with `run_vulkan_engine.command`

The repository includes a double-clickable Terminal launcher:

```sh
tools/macos/run_vulkan_engine.command
```

From Finder, double-click `tools/macos/run_vulkan_engine.command`. The script
resolves the repository root relative to itself, then looks for `VulkanEngine`
(or `VulkanEngine.app/Contents/MacOS/VulkanEngine`) under `build/debug`,
`build/release`, `build-mac`, and `build`, in that order. It sources a Vulkan
SDK `setup-env.sh` when available, sets `SDL_VIDEODRIVER=cocoa`, and runs the
renderer from whichever directory it found.

SDK lookup order:

1. `$VULKAN_SDK_ROOT/setup-env.sh`
2. `$VULKAN_SDK_ROOT/macOS/setup-env.sh`
3. `~/VulkanSDK/1.4.350.1/setup-env.sh`
4. Latest `~/VulkanSDK/*/setup-env.sh`

If the renderer exits with an error, the script waits for Return before closing
the Terminal window so the error log remains visible.

If macOS blocks the script because it is not executable, run:

```sh
chmod +x tools/macos/run_vulkan_engine.command
```

## Building `VulkanEngine.app`

The default build remains a plain executable. To build an app bundle:

```sh
cmake --preset release -DVULKAN_ENGINE_BUILD_MACOS_BUNDLE=ON
cmake --build build/release
```

The bundle is created at:

```sh
build/release/VulkanEngine.app
```

Run it from Finder by double-clicking `VulkanEngine.app`, or from Terminal:

```sh
open VulkanEngine.app
```

When bundle mode is enabled, CMake copies these folders into
`VulkanEngine.app/Contents/Resources`:

- `assets`
- `shaders`
- `config`

At runtime, macOS builds first look for `assets`, `shaders`, and `config` under
the app bundle `Contents/Resources` directory. Non-bundle runs fall back to the
existing repository/build paths.

## macOS Vulkan Handling

- Rendering uses MoltenVK through the LunarG Vulkan SDK loader, not by loading
  MoltenVK directly.
- SDL3 explicitly loads the Vulkan loader from `$VULKAN_SDK/lib/libvulkan.1.dylib`
  or `$VULKAN_SDK/lib/libvulkan.dylib`.
- Volk initializes from SDL3's `SDL_Vulkan_GetVkGetInstanceProcAddr()` result so
  SDL surface creation and engine Vulkan calls use the same loader path.
- Instance creation enables `VK_KHR_portability_enumeration` and sets
  `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` on Apple platforms.
- Logical device creation always requires `VK_KHR_swapchain` and enables
  `VK_KHR_portability_subset` only when the selected physical device reports it.

## Packaging Limitations

- `VulkanEngine.app` is not signed or notarized.
- The app may still require the LunarG Vulkan SDK, Vulkan loader, and MoltenVK
  to be installed on the machine unless those dylibs are bundled in a later
  packaging step.
- Full redistributable packaging, dependency dylib copying, code signing, and
  notarization are intentionally left for a later release step.

## Troubleshooting

### `VK_ERROR_INCOMPATIBLE_DRIVER` during `vkCreateInstance`

On macOS this usually means portability enumeration was not enabled. Confirm the
enabled instance extensions include `VK_KHR_portability_enumeration` and that the
instance create flags include `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`
on Apple builds.

### Segmentation fault inside `Cocoa_Vulkan_CreateSurface`

This indicates SDL and the engine may be using different Vulkan loader/proc
address paths. Confirm startup logs show SDL loading:

```sh
$VULKAN_SDK/lib/libvulkan.1.dylib
```

and Volk initializing from SDL's `vkGetInstanceProcAddr`.

### `SDL_CreateWindow failed`

Ensure the process has access to the macOS window server and the Cocoa backend is
selected:

```sh
export SDL_VIDEODRIVER=cocoa
```

This can fail in headless shells, SSH sessions, CI, or sandboxed environments
without display access.

### `setup-env.sh` path mistake

Source the SDK root script:

```sh
source "$HOME/VulkanSDK/1.4.350.1/setup-env.sh"
```

Do not hand-write `VULKAN_SDK` unless necessary. If runtime logs show SDL cannot
load the SDK Vulkan loader, check that `echo "$VULKAN_SDK"` ends with `/macOS`
and that `$VULKAN_SDK/lib/libvulkan.1.dylib` exists.
