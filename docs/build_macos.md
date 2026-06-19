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

The verified local SDK is:

```sh
/Users/zihanw/VulkanSDK/1.4.350.1
```

## Environment

Set up the LunarG SDK before configuring or running:

```sh
source /Users/zihanw/VulkanSDK/1.4.350.1/setup-env.sh
export SDL_VIDEODRIVER=cocoa
```

After sourcing, `VULKAN_SDK` should point at the SDK's macOS directory:

```sh
echo "$VULKAN_SDK"
# /Users/zihanw/VulkanSDK/1.4.350.1/macOS
```

## Build

```sh
mkdir -p build-mac
cd build-mac
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Debug
ninja
```

## Run

From `build-mac`:

```sh
./VulkanEngine
```

Expected startup logs include SDL using the Cocoa video driver, SDL loading the
Vulkan SDK loader, Vulkan API version 1.3, Apple M3 or the local Apple GPU, and
`VK_KHR_portability_subset` when MoltenVK exposes it.

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
source /Users/zihanw/VulkanSDK/1.4.350.1/setup-env.sh
```

Do not hand-write `VULKAN_SDK` unless necessary. If runtime logs show SDL cannot
load the SDK Vulkan loader, check that `echo "$VULKAN_SDK"` ends with `/macOS`
and that `$VULKAN_SDK/lib/libvulkan.1.dylib` exists.
