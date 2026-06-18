# Build Guide

This project is a C++20 Vulkan renderer. Building it should not require running
the renderer, so CI can validate compilation in a headless environment without a
GPU or display.

## Required tools

- CMake 3.25 or newer.
- A C++20 compiler:
  - Windows: Visual Studio 2022 MSVC x64.
  - Linux: GCC or Clang with C++20 support.
  - macOS: AppleClang with C++20 support, plus a Vulkan SDK/MoltenVK setup.
- Ninja for the checked-in CMake presets, or another CMake generator if you
  configure manually.
- Git, used by CMake FetchContent fallback dependencies.
- Vulkan SDK or distro Vulkan development packages that provide:
  - Vulkan headers, including `vulkan/vulkan.h`.
  - `glslc`, the shader compiler used to produce SPIR-V.

The CMake option `VULKAN_ENGINE_FETCH_DEPS=ON` keeps the existing fallback
behavior for SDL3, GLM, Volk, and Vulkan Memory Allocator. Dear ImGui,
`stb_image`, tinygltf, and nlohmann JSON are vendored in `external/`.

## Vulkan SDK requirement

The renderer includes Vulkan headers at build time and compiles GLSL shaders with
`glslc`. The full LunarG Vulkan SDK is the most consistent option on Windows and
macOS. On Linux, CI uses Ubuntu packages for `vulkan-headers`, `libvulkan-dev`,
and `glslc`.

CMake fails early if the Vulkan headers cannot be found. It also fails early if
`glslc` is missing, because the executable depends on compiled SPIR-V shader
files.

## Build locally with CMake presets

The presets use Ninja and enable dependency fallbacks:

```sh
cmake --preset debug
cmake --build --preset debug --parallel
```

For an optimized build:

```sh
cmake --preset release
cmake --build --preset release --parallel
```

The CI-oriented preset is also available locally:

```sh
cmake --preset ci-debug -DVULKAN_ENGINE_FETCH_DEPS=ON
cmake --build --preset ci-debug-shaders --parallel
cmake --build --preset ci-debug --parallel
```

If you prefer Visual Studio or Xcode, configure manually with your generator of
choice and keep `VULKAN_ENGINE_FETCH_DEPS=ON` when you want CMake to download
missing third-party dependencies.

## Shader compilation

GLSL sources live in `src/shaders/`. During configure, CMake locates `glslc`
from `Vulkan_GLSLC_EXECUTABLE` or from `PATH`. The `VulkanEngineShaders` target
compiles each GLSL file to SPIR-V under the build directory, for example
`build/debug/shaders`. The main `VulkanEngine` target depends on
`VulkanEngineShaders`, so building the executable also builds shaders.

The executable embeds the build-directory shader path and the source `assets/`
and `config/` paths. This makes IDE launches less sensitive to the current
working directory.

## Common failure cases

### Missing `glslc`

Symptom: CMake configure fails with a message that `glslc` was not found, or CI
fails in the Vulkan tool verification step.

Fix: install the Vulkan SDK and ensure its `Bin` directory is on `PATH`, or on
Ubuntu install the `glslc` package.

### Missing Vulkan SDK or headers

Symptom: CMake cannot satisfy `find_package(Vulkan REQUIRED)` or CI reports that
`vulkan/vulkan.h` is unavailable.

Fix: install the LunarG Vulkan SDK, or on Ubuntu install `vulkan-headers` and
`libvulkan-dev`. If the SDK is installed in a non-standard location, pass
`Vulkan_ROOT`, `Vulkan_INCLUDE_DIR`, or the relevant CMake Vulkan variables when
configuring.

### Missing Linux window-system development packages

Symptom: SDL3 FetchContent configuration fails on Linux while checking X11,
Wayland, EGL, or input headers.

Fix: install the platform development packages used by CI, including X11,
Wayland, EGL, xkbcommon, and ibus development packages.

### Unsupported GPU, driver, or headless environment

Builds do not prove that the renderer can create a Vulkan device. Running the
renderer requires a GPU and driver that support the Vulkan features used by the
engine. GitHub Actions CI intentionally builds shaders and the executable only;
it does not run runtime GPU tests because hosted runners do not guarantee Vulkan
GPU/display availability.
