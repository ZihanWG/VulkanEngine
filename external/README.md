# external

This folder vendors small header-only dependencies that are simpler than a package-manager integration.

- `stb_image.h` is used by Milestone 9 for RGBA image file loading.

CMake uses installed packages first and can fetch missing dependencies with `VULKAN_ENGINE_FETCH_DEPS=ON`:

- SDL3
- Volk
- Vulkan Memory Allocator
- GLM

For production work, pin dependency revisions or add your own package manager integration here.
