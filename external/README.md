# external

This project does not vendor third-party source by default.

CMake uses installed packages first and can fetch missing dependencies with `VULKAN_ENGINE_FETCH_DEPS=ON`:

- SDL3
- Volk
- Vulkan Memory Allocator
- GLM

For production work, pin dependency revisions or add your own package manager integration here.
