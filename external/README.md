# external

This folder vendors small header-only dependencies that are simpler than a package-manager integration.

- `stb_image.h` is used by Milestone 9 for RGBA image file loading.
- `tiny_gltf.h` and `json.hpp` are used by Milestone 24 for static glTF geometry loading.
- `bc7enc/` is Rich Geldreich's [bc7enc_rdo](https://github.com/richgel999/bc7enc_rdo), pinned at
  commit `e6990bc11829c072d9f9e37296f3335072aab4e4`. It provides the BC7 encoder (`bc7enc`) and the
  BC1-BC5 encoders (`rgbcx`) used by the offline texture cook. Only the six files the cook needs are
  vendored; the RDO, ISPC, and PNG parts of the upstream repository are not. MIT / public domain, see
  `bc7enc/LICENSE`. Built as its own CMake target so the project's warning policy does not apply to it.

CMake uses installed packages first and can fetch missing dependencies with `VULKAN_ENGINE_FETCH_DEPS=ON`:

- SDL3
- Volk
- Vulkan Memory Allocator
- GLM

For production work, pin dependency revisions or add your own package manager integration here.
