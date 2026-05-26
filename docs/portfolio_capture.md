# Portfolio Screenshot Capture

This workflow exports a clean PNG suitable for a portfolio page without adding an offline renderer or changing the normal frame path.

## Controls

- `F11`: toggles Portfolio Capture Mode.
- `F12`: enables the portfolio showcase if needed, then captures the next rendered frame.
- ImGui: `VulkanEngine Debug` -> `Portfolio Capture` -> `Load Portfolio Showcase Scene`, `Portfolio Capture Mode`, and `Capture Portfolio Screenshot`.

Screenshots are written to:

- `screenshots/vulkan_engine_portfolio_latest.png`
- `screenshots/vulkan_engine_portfolio_YYYYMMDD_HHMMSS.png`

The directory is created automatically.

## What Portfolio Capture Mode Changes

While enabled, the renderer applies a stable showcase setup:

- three-quarter camera angle
- 44 degree vertical FOV
- ACES tone mapping
- manual exposure
- subtle bloom enabled
- CSM texel snapping enabled
- cascade debug colors disabled
- a portfolio-only neutral studio floor and subtle gradient backdrop
- a smaller opaque rough ceramic hero sphere with smaller material samples arranged around it
- procedural UV-sphere material samples for matte gray, glossy blue dielectric, rough metal, and medium metal looks

The portfolio-only scene objects are skipped by draw-item construction while the mode is disabled, so normal rendering remains unchanged. The checkerboard cube/glTF test objects stay in the regular debug scene and are hidden only while Portfolio Capture Mode is active. A screenshot request automatically enables Portfolio Capture Mode before queuing the readback so `F12` cannot accidentally export the debug fallback pyramid.

## Capture Point

The screenshot is copied from the acquired swapchain image after `CompositePass` and before `ImGuiPass`.

That means the PNG includes:

- HDR exposure and tone mapping
- bloom composite
- PBR lighting
- IBL/skybox contribution
- CSM shadowing when visible
- the portfolio showcase material set when Portfolio Capture Mode is active

The PNG excludes:

- ImGui panels
- debug overlays

## Reflection Scope

Current reflective materials use environment-based specular IBL. Smooth metallic objects sample the prefiltered environment cubemap through the split-sum BRDF path; they do not reflect nearby scene objects because the renderer does not currently implement screen-space reflections, planar reflections, ray traced reflections, or local reflection probes. The portfolio showcase uses opaque PBR materials only; it does not implement true glass transmission or screen-space refraction.

## Readback Design

The swapchain is created with `VK_IMAGE_USAGE_TRANSFER_SRC_BIT` when the surface supports it. On a capture request, the renderer records these commands into the normal frame command buffer:

1. Transition swapchain color from color attachment to transfer source.
2. Copy the image to a host-visible per-frame readback buffer.
3. Barrier the buffer for host read after queue completion.
4. Transition the swapchain image back to color attachment for the ImGui pass.

The CPU reads and writes the PNG later, after the existing frame fence for that frame slot has completed. The normal frame loop does not add a permanent `vkDeviceWaitIdle`.

## Limitations

- Capture resolution is the current swapchain extent.
- PNG writing uses a small internal RGBA8 writer with uncompressed deflate blocks, so files are larger than optimized PNGs.
- Unsupported swapchain transfer-source usage disables screenshot capture with a UI/status warning.
- Supported swapchain capture formats are `R8G8B8A8` and `B8G8R8A8`, UNORM or sRGB.
