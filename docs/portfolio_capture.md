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
- an opaque ceramic hero material with smaller material samples arranged around it
- procedural UV-sphere material samples for matte gray, glossy blue dielectric, rough metal, and small polished metal looks

The portfolio-only scene objects are skipped by draw-item construction while the mode is disabled, so normal rendering remains unchanged. The checkerboard cube/glTF test objects stay in the regular debug scene and are hidden only while Portfolio Capture Mode is active. A screenshot request automatically enables Portfolio Capture Mode and verifies that portfolio showcase draw items are active before readback, so `F12` cannot accidentally export the debug fallback scene.

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

**The portfolio path always copies here, before the overlay** — a portfolio shot
with a debug panel across it is not a portfolio shot. Only the regression capture
(`--capture-frame`) can be asked for the other side of it, with
`--capture-include-ui`; see [headless_ci.md](headless_ci.md).

## Animated Captures

An animated clip comes from a scripted run rather than from F12:
`--capture-sequence FIRST COUNT --capture-output-dir DIR` reads back `COUNT`
consecutive frames starting at frame `FIRST` and writes them to `DIR` as
`frame_0000.png` onwards, and [`tools/make_gif.py`](../tools/make_gif.py) encodes
the directory as a looping GIF.

```
./build/release/VulkanEngine --scene sponza --deterministic --window-size 960x540 \
    --camera-orbit -0.0004 --capture-sequence 30 150 --capture-output-dir /tmp/frames
python tools/make_gif.py /tmp/frames screenshots/demo.gif --fps 30 --width 800 --no-dither
```

That is how [`screenshots/sponza_atrium.gif`](../screenshots/sponza_atrium.gif)
was made, except that the README clip keeps every fourth frame and plays them
forward and then back at 15 fps, so it loops without a jump, and that it ran
with `--settings` pointing at a copy of `config/runtime_settings.example.json`
changed in four places:

| Key | Default | Clip | Why |
| --- | --- | --- | --- |
| `taa.enabled` | `false` | `true` | Without it the curtain weave moirés and edges stair-step, which reads as over-sharpened. The composite sharpen only runs when upscaling, so it is not the cause at scale 1.0. |
| `toneMapping.toneMapper` | `Reinhard` | `ACES` | Reinhard leaves the interior dim: mean luma 36.8/255 by default, 51.4 with a 0.30 exposure target, 89.0 with ACES at that target. |
| `toneMapping.enableAutoExposure` | `true` | `false` | Auto exposure is still adapting at frame 30 and then follows the content, so the clip brightened from 67 to 93 mean luma end to end, which pumps in a loop. |
| `toneMapping.manualExposure` | `1.0` | `2.2` | About where auto exposure settles during this shot: 93.7 and 88.3 mean luma at the two ends. |

These are the clip's settings, not new defaults: the defaults feed the golden
pixel gate and every measured number.

- `--camera-orbit` pivots around the camera's target, and the Sponza preset puts
  that target half the building ahead, so the camera sweeps a wide arc: 0.01
  rad/frame leaves the building within a few seconds, and 0.0012 crosses the
  curtains into the side aisle by frame 180. -0.0004 stays in the atrium and
  drifts towards the hanging brazier.
- A moving camera changes nearly every pixel every frame, so GIF size is roughly
  frames x pixels. On this 148-frame, 800-wide clip `--no-dither` saved only
  10% (39.9 -> 35.8 MiB) with no visible banding; keeping every second frame of
  that at 15 fps halved it (17.9 MiB, before the settings below; 17.1 MiB with
  them). Frame count and width are the levers.
- The scene is the one the run was launched with, not the portfolio showcase:
  `--scene` picks it, `--camera-orbit` turns the preset's camera by a fixed angle
  per frame, and `--freeze-animation` holds the lights and animated objects still
  if only the camera should move.
- `--deterministic` is required. Each captured frame stalls the loop for a
  readback and an uncompressed PNG write, and on the wall clock every animation
  would jump by however long that took; on the fixed-step clock each frame is
  one 1/60 s step of scene time, so the clip plays smoothly at 60 fps of scene
  time whatever the capture cost.
- Start a few dozen frames in, so TAA history, auto exposure and streaming have
  settled before the first captured frame.
- The PNG writer does not compress: one 1920x1080 frame is about 8 MB, so a
  six-second clip at 1080p is well over a gigabyte of intermediate files.
  Capturing at the GIF's own size (960x540 above) keeps that to a quarter.
- `make_gif.py` maps every frame onto one shared palette built from frames
  across the whole clip, so flat regions do not shimmer between frames, and
  `--width` scales on encode if the capture was taken larger.

The copy is the same one `--capture-frame` records, before the ImGui overlay
unless `--capture-include-ui` is given. Every in-flight frame slot owns its own
readback buffer, and a slot's PNG is written after the timeline wait that
retires it and before the slot records again, so consecutive captures add no
synchronization. The run fails with exit code 3 unless every frame of the range
was written.

## Reflection Scope

Reflections are environment-based specular IBL plus screen-space reflections (see [ssr.md](ssr.md)); SSR is on by default, so polished materials pick up scene detail the cubemap cannot supply. There are no planar reflections, ray-traced reflections, local reflection probes, or real glass transmission/refraction, and SSR only finds what is on screen. Smooth metallic objects sample the prefiltered environment cubemap through the split-sum BRDF path, so polished materials are kept small and secondary in the portfolio PBR material showcase.

## Readback Design

The swapchain is created with `VK_IMAGE_USAGE_TRANSFER_SRC_BIT` when the surface supports it. On a capture request, the renderer records these commands into the normal frame command buffer:

1. Transition swapchain color from color attachment to transfer source.
2. Copy the image to a host-visible per-frame readback buffer.
3. Barrier the buffer for host read after queue completion.
4. Transition the swapchain image back to color attachment for the ImGui pass.

The CPU reads and writes the PNG later, after the frame loop's existing timeline wait for that frame slot has completed. The normal frame loop does not add a permanent `vkDeviceWaitIdle`.

## Limitations

- Capture resolution is the current swapchain extent.
- PNG writing uses a small internal RGBA8 writer with uncompressed deflate blocks, so files are larger than optimized PNGs.
- Unsupported swapchain transfer-source usage disables screenshot capture with a UI/status warning.
- Supported swapchain capture formats are `R8G8B8A8` and `B8G8R8A8`, UNORM or sRGB.
