# Post-Processing

Phase 6A keeps the renderer on a stable, non-temporal post-processing path. It adds mip-chain bloom and moves automatic exposure use into a GPU-readable exposure state buffer without adding TAA, motion vectors, temporal upscaling, or ray tracing.

## HDR Scene Color

`MainHDRPass` writes linear HDR skybox and mesh lighting to `SceneColorHDR`, a full-resolution `VK_FORMAT_R16G16B16A16_SFLOAT` image. The image is recreated with the swapchain and is sampled by bloom, luminance, histogram exposure, composite, and render-target debug previews.

## Bloom Pipeline

The legacy fallback path still exists:

- `BloomExtractPass` samples `SceneColorHDR` and writes thresholded highlights to `BloomExtract`.
- `BloomBlurHorizontal` samples `BloomExtract` and writes `BloomPing`.
- `BloomBlurVertical` samples `BloomPing` and writes `BloomPong`.

The default path is mip-chain bloom:

- `BloomDownsampleMip0` samples HDR scene color, applies the bloom threshold per sample, and writes the 1/2-resolution bloom level.
- Lower downsample passes write 1/4, 1/8, and 1/16-resolution levels when the viewport is large enough.
- Upsample passes run from the smallest useful level back toward 1/2 resolution.
- Each upsample pass combines the local current mip with the accumulated lower-resolution bloom.
- `CompositePass` can choose the mip-chain result or the legacy blur result at runtime.

Bloom resources are persistent renderer-owned images and are recreated on swapchain resize. They are declared as transient Render Graph resources for metadata, liveness, and conservative image transitions.

## Exposure Path

Automatic exposure uses the existing scene luminance inputs:

- `LuminancePass` writes per-workgroup log-average luminance partials to a GPU storage buffer.
- `HistogramExposurePass` clears and fills a 256-bin log2 luminance histogram.
- `exposure_reduce.comp` reads luminance partials, histogram bins, and the previous exposure state, then writes current exposure, log-average luminance, histogram-clipped luminance, and mode to `ExposureStateBufferN`.

`CompositePass` reads `ExposureStateBufferN` directly for auto exposure modes. The CPU reads that small exposure state only after the frame fence for ImGui/debug history. Manual exposure remains available and portfolio mode forces stable manual exposure.

## Composite Pass

`composite.frag` samples:

- HDR scene color
- legacy bloom
- mip-chain bloom
- GPU exposure state

The shader selects the active bloom method, multiplies bloom by strength, applies manual or GPU exposure, and then runs Reinhard or ACES tone mapping before writing the swapchain image. F12 portfolio screenshots still copy the swapchain after `CompositePass` and before `ImGuiPass`.

## Render Graph And Profiler

Render Graph metadata now includes:

- Legacy bloom extract and blur passes
- Bloom downsample mip passes
- Bloom upsample mip passes
- Luminance and histogram exposure buffers
- GPU exposure state buffer
- Composite reads of scene color, bloom, and exposure state

GPU profiler scopes include `Bloom Downsample Chain`, `Bloom Upsample Chain`, `Histogram Exposure`, and `CompositePass`. Buffer barriers for histogram reset, exposure reduce, and debug readback remain manual in `Renderer.cpp`.

## Debug UI

The ImGui debug UI exposes:

- Bloom enabled/disabled
- Bloom method
- Bloom mip count
- Bloom strength
- Bloom threshold
- Bloom radius
- Exposure mode
- Current debug exposure value
- Tone mapper selection
- Render target previews for scene color, legacy bloom, and selected bloom mip-chain levels

## Known Limitations

- No TAA.
- No motion vectors or velocity buffer.
- No temporal upscaling.
- No FSR/DLSS.
- No ray tracing.
- No local exposure.
- No automatic camera-cut handling.
