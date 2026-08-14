# Post-Processing

Phase 6A added mip-chain bloom and moved automatic exposure use into a GPU-readable exposure state buffer. Phase 6B added an optional Temporal AA foundation; the motion-vector upgrade later extended it with a main-pass velocity buffer and history reprojection, and it now also upsamples -- the resolve reconstructs at presentation resolution from the jittered low-resolution samples (see taa.md). Reactive masks, FSR2/DLSS/XeSS proper, and ray tracing remain out of scope.

## HDR Scene Color

`MainHDRPass` writes linear HDR skybox and mesh lighting to `SceneColorHDR`, a `VK_FORMAT_R16G16B16A16_SFLOAT` image at the internal render extent (the swapchain extent unless render scale is below 1.0 -- see [render_scale.md](render_scale.md)). The image is recreated with the swapchain and is sampled directly by post-processing when TAA is disabled.

When TAA is enabled, `TAAResolvePass` samples `SceneColorHDR` and the previous HDR history image, writes the resolved result to the current history image, and routes bloom, luminance, histogram exposure, composite, and render-target debug previews through that resolved history image.

## Temporal AA Foundation

TAA is disabled by default and is controlled from the `Temporal AA` ImGui panel or the `taa` object in runtime settings.

- Main skybox and mesh rendering use an 8-sample Halton jitter applied to the projection matrix.
- CPU frustum culling, CSM setup, and depth-pyramid validity continue to use the unjittered view-projection matrix.
- Two `VK_FORMAT_R16G16B16A16_SFLOAT` history images at the render extent ping-pong across frames.
- The first valid frame, resize, camera reset/edit, scene load, portfolio mode transition, material save/reload, and explicit UI reset invalidate history.
- The resolve shader clamps previous history to the current frame's 3x3 color neighborhood before feedback blending.
- Bloom, exposure, and composite descriptor variants select either `SceneColorHDR` or resolved TAA history as the active HDR source.

## Bloom Pipeline

The legacy fallback path still exists:

- `BloomExtractPass` samples the active HDR source and writes thresholded highlights to `BloomExtract`.
- `BloomBlurHorizontal` samples `BloomExtract` and writes `BloomPing`.
- `BloomBlurVertical` samples `BloomPing` and writes `BloomPong`.

The default path is mip-chain bloom:

- `BloomDownsampleMip0` samples the active HDR source, applies the bloom threshold per sample, and writes the 1/2-resolution bloom level.
- Lower downsample passes write 1/4, 1/8, and 1/16-resolution levels when the viewport is large enough.
- Upsample passes run from the smallest useful level back toward 1/2 resolution.
- Each upsample pass combines the local current mip with the accumulated lower-resolution bloom.
- `CompositePass` can choose the mip-chain result or the legacy blur result at runtime.

Bloom resources are persistent renderer-owned images and are recreated on swapchain resize. They are declared as transient Render Graph resources for metadata, liveness, and conservative image transitions.

## Exposure Path

Automatic exposure uses the active HDR source:

- `LuminancePass` writes per-workgroup log-average luminance partials to a GPU storage buffer. **It only runs in LogAverage mode.** Histogram mode used to run it too, scanning the whole scene image a second time for a value nothing read: the percentile walk uses the log-average only as a fallback for a completely empty histogram, and the debug readout consumed the rest. The reduce stage now derives the geometric mean from the histogram bins it already walks -- the bins *are* the luminance distribution -- which removes the 0.29 ms pass for 0.03 ms of extra bin sweep. Verified A/B/A/B: frame 15.094/15.037 -> 14.779/14.797 ms, with `average luminance` identical to four decimals (0.3144) and exposure unchanged. At 256 bins the quantisation is ~0.05 stops, far finer than either consumer needs.
- `HistogramExposurePass` clears and fills a 256-bin log2 luminance histogram. `luminance_histogram.comp` tallies each 16x16 workgroup into shared memory and flushes only its non-empty bins, so the pass costs one global atomic per non-empty bin per group rather than one per pixel. That took it from 6.07 ms to 1.96 ms on the demo scene, about 15% of the frame; the shared array's capacity is asserted against `kHistogramBinCount` in `ExposureTypes.h`.
- `exposure_reduce.comp` reads luminance partials, histogram bins, and the previous exposure state, then writes current exposure, log-average luminance, histogram-clipped luminance, and mode to `ExposureStateBufferN`. It is dispatched as a single 256-thread workgroup: there is one luminance partial per 16x16 tile of the scene (14,400 of them at a 2560x1440 drawable), so the threads stride through them and combine the sums through a shared-memory tree instead of one invocation walking the whole array. The percentile walk stays serial on thread 0, being inherently sequential over bins that are staged in shared memory by then. Reducing in parallel took the pass from 2.03 ms to 0.33 ms.

Between the shared-memory histogram and the parallel reduction, `Histogram Exposure` went from 6.07 ms to 0.33 ms on the demo scene. Both changes preserve output: `average luminance` and `histogram clipped luminance` are logged once per second and are the direct assertion on these two shaders.

`CompositePass` reads `ExposureStateBufferN` directly for auto exposure modes. The CPU reads that small exposure state only after the frame fence for ImGui/debug history. Manual exposure remains available and portfolio mode forces stable manual exposure.

## Composite Pass

`composite.frag` samples:

- active HDR scene color, either `SceneColorHDR` or TAA-resolved history
- legacy bloom
- mip-chain bloom
- GPU exposure state

The shader selects the active bloom method, multiplies bloom by strength, applies manual or GPU exposure, and then runs Reinhard or ACES tone mapping before writing the swapchain image. F12 portfolio screenshots still copy the swapchain after `CompositePass` and before `ImGuiPass`.

## Render Graph And Profiler

Render Graph metadata now includes:

- Legacy bloom extract and blur passes
- Bloom downsample mip passes
- Bloom upsample mip passes
- Optional `TAAResolvePass` and imported TAA history resources
- Luminance and histogram exposure buffers
- GPU exposure state buffer
- Composite reads of scene color, bloom, and exposure state

GPU profiler scopes include optional `TAAResolvePass`, `Bloom Downsample Chain`, `Bloom Upsample Chain`, `Histogram Exposure`, and `CompositePass`. Buffer barriers for histogram reset, exposure reduce, and debug readback remain manual in `RendererRecord.cpp`.

## Debug UI

The ImGui debug UI exposes:

- Bloom enabled/disabled
- Bloom method
- Bloom mip count
- Bloom strength
- Bloom threshold
- Bloom radius
- TAA enabled/disabled
- TAA jitter enabled/disabled
- TAA neighborhood clamp enabled/disabled
- TAA history feedback
- TAA history validity and read/write indices
- Exposure mode
- Current debug exposure value
- Tone mapper selection
- Render target previews for scene color, TAA history, legacy bloom, and selected bloom mip-chain levels

## Known Limitations

- TAA reprojects along the velocity buffer (see [taa.md](taa.md)), but there is no depth-based disocclusion classification -- a disoccluded pixel is caught by the neighborhood clamp rather than identified and rejected outright.
- Not FSR2/DLSS/XeSS: the upsampling is reconstruction plus neighbourhood
  rejection, without locks, reactive masks or disocclusion detection.
- No per-material reactive masks.
- No ray tracing.
- No local exposure.
- No automatic camera-cut handling.
