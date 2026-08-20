#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace ve {

struct CsmSettings {
    uint32_t cascadeCount = 4;
    float lambda = 0.5f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    float shadowDistance = 40.0f;
    bool enableTexelSnapping = true;
    bool enableCascadeDebugColors = false;
    float depthBiasConstant = 0.002f;
    float depthBiasSlope = 0.005f;
    // Offsets the shadow lookup along the surface normal instead of only along
    // depth. Expressed as a fraction of the cascade's own world extent, so one
    // value behaves across cascades whose sizes differ by an order of magnitude.
    // The punctual atlas has done this since it shipped; the cascades never did,
    // which is why they leaned harder on raster depth bias -- and depth bias is
    // what buys peter-panning. Zero restores the old depth-bias-only behaviour.
    float normalBias = 0.0f;
    // Width of the cross-fade at each cascade split, as a fraction of that
    // cascade's depth range. Without it the split is a hard line where filter
    // width and texel density change at once. Zero restores the hard split.
    float cascadeBlend = 0.0f;
    // Redraw a cascade only when the hash of what it draws moves. The punctual
    // atlas has cached this way since it shipped; the cascades cleared and
    // redrew unconditionally until now. Off restores that, and is the A/B
    // reference the cached path has to match pixel for pixel.
    bool enableCascadeCache = true;
    // Fits cascades to their slice's bounding sphere instead of its light-space
    // AABB. Off by default: it removes the rotation-driven shimmer the AABB fit
    // has, but a sphere circumscribing the slice is wider than the slice, so the
    // same texels cover more world. MEASURED: it does NOT help the cascade cache
    // -- see CascadeMath.h. Its value is stability, not cache hits.
    bool enableStableCascadeFit = false;

    // Defaulted rather than written out: a hand-written comparison would have to
    // be extended for every new field, which is exactly the failure that made
    // applyCsmSettings necessary in the first place.
    [[nodiscard]] bool operator==(const CsmSettings&) const = default;
};

// Applies persisted cascade settings onto the live ones.
//
// This exists as a free function so it can be tested: it used to be a
// field-by-field copy inside Renderer::applyRuntimeSettings, where nothing
// GPU-free could reach it, and it silently dropped `normalBias` and
// `cascadeBlend` for as long as both had existed -- they were clamped, saved and
// re-read, then thrown away on the way into the renderer. A copy that has to be
// extended per field fails by doing nothing, which is the failure mode that hid
// it.
//
// `cascadeCount` is the one field that cannot follow at runtime: it sizes the
// shadow-map image, and only shadow-map creation can change that. At startup it
// follows like everything else.
void applyCsmSettings(const CsmSettings& incoming, CsmSettings& current, bool startup);

struct ToneMappingSettings {
    float manualExposure = 1.0f;
    bool enableAutoExposure = true;

    int exposureMode = 2;
    // 0 = manual
    // 1 = log-average luminance
    // 2 = histogram percentile

    float targetLuminance = 0.18f;
    float minExposure = 0.1f;
    float maxExposure = 8.0f;
    float adaptationRate = 1.5f;

    float histogramMinLogLuminance = -10.0f;
    float histogramMaxLogLuminance = 4.0f;
    float lowPercentile = 0.05f;
    float highPercentile = 0.95f;

    int operatorType = 0;
    // 0 = Reinhard
    // 1 = ACES fitted approximation
};

// Auto-exposure selection, mirroring ToneMappingSettings::exposureMode. Lives
// with the settings (not in a renderer-internal header) so the GPU-independent
// clamping logic and its unit tests can use it.
enum class ExposureMode : int {
    Manual = 0,
    LogAverage = 1,
    Histogram = 2,
};

// Floor for average-luminance targets to avoid divide-by-zero in exposure math.
inline constexpr float kMinAverageLuminance = 0.0001f;

// Map an arbitrary stored int to a valid ExposureMode (defaults to Histogram).
[[nodiscard]] ExposureMode exposureModeValue(int exposureMode);

struct BloomSettings {
    bool enabled = true;
    bool useMipChain = true;
    float threshold = 1.0f;
    float intensity = 0.1f;
    float radius = 1.0f;
};

struct TaaSettings {
    bool enabled = false;
    bool jitterEnabled = true;
    bool neighborhoodClampEnabled = true;
    // Reprojects the history sample along the velocity buffer (camera + object
    // motion). Off falls back to same-UV history sampling for A/B comparison.
    bool reprojectionEnabled = true;
    float feedback = 0.88f;
    // Two stricter history-rejection mechanisms, BOTH OFF BY DEFAULT.
    //
    // They were built for ghosting that turned out not to exist in this scene,
    // and every form of stricter rejection costs accumulated history -- which is
    // the thing temporal upsampling exists to gather. With nothing wrong to
    // reject, they are cost without benefit, so the default is the behaviour that
    // was actually judged good rather than the one that sounds more advanced.
    //
    // They are kept because ghosting is real and this scene simply has almost no
    // object motion to produce it. Turn them on when content does.

    // Bounds the history by the neighbourhood's mean and standard deviation in
    // YCoCg instead of its RGB extremes. An extremes box is set by its two most
    // extreme samples, so one bright speck widens it enough for a ghost to sit
    // inside.
    bool varianceClipping = false;
    // Half-width of that box in standard deviations. Lower rejects more history.
    float varianceGamma = 1.0f;
    // Lowers a pixel's feedback by how far its history had to be pulled to become
    // acceptable. This is the one that actually removes a ghost -- clipping alone
    // leaves it at the nearest plausible colour with most of the pixel -- and
    // equally the one that costs most when there is no ghost.
    bool rejectionFeedback = false;
    // Resample the history with Catmull-Rom instead of one bilinear tap. A
    // bilinear tap is a low-pass filter, and accumulating through one for N
    // frames applies it N times -- the dominant source of TAA blur, and exactly
    // the detail temporal upsampling is trying to reconstruct. On by default
    // because it only affects how history is *read*; nothing else changes.
    bool catmullRomHistory = true;
};

// Ground-truth ambient occlusion (Jimenez et al. 2016): a dedicated horizon-
// search pass consumes the main depth buffer and the thin G-buffer normal and
// writes a visibility texture the composite pass multiplies into scene color.
// Disabled by default; the toggle is only honoured when the depth image supports
// sampling (Renderer::ssaoAvailable_). Lives with the other post-process
// settings structs so PostProcessStack and Renderer share the type without a
// circular include.
struct SsaoSettings {
    bool enabled = false;
    float radius = 0.5f;      // horizon-search radius in view-space units
    float intensity = 1.0f;   // occlusion strength multiplier
    float power = 2.0f;       // contrast curve applied to the visibility term
    int sliceCount = 3;       // GTAO slices swept around the view direction
    int stepsPerSlice = 6;    // horizon-march steps per slice, per side
    float falloff = 0.6f;     // 0..1 fraction of the radius over which samples fade
    float thickness = 0.5f;   // view-space thickness heuristic (reserved for denoise)
    // Applies occlusion to the ambient/indirect term inside the main pass rather
    // than multiplying the whole composited scene colour. The multiply darkens
    // direct lighting too, which is physically wrong -- a crease in full sunlight
    // should not go dark. False restores that older path for A/B comparison.
    bool ambientOnly = true;
};

// How far fog is evaluated. Past this the integration is clamped to its last
// slice, so distant geometry keeps the fog colour it had at the volume's edge
// instead of abruptly clearing. Lives here rather than with the other fog grid
// constants because it is the default of a persisted setting; VolumetricFog.h
// re-exports it so the fog math keeps its own vocabulary.
inline constexpr float kDefaultFogMaxDistance = 64.0f;

// Froxel volumetric fog. A 3D volume over the view frustum is filled with
// in-scattered light and integrated front to back, then applied in the main
// shading pass. Disabled by default. Like SsaoSettings this lives with the
// settings rather than in VolumetricFogPass.h, so serialization and clamping can
// see it without pulling in Vulkan -- which is also why the scattering colour is
// three floats instead of a glm::vec3.
struct FogSettings {
    bool enabled = false;
    float density = 0.03f;
    float maxDistance = kDefaultFogMaxDistance;
    // Forward scattering; positive values give the halo when looking toward a
    // light through fog.
    float anisotropy = 0.35f;
    float heightFalloff = 0.06f;
    float baseHeight = 0.0f;
    float ambientScale = 0.6f;
    float scatteringColor[3] = {1.0f, 1.0f, 1.0f};
    // Per-light importance cull. A froxel bounds what each light could still
    // contribute -- brightest channel * intensity * attenuation * the phase
    // peak -- and skips it before the shadow fetch when even that bound falls
    // below this. Zero disables the cull entirely, which is the reference the
    // culled result is judged against.
    float lightCullThreshold = 0.05f;
    // Temporal reprojection. One sample per froxel per frame aliases badly
    // under a high-frequency shadow; jittering the sample and blending against
    // the reprojected previous volume converges it instead.
    bool temporalEnabled = true;
    // Fraction of the previous frame kept. Higher is smoother but smears more
    // when lights or the camera move.
    float temporalBlend = 0.9f;
};

// Spot/point shadows through a quadtree-packed atlas, with optional GPU caster
// culling. These are renderer toggles rather than a tunable struct, so they sit
// at the RuntimeSettings level alongside the culling flags.
struct PunctualShadowSettings {
    bool enabled = true;
    // Measured slower than CPU-side caster culling on this device; off by
    // default, kept because that result is scene-dependent.
    bool gpuCasterCulling = false;
    // Renders the punctual shadow visibility term alone. A render debug view,
    // persisted for the same reason GiSettings::debugPattern is.
    bool debugView = false;
};

struct DebugUiSettings {
    // Master toggle: when false the debug window shows only the common post-process
    // knobs (tone mapping, bloom, SSAO, TAA, exposure); when true it reveals the
    // scene, GPU, diagnostics sections and the side panels.
    bool advancedMode = false;
    bool showRenderGraphPanel = true;
    bool showSceneHierarchyPanel = true;
    bool showMaterialInspectorPanel = true;
    bool showTextureDebugPanel = true;
    bool showRenderTargetDebugPanel = true;
    // Irradiance-probe controls and atlas previews. A window of its own because
    // the main debug panel is taller than the display and its previews would be
    // clipped rather than drawn.
    bool showIrradianceProbePanel = true;
    bool showGpuTimingGraphs = true;
    bool showCullingStats = true;
    bool showExposureGraphs = true;
    uint32_t selectedCsmCascade = 0;
    float renderTargetPreviewExposure = 1.0f;
    float renderTargetPreviewScale = 1.0f;
};

// Screen-space reflections: a view-space linear march with binary refinement
// against the main depth buffer, additively blended into scene color before
// TAA. Requires a samplable main depth image (same gate as SSAO).
struct SsrSettings {
    bool enabled = true;
    int maxSteps = 48;
    int refinementSteps = 5;
    float maxDistance = 30.0f;
    float thickness = 0.35f;
    float intensity = 1.0f;
    // Surfaces rougher than this trace no reflections (mirror-to-glossy fade).
    float maxRoughness = 0.6f;
    float screenEdgeFade = 0.1f;
};

// Discrete mesh level of detail. Selection runs per draw item inside the GPU cull
// dispatch (see cull.comp); these are the knobs uploaded to it each frame. The
// selection math itself lives in renderer/MeshLod.h.
struct LodSettings {
    bool enabled = true;
    // Projected sphere radius, in pixels, at which level 0 is still the right
    // choice. Each halving of the on-screen radius steps one level down.
    float referenceRadiusPixels = 220.0f;
    // Positive biases toward lower detail.
    float bias = 0.0f;
    // Added on top of `bias` for shadow-cascade dispatches, which tolerate
    // simplification far better than the main pass.
    float shadowBias = 1.0f;
    // >= 0 pins every draw item to that level; -1 selects by distance.
    int forcedLod = -1;
    // Tints geometry by the level the cull pass actually chose.
    bool debugHeatmap = false;
};

// Irradiance-probe global illumination. A grid of probes stores incoming
// radiance in small octahedral tiles, plus the distance to what it came from so
// a probe behind a wall can be rejected rather than lighting through it.
//
// This device exposes no ray tracing, so probe radiance is gathered by
// rasterising the scene from each probe rather than by tracing rays; the grid
// therefore stays coarse and the cost goes into update rate. See
// renderer/IrradianceProbes.h.
struct GiSettings {
    bool enabled = false;
    // World position of probe (0, 0, 0), and the spacing between adjacent
    // probes on each axis. Plain floats rather than a ProbeGridBounds so this
    // header stays free of renderer and glm types, like every other struct here.
    float gridOrigin[3] = {-14.0f, 0.5f, -14.0f};
    float gridSpacing[3] = {4.0f, 3.0f, 4.0f};
    // Probes captured per frame, round robin. This is the whole cost control:
    // the grid is 256 probes and each capture is six small rasterisations plus a
    // convolution, so this sets both the per-frame cost and how many frames the
    // grid takes to catch up with a change in the scene.
    int probesPerFrame = 4;
    // Fills the atlases with the direction each texel stands for instead of
    // captured radiance. The way to see whether probe *storage* is correct
    // independently of whether *capture* is, and a standing check on the
    // octahedral border.
    bool debugPattern = false;
    // Display-only gain for the atlas previews and the probe-only view. Probe
    // values are linear radiance well below 1.0, so at 1:1 a correct atlas is
    // indistinguishable from an empty one. Three puts a typical gathered value
    // (~0.26 here) in the upper half of the display range without clipping it,
    // which six did.
    float previewGain = 3.0f;
    // How strongly probe irradiance replaces the constant environment term.
    // Zero disables the lookup outright rather than scaling it to nothing, so
    // the shader needs no separate flag.
    float intensity = 1.0f;
    // How far off a surface the grid is sampled from, in world units. Too small
    // and flat surfaces self-occlude into darkness; too large and light leaks
    // through thin geometry.
    float surfaceBias = 0.3f;
    // Fraction of a probe's previous value kept when it is re-captured. Zero
    // makes each capture overwrite, which is the reference the accumulated
    // result is judged against.
    float hysteresis = 0.7f;
    // Multi-bounce: how much of a captured surface's indirect light comes from
    // the probe grid rather than the constant ambient. Zero is single bounce.
    // Feedback gain is albedo * this, so the series settles at
    // 1 / (1 - albedo * weight) -- see probeBounceAmplification.
    float bounceWeight = 0.5f;
    // Outputs probe irradiance alone instead of shaded colour.
    bool debugIrradianceOnly = false;
};

// Internal render resolution as a fraction of the presentation resolution. The
// scale itself and its bounds live in renderer/RenderScale.h, which this header
// deliberately does not include -- the clamp is applied in RuntimeSettings.cpp,
// which may pull in renderer headers, while this header stays a plain
// description of what is persisted.
struct RenderScaleSettings {
    float scale = 1.0f;
    // Contrast-adaptive sharpening applied in the composite, and only when the
    // frame is actually upscaled -- at scale 1.0 it does nothing, so the default
    // look is unchanged for anyone who never moves the scale.
    //
    // Non-zero by default because the measurement that motivated it was that a
    // 0.5-scale frame without sharpening is too soft to use, and requiring a
    // second opt-in to make the first feature usable is a bad default.
    float sharpness = 0.5f;
};

// Drives RenderScaleSettings::scale from measured GPU frame time. Off by
// default: a portfolio engine is usually being *measured*, and a resolution that
// moves under you invalidates every comparison. The controller itself lives in
// renderer/DynamicResolution.h, which is where the tuning constants are too --
// only the four values a user would actually want to set are persisted.
struct DynamicResolutionSettings {
    bool enabled = false;
    // Ceiling on GPU frame total, not an average to hover on: the controller
    // lowers the scale above this and only raises it once there is a clear
    // margin below. 16.67 ms is the 60 Hz budget.
    float targetFrameMs = 16.67f;
    float minScale = 0.5f;
    float maxScale = 1.0f;
};

// Virtual shadow map (docs/virtual_shadow_maps.md).
//
// Phase 1 scope: enableMarking only turns on the page-MARKING measurement pass.
// It changes nothing the renderer draws -- directional shadows still come
// entirely from the cascades. The point of the phase is to find out how many
// pages a real frame asks for before a physical pool, a per-frame page budget,
// or a sampling path is committed to.
//
// The numeric fields mirror renderer::VsmClipmapSettings, which is where they
// are clamped and unit-tested; this header stays glm-free by design, so the
// struct is duplicated as plain scalars rather than included.
struct VsmSettings {
    // Nested on purpose, each an A/B point: marking measures which pages a frame
    // needs and changes nothing drawn; page rendering additionally allocates and
    // draws them into the pool. Neither replaces the cascades.
    //
    // enableMarking is STARTUP-ONLY, like CsmSettings::cascadeCount: it decides
    // whether the page pool (4096x4096 D32 = 64 MiB) and its per-frame buffers
    // are allocated at all, and nothing but startup can answer that. Off means
    // the whole subsystem costs nothing, which is the point -- an off-by-default
    // feature holding 64 MiB would be a worse trade than the one this project
    // already measured and rejected for bloom aliasing.
    bool enableMarking = false;
    bool enablePageRendering = false;
    // Samples the page pool instead of the cascades for the directional light.
    // Requires the two above; the cascades keep running underneath so the A/B is
    // a single checkbox and the fallback is always one frame away.
    bool enableShadows = false;
    uint32_t clipmapLevels = 8;
    // World units spanned by the finest level's whole virtual texel grid.
    float level0Extent = 4.0f;
    // Above 1.0 asks for coarser levels (cheaper, blurrier); below, finer.
    float texelsPerPixel = 1.0f;
    // Half-thickness of every level's ortho depth range, in world units.
    float depthRange = 250.0f;
    // Pixels per marking thread along each axis. Higher is cheaper and coarser:
    // a block that straddles a page seam only marks the page its centre lands
    // in, which the coarser-level mark then covers.
    //
    // 8 rather than 4 because it was measured: the marking pass takes at most
    // 2x2 taps per block, so a wider block is genuinely fewer fetches, and 8
    // returned an identical page count on both the default and geometry-stress
    // scenes at roughly half the cost. See docs/virtual_shadow_maps.md.
    uint32_t markBlockStride = 8;

    [[nodiscard]] bool operator==(const VsmSettings&) const = default;
};

struct RuntimeSettings {
    RenderScaleSettings renderScale;
    DynamicResolutionSettings dynamicResolution;
    ToneMappingSettings toneMapping;
    BloomSettings bloom;
    TaaSettings taa;
    SsrSettings ssr;
    SsaoSettings ssao;
    FogSettings fog;
    PunctualShadowSettings punctualShadows;
    CsmSettings csm;
    VsmSettings vsm;
    LodSettings lod;
    GiSettings gi;
    DebugUiSettings debugUi;
    // Bind the bloom chain into one shared transient allocation, so resources
    // whose lifetimes do not overlap share bytes. OFF by default until the frame
    // cost of the alias-handoff barriers has been measured -- the memory it
    // saves is not scarce here, so an unmeasured cost is not worth paying by
    // default. Startup/resize applied: it changes how images are allocated.
    bool enableTransientAliasing = false;
    bool useGpuCulling = true;
    bool useGpuShadowCulling = true;
    // Hi-Z occlusion defaults ON now that two-phase re-testing removes the
    // disocclusion false negatives that used to make it opt-in.
    bool enableGpuOcclusionCulling = true;
    // Phase-1 (previous-frame pyramid) cull + phase-2 candidate re-test against a
    // mid-frame rebuild. Off falls back to the conservative single-phase test
    // that only runs while the camera holds still.
    bool enableTwoPhaseOcclusion = true;
    // Render every shadow cascade in one multiview pass instead of one pass per
    // cascade. Startup-only: the shadow pipelines bake the view mask.
    //
    // OFF because it was measured slower here, not because it is unfinished. The
    // output is pixel-identical; MoltenVK just does not have hardware for four
    // views (Apple vertex amplification tops out at two), so the emulation costs
    // more than the three command encoders it removes. See docs/render_scale.md.
    bool enableLayeredCascades = false;
    // Suspend Hi-Z occlusion culling (and the pyramid build that feeds it) while
    // it is culling nothing, re-probing periodically. Never changes the image --
    // skipping occlusion culling can only draw more, never less.
    bool enableAdaptiveOcclusion = true;
    // Main pass walks the per-froxel light list instead of brute-forcing every
    // light. Off is the brute-force reference the clustered result is judged
    // against, so it is worth persisting. Every use site already ANDs this with
    // ClusteredLighting::available(), so it needs no availability guard.
    bool useClusteredLighting = true;
    // Runs ClusterBuild/LightCull on the async compute queue, overlapping the
    // shadow passes. Ignored (graphics-queue fallback) when the device exposes
    // no async-capable queue.
    bool enableAsyncCompute = true;
    bool enableBindlessMaterialTextures = true;
};

// Clamp the GPU-independent runtime settings into valid ranges in place
// (tone-mapping, bloom, TAA, cascade, and debug-UI preview fields). Renderer
// state that is not part of these structs (e.g. GPU occlusion tuning) is clamped
// separately by the caller. Pure and unit-tested.
void clampRuntimeSettings(RenderScaleSettings& renderScale,
                          DynamicResolutionSettings& dynamicResolution,
                          ToneMappingSettings& toneMapping,
                          BloomSettings& bloom,
                          TaaSettings& taa,
                          SsrSettings& ssr,
                          SsaoSettings& ssao,
                          FogSettings& fog,
                          CsmSettings& csm,
                          VsmSettings& vsm,
                          LodSettings& lod,
                          GiSettings& gi,
                          DebugUiSettings& debugUi);

enum class RuntimeSettingsLoadStatus {
    Loaded,
    Missing,
    Malformed,
    IoError
};

struct RuntimeSettingsLoadResult {
    RuntimeSettingsLoadStatus status = RuntimeSettingsLoadStatus::Loaded;
    std::string message;
};

struct RuntimeSettingsSaveResult {
    bool saved = false;
    std::string message;
};

bool loadRuntimeSettings(const std::filesystem::path& path, RuntimeSettings& outSettings);
RuntimeSettingsLoadResult loadRuntimeSettingsDetailed(const std::filesystem::path& path,
                                                      RuntimeSettings& outSettings);

bool saveRuntimeSettings(const std::filesystem::path& path, const RuntimeSettings& settings);
RuntimeSettingsSaveResult saveRuntimeSettingsDetailed(const std::filesystem::path& path,
                                                      const RuntimeSettings& settings);

} // namespace ve
