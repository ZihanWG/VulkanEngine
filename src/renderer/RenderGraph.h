#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace ve::rhi {

class VulkanShadowMap;
class VulkanSwapchain;

} // namespace ve::rhi

namespace ve::renderer {

inline constexpr uint32_t kInvalidRenderGraphHandle = std::numeric_limits<uint32_t>::max();

// A handle names a resource *and* which of its versions. Every declared write
// produces a new version, and the builder hands back a handle for it, so the
// caller has to thread the result forward:
//
//     frame_.sceneColor = builder.writeTexture(frame_.sceneColor, ...);
//
// The version is what makes the declared data flow independent of where a pass
// sits: a reader names the version it consumes, so holding a handle from before
// an intervening write is a detectable mistake rather than an invisible one.
// Barriers and resource lookup ignore it -- all versions of a resource are the
// same VkImage.
struct RGTextureHandle {
    uint32_t index = kInvalidRenderGraphHandle;
    uint32_t version = 0;

    [[nodiscard]] bool valid() const
    {
        return index != kInvalidRenderGraphHandle;
    }
};

struct RGBufferHandle {
    uint32_t index = kInvalidRenderGraphHandle;
    uint32_t version = 0;

    [[nodiscard]] bool valid() const
    {
        return index != kInvalidRenderGraphHandle;
    }
};

enum class RGResourceKind {
    Texture,
    Buffer
};

enum class RGAccess {
    Unknown,
    ShaderRead,
    ColorAttachmentWrite,
    DepthStencilAttachmentWrite,
    StorageImageRead,
    StorageImageWrite,
    StorageImageReadWrite,
    TransferSrc,
    TransferDst,
    Present,
    StorageBufferRead,
    StorageBufferWrite,
    StorageBufferReadWrite,
    // A pass that clears the buffer with vkCmdFillBuffer and then reads and
    // writes it from a shader. The union of the transfer and shader scopes,
    // because a pass declares a resource once and this one genuinely touches it
    // in both.
    //
    // Plain StorageBufferReadWrite here builds a barrier that names the compute
    // stage while the first thing the pass does is a transfer write, so the
    // clear is ordered against nothing that came before. Every call stays legal,
    // which is why only synchronization validation ever saw it: the two GPU
    // culling passes and the histogram pass all described their reset as
    // "clears and writes" and none of their barriers covered the clear.
    StorageBufferClearAndReadWrite,
    IndirectRead,
    HostRead
};

enum class RenderPassType {
    Shadow,
    ShadowGpuCulling,
    VsmPageMark,
    VsmPage,
    VolumetricFog,
    ProbeCapture,
    IrradianceProbes,
    MainGpuCulling,
    DepthPyramid,
    DepthPrepass,
    MainHdr,
    Ssr,
    Gtao,
    GtaoBlur,
    Transparent,
    TaaResolve,
    BloomExtract,
    BloomBlur,
    BloomDownsample,
    BloomUpsample,
    Luminance,
    HistogramExposure,
    Composite,
    ImGui
};

enum class RenderPassExecutionType {
    Graphics,
    Compute,
    Transfer
};

enum class RenderResourceAccess {
    Read,
    Write,
    ReadWrite
};

// What a declared read consumes. The distinction is invisible to barriers -- a
// layout is a layout -- and decides everything else: which pass a reader
// depends on, whether reading before anything wrote is a mistake, and whether a
// producer is worth keeping alive.
enum class RGReadKind {
    // What this frame's graph produced. The ordinary case.
    Produced,
    // What the resource held at the end of the previous frame. Declaring it says
    // the read-before-write is deliberate rather than an ordering mistake.
    History,
    // Nothing. A descriptor binds the resource and the pass samples it, but the
    // result is discarded -- the shader picks between two bloom chains, or a
    // fallback path leaves a binding unused. The declaration exists only to put
    // the image in the layout the descriptor claims, which is the same reason
    // the punctual atlas and the VSM page pool are declared on frames that draw
    // nothing into them.
    //
    // Consuming nothing means keeping nothing alive: a producer whose only
    // reader is layout-only is dead, and culling removes it. That is what makes
    // the unselected bloom chain stop running.
    LayoutOnly
};

struct RGTextureDesc {
    std::string name;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    // Slices of a 3D image; 1 for everything else. Descriptive only -- barriers
    // name a subresource range, which has no depth -- but a volume reported as
    // its base slice reads as a 2D texture in the debug table, and the froxel
    // volumes are the first graph resources that are not flat.
    uint32_t depth = 1;
    VkImageUsageFlags usage = 0;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    VkClearValue clearValue{};
    bool hasClearValue = false;
    bool imported = true;
    // Pool-bound: see RenderGraphImageResource::aliased. Drives the first-use
    // barrier in transitionTexture.
    bool aliased = false;
};

struct RGBufferDesc {
    std::string name;
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    bool imported = true;
};

struct RenderResourceHandle {
    std::string name;
    RGResourceKind kind = RGResourceKind::Texture;
    uint32_t index = kInvalidRenderGraphHandle;
};

struct RenderResourceUsage {
    RenderResourceHandle resource{};
    RenderResourceAccess access = RenderResourceAccess::Read;
    RGAccess declaredAccess = RGAccess::Unknown;
    std::string description;
    // What the read consumes; see RGReadKind. Meaningless on a write.
    RGReadKind readKind = RGReadKind::Produced;
    // The version the declaring handle named, and the version the pass leaves
    // behind: outputVersion is inputVersion + 1 for a write, and equal to it for
    // a plain read. Version 0 is whatever the resource held before the frame.
    uint32_t inputVersion = 0;
    uint32_t outputVersion = 0;
};

// What a declared access means in barrier terms: the layout the image must be
// in, and the stage/access scopes to synchronize against.
struct TextureAccessState {
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
    RGAccess declaredAccess = RGAccess::Unknown;
};

struct BufferAccessState {
    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
    RGAccess declaredAccess = RGAccess::Unknown;
};

// Barrier derivation, split out of RenderGraph for the same reason
// cullUnusedPasses was: pure mappings that need no device to exercise.
//
// The texture mapping depends on the image's aspect, because a depth image asks
// for a depth layout where a colour image asks for the general shader-read one,
// and the presence of stencil picks between the combined and depth-only layouts.
// `currentLayout` is what an access that says nothing about layout falls back to;
// the graph passes the resource's tracked layout.
[[nodiscard]] TextureAccessState
textureAccessState(VkImageAspectFlags aspectMask, RGAccess access, VkImageLayout currentLayout);

[[nodiscard]] BufferAccessState bufferAccessState(RGAccess access);

// Whether a transition has to emit a barrier, split out of transitionTexture /
// transitionBuffer for the same reason cullUnusedPasses was: pure decisions that
// need no device, and both are easy to get subtly wrong in ways nothing checks.
//
// `usedThisFrame` and `previousDeclared` together answer "has this resource been
// touched yet". A resource on its first use with no recorded access needs no
// ordering -- there is nothing to order against -- which is why the two are
// OR'd rather than either being sufficient alone.
[[nodiscard]] bool bufferBarrierRequired(bool usedThisFrame,
                                         RGAccess previousDeclared,
                                         VkAccessFlags2 previousAccess,
                                         VkAccessFlags2 desiredAccess);

// A layout change always needs a barrier. Otherwise the same rule as buffers,
// plus: an UNDEFINED old layout means the contents are not being preserved, so
// there is nothing to order against either.
[[nodiscard]] bool textureBarrierRequired(VkImageLayout oldLayout,
                                          VkImageLayout desiredLayout,
                                          bool usedThisFrame,
                                          RGAccess previousDeclared,
                                          VkAccessFlags2 previousAccess,
                                          VkAccessFlags2 desiredAccess);

// Whether appending a barrier for `resource` to a batch that already holds
// barriers for the resources in `batched` has to submit what is accumulated
// first.
//
// A pass's inferred barriers all sit between the same two points in the command
// stream, so they belong in one vkCmdPipelineBarrier2 -- except that two
// barriers for the same resource inside one VkDependencyInfo are unordered with
// respect to each other. A resource transitioned twice within a single pass
// therefore has to see its first barrier submitted before the second is
// recorded; distinct resources carry no such constraint, which is the whole
// point of batching. Split out as pure index bookkeeping for the same reason
// cullUnusedPasses was: it needs no device to exercise.
//
// Texture and buffer indices live in separate tables, so callers pass the batch
// list matching the resource's kind.
[[nodiscard]] bool barrierBatchNeedsFlush(std::span<const uint32_t> batched, uint32_t resource);

struct RenderPassNode {
    std::string name;
    RenderPassType type = RenderPassType::MainHdr;
    std::vector<RenderResourceUsage> resourceUsages;
    RenderPassExecutionType executionType = RenderPassExecutionType::Graphics;
    std::string transitionSummary;
    bool sideEffect = false;
    bool executed = false;
    bool culled = false;
    std::string cullReason;
    uint32_t generatedBarrierCount = 0;
    uint32_t generatedImageBarrierCount = 0;
    uint32_t generatedBufferBarrierCount = 0;
    // vkCmdPipelineBarrier2 calls the above barriers were submitted in. Normally
    // one: the pass's whole barrier set goes in a single dependency info. Two
    // things raise it -- a resource transitioned more than once inside one pass
    // (see barrierBatchNeedsFlush), and the ImGui pass, whose present transition
    // is recorded after the pass body and so cannot join the pass's own set.
    uint32_t generatedBarrierSubmitCount = 0;
    // Declaration problems validateDeclarations found for this pass, joined for
    // display. Empty on a well-formed pass, which is the normal state.
    std::string declarationIssues;
};

// Declaration validation. The graph's declarations describe what each pass
// touches; nothing until now checked that the description is self-consistent, so
// a pass could read a transient nothing had produced and the only symptom would
// be wrong pixels. These are the three ways a declaration set can be wrong that
// are decidable from the declarations alone.
enum class RGDeclarationIssue {
    // Reads a graph-managed resource no earlier surviving pass wrote this frame,
    // without declaring the read as a history read. Either the pass is ordered
    // wrongly, or it means to read the previous frame and should say so.
    ReadsContentNoPassProduced,
    // A history read of a pool-bound resource. Aliasing hands the bytes to
    // another resource between frames and the handoff barrier discards the
    // contents, so there is no previous frame to read -- this is the rule that
    // says which transients are unsafe to wire into the transient memory pool.
    HistoryReadOfAliasedResource,
    // The same resource declared more than once by one pass. Usually a
    // copy-paste slip, and it costs a barrier submission: two barriers for one
    // resource cannot share a dependency info (see barrierBatchNeedsFlush).
    ResourceDeclaredTwice,
    // A read naming an older version than the one current at that point: the
    // declaring handle was taken before an intervening write and never
    // refreshed. The pixels are right -- every version is the same image -- but
    // the declared data flow points at the wrong producer, so anything that acts
    // on it, a reordering above all, acts on a lie.
    StaleVersionRead,
    // A history read placed after a pass that writes the resource this frame.
    // The declaration says "the previous frame's contents" and the schedule
    // hands it this frame's, so the pass reads something other than what it
    // claims. Unlike the two rules above this one applies to imported resources
    // as well: what makes it wrong is the ordering, not who owns the memory.
    HistoryReadAfterProducer
};

struct RenderGraphDeclarationIssue {
    uint32_t passIndex = 0;
    RGDeclarationIssue issue = RGDeclarationIssue::ReadsContentNoPassProduced;
    RGResourceKind resourceKind = RGResourceKind::Texture;
    uint32_t resourceIndex = 0;
};

// The pass dependency graph the declarations imply, and how much freedom the
// recorded order leaves within it.
//
// The edges are the three orderings a resource forces between two passes:
// read-after-write, write-after-read, and write-after-write. A history read
// creates no read-after-write edge -- it consumes the previous frame, not this
// frame's producer -- which is the one place the distinction changes the graph
// rather than just the diagnostics.
//
// Read the slack, not the legality. Because a pass's declarations are recorded
// in the order the passes run, every edge points backwards by construction and
// the recorded order can never violate one. What the analysis does say, today,
// is how tightly the declared data flow actually pins the order down: a pass
// with slack could be recorded earlier without breaking anything, and the
// longest chain bounds what any scheduler could achieve. Making the legality
// question real needs handle-threaded resource versions in the builder API, so
// that the data flow is stated independently of the order.
struct RenderGraphPassSchedule {
    // Passes that must be recorded before this one, ascending. Empty for a pass
    // nothing constrains, and for a culled pass.
    std::vector<uint32_t> predecessors;
    // Position among the surviving passes, in recording order.
    uint32_t recordedSlot = 0;
    // Earliest position the dependencies allow: the longest chain of
    // predecessors ending at this pass.
    uint32_t earliestSlot = 0;
    // recordedSlot - earliestSlot. Zero means the pass is pinned where it is.
    uint32_t slack = 0;
    // False for a culled pass, whose entry is left empty so the result stays
    // parallel to the pass list.
    bool scheduled = false;
};

// A dependency the proposed order breaks: `passIndex` is placed at or before
// `predecessorIndex`, which it must follow.
struct RenderGraphOrderViolation {
    uint32_t passIndex = 0;
    uint32_t predecessorIndex = 0;
};

// Run-level roll-up of the per-frame backstop results.
//
// The per-frame vectors are cleared by every beginFrame, so a violation in
// frame 7 that clears again by frame 30 leaves nothing behind to find at
// shutdown. What a scripted run needs to know is "did this ever happen", not
// "is it happening in the frame that happened to be last", so the peaks are
// accumulated across the run and reported once.
//
// Peaks rather than sums: the same declaration mistake repeats every frame, so
// a sum would report the frame count multiplied by the fault and read as though
// the run got progressively worse.
struct RenderGraphBackstopSummary {
    uint64_t framesObserved = 0;
    // Frames where any of the three counts was non-zero, or the order could not
    // be resolved. The peaks say how bad it got; this says how often.
    uint64_t framesWithIssues = 0;
    size_t peakOrderViolations = 0;
    size_t peakUnrecordedPasses = 0;
    size_t peakDeclarationIssues = 0;
    // A cycle means executionOrder_ is a partial order, so the schedule the rest
    // of the backstop judged against is itself suspect. Latched, never cleared.
    bool executionOrderCycleDetected = false;

    // Named passes from the first frame that reported anything, one entry per
    // finding. Counts alone turn a CI failure into a local-repro exercise, and
    // the configuration that failed is by definition one nobody was running.
    // Latched on the first offending frame rather than accumulated: the same
    // fault repeats every frame, so appending would grow without bound.
    std::vector<std::string> firstIssueDetails;

    [[nodiscard]] bool clean() const
    {
        return peakOrderViolations == 0 && peakUnrecordedPasses == 0 && peakDeclarationIssues == 0 &&
               !executionOrderCycleDetected;
    }
};

// The one line a scripted run greps for, in the shape of rhi::ValidationTally's
// "Validation tally:" report -- a fixed prefix, then counts a matcher can read.
//
// Free and pure so tests/test_render_graph.cpp can pin the wording: the string
// is the CI contract, and a reworded prefix breaks the gate silently rather
// than loudly.
[[nodiscard]] std::string formatRenderGraphBackstopSummary(const RenderGraphBackstopSummary& summary);

// What validateDeclarations needs to know about a resource, which is only
// whether the graph manages its bytes and whether they are shared.
struct RGResourceValidationInfo {
    // False for imported resources, whose contents persist by contract, so
    // reading one before writing it is always defined.
    bool graphManaged = false;
    // True when the resource is bound into the shared transient pool.
    bool aliased = false;
};

struct RenderGraphImageResource {
    std::string name;
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkExtent2D extent{};
    // See RGTextureDesc::depth.
    uint32_t depth = 1;
    VkImageLayout* layout = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    VkClearValue clearValue{};
    bool hasClearValue = false;
    bool imported = true;
    // True when this image is bound into the shared transient pool, so its bytes
    // belonged to a different resource earlier in the frame.
    bool aliased = false;
};

struct RenderGraphBufferResource {
    std::string name;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    bool imported = true;
};

struct RenderGraphFrameResources {
    // How much of the scene-sized targets this frame writes. They are allocated
    // at the maximum render resolution and only their top-left sub-rect is
    // rendered into, so the renderArea of every pass that writes one is this,
    // not the resource's own extent. Zero means "the whole resource", which is
    // what the passes whose targets are not sub-rected still want.
    VkExtent2D renderExtent{};
    RenderGraphImageResource sceneColor;
    RenderGraphImageResource velocity;
    RenderGraphImageResource normalRoughness;
    RenderGraphImageResource ambientOcclusion;
    RenderGraphImageResource ambientOcclusionRaw;
    RenderGraphImageResource ssrSceneColorCopy;
    RenderGraphImageResource taaHistoryRead;
    RenderGraphImageResource taaHistoryWrite;
    RenderGraphImageResource bloomExtract;
    RenderGraphImageResource bloomPing;
    RenderGraphImageResource bloomPong;
    std::vector<RenderGraphImageResource> bloomDownsampleChain;
    std::vector<RenderGraphImageResource> bloomUpsampleChain;
    RenderGraphImageResource depthPyramid;
    // The froxel volumes. Present whenever the subsystem allocated them, which
    // is not the same as fog being enabled: the material descriptor set binds
    // the integrated volume unconditionally, so it has to hold the layout that
    // sampler3D claims even on a frame that computes no fog.
    //
    // The two scatter volumes ping-pong -- this frame's injection target becomes
    // next frame's reprojection history -- so they are named by role rather than
    // by index, the way the TAA history pair is.
    RenderGraphImageResource fogScatterWrite;
    RenderGraphImageResource fogScatterRead;
    RenderGraphImageResource fogIntegrated;
    RenderGraphImageResource probeIrradianceAtlas;
    RenderGraphImageResource probeDepthAtlas;
    RenderGraphImageResource probeCaptureAtlas;
    RenderGraphImageResource probeCaptureDepth;
    RenderGraphBufferResource mainCullInput;
    RenderGraphBufferResource mainCullIndirectOutput;
    RenderGraphBufferResource mainCullVisibleCounts;
    RenderGraphBufferResource mainCullReadback;
    // The shadow cull's own three. One dispatch produces the union of every
    // cascade frustum, so unlike the main cull there is one set for all four
    // cascades rather than one per pass.
    RenderGraphBufferResource shadowCullIndirectOutput;
    RenderGraphBufferResource shadowCullVisibleCounts;
    RenderGraphBufferResource shadowCullReadback;
    // The punctual atlas cull's pair. One dispatch covers every slot, because
    // compute cannot run inside the atlas's single rendering scope.
    RenderGraphBufferResource punctualCullIndirectOutput;
    RenderGraphBufferResource punctualCullVisibleCounts;
    RenderGraphBufferResource luminancePartials;
    RenderGraphBufferResource luminanceReadback;
    RenderGraphBufferResource luminanceHistogram;
    RenderGraphBufferResource histogramReadback;
    RenderGraphBufferResource exposureState;
    bool taaEnabled = false;
    // Declares the two-phase occlusion passes (mid-frame depth pyramid, cull
    // phase 2, second main HDR pass) for this frame.
    bool twoPhaseOcclusionEnabled = false;
    // Declares the depth prepass for this frame. See
    // RuntimeSettings::enableDepthPrepass; it also selects the main pass's depth
    // load op, because the prepass is what filled the buffer.
    bool depthPrepassEnabled = false;
    // Declares the end-of-frame Hi-Z pyramid build. False when the build is
    // skipped -- occlusion culling off or suspended, and no VSM page marking to
    // feed -- in which case the recorder only invalidates the pyramid and
    // records no pass, so declaring one would model work that does not happen.
    bool depthPyramidBuildEnabled = false;
    // Declares the SSR copy + trace passes for this frame.
    bool ssrEnabled = false;
    // Declares the GTAO horizon-search pass for this frame.
    bool gtaoEnabled = false;
    // Number of alpha-blended draw items this frame. Zero skips declaring the
    // transparent pass entirely, so a fully opaque scene costs nothing.
    uint32_t transparentDrawCount = 0;
    // Punctual shadow atlas tiles allocated this frame. Zero skips declaring the
    // atlas pass, but the atlas texture is still imported and still declared as a
    // main-pass read so the graph transitions it into the layout the material
    // descriptors claim -- otherwise a frame that casts nothing would leave the
    // image in whatever layout it happened to be in.
    uint32_t punctualShadowSlotCount = 0;
    // Declares the virtual-shadow-map page marking compute pass. It reads the
    // previous frame's depth pyramid, so it only exists so the graph has that
    // texture in a sampled layout before the dispatch -- the request buffers it
    // writes are not graph resources and carry their own explicit barriers.
    bool vsmPageMarkEnabled = false;
    // Pages the residency update queued for drawing. Zero skips declaring the
    // page pass, but the pool is still imported and still read by the main pass,
    // so a frame that redraws nothing keeps the layout its sampler claims -- the
    // same asymmetry the punctual shadow atlas uses.
    uint32_t vsmDirtyPageCount = 0;
    // Declares the volumetric fog compute pass for this frame.
    bool volumetricFogEnabled = false;
    // Declares the probe-atlas update compute pass for this frame. The two
    // atlases are imported and read by the main pass whenever they exist, the
    // same asymmetry the punctual shadow atlas uses: a frame that updates no
    // probe still has to leave the atlases in the layout their samplers claim.
    bool irradianceProbeUpdateEnabled = false;
    // Declares the probe capture pass: the scene rasterised from each probe in
    // this frame's batch. Separate from the update flag because the update also
    // runs on frames with nothing to capture -- the cold-start seed, and the
    // debug-pattern path.
    bool probeCaptureEnabled = false;
    // Declares the main compute frustum/occlusion cull. Comes from
    // GpuCulling::willRecordMainCull rather than from a condition rebuilt here,
    // so the declaration and the recorder cannot drift. False on the CPU culling
    // fallback (renderer.useGpuCulling off) and on an empty draw list, both of
    // which the recorder returns on -- this pass was declared unconditionally
    // until the backstop was pointed at that configuration and reported it
    // unrecorded on every frame. The main pass still reads the indirect command
    // buffer either way: the CPU path uploads the same commands from the host.
    bool mainGpuCullingEnabled = false;
    // Declares the GPU shadow caster cull. Must be exactly the condition the
    // recorder uses -- GPU shadow culling active, at least one cascade needing a
    // redraw, and a non-empty draw list -- because a pass declared and never
    // recorded leaves the graph describing work that did not happen, which is
    // what the endFrame backstop exists to catch.
    bool shadowGpuCullingEnabled = false;
    // Declares the punctual atlas caster cull. Comes from
    // PunctualShadows::willRecordCull rather than from a condition rebuilt here,
    // so the declaration and the recorder cannot drift.
    bool punctualShadowCullEnabled = false;
    // Whether any cascade will be redrawn this frame. False skips declaring the
    // cascaded shadow pass, but the shadow map is still imported and still read
    // by the main pass -- the same asymmetry the punctual atlas uses. A fully
    // cached frame redraws no cascade, and declaring a pass the renderer never
    // records would leave the graph modelling work that did not happen.
    bool cascadeShadowRedrawRequired = false;
    // Whether the luminance reduction will record this frame. False skips
    // declaring the pass; histogram mode does not read its output, so the
    // exposure chain is unaffected.
    bool luminancePassEnabled = false;
    // Which bloom chain the composite will sample. The loser's output is
    // declared as a layout-only read, so culling drops the passes that would
    // have filled it; see PostProcessStack::willRecordMipChainBloom.
    bool mipChainBloomSelected = false;
    // Whether the histogram exposure reduction will record this frame. Manual and
    // log-average exposure both return before the pass begins, and declaring it
    // anyway makes the endFrame backstop report a mismatch on a frame where the
    // renderer is behaving correctly.
    bool histogramPassEnabled = false;
};

struct RenderGraphResourceDebugInfo {
    std::string name;
    RGResourceKind kind = RGResourceKind::Texture;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    uint32_t depth = 1;
    VkDeviceSize size = 0;
    VkImageUsageFlags imageUsage = 0;
    VkBufferUsageFlags bufferUsage = 0;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    bool imported = true;
    bool transient = false;
    bool graphManaged = false;
    VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout finalLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

// A recorder the graph may invoke, and the pass whose scheduled position decides
// when it runs. A recorder covering several declared passes names the first of
// them. A recorder with no pass of its own -- the synchronous cluster build, the
// fog volume's first-use clear -- leaves the index invalid and runs with the last
// anchored unit registered before it.
struct RenderGraphScheduledUnit {
    uint32_t passIndex = kInvalidRenderGraphHandle;
    std::function<void()> record;
};

// CPU wall time one scheduled unit spent recording. The GPU profiler answers
// "what did this pass cost the GPU"; this is the other half, and without it the
// only visible number is the whole-frame recording total, which cannot say what
// splitting recording across threads would actually buy.
//
// `name` has static storage duration -- it comes from renderPassTypeName -- so a
// cost outlives the frame whose passes_ produced it, which the per-frame reset
// would otherwise invalidate.
struct RenderGraphUnitCost {
    const char* name = "";
    float milliseconds = 0.0f;
    // Registration index of the unit. Two units can share a pass type name --
    // the cascade and punctual shadow passes both report "Shadow" -- and a
    // breakdown that cannot tell them apart sends you to read the wrong one.
    uint32_t unitIndex = 0;
};

// The passes a caller can anchor a scheduled unit to. Deliberately only the ones
// with a recorder factored out far enough to be invoked by the graph; the rest
// of the frame is still recorded in place.
enum class RenderGraphBuiltinPass {
    VsmPageMark,
    Shadow,
    PunctualShadow,
    MainGpuCulling,
    VolumetricFog,
    ProbeCapture,
    MainHdr,
    DepthPyramid,
    TaaResolve,
    BloomExtract,
    BloomDownsampleFirst,
    Luminance,
    HistogramExposure,
    Composite,
    ImGui
};

class RenderGraph;

class RenderGraphBuilder final {
public:
    RGTextureHandle readTexture(RGTextureHandle handle, RGAccess access, std::string description = {});
    // A read of the previous frame's contents. Identical to readTexture in
    // barrier terms -- the layout and scopes do not care where the pixels came
    // from -- but it tells validateDeclarations that reading before anything
    // wrote the resource this frame is the intent, not an ordering mistake.
    RGTextureHandle readHistoryTexture(RGTextureHandle handle, RGAccess access, std::string description = {});
    // A read whose content does not matter: see RGReadKind::LayoutOnly. It keeps
    // no producer alive, so declaring one is how a pass says "put this in the
    // right layout for my descriptor, but do not compute it on my account".
    RGTextureHandle readTextureForLayout(RGTextureHandle handle, RGAccess access, std::string description = {});
    RGTextureHandle writeTexture(RGTextureHandle handle, RGAccess access, std::string description = {});
    RGTextureHandle readWriteTexture(RGTextureHandle handle, RGAccess access, std::string description = {});
    RGBufferHandle readBuffer(RGBufferHandle handle, RGAccess access, std::string description = {});
    RGBufferHandle writeBuffer(RGBufferHandle handle, RGAccess access, std::string description = {});
    RGBufferHandle readWriteBuffer(RGBufferHandle handle, RGAccess access, std::string description = {});
    void sideEffect(std::string reason = {});

private:
    RenderGraphBuilder(RenderGraph& graph, RenderPassNode& pass);

    RenderGraph& graph_;
    RenderPassNode& pass_;

    friend class RenderGraph;
};

// Render Graph 2.0 keeps the renderer's existing pass recorders in place while
// adding logical handles, declarations, conservative inferred image and buffer
// barriers, transient/imported resource metadata, pass liveness state, and
// debug UI data.
class RenderGraph final {
public:
    using SetupCallback = std::function<void(RenderGraphBuilder&)>;

    RenderGraph();

    // punctualShadowAtlas and vsmPagePool are nullable: both are optional
    // subsystems, and a null pointer simply leaves that texture and its pass out
    // of this frame's graph.
    void beginFrame(VkCommandBuffer commandBuffer,
                    rhi::VulkanSwapchain& swapchain,
                    rhi::VulkanShadowMap& shadowMap,
                    rhi::VulkanShadowMap* punctualShadowAtlas,
                    rhi::VulkanShadowMap* vsmPagePool,
                    uint32_t imageIndex,
                    RenderGraphFrameResources frameResources);
    // Virtual shadow map page marking. Compute, no rendering scope; this exists
    // so the graph transitions the depth pyramid the marking dispatch samples.
    void beginVsmPageMarkPass();
    void endVsmPageMarkPass();
    // Virtual shadow map page rendering. One rendering scope covers the whole
    // physical page pool; the caller sets a viewport/scissor per dirty page
    // inside it, exactly as the punctual atlas does per slot.
    //
    // clearWholePool selects the load op, and for the same reason: false
    // preserves the pages already in the pool so cached ones survive, and the
    // caller clears only the pages it is about to redraw. True is for the first
    // frame after the image is created, when nothing in it can be trusted.
    void beginVsmPagePass(bool clearWholePool);
    void endVsmPagePass();
    void beginShadowPass(uint32_t cascadeLayer);
    // All cascades in one pass via multiview. Requires the shadow map's array
    // view and a pipeline built with the matching view mask.
    void beginLayeredShadowPass(uint32_t cascadeCount);
    void endShadowPass(bool finalCascade);
    // Punctual shadow atlas. One rendering scope covers the whole atlas; the
    // caller sets a viewport/scissor per slot inside it.
    //
    // clearWholeAtlas selects the load op. False preserves the tiles already in
    // the atlas so cached ones survive, which is what per-slot invalidation
    // needs -- the caller then clears just the tiles it is about to redraw.
    // True is for the first frame after the image is (re)created, when its
    // contents are undefined and no tile can be trusted.
    void beginPunctualShadowPass(bool clearWholeAtlas);
    void endPunctualShadowPass();
    // Volumetric fog's compute passes. No rendering scope; this exists so the
    // graph transitions the cascaded shadow map the injection pass samples.
    void beginVolumetricFogPass();
    void endVolumetricFogPass();
    // Irradiance probe atlas maintenance. Compute, no rendering scope; the graph
    // moves both atlases into GENERAL for the dispatches and back to a sampled
    // layout for whoever reads them.
    void beginIrradianceProbePass();
    void endIrradianceProbePass();
    // Probe capture. One rendering scope over the whole capture atlas; the
    // caller sets a viewport per (probe, cube face) inside it.
    void beginProbeCapturePass();
    void endProbeCapturePass();
    void beginMainGpuCullingPass();
    void endMainGpuCullingPass();
    // The GPU shadow caster cull. Recorded inside the cascade shadow recorder,
    // ahead of the first cascade, which is why its declaration sits immediately
    // before CSMShadowPass: a declared order that disagrees with the recorded one
    // is a reordering the graph would then act on.
    void beginShadowGpuCullingPass();
    void endShadowGpuCullingPass();
    // The punctual atlas caster cull, recorded ahead of the atlas pass for the
    // same reason the cascade cull is recorded ahead of the shadow passes.
    void beginPunctualShadowCullPass();
    void endPunctualShadowCullPass();
    // Depth-only replay of the opaque bucket. Declared and recorded only when
    // RuntimeSettings::enableDepthPrepass is on; when it is off the pass does not
    // exist in the graph at all, so the backstop's "declared but never recorded"
    // check keeps meaning what it says.
    void beginDepthPrepass();
    void endDepthPrepass();
    [[nodiscard]] bool hasDepthPrepass() const
    {
        return frame_.passIndices.depthPrepass != kInvalidRenderGraphHandle;
    }

    void beginMainHdrPass();
    void endMainHdrPass();
    // Two-phase occlusion: mid-frame pyramid build, candidate re-test, and the
    // second (load, no clear) main HDR pass for the rescued draws.
    void beginDepthPyramidMidPass();
    void endDepthPyramidMidPass();
    void beginMainGpuCullingPhase2Pass();
    void endMainGpuCullingPhase2Pass();
    void beginMainHdrPhase2Pass();
    void endMainHdrPhase2Pass();
    // SSR: transfer copy of the lit scene color (the trace's reflection source),
    // then the trace pass blending reflections back into scene color (LOAD).
    void beginSsrCopyPass();
    void endSsrCopyPass();
    void beginSsrTracePass();
    void endSsrTracePass();
    // GTAO: horizon-search fullscreen pass writing the raw AO target, then a
    // bilateral blur denoising it into the composite-visible AO target.
    void beginGtaoPass();
    void endGtaoPass();
    void beginGtaoBlurPass();
    void endGtaoBlurPass();
    // Alpha-blended geometry. Scene color only (no velocity, no G-buffer) with
    // depth test on and depth write off, recorded after every screen-space effect
    // that needs a clean opaque depth/G-buffer and before the TAA resolve.
    void beginTransparentPass();
    void endTransparentPass();
    void beginDepthPyramidPass();
    void endDepthPyramidPass();
    void beginTaaResolvePass();
    void endTaaResolvePass();
    void beginBloomExtractPass();
    void endBloomExtractPass();
    void beginBloomBlurPass(bool horizontal);
    void endBloomBlurPass();
    void beginBloomDownsamplePass(uint32_t level);
    void endBloomDownsamplePass();
    void beginBloomUpsamplePass(uint32_t level);
    void endBloomUpsamplePass();
    void beginLuminancePass();
    void endLuminancePass();
    void beginHistogramExposurePass();
    void endHistogramExposurePass();
    void beginCompositePass();
    void endCompositePass();
    void beginImGuiPass();
    void endImGuiPass();
    void endFrame();

    uint32_t addPass(std::string name, SetupCallback setup);
    uint32_t addPass(std::string name,
                     RenderPassType type,
                     RenderPassExecutionType executionType,
                     bool sideEffect,
                     SetupCallback setup);

    RGTextureHandle importTexture(const RenderGraphImageResource& resource);
    RGTextureHandle createTransientTexture(const RGTextureDesc& desc, const RenderGraphImageResource& backingResource);
    RGBufferHandle importBuffer(const RenderGraphBufferResource& resource);

    [[nodiscard]] VkImage image(RGTextureHandle handle) const;
    [[nodiscard]] VkImageView imageView(RGTextureHandle handle) const;
    [[nodiscard]] VkBuffer buffer(RGBufferHandle handle) const;

    [[nodiscard]] const std::vector<RenderPassNode>& passes() const
    {
        return passes_;
    }

    [[nodiscard]] const std::vector<RenderPassNode>& debugPasses() const
    {
        return passes_;
    }

    // Graph-transient textures of the frame just recorded, for the transient
    // memory allocator. Only the name and image handle: RenderGraph holds no
    // VkDevice by design, so querying memory requirements is the caller's job.
    struct TransientTextureRecord {
        std::string name;
        VkImage image = VK_NULL_HANDLE;
        // Index into the graph's texture table, so the caller can line this up
        // with computeTextureLifetimes' output.
        uint32_t resourceIndex = 0;
    };
    [[nodiscard]] std::vector<TransientTextureRecord> transientTextures() const;

    // Bounds the resource indices computeTextureLifetimes needs; the graph's
    // texture table is rebuilt every frame, so this is a per-frame value.
    [[nodiscard]] size_t textureCount() const
    {
        return textures_.size();
    }

    [[nodiscard]] const std::vector<RenderGraphResourceDebugInfo>& debugResources() const
    {
        return debugResources_;
    }

    // This frame's declaration problems. Empty is the normal state.
    [[nodiscard]] const std::vector<RenderGraphDeclarationIssue>& declarationIssues() const
    {
        return declarationIssues_;
    }

    // This frame's derived dependency graph, parallel to passes().
    [[nodiscard]] const std::vector<RenderGraphPassSchedule>& passSchedule() const
    {
        return passSchedule_;
    }

    // The order the graph scheduled this frame's passes in.
    [[nodiscard]] const std::vector<uint32_t>& executionOrder() const
    {
        return executionOrder_;
    }

    // Dependencies the order the passes were actually recorded in broke, and
    // scheduled passes that were never recorded at all. Both are filled at
    // endFrame and survive into the next frame for the debug panel. Empty is the
    // normal state for both.
    [[nodiscard]] const std::vector<RenderGraphOrderViolation>& recordedOrderViolations() const
    {
        return recordedOrderViolations_;
    }

    [[nodiscard]] const std::vector<uint32_t>& unrecordedPassIndices() const
    {
        return unrecordedPassIndices_;
    }

    // The same three numbers rolled up over every frame this graph has ended,
    // for the once-per-run report. See RenderGraphBackstopSummary.
    [[nodiscard]] const RenderGraphBackstopSummary& backstopSummary() const
    {
        return backstopSummary_;
    }

    // Last frame's per-unit recording cost, in the order the units ran. Survives
    // the next beginFrame so a once-per-second report can print it.
    [[nodiscard]] const std::vector<RenderGraphUnitCost>& unitRecordCosts() const
    {
        return unitRecordCosts_;
    }

    // True when the derived graph could not be fully ordered this frame.
    [[nodiscard]] bool executionOrderCycleDetected() const
    {
        return executionOrderCycleDetected_;
    }

    // Index of a pass a scheduled unit can anchor to, or kInvalidRenderGraphHandle
    // when this frame does not declare it.
    [[nodiscard]] uint32_t builtinPassIndex(RenderGraphBuiltinPass pass) const;

    // Invokes each recorder, ordered by where computeExecutionOrder put its pass.
    // Units with no pass this frame, or with equal positions, keep the order they
    // were registered in.
    //
    // This is where the graph stops describing the frame and starts driving it.
    // It drives only what the caller hands it; everything else is still recorded
    // in place, and the endFrame backstop is what covers both.
    void recordScheduledUnits(std::span<RenderGraphScheduledUnit> units);

    // Aliases kept so existing RenderGraph::TextureAccessState spellings still
    // compile; the types themselves live at namespace scope below so the pure
    // derivation functions can return them.
    using TextureAccessState = ve::renderer::TextureAccessState;
    using BufferAccessState = ve::renderer::BufferAccessState;

private:
    // Shared body of the per-cascade and layered shadow passes; viewMask == 0
    // selects the single-layer form.
    void beginShadowPassInternal(uint32_t cascadeLayer, uint32_t viewMask);
    enum class ActivePass {
        None,
        VsmPageMark,
        VsmPage,
        Shadow,
        PunctualShadow,
        VolumetricFog,
        ProbeCapture,
        IrradianceProbes,
        MainGpuCulling,
        ShadowGpuCulling,
        PunctualShadowCull,
        DepthPrepass,
        MainHdr,
        MainGpuCullingPhase2,
        MainHdrPhase2,
        SsrCopy,
        SsrTrace,
        Gtao,
        GtaoBlur,
        Transparent,
        DepthPyramidMid,
        DepthPyramid,
        TaaResolve,
        BloomExtract,
        BloomBlur,
        BloomDownsample,
        BloomUpsample,
        Luminance,
        HistogramExposure,
        Composite,
        ImGui
    };

    enum class TextureOwner {
        VsmPagePool,
        ExternalLayoutPointer,
        SwapchainColor,
        SwapchainDepth,
        ShadowMap,
        PunctualShadowAtlas
    };

    struct TextureResource {
        RGTextureDesc desc;
        // Versions produced so far this frame. 0 means nothing has written it.
        uint32_t currentVersion = 0;
        VkImage image = VK_NULL_HANDLE;
        VkImageView imageView = VK_NULL_HANDLE;
        VkImageLayout* externalLayout = nullptr;
        TextureOwner owner = TextureOwner::ExternalLayoutPointer;
        VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        TextureAccessState lastAccess{};
        bool usedThisFrame = false;
        bool graphManaged = false;
    };

    // One pass's inferred barriers, accumulated so the whole set is submitted as
    // a single vkCmdPipelineBarrier2 rather than one call per resource.
    //
    // The graph owns one of these and reuses it (barrierBatch_) rather than
    // building a fresh one per pass: this sits on the command-recording path, and
    // a local would mean a handful of small allocations for every pass of every
    // frame. reset() empties it without giving up the capacity.
    struct BarrierBatch {
        std::vector<VkImageMemoryBarrier2> imageBarriers;
        std::vector<VkBufferMemoryBarrier2> bufferBarriers;
        // Resource indices already represented in the batch, per resource table.
        std::vector<uint32_t> touchedTextures;
        std::vector<uint32_t> touchedBuffers;
        // Submissions made since the last reset, including the final flush.
        uint32_t submitCount = 0;

        void reset()
        {
            imageBarriers.clear();
            bufferBarriers.clear();
            touchedTextures.clear();
            touchedBuffers.clear();
            submitCount = 0;
        }
    };

    struct BufferResource {
        RGBufferDesc desc;
        uint32_t currentVersion = 0;
        VkBuffer buffer = VK_NULL_HANDLE;
        BufferAccessState lastAccess{};
        bool usedThisFrame = false;
        bool graphManaged = false;
    };

    struct BuiltinPassIndices {
        uint32_t vsmPageMark = kInvalidRenderGraphHandle;
        uint32_t vsmPage = kInvalidRenderGraphHandle;
        uint32_t shadow = kInvalidRenderGraphHandle;
        uint32_t punctualShadow = kInvalidRenderGraphHandle;
        uint32_t volumetricFog = kInvalidRenderGraphHandle;
        uint32_t probeCapture = kInvalidRenderGraphHandle;
        uint32_t irradianceProbes = kInvalidRenderGraphHandle;
        uint32_t mainGpuCulling = kInvalidRenderGraphHandle;
        uint32_t shadowGpuCulling = kInvalidRenderGraphHandle;
        uint32_t punctualShadowCull = kInvalidRenderGraphHandle;
        uint32_t depthPrepass = kInvalidRenderGraphHandle;
        uint32_t mainHdr = kInvalidRenderGraphHandle;
        uint32_t depthPyramidMid = kInvalidRenderGraphHandle;
        uint32_t mainGpuCullingPhase2 = kInvalidRenderGraphHandle;
        uint32_t mainHdrPhase2 = kInvalidRenderGraphHandle;
        uint32_t ssrCopy = kInvalidRenderGraphHandle;
        uint32_t ssrTrace = kInvalidRenderGraphHandle;
        uint32_t gtao = kInvalidRenderGraphHandle;
        uint32_t gtaoBlur = kInvalidRenderGraphHandle;
        uint32_t transparent = kInvalidRenderGraphHandle;
        uint32_t depthPyramid = kInvalidRenderGraphHandle;
        uint32_t taaResolve = kInvalidRenderGraphHandle;
        uint32_t bloomExtract = kInvalidRenderGraphHandle;
        uint32_t bloomBlurHorizontal = kInvalidRenderGraphHandle;
        uint32_t bloomBlurVertical = kInvalidRenderGraphHandle;
        std::vector<uint32_t> bloomDownsampleChain;
        std::vector<uint32_t> bloomUpsampleChain;
        uint32_t luminance = kInvalidRenderGraphHandle;
        uint32_t histogramExposure = kInvalidRenderGraphHandle;
        uint32_t composite = kInvalidRenderGraphHandle;
        uint32_t imgui = kInvalidRenderGraphHandle;
    };

    struct FrameState {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        rhi::VulkanSwapchain* swapchain = nullptr;
        rhi::VulkanShadowMap* shadowMap = nullptr;
        // Null when the punctual atlas failed to allocate; every use is guarded,
        // and the pass is simply never declared in that case.
        rhi::VulkanShadowMap* punctualShadowAtlas = nullptr;
        // Null when the page pool failed to allocate or the subsystem is off.
        rhi::VulkanShadowMap* vsmPagePool = nullptr;
        RenderGraphFrameResources resources{};
        BuiltinPassIndices passIndices{};
        uint32_t imageIndex = 0;
        VkImage swapchainImage = VK_NULL_HANDLE;
        RGTextureHandle swapchainColor{};
        RGTextureHandle mainDepth{};
        RGTextureHandle shadowMapDepth{};
        RGTextureHandle punctualShadowAtlasDepth{};
        RGTextureHandle vsmPagePoolDepth{};
        RGTextureHandle sceneColor{};
        RGTextureHandle velocity{};
        RGTextureHandle normalRoughness{};
        RGTextureHandle ambientOcclusion{};
        RGTextureHandle ambientOcclusionRaw{};
        RGTextureHandle ssrSceneColorCopy{};
        RGTextureHandle taaHistoryRead{};
        RGTextureHandle taaHistoryWrite{};
        // Deliberately absent: the post-process source used to be snapshotted
        // here before any pass had declared anything, which made every reader of
        // it name version 0 of a target four writes old. It is resolved at the
        // point of use instead; see RenderGraph::postProcessSource.
        bool taaHistoryIsPostProcessSource = false;
        RGTextureHandle bloomExtract{};
        RGTextureHandle bloomPing{};
        RGTextureHandle bloomPong{};
        std::vector<RGTextureHandle> bloomDownsampleChain;
        std::vector<RGTextureHandle> bloomUpsampleChain;
        RGTextureHandle depthPyramid{};
        RGTextureHandle fogScatterWrite{};
        RGTextureHandle fogScatterRead{};
        RGTextureHandle fogIntegrated{};
        RGTextureHandle probeIrradianceAtlas{};
        RGTextureHandle probeDepthAtlas{};
        RGTextureHandle probeCaptureAtlas{};
        RGTextureHandle probeCaptureDepth{};
        RGBufferHandle mainCullInput{};
        RGBufferHandle mainCullIndirectOutput{};
        RGBufferHandle mainCullVisibleCounts{};
        RGBufferHandle mainCullReadback{};
        RGBufferHandle shadowCullIndirectOutput{};
        RGBufferHandle shadowCullVisibleCounts{};
        RGBufferHandle shadowCullReadback{};
        RGBufferHandle punctualCullIndirectOutput{};
        RGBufferHandle punctualCullVisibleCounts{};
        RGBufferHandle luminancePartials{};
        RGBufferHandle luminanceReadback{};
        RGBufferHandle luminanceHistogram{};
        RGBufferHandle histogramReadback{};
        RGBufferHandle exposureState{};
    };

    // The HDR target the post-process chain reads: the TAA resolve's output when
    // TAA is on, the scene colour otherwise. Resolved on each call rather than
    // stored, so it always names the current version of whichever it is.
    [[nodiscard]] RGTextureHandle postProcessSource() const;
    void requireFrameActive(const char* operation) const;
    // beginFrame() helpers (see RenderGraph.cpp): import the externally-owned
    // swapchain/shadow targets, create the transient frame textures, and import
    // the per-frame buffers. Verbatim slices of the former monolithic beginFrame.
    void importExternalFrameTargets(rhi::VulkanSwapchain& swapchain,
                                    rhi::VulkanShadowMap& shadowMap,
                                    rhi::VulkanShadowMap* punctualShadowAtlas,
                                    rhi::VulkanShadowMap* vsmPagePool,
                                    uint32_t imageIndex);
    void createTransientFrameTextures();
    void importFrameBuffers();
    void buildFrameGraphDeclarations();
    // buildFrameGraphDeclarations() groups (see RenderGraph.cpp); each declares a
    // contiguous run of passes in the same order as before.
    void declareGeometryPasses();
    void declareBloomAndTaaPasses();
    void declareExposureCompositePasses();
    void compilePassCulling();
    // Runs validateDeclarations over the culled pass list and writes what it
    // finds onto the pass nodes, where the debug panel picks it up.
    void validateFrameDeclarations();
    bool beginDeclaredPass(uint32_t passIndex);
    // One human-readable line per finding in the frame current when this is
    // called. Latched into backstopSummary_ on the first offending frame.
    [[nodiscard]] std::vector<std::string> describeBackstopFindings() const;
    // Both record into `batch` instead of submitting, and return the number of
    // barriers they contributed (0 or 1) so the pass's counters keep meaning the
    // same thing they did when every transition submitted on its own. Either may
    // flush `batch` first when the resource is already in it.
    uint32_t transitionTexture(RGTextureHandle handle, RGAccess access, BarrierBatch& batch);
    uint32_t transitionBuffer(RGBufferHandle handle, RGAccess access, BarrierBatch& batch);
    // Submits the accumulated barriers as one dependency and empties the batch.
    // A no-op on an empty batch, so it does not count as a submission.
    void flushBarrierBatch(BarrierBatch& batch);
    [[nodiscard]] TextureAccessState accessStateForTexture(const TextureResource& resource, RGAccess access) const;
    [[nodiscard]] BufferAccessState accessStateForBuffer(RGAccess access) const;
    [[nodiscard]] VkImageLayout currentTextureLayout(const TextureResource& resource) const;
    void setTextureLayout(TextureResource& resource, VkImageLayout layout);
    [[nodiscard]] RenderResourceHandle textureResourceHandle(RGTextureHandle handle) const;
    [[nodiscard]] RenderResourceHandle bufferResourceHandle(RGBufferHandle handle) const;
    // Both return the handle naming the version the pass leaves behind: the next
    // one for a write, the caller's own for a read.
    RGTextureHandle addTextureUsage(RenderPassNode& pass,
                                    RGTextureHandle handle,
                                    RenderResourceAccess resourceAccess,
                                    RGAccess declaredAccess,
                                    std::string description,
                                    RGReadKind readKind = RGReadKind::Produced);
    RGBufferHandle addBufferUsage(RenderPassNode& pass,
                                  RGBufferHandle handle,
                                  RenderResourceAccess resourceAccess,
                                  RGAccess declaredAccess,
                                  std::string description);
    void refreshDebugResources();
    // extentOverride limits the renderArea to the sub-rect a scene-sized target
    // is actually written in; zero means the whole resource.
    void beginColorRendering(const TextureResource& resource,
                             VkClearValue clearValue,
                             VkExtent2D extentOverride = VkExtent2D{0, 0});
    // renderArea for a pass writing a scene-sized target. Clamped to the
    // resource so a render extent that has not caught up with a resize becomes a
    // smaller area rather than an out-of-bounds one.
    [[nodiscard]] VkExtent2D sceneRenderArea(VkExtent2D resourceExtent) const;
    // Shared main-HDR dynamic-rendering setup; loadExisting selects LOAD ops for
    // the phase-2 pass instead of the phase-1 clears.
    // `loadExisting` covers the colour attachments; `loadDepth` is separate
    // because the depth prepass makes the two differ -- it has already written
    // depth that must be kept, while the colour targets still need clearing.
    void beginMainHdrRendering(bool loadExisting, bool loadDepth);
    void beginSwapchainRendering(VkClearValue clearValue, VkAttachmentLoadOp loadOp);

    FrameState frame_{};
    // Reused across passes and frames; see BarrierBatch. Always empty between
    // passes, because every user resets it on entry and flushes it on exit.
    BarrierBatch barrierBatch_{};
    std::vector<TextureResource> textures_;
    std::vector<BufferResource> buffers_;
    std::vector<RenderPassNode> passes_;
    std::vector<RenderGraphResourceDebugInfo> debugResources_;
    // This frame's declaration problems, and the per-resource inputs the check
    // needs. All three are members rather than locals so the per-frame check
    // costs no allocations once the vectors have grown.
    std::vector<RenderGraphDeclarationIssue> declarationIssues_;
    std::vector<RGResourceValidationInfo> textureValidationInfo_;
    std::vector<RGResourceValidationInfo> bufferValidationInfo_;
    std::vector<RenderGraphPassSchedule> passSchedule_;
    std::vector<uint32_t> executionOrder_;
    bool executionOrderCycleDetected_ = false;
    // Pass indices in the order beginDeclaredPass first saw them, which is the
    // order the renderer actually recorded the frame in. A pass begun more than
    // once -- the cascaded shadow pass, once per cascade -- appears once.
    std::vector<uint32_t> recordedOrder_;
    std::vector<RenderGraphOrderViolation> recordedOrderViolations_;
    std::vector<uint32_t> unrecordedPassIndices_;
    // Accumulated across frames, never cleared by beginFrame -- that is the
    // whole point of it.
    RenderGraphBackstopSummary backstopSummary_;
    std::vector<RenderGraphUnitCost> unitRecordCosts_;
    bool frameActive_ = false;
    ActivePass activePass_ = ActivePass::None;

    friend class RenderGraphBuilder;
};

// Backward liveness sweep that marks passes whose declared writes nothing later
// reads. A free function rather than a private method because it is the graph's
// one piece of non-trivial pure logic and needs no device to exercise -- the same
// split ClusterGrid.h, CascadeMath.h, and VolumetricFog.h use.
//
// `textureCount` and `bufferCount` bound the resource indices; a usage pointing
// past either is ignored rather than treated as live, matching how the graph
// tolerates handles it never imported.
void cullUnusedPasses(std::vector<RenderPassNode>& passes, size_t textureCount, size_t bufferCount);

// Inclusive first/last slot each texture is live across, for the transient
// memory allocator (see TransientMemoryPlan.h). A free function next to
// cullUnusedPasses for the same reason: it is pure logic over declarations and
// needs no device to exercise.
//
// The slots are positions in `order` -- the order the passes actually run in --
// not declaration indices, and this is the one analysis where the difference is
// not cosmetic. The intervals decide which resources may share bytes, so an
// interval that does not describe the real timeline lets two simultaneously live
// resources overlap in memory. Nothing catches that: not the validation layer,
// not a barrier, not a test that does not happen to read the clobbered pixels.
//
// Passing the execution order also means culled passes are absent by
// construction rather than skipped, so the intervals no longer carry gaps where
// a culled pass sat. Tighter intervals only ever allow more sharing.
//
// A texture no surviving pass touches comes back with used == false and
// firstPass > lastPass, matching the empty-lifetime convention
// TransientAllocationRequest uses to drop a resource.
//
// A texture anything history-reads comes back live for the whole frame,
// whatever its usages say. Its bytes have to survive from the write in frame N
// to the read in frame N+1, and an interval derived from one frame cannot
// express that: written at slot 17 and history-read at slot 4, it looks free
// over [0, 3], and whatever the allocator packs there overwrites at the start of
// frame N+1 exactly what frame N left to be read. The read happens after the
// clobber, so only the pixels ever say so.
struct RenderGraphResourceLifetime {
    uint32_t firstPass = 1;
    uint32_t lastPass = 0;
    bool used = false;
};

[[nodiscard]] std::vector<RenderGraphResourceLifetime> computeTextureLifetimes(
    const std::vector<RenderPassNode>& passes, std::span<const uint32_t> order, size_t textureCount);

// Checks a frame's declarations against the three rules above. Pure logic over
// declarations, next to cullUnusedPasses for the same reason: no device needed,
// and a mistake here is invisible to the validation layer.
//
// Culled passes are skipped: their declarations describe work that will not run.
// Resource indices past either span are ignored, matching how the rest of the
// graph tolerates handles it never imported.
//
// Reports rather than throws. Every rule here has a legitimate-looking shape
// that only the author can adjudicate, and a graph that refuses to render is a
// worse diagnostic than one that renders and says what looks wrong.
[[nodiscard]] std::vector<RenderGraphDeclarationIssue>
validateDeclarations(const std::vector<RenderPassNode>& passes,
                     std::span<const RGResourceValidationInfo> textures,
                     std::span<const RGResourceValidationInfo> buffers);

// Derives the schedule above from the declarations. Pure logic next to
// cullUnusedPasses and validateDeclarations, for the same reason: no device, and
// it is the data any future reordering would consume.
//
// Culled passes are skipped entirely -- they neither produce for nor constrain
// anything, since they do not run.
[[nodiscard]] std::vector<RenderGraphPassSchedule> computePassSchedule(const std::vector<RenderPassNode>& passes);

// Checks a proposed execution order against the derived dependency graph. This
// is the question versioned handles make answerable: the edges come from the
// declared version chain rather than from where a pass happens to sit, so an
// order that is not the recorded one can be judged, and rejected.
//
// `order` lists the pass indices to run, in the proposed order. A pass that is
// scheduled but missing from `order` is reported against each predecessor it
// would have followed, since dropping it is not an ordering the graph allows.
[[nodiscard]] std::vector<RenderGraphOrderViolation>
validatePassOrder(const std::vector<RenderGraphPassSchedule>& schedule, std::span<const uint32_t> order);

// A topological order over the derived dependency graph: the order the graph
// says the passes may run in.
//
// Ties break on declaration index, which makes the result deterministic and,
// whenever the declaration order is itself legal, identical to it. That is the
// intended outcome rather than a limitation: every derived edge points backwards
// by construction, so making the graph the authority is meant to change nothing
// on its own. What it changes is who decides.
struct RenderGraphExecutionOrder {
    // Scheduled passes, in execution order. Culled passes are absent.
    std::vector<uint32_t> order;
    // True when the graph could not be fully ordered. The passes involved are
    // left out of `order` rather than emitted in an arbitrary place, so a cycle
    // shows up as unrecorded passes instead of as silently wrong output. It
    // cannot happen while declarations are written in a legal order; the flag
    // exists so that a policy or a mis-declared frame that breaks that has a
    // symptom.
    bool cycleDetected = false;
};

[[nodiscard]] RenderGraphExecutionOrder computeExecutionOrder(const std::vector<RenderGraphPassSchedule>& schedule);

// Scheduled passes missing from an order, ascending.
//
// validatePassOrder reports a missing pass only through the edges it breaks, so
// one with no predecessors would otherwise go unnoticed. A declared pass the
// renderer never records means the graph's model of the frame -- its culling,
// its barriers, its resource lifetimes -- describes work that did not happen.
[[nodiscard]] std::vector<uint32_t> unrecordedPasses(const std::vector<RenderGraphPassSchedule>& schedule,
                                                     std::span<const uint32_t> order);

// Longest chain of dependent passes in a schedule, in passes. The floor on how
// many sequential steps the frame needs however it is reordered; 0 for an empty
// or fully culled graph.
[[nodiscard]] uint32_t longestPassChain(const std::vector<RenderGraphPassSchedule>& schedule);

// One-line description of an issue, for the debug panel and for tests.
[[nodiscard]] const char* renderGraphDeclarationIssueName(RGDeclarationIssue issue);

[[nodiscard]] const char* renderPassTypeName(RenderPassType type);
[[nodiscard]] const char* renderPassExecutionTypeName(RenderPassExecutionType executionType);
[[nodiscard]] const char* renderResourceAccessName(RenderResourceAccess access);
[[nodiscard]] const char* renderGraphResourceKindName(RGResourceKind kind);
[[nodiscard]] const char* rgAccessName(RGAccess access);

} // namespace ve::renderer
