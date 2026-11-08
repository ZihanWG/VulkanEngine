#include "renderer/Renderer.h"

#include "renderer/RendererInternal.h"

#include <cstdint>
#include <string>

// The frame's passes, declared beside the commands that record them: each
// addPass takes its resource declarations and its execute callback together.
// Declaration order is the frame's order -- every edge the graph derives points
// backwards, and the schedule keeps ties in declaration order.

namespace ve {

namespace {

// Every pass that runs the main fragment shader declares this, because that
// shader is what traces: the main pass, the phase-2 re-draw and the transparent
// pass. Declares nothing on a frame that does not trace sun shadows -- including
// one on which only the probe gather traces, where the structure is imported
// but the main shader stays on the shadow-map path.
void declareRayTracedShadowRead(renderer::RenderGraphBuilder& builder,
                                const renderer::RenderGraphBlackboard& board,
                                const renderer::RenderGraphFrameResources& resources)
{
    if (board.rayTracingTopLevel.valid() && resources.rayTracedShadowsEnabled) {
        builder.readBuffer(board.rayTracingTopLevel,
                           renderer::RGAccess::AccelerationStructureRead,
                           "Traces sun shadow rays against the top-level acceleration structure.");
    }
}

// The same three passes read the ray-traced shadow mask. Declared on every frame
// the mask exists, unlike the TLAS read: the ray-query variant samples binding 17
// on every frame it runs, traced or not.
void declareRayTracedShadowMaskRead(renderer::RenderGraphBuilder& builder, const renderer::RenderGraphBlackboard& board)
{
    if (board.rayTracedShadowMask.valid()) {
        builder.readTexture(board.rayTracedShadowMask,
                            renderer::RGAccess::ShaderRead,
                            "Reads each pixel's traced sun visibility where the depth prepass drew it.");
    }
}

} // namespace

void Renderer::declareFrameGraph(MainPassFrameState& mainPassState, uint32_t imageIndex, float probeDebugGain)
{
    using namespace renderer;
    RenderGraph& graph = renderGraph_;
    RenderGraphBlackboard& board = graph.blackboard();
    const RenderGraphFrameResources& resources = graph.frameResources();
    FrameGraphPasses& passes = frameGraphPasses_;
    passes = {};

    // First pass of the frame: it reads the depth pyramid the PREVIOUS frame
    // left behind, so it has to be declared before anything this frame writes.
    if (resources.vsmPageMarkEnabled) {
        passes.vsmPageMark = graph.addPass(
            "VsmPageMarkPass",
            RenderPassType::VsmPageMark,
            RenderPassExecutionType::Compute,
            // Side effect: the page-request bitmask it writes is not a graph
            // resource, so nothing downstream declares a read and liveness
            // analysis would cull the pass away.
            true,
            [&](RenderGraphBuilder& builder) {
                builder.readHistoryTexture(board.depthPyramid,
                                           RGAccess::ShaderRead,
                                           "Samples the previous frame's Hi-Z depth to work out which shadow "
                                           "pages this frame's visible surfaces need.");
            },
            {"", nullptr, [this](RenderGraphPassContext& context) { recordVsmPageMarkPass(context.commandBuffer()); }});
    }

    // Only declared when the residency update actually queued pages. The pool is
    // still imported and still read by the main pass below, so a frame that
    // redraws nothing gets the read-layout transition without the write pass --
    // the same asymmetry the punctual shadow atlas uses.
    //
    // The page cull feeds it: one compute dispatch over every dirty page, on
    // request buffers that are not graph resources, so its place between the
    // marking and the page pass is an explicit edge rather than an inferred one.
    if (board.vsmPagePoolImported && resources.vsmDirtyPageCount > 0) {
        passes.vsmPageCull = graph.addWorkPass(
            "VsmPageCullPass",
            RenderPassType::VsmPageCull,
            RenderPassExecutionType::Compute,
            {passes.vsmPageMark},
            {"", nullptr, [this](RenderGraphPassContext& context) { recordVsmPageCull(context.commandBuffer()); }});
        passes.vsmPage =
            graph.addPass("VsmPagePass",
                          RenderPassType::VsmPage,
                          RenderPassExecutionType::Graphics,
                          false,
                          [&](RenderGraphBuilder& builder) {
                              builder.orderAfter(passes.vsmPageCull);
                              board.vsmPagePoolDepth = builder.writeTexture(
                                  board.vsmPagePoolDepth,
                                  RGAccess::DepthStencilAttachmentWrite,
                                  "Draws this frame's dirty clipmap pages into the virtual shadow page pool.");
                          },
                          {"VsmPagePass", "VsmPagePass", [this](RenderGraphPassContext& context) {
                               recordVsmPagePass(context.commandBuffer());
                           }});
    }

    // Marking on, page rendering off: the pool exists but is not a graph
    // resource this frame, and still owes its one move to the layout the
    // material descriptors claim.
    if (resources.vsmPagePoolIdleTransitionPending) {
        passes.vsmPagePoolIdle = graph.addWorkPass("VsmPagePoolIdle",
                                                   RenderPassType::VsmPagePoolIdle,
                                                   RenderPassExecutionType::Transfer,
                                                   {},
                                                   {"", nullptr, [this](RenderGraphPassContext& context) {
                                                        recordVsmPagePoolIdleTransition(context.commandBuffer());
                                                    }});
    }

    // Skipped on a fully cached frame, when the renderer redraws no cascade.
    // The shadow map stays imported and the main pass still declares its read,
    // so it keeps the layout its sampler claims; only the write pass goes away.
    // Ahead of CSMShadowPass, because that is where it records: the cull runs
    // once ahead of the cascade loop. One dispatch produces the union of every
    // cascade frustum, so there is one of these however many cascades are
    // redrawn.
    //
    // Both sit in one scope, which is what the cascade profiler number has
    // always covered: the cull and every cascade.
    graph.pushScope("CSMShadowPass", "CSMShadowPass");
    const bool shadowCullDeclared = resources.shadowGpuCullingEnabled && board.shadowCullIndirectOutput.valid() &&
                                    board.shadowCullVisibleCounts.valid();
    if (shadowCullDeclared) {
        passes.shadowGpuCulling = graph.addPass(
            "ShadowGpuCullingPass",
            RenderPassType::ShadowGpuCulling,
            RenderPassExecutionType::Compute,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(board.depthPyramid,
                                    RGAccess::ShaderRead,
                                    "Samples the previous-frame Hi-Z pyramid for caster occlusion tests.");
                board.shadowCullIndirectOutput =
                    builder.writeBuffer(board.shadowCullIndirectOutput,
                                        RGAccess::StorageBufferClearAndReadWrite,
                                        "Clears and writes indirect draw commands for the cascade union.");
                board.shadowCullVisibleCounts =
                    builder.writeBuffer(board.shadowCullVisibleCounts,
                                        RGAccess::StorageBufferClearAndReadWrite,
                                        "Clears and writes per-batch visible caster counts.");
                if (board.shadowCullReadback.valid()) {
                    board.shadowCullReadback = builder.writeBuffer(board.shadowCullReadback,
                                                                   RGAccess::TransferDst,
                                                                   "Receives the caster counts for CPU readback.");
                }
            },
            {"", "ShadowGpuCulling", [this](RenderGraphPassContext& context) {
                 recordGpuShadowCullingCommands(context.commandBuffer());
             }});
    }

    if (resources.cascadeShadowRedrawRequired) {
        passes.shadow =
            graph.addPass("CSMShadowPass",
                          RenderPassType::Shadow,
                          RenderPassExecutionType::Graphics,
                          false,
                          [&, shadowCullDeclared](RenderGraphBuilder& builder) {
                              board.shadowMapDepth =
                                  builder.writeTexture(board.shadowMapDepth,
                                                       RGAccess::DepthStencilAttachmentWrite,
                                                       "Writes cascaded shadow-map depth array layers.");
                              if (shadowCullDeclared) {
                                  // The consumer edge the cull's own barriers used to
                                  // carry: compute writes the commands, the cascade
                                  // replay reads them as indirect draws.
                                  builder.readBuffer(board.shadowCullIndirectOutput,
                                                     RGAccess::IndirectRead,
                                                     "Replays the culled caster list as indirect draws per cascade.");
                                  builder.readBuffer(board.shadowCullVisibleCounts,
                                                     RGAccess::IndirectRead,
                                                     "Reads per-batch caster counts for indirect-count drawing.");
                              }
                          },
                          {"", nullptr, [this](RenderGraphPassContext& context) {
                               recordCascadeShadowCascades(context.commandBuffer());
                           }});
    }
    // What the cascades now hold, committed every frame -- including a fully
    // cached one, which declares neither pass above. Inside the scope, so the
    // CSMShadowPass profiler number exists every frame, at its real cost of
    // nothing when everything was cached.
    passes.cascadeShadowCache =
        graph.addWorkPass("CSMShadowCache",
                          RenderPassType::CascadeShadowCache,
                          RenderPassExecutionType::Graphics,
                          {passes.shadow},
                          {"", nullptr, [this](RenderGraphPassContext&) { finishCascadeShadowFrame(); }});
    graph.popScope();

    // Only declared when a light actually got a tile. The atlas texture is
    // still imported and still read by the main pass below, so a frame that
    // casts nothing gets the read-layout transition without the write pass.
    // Ahead of the atlas pass, because that is where it records: the cull cannot
    // run inside the atlas's rendering scope, so every slot is culled up front.
    const bool punctualCullDeclared = resources.punctualShadowCullEnabled && board.punctualCullIndirectOutput.valid() &&
                                      board.punctualCullVisibleCounts.valid();
    if (punctualCullDeclared) {
        passes.punctualShadowCull =
            graph.addPass("PunctualShadowCullPass",
                          RenderPassType::ShadowGpuCulling,
                          RenderPassExecutionType::Compute,
                          false,
                          [&](RenderGraphBuilder& builder) {
                              board.punctualCullIndirectOutput = builder.writeBuffer(
                                  board.punctualCullIndirectOutput,
                                  RGAccess::StorageBufferClearAndReadWrite,
                                  "Clears and writes per-slot indirect draw commands for the atlas.");
                              board.punctualCullVisibleCounts =
                                  builder.writeBuffer(board.punctualCullVisibleCounts,
                                                      RGAccess::StorageBufferClearAndReadWrite,
                                                      "Clears and writes per-slot, per-batch caster counts.");
                          },
                          {"", "PunctualShadowGpuCull", [this](RenderGraphPassContext& context) {
                               recordPunctualShadowCull(context.commandBuffer());
                           }});
    }

    if (board.punctualShadowAtlasImported && resources.punctualShadowSlotCount > 0) {
        passes.punctualShadow =
            graph.addPass("PunctualShadowAtlasPass",
                          RenderPassType::Shadow,
                          RenderPassExecutionType::Graphics,
                          false,
                          [&, punctualCullDeclared](RenderGraphBuilder& builder) {
                              board.punctualShadowAtlasDepth = builder.writeTexture(
                                  board.punctualShadowAtlasDepth,
                                  RGAccess::DepthStencilAttachmentWrite,
                                  "Writes per-slot spot-light depth tiles into the punctual shadow atlas.");
                              if (punctualCullDeclared) {
                                  builder.readBuffer(board.punctualCullIndirectOutput,
                                                     RGAccess::IndirectRead,
                                                     "Replays the culled caster list as per-slot indirect draws.");
                                  builder.readBuffer(board.punctualCullVisibleCounts,
                                                     RGAccess::IndirectRead,
                                                     "Reads per-slot caster counts as the indirect draw count.");
                              }
                          },
                          {"PunctualShadowAtlas", "PunctualShadowAtlas", [this](RenderGraphPassContext& context) {
                               recordPunctualShadowPass(context.commandBuffer(), willRecordPunctualShadowCull());
                           }});
    }

    // Skipped on the CPU culling fallback (renderer.useGpuCulling off) and on an
    // empty draw list -- GpuCulling::recordMainCull returns on both, and this
    // pass was declared unconditionally until the backstop was finally pointed
    // at that configuration and reported it unrecorded on every frame.
    //
    // The main pass keeps reading the indirect command buffer either way, unlike
    // the shadow and punctual culls above whose consumers drop their reads with
    // the pass: the CPU path uploads the identical commands from the host, so
    // the read is real, it just has no producer inside the frame.
    // The fog volume's first-use clear, and the neutral volume fog switching off
    // has to leave: the skybox and the main pass sample it whether or not fog
    // runs. A no-op after the first frame.
    passes.fogVolumeInit = graph.addWorkPass("FogVolumeInit",
                                             RenderPassType::FogVolumeInit,
                                             RenderPassExecutionType::Transfer,
                                             {},
                                             {"", nullptr, [this](RenderGraphPassContext& context) {
                                                  // Runs whether or not fog is on: binding 8 claims a
                                                  // sampled layout unconditionally, so the volume has to
                                                  // reach it even on a cold start with fog disabled. Fog
                                                  // switching off has to leave a neutral volume behind,
                                                  // because the skybox samples it unconditionally, and
                                                  // detecting that here covers every path that can
                                                  // disable fog -- settings load, preset, toggle.
                                                  const bool fogActiveThisFrame = isVolumetricFogActive();
                                                  if (fogWasActive_ && !fogActiveThisFrame) {
                                                      volumetricFog_.markVolumeNeedsClear();
                                                  }
                                                  fogWasActive_ = fogActiveThisFrame;
                                                  volumetricFog_.ensureVolumeInitialized(context.commandBuffer());
                                              }});

    if (resources.mainGpuCullingEnabled) {
        passes.mainGpuCulling = graph.addPass(
            "MainGpuCullingPass",
            RenderPassType::MainGpuCulling,
            RenderPassExecutionType::Compute,
            true,
            [&](RenderGraphBuilder& builder) {
                builder.readBuffer(board.mainCullInput,
                                   RGAccess::StorageBufferRead,
                                   "Reads per-draw AABB, draw-command, and batch metadata.");
                builder.readHistoryTexture(
                    board.depthPyramid,
                    RGAccess::ShaderRead,
                    "Optionally samples the previous-frame Hi-Z depth pyramid for occlusion tests.");
                board.mainCullIndirectOutput = builder.writeBuffer(board.mainCullIndirectOutput,
                                                                   RGAccess::StorageBufferWrite,
                                                                   "Writes indirect draw commands for the main pass.");
                board.mainCullVisibleCounts =
                    builder.writeBuffer(board.mainCullVisibleCounts,
                                        RGAccess::StorageBufferClearAndReadWrite,
                                        "Clears and writes visible counts plus culling debug counters.");
                board.mainCullReadback =
                    builder.writeBuffer(board.mainCullReadback,
                                        RGAccess::TransferDst,
                                        "Receives copied culling counters for frame-latency CPU readback.");
            },
            {"", "MainGpuCullingPass", [this](RenderGraphPassContext& context) {
                 recordGpuCullingCommands(context.commandBuffer());
             }});
    }

    // Clustered light assignment on this queue: rebuild the froxel AABBs, then
    // cull every light into its froxels. The lists are buffers the graph does
    // not track -- the async path submits the same work on the compute queue --
    // so fog and the main pass, which read them, name this pass explicitly.
    if (resources.clusterBuildOnGraphicsQueue) {
        passes.clusterBuild = graph.addWorkPass(
            "ClusterBuild",
            RenderPassType::ClusterBuild,
            RenderPassExecutionType::Compute,
            {passes.mainGpuCulling},
            {"", nullptr, [this](RenderGraphPassContext& context) {
                 const VkCommandBuffer cb = context.commandBuffer();
                 {
                     const renderer::GpuProfileScope buildScope(gpuProfiler_, currentFrame_, cb, "ClusterBuild");
                     rhi::debug::beginLabel(cb, "ClusterBuild");
                     clusteredLighting_.recordClusterBuild(cb, currentFrame_);
                     rhi::debug::endLabel(cb);
                 }
                 {
                     const renderer::GpuProfileScope cullScope(gpuProfiler_, currentFrame_, cb, "LightCull");
                     rhi::debug::beginLabel(cb, "LightCull");
                     clusteredLighting_.recordLightCull(cb, currentFrame_, frameClusterConsumerStages_);
                     rhi::debug::endLabel(cb);
                 }
             }});
    }

    // Declared after the main cull, which is where the renderer records them.
    // They used to be declared before it and recorded after, a disagreement that
    // cost nothing while nothing acted on the declared order and became a
    // reordering the moment the graph started sequencing the frame. Fog in
    // particular has a prerequisite the graph does not model -- injection walks
    // the per-cluster light lists the synchronous cluster build produces -- so
    // its declared position has to agree with where it actually runs.
    //
    // Safe to move: none of these three touch a resource MainGpuCullingPass
    // touches, so no edge, version chain or culling outcome changes with them.
    if (resources.volumetricFogEnabled) {
        passes.volumetricFog = graph.addPass(
            "VolumetricFogPass",
            RenderPassType::VolumetricFog,
            RenderPassExecutionType::Compute,
            // No side effect any more. The volumes it writes are graph
            // resources now and the main pass declares a read on the
            // integrated one, so liveness keeps the pass for the reason it
            // actually runs rather than because it was exempted.
            false,
            [&](RenderGraphBuilder& builder) {
                builder.orderAfter(passes.fogVolumeInit);
                builder.orderAfter(passes.clusterBuild);
                builder.readTexture(board.shadowMapDepth,
                                    RGAccess::ShaderRead,
                                    "Samples the cascaded shadow map to shadow the fog froxels.");
                if (board.punctualShadowAtlasImported) {
                    // Fog runs between the atlas pass and the main pass, so
                    // without this the atlas is still a depth attachment when
                    // the injection dispatch samples it for light shafts.
                    builder.readTexture(board.punctualShadowAtlasDepth,
                                        RGAccess::ShaderRead,
                                        "Samples the punctual shadow atlas for fog light shafts.");
                }
                // One declaration per resource per pass, so the scatter
                // target is read-write: injection writes every froxel and
                // integration marches back through them.
                board.fogScatterWrite =
                    builder.readWriteTexture(board.fogScatterWrite,
                                             RGAccess::StorageImageReadWrite,
                                             "Injects scattering and extinction, then integrates it.");
                // The other half of the ping-pong, holding what the previous
                // frame injected. A history read: reading it before anything
                // wrote it this frame is the intent, not an ordering mistake.
                builder.readHistoryTexture(board.fogScatterRead,
                                           RGAccess::ShaderRead,
                                           "Samples the previous frame's froxels for temporal reprojection.");
                board.fogIntegrated = builder.writeTexture(board.fogIntegrated,
                                                           RGAccess::StorageImageWrite,
                                                           "Writes the front-to-back integrated scattering volume.");
            },
            {"", "VolumetricFog", [this](RenderGraphPassContext& context) {
                 recordVolumetricFogPass(context.commandBuffer());
             }});
    }

    // Ahead of the first pass that traces: the probe gather, then the passes
    // that run the main fragment shader, whose reads are what keep it alive; the
    // previous frame's reads of the same structure are ordered by the barrier
    // RayTracingScene records at the top of the build, because buffer state
    // here is rebuilt every frame and has nothing to order them against. Absent
    // on a frame that keeps the structure an earlier frame built: the reads are
    // still declared, and with no write before them they emit nothing.
    if (board.rayTracingTopLevel.valid() && resources.rayTracingSceneBuildEnabled) {
        passes.rayTracingSceneBuild =
            graph.addPass("RayTracingSceneBuild",
                          RenderPassType::RayTracingSceneBuild,
                          RenderPassExecutionType::Compute,
                          false,
                          [&](RenderGraphBuilder& builder) {
                              board.rayTracingTopLevel = builder.writeBuffer(
                                  board.rayTracingTopLevel,
                                  RGAccess::AccelerationStructureBuild,
                                  "Rebuilds the top-level acceleration structure from this frame's draw items.");
                          },
                          {"RayTracingSceneBuild", "RayTracingSceneBuild", [this](RenderGraphPassContext& context) {
                               rayTracingScene_.recordTopLevelBuild(context.commandBuffer(), currentFrame_);
                           }});
    }

    // The probe shading parameters, refreshed in this frame's own command buffer
    // so one buffer serves every frame in flight. The capture reads them for the
    // multi-bounce lookup and the main pass for shading; neither through a graph
    // resource, so both name this pass.
    passes.probeShadingParams = graph.addWorkPass(
        "ProbeShadingParams",
        RenderPassType::ProbeShadingParams,
        RenderPassExecutionType::Transfer,
        {},
        {"", nullptr, [this](RenderGraphPassContext& context) { recordProbeShadingParams(context.commandBuffer()); }});

    if (resources.probeCaptureEnabled && board.probeCaptureAtlas.valid() && board.probeCaptureDepth.valid()) {
        passes.probeCapture = graph.addPass(
            "ProbeCapture",
            RenderPassType::ProbeCapture,
            RenderPassExecutionType::Graphics,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.orderAfter(passes.probeShadingParams);
                builder.readTexture(board.shadowMapDepth,
                                    RGAccess::ShaderRead,
                                    "Samples the cascaded shadow map so captured radiance is shadowed.");
                if (board.punctualShadowAtlasImported) {
                    // The capture evaluates punctual lights too, so the atlas has
                    // to be out of its depth-attachment layout before this pass
                    // rather than only before the main pass.
                    builder.readTexture(board.punctualShadowAtlasDepth,
                                        RGAccess::ShaderRead,
                                        "Samples the punctual shadow atlas so captured radiance is shadowed.");
                }
                if (board.probeIrradianceAtlas.valid() && board.probeDepthAtlas.valid()) {
                    // Multi-bounce: the capture reads the irradiance the grid
                    // already holds. The update pass writes those same images
                    // later this frame, so what the capture sees is the previous
                    // frame's contents -- which is exactly the feedback wanted,
                    // and why the read has to be declared before that write.
                    builder.readTexture(board.probeIrradianceAtlas,
                                        RGAccess::ShaderRead,
                                        "Reads the previous bounce's irradiance so light can bounce again.");
                    builder.readTexture(board.probeDepthAtlas,
                                        RGAccess::ShaderRead,
                                        "Reads probe visibility to weight the previous bounce.");
                }
                board.probeCaptureAtlas =
                    builder.writeTexture(board.probeCaptureAtlas,
                                         RGAccess::ColorAttachmentWrite,
                                         "Writes radiance and distance for every face of this frame's probes.");
                board.probeCaptureDepth = builder.writeTexture(board.probeCaptureDepth,
                                                               RGAccess::DepthStencilAttachmentWrite,
                                                               "Resolves which surface each capture texel sees.");
            },
            {"ProbeCapture", "ProbeCapture", [this](RenderGraphPassContext& context) {
                 recordProbeCapturePass(context.commandBuffer());
             }});
    }

    // The ray-traced gather, on the frames that trace the batch instead of
    // rasterising it: the same capture atlas, filled by one ray per texel, for
    // the same update pass to convolve. It needs no cascades -- its sun shadow is
    // a ray -- which is what lets the cascades go entirely on a frame that also
    // traces the camera's shadows.
    if (resources.probeTraceEnabled && board.probeCaptureAtlas.valid() && board.rayTracingTopLevel.valid()) {
        passes.probeTrace = graph.addPass(
            "ProbeTrace",
            RenderPassType::ProbeTrace,
            RenderPassExecutionType::Compute,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.orderAfter(passes.probeShadingParams);
                builder.readBuffer(board.rayTracingTopLevel,
                                   RGAccess::AccelerationStructureComputeRead,
                                   "Traces each capture texel's ray, and its hit's sun ray, against the scene.");
                if (board.punctualShadowAtlasImported) {
                    builder.readTexture(board.punctualShadowAtlasDepth,
                                        RGAccess::ShaderRead,
                                        "Samples the punctual shadow atlas so traced radiance is shadowed.");
                }
                if (board.probeIrradianceAtlas.valid() && board.probeDepthAtlas.valid()) {
                    // Multi-bounce, as for the raster capture: last frame's grid,
                    // read before the update pass below writes it.
                    builder.readTexture(board.probeIrradianceAtlas,
                                        RGAccess::ShaderRead,
                                        "Reads the previous bounce's irradiance so light can bounce again.");
                    builder.readTexture(board.probeDepthAtlas,
                                        RGAccess::ShaderRead,
                                        "Reads probe visibility to weight the previous bounce.");
                }
                board.probeCaptureAtlas =
                    builder.writeTexture(board.probeCaptureAtlas,
                                         RGAccess::StorageImageWrite,
                                         "Writes radiance and distance for every face of this frame's probes.");
            },
            {"ProbeTrace", "ProbeTrace", [this](RenderGraphPassContext& context) {
                 recordProbeTracePass(context.commandBuffer());
             }});
    }

    if (resources.irradianceProbeUpdateEnabled && board.probeIrradianceAtlas.valid() && board.probeDepthAtlas.valid()) {
        passes.irradianceProbes = graph.addPass(
            "IrradianceProbeUpdate",
            RenderPassType::IrradianceProbes,
            RenderPassExecutionType::Compute,
            // Side effect, even though both atlases are graph resources with a
            // declared reader. The atlases persist across frames by design, so
            // the pass's real consumer is the *next* frame's update, which
            // liveness analysis cannot see. Without this the pass survives only
            // because the main pass declares a read it does not yet use, and the
            // day that read is conditioned or removed the pass is culled while
            // the renderer still tries to record it -- which throws rather than
            // degrades.
            true,
            [&](RenderGraphBuilder& builder) {
                builder.orderAfter(passes.probeShadingParams);
                // Read-write, not write: the border dispatch copies core texels
                // the fill dispatch produced, and later phases blend new radiance
                // against what the atlas already holds.
                board.probeIrradianceAtlas =
                    builder.readWriteTexture(board.probeIrradianceAtlas,
                                             RGAccess::StorageImageReadWrite,
                                             "Writes probe irradiance tiles and wraps their octahedral border.");
                board.probeDepthAtlas =
                    builder.readWriteTexture(board.probeDepthAtlas,
                                             RGAccess::StorageImageReadWrite,
                                             "Writes probe visibility tiles and wraps their octahedral border.");
                if ((resources.probeCaptureEnabled || resources.probeTraceEnabled) && board.probeCaptureAtlas.valid()) {
                    // Moves the capture atlas out of the layout whichever gather
                    // above left it in -- colour attachment or storage write --
                    // and into the one the convolution reads it as.
                    builder.readTexture(board.probeCaptureAtlas,
                                        RGAccess::StorageImageRead,
                                        "Reads this frame's captured cube faces to convolve into probe tiles.");
                }
            },
            {"", "IrradianceProbeUpdate", [this](RenderGraphPassContext& context) {
                 irradianceProbes_.recordUpdate(context.commandBuffer(), giSettings_.debugPattern);
             }});
    }
    if (resources.depthPrepassEnabled) {
        passes.depthPrepass = graph.addPass(
            "DepthPrepass",
            RenderPassType::DepthPrepass,
            RenderPassExecutionType::Graphics,
            // sideEffect, so liveness cannot drop it. Its only product
            // is depth, and the pass after it LOADS that depth rather
            // than reading it as a texture -- an attachment hand-off the
            // graph does not model as a dependency. Declared without
            // this the graph culls the pass as dead (its write is
            // followed by another write and never read), and the first
            // build did exactly that: the renderer then tried to record
            // a culled pass and threw.
            //
            // Worth noting what the failure would have been had it not
            // thrown: the main pass would test LESS_OR_EQUAL against
            // cleared depth, draw the scene correctly, and measure
            // nothing.
            true,
            [&](RenderGraphBuilder& builder) {
                board.mainDepth = builder.writeTexture(board.mainDepth,
                                                       RGAccess::DepthStencilAttachmentWrite,
                                                       "Clears and writes opaque depth ahead of the main pass.");
                builder.readBuffer(board.mainCullIndirectOutput,
                                   RGAccess::IndirectRead,
                                   "Replays the main pass's indirect draw commands, depth only.");
                builder.readBuffer(board.mainCullVisibleCounts,
                                   RGAccess::IndirectRead,
                                   "Reads per-batch visible counts when indirect-count drawing is active.");
            },
            {"DepthPrepass", "DepthPrepass", [this](RenderGraphPassContext& context) {
                 recordDepthPrepass(context.commandBuffer());
             }});
    }

    // After the prepass whose depth they trace from and the build whose structure
    // they trace against, before the main pass that reads what they write. The
    // renderer records both on exactly this condition; the images and the TLAS
    // handles are valid whenever the flag is set, since the flag implies a frame
    // that traces on a device whose mask is available.
    if (resources.rayTracedShadowMaskEnabled && board.rayTracedShadowMask.valid() &&
        board.rayTracedShadowTrace.valid() && board.rayTracingTopLevel.valid()) {
        passes.rayTracedShadowTrace = graph.addPass(
            "RayTracedShadowTrace",
            RenderPassType::RayTracedShadowTrace,
            RenderPassExecutionType::Compute,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(board.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Reconstructs each traced pixel's surface from the depth the prepass wrote.");
                builder.readBuffer(board.rayTracingTopLevel,
                                   RGAccess::AccelerationStructureComputeRead,
                                   "Traces sun shadow rays against the top-level acceleration structure.");
                board.rayTracedShadowTrace =
                    builder.writeTexture(board.rayTracedShadowTrace,
                                         RGAccess::StorageImageWrite,
                                         "Writes each trace texel's visibility, hit distance and camera distance.");
            },
            {"RayTracedShadowTrace", "RayTracedShadowTrace", [this](RenderGraphPassContext& context) {
                 recordRayTracedShadowTrace(context.commandBuffer());
             }});

        // The soft shadow filter, between the two at the trace's resolution. Two
        // images serve three passes by ping-pong: the radius pass reads the trace
        // image into the filter image, the horizontal axis reads that back into
        // the trace image, and the vertical axis writes the filter image again,
        // which is what the resolve then reads.
        const bool filtered = resources.rayTracedShadowFilterEnabled && board.rayTracedShadowFilter.valid();
        // Temporal accumulation, ahead of the filter whose radius pass reads it.
        if (filtered && resources.rayTracedShadowTemporalEnabled && board.rayTracedShadowHistoryRead.valid() &&
            board.rayTracedShadowHistoryWrite.valid()) {
            passes.rayTracedShadowTemporal = graph.addPass(
                "RayTracedShadowTemporal",
                RenderPassType::RayTracedShadowTemporal,
                RenderPassExecutionType::Compute,
                false,
                [&](RenderGraphBuilder& builder) {
                    builder.readTexture(
                        board.mainDepth, RGAccess::ShaderRead, "Reprojects each trace texel through its depth.");
                    builder.readTexture(
                        board.rayTracedShadowTrace, RGAccess::ShaderRead, "Reads this frame's ray per texel.");
                    builder.readTexture(board.rayTracedShadowHistoryRead,
                                        RGAccess::ShaderRead,
                                        "Reads last frame's history at the reprojected position.");
                    board.rayTracedShadowHistoryWrite =
                        builder.writeTexture(board.rayTracedShadowHistoryWrite,
                                             RGAccess::StorageImageWrite,
                                             "Writes each texel's running mean, frame count and view depth.");
                },
                {"RayTracedShadowTemporal", "RayTracedShadowTemporal", [this](RenderGraphPassContext& context) {
                     recordRayTracedShadowTemporal(context.commandBuffer());
                 }});
        }
        if (filtered) {
            static_assert(kRayTracedShadowFilterStageCount == 3);
            struct FilterPassDeclaration {
                const char* name;
                RayTracedShadowFilterStage stage;
                const char* readPurpose;
                const char* writePurpose;
            };
            static constexpr std::array<FilterPassDeclaration, kRayTracedShadowFilterStageCount> kFilterPasses{{
                {"RayTracedShadowRadius",
                 RayTracedShadowFilterStage::Radius,
                 "Reads the hit distances around each trace texel.",
                 "Writes each trace texel with the penumbra radius its blur takes."},
                {"RayTracedShadowBlurH",
                 RayTracedShadowFilterStage::BlurHorizontal,
                 "Reads each texel's row neighbours and radius.",
                 "Writes the visibility blurred along the row."},
                {"RayTracedShadowBlurV",
                 RayTracedShadowFilterStage::BlurVertical,
                 "Reads each texel's column neighbours and radius.",
                 "Writes the visibility blurred along both axes, which the resolve reads."},
            }};
            // One profiler scope over the three dispatches: the filter's cost is
            // the question, and each scope is two more timestamps in a pass this
            // small.
            graph.pushScope("RayTracedShadowFilter", "RayTracedShadowFilter");
            for (const FilterPassDeclaration& declaration : kFilterPasses) {
                const bool toFilterImage = declaration.stage != RayTracedShadowFilterStage::BlurHorizontal;
                passes.rayTracedShadowFilter.at(static_cast<size_t>(declaration.stage)) =
                    graph.addPass(declaration.name,
                                  RenderPassType::RayTracedShadowFilter,
                                  RenderPassExecutionType::Compute,
                                  false,
                                  [&, toFilterImage, declaration](RenderGraphBuilder& builder) {
                                      RGTextureHandle& source =
                                          toFilterImage ? board.rayTracedShadowTrace : board.rayTracedShadowFilter;
                                      RGTextureHandle& destination =
                                          toFilterImage ? board.rayTracedShadowFilter : board.rayTracedShadowTrace;
                                      builder.readTexture(source, RGAccess::ShaderRead, declaration.readPurpose);
                                      // The radius pass's set samples this frame's
                                      // history whether or not frames accumulate.
                                      if (declaration.stage == RayTracedShadowFilterStage::Radius &&
                                          board.rayTracedShadowHistoryWrite.valid()) {
                                          builder.readTexture(board.rayTracedShadowHistoryWrite,
                                                              RGAccess::ShaderRead,
                                                              "Reads each texel's accumulated visibility.");
                                      }
                                      destination = builder.writeTexture(
                                          destination, RGAccess::StorageImageWrite, declaration.writePurpose);
                                  },
                                  {"", nullptr, [this, stage = declaration.stage](RenderGraphPassContext& context) {
                                       recordRayTracedShadowFilter(context.commandBuffer(), stage);
                                   }});
            }
            graph.popScope();
        }

        passes.rayTracedShadowResolve = graph.addPass(
            "RayTracedShadowResolve",
            RenderPassType::RayTracedShadowResolve,
            RenderPassExecutionType::Compute,
            false,
            [&, filtered](RenderGraphBuilder& builder) {
                builder.readTexture(board.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Matches each full-resolution pixel to the trace texels about its surface.");
                builder.readTexture(filtered ? board.rayTracedShadowFilter : board.rayTracedShadowTrace,
                                    RGAccess::ShaderRead,
                                    "Reads the trace texels around each pixel, filtered or as traced.");
                board.rayTracedShadowMask =
                    builder.writeTexture(board.rayTracedShadowMask,
                                         RGAccess::StorageImageWrite,
                                         "Writes each pixel's sun visibility and the depth it is about.");
            },
            {"RayTracedShadowResolve", "RayTracedShadowResolve", [this](RenderGraphPassContext& context) {
                 recordRayTracedShadowResolve(context.commandBuffer());
             }});
    }

    passes.mainHdr = graph.addPass(
        "MainHDRPass",
        RenderPassType::MainHdr,
        RenderPassExecutionType::Graphics,
        false,
        [&](RenderGraphBuilder& builder) {
            // What it samples or walks that the graph does not track: the cluster
            // lists, the probe shading parameters, the fog volume's first-use
            // clear, and the idle VSM pool's layout.
            builder.orderAfter(passes.clusterBuild);
            builder.orderAfter(passes.probeShadingParams);
            builder.orderAfter(passes.fogVolumeInit);
            builder.orderAfter(passes.vsmPagePoolIdle);
            builder.readTexture(
                board.shadowMapDepth, RGAccess::ShaderRead, "Samples the cascaded shadow-map array for lighting.");
            declareRayTracedShadowRead(builder, board, resources);
            declareRayTracedShadowMaskRead(builder, board);
            if (board.punctualShadowAtlasImported) {
                // Declared unconditionally (not just when slots exist) so the
                // atlas always reaches the layout the material descriptors
                // record, even on frames where nothing cast.
                builder.readTexture(board.punctualShadowAtlasDepth,
                                    RGAccess::ShaderRead,
                                    "Samples the punctual shadow atlas for spot-light visibility.");
            }
            if (board.vsmPagePoolImported) {
                // Unconditional for the same reason as the atlas above: the pool
                // has to reach the layout its sampler claims even on a frame
                // that redrew no page at all.
                builder.readTexture(board.vsmPagePoolDepth,
                                    RGAccess::ShaderRead,
                                    "Samples the virtual shadow page pool for directional visibility.");
            }
            if (board.probeIrradianceAtlas.valid() && board.probeDepthAtlas.valid()) {
                // Same asymmetry as the punctual atlas above, for the same
                // reason: declared whenever the atlases exist rather than only
                // when probes updated, so they always reach a sampled layout.
                // Nothing samples them until the shading phase lands; the
                // declaration is what keeps them out of UNDEFINED until then.
                builder.readTexture(
                    board.probeIrradianceAtlas, RGAccess::ShaderRead, "Samples probe irradiance for indirect diffuse.");
                builder.readTexture(board.probeDepthAtlas,
                                    RGAccess::ShaderRead,
                                    "Samples probe visibility to reject leaked probe contributions.");
            }
            if (board.ambientOcclusion.valid()) {
                // Read before the GTAO pass writes it later this frame, so what
                // lands here is the previous frame's occlusion, reprojected per
                // fragment in the shader. Declared whenever the target exists,
                // for the same reason as the atlases above: the declaration is
                // what keeps it in a sampled layout rather than UNDEFINED, and
                // the shader gates the fetch on a push constant anyway.
                builder.readHistoryTexture(board.ambientOcclusion,
                                           RGAccess::ShaderRead,
                                           "Samples the previous frame's ambient occlusion for the ambient term.");
            }
            if (board.fogIntegrated.valid()) {
                // Binding 8 of the material set is a sampler3D bound to this
                // volume on every frame the subsystem allocated one, so the image
                // has to hold the layout that descriptor claims whether or not
                // fog ran. With fog off the shader's max-distance gate is zero
                // and the sample never happens, which is exactly a layout-only
                // read -- and declaring it that way keeps no producer alive, so
                // it cannot resurrect the fog pass on a frame that skipped it.
                if (resources.volumetricFogEnabled) {
                    builder.readTexture(board.fogIntegrated,
                                        RGAccess::ShaderRead,
                                        "Samples the integrated froxel volume for in-scattering.");
                } else {
                    builder.readTextureForLayout(board.fogIntegrated,
                                                 RGAccess::ShaderRead,
                                                 "Bound as the fog volume while fog is off, so the sample is gated "
                                                 "out and only the layout matters.");
                }
            }
            board.sceneColor = builder.writeTexture(
                board.sceneColor, RGAccess::ColorAttachmentWrite, "Writes linear HDR skybox and mesh lighting.");
            board.velocity = builder.writeTexture(board.velocity,
                                                  RGAccess::ColorAttachmentWrite,
                                                  "Writes UV-space motion vectors for TAA history reprojection.");
            board.normalRoughness =
                builder.writeTexture(board.normalRoughness,
                                     RGAccess::ColorAttachmentWrite,
                                     "Writes the thin G-buffer (normal, roughness, metallic) for SSR.");
            board.mainDepth = builder.writeTexture(board.mainDepth,
                                                   RGAccess::DepthStencilAttachmentWrite,
                                                   resources.depthPrepassEnabled
                                                       ? "Loads the prepass depth and writes the main depth attachment."
                                                       : "Clears and writes the main depth attachment.");
            builder.readBuffer(board.mainCullIndirectOutput,
                               RGAccess::IndirectRead,
                               "Reads CPU- or GPU-generated indirect draw commands.");
            builder.readBuffer(board.mainCullVisibleCounts,
                               RGAccess::IndirectRead,
                               "Reads per-batch visible counts when indirect-count drawing is active.");
        },
        {"MainHDRPass", "MainHDRPass", [this, &mainPassState](RenderGraphPassContext& context) {
             recordMainHdrPass(context.commandBuffer(), mainPassState);
         }});

    if (resources.twoPhaseOcclusionEnabled) {
        passes.depthPyramidMid = graph.addPass(
            "DepthPyramidMidPass",
            RenderPassType::DepthPyramid,
            RenderPassExecutionType::Compute,
            true,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(board.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Samples phase-1 main depth for the mid-frame Hi-Z rebuild.");
                board.depthPyramid =
                    builder.writeTexture(board.depthPyramid,
                                         RGAccess::StorageImageWrite,
                                         "Rebuilds the Hi-Z pyramid so phase 2 can re-test occlusion candidates.");
            },
            {"", "DepthPyramidMid", [this](RenderGraphPassContext& context) {
                 rhi::debug::beginLabel(context.commandBuffer(), "DepthPyramid");
                 depthPyramid_.recordBuild(context.commandBuffer(), frameViewProjection_, frameCameraPosition_);
                 rhi::debug::endLabel(context.commandBuffer());
             }});

        passes.mainGpuCullingPhase2 =
            graph.addPass("MainGpuCullingPhase2",
                          RenderPassType::MainGpuCulling,
                          RenderPassExecutionType::Compute,
                          true,
                          [&](RenderGraphBuilder& builder) {
                              builder.readBuffer(board.mainCullInput,
                                                 RGAccess::StorageBufferRead,
                                                 "Re-reads per-draw AABBs for the phase-1 occlusion candidates.");
                              builder.readTexture(board.depthPyramid,
                                                  RGAccess::ShaderRead,
                                                  "Samples the mid-frame Hi-Z pyramid for the candidate re-test.");
                              board.mainCullIndirectOutput =
                                  builder.writeBuffer(board.mainCullIndirectOutput,
                                                      RGAccess::StorageBufferWrite,
                                                      "Writes indirect draw commands for rescued (disoccluded) draws.");
                              board.mainCullVisibleCounts =
                                  builder.writeBuffer(board.mainCullVisibleCounts,
                                                      RGAccess::StorageBufferClearAndReadWrite,
                                                      "Resets per-batch counts and appends the rescued stats counter.");
                              board.mainCullReadback = builder.writeBuffer(
                                  board.mainCullReadback,
                                  RGAccess::TransferDst,
                                  "Receives the combined two-phase culling counters for CPU readback.");
                          },
                          {"", "MainGpuCullingPhase2", [this](RenderGraphPassContext& context) {
                               gpuCulling_.recordMainCullPhase2(context.commandBuffer(),
                                                                currentFrame_,
                                                                isGpuCullingActive(),
                                                                static_cast<uint32_t>(allDrawItems_.size()),
                                                                frameFrustumPlanes_,
                                                                isMainPassMultiDrawIndirectActive());
                           }});

        passes.mainHdrPhase2 = graph.addPass(
            "MainHDRPhase2",
            RenderPassType::MainHdr,
            RenderPassExecutionType::Graphics,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(
                    board.shadowMapDepth, RGAccess::ShaderRead, "Samples the cascaded shadow-map array for lighting.");
                declareRayTracedShadowRead(builder, board, resources);
                declareRayTracedShadowMaskRead(builder, board);
                if (board.ambientOcclusion.valid()) {
                    // A history read for the same reason phase 1's is: GTAO has
                    // not run yet at this point in the frame.
                    builder.readHistoryTexture(
                        board.ambientOcclusion,
                        RGAccess::ShaderRead,
                        "Samples ambient occlusion for the ambient term, as the main pass does.");
                }
                board.sceneColor =
                    builder.writeTexture(board.sceneColor,
                                         RGAccess::ColorAttachmentWrite,
                                         "Draws rescued disoccluded objects into the existing HDR color.");
                board.velocity = builder.writeTexture(
                    board.velocity, RGAccess::ColorAttachmentWrite, "Appends motion vectors for the rescued draws.");
                board.normalRoughness = builder.writeTexture(board.normalRoughness,
                                                             RGAccess::ColorAttachmentWrite,
                                                             "Appends thin G-buffer data for the rescued draws.");
                board.mainDepth = builder.writeTexture(board.mainDepth,
                                                       RGAccess::DepthStencilAttachmentWrite,
                                                       "Loads and extends phase-1 depth with the rescued draws.");
                builder.readBuffer(board.mainCullIndirectOutput,
                                   RGAccess::IndirectRead,
                                   "Reads the phase-2 compacted indirect draw commands.");
                builder.readBuffer(
                    board.mainCullVisibleCounts, RGAccess::IndirectRead, "Reads the phase-2 per-batch visible counts.");
            },
            {"MainHDRPhase2", "MainHDRPhase2", [this, &mainPassState](RenderGraphPassContext& context) {
                 recordMainHdrPhase2Pass(context.commandBuffer(), mainPassState);
             }});
    }

    // The ray-traced reference for probe GI: a measuring instrument that changes no
    // pixel, so nothing reads what it writes in the frame -- a side effect, kept
    // alive by that rather than by a consumer. After the main depth and the thin
    // G-buffer are final, against this frame's probe atlases.
    if (resources.probeReferenceEnabled && board.probeReferenceAccumulation.valid() &&
        board.probeReferenceEstimate.valid() && board.rayTracingTopLevel.valid() &&
        board.probeIrradianceAtlas.valid() && board.probeDepthAtlas.valid()) {
        passes.probeReference = graph.addPass(
            "ProbeReference",
            RenderPassType::ProbeReference,
            RenderPassExecutionType::Compute,
            true,
            [&](RenderGraphBuilder& builder) {
                builder.orderAfter(passes.probeShadingParams);
                builder.readTexture(board.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Reconstructs each visible surface point from the final main depth.");
                builder.readTexture(board.normalRoughness,
                                    RGAccess::ShaderRead,
                                    "Reads each visible surface's shading normal from the thin G-buffer.");
                builder.readBuffer(board.rayTracingTopLevel,
                                   RGAccess::AccelerationStructureComputeRead,
                                   "Traces the reference's hemisphere rays against the scene.");
                if (board.punctualShadowAtlasImported) {
                    builder.readTexture(board.punctualShadowAtlasDepth,
                                        RGAccess::ShaderRead,
                                        "Samples the punctual shadow atlas so traced radiance is shadowed.");
                }
                builder.readTexture(board.probeIrradianceAtlas,
                                    RGAccess::ShaderRead,
                                    "Looks up the probes' irradiance at each pixel, to compare with the reference.");
                builder.readTexture(
                    board.probeDepthAtlas, RGAccess::ShaderRead, "Reads probe visibility for that lookup.");
                board.probeReferenceAccumulation =
                    builder.writeTexture(board.probeReferenceAccumulation,
                                         RGAccess::StorageImageReadWrite,
                                         "Adds this frame's traced radiance to the running sum.");
                board.probeReferenceEstimate = builder.writeTexture(board.probeReferenceEstimate,
                                                                    RGAccess::StorageImageWrite,
                                                                    "Writes the probes' irradiance per pixel.");
            },
            {"ProbeReference", "ProbeReference", [this](RenderGraphPassContext& context) {
                 recordProbeReferencePass(context.commandBuffer());
             }});
    }

    if (resources.ssrEnabled && board.ssrSceneColorCopy.valid()) {
        passes.ssrCopy = graph.addPass(
            "SSRCopyPass",
            RenderPassType::Ssr,
            RenderPassExecutionType::Transfer,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(board.sceneColor,
                                    RGAccess::TransferSrc,
                                    "Copies the lit opaque scene color as the SSR reflection source.");
                board.ssrSceneColorCopy = builder.writeTexture(
                    board.ssrSceneColorCopy, RGAccess::TransferDst, "Receives the scene-color copy the trace samples.");
            },
            {"SSRCopyPass", "SSRCopy", [this](RenderGraphPassContext& context) {
                 ssr_.recordCopy(context.commandBuffer(),
                                 currentFrame_,
                                 postProcess_.sceneColor().image(),
                                 renderResolution_.extent());
             }});

        passes.ssrTrace = graph.addPass(
            "SSRTracePass",
            RenderPassType::Ssr,
            RenderPassExecutionType::Graphics,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(board.ssrSceneColorCopy,
                                    RGAccess::ShaderRead,
                                    "Samples the pre-reflection scene color at ray hit points.");
                builder.readTexture(board.normalRoughness,
                                    RGAccess::ShaderRead,
                                    "Reads surface normal/roughness/metallic for ray setup and weighting.");
                builder.readTexture(
                    board.mainDepth, RGAccess::ShaderRead, "Marches rays against the main depth buffer.");
                // Read-modify-write, like the transparent pass: an additive blend
                // reads the destination. Declared write-only, the culler is free to
                // treat the main pass's write to scene colour as dead -- harmless
                // only for as long as nothing else writes scene colour in between.
                // The transparent pass hit exactly this and the note in
                // docs/transparency.md flagged this one as the same latent gap.
                board.sceneColor =
                    builder.readWriteTexture(board.sceneColor,
                                             RGAccess::ColorAttachmentWrite,
                                             "Blends the reflection correction into scene color; additive, and the "
                                             "correction is signed, so it reads what the main pass already wrote.");
                // The hybrid variant binds the global material set, so it reads
                // what the main pass reads from it: the TLAS for its rays, and
                // the atlases its hit shading samples or statically names.
                if (resources.ssrRayTracedEnabled && board.rayTracingTopLevel.valid()) {
                    builder.readBuffer(board.rayTracingTopLevel,
                                       RGAccess::AccelerationStructureRead,
                                       "Traces a reflection ray where the screen-space march misses.");
                    if (board.punctualShadowAtlasImported) {
                        builder.readTexture(board.punctualShadowAtlasDepth,
                                            RGAccess::ShaderRead,
                                            "Shadows the punctual lights at a traced reflection hit.");
                    }
                    if (board.probeIrradianceAtlas.valid() && board.probeDepthAtlas.valid()) {
                        builder.readTexture(board.probeIrradianceAtlas,
                                            RGAccess::ShaderRead,
                                            "Named by the shared hit shading; kept in its sampled layout.");
                        builder.readTexture(board.probeDepthAtlas,
                                            RGAccess::ShaderRead,
                                            "Named by the shared hit shading; kept in its sampled layout.");
                    }
                }
            },
            {"SSRTracePass",
             "SSRTrace",
             [this, hybrid = resources.ssrRayTracedEnabled](RenderGraphPassContext& context) {
                 if (hybrid) {
                     if (!ssrHybridLogged_) {
                         ssrHybridLogged_ = true;
                         Logger::info("Screen-space reflections: hybrid trace recorded (ray query where the march "
                                      "misses, max distance " +
                                      std::to_string(ssrSettings_.rayTracedMaxDistance) + ").");
                     }
                     const renderer::SsrRayQueryBinding binding = ssrRayQueryBinding();
                     ssr_.recordTrace(context.commandBuffer(), currentFrame_, renderResolution_.extent(), &binding);
                 } else {
                     ssr_.recordTrace(context.commandBuffer(), currentFrame_, renderResolution_.extent());
                 }
             }});
    }

    if (resources.gtaoEnabled && board.ambientOcclusion.valid() && board.ambientOcclusionRaw.valid()) {
        passes.gtao = graph.addPass(
            "GTAOPass",
            RenderPassType::Gtao,
            RenderPassExecutionType::Graphics,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(board.mainDepth,
                                    RGAccess::ShaderRead,
                                    "Reconstructs view positions from the main depth buffer for the horizon search.");
                builder.readTexture(board.normalRoughness,
                                    RGAccess::ShaderRead,
                                    "Reads the surface normal for GTAO slice integration.");
                board.ambientOcclusionRaw = builder.writeTexture(board.ambientOcclusionRaw,
                                                                 RGAccess::ColorAttachmentWrite,
                                                                 "Writes the raw (pre-denoise) GTAO visibility term.");
            },
            {"GTAOPass", "GTAO", [this](RenderGraphPassContext& context) {
                 gtao_.recordTrace(context.commandBuffer(), currentFrame_, renderResolution_.extent());
             }});

        passes.gtaoBlur =
            graph.addPass("GTAOBlurPass",
                          RenderPassType::GtaoBlur,
                          RenderPassExecutionType::Graphics,
                          false,
                          [&](RenderGraphBuilder& builder) {
                              builder.readTexture(board.ambientOcclusionRaw,
                                                  RGAccess::ShaderRead,
                                                  "Samples the raw GTAO term for depth-aware bilateral denoising.");
                              builder.readTexture(board.mainDepth,
                                                  RGAccess::ShaderRead,
                                                  "Samples main depth for the bilateral blur's edge-stopping weights.");
                              board.ambientOcclusion = builder.writeTexture(
                                  board.ambientOcclusion,
                                  RGAccess::ColorAttachmentWrite,
                                  "Writes the denoised GTAO visibility term the composite multiplies in.");
                          },
                          {"GTAOBlurPass", "GTAOBlur", [this](RenderGraphPassContext& context) {
                               gtao_.recordBlur(context.commandBuffer(), currentFrame_, renderResolution_.extent());
                           }});
    }

    // Transparents come after SSR and GTAO, which both need an opaque-only depth
    // buffer and G-buffer, and before the TAA resolve so blended edges still get
    // antialiased. They read depth without writing it, so the depth pyramid built
    // below is unaffected.
    if (resources.transparentDrawCount > 0) {
        passes.transparent = graph.addPass(
            "TransparentPass",
            RenderPassType::Transparent,
            RenderPassExecutionType::Graphics,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(board.mainDepth,
                                    RGAccess::DepthStencilAttachmentWrite,
                                    "Depth-tests blended geometry against the opaque depth buffer (no writes).");
                // The shared fragment shader traces for blended surfaces too:
                // glass receives the sun shadow, it just does not cast one.
                declareRayTracedShadowRead(builder, board, resources);
                declareRayTracedShadowMaskRead(builder, board);
                if (board.ambientOcclusion.valid()) {
                    // Same shared-shader reason as normalRoughness below: this
                    // pass reuses the main fragment shader, so it samples the
                    // occlusion binding whether or not blended surfaces need it.
                    // GTAO already ran and left the image a colour attachment,
                    // so without this declaration it would still be in that
                    // layout when the descriptor is read.
                    // Which read this is depends on whether GTAO ran: with it on,
                    // the blur wrote this image earlier in the frame and this is
                    // an ordinary dependency; with it off, nothing produced the
                    // image this frame and the shader gates the fetch away.
                    // Declaring the difference is what keeps the GTAO-off frame
                    // from reading as an ordering mistake.
                    const char* transparentAoDescription =
                        "Samples ambient occlusion through the shared main-pass fragment shader.";
                    if (resources.gtaoEnabled) {
                        builder.readTexture(board.ambientOcclusion, RGAccess::ShaderRead, transparentAoDescription);
                    } else {
                        builder.readHistoryTexture(
                            board.ambientOcclusion, RGAccess::ShaderRead, transparentAoDescription);
                    }
                }
                // Read-modify-write, not a plain write: the "over" blend reads the
                // destination. Declaring it write-only makes the pass culler treat
                // every earlier write to scene color -- the main pass, and SSR's
                // additive blend -- as dead, and cull them.
                board.sceneColor =
                    builder.readWriteTexture(board.sceneColor,
                                             RGAccess::ColorAttachmentWrite,
                                             "Alpha-blends sorted transparent geometry over the lit opaque "
                                             "scene color.");
                board.velocity =
                    builder.readWriteTexture(board.velocity,
                                             RGAccess::ColorAttachmentWrite,
                                             "Overwrites motion vectors for blended pixels so TAA reprojects them "
                                             "with their own motion; loaded, so opaque velocity survives.");
                board.normalRoughness =
                    builder.readWriteTexture(board.normalRoughness,
                                             RGAccess::ColorAttachmentWrite,
                                             "Written because the shared fragment shader emits it; SSR and GTAO, its "
                                             "only readers, already ran earlier this frame.");
            },
            {"Transparent " +
                 std::to_string(frameVisibleBucketRanges_[static_cast<size_t>(RenderBucket::Blend)].count()) + " draws",
             "Transparent",
             [this, &mainPassState](RenderGraphPassContext& context) {
                 recordTransparentPass(context.commandBuffer(), mainPassState);
             }});
    }

    // Declared without a side effect, which it used to need. Nothing this frame
    // reads what this pass writes -- the phase-2 re-test reads the mid-frame build
    // -- and the next frame's page marking and main cull declare history reads on
    // it. cullUnusedPasses keeps the last writer of a history-read resource alive,
    // so the declarations are what save this pass from culling, not a hand-set
    // flag.
    //
    // Whether it is declared at all is a different question, and it does need
    // the flag: the recorder skips the build outright when nothing consumes the
    // pyramid, so on those frames this used to declare a pass that was never
    // recorded. The next frame's readers say readHistoryTexture, which means
    // "the previous frame's contents" and needs no producer this frame, so not
    // declaring it is honest rather than merely quieter.
    if (resources.depthPyramidBuildEnabled) {
        passes.depthPyramid = graph.addPass(
            "DepthPyramidPass",
            RenderPassType::DepthPyramid,
            RenderPassExecutionType::Compute,
            false,
            [&](RenderGraphBuilder& builder) {
                builder.readTexture(
                    board.mainDepth, RGAccess::ShaderRead, "Samples the completed normal-Z main depth buffer.");
                board.depthPyramid =
                    builder.writeTexture(board.depthPyramid,
                                         RGAccess::StorageImageWrite,
                                         "Writes the max-depth Hi-Z pyramid for later-frame occlusion culling.");
            },
            {"DepthPyramid", "DepthPyramid", [this](RenderGraphPassContext& context) {
                 depthPyramid_.recordBuild(context.commandBuffer(), frameViewProjection_, frameCameraPosition_);
             }});
    }

    // TAA, both bloom chains, exposure and the composite are the post-process
    // stack's, declared beside its recorders.
    passes.composite = postProcess_.declareGraphPasses(frameJitteredProjection_,
                                                       probeDebugGain,
                                                       renderScaleSettings_.sharpness,
                                                       showSharpenDelta_ ? sharpenDeltaGain_ : 0.0f);

    // The portfolio screenshot, and the regression capture unless it asked for
    // the overlay: copies of the swapchain between the composite and the
    // overlay. recordCopy reads the swapchain's tracked layout on entry and
    // leaves it where it found it, so the copies need no graph declaration.
    passes.frameCapture = graph.addWorkPass("FrameCapture",
                                            RenderPassType::FrameCapture,
                                            RenderPassExecutionType::Transfer,
                                            {passes.composite},
                                            {"", nullptr, [this, imageIndex](RenderGraphPassContext& context) {
                                                 recordPortfolioScreenshotCopy(context.commandBuffer(), imageIndex);
                                                 if (!frameCaptureIncludesUi_) {
                                                     recordFrameCaptureCopy(context.commandBuffer(), imageIndex);
                                                 }
                                             }});

    passes.imgui = graph.addPass("ImGuiPass",
                                 RenderPassType::ImGui,
                                 RenderPassExecutionType::Graphics,
                                 true,
                                 [&](RenderGraphBuilder& builder) {
                                     builder.orderAfter(passes.frameCapture);
                                     board.swapchainColor = builder.readWriteTexture(
                                         board.swapchainColor,
                                         RGAccess::ColorAttachmentWrite,
                                         "Loads the composited swapchain image and draws the debug overlay.");
                                     builder.colorAttachment(board.swapchainColor, VK_ATTACHMENT_LOAD_OP_LOAD);
                                     // The last pass that draws into the swapchain, so the
                                     // present transition is recorded behind its body.
                                     builder.transitionAfterPass(board.swapchainColor, RGAccess::Present);
                                 },
                                 {"ImGuiPass", "ImGuiPass", [this](RenderGraphPassContext& context) {
                                      imguiLayer_.render(context.commandBuffer());
                                  }});

    // The regression capture taken behind the overlay and its present transition.
    passes.frameCaptureAfterOverlay =
        graph.addWorkPass("FrameCaptureAfterOverlay",
                          RenderPassType::FrameCapture,
                          RenderPassExecutionType::Transfer,
                          {passes.imgui},
                          {"", nullptr, [this, imageIndex](RenderGraphPassContext& context) {
                               if (frameCaptureIncludesUi_) {
                                   recordFrameCaptureCopy(context.commandBuffer(), imageIndex);
                               }
                           }});
}

} // namespace ve
