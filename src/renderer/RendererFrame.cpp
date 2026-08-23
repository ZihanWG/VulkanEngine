// Per-frame CPU preparation, everything between "a frame starts" and "recording
// begins": light and fog parameters, cascade fitting, draw-item and batch
// construction, and the GPU-cull and object-data uploads.
//
// Split out of Renderer.cpp. Definitions only -- these remain Renderer member
// functions, so no call site changed.
#include "renderer/Renderer.h"
#include "renderer/RendererInternal.h"

#include "core/Logger.h"
#include "renderer/Bounds.h"
#include "rhi/VulkanDebugUtils.h"

#include <imgui.h>
#include <ImGuizmo.h> // must follow imgui.h (relies on its types)

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <functional>
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>


namespace ve {

uint32_t Renderer::activeCascadeCount() const
{
    return std::clamp(csmSettings_.cascadeCount, 1U, kMaxShadowCascades);
}

void Renderer::updateDemoLights(float elapsedSeconds)
{
    constexpr float kDegToRad = 3.1415926535897932385f / 180.0f;
    constexpr float kGoldenAngle = 2.39996322972865332f;

    // Deterministic fractional hash so each light gets a stable radius/height/
    // speed/hue without storing per-light state; the count slider just adds or
    // removes entries from the end of the swarm.
    const auto hash01 = [](int index, float seed) {
        const float value = std::sin(static_cast<float>(index) * 12.9898f + seed) * 43758.5453f;
        return value - std::floor(value);
    };

    // Minimal HSV->RGB for evenly spread, saturated light colors.
    const auto hueColor = [](float hue) {
        const glm::vec3 k{1.0f, 2.0f / 3.0f, 1.0f / 3.0f};
        const glm::vec3 p =
            glm::abs(glm::fract(glm::vec3(hue) + k) * 6.0f - glm::vec3(3.0f));
        return glm::clamp(p - glm::vec3(1.0f), glm::vec3(0.0f), glm::vec3(1.0f));
    };

    clusteredLighting_.clear();

    if (fragmentStressSceneActive_) {
        // Densely packed in the volume the slabs occupy, with a range far larger
        // than their spacing, so froxels see many lights at once. That is the
        // point: ablation put the per-fragment punctual loop at ~64% of
        // MainHDRPass, and it scales with lights *per froxel*, not lights total.
        for (int lightIndex = 0; lightIndex < renderer::kFragmentStressLightCount; ++lightIndex) {
            const float u = hash01(lightIndex, 3.0f);
            const float v = hash01(lightIndex, 17.0f);
            const float w = hash01(lightIndex, 31.0f);
            const glm::vec3 position{-12.0f + u * 24.0f, 0.5f + v * 8.0f, -2.0f - w * 20.0f};
            clusteredLighting_.addPointLight(
                position, hueColor(hash01(lightIndex, 41.0f)), 6.0f, renderer::kFragmentStressLightRange);
        }
        return;
    }

    if (cornellBoxSceneActive_) {
        // One white light near the ceiling, and nothing else. The orbiting swarm
        // would light every surface directly, which is precisely what has to not
        // happen here: the floor between the blocks has to be lit by bounce
        // alone or there is nothing to look at.
        //
        // A point light rather than a spot so it fills the room the way the
        // original's area light does, and placed below the ceiling slab so the
        // ceiling itself is lit by its own bounce rather than from inside solid
        // geometry.
        clusteredLighting_.addPointLight(glm::vec3{0.0f, renderer::kCornellBoxHalfExtent * 2.0f - 1.2f, 0.0f},
                                         glm::vec3{1.0f, 0.96f, 0.90f},
                                         60.0f,
                                         26.0f);
        return;
    }

    const int lightCount = std::clamp(demoLightCount_, 0, 512);
    for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
        const float radius = 2.0f + 4.0f * hash01(lightIndex, 0.0f);
        const float height = 0.6f + 4.0f * hash01(lightIndex, 7.0f);
        const float speed = 0.15f + 0.55f * hash01(lightIndex, 13.0f);
        const float baseAngle = static_cast<float>(lightIndex) * kGoldenAngle;
        const float angle = baseAngle + (animateLights_ ? elapsedSeconds * speed : 0.0f);
        const glm::vec3 position{radius * std::cos(angle), height, radius * std::sin(angle)};
        const glm::vec3 color = hueColor(hash01(lightIndex, 21.0f));
        clusteredLighting_.addPointLight(position, color, demoLightIntensity_, demoLightRange_);
    }

    // A white overhead spot anchors the scene regardless of the swarm size.
    const float spotAngle = animateLights_ ? elapsedSeconds * 0.25f : 0.0f;
    clusteredLighting_.addSpotLight(glm::vec3{2.5f * std::cos(spotAngle), 6.0f, 2.5f * std::sin(spotAngle)},
                                    glm::vec3{0.0f, -1.0f, 0.0f},
                                    glm::vec3{1.0f, 1.0f, 1.0f},
                                    40.0f,
                                    16.0f,
                                    18.0f * kDegToRad,
                                    30.0f * kDegToRad);
}

void Renderer::updateVolumetricFogParams(uint32_t frameIndex, float aspectRatio)
{
    if (!volumetricFog_.available()) {
        return;
    }

    renderer::FogInjectParams params{};
    const glm::mat4 view = camera_.viewMatrix();
    const glm::mat4 projection = camera_.projectionMatrix(aspectRatio);
    params.inverseView = glm::inverse(view);
    // The camera's own projection, y-flip included, so the volume stays aligned
    // with the image the main pass will sample it from.
    params.inverseProjection = glm::inverse(projection);

    const uint32_t cascadeCount = activeCascadeCount();
    for (uint32_t cascade = 0; cascade < renderer::kMaxFogCascades; ++cascade) {
        // Unused slots repeat the last active cascade so an out-of-range read
        // still samples something valid rather than an identity matrix.
        const uint32_t source = std::min(cascade, cascadeCount - 1);
        params.cascadeViewProjection[cascade] = frameCascades_[source].lightViewProjection;
    }
    params.cascadeSplits = frameCascadeSplits_;

    params.cameraPosition = glm::vec4(camera_.position, 1.0f);
    const glm::vec4 lightDirection = activeDirectionalLightDirection();
    params.lightDirection = glm::vec4(glm::vec3(lightDirection), static_cast<float>(cascadeCount));
    params.lightColor = activeDirectionalLightColor();
    params.ambientColor =
        portfolioCaptureMode_ ? kPortfolioAmbientLightColor : kAmbientLightColor;

    params.fogParams = glm::vec4(std::max(fogSettings_.density, 0.0f),
                                 std::max(fogSettings_.maxDistance, renderer::kFogNearPlane + 1.0f),
                                 std::clamp(fogSettings_.anisotropy, -0.95f, 0.95f),
                                 std::max(fogSettings_.heightFalloff, 0.0f));
    params.fogParams2 = glm::vec4(fogSettings_.baseHeight,
                                 std::max(fogSettings_.ambientScale, 0.0f),
                                 std::max(fogSettings_.lightCullThreshold, 0.0f),
                                 0.0f);
    params.scatteringColor = glm::vec4(glm::max(glm::vec3{fogSettings_.scatteringColor[0],
                                                          fogSettings_.scatteringColor[1],
                                                          fogSettings_.scatteringColor[2]},
                                                glm::vec3{0.0f}),
                                       1.0f);

    // Temporal reprojection. The matrices come from the same previous-frame
    // state TAA already tracks, so fog and TAA cannot disagree about where the
    // camera was.
    params.previousViewProjection = previousFrameViewProjection_;
    params.previousView = previousFrameView_;

    if (fogSettings_.temporalEnabled) {
        // Halton 2,3,5 over a 16-frame cycle: the same low-discrepancy sequence
        // the TAA jitter uses, extended to the volume's third axis so slices are
        // sampled at varying depths too.
        const uint32_t sampleIndex = fogTemporalSampleIndex_ % 16u;
        params.jitterAndBlend = glm::vec4(halton(sampleIndex + 1u, 2u),
                                          halton(sampleIndex + 1u, 3u),
                                          halton(sampleIndex + 1u, 5u),
                                          std::clamp(fogSettings_.temporalBlend, 0.0f, 0.98f));
        ++fogTemporalSampleIndex_;
    } else {
        // Froxel centres and no history: deterministic, and the reference the
        // temporal path is judged against.
        params.jitterAndBlend = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f);
    }

    volumetricFog_.updateParams(frameIndex, params);
}

// Computes a content hash per atlas tile and works out which tiles this frame
// would draw differently from what the atlas already holds.
//
// Per tile rather than per frame: one moved caster used to invalidate every
// tile, so a mostly-static scene with one moving object paid the full atlas
// cost. A tile only cares about the casters inside its own frustum.
//
// The enumeration below is the load-bearing part. Anything that can change a
// tile and is not hashed here becomes a stale shadow, which surfaces far from
// its cause. Hashing inputs rather than tracking dirty flags concentrates that
// risk into one list instead of spreading it over every mutable input.
void Renderer::updatePunctualShadowCacheState()
{
    const uint32_t slotCount = punctualShadows_.slotCount();
    punctualShadowSlotKeys_.assign(slotCount, 0);
    punctualShadowDirtySlots_.clear();

    for (uint32_t slot = 0; slot < slotCount; ++slot) {
        const renderer::ShadowAtlasRect rect = punctualShadows_.slotRect(slot);

        punctualShadowCacheKey_.reset();
        // The projection carries the light's position, direction, range, cone
        // and near plane; the rect carries where and at what resolution.
        punctualShadowCacheKey_.add(punctualShadows_.slotViewProjection(slot));
        punctualShadowCacheKey_.add(rect);
        // Raster depth bias is pipeline state the tile is rendered with.
        punctualShadowCacheKey_.add(shadowSettings_.rasterDepthBiasConstantFactor);
        punctualShadowCacheKey_.add(shadowSettings_.rasterDepthBiasSlopeFactor);

        // Only the casters this tile actually draws. The cull below mirrors the
        // one in recordPunctualShadowPass exactly -- if the two ever diverge,
        // the hash stops describing what gets drawn, so they are kept adjacent
        // in intent even though they live in different translation units.
        const renderer::Frustum& slotFrustum = punctualShadows_.slotFrustum(slot);
        for (const DrawItem& drawItem : allDrawItems_) {
            if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems || drawItem.indexCount == 0) {
                continue;
            }
            if (drawItem.bucket == RenderBucket::Blend) {
                continue;
            }
            if (drawItem.objectIndex < frameWorldBounds_.size() &&
                !slotFrustum.testAabb(frameWorldBounds_[drawItem.objectIndex])) {
                continue;
            }

            punctualShadowCacheKey_.add(static_cast<const void*>(drawItem.mesh));
            punctualShadowCacheKey_.add(drawItem.firstIndex);
            punctualShadowCacheKey_.add(drawItem.indexCount);
            punctualShadowCacheKey_.add(renderObjects_[drawItem.objectIndex].transform.modelMatrix());
        }

        // The skinned caster, on the same rule and for the same reason as the
        // cascades: it is in no draw-item list, and what changes about it is the
        // pose rather than anything the loop above hashes. The recorder tests
        // the identical frustum, so a tile that draws it always has it in its
        // key -- and a tile it cannot reach keeps caching.
        if (skinnedCasterCastsIntoFrustum(slotFrustum)) {
            const uint64_t pose = skinnedMesh_.poseHash();
            punctualShadowCacheKey_.addBytes(&pose, sizeof(pose));
        }

        const uint64_t key = punctualShadowCacheKey_.value();
        punctualShadowSlotKeys_[slot] = key;

        if (punctualShadowNeedsFullClear_) {
            // Nothing in the image is trustworthy yet, so every tile is dirty
            // regardless of what the hashes say.
            punctualShadowDirtySlots_.push_back(slot);
            continue;
        }

        const auto resident = punctualShadowResidentTiles_.find(packShadowAtlasRect(rect));
        if (resident == punctualShadowResidentTiles_.end() || resident->second != key) {
            punctualShadowDirtySlots_.push_back(slot);
        }
    }

    // Nothing to redraw means the pass is skipped outright: no clear, no draws,
    // no transition.
    punctualShadowCacheHit_ = punctualShadowDirtySlots_.empty() && slotCount > 0;
}

void Renderer::invalidatePunctualShadowCache()
{
    punctualShadowResidentTiles_.clear();
    punctualShadowNeedsFullClear_ = true;
    punctualShadowCacheHit_ = false;
}

// Computes a content hash per cascade and works out which cascades this frame
// would draw differently from what the shadow map already holds.
//
// Per cascade rather than per pass: the cascades cover nested but different
// volumes, so a caster moving inside cascade 0 says nothing about cascade 3.
// Each cascade also owns its own image layer for the life of the shadow map,
// which is why this keys by cascade index where the atlas has to key by tile
// rect -- a cascade's storage never migrates the way an atlas slot's does.
//
// As with the atlas, the enumeration below is the load-bearing part: anything
// that can change a cascade's image and is not hashed here becomes a stale
// shadow, which surfaces far from its cause.
void Renderer::advanceSkinnedAnimation(uint32_t frameIndex)
{
    // Called from drawFrame BEFORE updateVsmResidency, not from updateFrameData
    // with the rest of the per-frame animation.
    //
    // Residency decides which pages to invalidate from the caster's bounds and
    // pose digest, and the page pass then draws whatever pose is in the palette.
    // Advancing the pose after that decision means the two describe different
    // poses: a page the caster newly moves into is not invalidated, stays
    // cached, and holds depth that omits the caster -- for one frame, in the
    // direction that loses a shadow. The cascade and punctual keys were already
    // safe because they are built inside updateFrameData, after this ran; the
    // page invalidation was the one consumer that ran earlier.
    //
    // Writing the palette buffer this early is safe: drawFrame waits on this
    // frame slot's fence before any of this, so the GPU is done reading it.
    const float elapsedSeconds = static_cast<float>(frameClock_.elapsedSeconds());
    // Speed-scaled frame delta so play/pause holds the current pose and the
    // speed slider changes playback continuously.
    const float skinnedDelta = elapsedSeconds - previousElapsedSeconds_;
    previousElapsedSeconds_ = elapsedSeconds;
    if (animateSkinnedMesh_) {
        skinnedAnimationTime_ += skinnedDelta * skinnedAnimationSpeed_;
    }
    if (skinnedMesh_.valid()) {
        skinnedMesh_.update(frameIndex, skinnedAnimationTime_);
    }
}

void Renderer::updateCascadeShadowCacheState()
{
    const uint32_t cascadeCount = activeCascadeCount();
    cascadeShadowKeys_.fill(0);
    cascadeShadowDirty_.fill(false);
    cascadeShadowCascadesRedrawn_ = 0;

    // The GPU cull picks a LOD level per draw item from the projected screen
    // radius, so on that path the level -- not the CPU draw item's index range
    // -- is what decides the geometry a cascade rasterizes. Mirror the same
    // selection here. The mirror is renderer/MeshLod.h, which cull.comp is
    // written against and which the LOD tests cover.
    const bool gpuLodSelectionActive = isGpuShadowCullingActive() && !allDrawItems_.empty();
    const VkExtent2D renderExtent = renderResolution_.extent();
    const float projScaleY =
        0.5f * static_cast<float>(renderExtent.height) * std::abs(frameJitteredProjection_[1][1]);
    renderer::LodSelectionSettings lodSelection{};
    lodSelection.referenceRadiusPixels = lodSettings_.referenceRadiusPixels;
    // Shadow dispatches add the shadow bias on top of the shared one, matching
    // the `pc.params.w & 4` branch in cull.comp.
    lodSelection.bias = lodSettings_.bias + lodSettings_.shadowBias;
    lodSelection.forcedLod = lodSettings_.enabled ? lodSettings_.forcedLod : 0;

    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        renderer::CascadeShadowPassState state{};
        state.lightViewProjection = frameCascades_[cascadeIndex].lightViewProjection;
        state.rasterDepthBiasConstantFactor = shadowSettings_.rasterDepthBiasConstantFactor;
        state.rasterDepthBiasSlopeFactor = shadowSettings_.rasterDepthBiasSlopeFactor;
        state.shadowResolution = shadowSettings_.resolution;
        state.gpuLodSelectionActive = gpuLodSelectionActive;
        // Exactly the test the recorder makes before drawing it. The two have to
        // agree: a cascade that draws the skinned mesh without its pose in the
        // key would cache a shadow that never updates again.
        state.skinnedCasterPose = skinnedCasterCastsIntoCascade(cascadeIndex) ? skinnedMesh_.poseHash() : 0;

        // shadowCascadeDrawItems_ is already this cascade's frustum-culled
        // caster list, built by buildShadowDrawItems against the very frustum
        // the pass renders with. Reusing it rather than re-culling here is what
        // keeps the hash describing what actually gets drawn: there is only one
        // cull to keep correct, not two that can drift apart.
        const std::vector<DrawItem>& cascadeDrawItems = shadowCascadeDrawItems_[cascadeIndex];
        cascadeShadowCasterScratch_.clear();
        cascadeShadowCasterScratch_.reserve(cascadeDrawItems.size());

        for (const DrawItem& drawItem : cascadeDrawItems) {
            // Same three rejections the shadow recording applies, so an item the
            // pass never draws never enters the hash.
            if (!drawItem.mesh || drawItem.frameDataIndex >= kMaxDrawItems || drawItem.indexCount == 0) {
                continue;
            }
            if (drawItem.bucket == RenderBucket::Blend) {
                continue;
            }

            renderer::CascadeShadowCaster caster{};
            caster.mesh = static_cast<const void*>(drawItem.mesh);
            caster.material = static_cast<const void*>(drawItem.material);
            caster.firstIndex = drawItem.firstIndex;
            caster.indexCount = drawItem.indexCount;
            caster.bucket = static_cast<uint32_t>(drawItem.bucket);
            caster.alphaCutoff =
                drawItem.material != nullptr ? drawItem.material->alphaTestCutoff()
                                             : renderer::kNoAlphaTestCutoff;
            if (drawItem.objectIndex < renderObjects_.size()) {
                caster.modelMatrix = renderObjects_[drawItem.objectIndex].transform.modelMatrix();
            }
            caster.lodLevel = gpuLodSelectionActive ? selectedShadowLodLevel(drawItem, projScaleY, lodSelection) : 0u;

            cascadeShadowCasterScratch_.push_back(caster);
        }

        const uint64_t key = renderer::computeCascadeShadowKey(
            state,
            std::span<const renderer::CascadeShadowCaster>(cascadeShadowCasterScratch_.data(),
                                                           cascadeShadowCasterScratch_.size()));
        cascadeShadowKeys_[cascadeIndex] = key;

        if (cascadeShadowNeedsFullRedraw_ || cascadeShadowResidentKeys_[cascadeIndex] != key) {
            cascadeShadowDirty_[cascadeIndex] = true;
            ++cascadeShadowCascadesRedrawn_;
        }
    }

    // Inactive cascades hold whatever they last held. Marking them clean keeps
    // them out of the redraw count; raising the cascade count invalidates the
    // whole map anyway, since it reconfigures the image.
    cascadeShadowCacheHit_ = cascadeShadowCascadesRedrawn_ == 0 && cascadeCount > 0;
    cascadeShadowCachedFrames_ = cascadeShadowCacheHit_ ? cascadeShadowCachedFrames_ + 1 : 0;
}

bool Renderer::skinnedCasterCastsIntoFrustum(const renderer::Frustum& frustum) const
{
    // The same test every other caster gets, against whichever light frustum the
    // caller holds: a cascade's, or an atlas slot's. Shared so a cache key and
    // the recorder that fills it cannot answer differently.
    return skinnedCasterActive() && frustum.testAabb(skinnedMesh_.worldBounds());
}

bool Renderer::skinnedCasterCastsIntoCascade(uint32_t cascadeIndex) const
{
    if (!skinnedCasterActive() || cascadeIndex >= frameCascades_.size()) {
        return false;
    }

    // The layered path is a "no" for every cascade, not just for the recorder.
    // One draw fans out to all layers there and picks its matrix from
    // gl_ViewIndex, which a pushed matrix cannot answer, so the caster is
    // skipped -- and a pose that is never drawn must not enter a cache key
    // either. It would dirty every cascade on every frame of the animation, and
    // the layered pass redraws all of them together, so the whole shadow map
    // would redraw for geometry it does not contain.
    if (isLayeredCascadeRenderingActive()) {
        return false;
    }

    return skinnedCasterCastsIntoFrustum(frameCascades_[cascadeIndex].lightFrustum);
}

// The level cull.comp would pick for this item on a shadow dispatch. Kept
// beside the hash rather than in the shared LOD header because it needs the
// renderer's per-frame world bounds, which the GPU-free header cannot see.
uint32_t Renderer::selectedShadowLodLevel(const DrawItem& drawItem,
                                          float projScaleY,
                                          const renderer::LodSelectionSettings& settings) const
{
    const size_t drawIndex = static_cast<size_t>(&drawItem - allDrawItems_.data());
    if (drawIndex >= frameDrawItemLodRanges_.size()) {
        return 0;
    }
    const uint32_t lodCount = frameDrawItemLodRanges_[drawIndex].y;
    if (lodCount <= 1) {
        return 0;
    }
    if (drawItem.objectIndex >= frameWorldBounds_.size()) {
        return 0;
    }

    const renderer::Aabb& worldBounds = frameWorldBounds_[drawItem.objectIndex];
    if (!worldBounds.valid()) {
        return 0;
    }

    const glm::vec3 center = (worldBounds.min + worldBounds.max) * 0.5f;
    const float radius = glm::length(worldBounds.max - center);
    const float distanceToCamera = glm::length(center - frameCameraPosition_);
    return renderer::selectLodIndex(
        renderer::projectedScreenRadius(radius, distanceToCamera, projScaleY), lodCount, settings);
}

void Renderer::invalidateCascadeShadowCache()
{
    cascadeShadowResidentKeys_.fill(0);
    cascadeShadowNeedsFullRedraw_ = true;
    cascadeShadowCacheHit_ = false;
    cascadeShadowCachedFrames_ = 0;
}

void Renderer::updatePunctualShadowSlots(uint32_t frameIndex, float aspectRatio)
{
    punctualShadows_.beginFrame();

    std::vector<renderer::GpuLight>& lights = clusteredLighting_.lights();
    const bool atlasUsable = punctualShadows_.valid() && usePunctualShadows_;

    // Start every light unshadowed; the passes below only ever upgrade one.
    //
    // The normal bias rides along in .w so the shader can apply it without
    // reading the shadow slot. It has to bias the position *before* it can pick
    // a cube face, so taking it from the slot forced a second read of that
    // buffer purely to reach one float. The bias is a single global value, and
    // GpuLight is already in registers by then, so this costs nothing.
    const float normalBias = punctualShadows_.normalBias();
    for (renderer::GpuLight& light : lights) {
        light.spotScaleOffset.z = renderer::punctualShadowSlotToFloat(renderer::kInvalidPunctualShadowSlot);
        light.spotScaleOffset.w = normalBias;
    }

    if (!atlasUsable) {
        punctualShadowSlotsUsed_ = 0;
        punctualShadows_.upload(frameIndex);
        return;
    }

    // Rank every punctual light by how large it projects on screen, using the
    // same projScaleY the mesh-LOD selection uses so "how big does this look"
    // means one thing across the engine. That drives both who gets a tile at all
    // and how big a tile they get -- a light dominating the view is worth four
    // times the resolution of one barely visible, and the old distance-only
    // ordering could not express that.
    const float projScaleY =
        static_cast<float>(renderResolution_.extent().height) * 0.5f * camera_.projectionMatrix(aspectRatio)[1][1];

    punctualShadowCandidates_.clear();
    punctualShadowCandidates_.reserve(lights.size());
    for (const renderer::GpuLight& light : lights) {
        renderer::PunctualShadowCandidateInput candidate{};
        candidate.isSpot = light.directionType.w > 0.5f;
        candidate.range = light.positionRange.w;
        candidate.projectedRadius = renderer::punctualShadowProjectedRadius(
            glm::vec3(light.positionRange), light.positionRange.w, camera_.position, std::abs(projScaleY));
        punctualShadowCandidates_.push_back(candidate);
    }

    // Ranking, size classes and the point-light budget all live in
    // renderer/PunctualShadowAtlas.h, where they are unit-tested. What stays here
    // is the GPU-side half: inserting into the atlas and writing the resulting
    // slot back into the light.
    renderer::rankPunctualShadowAssignments(punctualShadowCandidates_,
                                            static_cast<uint32_t>(std::max(maxShadowCastingPointLights_, 0)),
                                            punctualShadowAssignments_);

    for (const renderer::PunctualShadowAssignment& assignment : punctualShadowAssignments_) {
        renderer::GpuLight& light = lights[assignment.lightIndex];

        if (assignment.isSpot) {
            const float cosOuter = std::clamp(light.spotScaleOffset.x, -1.0f, 1.0f);
            const float outerAngle = std::acos(cosOuter);
            const uint32_t slot = punctualShadows_.addSpotLight(glm::vec3(light.positionRange),
                                                                glm::vec3(light.directionType),
                                                                outerAngle,
                                                                light.positionRange.w,
                                                                assignment.sizeClass);
            // A full atlas degrades to an unshadowed light rather than an error,
            // and lower-ranked lights may still fit in a smaller leftover tile,
            // so this keeps walking instead of breaking out.
            light.spotScaleOffset.z = renderer::punctualShadowSlotToFloat(slot);
            continue;
        }

        const uint32_t baseSlot = punctualShadows_.addPointLight(
            glm::vec3(light.positionRange), light.positionRange.w, assignment.sizeClass);
        if (baseSlot == renderer::kInvalidPunctualShadowSlot) {
            continue;
        }

        light.spotScaleOffset.z = renderer::punctualShadowSlotToFloat(baseSlot);
    }

    // Measure how much the assignment moved. Lights popping in and out of the
    // shadowed set frame to frame is what reads as flicker, so it gets counted
    // rather than eyeballed.
    punctualShadowLightState_.resize(lights.size());
    punctualShadowAssignmentChurn_ = 0;
    for (size_t lightIndex = 0; lightIndex < lights.size(); ++lightIndex) {
        const bool shadowed = lights[lightIndex].spotScaleOffset.z >= 0.0f;
        PunctualShadowLightState& state = punctualShadowLightState_[lightIndex];
        if (state.valid && state.shadowed != shadowed) {
            ++punctualShadowAssignmentChurn_;
            ++punctualShadowAssignmentChurnTotal_;
        }
        state.shadowed = shadowed;
        state.valid = true;
    }

    punctualShadowSlotsUsed_ = punctualShadows_.slotCount();
    punctualShadows_.upload(frameIndex);
}

glm::vec4 Renderer::activeDirectionalLightDirection() const
{
    if (portfolioCaptureMode_) {
        return kPortfolioLightDirection;
    }

    const glm::vec3 direction =
        normalizedOrFallback(directionalLightSettings_.direction,
                             glm::normalize(glm::vec3{kDirectionalLightDirection.x,
                                                      kDirectionalLightDirection.y,
                                                      kDirectionalLightDirection.z}));
    return glm::vec4(direction, 0.0f);
}

glm::vec4 Renderer::activeDirectionalLightColor() const
{
    if (portfolioCaptureMode_) {
        return kPortfolioLightColor;
    }

    const float intensity = std::max(directionalLightSettings_.intensity, 0.0f);
    return glm::vec4(glm::max(directionalLightSettings_.color, glm::vec3{0.0f}) * intensity, 1.0f);
}

void Renderer::updateCascades(float aspectRatio)
{
    // The cascade fitting math is GPU-independent and lives in CascadeMath.h so
    // it can be unit-tested. Gather the inputs from renderer state, run the pure
    // solver, then cache the results plus their packed frustum planes.
    const glm::vec4 activeLightDirection = activeDirectionalLightDirection();

    renderer::CascadeBuildInput cascadeInput{};
    cascadeInput.requestedCascadeCount = csmSettings_.cascadeCount;
    cascadeInput.nearPlane = csmSettings_.nearPlane;
    cascadeInput.farPlane = csmSettings_.farPlane;
    cascadeInput.shadowDistance = csmSettings_.shadowDistance;
    cascadeInput.lambda = csmSettings_.lambda;
    cascadeInput.enableTexelSnapping = csmSettings_.enableTexelSnapping;
    cascadeInput.enableStableFit = csmSettings_.enableStableCascadeFit;
    cascadeInput.shadowResolution = shadowSettings_.resolution;
    cascadeInput.cameraPosition = camera_.position;
    cascadeInput.cameraTarget = camera_.target;
    cascadeInput.cameraUp = camera_.up;
    cascadeInput.cameraVerticalFovRadians = camera_.verticalFovRadians;
    cascadeInput.lightDirection =
        glm::vec3{activeLightDirection.x, activeLightDirection.y, activeLightDirection.z};
    cascadeInput.aspectRatio = aspectRatio;

    const renderer::CascadeBuildOutput cascadeOutput = renderer::computeShadowCascades(cascadeInput);

    frameCascades_ = cascadeOutput.cascades;
    frameCascadeSplits_ = cascadeOutput.splitDistances;

    for (uint32_t cascadeIndex = 0; cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
        const renderer::Frustum& lightFrustum = frameCascades_[cascadeIndex].lightFrustum;
        for (size_t planeIndex = 0; planeIndex < frameShadowCascadeFrustumPlanes_[cascadeIndex].size(); ++planeIndex) {
            const renderer::FrustumPlane& lightPlane = lightFrustum.planes[planeIndex];
            frameShadowCascadeFrustumPlanes_[cascadeIndex][planeIndex] =
                glm::vec4(lightPlane.normal, lightPlane.distance);
        }
    }
}

const renderer::Material* Renderer::resolveMaterial(const renderer::RenderObject& object,
                                                    const renderer::MeshPrimitive* primitive) const
{
    if (primitive && object.materialTable && primitive->materialIndex < object.materialCount) {
        return &object.materialTable[primitive->materialIndex];
    }

    return object.material;
}

bool Renderer::isRenderObjectActive(const renderer::RenderObject& object) const
{
    if (!object.visible) {
        return false;
    }
    if (object.sourceType == renderer::RenderObjectSourceType::OcclusionTest) {
        return occlusionTestSceneActive_ && !portfolioCaptureMode_;
    }
    if (object.sourceType == renderer::RenderObjectSourceType::CornellBox) {
        return cornellBoxSceneActive_ && !portfolioCaptureMode_;
    }
    if ((occlusionTestSceneActive_ || cornellBoxSceneActive_) && !portfolioCaptureMode_) {
        return false;
    }
    if (portfolioCaptureMode_ && object.hideInPortfolio) {
        return false;
    }
    return !object.portfolioOnly || portfolioCaptureMode_;
}

Renderer::RenderBucket Renderer::renderBucketForMaterial(const renderer::Material* material)
{
    if (!material) {
        return RenderBucket::Opaque;
    }

    switch (material->alphaModeValue()) {
    case renderer::AlphaMode::Mask:
        return RenderBucket::Mask;
    case renderer::AlphaMode::Blend:
        return RenderBucket::Blend;
    case renderer::AlphaMode::Opaque:
        break;
    }
    return RenderBucket::Opaque;
}

const char* Renderer::renderBucketName(RenderBucket bucket)
{
    switch (bucket) {
    case RenderBucket::Mask:
        return "Mask";
    case RenderBucket::Blend:
        return "Blend";
    case RenderBucket::Opaque:
        break;
    }
    return "Opaque";
}

void Renderer::appendDrawItemsForObject(uint32_t objectIndex,
                                        renderer::FrameCapacityBudget& budget,
                                        std::vector<DrawItem>& drawItems) const
{
    if (objectIndex >= renderObjects_.size()) {
        return;
    }

    const renderer::RenderObject& object = renderObjects_[objectIndex];
    if (!isRenderObjectActive(object)) {
        return;
    }

    const renderer::Mesh* mesh = object.mesh;
    if (!mesh || !mesh->valid()) {
        return;
    }

    if (mesh->hasSubMeshes()) {
        const std::span<const renderer::MeshPrimitive> primitives = mesh->primitives();
        for (size_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
            const renderer::MeshPrimitive& primitive = primitives[primitiveIndex];
            // Order matters: an empty primitive is charged to nothing, so it can
            // neither consume a slot nor be reported as dropped geometry.
            if (primitive.indexCount == 0) {
                continue;
            }
            // `continue`, not `return`: the remaining primitives still have to be
            // offered so the drop count is a real total. The old early return made
            // the size of the loss unknowable, which is what this replaces.
            if (!budget.admitDrawItem()) {
                continue;
            }

            DrawItem drawItem{};
            drawItem.mesh = mesh;
            drawItem.material = resolveMaterial(object, &primitive);
            drawItem.objectIndex = objectIndex;
            drawItem.submeshIndex = static_cast<uint32_t>(primitiveIndex);
            drawItem.firstIndex = primitive.firstIndex;
            drawItem.indexCount = primitive.indexCount;
            drawItem.frameDataIndex = static_cast<uint32_t>(drawItems.size());
            drawItem.bucket = renderBucketForMaterial(drawItem.material);
            drawItems.push_back(drawItem);
        }
        return;
    }

    if (mesh->indexCount() == 0) {
        return;
    }
    if (!budget.admitDrawItem()) {
        return;
    }

    DrawItem drawItem{};
    drawItem.mesh = mesh;
    drawItem.material = object.material;
    drawItem.objectIndex = objectIndex;
    drawItem.indexCount = mesh->indexCount();
    drawItem.frameDataIndex = static_cast<uint32_t>(drawItems.size());
    drawItem.bucket = renderBucketForMaterial(drawItem.material);
    drawItems.push_back(drawItem);
}

void Renderer::framePrepParallelFor(size_t count, const std::function<void(size_t, size_t)>& body)
{
    // Chunks below this size cost more to dispatch than to run inline.
    constexpr size_t kMinChunkSize = 64;
    if (parallelFramePrepEnabled_) {
        jobSystem_.parallelFor(count, kMinChunkSize, body);
    } else if (count > 0) {
        body(0, count);
    }
}

void Renderer::updateFrameWorldBounds()
{
    const size_t objectCount = renderObjects_.size();
    frameWorldBounds_.resize(objectCount);
    framePrepParallelFor(objectCount, [this](size_t begin, size_t end) {
        for (size_t objectIndex = begin; objectIndex < end; ++objectIndex) {
            frameWorldBounds_[objectIndex] = renderObjects_[objectIndex].worldBounds();
        }
    });
}

void Renderer::reportFrameCapacityOverflow()
{
    const uint32_t droppedObjects = frameCapacityBudget_.droppedObjects();
    const uint32_t droppedDrawItems = frameCapacityBudget_.droppedDrawItems();

    // Log on change, not per frame. A scene that overflows overflows on every
    // frame, and a per-frame line would bury the log it is meant to make useful.
    // Latching on the counts rather than on a bool also reports a scene that
    // grows further past the cap, which a plain "already warned" flag would hide.
    if (droppedObjects == loggedDroppedObjects_ && droppedDrawItems == loggedDroppedDrawItems_) {
        return;
    }
    const bool wasOverflowing = loggedDroppedObjects_ != 0 || loggedDroppedDrawItems_ != 0;
    loggedDroppedObjects_ = droppedObjects;
    loggedDroppedDrawItems_ = droppedDrawItems;

    if (!frameCapacityBudget_.overflowed()) {
        if (wasOverflowing) {
            Logger::info("Frame capacity is back under its caps; all geometry is being submitted again.");
        }
        return;
    }

    std::ostringstream message;
    message << "Frame capacity exceeded -- geometry is missing from this frame.";
    if (droppedObjects > 0) {
        message << " " << droppedObjects << " render object(s) past the " << kMaxFrameObjects
                << "-object cap were skipped entirely (invisible to culling, shadows, and GPU-cull input).";
    }
    if (droppedDrawItems > 0) {
        message << " " << droppedDrawItems << " draw item(s) past the " << kMaxDrawItems
                << "-draw-item cap were dropped.";
    }
    message << " Raise kMaxFrameObjects / kMaxDrawItems in renderer/RendererInternal.h (ceiling 65535).";
    Logger::warn(message.str());
}

size_t Renderer::sweptObjectCount() const
{
    return std::min(renderObjects_.size(), static_cast<size_t>(kMaxFrameObjects));
}

void Renderer::buildDrawItems()
{
    allDrawItems_.clear();
    allDrawItems_.reserve(renderObjects_.size());

    frameCapacityBudget_ = renderer::FrameCapacityBudget(kMaxFrameObjects, kMaxDrawItems);
    const size_t objectCount = frameCapacityBudget_.admitObjects(renderObjects_.size());
    // No early break once the draw-item cap fills. The loop keeps offering items
    // so the budget can report how many were actually lost, which costs a walk of
    // the remaining objects -- but only in a scene that is already being drawn
    // incompletely, and a wrong number there is worse than a slow one.
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        appendDrawItemsForObject(static_cast<uint32_t>(objectIndex), frameCapacityBudget_, allDrawItems_);
    }
    reportFrameCapacityOverflow();

    // Bucket is the primary key so each bucket is one contiguous range (the pass
    // and pipeline split reduces to a range walk); mesh stays the secondary key so
    // batching inside a bucket still coalesces vertex/index buffer binds.
    std::stable_sort(allDrawItems_.begin(), allDrawItems_.end(), [](const DrawItem& lhs, const DrawItem& rhs) {
        if (lhs.bucket != rhs.bucket) {
            return lhs.bucket < rhs.bucket;
        }
        return std::less<const renderer::Mesh*>{}(lhs.mesh, rhs.mesh);
    });

    // Blend is the last bucket, so its items form a suffix that gets reordered
    // back to front. Runs before frameDataIndex is assigned so the object-data
    // slots follow the final order.
    sortTransparentDrawItems();

    for (size_t drawIndex = 0; drawIndex < allDrawItems_.size(); ++drawIndex) {
        allDrawItems_[drawIndex].frameDataIndex = static_cast<uint32_t>(drawIndex);
    }
}

void Renderer::buildVisibleDrawItems(const renderer::Frustum& frustum)
{
    visibleDrawItems_.clear();
    visibleDrawItems_.reserve(allDrawItems_.size());
    cullingStats_ = {};
    cullingStats_.totalDrawItems = allDrawItems_.size();

    const size_t objectCount = sweptObjectCount();
    // uint8_t instead of vector<bool>: parallel chunks write disjoint indices,
    // which vector<bool>'s packed bits would turn into data races.
    std::vector<uint8_t> objectVisible(objectCount, 0);
    std::atomic<size_t> totalObjects{0};
    std::atomic<size_t> culledObjects{0};
    std::atomic<size_t> visibleObjects{0};
    framePrepParallelFor(objectCount, [&](size_t begin, size_t end) {
        size_t chunkTotal = 0;
        size_t chunkCulled = 0;
        size_t chunkVisible = 0;
        for (size_t objectIndex = begin; objectIndex < end; ++objectIndex) {
            const renderer::RenderObject& object = renderObjects_[objectIndex];
            if (!isRenderObjectActive(object)) {
                continue;
            }
            if (!object.mesh || !object.mesh->valid()) {
                continue;
            }

            ++chunkTotal;
            const renderer::Aabb& worldBounds = frameWorldBounds_[objectIndex];
            if (worldBounds.valid() && !frustum.testAabb(worldBounds)) {
                ++chunkCulled;
                continue;
            }

            ++chunkVisible;
            objectVisible[objectIndex] = 1;
        }
        totalObjects.fetch_add(chunkTotal, std::memory_order_relaxed);
        culledObjects.fetch_add(chunkCulled, std::memory_order_relaxed);
        visibleObjects.fetch_add(chunkVisible, std::memory_order_relaxed);
    });
    cullingStats_.totalObjects = totalObjects.load();
    cullingStats_.culledObjects = culledObjects.load();
    cullingStats_.visibleObjects = visibleObjects.load();

    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex < objectVisible.size() && objectVisible[drawItem.objectIndex] != 0) {
            visibleDrawItems_.push_back(drawItem);
        }
    }
}

void Renderer::updateVisibleBucketRanges()
{
    // visibleDrawItems_ inherits allDrawItems_' (bucket, mesh) ordering, so each
    // bucket is one contiguous run and the ranges are a single linear scan.
    for (RenderBucketRange& range : frameVisibleBucketRanges_) {
        range = {};
    }

    size_t index = 0;
    for (size_t bucket = 0; bucket < kRenderBucketCount; ++bucket) {
        const RenderBucket wanted = static_cast<RenderBucket>(bucket);
        const size_t begin = index;
        while (index < visibleDrawItems_.size() && visibleDrawItems_[index].bucket == wanted) {
            ++index;
        }
        frameVisibleBucketRanges_[bucket] = {static_cast<uint32_t>(begin), static_cast<uint32_t>(index)};
    }
}

void Renderer::sortTransparentDrawItems()
{
    // Operates on allDrawItems_, not visibleDrawItems_: when GPU culling is on,
    // the cull input is marshalled from allDrawItems_ while the batches index
    // visibleDrawItems_, and the two are only interchangeable because they share
    // an ordering. Sorting one and not the other would silently mismatch every
    // batch's command range.
    const auto blendBegin = std::find_if(allDrawItems_.begin(), allDrawItems_.end(), [](const DrawItem& drawItem) {
        return drawItem.bucket == RenderBucket::Blend;
    });
    if (std::distance(blendBegin, allDrawItems_.end()) < 2) {
        return;
    }

    // Back to front by distance from the camera to the object's world-bounds
    // centre. Unlike the opaque buckets, the blend range cannot stay grouped by
    // mesh: correct compositing order beats batching, and transparent draws are
    // few enough for that to be the right trade.
    const glm::vec3 cameraPosition = camera_.position;
    const auto viewDistanceSquared = [&](const DrawItem& drawItem) {
        if (drawItem.objectIndex >= frameWorldBounds_.size()) {
            return 0.0f;
        }
        const renderer::Aabb& bounds = frameWorldBounds_[drawItem.objectIndex];
        if (!bounds.valid()) {
            return 0.0f;
        }
        const glm::vec3 offset = (bounds.min + bounds.max) * 0.5f - cameraPosition;
        return glm::dot(offset, offset);
    };

    std::stable_sort(blendBegin, allDrawItems_.end(), [&](const DrawItem& lhs, const DrawItem& rhs) {
        return viewDistanceSquared(lhs) > viewDistanceSquared(rhs);
    });
}

void Renderer::buildMeshDrawBatches()
{
    meshDrawBatches_.clear();
    cullingStats_.commandCount = visibleDrawItems_.size();
    buildMeshDrawBatchesForItems(visibleDrawItems_, meshDrawBatches_);
    cullingStats_.batchCount = meshDrawBatches_.size();
}

void Renderer::buildShadowDrawItems(uint32_t cascadeIndex, const renderer::Frustum& lightFrustum)
{
    if (cascadeIndex >= shadowCascadeDrawItems_.size()) {
        return;
    }

    std::vector<DrawItem>& cascadeDrawItems = shadowCascadeDrawItems_[cascadeIndex];
    cascadeDrawItems.clear();
    cascadeDrawItems.reserve(allDrawItems_.size());

    // Serial on purpose: this runs inside the per-cascade parallel loop in
    // buildShadowFrameData, and framePrepParallelFor must not nest.
    const size_t objectCount = sweptObjectCount();
    std::vector<uint8_t> objectVisible(objectCount, 0);
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        const renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!isRenderObjectActive(object)) {
            continue;
        }
        if (!object.mesh || !object.mesh->valid()) {
            continue;
        }

        const renderer::Aabb& worldBounds = frameWorldBounds_[objectIndex];
        if (worldBounds.valid() && !lightFrustum.testAabb(worldBounds)) {
            continue;
        }

        objectVisible[objectIndex] = 1;
    }

    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex < objectVisible.size() && objectVisible[drawItem.objectIndex] != 0) {
            cascadeDrawItems.push_back(drawItem);
        }
    }
}

// Same object sweep as buildShadowDrawItems, but an object survives if ANY
// active cascade wants it. Matches what the GPU shadow cull now computes, so
// both paths feed the layered pass the same set.
void Renderer::buildUnionShadowDrawItems(uint32_t cascadeCount)
{
    shadowDrawItems_.clear();
    shadowDrawItems_.reserve(allDrawItems_.size());

    const size_t objectCount = sweptObjectCount();
    std::vector<uint8_t> objectVisible(objectCount, 0);
    for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
        const renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!isRenderObjectActive(object)) {
            continue;
        }
        if (!object.mesh || !object.mesh->valid()) {
            continue;
        }

        const renderer::Aabb& worldBounds = frameWorldBounds_[objectIndex];
        if (!worldBounds.valid()) {
            objectVisible[objectIndex] = 1;
            continue;
        }
        for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
            if (frameCascades_[cascadeIndex].lightFrustum.testAabb(worldBounds)) {
                objectVisible[objectIndex] = 1;
                break;
            }
        }
    }

    for (const DrawItem& drawItem : allDrawItems_) {
        if (drawItem.objectIndex < objectVisible.size() && objectVisible[drawItem.objectIndex] != 0) {
            shadowDrawItems_.push_back(drawItem);
        }
    }
}

void Renderer::buildShadowMeshDrawBatches()
{
    shadowMeshDrawBatches_.clear();
    buildMeshDrawBatchesForItems(shadowDrawItems_, shadowMeshDrawBatches_);
    shadowCullingStats_.batchCount = shadowMeshDrawBatches_.size();
}

void Renderer::buildMeshDrawBatchesForItems(const std::vector<DrawItem>& drawItems,
                                            std::vector<MeshDrawBatch>& batches) const
{
    renderer::buildMeshDrawBatches(drawItems, kMaxDrawItems, batches);
}

void Renderer::buildFrameMeshLodTable()
{
    frameMeshLodTable_.clear();
    frameDrawItemLodRanges_.assign(allDrawItems_.size(), glm::uvec2(0));

    // Dedup by mesh: a mesh's whole chain is uploaded once and every draw item
    // referencing it offsets into that block. Serial on purpose -- the dedup map
    // is shared state, and the loop is one hash lookup per draw item.
    std::unordered_map<const renderer::Mesh*, uint32_t> meshLodBases;
    for (size_t drawIndex = 0; drawIndex < allDrawItems_.size(); ++drawIndex) {
        const DrawItem& drawItem = allDrawItems_[drawIndex];
        const renderer::Mesh* mesh = drawItem.mesh;
        if (!mesh) {
            continue;
        }

        const std::span<const renderer::MeshLod> meshLods = mesh->lods();
        if (meshLods.empty()) {
            continue;
        }

        uint32_t meshBase = 0;
        const auto found = meshLodBases.find(mesh);
        if (found != meshLodBases.end()) {
            meshBase = found->second;
        } else {
            if (frameMeshLodTable_.size() + meshLods.size() > kMaxMeshLodEntries) {
                // Table full: leave this draw item without a chain so the cull
                // shader falls back to its authored index range.
                continue;
            }
            meshBase = static_cast<uint32_t>(frameMeshLodTable_.size());
            frameMeshLodTable_.insert(frameMeshLodTable_.end(), meshLods.begin(), meshLods.end());
            meshLodBases.emplace(mesh, meshBase);
        }

        // Sub-meshed meshes carry a chain per primitive; the rest use the
        // whole-mesh range.
        uint32_t localBase = 0;
        uint32_t localCount = 0;
        if (mesh->hasSubMeshes()) {
            const std::span<const renderer::MeshPrimitive> primitives = mesh->primitives();
            if (drawItem.submeshIndex < primitives.size()) {
                localBase = primitives[drawItem.submeshIndex].lodBase;
                localCount = primitives[drawItem.submeshIndex].lodCount;
            }
        } else {
            localBase = mesh->lodBase();
            localCount = mesh->lodCount();
        }

        if (localCount == 0 || static_cast<size_t>(localBase) + localCount > meshLods.size()) {
            continue;
        }

        frameDrawItemLodRanges_[drawIndex] = glm::uvec2(meshBase + localBase, localCount);
    }
}

void Renderer::uploadGpuCullFrameParams(uint32_t frameIndex, bool occlusionEnabledThisFrame)
{
    // Render extent: every threshold downstream of this is in rendered pixels --
    // the Hi-Z pyramid's dimensions, the occlusion pass's minimum screen size,
    // and the LOD reference radius. A half-scale frame genuinely does need less
    // mesh detail, so letting LOD follow the scale is the correct behaviour, not
    // a rounding artefact.
    const VkExtent2D extent = renderResolution_.extent();

    GpuCullFrameParams frameParams{};
    frameParams.occlusionViewProjection = depthPyramid_.viewProjection();
    frameParams.occlusionViewProjectionPhase2 = frameViewProjection_;
    frameParams.cameraPosition = glm::vec4(frameCameraPosition_, 0.0f);

    // projScaleY converts a world-space radius at unit distance into vertical
    // pixels: |proj[1][1]| is 1/tan(fovY/2), and half the viewport height maps NDC
    // [-1,1] onto pixels. The cull shader divides by view distance to get the
    // projected pixel radius it selects a LOD from.
    //
    // The abs() is load-bearing: these projections carry the Vulkan Y-flip, so
    // proj[1][1] is negative (see the ImGuizmo un-flip in drawViewportGizmo).
    // Passing it through signed makes the shader's projScaleY <= 0 guard fire on
    // every draw item, which silently pins everything to level 0.
    const float projScaleY = 0.5f * static_cast<float>(extent.height) * std::abs(frameJitteredProjection_[1][1]);
    frameParams.viewportAndMipCount = glm::vec4(static_cast<float>(extent.width),
                                                static_cast<float>(extent.height),
                                                static_cast<float>(depthPyramid_.mipLevels()),
                                                projScaleY);
    frameParams.occlusionSettings = glm::vec4(gpuOcclusionDepthBias_,
                                              gpuOcclusionNearDisableDistance_,
                                              gpuOcclusionMaxScreenCoverage_,
                                              gpuOcclusionMinScreenPixels_);
    // Disabling LOD selection is expressed as pinning level 0, so the shader has
    // one code path rather than an extra branch.
    const float forcedLod =
        lodSettings_.enabled ? static_cast<float>(lodSettings_.forcedLod) : 0.0f;
    frameParams.lodSettings = glm::vec4(lodSettings_.referenceRadiusPixels,
                                        lodSettings_.bias,
                                        forcedLod,
                                        lodSettings_.shadowBias);
    const uint32_t cascadeCount = activeCascadeCount();
    frameParams.counterAndFlags =
        glm::uvec4(kGpuCullStatsCounterOffset, occlusionEnabledThisFrame ? 1u : 0u, cascadeCount, 0u);
    // Cascade-major, active cascades only; the shader reads counterAndFlags.z of
    // them. Inactive slots stay zeroed rather than carrying a stale frustum.
    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        for (size_t planeIndex = 0; planeIndex < 6; ++planeIndex) {
            frameParams.shadowCascadePlanes[cascadeIndex * 6 + planeIndex] =
                frameShadowCascadeFrustumPlanes_[cascadeIndex][planeIndex];
        }
    }
    const VkExtent2D pyramidAllocation = renderResolution_.allocationExtent();
    frameParams.pyramidBaseSizes =
        glm::uvec4(extent.width, extent.height, pyramidAllocation.width, pyramidAllocation.height);
    gpuCulling_.paramBuffer(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullFrameParams>(&frameParams, 1)));
}

void Renderer::updateGpuCullInputBuffer(uint32_t frameIndex)
{
    if (allDrawItems_.empty()) {
        return;
    }
    if (!gpuCulling_.available()) {
        return;
    }

    // Two-phase mode drops the camera-still requirement: phase 1 projects with
    // the pyramid's stored (previous-frame) view-projection, and phase 2 corrects
    // any resulting false negatives against the mid-frame rebuild. Single-phase
    // keeps the conservative previous-frame validity gate.
    const bool occlusionEnabledThisFrame =
        isGpuOcclusionCullingActive() &&
        (frameTwoPhaseOcclusionActive_ || previousFrameDepthValidForOcclusion());
    // Remembered per frame SLOT, not per frame: the counters this slot produces
    // are not read back until it comes round again, and the yield controller has
    // to know whether occlusion was running when they were written.
    if (frameIndex < frameOcclusionTested_.size()) {
        frameOcclusionTested_[frameIndex] = occlusionEnabledThisFrame ? 1u : 0u;
    }
    uploadGpuCullFrameParams(frameIndex, occlusionEnabledThisFrame);

    if (!frameMeshLodTable_.empty()) {
        gpuCulling_.meshLodBuffer(frameIndex)
            .upload(std::as_bytes(
                std::span<const renderer::MeshLod>(frameMeshLodTable_.data(), frameMeshLodTable_.size())));
    }

    std::vector<GpuCullDrawItem> cullDrawItems(allDrawItems_.size());
    framePrepParallelFor(allDrawItems_.size(), [&](size_t begin, size_t end) {
        for (size_t drawIndex = begin; drawIndex < end; ++drawIndex) {
            const DrawItem& drawItem = allDrawItems_[drawIndex];
            GpuCullDrawItem& gpuDrawItem = cullDrawItems[drawIndex];

            renderer::Aabb worldBounds{};
            if (drawItem.objectIndex < frameWorldBounds_.size()) {
                worldBounds = frameWorldBounds_[drawItem.objectIndex];
            }

            if (worldBounds.valid()) {
                gpuDrawItem.boundsMin = glm::vec4(worldBounds.min, 0.0f);
                gpuDrawItem.boundsMax = glm::vec4(worldBounds.max, 0.0f);
            } else {
                gpuDrawItem.boundsMin =
                    glm::vec4(-kUnboundedCullExtent, -kUnboundedCullExtent, -kUnboundedCullExtent, 0.0f);
                gpuDrawItem.boundsMax =
                    glm::vec4(kUnboundedCullExtent, kUnboundedCullExtent, kUnboundedCullExtent, 0.0f);
            }

            gpuDrawItem.indexCount = drawItem.indexCount;
            gpuDrawItem.firstIndex = drawItem.firstIndex;
            gpuDrawItem.vertexOffset = drawItem.vertexOffset;
            gpuDrawItem.objectFrameDataIndex = drawItem.frameDataIndex;

            // (base, count) into the per-frame LOD table; (0, 0) when the mesh
            // has no chain, which makes the shader emit the authored range.
            const glm::uvec2 lodRange = drawIndex < frameDrawItemLodRanges_.size()
                                            ? frameDrawItemLodRanges_[drawIndex]
                                            : glm::uvec2(0);
            gpuDrawItem.lodBase = lodRange.x;
            gpuDrawItem.lodCount = lodRange.y;
        }
    });

    for (size_t batchIndex = 0; batchIndex < meshDrawBatches_.size(); ++batchIndex) {
        const MeshDrawBatch& batch = meshDrawBatches_[batchIndex];
        const uint32_t endDrawItem =
            std::min<uint32_t>(batch.beginDrawItem + batch.drawItemCount, static_cast<uint32_t>(cullDrawItems.size()));
        for (uint32_t drawItemIndex = batch.beginDrawItem; drawItemIndex < endDrawItem; ++drawItemIndex) {
            cullDrawItems[drawItemIndex].batchIndex = static_cast<uint32_t>(batchIndex);
            cullDrawItems[drawItemIndex].batchOutputBase = batch.compactedCommandOffset;
        }
    }

    gpuCulling_.cullInputBuffer(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullDrawItem>(cullDrawItems.data(), cullDrawItems.size())));
}

void Renderer::updateGpuShadowCullInputBuffer(uint32_t frameIndex)
{
    if (allDrawItems_.empty()) {
        return;
    }
    if (!gpuCulling_.shadowAvailable()) {
        return;
    }

    // Same params as the main dispatch (occlusion off): both read one buffer, so
    // writing identical values keeps the result independent of upload order.
    if (frameIndex < frameOcclusionTested_.size()) {
        frameOcclusionTested_[frameIndex] = 0u;
    }
    uploadGpuCullFrameParams(frameIndex, /*occlusionEnabledThisFrame=*/false);

    if (!frameMeshLodTable_.empty()) {
        gpuCulling_.meshLodBuffer(frameIndex)
            .upload(std::as_bytes(
                std::span<const renderer::MeshLod>(frameMeshLodTable_.data(), frameMeshLodTable_.size())));
    }

    std::vector<GpuCullDrawItem> cullDrawItems(allDrawItems_.size());
    framePrepParallelFor(allDrawItems_.size(), [&](size_t begin, size_t end) {
        for (size_t drawIndex = begin; drawIndex < end; ++drawIndex) {
            const DrawItem& drawItem = allDrawItems_[drawIndex];
            GpuCullDrawItem& gpuDrawItem = cullDrawItems[drawIndex];

            renderer::Aabb worldBounds{};
            if (drawItem.objectIndex < frameWorldBounds_.size()) {
                worldBounds = frameWorldBounds_[drawItem.objectIndex];
            }

            if (worldBounds.valid()) {
                gpuDrawItem.boundsMin = glm::vec4(worldBounds.min, 0.0f);
                gpuDrawItem.boundsMax = glm::vec4(worldBounds.max, 0.0f);
            } else {
                gpuDrawItem.boundsMin =
                    glm::vec4(-kUnboundedCullExtent, -kUnboundedCullExtent, -kUnboundedCullExtent, 0.0f);
                gpuDrawItem.boundsMax =
                    glm::vec4(kUnboundedCullExtent, kUnboundedCullExtent, kUnboundedCullExtent, 0.0f);
            }

            gpuDrawItem.indexCount = drawItem.indexCount;
            gpuDrawItem.firstIndex = drawItem.firstIndex;
            gpuDrawItem.vertexOffset = drawItem.vertexOffset;
            gpuDrawItem.objectFrameDataIndex = drawItem.frameDataIndex;

            // (base, count) into the per-frame LOD table; (0, 0) when the mesh
            // has no chain, which makes the shader emit the authored range.
            const glm::uvec2 lodRange = drawIndex < frameDrawItemLodRanges_.size()
                                            ? frameDrawItemLodRanges_[drawIndex]
                                            : glm::uvec2(0);
            gpuDrawItem.lodBase = lodRange.x;
            gpuDrawItem.lodCount = lodRange.y;
        }
    });

    for (size_t batchIndex = 0; batchIndex < gpuShadowMeshDrawBatches_.size(); ++batchIndex) {
        const MeshDrawBatch& batch = gpuShadowMeshDrawBatches_[batchIndex];
        const uint32_t endDrawItem =
            std::min<uint32_t>(batch.beginDrawItem + batch.drawItemCount, static_cast<uint32_t>(cullDrawItems.size()));
        for (uint32_t drawItemIndex = batch.beginDrawItem; drawItemIndex < endDrawItem; ++drawItemIndex) {
            cullDrawItems[drawItemIndex].batchIndex = static_cast<uint32_t>(batchIndex);
            cullDrawItems[drawItemIndex].batchOutputBase = batch.compactedCommandOffset;
        }
    }

    gpuCulling_.shadowCullInputBuffer(frameIndex)
        .upload(std::as_bytes(std::span<const GpuCullDrawItem>(cullDrawItems.data(), cullDrawItems.size())));
}

void Renderer::updateIndirectDrawBuffer(uint32_t frameIndex)
{
    if (visibleDrawItems_.empty()) {
        return;
    }

    std::vector<VkDrawIndexedIndirectCommand> indirectCommands(visibleDrawItems_.size());
    const bool objectDataArrayIndexingActive = isMainPassMultiDrawIndirectActive();
    for (size_t drawIndex = 0; drawIndex < visibleDrawItems_.size(); ++drawIndex) {
        const DrawItem& drawItem = visibleDrawItems_[drawIndex];
        VkDrawIndexedIndirectCommand& command = indirectCommands[drawIndex];
        command.indexCount = drawItem.indexCount;
        command.instanceCount = 1;
        command.firstIndex = drawItem.firstIndex;
        command.vertexOffset = drawItem.vertexOffset;
        command.firstInstance = objectDataArrayIndexingActive ? drawItem.frameDataIndex : 0;
    }

    frameIndirectDrawBuffers_.at(frameIndex)
        .upload(std::as_bytes(
            std::span<const VkDrawIndexedIndirectCommand>(indirectCommands.data(), indirectCommands.size())));
}

void Renderer::updateShadowIndirectDrawBuffer(uint32_t frameIndex)
{
    if (!isShadowIndirectActive() || shadowDrawItems_.empty()) {
        return;
    }

    std::vector<VkDrawIndexedIndirectCommand> indirectCommands(shadowDrawItems_.size());
    for (size_t drawIndex = 0; drawIndex < shadowDrawItems_.size(); ++drawIndex) {
        const DrawItem& drawItem = shadowDrawItems_[drawIndex];
        VkDrawIndexedIndirectCommand& command = indirectCommands[drawIndex];
        command.indexCount = drawItem.indexCount;
        command.instanceCount = 1;
        command.firstIndex = drawItem.firstIndex;
        command.vertexOffset = drawItem.vertexOffset;
        command.firstInstance = drawItem.frameDataIndex;
    }

    frameShadowIndirectDrawBuffers_.at(frameIndex)
        .upload(std::as_bytes(
            std::span<const VkDrawIndexedIndirectCommand>(indirectCommands.data(), indirectCommands.size())));
}

void Renderer::updateFrameData(uint32_t frameIndex)
{
    // From the frame clock, not the steady clock: this value drives the demo
    // light orbit, the skeletal animation delta, and every animated object
    // transform, so it is the single biggest determinism lever in the renderer.
    const float elapsedSeconds = static_cast<float>(frameClock_.elapsedSeconds());

    // Render extent, not swapchain: this feeds the TAA jitter (an NDC offset of
    // half a *render* pixel) and the cluster grid's screen dimensions. Aspect is
    // the same either way, since the scale is uniform.
    const VkExtent2D extent = renderResolution_.extent();
    const float aspect =
        extent.height == 0 ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height);
    const glm::mat4 view = camera_.viewMatrix();
    const glm::mat4 projection = camera_.projectionMatrix(aspect);
    const glm::mat4 viewProjection = projection * view;
    // PostProcessStack owns the TAA jitter sequence/state; it returns the NDC
    // offset to fold into the main projection (zero when jitter is inactive).
    glm::mat4 jitteredProjection = projection;
    const glm::vec2 jitterNdc = postProcess_.advanceJitter(extent);
    jitteredProjection[2][0] += jitterNdc.x;
    jitteredProjection[2][1] += jitterNdc.y;
    frameJitteredProjection_ = jitteredProjection;
    frameJitteredViewProjection_ = jitteredProjection * view;
    frameViewProjection_ = viewProjection;
    frameView_ = view;
    frameCameraPosition_ = camera_.position;
    // First frame after a history reset: reproject against the current matrices
    // so the velocity buffer reads as zero motion instead of garbage.
    if (!previousFrameViewProjectionValid_) {
        previousFrameViewProjection_ = viewProjection;
        previousFrameView_ = view;
        previousFrameViewProjectionValid_ = true;
    }

    // Regenerate the animated demo light swarm, then hand the froxel grid + light
    // culling the current view/inverse-projection and camera planes (view-space
    // work, so it runs regardless of scene contents).
    updateDemoLights(elapsedSeconds);
    // Assign atlas tiles before the light buffer is uploaded below: this stamps
    // the slot index into each GpuLight, so it has to run between the rebuild
    // and clusteredLighting_.upload().
    updatePunctualShadowSlots(frameIndex, aspect);
    // The skinned pose is NOT advanced here. It has to be in place before
    // updateVsmResidency, which decides which pages to invalidate, and that runs
    // earlier in drawFrame than this does -- see advanceSkinnedAnimation.
    clusteredLighting_.updateParams(frameIndex,
                                    view,
                                    glm::inverse(projection),
                                    camera_.nearPlane,
                                    camera_.farPlane,
                                    static_cast<float>(extent.width),
                                    static_cast<float>(extent.height));

    if (renderObjects_.empty()) {
        resetFrameStateForEmptyScene(frameIndex);
        return;
    }

    updateCascades(aspect);
    // After updateCascades so the injection pass gets this frame's cascade
    // matrices, and before recording, which reads the uploaded buffer.
    updateVolumetricFogParams(frameIndex, aspect);

    if (updateAnimatedTransforms(elapsedSeconds)) {
        invalidateDepthPyramid();
    }

    // Transforms are final for this frame; cache every object's world AABB once
    // for the visibility, shadow-cascade, and GPU-cull-input passes below.
    updateFrameWorldBounds();

    buildDrawItems();

    // Deliberately here and not with the slot assignment above. The hash covers
    // caster transforms and draw items, and both are still the previous frame's
    // values at that point -- updateAnimatedTransforms and buildDrawItems run
    // between. Hashing there would notice every change exactly one frame late,
    // which is a single stale shadow frame on every edit: visible, and hard to
    // attribute. The slots are final by now, so nothing is lost by waiting.
    updatePunctualShadowCacheState();
    buildFrameMeshLodTable();
    resetGpuCullFrameCounters(frameIndex);

    const renderer::Frustum cameraFrustum = renderer::Frustum::fromViewProjection(viewProjection);
    for (size_t planeIndex = 0; planeIndex < frameFrustumPlanes_.size(); ++planeIndex) {
        const renderer::FrustumPlane& cameraPlane = cameraFrustum.planes[planeIndex];
        frameFrustumPlanes_[planeIndex] = glm::vec4(cameraPlane.normal, cameraPlane.distance);
    }

    buildShadowFrameData(frameIndex);
    // After buildShadowFrameData, which is what fills the per-cascade caster
    // lists the hash reads, and after buildFrameMeshLodTable above, which fills
    // the LOD ranges the mirrored level selection needs.
    updateCascadeShadowCacheState();
    buildMainCullingFrameData(frameIndex, cameraFrustum);
    uploadObjectFrameData(frameIndex);
    clusteredLighting_.upload(frameIndex);

    // Resolved here (not in recordRenderCommands) so drawFrame can submit the
    // async compute work before the graphics command buffer is even recorded.
    const bool clusteredLightingActiveThisFrame = clusteredLighting_.available() && useClusteredLighting_ &&
                                                  clusteredLighting_.lightCount() > 0 && !allDrawItems_.empty();
    frameAsyncComputeActive_ = clusteredLightingActiveThisFrame && useAsyncCompute_ && asyncCompute_.available();

    // isIblBound() is part of the gate, not an optimisation: the trace subtracts
    // the specular IBL it replaces, so without those bindings it has no idea what
    // scene colour already contains -- and the descriptors would be statically
    // used while unwritten, which validation rejects outright.
    frameSsrActive_ =
        ssrSettings_.enabled && ssr_.available() && ssr_.isIblBound() && !allDrawItems_.empty();
    if (frameSsrActive_) {
        // The trace reconstructs positions from the jitter-rendered depth, so it
        // uses the same jittered projection the rasterizer used.
        ssr_.uploadParams(frameIndex, view, frameJitteredProjection_, ssrFrameCounter_++);
    }

    frameGtaoActive_ = ssaoSettings_.enabled && gtao_.available() && !allDrawItems_.empty();
    if (frameGtaoActive_) {
        // The horizon search reconstructs positions from the same jitter-rendered
        // depth the composite consumes, so it uses the jittered projection too.
        gtao_.uploadParams(frameIndex, view, frameJitteredProjection_, gtaoFrameCounter_++);
    }

    // Latched here rather than recomputed at record time because the batch has
    // to be chosen once: the graph declaration, the capture draws and the
    // convolution all have to agree on which probes this frame is about, and
    // beginCaptureBatch advances the round-robin cursor as a side effect.
    frameProbeCaptureActive_ = isProbeCaptureActive();
    if (frameProbeCaptureActive_) {
        frameProbeCaptureActive_ =
            !irradianceProbes_.beginCaptureBatch(static_cast<uint32_t>(giSettings_.probesPerFrame)).empty();
    } else {
        // Leaves the update pass with nothing to convolve, which is what makes
        // it fall back to the fill.
        irradianceProbes_.clearCaptureBatch();
    }
}

// Called from drawFrame after command recording: both the object-data upload and
// the skybox push constants must still see the previous frame's matrices, so the
// capture happens only once the frame has been fully recorded.

void Renderer::capturePreviousFrameMatrices()
{
    previousFrameViewProjection_ = frameViewProjection_;
    previousFrameView_ = frameView_;
    previousFrameViewProjectionValid_ = true;
    for (renderer::RenderObject& object : renderObjects_) {
        object.previousModelMatrix = object.transform.modelMatrix();
        object.previousModelValid = true;
    }
    if (skinnedMesh_.valid()) {
        previousSkinnedModelMatrix_ = skinnedMesh_.modelMatrix();
        previousSkinnedModelValid_ = true;
    }
}

void Renderer::resetFrameStateForEmptyScene(uint32_t frameIndex)
{
    // This path returns before the cache hash is recomputed, so the flag would
    // otherwise keep the previous frame's answer and suppress a pass that has
    // no business being suppressed.
    punctualShadowCacheHit_ = false;
    cascadeShadowCacheHit_ = false;
    cascadeShadowDirty_.fill(false);
    cascadeShadowCascadesRedrawn_ = 0;
    frameTwoPhaseOcclusionActive_ = false;
    frameAsyncComputeActive_ = false;
    frameSsrActive_ = false;
    frameGtaoActive_ = false;
    frameProbeCaptureActive_ = false;
    irradianceProbes_.clearCaptureBatch();
    // Same reasoning as the cache flags above: this path never runs buildDrawItems,
    // so the budget would keep reporting the previous scene's overflow forever.
    frameCapacityBudget_ = renderer::FrameCapacityBudget(kMaxFrameObjects, kMaxDrawItems);
    reportFrameCapacityOverflow();
    allDrawItems_.clear();
    visibleDrawItems_.clear();
    shadowDrawItems_.clear();
    meshDrawBatches_.clear();
    shadowMeshDrawBatches_.clear();
    gpuShadowMeshDrawBatches_.clear();
    for (std::vector<DrawItem>& cascadeDrawItems : shadowCascadeDrawItems_) {
        cascadeDrawItems.clear();
    }
    for (std::vector<MeshDrawBatch>& cascadeBatches : shadowCascadeMeshDrawBatches_) {
        cascadeBatches.clear();
    }
    shadowVisibleDrawItemsPerCascade_.fill(0);
    shadowBatchCountPerCascade_.fill(0);
    cullingStats_ = {};
    shadowCullingStats_ = {};
    gpuCulling_.resetFrameCounters(frameIndex, 0);
}

bool Renderer::updateAnimatedTransforms(float elapsedSeconds)
{
    bool animatedTransformUpdated = false;
    for (size_t objectIndex = 0; objectIndex < renderObjects_.size(); ++objectIndex) {
        renderer::RenderObject& object = renderObjects_[objectIndex];
        if (!object.animateTransform) {
            continue;
        }

        animatedTransformUpdated = true;
        switch (objectIndex) {
        case 0:
            object.transform.rotationRadians = {0.2f, elapsedSeconds, 0.0f};
            break;
        case 1:
            object.transform.rotationRadians = {elapsedSeconds * 1.15f, 0.35f, 0.2f};
            break;
        case 2:
            object.transform.rotationRadians = {0.25f, -0.35f, elapsedSeconds * 0.9f};
            break;
        case 3:
            object.transform.rotationRadians = {elapsedSeconds * 0.35f, elapsedSeconds * 0.55f, 0.45f};
            break;
        default:
            object.transform.rotationRadians = {elapsedSeconds * (0.2f + 0.05f * static_cast<float>(objectIndex)),
                                                elapsedSeconds * 0.4f,
                                                elapsedSeconds * 0.3f};
            break;
        }
    }
    return animatedTransformUpdated;
}

void Renderer::resetGpuCullFrameCounters(uint32_t frameIndex)
{
    gpuCulling_.resetFrameCounters(
        frameIndex, static_cast<uint32_t>(std::min(allDrawItems_.size(), static_cast<size_t>(kMaxDrawItems))));
}

void Renderer::buildShadowFrameData(uint32_t frameIndex)
{
    const uint32_t cascadeCount = activeCascadeCount();
    shadowDrawItems_.clear();
    shadowMeshDrawBatches_.clear();
    gpuShadowMeshDrawBatches_.clear();
    shadowCullingStats_ = {};
    shadowCullingStats_.cascadeCount = cascadeCount;
    shadowCullingStats_.totalDrawItems = allDrawItems_.size() * cascadeCount;
    shadowVisibleDrawItemsPerCascade_.fill(0);
    shadowBatchCountPerCascade_.fill(0);
    for (std::vector<DrawItem>& cascadeDrawItems : shadowCascadeDrawItems_) {
        cascadeDrawItems.clear();
    }
    for (std::vector<MeshDrawBatch>& cascadeBatches : shadowCascadeMeshDrawBatches_) {
        cascadeBatches.clear();
    }

    // Cascades are independent (each writes only its own draw-item/batch slot),
    // so they run as parallel jobs; the stats reduction happens after the join.
    if (parallelFramePrepEnabled_) {
        jobSystem_.parallelFor(cascadeCount, 1, [this](size_t begin, size_t end) {
            for (size_t cascadeIndex = begin; cascadeIndex < end; ++cascadeIndex) {
                buildShadowDrawItems(static_cast<uint32_t>(cascadeIndex),
                                     frameCascades_[cascadeIndex].lightFrustum);
                buildMeshDrawBatchesForItems(shadowCascadeDrawItems_[cascadeIndex],
                                             shadowCascadeMeshDrawBatches_[cascadeIndex]);
            }
        });
    } else {
        for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
            buildShadowDrawItems(cascadeIndex, frameCascades_[cascadeIndex].lightFrustum);
            buildMeshDrawBatchesForItems(shadowCascadeDrawItems_[cascadeIndex],
                                         shadowCascadeMeshDrawBatches_[cascadeIndex]);
        }
    }
    // A layered pass draws one list for every cascade, so the CPU-culling
    // fallback needs the union of the cascade frusta rather than a list each.
    if (isLayeredCascadeRenderingActive() && !isGpuShadowCullingActive()) {
        buildUnionShadowDrawItems(cascadeCount);
        buildMeshDrawBatchesForItems(shadowDrawItems_, shadowMeshDrawBatches_);
    }

    for (uint32_t cascadeIndex = 0; cascadeIndex < cascadeCount; ++cascadeIndex) {
        shadowVisibleDrawItemsPerCascade_[cascadeIndex] =
            static_cast<uint32_t>(shadowCascadeDrawItems_[cascadeIndex].size());
        shadowBatchCountPerCascade_[cascadeIndex] =
            static_cast<uint32_t>(shadowCascadeMeshDrawBatches_[cascadeIndex].size());
        shadowCullingStats_.visibleDrawItems += shadowCascadeDrawItems_[cascadeIndex].size();
        shadowCullingStats_.batchCount += shadowCascadeMeshDrawBatches_[cascadeIndex].size();
    }
    shadowCullingStats_.culledDrawItems =
        shadowCullingStats_.totalDrawItems > shadowCullingStats_.visibleDrawItems
            ? shadowCullingStats_.totalDrawItems - shadowCullingStats_.visibleDrawItems
            : 0;

    const bool gpuShadowCullingActive = isGpuShadowCullingActive();
    if (gpuShadowCullingActive) {
        buildMeshDrawBatchesForItems(allDrawItems_, gpuShadowMeshDrawBatches_);

        bool shadowIndirectCountPathActive = isShadowIndirectCountSupported();
        if (shadowIndirectCountPathActive) {
            const uint32_t maxDrawIndirectCount = context_.device().maxDrawIndirectCount();
            for (const MeshDrawBatch& batch : gpuShadowMeshDrawBatches_) {
                if (batch.drawItemCount > maxDrawIndirectCount) {
                    shadowIndirectCountPathActive = false;
                    break;
                }
            }
        }

        gpuCulling_.setShadowCullFrameInfo(
            frameIndex,
            static_cast<uint32_t>(
                std::min(allDrawItems_.size() * cascadeCount, static_cast<size_t>(kMaxShadowCullStatsDrawItems))),
            static_cast<uint32_t>(gpuShadowMeshDrawBatches_.size() * cascadeCount),
            shadowIndirectCountPathActive);

        shadowCullingStats_.gpuCulling = true;
        shadowCullingStats_.indirectDrawing = true;
        updateGpuShadowCullInputBuffer(frameIndex);
    } else {
        shadowCullingStats_.indirectDrawing = false;
    }
}

void Renderer::buildMainCullingFrameData(uint32_t frameIndex, const renderer::Frustum& cameraFrustum)
{
    const bool gpuCullingActive = isGpuCullingActive();
    if (gpuCullingActive) {
        visibleDrawItems_ = allDrawItems_;
        cullingStats_ = {};
        cullingStats_.gpuCulling = true;
        const size_t objectCount = sweptObjectCount();
        for (size_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
            const renderer::RenderObject& object = renderObjects_[objectIndex];
            if (isRenderObjectActive(object) && object.mesh && object.mesh->valid()) {
                ++cullingStats_.totalObjects;
            }
        }
        cullingStats_.totalDrawItems = allDrawItems_.size();
    } else {
        buildVisibleDrawItems(cameraFrustum);
    }
    // allDrawItems_ was already bucket-sorted and depth-sorted in buildDrawItems,
    // so visibleDrawItems_ inherits the final order and this is just a scan.
    updateVisibleBucketRanges();
    buildMeshDrawBatches();

    if (gpuCullingActive) {
        bool indirectCountPathActive = isMainPassIndirectCountSupported();
        if (indirectCountPathActive) {
            const uint32_t maxDrawIndirectCount = context_.device().maxDrawIndirectCount();
            for (const MeshDrawBatch& batch : meshDrawBatches_) {
                if (batch.drawItemCount > maxDrawIndirectCount) {
                    indirectCountPathActive = false;
                    break;
                }
            }
        }

        gpuCulling_.setMainCullFrameInfo(
            frameIndex, static_cast<uint32_t>(meshDrawBatches_.size()), indirectCountPathActive);
        // Two-phase occlusion needs the bindless multi-draw-indirect path (the
        // phase-2 pass replays the batch draws) and a valid previous-frame
        // pyramid. With the indirect-count path phase 2 re-compacts into the
        // batch regions; without it, phase 2 rewrites the fixed per-item slots.
        frameTwoPhaseOcclusionActive_ =
            useTwoPhaseOcclusion_ && isMainPassMultiDrawIndirectActive() && isGpuOcclusionCullingActive();
        updateGpuCullInputBuffer(frameIndex);
    } else {
        frameTwoPhaseOcclusionActive_ = false;
        updateIndirectDrawBuffer(frameIndex);
    }
}

// Feeds the yield controller the completed frame's occlusion result so it can
// decide whether the next pyramid build is worth doing.
//
// Call site matters: this has to run after the fence wait and BEFORE
// resetGpuCullFrameCounters, which zeroes the readback-ready flag for the slot
// about to be reused. Called from frame prep instead, readMainCounters always
// returns false and the controller sits in Active forever with zero==0.
void Renderer::updateOcclusionYield(uint32_t frameIndex)
{
    if (!useAdaptiveOcclusion_) {
        // Held active so that turning the setting back on does not inherit a
        // suspension decided while nobody was acting on it.
        occlusionYield_.reset();
        return;
    }

    // Whether occlusion ran in the frame that WROTE these counters, which is two
    // frames back, not whether it is enabled now. Reading the live state instead
    // is the bug this replaced: a probe frame saw occlusionTestRan == true but
    // counters from the suspended frame that last used the slot, so every probe
    // read a yield of zero and re-suspended, and the controller could never
    // recover once suspended.
    const bool occlusionTestRan =
        frameIndex < frameOcclusionTested_.size() && frameOcclusionTested_[frameIndex] != 0u;
    renderer::GpuCullCounters counters{};
    if (!occlusionTestRan || !readGpuCullCounters(frameIndex, counters)) {
        occlusionYield_.update(0, /*occlusionTestRan=*/false);
        return;
    }

    // Phase-2 rescues were occluded in phase 1 and drawn anyway, so they are not
    // yield -- counting them would keep a scene alive on work it did not save.
    const uint32_t culled = counters.occlusionCulledDrawItems > counters.phase2RescuedDrawItems
                                ? counters.occlusionCulledDrawItems - counters.phase2RescuedDrawItems
                                : 0;
    occlusionYield_.update(culled, /*occlusionTestRan=*/true);
}

void Renderer::uploadObjectFrameData(uint32_t frameIndex)
{
    const uint32_t cascadeCount = activeCascadeCount();
    const size_t objectFrameCount = std::min(allDrawItems_.size(), static_cast<size_t>(kMaxDrawItems));
    std::vector<ObjectFrameData> objectFrameData(objectFrameCount);

    // Per-item fill is the heaviest CPU loop of the frame (six mat4 multiplies
    // per draw item); every iteration writes only objectFrameData[drawIndex] and
    // reads shared frame state, so it chunks cleanly across the JobSystem.
    framePrepParallelFor(objectFrameCount, [&](size_t begin, size_t end) {
        for (size_t drawIndex = begin; drawIndex < end; ++drawIndex) {
            const DrawItem& drawItem = allDrawItems_[drawIndex];
            if (drawItem.objectIndex >= renderObjects_.size()) {
                continue;
            }

            const renderer::RenderObject& object = renderObjects_[drawItem.objectIndex];
            if (!object.mesh) {
                continue;
            }

            const glm::mat4 model = object.transform.modelMatrix();
            ObjectFrameData& frameData = objectFrameData[drawIndex];
            frameData.model = model;
            frameData.prevMvpNoJitter =
                previousFrameViewProjection_ * (object.previousModelValid ? object.previousModelMatrix : model);
            const renderer::Material* material = drawItem.material ? drawItem.material : object.material;
            if (material) {
                frameData.baseColorFactor = material->baseColorFactor;
                frameData.materialParams = {material->metallic,
                                            material->roughness,
                                            material->multiScatterStrength,
                                            material->alphaTestCutoff()};
                frameData.textureIndices = {material->baseColorTextureIndex,
                                            material->normalTextureIndex,
                                            material->metallicRoughnessTextureIndex,
                                            material->emissiveTextureIndex};
                frameData.emissiveFactor =
                    glm::vec4(material->emissiveFactor, material->hasEmissiveTexture ? 1.0f : 0.0f);
            }
        }
    });

    uploadFrameConstants(frameIndex, cascadeCount);

    frameObjectDataBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const ObjectFrameData>(objectFrameData.data(), objectFrameData.size())));

    // The skinned demo mesh isn't a RenderObject; give it its own ObjectFrameData
    // in the reserved last slot (same lighting/camera state as the scene draws).
    if (skinnedMesh_.valid()) {
        const glm::mat4 model = skinnedMesh_.modelMatrix();
        ObjectFrameData skinnedData{};
        skinnedData.model = model;
        skinnedData.prevMvpNoJitter =
            previousFrameViewProjection_ * (previousSkinnedModelValid_ ? previousSkinnedModelMatrix_ : model);
        skinnedData.baseColorFactor = glm::vec4(0.85f, 0.45f, 0.32f, 1.0f);
        skinnedData.materialParams = {0.1f, 0.55f, 1.0f, renderer::kNoAlphaTestCutoff};
        skinnedData.textureIndices = {bindlessBaseColorFallbackIndex_,
                                      bindlessNormalFallbackIndex_,
                                      bindlessMetallicRoughnessFallbackIndex_,
                                      0};
        frameObjectDataBuffers_.at(frameIndex)
            .upload(std::as_bytes(std::span<const ObjectFrameData>(&skinnedData, 1)),
                    static_cast<VkDeviceSize>(kSkinnedObjectFrameSlot) * sizeof(ObjectFrameData));
    }
}

// The values every draw item shares. Written once per frame instead of into all
// N object records, which is where 112 of the old 688-byte record went.
void Renderer::uploadFrameConstants(uint32_t frameIndex, uint32_t cascadeCount)
{
    if (frameIndex >= frameConstantsBuffers_.size()) {
        return;
    }

    FrameConstants constants{};
    constants.jitteredViewProjection = frameJitteredViewProjection_;
    constants.viewProjection = frameViewProjection_;
    for (uint32_t cascade = 0; cascade < kMaxShadowCascades; ++cascade) {
        constants.cascadeViewProjection[cascade] = frameCascades_[cascade].lightViewProjection;
    }
    constants.lightDirection = activeDirectionalLightDirection();
    constants.lightColor = activeDirectionalLightColor();
    constants.ambientColor = portfolioCaptureMode_ ? kPortfolioAmbientLightColor : kAmbientLightColor;
    constants.cascadeSplits = frameCascadeSplits_;
    constants.shadowSettings = {csmSettings_.depthBiasConstant,
                                csmSettings_.depthBiasSlope,
                                shadowSettings_.enablePcf ? 1.0f : 0.0f,
                                static_cast<float>(std::max(shadowSettings_.pcfRadius, 0))};
    constants.shadowQuality = {std::max(csmSettings_.normalBias, 0.0f),
                               std::clamp(csmSettings_.cascadeBlend, 0.0f, 0.5f),
                               0.0f,
                               0.0f};
    constants.cameraPosition =
        glm::vec4(camera_.position, csmSettings_.enableCascadeDebugColors ? 1.0f : 0.0f);
    constants.cameraForward =
        glm::vec4(glm::normalize(camera_.target - camera_.position), static_cast<float>(cascadeCount));

    // Virtual shadow maps. The enable flag is the only thing the shader branches
    // on, so leaving the rest at zero when the subsystem is off is safe -- and
    // the page-pool descriptor stays bound either way, because an unwritten
    // descriptor is a validation error rather than a no-op.
    if (isVsmDirectionalShadowActive()) {
        const renderer::VsmClipmapSettings clipmap = vsmClipmapSettings();
        const glm::mat4 lightView = renderer::vsmLightView(directionalLightSettings_.direction);
        const glm::vec2 cameraLightSpaceXy = glm::vec2(lightView * glm::vec4(frameCameraPosition_, 1.0f));
        const VkDeviceAddress pageTableAddress = virtualShadowMap_.pageTableAddress(frameIndex);

        constants.vsmLightView = lightView;
        constants.vsmPageTable = glm::uvec4(static_cast<uint32_t>(pageTableAddress & 0xFFFFFFFFull),
                                            static_cast<uint32_t>(pageTableAddress >> 32),
                                            clipmap.levelCount,
                                            // Flag word, not a bool: bit 0 gates the whole path,
                                            // bit 1 asks the sampler to report which clipmap level
                                            // it used so the debug view can tint by it. Mirrored by
                                            // kVsmFlag* in virtual_shadow_map.glsl.
                                            (pageTableAddress != 0 ? 1u : 0u) |
                                                (vsmSettings_.debugLevelColors ? 2u : 0u));
        constants.vsmParams = glm::vec4(clipmap.level0Extent,
                                        clipmap.texelsPerPixel,
                                        clipmap.depthRange,
                                        1.0f / static_cast<float>(renderer::kVsmPagePoolSize));
        // The normal offset is in world units here, not a fraction of a cascade
        // extent: a clipmap level's texel size varies by a factor of 2^levels, so
        // there is no single extent to be a fraction of. Scaled by the finest
        // texel so the default reads the same way the cascade setting does.
        constants.vsmCamera = glm::vec4(cameraLightSpaceXy.x,
                                        cameraLightSpaceXy.y,
                                        std::max(csmSettings_.normalBias, 0.0f) *
                                            renderer::vsmTexelWorldSize(clipmap, 0) *
                                            static_cast<float>(renderer::kVsmPageSize),
                                        // In texels of the sampled level; the shader turns it into
                                        // world units there, because the level is not known until
                                        // the page walk finds one. This used to pass the cascades'
                                        // depthBiasConstant straight through, which is a fraction
                                        // of a cascade's ortho depth and means something ~250x
                                        // larger against a page's 2*depthRange axis.
                                        std::max(vsmSettings_.depthBiasTexels, 0.0f));
    }

    frameConstantsBuffers_.at(frameIndex)
        .upload(std::as_bytes(std::span<const FrameConstants>(&constants, 1)));
}

} // namespace ve
