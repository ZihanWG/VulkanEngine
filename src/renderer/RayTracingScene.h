#pragma once

// The scene as rays see it: one bottom-level acceleration structure per mesh
// primitive at LOD 0, and one top-level structure rebuilt from this frame's
// draw items. Feeds the ray-traced sun shadows in the main pass
// (docs/ray_traced_shadows.md); the GPU-free half is renderer/RayTracedShadows.h.
//
// Ownership and lifetime, which is most of what this class is for:
//
// - BLAS: one per (mesh, primitive), built once by buildBottomLevel after the
//   scene's meshes exist. The renderer's mesh set is fixed after construction --
//   every scene preset draws the built-in cube and sphere, and the sample scene
//   is imported at startup only -- so nothing rebuilds them. A BLAS holds its
//   own copy of the geometry; the mesh buffers are only its build input.
// - TLAS: ONE structure, rebuilt in place by every frame that traces. It is
//   bound once into the bindless environment descriptor set, which is shared by
//   every material and cannot be rewritten while a frame in flight uses it, so a
//   copy per frame slot would need a descriptor per slot. The shadow map works
//   the same way: one image, redrawn every frame, read by the frame before.
// - Instance and geometry-record buffers: one per frame slot, host-visible,
//   written by the CPU after that slot's fence has signalled.
// - Scratch: one buffer, shared by every build on the graphics queue.
//
// Synchronization (the edges, not just the barriers):
// - CPU instance writes -> TLAS build: the slot's fence plus the queue submit's
//   host-write guarantee. No barrier.
// - previous frame's ray queries (fragment, AS read) and previous builds (scratch
//   and TLAS writes) -> this frame's build: recordTopLevelBuild's entry barrier.
//   The render graph rebuilds buffer state every frame, so it cannot order a
//   frame's first write against the frame before; this is the same situation as
//   the probe shading-parameter buffer in docs/render_graph.md.
// - this frame's build -> this frame's ray queries: the render graph, from the
//   AccelerationStructureBuild / AccelerationStructureRead declarations on the
//   TLAS storage buffer.
// - mesh upload (transfer) -> BLAS build: a barrier at the top of the one-time
//   build command buffer, whose first scope covers the earlier upload
//   submissions on the same queue.
// - every BLAS -> the rays: the graph's barrier names only the TLAS buffer, and a
//   ray reads the BLASes it reaches through it. The entry barrier makes the
//   startup builds visible to fragment shaders, and the barrier after the skinned
//   refit does the same for the one BLAS written inside the frame.
// - skinned caster (renderer/RayTracingSkinning.h): previous refit's reads of the
//   positions -> this frame's pre-skin writes, by the entry barrier; pre-skin
//   writes -> refit reads, inside RayTracingSkinnedCaster::record; refit ->
//   TLAS build and rays, by the barrier after it.

#include "renderer/RayTracedShadows.h"
#include "renderer/RayTracingSkinning.h"
#include "rhi/VulkanAccelerationStructure.h"
#include "rhi/VulkanBuffer.h"

#include <glm/mat4x4.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <unordered_map>
#include <vector>

namespace ve::rhi {
class VulkanCommandContext;
class VulkanContext;
} // namespace ve::rhi

namespace ve::renderer {

class Mesh;

// One instance this frame's TLAS should hold.
struct RayTracingInstanceInput {
    const Mesh* mesh = nullptr;
    // The draw item's level-0 index range, which is what names its BLAS.
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    glm::mat4 model{1.0f};
    // Becomes the instance's custom index: the shader reaches this draw item's
    // ObjectFrameData and RayTracingGeometryRecord through it.
    uint32_t frameDataIndex = 0;
    RayTracedCasterKind kind = RayTracedCasterKind::Opaque;
    // The skinned caster instead of a mesh primitive: mesh and index range are
    // ignored, the instance references the skinned BLAS, and this frame's build
    // pre-skins and refits it from jointPalette first.
    bool skinned = false;
    VkDeviceAddress jointPalette = 0;
};

struct RayTracingSceneStats {
    uint32_t bottomLevelCount = 0;
    uint64_t bottomLevelTriangles = 0;
    VkDeviceSize bottomLevelBytes = 0;
    VkDeviceSize topLevelBytes = 0;
    VkDeviceSize scratchBytes = 0;
    double bottomLevelBuildMs = 0.0;
    // The last frame written, not an average.
    uint32_t instances = 0;
    uint32_t alphaTestedInstances = 0;
    // Inputs naming geometry no BLAS was built for. Zero unless the mesh set
    // changed after buildBottomLevel, which nothing in the renderer does.
    uint32_t instancesWithoutBottomLevel = 0;
    // Whether the last frame written held the skinned caster, and how often its
    // BLAS has been built in full and refitted since startup.
    bool skinnedInstance = false;
    uint64_t skinnedBuilds = 0;
    uint64_t skinnedRefits = 0;
};

class RayTracingScene final {
public:
    RayTracingScene() = default;
    ~RayTracingScene();

    RayTracingScene(const RayTracingScene&) = delete;
    RayTracingScene& operator=(const RayTracingScene&) = delete;
    RayTracingScene(RayTracingScene&&) = delete;
    RayTracingScene& operator=(RayTracingScene&&) = delete;

    // The TLAS, its scratch, and the per-frame instance and geometry-record
    // buffers, sized for maxInstances. Nothing is built. Separate from the BLAS
    // build because the TLAS has to exist before the descriptor set that names
    // it, which is before the scene's meshes do. Throws on failure and leaves
    // nothing allocated.
    void createTopLevel(rhi::VulkanContext& context, uint32_t framesInFlight, uint32_t maxInstances);

    // One BLAS per mesh primitive at level 0, then an empty TLAS build, so the
    // descriptor names a built structure from the first frame whether or not
    // anything traces. Startup only: one submission, waited on with a fence.
    void buildBottomLevel(const rhi::VulkanCommandContext& commandContext, std::span<const Mesh* const> meshes);

    // The skinned caster: its positions buffer, its BLAS storage and the pre-skin
    // pipeline, with the shared scratch grown to cover its build. Nothing is
    // built here; the first frame that traces with it builds, later ones refit.
    // Throws on failure, leaving the static scene usable.
    void createSkinned(const RayTracingSkinnedSource& source, const std::filesystem::path& preSkinShaderPath);
    [[nodiscard]] bool hasSkinned() const
    {
        return skinned_.valid();
    }

    void destroy();

    // The TLAS exists, so its descriptor can be written.
    [[nodiscard]] bool hasTopLevel() const
    {
        return topLevel_.valid();
    }
    // The BLAS build has run, so instances can be written and traced.
    [[nodiscard]] bool ready() const
    {
        return ready_;
    }

    // Writes this frame slot's instances and geometry records. Inputs whose
    // geometry has no BLAS are skipped and counted; past maxInstances the rest
    // are dropped and counted the same way. Returns the instance count the next
    // recordTopLevelBuild for this slot will build.
    uint32_t writeInstances(uint32_t frameIndex, std::span<const RayTracingInstanceInput> inputs);

    // The entry barrier, the skinned caster's pre-skin and refit when this slot
    // holds it, then a full TLAS build over the instances last written to it.
    void recordTopLevelBuild(VkCommandBuffer commandBuffer, uint32_t frameIndex);

    [[nodiscard]] VkAccelerationStructureKHR topLevelHandle() const
    {
        return topLevel_.handle();
    }
    [[nodiscard]] VkBuffer topLevelBuffer() const
    {
        return topLevel_.buffer();
    }
    [[nodiscard]] VkDeviceSize topLevelSize() const
    {
        return topLevel_.size();
    }
    [[nodiscard]] VkDeviceAddress geometryTableAddress(uint32_t frameIndex) const;
    [[nodiscard]] const RayTracingSceneStats& stats() const
    {
        return stats_;
    }

private:
    struct BottomLevelKey {
        const Mesh* mesh = nullptr;
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;

        [[nodiscard]] bool operator==(const BottomLevelKey&) const = default;
    };
    struct BottomLevelKeyHash {
        [[nodiscard]] size_t operator()(const BottomLevelKey& key) const;
    };
    struct FrameBuffers {
        rhi::VulkanBuffer instances;
        rhi::VulkanBuffer geometry;
        uint32_t instanceCount = 0;
        // Non-zero when this slot's instances include the skinned caster: the
        // palette its pre-skin reads.
        VkDeviceAddress skinnedPalette = 0;
    };

    // Grows the shared scratch to hold `bytes` past an aligned base. Startup only:
    // it replaces the buffer.
    void ensureScratch(VkDeviceSize bytes);
    [[nodiscard]] VkDeviceAddress scratchBase() const;

    rhi::VulkanContext* context_ = nullptr;
    uint32_t maxInstances_ = 0;
    std::vector<rhi::VulkanAccelerationStructure> bottomLevels_;
    std::unordered_map<BottomLevelKey, uint32_t, BottomLevelKeyHash> bottomLevelIndex_;
    rhi::VulkanAccelerationStructure topLevel_;
    VkDeviceSize topLevelScratchSize_ = 0;
    rhi::VulkanBuffer scratch_;
    std::vector<FrameBuffers> frames_;
    RayTracingSkinnedCaster skinned_;
    RayTracingSceneStats stats_{};
    bool ready_ = false;
};

} // namespace ve::renderer
