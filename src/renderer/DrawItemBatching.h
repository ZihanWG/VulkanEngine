#pragma once

// Draw-item bucketing and mesh batching, with no Vulkan in sight.
//
// This is the arithmetic that decides where every indirect draw command lands.
// `compactedCommandOffset` and `visibleCountOffset` are consumed by the cull
// shader, by `vkCmdDrawIndexedIndirect`, and by the per-batch visible-count
// readback; if any of them disagrees by one, the GPU draws the wrong commands
// and nothing reports an error. It lived inside Renderer, where it could not be
// tested, which is a poor place for arithmetic whose failure mode is silence.
//
// Mesh and Material appear only as opaque pointers -- batching cares whether two
// items share a mesh, never what a mesh is -- so tests can pass arbitrary
// non-null addresses. Same reasoning as renderer/CascadeMath.h and
// renderer/ClusterGrid.h: the logic is pure, so it lives in VulkanEngineCore and
// the renderer aliases it.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ve::renderer {

class Mesh;
struct Material;

// Which pass and pipeline a draw item belongs to. The numeric order is also the
// render order: opaque and mask share the main pipeline (the alpha cutoff in
// ObjectFrameData::materialParams.w makes the test data-driven) while blend needs
// its own pass, sort order and blend state.
enum class RenderBucket : uint8_t {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
};

inline constexpr size_t kRenderBucketCount = 3;

// Half-open [begin, end) range of draw items belonging to one bucket.
struct RenderBucketRange {
    uint32_t begin = 0;
    uint32_t end = 0;

    [[nodiscard]] uint32_t count() const { return end - begin; }
    [[nodiscard]] bool empty() const { return end <= begin; }
};

struct DrawItem {
    const Mesh* mesh = nullptr;
    const Material* material = nullptr;
    uint32_t objectIndex = 0;
    uint32_t submeshIndex = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    int32_t vertexOffset = 0;
    uint32_t frameDataIndex = 0;
    RenderBucket bucket = RenderBucket::Opaque;
    // Material::doubleSided, carried here so batching can break on it. A batch
    // binds one pipeline and the cull mode lives in pipeline state, so a batch
    // straddling this would draw single-sided geometry with the two-sided
    // pipeline or the reverse -- silently, like every other disagreement in this
    // file.
    bool doubleSided = false;
};

// One indirect draw with one pipeline bound, over a contiguous run of draw items
// that share a mesh and a bucket.
struct MeshDrawBatch {
    const Mesh* mesh = nullptr;
    uint32_t beginDrawItem = 0;
    uint32_t drawItemCount = 0;
    uint32_t compactedCommandOffset = 0;
    uint32_t visibleCountOffset = 0;
    RenderBucket bucket = RenderBucket::Opaque;
    // Which of the two main pipelines this batch binds. Uniform across the batch
    // by construction -- buildMeshDrawBatches breaks a run when it changes.
    bool doubleSided = false;
};

// Group a draw-item list into batches. Appends to `batches` rather than clearing
// it, matching the callers that build several lists into one vector.
//
// A run breaks on a change of mesh, of bucket, OR of doubleSided: a batch is one
// indirect draw with one pipeline bound, so it must not straddle a boundary that
// selects a different pipeline even when consecutive items share a mesh. Items
// with no mesh, or whose frameDataIndex is past `maxDrawItems`, break the run and
// are skipped -- they have no valid object record for the shader to read.
void buildMeshDrawBatches(const std::vector<DrawItem>& drawItems,
                          uint32_t maxDrawItems,
                          std::vector<MeshDrawBatch>& batches);

// Contiguous per-bucket ranges over a draw-item list that is already sorted by
// bucket. Scans runs rather than counting, so a list that is *not* sorted yields
// only the first run of each bucket -- which is the correct way for it to fail:
// visibly, at the boundary, rather than by silently merging two disjoint ranges
// into one that spans the items between them.
void computeVisibleBucketRanges(const std::vector<DrawItem>& drawItems,
                                std::array<RenderBucketRange, kRenderBucketCount>& ranges);

} // namespace ve::renderer
