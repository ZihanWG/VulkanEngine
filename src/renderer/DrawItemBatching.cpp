#include "renderer/DrawItemBatching.h"

namespace ve::renderer {

void buildMeshDrawBatches(const std::vector<DrawItem>& drawItems,
                          uint32_t maxDrawItems,
                          std::vector<MeshDrawBatch>& batches)
{
    const Mesh* currentMesh = nullptr;
    RenderBucket currentBucket = RenderBucket::Opaque;
    bool currentDoubleSided = false;
    // An index, not a pointer into `batches`: push_back may reallocate, and a
    // retained pointer only survives because it is reassigned immediately after
    // every push. That is one edit away from dangling and no sanitizer would
    // catch the reordering that breaks it.
    size_t currentBatch = batches.size();
    bool haveBatch = false;

    for (size_t drawItemIndex = 0; drawItemIndex < drawItems.size(); ++drawItemIndex) {
        const DrawItem& drawItem = drawItems[drawItemIndex];
        if (drawItem.mesh == nullptr || drawItem.frameDataIndex >= maxDrawItems) {
            currentMesh = nullptr;
            haveBatch = false;
            continue;
        }

        if (!haveBatch || currentMesh != drawItem.mesh || currentBucket != drawItem.bucket ||
            currentDoubleSided != drawItem.doubleSided) {
            MeshDrawBatch batch{};
            batch.mesh = drawItem.mesh;
            batch.beginDrawItem = static_cast<uint32_t>(drawItemIndex);
            batch.compactedCommandOffset = static_cast<uint32_t>(drawItemIndex);
            batch.visibleCountOffset = static_cast<uint32_t>(batches.size() * sizeof(uint32_t));
            batch.bucket = drawItem.bucket;
            batch.doubleSided = drawItem.doubleSided;
            currentBatch = batches.size();
            batches.push_back(batch);
            haveBatch = true;
            currentMesh = drawItem.mesh;
            currentBucket = drawItem.bucket;
            currentDoubleSided = drawItem.doubleSided;
        }

        ++batches[currentBatch].drawItemCount;
    }
}

void computeVisibleBucketRanges(const std::vector<DrawItem>& drawItems,
                                std::array<RenderBucketRange, kRenderBucketCount>& ranges)
{
    // No clearing pass: the loop below assigns every bucket unconditionally, so a
    // reset first would be dead. Mutation-testing the extraction is what showed
    // that -- deleting the reset broke no test, because there was nothing to break.
    size_t index = 0;
    for (size_t bucket = 0; bucket < kRenderBucketCount; ++bucket) {
        const RenderBucket wanted = static_cast<RenderBucket>(bucket);
        const size_t begin = index;
        while (index < drawItems.size() && drawItems[index].bucket == wanted) {
            ++index;
        }
        ranges[bucket] = {static_cast<uint32_t>(begin), static_cast<uint32_t>(index)};
    }
}

} // namespace ve::renderer
