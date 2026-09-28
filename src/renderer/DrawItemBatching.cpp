#include "renderer/DrawItemBatching.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>

namespace ve::renderer {

namespace {

struct BatchOrderKey {
    RenderBucket bucket = RenderBucket::Opaque;
    bool doubleSided = false;
    const Mesh* mesh = nullptr;
};

BatchOrderKey batchOrderKey(const DrawItem& drawItem)
{
    return {drawItem.bucket, drawItem.doubleSided, drawItem.mesh};
}

bool batchOrderKeyLess(const BatchOrderKey& lhs, const BatchOrderKey& rhs)
{
    if (lhs.bucket != rhs.bucket) {
        return lhs.bucket < rhs.bucket;
    }
    if (lhs.doubleSided != rhs.doubleSided) {
        return static_cast<int>(lhs.doubleSided) < static_cast<int>(rhs.doubleSided);
    }
    return std::less<const Mesh*>{}(lhs.mesh, rhs.mesh);
}

bool sameBatchOrderKey(const BatchOrderKey& lhs, const BatchOrderKey& rhs)
{
    return lhs.bucket == rhs.bucket && lhs.doubleSided == rhs.doubleSided && lhs.mesh == rhs.mesh;
}

} // namespace

bool drawItemBatchOrderLess(const DrawItem& lhs, const DrawItem& rhs)
{
    return batchOrderKeyLess(batchOrderKey(lhs), batchOrderKey(rhs));
}

void sortDrawItemsForBatching(std::vector<DrawItem>& drawItems, std::vector<DrawItem>& scratch)
{
    const size_t itemCount = drawItems.size();
    if (itemCount < 2) {
        return;
    }

    // Distinct keys in first-seen order, and each item's slot among them. Items
    // arrive grouped by object, so consecutive items usually share a key and the
    // previous slot is checked before the linear search.
    std::array<BatchOrderKey, kMaxCountingSortKeys> keys{};
    size_t keyCount = 0;
    std::vector<uint8_t> itemKey(itemCount);
    static_assert(kMaxCountingSortKeys <= 256, "itemKey stores a key slot in one byte.");
    size_t lastKey = 0;
    for (size_t itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        const BatchOrderKey key = batchOrderKey(drawItems[itemIndex]);
        if (keyCount == 0 || !sameBatchOrderKey(keys[lastKey], key)) {
            size_t found = 0;
            while (found < keyCount && !sameBatchOrderKey(keys[found], key)) {
                ++found;
            }
            if (found == keyCount) {
                if (keyCount == kMaxCountingSortKeys) {
                    std::stable_sort(drawItems.begin(), drawItems.end(), drawItemBatchOrderLess);
                    return;
                }
                keys[keyCount++] = key;
            }
            lastKey = found;
        }
        itemKey[itemIndex] = static_cast<uint8_t>(lastKey);
    }

    // Keys are distinct, so sorting them is a total order; rank maps a key slot
    // to where its items start in the output.
    std::array<uint8_t, kMaxCountingSortKeys> order{};
    for (size_t slot = 0; slot < keyCount; ++slot) {
        order[slot] = static_cast<uint8_t>(slot);
    }
    std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keyCount), [&](uint8_t lhs, uint8_t rhs) {
        return batchOrderKeyLess(keys[lhs], keys[rhs]);
    });

    std::array<size_t, kMaxCountingSortKeys> counts{};
    for (size_t itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        ++counts[itemKey[itemIndex]];
    }
    std::array<size_t, kMaxCountingSortKeys> offsets{};
    size_t next = 0;
    for (size_t rank = 0; rank < keyCount; ++rank) {
        offsets[order[rank]] = next;
        next += counts[order[rank]];
    }

    // Placing items in input order within each key is what makes this stable.
    scratch.resize(itemCount);
    for (size_t itemIndex = 0; itemIndex < itemCount; ++itemIndex) {
        scratch[offsets[itemKey[itemIndex]]++] = drawItems[itemIndex];
    }
    drawItems.swap(scratch);
}

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
