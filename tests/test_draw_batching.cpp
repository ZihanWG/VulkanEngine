#include "renderer/DrawItemBatching.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

using ve::renderer::buildMeshDrawBatches;
using ve::renderer::computeVisibleBucketRanges;
using ve::renderer::DrawItem;
using ve::renderer::kRenderBucketCount;
using ve::renderer::Mesh;
using ve::renderer::MeshDrawBatch;
using ve::renderer::RenderBucket;
using ve::renderer::RenderBucketRange;

namespace {

// Batching only ever compares mesh pointers, so these need to be distinct and
// non-null and nothing else. Never dereferenced.
const Mesh* meshA()
{
    return reinterpret_cast<const Mesh*>(0x1000);
}
const Mesh* meshB()
{
    return reinterpret_cast<const Mesh*>(0x2000);
}

constexpr uint32_t kMaxDrawItems = 8192;

DrawItem item(const Mesh* mesh, RenderBucket bucket = RenderBucket::Opaque, uint32_t frameDataIndex = 0)
{
    DrawItem drawItem{};
    drawItem.mesh = mesh;
    drawItem.bucket = bucket;
    drawItem.frameDataIndex = frameDataIndex;
    return drawItem;
}

std::vector<MeshDrawBatch> batchesFor(const std::vector<DrawItem>& items, uint32_t maxDrawItems = kMaxDrawItems)
{
    std::vector<MeshDrawBatch> batches;
    buildMeshDrawBatches(items, maxDrawItems, batches);
    return batches;
}

} // namespace

TEST_CASE("An empty draw list produces no batches")
{
    CHECK(batchesFor({}).empty());
}

TEST_CASE("Consecutive items sharing a mesh and bucket form one batch")
{
    const std::vector<MeshDrawBatch> batches = batchesFor({item(meshA()), item(meshA()), item(meshA())});

    REQUIRE(batches.size() == 1);
    CHECK(batches[0].mesh == meshA());
    CHECK(batches[0].beginDrawItem == 0);
    CHECK(batches[0].drawItemCount == 3);
}

TEST_CASE("A change of mesh starts a new batch")
{
    const std::vector<MeshDrawBatch> batches = batchesFor({item(meshA()), item(meshB()), item(meshB())});

    REQUIRE(batches.size() == 2);
    CHECK(batches[0].drawItemCount == 1);
    CHECK(batches[1].mesh == meshB());
    CHECK(batches[1].beginDrawItem == 1);
    CHECK(batches[1].drawItemCount == 2);
}

TEST_CASE("A change of bucket starts a new batch even on the same mesh")
{
    // This is the load-bearing one: a batch is a single indirect draw with one
    // pipeline bound, so merging across a bucket boundary would draw masked
    // geometry with the opaque pipeline.
    const std::vector<MeshDrawBatch> batches =
        batchesFor({item(meshA(), RenderBucket::Opaque), item(meshA(), RenderBucket::Mask)});

    REQUIRE(batches.size() == 2);
    CHECK(batches[0].bucket == RenderBucket::Opaque);
    CHECK(batches[1].bucket == RenderBucket::Mask);
    CHECK(batches[1].beginDrawItem == 1);
}

TEST_CASE("compactedCommandOffset is the draw item index the batch starts at")
{
    const std::vector<MeshDrawBatch> batches =
        batchesFor({item(meshA()), item(meshA()), item(meshB()), item(meshA())});

    REQUIRE(batches.size() == 3);
    CHECK(batches[0].compactedCommandOffset == 0);
    CHECK(batches[1].compactedCommandOffset == 2);
    CHECK(batches[2].compactedCommandOffset == 3);
}

TEST_CASE("visibleCountOffset advances one uint32 per batch")
{
    const std::vector<MeshDrawBatch> batches = batchesFor({item(meshA()), item(meshB()), item(meshA())});

    REQUIRE(batches.size() == 3);
    CHECK(batches[0].visibleCountOffset == 0);
    CHECK(batches[1].visibleCountOffset == sizeof(uint32_t));
    CHECK(batches[2].visibleCountOffset == 2 * sizeof(uint32_t));
}

TEST_CASE("visibleCountOffset counts batches already in the output vector")
{
    // Callers build several draw lists into one batch vector, and the count
    // buffer is shared across all of them, so the offset must continue rather
    // than restart.
    std::vector<MeshDrawBatch> batches;
    buildMeshDrawBatches({item(meshA())}, kMaxDrawItems, batches);
    buildMeshDrawBatches({item(meshB())}, kMaxDrawItems, batches);

    REQUIRE(batches.size() == 2);
    CHECK(batches[1].visibleCountOffset == sizeof(uint32_t));
}

TEST_CASE("Items without a mesh are skipped and break the run")
{
    const std::vector<MeshDrawBatch> batches = batchesFor({item(meshA()), item(nullptr), item(meshA())});

    // Same mesh either side, but the gap must not be bridged: the batch covers a
    // contiguous index range, and spanning the hole would pull the mesh-less item
    // into an indirect draw.
    REQUIRE(batches.size() == 2);
    CHECK(batches[0].beginDrawItem == 0);
    CHECK(batches[0].drawItemCount == 1);
    CHECK(batches[1].beginDrawItem == 2);
    CHECK(batches[1].drawItemCount == 1);
}

TEST_CASE("Items past the object-record capacity are skipped and break the run")
{
    const std::vector<MeshDrawBatch> batches =
        batchesFor({item(meshA(), RenderBucket::Opaque, 0), item(meshA(), RenderBucket::Opaque, 16),
                    item(meshA(), RenderBucket::Opaque, 1)},
                   /*maxDrawItems=*/16);

    REQUIRE(batches.size() == 2);
    CHECK(batches[0].drawItemCount == 1);
    CHECK(batches[1].beginDrawItem == 2);
}

TEST_CASE("Every batch's index range stays inside the draw list")
{
    const std::vector<DrawItem> items{item(meshA()), item(meshA()), item(meshB()),
                                      item(meshB(), RenderBucket::Blend), item(meshA())};
    const std::vector<MeshDrawBatch> batches = batchesFor(items);

    uint32_t covered = 0;
    for (const MeshDrawBatch& batch : batches) {
        CHECK(batch.drawItemCount > 0);
        CHECK(batch.beginDrawItem + batch.drawItemCount <= items.size());
        covered += batch.drawItemCount;
    }
    // Nothing drawable was dropped and nothing was counted twice.
    CHECK(covered == items.size());
}

TEST_CASE("Bucket ranges are contiguous and cover a sorted list")
{
    const std::vector<DrawItem> items{item(meshA(), RenderBucket::Opaque), item(meshA(), RenderBucket::Opaque),
                                      item(meshA(), RenderBucket::Mask), item(meshB(), RenderBucket::Blend),
                                      item(meshB(), RenderBucket::Blend)};

    std::array<RenderBucketRange, kRenderBucketCount> ranges{};
    computeVisibleBucketRanges(items, ranges);

    CHECK(ranges[0].begin == 0);
    CHECK(ranges[0].end == 2);
    CHECK(ranges[1].begin == 2);
    CHECK(ranges[1].end == 3);
    CHECK(ranges[2].begin == 3);
    CHECK(ranges[2].end == 5);
    CHECK(ranges[0].count() == 2);
    CHECK_FALSE(ranges[0].empty());
}

TEST_CASE("A bucket with no items yields an empty range that does not disturb the others")
{
    const std::vector<DrawItem> items{item(meshA(), RenderBucket::Opaque), item(meshB(), RenderBucket::Blend)};

    std::array<RenderBucketRange, kRenderBucketCount> ranges{};
    computeVisibleBucketRanges(items, ranges);

    CHECK(ranges[0].end == 1);
    CHECK(ranges[1].empty());
    CHECK(ranges[1].count() == 0);
    CHECK(ranges[2].begin == 1);
    CHECK(ranges[2].end == 2);
}

TEST_CASE("Bucket ranges over an empty list are all empty")
{
    std::array<RenderBucketRange, kRenderBucketCount> ranges{};
    computeVisibleBucketRanges({}, ranges);

    for (const RenderBucketRange& range : ranges) {
        CHECK(range.empty());
    }
}

TEST_CASE("Bucket ranges are reset rather than accumulated across calls")
{
    std::array<RenderBucketRange, kRenderBucketCount> ranges{};
    computeVisibleBucketRanges({item(meshA(), RenderBucket::Opaque), item(meshA(), RenderBucket::Opaque)}, ranges);
    REQUIRE(ranges[0].end == 2);

    computeVisibleBucketRanges({item(meshA(), RenderBucket::Opaque)}, ranges);
    CHECK(ranges[0].end == 1);
}
