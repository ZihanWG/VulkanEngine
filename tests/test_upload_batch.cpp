#include "rhi/VulkanUploadBatch.h"

#include <catch2/catch_test_macros.hpp>

using ve::rhi::kUploadBatchStagingBudgetBytes;
using ve::rhi::uploadBatchShouldFlush;

namespace {

constexpr VkDeviceSize kMiB = 1024ULL * 1024ULL;

} // namespace

TEST_CASE("An empty batch accepts any upload, however large", "[upload-batch]")
{
    // The batch cannot ask a caller to split one staging allocation: an image
    // copy needs its source contiguous. So a lone oversized upload has to go
    // through rather than deadlocking against the budget.
    CHECK_FALSE(uploadBatchShouldFlush(0, 1 * kMiB));
    CHECK_FALSE(uploadBatchShouldFlush(0, kUploadBatchStagingBudgetBytes));
    CHECK_FALSE(uploadBatchShouldFlush(0, kUploadBatchStagingBudgetBytes * 4));
}

TEST_CASE("A batch accumulates until the budget is crossed", "[upload-batch]")
{
    const VkDeviceSize budget = 64 * kMiB;

    CHECK_FALSE(uploadBatchShouldFlush(1 * kMiB, 1 * kMiB, budget));
    CHECK_FALSE(uploadBatchShouldFlush(32 * kMiB, 16 * kMiB, budget));
    // Exactly at the budget still fits; the test is "over", not "at".
    CHECK_FALSE(uploadBatchShouldFlush(48 * kMiB, 16 * kMiB, budget));
    CHECK(uploadBatchShouldFlush(48 * kMiB, 17 * kMiB, budget));
    CHECK(uploadBatchShouldFlush(budget, 1, budget));
}

TEST_CASE("An oversized upload flushes what is already retained first", "[upload-batch]")
{
    const VkDeviceSize budget = 64 * kMiB;

    // Whatever is already staged gets submitted, and the big one then lands in a
    // fresh, empty batch -- where the first case above lets it through.
    CHECK(uploadBatchShouldFlush(1 * kMiB, budget * 2, budget));
    CHECK_FALSE(uploadBatchShouldFlush(0, budget * 2, budget));
}

TEST_CASE("The default budget bounds a production-scale scene", "[upload-batch]")
{
    // Sponza is ~91 MiB of cooked texture staging and ~366 MiB uncompressed.
    // Neither may be retained in one go; the point of the budget is that peak
    // staging stops depending on how big the scene is.
    CHECK(kUploadBatchStagingBudgetBytes < 91 * kMiB);

    const auto submitsFor = [](VkDeviceSize total, VkDeviceSize perUpload) {
        VkDeviceSize retained = 0;
        uint32_t submits = 0;
        for (VkDeviceSize uploaded = 0; uploaded < total; uploaded += perUpload) {
            if (uploadBatchShouldFlush(retained, perUpload)) {
                ++submits;
                retained = 0;
            }
            retained += perUpload;
        }
        return retained > 0 ? submits + 1 : submits;
    };

    // 77 textures averaging ~1.2 MiB cooked: a handful of submits, not 77.
    const uint32_t cookedSubmits = submitsFor(91 * kMiB, 1200 * 1024);
    CHECK(cookedSubmits >= 1);
    CHECK(cookedSubmits <= 4);

    // The uncompressed path is four times the bytes and must still stay bounded.
    const uint32_t uncookedSubmits = submitsFor(366 * kMiB, 5 * kMiB);
    CHECK(uncookedSubmits <= 8);
}
