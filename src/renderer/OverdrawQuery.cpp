#include "renderer/OverdrawQuery.h"

#include "core/Logger.h"
#include "rhi/VulkanCommon.h"
#include "rhi/VulkanContext.h"

#include <utility>

namespace ve::renderer {

namespace {

// One statistic, so one value per query. Paired with the availability word,
// which is what makes a non-blocking read possible: without it a not-yet-ready
// query is indistinguishable from one that legitimately counted zero.
struct StatisticsResult {
    uint64_t fragmentInvocations;
    uint64_t available;
};

} // namespace

OverdrawQuery::~OverdrawQuery()
{
    shutdown();
}

void OverdrawQuery::initialize(const rhi::VulkanContext& context, uint32_t framesInFlight)
{
    shutdown();

    device_ = context.vkDevice();
    unavailableReason_.clear();

    if (device_ == VK_NULL_HANDLE) {
        disable("missing Vulkan device.");
        return;
    }
    if (framesInFlight == 0) {
        disable("frames-in-flight count is zero.");
        return;
    }
    if (!context.device().pipelineStatisticsQueryEnabled()) {
        disable("the device does not support pipelineStatisticsQuery.");
        return;
    }

    frames_.resize(framesInFlight);
    for (FrameState& frame : frames_) {
        VkQueryPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        poolInfo.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        poolInfo.queryCount = 1;
        poolInfo.pipelineStatistics = VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;

        const VkResult result = vkCreateQueryPool(device_, &poolInfo, nullptr, &frame.queryPool);
        if (result != VK_SUCCESS) {
            disable(std::string("query pool creation failed: ") + rhi::vkResultToString(result));
            return;
        }
    }

    available_ = true;
}

void OverdrawQuery::shutdown()
{
    for (FrameState& frame : frames_) {
        if (frame.queryPool != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, frame.queryPool, nullptr);
        }
        frame.queryPool = VK_NULL_HANDLE;
    }
    frames_.clear();
    available_ = false;
    device_ = VK_NULL_HANDLE;
}

bool OverdrawQuery::validFrameIndex(uint32_t frameIndex) const
{
    return frameIndex < frames_.size();
}

void OverdrawQuery::disable(std::string reason)
{
    available_ = false;
    unavailableReason_ = std::move(reason);
}

void OverdrawQuery::resetFrame(uint32_t frameIndex, VkCommandBuffer commandBuffer)
{
    if (!available_ || !validFrameIndex(frameIndex) || commandBuffer == VK_NULL_HANDLE) {
        return;
    }

    FrameState& frame = frames_[frameIndex];
    // The whole frame's state, not just the pool: a frame that reset but never
    // recorded must not read back the previous frame's counts through a pool
    // that still holds them.
    frame.recorded = false;
    frame.submitted = false;
    frame.active = false;
    frame.renderedPixels = 0;
    vkCmdResetQueryPool(commandBuffer, frame.queryPool, 0, 1);
}

void OverdrawQuery::begin(uint32_t frameIndex, VkCommandBuffer commandBuffer, uint64_t renderedPixels)
{
    if (!available_ || !validFrameIndex(frameIndex) || commandBuffer == VK_NULL_HANDLE) {
        return;
    }

    FrameState& frame = frames_[frameIndex];
    if (frame.active || frame.recorded) {
        // One scope per frame. Beginning twice would be a validation error, and
        // silently counting the second scope only would be worse than not
        // counting at all.
        return;
    }

    frame.renderedPixels = renderedPixels;
    frame.active = true;
    vkCmdBeginQuery(commandBuffer, frame.queryPool, 0, 0);
}

void OverdrawQuery::end(uint32_t frameIndex, VkCommandBuffer commandBuffer)
{
    if (!available_ || !validFrameIndex(frameIndex) || commandBuffer == VK_NULL_HANDLE) {
        return;
    }

    FrameState& frame = frames_[frameIndex];
    if (!frame.active) {
        return;
    }

    vkCmdEndQuery(commandBuffer, frame.queryPool, 0);
    frame.active = false;
    frame.recorded = true;
}

void OverdrawQuery::markFrameSubmitted(uint32_t frameIndex)
{
    if (!available_ || !validFrameIndex(frameIndex)) {
        return;
    }
    FrameState& frame = frames_[frameIndex];
    // Only a frame that actually recorded the scope has anything to read. A
    // frame where the pass was skipped stays unsubmitted, so readFrame reports
    // "not ready" forever rather than returning a stale or zero count as though
    // it were measured.
    frame.submitted = frame.recorded;
}

bool OverdrawQuery::readFrame(uint32_t frameIndex, FrameResult& result)
{
    result = {};
    if (!available_ || !validFrameIndex(frameIndex)) {
        return false;
    }

    const FrameState& frame = frames_[frameIndex];
    if (!frame.submitted) {
        return false;
    }

    StatisticsResult statistics{};
    const VkResult queryResult = vkGetQueryPoolResults(device_,
                                                       frame.queryPool,
                                                       0,
                                                       1,
                                                       sizeof(statistics),
                                                       &statistics,
                                                       sizeof(statistics),
                                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (queryResult == VK_NOT_READY) {
        return false;
    }
    if (queryResult != VK_SUCCESS) {
        Logger::warn(std::string("Overdraw query readback failed: ") + rhi::vkResultToString(queryResult));
        return false;
    }
    if (statistics.available == 0) {
        return false;
    }

    result.valid = true;
    result.fragmentInvocations = statistics.fragmentInvocations;
    result.renderedPixels = frame.renderedPixels;
    return true;
}

} // namespace ve::renderer
