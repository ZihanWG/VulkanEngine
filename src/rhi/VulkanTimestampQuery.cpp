#include "rhi/VulkanTimestampQuery.h"

#include "core/Logger.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <array>
#include <string>

namespace ve::rhi {

namespace {

constexpr uint32_t kBeginEdge = 0;
constexpr uint32_t kEndEdge = 1;

void warnTimestampProfilingDisabled(const std::string& reason)
{
    Logger::warn("GPU timestamp profiling disabled: " + reason);
}

} // namespace

VulkanTimestampQuery::~VulkanTimestampQuery()
{
    cleanup();
}

void VulkanTimestampQuery::initialize(const VulkanContext& context, uint32_t frameCount)
{
    cleanup();

    device_ = context.vkDevice();
    frameCount_ = frameCount;
    queryCount_ = frameCount_ * kQueriesPerFrame;
    frameHasResults_.assign(frameCount_, false);

    if (device_ == VK_NULL_HANDLE || frameCount_ == 0) {
        warnTimestampProfilingDisabled("missing device or frame slots.");
        cleanup();
        return;
    }
    if (vkCmdWriteTimestamp2 == nullptr) {
        warnTimestampProfilingDisabled("vkCmdWriteTimestamp2 is unavailable.");
        cleanup();
        return;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context.physicalDevice(), &properties);
    timestampPeriod_ = properties.limits.timestampPeriod;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context.physicalDevice(), &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(context.physicalDevice(), &queueFamilyCount, queueFamilies.data());

    const uint32_t graphicsFamily = context.queueFamilies().graphicsFamily.value();
    if (graphicsFamily >= queueFamilies.size()) {
        warnTimestampProfilingDisabled("graphics queue family index is invalid.");
        cleanup();
        return;
    }

    timestampValidBits_ = queueFamilies[graphicsFamily].timestampValidBits;
    if (timestampValidBits_ == 0 || timestampPeriod_ <= 0.0f) {
        warnTimestampProfilingDisabled("graphics queue does not expose timestamp queries.");
        cleanup();
        return;
    }

    VkQueryPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    createInfo.queryCount = queryCount_;

    const VkResult createResult = vkCreateQueryPool(device_, &createInfo, nullptr, &queryPool_);
    if (createResult != VK_SUCCESS) {
        warnTimestampProfilingDisabled(std::string("vkCreateQueryPool failed with ") + vkResultToString(createResult) +
                                       ".");
        cleanup();
        return;
    }

    debug::setObjectName(device_, queryPool_, VK_OBJECT_TYPE_QUERY_POOL, "GpuTimestampQueryPool");
    Logger::info("GPU timestamp profiling enabled.");
}

void VulkanTimestampQuery::cleanup()
{
    if (queryPool_) {
        vkDestroyQueryPool(device_, queryPool_, nullptr);
        queryPool_ = VK_NULL_HANDLE;
    }

    device_ = VK_NULL_HANDLE;
    frameCount_ = 0;
    queryCount_ = 0;
    timestampValidBits_ = 0;
    timestampPeriod_ = 0.0f;
    frameHasResults_.clear();
}

void VulkanTimestampQuery::resetFrame(VkCommandBuffer commandBuffer, uint32_t frameIndex)
{
    if (!available() || commandBuffer == VK_NULL_HANDLE || frameIndex >= frameCount_) {
        return;
    }

    frameHasResults_[frameIndex] = false;
    vkCmdResetQueryPool(commandBuffer, queryPool_, firstQueryForFrame(frameIndex), kQueriesPerFrame);
}

void VulkanTimestampQuery::writeBegin(VkCommandBuffer commandBuffer, uint32_t frameIndex, Timer timer)
{
    if (!available() || commandBuffer == VK_NULL_HANDLE || frameIndex >= frameCount_) {
        return;
    }

    vkCmdWriteTimestamp2(
        commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queryPool_, queryIndex(frameIndex, timer, kBeginEdge));
}

void VulkanTimestampQuery::writeEnd(VkCommandBuffer commandBuffer, uint32_t frameIndex, Timer timer)
{
    if (!available() || commandBuffer == VK_NULL_HANDLE || frameIndex >= frameCount_) {
        return;
    }

    vkCmdWriteTimestamp2(
        commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, queryPool_, queryIndex(frameIndex, timer, kEndEdge));
}

void VulkanTimestampQuery::markFrameSubmitted(uint32_t frameIndex)
{
    if (!available() || frameIndex >= frameHasResults_.size()) {
        return;
    }

    frameHasResults_[frameIndex] = true;
}

bool VulkanTimestampQuery::readFrame(uint32_t frameIndex, Results& results)
{
    results = {};
    if (!available() || frameIndex >= frameCount_ || !frameHasResults_[frameIndex]) {
        return false;
    }

    std::array<uint64_t, kQueriesPerFrame> timestamps{};
    const VkResult result = vkGetQueryPoolResults(device_,
                                                  queryPool_,
                                                  firstQueryForFrame(frameIndex),
                                                  kQueriesPerFrame,
                                                  sizeof(uint64_t) * timestamps.size(),
                                                  timestamps.data(),
                                                  sizeof(uint64_t),
                                                  VK_QUERY_RESULT_64_BIT);
    if (result == VK_NOT_READY) {
        return false;
    }
    if (result != VK_SUCCESS) {
        Logger::warn(std::string("Failed to read GPU timestamp queries: ") + vkResultToString(result));
        return false;
    }

    const auto elapsed = [this, &timestamps, frameIndex](Timer timer) {
        const uint32_t begin = queryIndex(frameIndex, timer, kBeginEdge) - firstQueryForFrame(frameIndex);
        const uint32_t end = queryIndex(frameIndex, timer, kEndEdge) - firstQueryForFrame(frameIndex);
        return elapsedMilliseconds(timestamps[begin], timestamps[end]);
    };

    results.valid = true;
    results.shadowPassMs = elapsed(Timer::ShadowPass);
    results.mainPassMs = elapsed(Timer::MainPass);
    results.skyboxMs = elapsed(Timer::Skybox);
    results.renderObjectsMs = elapsed(Timer::RenderObjects);
    return true;
}

uint32_t VulkanTimestampQuery::firstQueryForFrame(uint32_t frameIndex) const
{
    return frameIndex * kQueriesPerFrame;
}

uint32_t VulkanTimestampQuery::queryIndex(uint32_t frameIndex, Timer timer, uint32_t edge) const
{
    return firstQueryForFrame(frameIndex) + (static_cast<uint32_t>(timer) * kQueriesPerTimer) + edge;
}

double VulkanTimestampQuery::elapsedMilliseconds(uint64_t begin, uint64_t end) const
{
    if (timestampValidBits_ < 64) {
        const uint64_t timestampMask = (uint64_t{1} << timestampValidBits_) - 1;
        begin &= timestampMask;
        end &= timestampMask;
    }

    uint64_t delta = 0;
    if (end >= begin) {
        delta = end - begin;
    } else if (timestampValidBits_ < 64) {
        const uint64_t timestampMask = (uint64_t{1} << timestampValidBits_) - 1;
        delta = timestampMask - begin + end + 1;
    }

    return static_cast<double>(delta) * static_cast<double>(timestampPeriod_) / 1'000'000.0;
}

} // namespace ve::rhi
