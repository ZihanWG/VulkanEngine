#include "renderer/GpuProfiler.h"

#include "core/Logger.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ve::renderer {

namespace {

struct TimestampQueryResult {
    uint64_t timestamp = 0;
    uint64_t available = 0;
};

} // namespace

GpuProfiler::~GpuProfiler()
{
    shutdown();
}

void GpuProfiler::initialize(const rhi::VulkanContext& context,
                             uint32_t framesInFlight,
                             uint32_t maxTimestampQueriesPerFrame)
{
    shutdown();

    device_ = context.vkDevice();
    maxTimestampQueriesPerFrame_ = maxTimestampQueriesPerFrame;
    unavailableReason_.clear();

    if (device_ == VK_NULL_HANDLE || context.physicalDevice() == VK_NULL_HANDLE) {
        disable("missing Vulkan device.");
        return;
    }
    if (framesInFlight == 0) {
        disable("frames-in-flight count is zero.");
        return;
    }
    if (maxTimestampQueriesPerFrame_ < 2) {
        disable("timestamp query capacity is too small.");
        return;
    }
    if (vkCmdWriteTimestamp2 == nullptr) {
        disable("vkCmdWriteTimestamp2 is unavailable.");
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
        disable("graphics queue family index is invalid.");
        return;
    }

    timestampValidBits_ = queueFamilies[graphicsFamily].timestampValidBits;
    if (timestampValidBits_ == 0 || timestampPeriod_ <= 0.0f) {
        disable("graphics queue does not support timestamp queries.");
        return;
    }

    frames_.resize(framesInFlight);
    for (uint32_t frameIndex = 0; frameIndex < framesInFlight; ++frameIndex) {
        VkQueryPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        createInfo.queryCount = maxTimestampQueriesPerFrame_;

        const VkResult result = vkCreateQueryPool(device_, &createInfo, nullptr, &frames_[frameIndex].queryPool);
        if (result != VK_SUCCESS) {
            disable(std::string("vkCreateQueryPool failed with ") + rhi::vkResultToString(result) + ".");
            return;
        }

        rhi::debug::setObjectName(device_,
                                  frames_[frameIndex].queryPool,
                                  VK_OBJECT_TYPE_QUERY_POOL,
                                  "GpuProfilerQueryPool" + std::to_string(frameIndex));
    }

    available_ = true;
    enabled_ = true;
    Logger::info("GPU profiler enabled with " + std::to_string(framesInFlight) + " frame query pool(s), " +
                 std::to_string(maxTimestampQueriesPerFrame_) + " timestamp queries per frame, timestampPeriod=" +
                 std::to_string(timestampPeriod_) + " ns.");
}

void GpuProfiler::shutdown()
{
    if (device_ != VK_NULL_HANDLE) {
        for (FrameState& frame : frames_) {
            if (frame.queryPool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device_, frame.queryPool, nullptr);
                frame.queryPool = VK_NULL_HANDLE;
            }
        }
    }

    frames_.clear();
    device_ = VK_NULL_HANDLE;
    maxTimestampQueriesPerFrame_ = 0;
    timestampValidBits_ = 0;
    timestampPeriod_ = 0.0f;
    available_ = false;
    enabled_ = true;
    unavailableReason_.clear();
}

void GpuProfiler::beginFrame(uint32_t frameIndex, VkCommandBuffer commandBuffer)
{
    if (!available_ || !enabled_ || commandBuffer == VK_NULL_HANDLE || !validFrameIndex(frameIndex)) {
        return;
    }

    FrameState& frame = frames_[frameIndex];
    frame.scopes.clear();
    frame.activeScopeStack.clear();
    frame.nextQuery = 0;
    frame.submittedQueryCount = 0;
    frame.frameBeginQuery = 0;
    frame.frameEndQuery = 0;
    frame.frameActive = true;
    frame.frameSubmitted = false;
    frame.frameQueriesWritten = false;
    frame.queryLimitExceeded = false;

    vkCmdResetQueryPool(commandBuffer, frame.queryPool, 0, maxTimestampQueriesPerFrame_);

    uint32_t frameQueries = 0;
    if (!allocateQueries(frame, 2, frameQueries)) {
        return;
    }

    frame.frameBeginQuery = frameQueries;
    frame.frameEndQuery = frameQueries + 1;
    frame.frameQueriesWritten = true;
    writeTimestamp(commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame.queryPool, frame.frameBeginQuery);
}

void GpuProfiler::endFrame(uint32_t frameIndex, VkCommandBuffer commandBuffer)
{
    if (!available_ || !enabled_ || commandBuffer == VK_NULL_HANDLE || !validFrameIndex(frameIndex)) {
        return;
    }

    FrameState& frame = frames_[frameIndex];
    if (!frame.frameActive) {
        return;
    }

    while (!frame.activeScopeStack.empty()) {
        endScope(frameIndex, commandBuffer);
    }

    if (frame.frameQueriesWritten) {
        writeTimestamp(commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame.queryPool, frame.frameEndQuery);
    }

    frame.submittedQueryCount = frame.nextQuery;
    frame.frameActive = false;
}

bool GpuProfiler::beginScope(uint32_t frameIndex, VkCommandBuffer commandBuffer, std::string_view name)
{
    if (!available_ || !enabled_ || commandBuffer == VK_NULL_HANDLE || name.empty() || !validFrameIndex(frameIndex)) {
        return false;
    }

    FrameState& frame = frames_[frameIndex];
    if (!frame.frameActive) {
        return false;
    }

    uint32_t firstQuery = 0;
    if (!allocateQueries(frame, 2, firstQuery)) {
        return false;
    }

    ScopeRecord scope{};
    scope.name = std::string(name);
    scope.beginQuery = firstQuery;
    scope.endQuery = firstQuery + 1;

    frame.scopes.push_back(std::move(scope));
    frame.activeScopeStack.push_back(frame.scopes.size() - 1);
    writeTimestamp(commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame.queryPool, firstQuery);
    return true;
}

void GpuProfiler::endScope(uint32_t frameIndex, VkCommandBuffer commandBuffer)
{
    if (!available_ || !enabled_ || commandBuffer == VK_NULL_HANDLE || !validFrameIndex(frameIndex)) {
        return;
    }

    FrameState& frame = frames_[frameIndex];
    if (!frame.frameActive || frame.activeScopeStack.empty()) {
        return;
    }

    const size_t scopeIndex = frame.activeScopeStack.back();
    frame.activeScopeStack.pop_back();
    if (scopeIndex >= frame.scopes.size()) {
        return;
    }

    ScopeRecord& scope = frame.scopes[scopeIndex];
    writeTimestamp(commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame.queryPool, scope.endQuery);
    scope.endWritten = true;
}

void GpuProfiler::markFrameSubmitted(uint32_t frameIndex)
{
    if (!available_ || !enabled_ || !validFrameIndex(frameIndex)) {
        return;
    }

    FrameState& frame = frames_[frameIndex];
    frame.frameSubmitted = frame.submittedQueryCount > 0;
}

bool GpuProfiler::readFrame(uint32_t frameIndex, FrameResults& results)
{
    results = {};
    if (!available_ || !validFrameIndex(frameIndex)) {
        return false;
    }

    const FrameState& frame = frames_[frameIndex];
    if (!enabled_ || !frame.frameSubmitted || frame.submittedQueryCount == 0) {
        return false;
    }

    std::vector<TimestampQueryResult> queryResults(frame.submittedQueryCount);
    const VkResult result = vkGetQueryPoolResults(device_,
                                                  frame.queryPool,
                                                  0,
                                                  frame.submittedQueryCount,
                                                  sizeof(TimestampQueryResult) * queryResults.size(),
                                                  queryResults.data(),
                                                  sizeof(TimestampQueryResult),
                                                  VK_QUERY_RESULT_64_BIT |
                                                      VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result == VK_NOT_READY) {
        return false;
    }
    if (result != VK_SUCCESS) {
        Logger::warn(std::string("GPU profiler query readback failed: ") + rhi::vkResultToString(result));
        return false;
    }

    const auto queryAvailable = [&queryResults](uint32_t query) {
        return query < queryResults.size() && queryResults[query].available != 0;
    };
    const auto elapsed = [this, &queryResults](uint32_t beginQuery, uint32_t endQuery) {
        return elapsedMilliseconds(queryResults[beginQuery].timestamp, queryResults[endQuery].timestamp);
    };

    if (!frame.frameQueriesWritten || !queryAvailable(frame.frameBeginQuery) || !queryAvailable(frame.frameEndQuery)) {
        return false;
    }
    results.totalGpuTimeMs = elapsed(frame.frameBeginQuery, frame.frameEndQuery);

    results.scopes.reserve(frame.scopes.size());
    for (const ScopeRecord& scope : frame.scopes) {
        if (!scope.endWritten || !queryAvailable(scope.beginQuery) || !queryAvailable(scope.endQuery)) {
            continue;
        }

        results.scopes.push_back(ScopeResult{
            scope.name,
            elapsed(scope.beginQuery, scope.endQuery),
            scope.beginQuery,
            scope.endQuery,
        });
    }

    results.valid = true;
    results.queryCount = frame.submittedQueryCount;
    results.maxQueryCount = maxTimestampQueriesPerFrame_;
    results.queryLimitExceeded = frame.queryLimitExceeded;
    return true;
}

bool GpuProfiler::validFrameIndex(uint32_t frameIndex) const
{
    return frameIndex < frames_.size();
}

bool GpuProfiler::allocateQueries(FrameState& frame, uint32_t count, uint32_t& firstQuery)
{
    if (count == 0) {
        firstQuery = frame.nextQuery;
        return true;
    }
    if (frame.nextQuery > maxTimestampQueriesPerFrame_ || count > maxTimestampQueriesPerFrame_ - frame.nextQuery) {
        frame.queryLimitExceeded = true;
        firstQuery = 0;
        return false;
    }

    firstQuery = frame.nextQuery;
    frame.nextQuery += count;
    return true;
}

void GpuProfiler::writeTimestamp(VkCommandBuffer commandBuffer,
                                 VkPipelineStageFlags2 stage,
                                 VkQueryPool queryPool,
                                 uint32_t query)
{
    vkCmdWriteTimestamp2(commandBuffer, stage, queryPool, query);
}

double GpuProfiler::elapsedMilliseconds(uint64_t begin, uint64_t end) const
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

void GpuProfiler::disable(std::string reason)
{
    unavailableReason_ = std::move(reason);
    Logger::warn("GPU profiler unavailable: " + unavailableReason_);

    if (device_ != VK_NULL_HANDLE) {
        for (FrameState& frame : frames_) {
            if (frame.queryPool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device_, frame.queryPool, nullptr);
                frame.queryPool = VK_NULL_HANDLE;
            }
        }
    }

    frames_.clear();
    maxTimestampQueriesPerFrame_ = 0;
    timestampValidBits_ = 0;
    timestampPeriod_ = 0.0f;
    available_ = false;
    enabled_ = false;
}

GpuProfileScope::GpuProfileScope(GpuProfiler& profiler,
                                 uint32_t frameIndex,
                                 VkCommandBuffer commandBuffer,
                                 std::string_view name)
    : profiler_(&profiler),
      commandBuffer_(commandBuffer),
      frameIndex_(frameIndex),
      active_(profiler.beginScope(frameIndex, commandBuffer, name))
{}

GpuProfileScope::~GpuProfileScope()
{
    if (active_ && profiler_) {
        profiler_->endScope(frameIndex_, commandBuffer_);
    }
}

} // namespace ve::renderer
