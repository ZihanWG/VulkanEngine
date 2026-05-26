#pragma once

#include "rhi/VulkanCommon.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ve::rhi {

class VulkanContext;

} // namespace ve::rhi

namespace ve::renderer {

class GpuProfiler final {
public:
    static constexpr uint32_t kDefaultMaxTimestampQueriesPerFrame = 256;

    struct ScopeResult {
        std::string name;
        double elapsedMs = 0.0;
        uint32_t beginQuery = 0;
        uint32_t endQuery = 0;
    };

    struct FrameResults {
        bool valid = false;
        double totalGpuTimeMs = 0.0;
        uint32_t queryCount = 0;
        uint32_t maxQueryCount = 0;
        bool queryLimitExceeded = false;
        std::vector<ScopeResult> scopes;
    };

    GpuProfiler() = default;
    ~GpuProfiler();

    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;
    GpuProfiler(GpuProfiler&&) = delete;
    GpuProfiler& operator=(GpuProfiler&&) = delete;

    void initialize(const rhi::VulkanContext& context,
                    uint32_t framesInFlight,
                    uint32_t maxTimestampQueriesPerFrame = kDefaultMaxTimestampQueriesPerFrame);
    void shutdown();

    void beginFrame(uint32_t frameIndex, VkCommandBuffer commandBuffer);
    void endFrame(uint32_t frameIndex, VkCommandBuffer commandBuffer);
    bool beginScope(uint32_t frameIndex, VkCommandBuffer commandBuffer, std::string_view name);
    void endScope(uint32_t frameIndex, VkCommandBuffer commandBuffer);
    void markFrameSubmitted(uint32_t frameIndex);
    [[nodiscard]] bool readFrame(uint32_t frameIndex, FrameResults& results);

    void setEnabled(bool enabled) { enabled_ = enabled; }
    [[nodiscard]] bool enabled() const { return enabled_; }
    [[nodiscard]] bool available() const { return available_; }
    [[nodiscard]] const std::string& unavailableReason() const { return unavailableReason_; }
    [[nodiscard]] uint32_t maxTimestampQueriesPerFrame() const { return maxTimestampQueriesPerFrame_; }

private:
    struct ScopeRecord {
        std::string name;
        uint32_t beginQuery = 0;
        uint32_t endQuery = 0;
        bool endWritten = false;
    };

    struct FrameState {
        VkQueryPool queryPool = VK_NULL_HANDLE;
        std::vector<ScopeRecord> scopes;
        std::vector<size_t> activeScopeStack;
        uint32_t nextQuery = 0;
        uint32_t submittedQueryCount = 0;
        uint32_t frameBeginQuery = 0;
        uint32_t frameEndQuery = 0;
        bool frameActive = false;
        bool frameSubmitted = false;
        bool frameQueriesWritten = false;
        bool queryLimitExceeded = false;
    };

    [[nodiscard]] bool validFrameIndex(uint32_t frameIndex) const;
    [[nodiscard]] bool allocateQueries(FrameState& frame, uint32_t count, uint32_t& firstQuery);
    void writeTimestamp(VkCommandBuffer commandBuffer, VkPipelineStageFlags2 stage, VkQueryPool queryPool, uint32_t query);
    [[nodiscard]] double elapsedMilliseconds(uint64_t begin, uint64_t end) const;
    void disable(std::string reason);

    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<FrameState> frames_;
    uint32_t maxTimestampQueriesPerFrame_ = 0;
    uint32_t timestampValidBits_ = 0;
    float timestampPeriod_ = 0.0f;
    bool available_ = false;
    bool enabled_ = true;
    std::string unavailableReason_;
};

class GpuProfileScope final {
public:
    GpuProfileScope(GpuProfiler& profiler,
                    uint32_t frameIndex,
                    VkCommandBuffer commandBuffer,
                    std::string_view name);
    ~GpuProfileScope();

    GpuProfileScope(const GpuProfileScope&) = delete;
    GpuProfileScope& operator=(const GpuProfileScope&) = delete;
    GpuProfileScope(GpuProfileScope&&) = delete;
    GpuProfileScope& operator=(GpuProfileScope&&) = delete;

private:
    GpuProfiler* profiler_ = nullptr;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    uint32_t frameIndex_ = 0;
    bool active_ = false;
};

} // namespace ve::renderer
