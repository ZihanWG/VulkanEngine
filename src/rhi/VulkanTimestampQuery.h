#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <vector>

namespace ve::rhi {

class VulkanContext;

class VulkanTimestampQuery final {
public:
    enum class Timer : uint32_t {
        ShadowPass = 0,
        MainPass,
        Skybox,
        RenderObjects,
        Count
    };

    struct Results {
        bool valid = false;
        double shadowPassMs = 0.0;
        double mainPassMs = 0.0;
        double skyboxMs = 0.0;
        double renderObjectsMs = 0.0;
    };

    VulkanTimestampQuery() = default;
    ~VulkanTimestampQuery();

    VulkanTimestampQuery(const VulkanTimestampQuery&) = delete;
    VulkanTimestampQuery& operator=(const VulkanTimestampQuery&) = delete;
    VulkanTimestampQuery(VulkanTimestampQuery&&) = delete;
    VulkanTimestampQuery& operator=(VulkanTimestampQuery&&) = delete;

    void initialize(const VulkanContext& context, uint32_t frameCount);
    void cleanup();

    void resetFrame(VkCommandBuffer commandBuffer, uint32_t frameIndex);
    void writeBegin(VkCommandBuffer commandBuffer, uint32_t frameIndex, Timer timer);
    void writeEnd(VkCommandBuffer commandBuffer, uint32_t frameIndex, Timer timer);
    void markFrameSubmitted(uint32_t frameIndex);

    [[nodiscard]] bool readFrame(uint32_t frameIndex, Results& results);
    [[nodiscard]] bool available() const
    {
        return queryPool_ != VK_NULL_HANDLE;
    }

private:
    static constexpr uint32_t kTimerCount = static_cast<uint32_t>(Timer::Count);
    static constexpr uint32_t kQueriesPerTimer = 2;
    static constexpr uint32_t kQueriesPerFrame = kTimerCount * kQueriesPerTimer;

    [[nodiscard]] uint32_t firstQueryForFrame(uint32_t frameIndex) const;
    [[nodiscard]] uint32_t queryIndex(uint32_t frameIndex, Timer timer, uint32_t edge) const;
    [[nodiscard]] double elapsedMilliseconds(uint64_t begin, uint64_t end) const;

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    uint32_t frameCount_ = 0;
    uint32_t queryCount_ = 0;
    uint32_t timestampValidBits_ = 0;
    float timestampPeriod_ = 0.0f;
    std::vector<bool> frameHasResults_;
};

} // namespace ve::rhi
