#include "rhi/VulkanSyncValidationProbe.h"

#include "rhi/ValidationTally.h"
#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"

#include <string>

namespace ve::rhi {

namespace {

// Large enough that the layer tracks it as an ordinary range and small enough to
// cost nothing. The value written does not matter; only that the two writes
// overlap.
constexpr VkDeviceSize kProbeBufferBytes = 1024;

} // namespace

SyncValidationProbeResult probeSynchronizationValidation(VulkanContext& context,
                                                         const VulkanCommandContext& commandContext)
{
    SyncValidationProbeResult result{};

    const VkDevice device = context.vkDevice();

    VulkanBuffer probeBuffer;
    VulkanBufferCreateInfo bufferInfo{};
    bufferInfo.size = kProbeBufferBytes;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO;
    probeBuffer.createBuffer(context, bufferInfo);

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandContext.commandPool();
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer));

    // Snapshotted around the recording and the submit together. Same thread, and
    // the layer calls the messenger synchronously from inside the vkCmd/vkQueue
    // calls below, so the delta is exactly what this probe caused.
    const uint64_t hazardsBefore = ValidationTally::syncHazardCount();
    const uint64_t errorsBefore = ValidationTally::errorCount();

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    // The hazard: two writes to the same range, nothing ordering them. Both
    // calls are individually legal, which is the whole point -- core validation
    // has no complaint to make here and only synchronization validation can see
    // the mistake.
    vkCmdFillBuffer(commandBuffer, probeBuffer.buffer(), 0, kProbeBufferBytes, 0x11111111u);
    vkCmdFillBuffer(commandBuffer, probeBuffer.buffer(), 0, kProbeBufferBytes, 0x22222222u);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = commandBuffer;
    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    VK_CHECK(vkQueueSubmit2(context.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(context.graphicsQueue()));

    result.syncHazardsObserved = ValidationTally::syncHazardCount() - hazardsBefore;
    result.otherErrorsObserved = (ValidationTally::errorCount() - errorsBefore) - result.syncHazardsObserved;
    result.hazardReported = result.syncHazardsObserved > 0;

    vkFreeCommandBuffers(device, commandContext.commandPool(), 1, &commandBuffer);

    result.detail = result.hazardReported
                        ? "two unsynchronized writes to one buffer were reported as a hazard"
                        : "two unsynchronized writes to one buffer went unreported -- nothing is checking ordering";
    return result;
}

} // namespace ve::rhi
