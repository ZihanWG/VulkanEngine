#include "rhi/VulkanTransientMemoryPool.h"

#include "core/Logger.h"
#include "rhi/VulkanContext.h"

#include <string>

namespace ve::rhi {

VulkanTransientMemoryPool::~VulkanTransientMemoryPool()
{
    reset();
}

bool VulkanTransientMemoryPool::create(VulkanContext& context,
                                       VkDeviceSize sizeBytes,
                                       uint32_t memoryTypeBits,
                                       VkDeviceSize alignment)
{
    reset();

    context_ = &context;

    if (sizeBytes == 0) {
        unavailableReason_ = "nothing to allocate";
        return false;
    }
    if (memoryTypeBits == 0) {
        // No single memory type suits every image. Supported outcome: the caller
        // gives each image its own allocation, as it did before the pool existed.
        unavailableReason_ = "no memory type accepts every transient image";
        return false;
    }

    VkMemoryRequirements requirements{};
    requirements.size = sizeBytes;
    requirements.alignment = alignment == 0 ? 1 : alignment;
    requirements.memoryTypeBits = memoryTypeBits;

    VmaAllocationCreateInfo allocationInfo{};
    // Not VMA_MEMORY_USAGE_AUTO: it infers the memory type from the resource
    // being created and asserts when there is none, which is exactly the case
    // for a raw pool allocation (vk_mem_alloc.h:4053).
    allocationInfo.usage = VMA_MEMORY_USAGE_UNKNOWN;
    allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    // Without CAN_ALIAS, VMA is entitled to assume one resource per allocation.
    allocationInfo.flags = VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;

    VmaAllocationInfo detail{};
    const VkResult result =
        vmaAllocateMemory(context.allocator(), &requirements, &allocationInfo, &allocation_, &detail);
    if (result != VK_SUCCESS) {
        allocation_ = VK_NULL_HANDLE;
        unavailableReason_ = "the driver refused the shared allocation";
        return false;
    }

    size_ = detail.size;
    unavailableReason_ = "";
    return true;
}

void VulkanTransientMemoryPool::reset()
{
    if (allocation_ != VK_NULL_HANDLE && context_ != nullptr) {
        // Images bound into this memory must already be destroyed; freeing under
        // a live image is undefined, and no amount of validation will say so
        // reliably after the fact.
        vmaFreeMemory(context_->allocator(), allocation_);
    }
    allocation_ = VK_NULL_HANDLE;
    size_ = 0;
    unavailableReason_ = "not created";
}

} // namespace ve::rhi
