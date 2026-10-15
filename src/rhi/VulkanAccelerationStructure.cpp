#include "rhi/VulkanAccelerationStructure.h"

#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ve::rhi {

VulkanAccelerationStructure::~VulkanAccelerationStructure()
{
    reset();
}

VulkanAccelerationStructure::VulkanAccelerationStructure(VulkanAccelerationStructure&& other) noexcept
{
    moveFrom(other);
}

VulkanAccelerationStructure& VulkanAccelerationStructure::operator=(VulkanAccelerationStructure&& other) noexcept
{
    if (this != &other) {
        reset();
        moveFrom(other);
    }
    return *this;
}

void VulkanAccelerationStructure::create(VulkanContext& context,
                                         VkAccelerationStructureTypeKHR type,
                                         VkDeviceSize size,
                                         std::string_view debugName)
{
    reset();

    if (!context.rayQueryEnabled()) {
        throw std::logic_error("Cannot create an acceleration structure on a device without ray query.");
    }
    if (size == 0) {
        throw std::runtime_error("Cannot create a zero-sized acceleration structure.");
    }

    context_ = &context;
    type_ = type;

    try {
        // Device-local and addressable: the build writes it on the GPU, and a TLAS
        // references a BLAS by address, not by handle.
        VulkanBufferCreateInfo storageInfo{};
        storageInfo.size = size;
        storageInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
        storageInfo.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        storageInfo.requestDeviceAddress = true;
        storage_.create(context, storageInfo);

        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = storage_.buffer();
        createInfo.offset = 0;
        createInfo.size = size;
        createInfo.type = type;
        VK_CHECK(vkCreateAccelerationStructureKHR(context.vkDevice(), &createInfo, nullptr, &handle_));

        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = handle_;
        deviceAddress_ = vkGetAccelerationStructureDeviceAddressKHR(context.vkDevice(), &addressInfo);
        if (deviceAddress_ == 0) {
            throw std::runtime_error("Acceleration structure reported a null device address.");
        }
    } catch (...) {
        reset();
        throw;
    }

    if (!debugName.empty()) {
        debug::setObjectName(context.vkDevice(), handle_, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, debugName);
        debug::setObjectName(
            context.vkDevice(), storage_.buffer(), VK_OBJECT_TYPE_BUFFER, std::string(debugName) + "Storage");
    }
}

void VulkanAccelerationStructure::reset()
{
    // The structure is a view over storage_, so it has to go first.
    if (handle_ != VK_NULL_HANDLE && context_ != nullptr) {
        vkDestroyAccelerationStructureKHR(context_->vkDevice(), handle_, nullptr);
    }
    handle_ = VK_NULL_HANDLE;
    deviceAddress_ = 0;
    storage_.reset();
    context_ = nullptr;
    type_ = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
}

void VulkanAccelerationStructure::moveFrom(VulkanAccelerationStructure& other) noexcept
{
    context_ = std::exchange(other.context_, nullptr);
    storage_ = std::move(other.storage_);
    handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
    deviceAddress_ = std::exchange(other.deviceAddress_, 0);
    type_ = std::exchange(other.type_, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR);
}

} // namespace ve::rhi
