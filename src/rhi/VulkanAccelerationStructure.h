#pragma once

#include "rhi/VulkanBuffer.h"
#include "rhi/VulkanCommon.h"

#include <string_view>

namespace ve::rhi {

class VulkanContext;

// One acceleration structure and the buffer that stores it.
//
// A VkAccelerationStructureKHR owns no memory: it is a view over a range of a
// buffer created with ACCELERATION_STRUCTURE_STORAGE usage, and destroying the
// buffer first leaves a handle pointing at freed memory that nothing reports
// until a build or trace reads it. Holding both here makes the order a property
// of one destructor rather than of every call site -- the structure goes first,
// then its storage.
//
// Creation only reserves storage. What the structure contains is decided by a
// build recorded into a command buffer, and the size passed in is the
// accelerationStructureSize vkGetAccelerationStructureBuildSizesKHR returned for
// that build; scratch memory is the caller's, because it is needed only while a
// build runs and can be shared between builds.
//
// Requires a device created with ray query (VulkanContext::rayQueryEnabled());
// create() refuses otherwise rather than reaching a null entry point.
class VulkanAccelerationStructure final {
public:
    VulkanAccelerationStructure() = default;
    ~VulkanAccelerationStructure();

    VulkanAccelerationStructure(const VulkanAccelerationStructure&) = delete;
    VulkanAccelerationStructure& operator=(const VulkanAccelerationStructure&) = delete;
    VulkanAccelerationStructure(VulkanAccelerationStructure&& other) noexcept;
    VulkanAccelerationStructure& operator=(VulkanAccelerationStructure&& other) noexcept;

    // Replaces whatever this held. On failure nothing is left allocated.
    void
    create(VulkanContext& context, VkAccelerationStructureTypeKHR type, VkDeviceSize size, std::string_view debugName);
    void reset();

    [[nodiscard]] VkAccelerationStructureKHR handle() const
    {
        return handle_;
    }
    // What a TLAS instance stores to reference this structure as its BLAS.
    [[nodiscard]] VkDeviceAddress deviceAddress() const
    {
        return deviceAddress_;
    }
    // The storage buffer, for barriers: a build writes and a trace reads the
    // structure through this buffer's memory.
    [[nodiscard]] VkBuffer buffer() const
    {
        return storage_.buffer();
    }
    [[nodiscard]] VkDeviceSize size() const
    {
        return storage_.size();
    }
    [[nodiscard]] VkAccelerationStructureTypeKHR type() const
    {
        return type_;
    }
    [[nodiscard]] bool valid() const
    {
        return handle_ != VK_NULL_HANDLE;
    }

private:
    void moveFrom(VulkanAccelerationStructure& other) noexcept;

    VulkanContext* context_ = nullptr;
    VulkanBuffer storage_;
    VkAccelerationStructureKHR handle_ = VK_NULL_HANDLE;
    VkDeviceAddress deviceAddress_ = 0;
    VkAccelerationStructureTypeKHR type_ = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
};

} // namespace ve::rhi
