#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace ve::rhi::debug {

void setObjectName(VkDevice device, uint64_t objectHandle, VkObjectType objectType, std::string_view name);
void beginLabel(VkCommandBuffer commandBuffer, std::string_view name);
void endLabel(VkCommandBuffer commandBuffer);
void insertLabel(VkCommandBuffer commandBuffer, std::string_view name);

template <typename Handle>
void setObjectName(VkDevice device, Handle objectHandle, VkObjectType objectType, std::string_view name)
{
    uint64_t handleValue = 0;
    if constexpr (std::is_pointer_v<Handle>) {
        handleValue = reinterpret_cast<uint64_t>(objectHandle);
    } else {
        handleValue = static_cast<uint64_t>(objectHandle);
    }

    setObjectName(device, handleValue, objectType, name);
}

} // namespace ve::rhi::debug
