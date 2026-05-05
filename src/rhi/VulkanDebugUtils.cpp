#include "rhi/VulkanDebugUtils.h"

#include <string>

namespace ve::rhi::debug {

void setObjectName(VkDevice device, uint64_t objectHandle, VkObjectType objectType, std::string_view name)
{
    if (device == VK_NULL_HANDLE || objectHandle == 0 || name.empty() || vkSetDebugUtilsObjectNameEXT == nullptr) {
        return;
    }

    const std::string objectName{name};
    VkDebugUtilsObjectNameInfoEXT nameInfo{};
    nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    nameInfo.objectType = objectType;
    nameInfo.objectHandle = objectHandle;
    nameInfo.pObjectName = objectName.c_str();
    vkSetDebugUtilsObjectNameEXT(device, &nameInfo);
}

void beginLabel(VkCommandBuffer commandBuffer, std::string_view name)
{
    if (commandBuffer == VK_NULL_HANDLE || name.empty() || vkCmdBeginDebugUtilsLabelEXT == nullptr) {
        return;
    }

    const std::string labelName{name};
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = labelName.c_str();
    label.color[0] = 0.2f;
    label.color[1] = 0.6f;
    label.color[2] = 1.0f;
    label.color[3] = 1.0f;
    vkCmdBeginDebugUtilsLabelEXT(commandBuffer, &label);
}

void endLabel(VkCommandBuffer commandBuffer)
{
    if (commandBuffer == VK_NULL_HANDLE || vkCmdEndDebugUtilsLabelEXT == nullptr) {
        return;
    }

    vkCmdEndDebugUtilsLabelEXT(commandBuffer);
}

void insertLabel(VkCommandBuffer commandBuffer, std::string_view name)
{
    if (commandBuffer == VK_NULL_HANDLE || name.empty() || vkCmdInsertDebugUtilsLabelEXT == nullptr) {
        return;
    }

    const std::string labelName{name};
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = labelName.c_str();
    label.color[0] = 0.8f;
    label.color[1] = 0.8f;
    label.color[2] = 0.2f;
    label.color[3] = 1.0f;
    vkCmdInsertDebugUtilsLabelEXT(commandBuffer, &label);
}

} // namespace ve::rhi::debug
