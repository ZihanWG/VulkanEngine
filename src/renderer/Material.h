#pragma once

#include "rhi/VulkanCommon.h"

#include <string>

namespace ve::rhi {

class VulkanTexture;

} // namespace ve::rhi

namespace ve::renderer {

struct Material {
    std::string debugName;
    const rhi::VulkanTexture* baseColorTexture = nullptr;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

} // namespace ve::renderer
