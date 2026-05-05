#pragma once

#include "rhi/VulkanCommon.h"

#include <glm/vec4.hpp>
#include <string>

namespace ve::rhi {

class VulkanTexture;

} // namespace ve::rhi

namespace ve::renderer {

struct Material {
    std::string debugName;
    const rhi::VulkanTexture* baseColorTexture = nullptr;
    const rhi::VulkanTexture* normalTexture = nullptr;
    const rhi::VulkanTexture* metallicRoughnessTexture = nullptr;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    float multiScatterStrength = 1.0f;
    bool hasNormalMap = false;
    bool hasMetallicRoughnessMap = false;
};

} // namespace ve::renderer
