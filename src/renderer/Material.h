#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <filesystem>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>

namespace ve::rhi {

class VulkanTexture;

} // namespace ve::rhi

namespace ve::renderer {

enum class MaterialSource {
    BuiltIn,
    Gltf,
    MaterialAsset,
    Fallback
};

struct Material {
    std::string debugName;
    std::string assetName;
    std::filesystem::path sourceAssetPath;
    std::string shader = "pbr_opaque";
    std::filesystem::path baseColorTexturePath;
    std::filesystem::path normalTexturePath;
    std::filesystem::path metallicRoughnessTexturePath;
    std::string alphaMode = "OPAQUE";
    const rhi::VulkanTexture* baseColorTexture = nullptr;
    const rhi::VulkanTexture* normalTexture = nullptr;
    const rhi::VulkanTexture* metallicRoughnessTexture = nullptr;
    // Emissive maps reuse the sRGB base-color bindless array, so the index is into
    // that array; hasEmissiveTexture gates sampling so factor-only emissive needs
    // no texture.
    const rhi::VulkanTexture* emissiveTexture = nullptr;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    uint32_t baseColorTextureIndex = 0;
    uint32_t normalTextureIndex = 0;
    uint32_t metallicRoughnessTextureIndex = 0;
    uint32_t emissiveTextureIndex = 0;
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    glm::vec3 emissiveFactor = glm::vec3(0.0f);
    float metallic = 0.0f;
    float roughness = 0.5f;
    float multiScatterStrength = 1.0f;
    float alphaCutoff = 0.5f;
    MaterialSource source = MaterialSource::BuiltIn;
    bool hasNormalMap = false;
    bool hasMetallicRoughnessMap = false;
    bool doubleSided = false;
    bool baseColorTextureFallback = false;
    bool normalTextureFallback = false;
    bool metallicRoughnessTextureFallback = false;
    bool hasEmissiveTexture = false;
};

} // namespace ve::renderer
