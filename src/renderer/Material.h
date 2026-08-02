#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <filesystem>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>
#include <string_view>

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

// glTF 2.0 alpha modes. Material::alphaMode keeps the glTF spelling because it is
// what the asset JSON round-trips; this enum is the parsed form the render path
// switches on (render-bucket assignment, pipeline choice, alpha-test cutoff).
enum class AlphaMode {
    Opaque,
    Mask,
    Blend
};

[[nodiscard]] inline AlphaMode alphaModeFromString(std::string_view mode)
{
    if (mode == "MASK") {
        return AlphaMode::Mask;
    }
    if (mode == "BLEND") {
        return AlphaMode::Blend;
    }
    return AlphaMode::Opaque;
}

[[nodiscard]] inline std::string_view alphaModeToString(AlphaMode mode)
{
    switch (mode) {
    case AlphaMode::Mask:
        return "MASK";
    case AlphaMode::Blend:
        return "BLEND";
    case AlphaMode::Opaque:
        break;
    }
    return "OPAQUE";
}

// Value written to ObjectFrameData::materialParams.w. The shaders alpha-test when
// it is non-negative, so OPAQUE and BLEND (which never clip) encode as -1 and the
// fragment shaders need no extra varying or uniform to branch on.
inline constexpr float kNoAlphaTestCutoff = -1.0f;

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

    [[nodiscard]] AlphaMode alphaModeValue() const { return alphaModeFromString(alphaMode); }

    // MASK materials clip in the main and shadow passes; OPAQUE and BLEND disable
    // the test. See kNoAlphaTestCutoff.
    [[nodiscard]] float alphaTestCutoff() const
    {
        return alphaModeValue() == AlphaMode::Mask ? alphaCutoff : kNoAlphaTestCutoff;
    }
};

} // namespace ve::renderer
