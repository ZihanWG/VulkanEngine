#include "renderer/BuiltinTextureFactory.h"

#include "core/Logger.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <glm/common.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <span>
#include <string>

namespace ve::renderer {

void BuiltinTextureFactory::createCheckerboardBaseColor(rhi::VulkanContext& context,
                                                        const rhi::VulkanCommandContext& commandContext,
                                                        const std::filesystem::path& assetDirectory,
                                                        rhi::VulkanTexture& out) const
{
    const std::filesystem::path texturePath = assetDirectory / "textures/checker.png";
    if (std::filesystem::exists(texturePath)) {
        try {
            out.createFromFile(context, commandContext, texturePath, rhi::TextureColorSpace::SRGB, true);
            out.setDebugMetadata(rhi::TextureDebugMetadata{
                "Checkerboard base color",
                texturePath.string(),
                rhi::TextureColorSpace::SRGB,
                rhi::TextureDebugSource::LoadedFromDisk,
                false,
            });
            nameTexture(context, out, "BaseColorTexture");
            Logger::info("Loaded base color texture as sRGB: " + texturePath.string());
            return;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load texture '" + texturePath.string() + "': " + error.what());
        }
    } else {
        Logger::warn("Texture asset missing, using procedural checkerboard fallback: " + texturePath.string());
    }

    out.createCheckerboard(context, commandContext, 256, 256, rhi::TextureColorSpace::SRGB);
    out.setDebugMetadata(rhi::TextureDebugMetadata{
        "Procedural checkerboard base color",
        {},
        rhi::TextureColorSpace::SRGB,
        rhi::TextureDebugSource::ProceduralFallback,
        true,
    });
    nameTexture(context, out, "BaseColorTexture");
    Logger::info("Created procedural checkerboard base color texture as sRGB.");
}

void BuiltinTextureFactory::createPortfolioBaseColor(rhi::VulkanContext& context,
                                                     const rhi::VulkanCommandContext& commandContext,
                                                     rhi::VulkanTexture& out) const
{
    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::array<uint8_t, width * height * 4> pixels{};
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        pixels[offset + 0] = 255;
        pixels[offset + 1] = 255;
        pixels[offset + 2] = 255;
        pixels[offset + 3] = 255;
    }

    out.createFromRgba8(context,
                        commandContext,
                        width,
                        height,
                        std::span<const uint8_t>(pixels.data(), pixels.size()),
                        VK_FORMAT_R8G8B8A8_SRGB,
                        false);
    out.setDebugMetadata(rhi::TextureDebugMetadata{
        "Portfolio solid base color",
        {},
        rhi::TextureColorSpace::SRGB,
        rhi::TextureDebugSource::ProceduralFallback,
        false,
    });
    nameTexture(context, out, "PortfolioBaseColorTexture");
}

void BuiltinTextureFactory::createPortfolioBackdrop(rhi::VulkanContext& context,
                                                    const rhi::VulkanCommandContext& commandContext,
                                                    rhi::VulkanTexture& out) const
{
    constexpr uint32_t width = 16;
    constexpr uint32_t height = 64;
    std::array<uint8_t, width * height * 4> pixels{};
    const glm::vec3 bottom{132.0f, 144.0f, 154.0f};
    const glm::vec3 top{78.0f, 94.0f, 112.0f};

    for (uint32_t y = 0; y < height; ++y) {
        const float t = static_cast<float>(y) / static_cast<float>(height - 1);
        const glm::vec3 color = glm::mix(bottom, top, t);
        for (uint32_t x = 0; x < width; ++x) {
            const size_t offset = (static_cast<size_t>(y) * width + x) * 4U;
            pixels[offset + 0] = static_cast<uint8_t>(std::clamp(color.r, 0.0f, 255.0f));
            pixels[offset + 1] = static_cast<uint8_t>(std::clamp(color.g, 0.0f, 255.0f));
            pixels[offset + 2] = static_cast<uint8_t>(std::clamp(color.b, 0.0f, 255.0f));
            pixels[offset + 3] = 255;
        }
    }

    out.createFromRgba8(context,
                        commandContext,
                        width,
                        height,
                        std::span<const uint8_t>(pixels.data(), pixels.size()),
                        VK_FORMAT_R8G8B8A8_SRGB,
                        false);
    out.setDebugMetadata(rhi::TextureDebugMetadata{
        "Portfolio studio backdrop gradient",
        {},
        rhi::TextureColorSpace::SRGB,
        rhi::TextureDebugSource::ProceduralFallback,
        false,
    });
    nameTexture(context, out, "PortfolioBackdropTexture");
}

void BuiltinTextureFactory::createNormal(rhi::VulkanContext& context,
                                         const rhi::VulkanCommandContext& commandContext,
                                         const std::filesystem::path& assetDirectory,
                                         rhi::VulkanTexture& out,
                                         rhi::VulkanTexture& flatOut,
                                         bool& outAssetLoaded) const
{
    outAssetLoaded = false;
    bool loadedAsset = false;

    const std::filesystem::path texturePath = assetDirectory / "textures/checker_normal.png";
    if (std::filesystem::exists(texturePath)) {
        try {
            out.createFromFile(context, commandContext, texturePath, rhi::TextureColorSpace::Linear, true);
            out.setDebugMetadata(rhi::TextureDebugMetadata{
                "Checker normal map",
                texturePath.string(),
                rhi::TextureColorSpace::Linear,
                rhi::TextureDebugSource::LoadedFromDisk,
                false,
            });
            outAssetLoaded = true;
            nameTexture(context, out, "NormalTexture");
            Logger::info("Loaded normal texture as linear UNORM: " + texturePath.string());
            loadedAsset = true;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load normal texture '" + texturePath.string() + "': " + error.what());
        }
    } else {
        Logger::warn("Normal texture asset missing, using procedural flat normal fallback: " + texturePath.string());
    }

    if (!loadedAsset) {
        constexpr uint32_t width = 4;
        constexpr uint32_t height = 4;
        std::array<uint8_t, width * height * 4> pixels{};
        for (size_t offset = 0; offset < pixels.size(); offset += 4) {
            pixels[offset + 0] = 128;
            pixels[offset + 1] = 128;
            pixels[offset + 2] = 255;
            pixels[offset + 3] = 255;
        }

        out.createFromRgba8(context,
                            commandContext,
                            width,
                            height,
                            std::span<const uint8_t>(pixels.data(), pixels.size()),
                            VK_FORMAT_R8G8B8A8_UNORM,
                            false);
        out.setDebugMetadata(rhi::TextureDebugMetadata{
            "Procedural flat normal map",
            {},
            rhi::TextureColorSpace::Linear,
            rhi::TextureDebugSource::ProceduralFallback,
            true,
        });
        nameTexture(context, out, "NormalTexture");
        Logger::info("Created procedural flat normal texture as linear UNORM.");
    }

    createFlatNormal(context, commandContext, flatOut);
}

void BuiltinTextureFactory::createFlatNormal(rhi::VulkanContext& context,
                                             const rhi::VulkanCommandContext& commandContext,
                                             rhi::VulkanTexture& out) const
{
    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::array<uint8_t, width * height * 4> pixels{};
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        pixels[offset + 0] = 128;
        pixels[offset + 1] = 128;
        pixels[offset + 2] = 255;
        pixels[offset + 3] = 255;
    }

    out.createFromRgba8(context,
                        commandContext,
                        width,
                        height,
                        std::span<const uint8_t>(pixels.data(), pixels.size()),
                        VK_FORMAT_R8G8B8A8_UNORM,
                        false);
    out.setDebugMetadata(rhi::TextureDebugMetadata{
        "Flat normal fallback",
        {},
        rhi::TextureColorSpace::Linear,
        rhi::TextureDebugSource::ProceduralFallback,
        true,
    });
    nameTexture(context, out, "FlatNormalTexture");
}

void BuiltinTextureFactory::createMetallicRoughness(rhi::VulkanContext& context,
                                                    const rhi::VulkanCommandContext& commandContext,
                                                    const std::filesystem::path& assetDirectory,
                                                    rhi::VulkanTexture& out,
                                                    rhi::VulkanTexture& neutralOut,
                                                    bool& outAssetLoaded) const
{
    outAssetLoaded = false;
    bool loadedAsset = false;

    const std::filesystem::path texturePath = assetDirectory / "textures/checker_mr.png";
    if (std::filesystem::exists(texturePath)) {
        try {
            out.createFromFile(context, commandContext, texturePath, rhi::TextureColorSpace::Linear, true);
            out.setDebugMetadata(rhi::TextureDebugMetadata{
                "Checker metallic-roughness map",
                texturePath.string(),
                rhi::TextureColorSpace::Linear,
                rhi::TextureDebugSource::LoadedFromDisk,
                false,
            });
            outAssetLoaded = true;
            nameTexture(context, out, "MetallicRoughnessTexture");
            Logger::info("Loaded metallic-roughness texture as linear UNORM: " + texturePath.string());
            loadedAsset = true;
        } catch (const std::exception& error) {
            Logger::warn("Failed to load metallic-roughness texture '" + texturePath.string() + "': " + error.what());
        }
    } else {
        Logger::warn("Metallic-roughness texture asset missing, using procedural neutral fallback: " +
                     texturePath.string());
    }

    if (!loadedAsset) {
        constexpr uint32_t width = 4;
        constexpr uint32_t height = 4;
        std::array<uint8_t, width * height * 4> pixels{};
        for (size_t offset = 0; offset < pixels.size(); offset += 4) {
            pixels[offset + 0] = 255;
            pixels[offset + 1] = 255;
            pixels[offset + 2] = 0;
            pixels[offset + 3] = 255;
        }

        out.createFromRgba8(context,
                            commandContext,
                            width,
                            height,
                            std::span<const uint8_t>(pixels.data(), pixels.size()),
                            VK_FORMAT_R8G8B8A8_UNORM,
                            false);
        out.setDebugMetadata(rhi::TextureDebugMetadata{
            "Procedural neutral metallic-roughness map",
            {},
            rhi::TextureColorSpace::Linear,
            rhi::TextureDebugSource::ProceduralFallback,
            true,
        });
        nameTexture(context, out, "MetallicRoughnessTexture");
        Logger::info("Created procedural neutral metallic-roughness texture as linear UNORM.");
    }

    createNeutralMetallicRoughness(context, commandContext, neutralOut);
}

void BuiltinTextureFactory::createNeutralMetallicRoughness(rhi::VulkanContext& context,
                                                           const rhi::VulkanCommandContext& commandContext,
                                                           rhi::VulkanTexture& out) const
{
    constexpr uint32_t width = 4;
    constexpr uint32_t height = 4;
    std::array<uint8_t, width * height * 4> pixels{};
    for (size_t offset = 0; offset < pixels.size(); offset += 4) {
        pixels[offset + 0] = 255;
        pixels[offset + 1] = 255;
        pixels[offset + 2] = 0;
        pixels[offset + 3] = 255;
    }

    out.createFromRgba8(context,
                        commandContext,
                        width,
                        height,
                        std::span<const uint8_t>(pixels.data(), pixels.size()),
                        VK_FORMAT_R8G8B8A8_UNORM,
                        false);
    out.setDebugMetadata(rhi::TextureDebugMetadata{
        "Neutral metallic-roughness fallback",
        {},
        rhi::TextureColorSpace::Linear,
        rhi::TextureDebugSource::ProceduralFallback,
        true,
    });
    nameTexture(context, out, "NeutralMetallicRoughnessTexture");
}

void BuiltinTextureFactory::nameTexture(const rhi::VulkanContext& context,
                                        const rhi::VulkanTexture& texture,
                                        std::string_view name) const
{
    if (!texture.valid()) {
        return;
    }

    const std::string prefix{name};
    rhi::debug::setObjectName(context.vkDevice(), texture.image(), VK_OBJECT_TYPE_IMAGE, prefix + "Image");
    rhi::debug::setObjectName(context.vkDevice(), texture.imageView(), VK_OBJECT_TYPE_IMAGE_VIEW, prefix + "View");
    rhi::debug::setObjectName(context.vkDevice(), texture.sampler(), VK_OBJECT_TYPE_SAMPLER, prefix + "Sampler");
}

} // namespace ve::renderer
