#include "renderer/BuiltinTextureFactory.h"

#include "core/Logger.h"
#include "rhi/VulkanCommandContext.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <glm/common.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace ve::renderer {

namespace {

// The builtin textures are real files in real material slots, so they take the
// cooked sidecar when one is there. Same rule as the material-asset and glTF
// paths: any problem with the cooked file loses to the uncompressed source
// rather than failing the texture, so a scene never goes untextured over it.
void loadPreferringCooked(rhi::VulkanContext& context,
                          const rhi::VulkanCommandContext& commandContext,
                          const std::filesystem::path& texturePath,
                          assets::TextureUsage usage,
                          rhi::VulkanTexture& out)
{
    if (rhi::cookedTextureAvailable(context.physicalDevice(), texturePath, usage)) {
        const std::filesystem::path cookedPath = assets::ktx2SidecarPath(texturePath, usage);
        try {
            out.createFromKtx2(context, commandContext, cookedPath, usage);
            Logger::info("Loaded cooked builtin texture: " + cookedPath.string());
            return;
        } catch (const std::exception& error) {
            Logger::warn("Cooked builtin texture '" + cookedPath.string() +
                         "' could not be used; loading the uncompressed source instead: " + error.what());
        }
    }

    out.createFromFile(context,
                       commandContext,
                       texturePath,
                       assets::textureUsageIsSrgb(usage) ? rhi::TextureColorSpace::SRGB
                                                         : rhi::TextureColorSpace::Linear,
                       true);
}

} // namespace

void BuiltinTextureFactory::createCheckerboardBaseColor(rhi::VulkanContext& context,
                                                        const rhi::VulkanCommandContext& commandContext,
                                                        const std::filesystem::path& assetDirectory,
                                                        rhi::VulkanTexture& out) const
{
    const std::filesystem::path texturePath = assetDirectory / "textures/checker.png";
    if (std::filesystem::exists(texturePath)) {
        try {
            loadPreferringCooked(context, commandContext, texturePath, assets::TextureUsage::BaseColor, out);
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

void BuiltinTextureFactory::createCutoutLattice(rhi::VulkanContext& context,
                                                const rhi::VulkanCommandContext& commandContext,
                                                rhi::VulkanTexture& out) const
{
    constexpr uint32_t width = 256;
    constexpr uint32_t height = 256;
    constexpr uint32_t cellCount = 8;
    constexpr float cellSize = static_cast<float>(width) / static_cast<float>(cellCount);
    // Hole radius as a fraction of the cell. Leaves a solid frame between holes so
    // the panel still reads as a surface rather than a sieve.
    constexpr float holeRadius = cellSize * 0.34f;

    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4U);
    const glm::vec3 panelColor{198.0f, 176.0f, 132.0f};
    const glm::vec3 rimColor{150.0f, 126.0f, 88.0f};

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            // Distance from this texel to the centre of the cell it falls in.
            const float cellX = std::fmod(static_cast<float>(x), cellSize) - cellSize * 0.5f;
            const float cellY = std::fmod(static_cast<float>(y), cellSize) - cellSize * 0.5f;
            const float distance = std::sqrt(cellX * cellX + cellY * cellY);

            // One-texel-wide ramp instead of a hard step: the alpha test still
            // produces a crisp edge, but the gradient keeps the cutoff meaningful
            // and gives mip levels something sane to filter.
            const float alpha = std::clamp(distance - holeRadius, 0.0f, 1.0f);
            // Darken toward the hole rim so the perforation reads in the shading
            // and not only in the silhouette.
            const glm::vec3 color = glm::mix(rimColor, panelColor, std::clamp((distance - holeRadius) / 3.0f, 0.0f, 1.0f));

            const size_t offset = (static_cast<size_t>(y) * width + x) * 4U;
            pixels[offset + 0] = static_cast<uint8_t>(std::clamp(color.r, 0.0f, 255.0f));
            pixels[offset + 1] = static_cast<uint8_t>(std::clamp(color.g, 0.0f, 255.0f));
            pixels[offset + 2] = static_cast<uint8_t>(std::clamp(color.b, 0.0f, 255.0f));
            pixels[offset + 3] = static_cast<uint8_t>(std::clamp(alpha * 255.0f, 0.0f, 255.0f));
        }
    }

    out.createFromRgba8(context,
                        commandContext,
                        width,
                        height,
                        std::span<const uint8_t>(pixels.data(), pixels.size()),
                        VK_FORMAT_R8G8B8A8_SRGB,
                        true);
    out.setDebugMetadata(rhi::TextureDebugMetadata{
        "Procedural cutout lattice (alpha-mask demo)",
        {},
        rhi::TextureColorSpace::SRGB,
        rhi::TextureDebugSource::ProceduralFallback,
        false,
    });
    nameTexture(context, out, "CutoutLatticeTexture");
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
            loadPreferringCooked(context, commandContext, texturePath, assets::TextureUsage::NormalMap, out);
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
            loadPreferringCooked(
                context, commandContext, texturePath, assets::TextureUsage::MetallicRoughness, out);
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
