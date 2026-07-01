#pragma once

#include "rhi/VulkanTexture.h"

#include <filesystem>
#include <string_view>

namespace ve::rhi {
class VulkanContext;
class VulkanCommandContext;
} // namespace ve::rhi

namespace ve::renderer {

// Creates the engine's built-in procedural / asset-fallback textures into
// caller-owned rhi::VulkanTexture references. Stateless and owns no GPU
// resources: each method borrows a VulkanContext + command context and writes
// into destination texture(s) passed by reference, mirroring the borrowing
// style of renderer::SceneBuilder. Kept decoupled from Renderer (it takes the
// asset directory as a parameter rather than reaching into RendererInternal.h),
// so it does not pull the renderer's god-object header back in.
class BuiltinTextureFactory {
public:
    BuiltinTextureFactory() = default;

    // Base color: try "<assetDirectory>/textures/checker.png" (sRGB, mipmapped),
    // else a procedural 256x256 sRGB checkerboard.
    void createCheckerboardBaseColor(rhi::VulkanContext& context,
                                     const rhi::VulkanCommandContext& commandContext,
                                     const std::filesystem::path& assetDirectory,
                                     rhi::VulkanTexture& out) const;

    // 4x4 solid-white sRGB base color for the portfolio material.
    void createPortfolioBaseColor(rhi::VulkanContext& context,
                                  const rhi::VulkanCommandContext& commandContext,
                                  rhi::VulkanTexture& out) const;

    // 16x64 vertical gradient sRGB studio backdrop for the portfolio scene.
    void createPortfolioBackdrop(rhi::VulkanContext& context,
                                 const rhi::VulkanCommandContext& commandContext,
                                 rhi::VulkanTexture& out) const;

    // Normal map: try "<assetDirectory>/textures/checker_normal.png" (linear),
    // else a procedural flat normal. Reports whether the asset loaded via
    // outAssetLoaded, and also populates flatOut with the standalone flat-normal
    // fallback texture, preserving the original chained creation order.
    void createNormal(rhi::VulkanContext& context,
                      const rhi::VulkanCommandContext& commandContext,
                      const std::filesystem::path& assetDirectory,
                      rhi::VulkanTexture& out,
                      rhi::VulkanTexture& flatOut,
                      bool& outAssetLoaded) const;

    // Metallic-roughness: try "<assetDirectory>/textures/checker_mr.png"
    // (linear), else a procedural neutral map. Reports asset load via
    // outAssetLoaded, and also populates neutralOut with the standalone neutral
    // fallback, preserving the original chained creation order.
    void createMetallicRoughness(rhi::VulkanContext& context,
                                 const rhi::VulkanCommandContext& commandContext,
                                 const std::filesystem::path& assetDirectory,
                                 rhi::VulkanTexture& out,
                                 rhi::VulkanTexture& neutralOut,
                                 bool& outAssetLoaded) const;

private:
    void createFlatNormal(rhi::VulkanContext& context,
                          const rhi::VulkanCommandContext& commandContext,
                          rhi::VulkanTexture& out) const;
    void createNeutralMetallicRoughness(rhi::VulkanContext& context,
                                        const rhi::VulkanCommandContext& commandContext,
                                        rhi::VulkanTexture& out) const;
    void nameTexture(const rhi::VulkanContext& context,
                     const rhi::VulkanTexture& texture,
                     std::string_view name) const;
};

} // namespace ve::renderer
