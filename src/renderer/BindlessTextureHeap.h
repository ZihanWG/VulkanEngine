#pragma once

#include "rhi/VulkanCommon.h"
#include "rhi/VulkanDescriptor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace ve::rhi {

class VulkanContext;
class VulkanTexture;

} // namespace ve::rhi

namespace ve::renderer {

class BindlessTextureHeap final {
public:
    enum class TextureKind : uint32_t {
        BaseColor = 0,
        Normal = 1,
        MetallicRoughness = 2,
        Count = 3
    };

    static constexpr uint32_t kDefaultMaxTextures = 256;

    BindlessTextureHeap() = default;
    ~BindlessTextureHeap() = default;

    BindlessTextureHeap(const BindlessTextureHeap&) = delete;
    BindlessTextureHeap& operator=(const BindlessTextureHeap&) = delete;
    BindlessTextureHeap(BindlessTextureHeap&&) = delete;
    BindlessTextureHeap& operator=(BindlessTextureHeap&&) = delete;

    void create(rhi::VulkanContext& context, uint32_t maxTextures = kDefaultMaxTextures);
    void reset();

    [[nodiscard]] uint32_t registerTexture(TextureKind textureKind, const rhi::VulkanTexture& texture);

    [[nodiscard]] VkDescriptorSetLayout descriptorSetLayout() const { return descriptorSetLayout_.handle(); }
    [[nodiscard]] VkDescriptorSet descriptorSet() const { return descriptorSet_; }
    [[nodiscard]] uint32_t maxTextures() const { return maxTextures_; }
    [[nodiscard]] bool valid() const
    {
        return device_ != VK_NULL_HANDLE && descriptorSetLayout_.handle() != VK_NULL_HANDLE &&
               descriptorPool_.handle() != VK_NULL_HANDLE && descriptorSet_ != VK_NULL_HANDLE;
    }

private:
    static constexpr size_t kTextureKindCount = static_cast<size_t>(TextureKind::Count);

    [[nodiscard]] static uint32_t bindingFor(TextureKind textureKind);
    [[nodiscard]] static size_t indexFor(TextureKind textureKind);

    VkDevice device_ = VK_NULL_HANDLE;
    uint32_t maxTextures_ = 0;
    rhi::VulkanDescriptorSetLayout descriptorSetLayout_;
    rhi::VulkanDescriptorPool descriptorPool_;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    std::array<uint32_t, kTextureKindCount> nextIndices_{};
    std::array<std::unordered_map<const rhi::VulkanTexture*, uint32_t>, kTextureKindCount> registeredTextures_{};
};

} // namespace ve::renderer
