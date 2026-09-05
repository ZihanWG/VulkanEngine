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

    // Descriptors reserved per texture class. This is a self-imposed default, not
    // a device limit: an RTX 3080 Ti with update-after-bind reports a budget of
    // 349514 per class, so 256 was leaving three orders of magnitude unused while
    // registerTexture throws the moment a scene needs one more -- and glTF imports
    // are user-supplied, so that cliff is reachable by an asset rather than only by
    // a code change.
    //
    // 4096 costs 3 x 4096 descriptor slots in one pool, which is nothing, and
    // covers imports far larger than anything this repo ships. Devices that cannot
    // afford it are unaffected: create() already clamps to the queried budget and
    // warns, and PARTIALLY_BOUND means the unused slots never have to be written.
    static constexpr uint32_t kDefaultMaxTextures = 4096;

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
