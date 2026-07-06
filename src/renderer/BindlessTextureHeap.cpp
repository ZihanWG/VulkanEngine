#include "renderer/BindlessTextureHeap.h"

#include "core/Logger.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanTexture.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace ve::renderer {

namespace {

// Per-stage descriptor limits cover every set in the pipeline layout, so leave
// headroom for the non-bindless descriptor sets bound alongside the heap.
constexpr uint32_t kReservedDescriptors = 32;

uint32_t perClassDescriptorBudget(VkPhysicalDevice physicalDevice, bool updateAfterBind, uint32_t textureKindCount)
{
    VkPhysicalDeviceVulkan12Properties properties12{};
    properties12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;

    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &properties12;
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties);

    // Every heap binding is a fragment-stage combined image sampler, so the
    // total descriptor count is validated against the sampler, sampled-image,
    // and total-resource limits, both per stage and per set.
    const VkPhysicalDeviceLimits& limits = properties.properties.limits;
    const uint32_t budget = updateAfterBind
        ? std::min({properties12.maxPerStageDescriptorUpdateAfterBindSamplers,
                    properties12.maxPerStageDescriptorUpdateAfterBindSampledImages,
                    properties12.maxDescriptorSetUpdateAfterBindSamplers,
                    properties12.maxDescriptorSetUpdateAfterBindSampledImages,
                    properties12.maxPerStageUpdateAfterBindResources})
        : std::min({limits.maxPerStageDescriptorSamplers,
                    limits.maxPerStageDescriptorSampledImages,
                    limits.maxDescriptorSetSamplers,
                    limits.maxDescriptorSetSampledImages,
                    limits.maxPerStageResources});

    if (budget <= kReservedDescriptors) {
        return 0;
    }
    return (budget - kReservedDescriptors) / textureKindCount;
}

} // namespace

void BindlessTextureHeap::create(rhi::VulkanContext& context, uint32_t maxTextures)
{
    reset();

    if (maxTextures == 0) {
        throw std::runtime_error("BindlessTextureHeap requires at least one descriptor slot.");
    }

    device_ = context.vkDevice();

    const bool updateAfterBind = context.device().descriptorUpdateAfterBindEnabled();
    const uint32_t descriptorBudget = perClassDescriptorBudget(
        context.physicalDevice(), updateAfterBind, static_cast<uint32_t>(kTextureKindCount));
    if (descriptorBudget == 0) {
        throw std::runtime_error("Device descriptor limits are too small for the bindless material texture heap.");
    }
    if (descriptorBudget < maxTextures) {
        Logger::warn("Bindless material texture heap clamped from " + std::to_string(maxTextures) + " to " +
                     std::to_string(descriptorBudget) + " descriptors per class by device descriptor limits.");
    }
    maxTextures_ = std::min(maxTextures, descriptorBudget);

    std::array<VkDescriptorSetLayoutBinding, kTextureKindCount> bindings{};
    for (uint32_t binding = 0; binding < static_cast<uint32_t>(bindings.size()); ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[binding].descriptorCount = maxTextures_;
        bindings[binding].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    // Update-after-bind moves the heap out of the small non-update-after-bind
    // per-stage descriptor limits and into the maxPerStageDescriptorUpdateAfterBind*
    // limits, which are orders of magnitude larger on most drivers.
    std::array<VkDescriptorBindingFlags, kTextureKindCount> bindingFlags{};
    bindingFlags.fill(updateAfterBind
                          ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT
                          : VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{};
    bindingFlagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bindingFlagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    bindingFlagsInfo.pBindingFlags = bindingFlags.data();

    descriptorSetLayout_.create(
        device_,
        std::span<const VkDescriptorSetLayoutBinding>(bindings.data(), bindings.size()),
        updateAfterBind ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0,
        &bindingFlagsInfo);
    rhi::debug::setObjectName(
        device_, descriptorSetLayout_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, "BindlessMaterialTextureSetLayout");

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = maxTextures_ * static_cast<uint32_t>(bindings.size());

    descriptorPool_.create(device_,
                           std::span<const VkDescriptorPoolSize>(&poolSize, 1),
                           1,
                           updateAfterBind ? VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT : 0);
    rhi::debug::setObjectName(device_, descriptorPool_.handle(), VK_OBJECT_TYPE_DESCRIPTOR_POOL, "BindlessMaterialTexturePool");

    const VkDescriptorSetLayout setLayout = descriptorSetLayout_.handle();
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool_.handle();
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &setLayout;
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocateInfo, &descriptorSet_));
    rhi::debug::setObjectName(device_, descriptorSet_, VK_OBJECT_TYPE_DESCRIPTOR_SET, "BindlessMaterialTextureSet");
}

void BindlessTextureHeap::reset()
{
    descriptorSet_ = VK_NULL_HANDLE;
    descriptorPool_.reset();
    descriptorSetLayout_.reset();
    for (auto& textures : registeredTextures_) {
        textures.clear();
    }
    nextIndices_.fill(0);
    maxTextures_ = 0;
    device_ = VK_NULL_HANDLE;
}

uint32_t BindlessTextureHeap::registerTexture(TextureKind textureKind, const rhi::VulkanTexture& texture)
{
    if (!valid()) {
        throw std::runtime_error("Cannot register a bindless texture before the heap is created.");
    }
    if (!texture.valid()) {
        throw std::runtime_error("Cannot register an invalid texture in the bindless texture heap.");
    }

    const size_t kindIndex = indexFor(textureKind);
    auto& textures = registeredTextures_[kindIndex];
    const auto found = textures.find(&texture);
    if (found != textures.end()) {
        return found->second;
    }

    uint32_t& nextIndex = nextIndices_[kindIndex];
    if (nextIndex >= maxTextures_) {
        throw std::runtime_error("Bindless material texture heap capacity exceeded (" +
                                 std::to_string(maxTextures_) + " descriptors per material texture class).");
    }

    const uint32_t textureIndex = nextIndex++;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = texture.sampler();
    imageInfo.imageView = texture.imageView();
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet_;
    write.dstBinding = bindingFor(textureKind);
    write.dstArrayElement = textureIndex;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    textures.emplace(&texture, textureIndex);
    return textureIndex;
}

uint32_t BindlessTextureHeap::bindingFor(TextureKind textureKind)
{
    return static_cast<uint32_t>(textureKind);
}

size_t BindlessTextureHeap::indexFor(TextureKind textureKind)
{
    const size_t index = static_cast<size_t>(textureKind);
    if (index >= kTextureKindCount) {
        throw std::runtime_error("Invalid bindless texture kind.");
    }
    return index;
}

} // namespace ve::renderer
