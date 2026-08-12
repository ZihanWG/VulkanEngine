#pragma once

#include <filesystem>
#include <string>

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <cstddef>
#include <functional>
#include <unordered_map>

typedef union SDL_Event SDL_Event;

namespace ve {

class Window;

namespace rhi {
class VulkanContext;
}

namespace ui {

class ImGuiLayer final {
public:
    ImGuiLayer() = default;
    ~ImGuiLayer();

    ImGuiLayer(const ImGuiLayer&) = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;
    ImGuiLayer(ImGuiLayer&&) = delete;
    ImGuiLayer& operator=(ImGuiLayer&&) = delete;

    // layoutFile is where ImGui persists window positions, sizes and open state.
    // Without it every launch starts with the panels back at their defaults,
    // which is a small thing that costs a few seconds every single run.
    void initialize(Window& window,
                    rhi::VulkanContext& context,
                    VkFormat colorFormat,
                    uint32_t imageCount,
                    const std::filesystem::path& layoutFile = {});
    void shutdown();

    void handleEvent(const SDL_Event& event);
    void beginFrame();
    void endFrame();
    void render(VkCommandBuffer commandBuffer);
    void onSwapchainRecreated(VkFormat colorFormat, uint32_t imageCount);
    [[nodiscard]] VkDescriptorSet texturePreviewDescriptor(
        VkImageView imageView,
        VkSampler sampler,
        VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    [[nodiscard]] VkDescriptorSet renderTargetPreviewDescriptor(
        VkImageView imageView,
        VkSampler sampler,
        VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    void clearTexturePreviewDescriptors();
    void clearRenderTargetPreviewDescriptors();

    [[nodiscard]] bool wantsMouseCapture() const;
    [[nodiscard]] bool wantsKeyboardCapture() const;
    [[nodiscard]] bool initialized() const { return contextInitialized_ && platformInitialized_ && rendererInitialized_; }

private:
    struct TexturePreviewKey {
        VkImageView imageView = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        [[nodiscard]] bool operator==(const TexturePreviewKey& other) const
        {
            return imageView == other.imageView && sampler == other.sampler && imageLayout == other.imageLayout;
        }
    };

    struct TexturePreviewKeyHash {
        [[nodiscard]] size_t operator()(const TexturePreviewKey& key) const
        {
            size_t hash = std::hash<VkImageView>{}(key.imageView);
            hash ^= std::hash<VkSampler>{}(key.sampler) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
            hash ^= std::hash<int>{}(static_cast<int>(key.imageLayout)) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
            return hash;
        }
    };

    static void checkVkResult(VkResult result);

    void createDescriptorPool();
    void destroyDescriptorPool();
    void initializeVulkanBackend(VkFormat colorFormat, uint32_t imageCount);
    void clearDescriptorCache(std::unordered_map<TexturePreviewKey, VkDescriptorSet, TexturePreviewKeyHash>& cache);
    [[nodiscard]] uint32_t minImageCount(uint32_t imageCount) const;

    rhi::VulkanContext* context_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED;
    uint32_t imageCount_ = 0;
    std::unordered_map<TexturePreviewKey, VkDescriptorSet, TexturePreviewKeyHash> texturePreviewDescriptors_;
    std::unordered_map<TexturePreviewKey, VkDescriptorSet, TexturePreviewKeyHash> renderTargetPreviewDescriptors_;
    // ImGui stores this as a raw pointer, so the string has to outlive the
    // context rather than being a temporary at the call site.
    std::string layoutFilePath_;
    bool contextInitialized_ = false;
    bool platformInitialized_ = false;
    bool rendererInitialized_ = false;
};

} // namespace ui
} // namespace ve
