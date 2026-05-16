#pragma once

#include "rhi/VulkanCommon.h"

#include <cstdint>

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

    void initialize(Window& window, rhi::VulkanContext& context, VkFormat colorFormat, uint32_t imageCount);
    void shutdown();

    void handleEvent(const SDL_Event& event);
    void beginFrame();
    void endFrame();
    void render(VkCommandBuffer commandBuffer);
    void onSwapchainRecreated(VkFormat colorFormat, uint32_t imageCount);

    [[nodiscard]] bool wantsMouseCapture() const;
    [[nodiscard]] bool wantsKeyboardCapture() const;
    [[nodiscard]] bool initialized() const { return contextInitialized_ && platformInitialized_ && rendererInitialized_; }

private:
    static void checkVkResult(VkResult result);

    void createDescriptorPool();
    void destroyDescriptorPool();
    void initializeVulkanBackend(VkFormat colorFormat, uint32_t imageCount);
    [[nodiscard]] uint32_t minImageCount(uint32_t imageCount) const;

    rhi::VulkanContext* context_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED;
    uint32_t imageCount_ = 0;
    bool contextInitialized_ = false;
    bool platformInitialized_ = false;
    bool rendererInitialized_ = false;
};

} // namespace ui
} // namespace ve
