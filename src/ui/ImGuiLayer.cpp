#include "ui/ImGuiLayer.h"

#include "core/Window.h"
#include "rhi/VulkanContext.h"
#include "rhi/VulkanDebugUtils.h"

#include <SDL3/SDL.h>
#include <backends/imgui_impl_sdl3.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace ve::ui {

namespace {

constexpr uint32_t kSampledImageDescriptorCount = 64;
constexpr uint32_t kSamplerDescriptorCount = 16;

} // namespace

ImGuiLayer::~ImGuiLayer()
{
    shutdown();
}

void ImGuiLayer::initialize(Window& window, rhi::VulkanContext& context, VkFormat colorFormat, uint32_t imageCount)
{
    if (initialized()) {
        return;
    }
    if (imageCount < 2) {
        throw std::runtime_error("ImGui requires at least two swapchain images.");
    }

    context_ = &context;
    device_ = context.vkDevice();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    contextInitialized_ = true;

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();

    if (!ImGui_ImplSDL3_InitForVulkan(window.nativeHandle())) {
        throw std::runtime_error(std::string("ImGui SDL3 backend initialization failed: ") + SDL_GetError());
    }
    platformInitialized_ = true;

    createDescriptorPool();
    initializeVulkanBackend(colorFormat, imageCount);
}

void ImGuiLayer::shutdown()
{
    if (device_ != VK_NULL_HANDLE && rendererInitialized_) {
        VK_CHECK(vkDeviceWaitIdle(device_));
        ImGui_ImplVulkan_Shutdown();
        rendererInitialized_ = false;
    }

    destroyDescriptorPool();

    if (platformInitialized_) {
        ImGui_ImplSDL3_Shutdown();
        platformInitialized_ = false;
    }

    if (contextInitialized_) {
        ImGui::DestroyContext();
        contextInitialized_ = false;
    }

    context_ = nullptr;
    device_ = VK_NULL_HANDLE;
    colorFormat_ = VK_FORMAT_UNDEFINED;
    imageCount_ = 0;
}

void ImGuiLayer::handleEvent(const SDL_Event& event)
{
    if (platformInitialized_) {
        ImGui_ImplSDL3_ProcessEvent(&event);
    }
}

void ImGuiLayer::beginFrame()
{
    if (!initialized()) {
        return;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::endFrame()
{
    if (initialized()) {
        ImGui::Render();
    }
}

void ImGuiLayer::render(VkCommandBuffer commandBuffer)
{
    if (!initialized() || commandBuffer == VK_NULL_HANDLE) {
        return;
    }

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

void ImGuiLayer::onSwapchainRecreated(VkFormat colorFormat, uint32_t imageCount)
{
    if (!rendererInitialized_) {
        return;
    }

    if (imageCount < 2) {
        throw std::runtime_error("ImGui requires at least two swapchain images.");
    }

    if (colorFormat == colorFormat_ && imageCount == imageCount_) {
        ImGui_ImplVulkan_SetMinImageCount(minImageCount(imageCount));
        return;
    }

    VK_CHECK(vkDeviceWaitIdle(device_));
    ImGui_ImplVulkan_Shutdown();
    rendererInitialized_ = false;

    destroyDescriptorPool();
    createDescriptorPool();
    initializeVulkanBackend(colorFormat, imageCount);
}

bool ImGuiLayer::wantsMouseCapture() const
{
    return contextInitialized_ && ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiLayer::wantsKeyboardCapture() const
{
    return contextInitialized_ && ImGui::GetIO().WantCaptureKeyboard;
}

void ImGuiLayer::checkVkResult(VkResult result)
{
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("ImGui Vulkan backend call failed: ") + rhi::vkResultToString(result));
    }
}

void ImGuiLayer::createDescriptorPool()
{
    if (device_ == VK_NULL_HANDLE || descriptorPool_ != VK_NULL_HANDLE) {
        return;
    }

    const std::array<VkDescriptorPoolSize, 2> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kSampledImageDescriptorCount},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, kSamplerDescriptorCount},
    };

    VkDescriptorPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    createInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    createInfo.maxSets = kSampledImageDescriptorCount + kSamplerDescriptorCount;
    createInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    createInfo.pPoolSizes = poolSizes.data();

    VK_CHECK(vkCreateDescriptorPool(device_, &createInfo, nullptr, &descriptorPool_));
    rhi::debug::setObjectName(device_, descriptorPool_, VK_OBJECT_TYPE_DESCRIPTOR_POOL, "ImGuiDescriptorPool");
}

void ImGuiLayer::destroyDescriptorPool()
{
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }
}

void ImGuiLayer::initializeVulkanBackend(VkFormat colorFormat, uint32_t imageCount)
{
    if (!context_ || descriptorPool_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot initialize ImGui Vulkan backend without a context and descriptor pool.");
    }

    VkPipelineRenderingCreateInfoKHR renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = context_->instance();
    initInfo.PhysicalDevice = context_->physicalDevice();
    initInfo.Device = context_->vkDevice();
    initInfo.QueueFamily = context_->queueFamilies().graphicsFamily.value();
    initInfo.Queue = context_->graphicsQueue();
    initInfo.DescriptorPool = descriptorPool_;
    initInfo.MinImageCount = minImageCount(imageCount);
    initInfo.ImageCount = imageCount;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;
    initInfo.UseDynamicRendering = true;
    initInfo.CheckVkResultFn = &ImGuiLayer::checkVkResult;
    initInfo.MinAllocationSize = 1024 * 1024;

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        throw std::runtime_error("ImGui Vulkan backend initialization failed.");
    }

    colorFormat_ = colorFormat;
    imageCount_ = imageCount;
    rendererInitialized_ = true;
}

uint32_t ImGuiLayer::minImageCount(uint32_t imageCount) const
{
    return std::min(imageCount, 2u);
}

} // namespace ve::ui
