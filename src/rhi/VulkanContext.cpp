#include "rhi/VulkanContext.h"

#include "core/Logger.h"
#include "core/Window.h"
#include "rhi/ValidationTally.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

#if defined(VULKAN_ENGINE_ENABLE_VALIDATION) && VULKAN_ENGINE_ENABLE_VALIDATION
constexpr bool kEnableValidationLayers = true;
#else
constexpr bool kEnableValidationLayers = false;
#endif

constexpr std::array<const char*, 1> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};
constexpr uint32_t kRequiredVulkanApiVersion = VK_API_VERSION_1_3;

bool containsExtension(const std::vector<const char*>& extensions, const char* extensionName)
{
    return std::find_if(extensions.begin(), extensions.end(), [extensionName](const char* enabledExtension) {
               return std::strcmp(enabledExtension, extensionName) == 0;
           }) != extensions.end();
}

void appendUniqueExtension(std::vector<const char*>& extensions, const char* extensionName)
{
    if (!containsExtension(extensions, extensionName)) {
        extensions.push_back(extensionName);
    }
}

std::string vulkanApiVersionString(uint32_t version)
{
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

uint32_t selectVulkanApiVersion()
{
    uint32_t supportedApiVersion = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) {
        VK_CHECK(vkEnumerateInstanceVersion(&supportedApiVersion));
    }

    if (supportedApiVersion < kRequiredVulkanApiVersion) {
        throw std::runtime_error("Vulkan 1.3 is required, but the loader reports Vulkan " +
                                 vulkanApiVersionString(supportedApiVersion) + ".");
    }

    return kRequiredVulkanApiVersion;
}

#ifndef NDEBUG
void logEnabledExtensions(const char* label, const std::vector<const char*>& extensions)
{
    std::string message(label);
    for (const char* extension : extensions) {
        message += "\n  ";
        message += extension;
    }
    Logger::info(message);
}
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL validationCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                  VkDebugUtilsMessageTypeFlagsEXT,
                                                  const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                  void*)
{
    const char* message =
        callbackData && callbackData->pMessage ? callbackData->pMessage : "Unknown validation message";

    // Tally before logging so a message is counted even if logging is filtered.
    // Counting is unconditional; whether a non-zero tally fails the process is
    // Application's policy (--fail-on-validation-error).
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        ValidationTally::recordError();
        Logger::error(message);
    } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        ValidationTally::recordWarning();
        Logger::warn(message);
    } else {
        Logger::trace(message);
    }

    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo()
{
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = validationCallback;
    return createInfo;
}

} // namespace

VulkanContext::~VulkanContext()
{
    cleanup();
}

void VulkanContext::initialize(const Window& window, std::filesystem::path shaderDirectory)
{
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = window.vulkanGetInstanceProcAddr();
    if (getInstanceProcAddr == nullptr) {
        throw std::runtime_error("SDL Vulkan vkGetInstanceProcAddr is unavailable.");
    }
    volkInitializeCustom(getInstanceProcAddr);
    Logger::info("Volk initialized from SDL Vulkan vkGetInstanceProcAddr.");

    createInstance(window);
    volkLoadInstance(instance_);

    setupDebugMessenger();
    surface_ = window.createSurface(instance_);

    device_.initialize(instance_, surface_, std::move(shaderDirectory));
    createAllocator();
}

void VulkanContext::cleanup()
{
    if (device_.device()) {
        vkDeviceWaitIdle(device_.device());
    }

    if (allocator_) {
        vmaDestroyAllocator(allocator_);
        allocator_ = VK_NULL_HANDLE;
    }

    device_.cleanup();

    if (surface_) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }

    if (debugMessenger_) {
        if (vkDestroyDebugUtilsMessengerEXT != nullptr) {
            vkDestroyDebugUtilsMessengerEXT(instance_, debugMessenger_, nullptr);
        }
        debugMessenger_ = VK_NULL_HANDLE;
    }

    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::waitIdle() const
{
    if (device_.device()) {
        VK_CHECK(vkDeviceWaitIdle(device_.device()));
    }
}

void VulkanContext::createInstance(const Window& window)
{
    if (kEnableValidationLayers && !validationLayersAvailable()) {
        throw std::runtime_error("Validation layers were requested but VK_LAYER_KHRONOS_validation is not available.");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanEngine";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "VulkanEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = selectVulkanApiVersion();
    Logger::info("Selected Vulkan API version: " + vulkanApiVersionString(appInfo.apiVersion));

    const std::vector<const char*> extensions = requiredInstanceExtensions(window);
#ifndef NDEBUG
    logEnabledExtensions("Enabled Vulkan instance extensions:", extensions);
#endif
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = debugMessengerCreateInfo();

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
#if defined(__APPLE__)
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    createInfo.pNext = kEnableValidationLayers ? &debugCreateInfo : nullptr;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = kEnableValidationLayers ? static_cast<uint32_t>(kValidationLayers.size()) : 0;
    createInfo.ppEnabledLayerNames = kEnableValidationLayers ? kValidationLayers.data() : nullptr;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));
}

void VulkanContext::setupDebugMessenger()
{
    if (!kEnableValidationLayers) {
        return;
    }
    if (vkCreateDebugUtilsMessengerEXT == nullptr) {
        Logger::warn("Vulkan validation messenger unavailable because VK_EXT_debug_utils functions were not loaded.");
        return;
    }

    const VkDebugUtilsMessengerCreateInfoEXT createInfo = debugMessengerCreateInfo();
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(instance_, &createInfo, nullptr, &debugMessenger_));
    Logger::info("Vulkan validation layers enabled.");
}

void VulkanContext::createAllocator()
{
    vmaVulkanFunctions_.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaVulkanFunctions_.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorInfo.physicalDevice = device_.physicalDevice();
    allocatorInfo.device = device_.device();
    allocatorInfo.instance = instance_;
    allocatorInfo.vulkanApiVersion = kRequiredVulkanApiVersion;
    allocatorInfo.pVulkanFunctions = &vmaVulkanFunctions_;

    VK_CHECK(vmaCreateAllocator(&allocatorInfo, &allocator_));
}

bool VulkanContext::validationLayersAvailable() const
{
    uint32_t layerCount = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));
    std::vector<VkLayerProperties> availableLayers(layerCount);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()));

    for (const char* layerName : kValidationLayers) {
        const auto layerIt =
            std::find_if(availableLayers.begin(), availableLayers.end(), [layerName](const VkLayerProperties& layer) {
                return std::strcmp(layer.layerName, layerName) == 0;
            });

        if (layerIt == availableLayers.end()) {
            return false;
        }
    }

    return true;
}

std::vector<const char*> VulkanContext::requiredInstanceExtensions(const Window& window) const
{
    std::vector<const char*> extensions = window.requiredVulkanInstanceExtensions();

    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr));
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data()));

    const auto hasExtension = [&availableExtensions](const char* extensionName) {
        return std::find_if(availableExtensions.begin(),
                            availableExtensions.end(),
                            [extensionName](const VkExtensionProperties& extension) {
                                return std::strcmp(extension.extensionName, extensionName) == 0;
                            }) != availableExtensions.end();
    };

#if defined(__APPLE__)
    appendUniqueExtension(extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    const bool debugUtilsAvailable = hasExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (!containsExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) && debugUtilsAvailable) {
        appendUniqueExtension(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    if (kEnableValidationLayers && !debugUtilsAvailable) {
        throw std::runtime_error(std::string("Required Vulkan instance extension is missing: ") +
                                 VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    for (const char* requiredExtension : extensions) {
        if (!hasExtension(requiredExtension)) {
            throw std::runtime_error(std::string("Required Vulkan instance extension is missing: ") +
                                     requiredExtension);
        }
    }

    return extensions;
}

} // namespace ve::rhi
