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
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." + std::to_string(VK_API_VERSION_MINOR(version)) + "." +
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

    // Every synchronization-validation finding is named SYNC-HAZARD-...; the id
    // name is the layer's own classification, so reading it is cheaper and
    // steadier than matching on the message text. See
    // ValidationTally::recordSyncHazard for why these are counted apart.
    const char* messageId = callbackData != nullptr ? callbackData->pMessageIdName : nullptr;
    const bool syncHazard = messageId != nullptr && std::strncmp(messageId, "SYNC-", 5) == 0;

    // Tally before logging so a message is counted even if logging is filtered.
    // Counting is unconditional; whether a non-zero tally fails the process is
    // Application's policy (--fail-on-validation-error).
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        ValidationTally::recordError();
        if (syncHazard) {
            ValidationTally::recordSyncHazard();
        }
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

void VulkanContext::initialize(const Window& window,
                               std::filesystem::path shaderDirectory,
                               VulkanContextOptions options)
{
    // Before createInstance, which is the only place they can take effect.
    options_ = options;

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

    // A build with the layer compiled out cannot run the check at all, and a run
    // that silently skipped it would report a clean frame as evidence of
    // synchronization it never examined.
    if (options_.synchronizationValidation && !kEnableValidationLayers) {
        throw std::runtime_error(
            "Synchronization validation was requested but this build has the validation layer compiled out. "
            "Build a Debug configuration (VULKAN_ENGINE_ENABLE_VALIDATION).");
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

    // The layer reads this at instance creation, which is the only moment
    // synchronization validation can be turned on -- there is no runtime toggle.
    // VK_EXT_layer_settings replaced the deprecated VK_EXT_validation_features
    // for exactly this; requiredInstanceExtensions has already established that
    // the extension is there, so reaching here means it can be set.
    const VkBool32 syncValidationEnabled = VK_TRUE;
    VkLayerSettingEXT syncValidationSetting{};
    syncValidationSetting.pLayerName = kValidationLayers[0];
    syncValidationSetting.pSettingName = "validate_sync";
    syncValidationSetting.type = VK_LAYER_SETTING_TYPE_BOOL32_EXT;
    syncValidationSetting.valueCount = 1;
    syncValidationSetting.pValues = &syncValidationEnabled;

    VkLayerSettingsCreateInfoEXT layerSettingsCreateInfo{};
    layerSettingsCreateInfo.sType = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT;
    layerSettingsCreateInfo.settingCount = 1;
    layerSettingsCreateInfo.pSettings = &syncValidationSetting;
    if (options_.synchronizationValidation) {
        layerSettingsCreateInfo.pNext = &debugCreateInfo;
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
#if defined(__APPLE__)
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    if (options_.synchronizationValidation) {
        createInfo.pNext = &layerSettingsCreateInfo;
    } else {
        createInfo.pNext = kEnableValidationLayers ? &debugCreateInfo : nullptr;
    }
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = kEnableValidationLayers ? static_cast<uint32_t>(kValidationLayers.size()) : 0;
    createInfo.ppEnabledLayerNames = kEnableValidationLayers ? kValidationLayers.data() : nullptr;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance_));

    // Said out loud so a scripted run has machine-checkable evidence of which
    // check it was running. A log that cannot distinguish "clean under
    // synchronization validation" from "clean without it" makes the two look
    // like the same result.
    if (options_.synchronizationValidation) {
        Logger::info("Synchronization validation enabled via VK_EXT_layer_settings (validate_sync).");
    }
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

    // VK_EXT_layer_settings is provided by the validation layer itself, so it
    // does not appear in the driver's own extension list -- it has to be asked
    // for by layer name. Missing means the installed layer predates it and
    // synchronization validation cannot be configured through the API, which is
    // a hard failure rather than a quiet downgrade: see
    // VulkanContextOptions::synchronizationValidation.
    if (options_.synchronizationValidation) {
        bool layerSettingsAvailable = hasExtension(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
        if (!layerSettingsAvailable) {
            uint32_t layerExtensionCount = 0;
            VK_CHECK(vkEnumerateInstanceExtensionProperties(kValidationLayers[0], &layerExtensionCount, nullptr));
            std::vector<VkExtensionProperties> layerExtensions(layerExtensionCount);
            VK_CHECK(vkEnumerateInstanceExtensionProperties(
                kValidationLayers[0], &layerExtensionCount, layerExtensions.data()));
            layerSettingsAvailable =
                std::find_if(
                    layerExtensions.begin(), layerExtensions.end(), [](const VkExtensionProperties& extension) {
                        return std::strcmp(extension.extensionName, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME) == 0;
                    }) != layerExtensions.end();
        }

        if (!layerSettingsAvailable) {
            throw std::runtime_error(
                std::string("Synchronization validation was requested but the installed validation layer does not "
                            "provide ") +
                VK_EXT_LAYER_SETTINGS_EXTENSION_NAME + ". Update the Vulkan SDK.");
        }
    }

    for (const char* requiredExtension : extensions) {
        if (!hasExtension(requiredExtension)) {
            throw std::runtime_error(std::string("Required Vulkan instance extension is missing: ") +
                                     requiredExtension);
        }
    }

    // After the loop above, not before it: that check reads the driver's own
    // extension list, and a layer-provided extension is absent from it by
    // construction. Its availability was established against the layer a few
    // lines up, which is the list that can actually answer for it.
    if (options_.synchronizationValidation) {
        appendUniqueExtension(extensions, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
    }

    return extensions;
}

} // namespace ve::rhi
