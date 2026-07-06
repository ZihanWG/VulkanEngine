#include "rhi/VulkanDevice.h"

#include "core/Logger.h"
#include "rhi/VulkanDebugUtils.h"
#include "rhi/VulkanPipelineCache.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ve::rhi {

namespace {

constexpr std::array<const char*, 1> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#if defined(__APPLE__)
#if defined(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)
constexpr const char* kPortabilitySubsetExtensionName = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
#else
constexpr const char* kPortabilitySubsetExtensionName = "VK_KHR_portability_subset";
#endif
#endif

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

std::vector<VkExtensionProperties> enumerateDeviceExtensions(VkPhysicalDevice candidate)
{
    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, nullptr));
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, availableExtensions.data()));
    return availableExtensions;
}

bool supportsDeviceExtension(VkPhysicalDevice candidate, const char* extensionName)
{
    const std::vector<VkExtensionProperties> availableExtensions = enumerateDeviceExtensions(candidate);
    return std::find_if(availableExtensions.begin(),
                        availableExtensions.end(),
                        [extensionName](const VkExtensionProperties& extension) {
                            return std::strcmp(extension.extensionName, extensionName) == 0;
                        }) != availableExtensions.end();
}

std::vector<const char*> deviceExtensionsFor(VkPhysicalDevice physicalDevice)
{
    std::vector<const char*> extensions(kRequiredDeviceExtensions.begin(), kRequiredDeviceExtensions.end());
#if defined(__APPLE__)
    if (supportsDeviceExtension(physicalDevice, kPortabilitySubsetExtensionName)) {
        appendUniqueExtension(extensions, kPortabilitySubsetExtensionName);
    }
#endif
    return extensions;
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

bool supportsDescriptorIndexing(const VkPhysicalDeviceVulkan12Features& features)
{
    return features.descriptorIndexing == VK_TRUE
        && features.runtimeDescriptorArray == VK_TRUE
        && features.descriptorBindingPartiallyBound == VK_TRUE
        && features.shaderSampledImageArrayNonUniformIndexing == VK_TRUE;
}

} // namespace

VulkanDevice::~VulkanDevice()
{
    cleanup();
}

void VulkanDevice::initialize(VkInstance instance, VkSurfaceKHR surface)
{
    instance_ = instance;
    surface_ = surface;

    pickPhysicalDevice();
    createLogicalDevice();
}

void VulkanDevice::cleanup()
{
    // The pipeline cache is a device child: persist and destroy it before the
    // device it was created from.
    destroyPipelineCache();

    if (device_) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    physicalDevice_ = VK_NULL_HANDLE;
    graphicsQueue_ = VK_NULL_HANDLE;
    presentQueue_ = VK_NULL_HANDLE;
    asyncComputeQueue_ = VK_NULL_HANDLE;
    asyncComputeQueueFamily_ = UINT32_MAX;
    asyncComputeAvailable_ = false;
    asyncComputeDedicatedFamily_ = false;
    queueFamilies_ = {};
    descriptorIndexingEnabled_ = false;
    bufferDeviceAddressEnabled_ = false;
    multiDrawIndirectEnabled_ = false;
    drawIndirectFirstInstanceEnabled_ = false;
    drawIndexedIndirectCountAvailable_ = false;
    maxDrawIndirectCount_ = 0;
}

void VulkanDevice::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr));
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan physical devices were found.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data()));

    int bestScore = -1;
    VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
    for (VkPhysicalDevice candidate : devices) {
        if (!isDeviceSuitable(candidate)) {
            continue;
        }

        const int score = scoreDevice(candidate);
        if (score > bestScore) {
            bestScore = score;
            bestDevice = candidate;
        }
    }

    if (bestDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("No suitable Vulkan 1.3 GPU was found.");
    }

    physicalDevice_ = bestDevice;
    queueFamilies_ = findQueueFamilies(physicalDevice_);

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    Logger::info(std::string("Selected Vulkan physical device: ") + properties.deviceName);
}

bool VulkanDevice::isDeviceSuitable(VkPhysicalDevice candidate) const
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(candidate, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_3) {
        return false;
    }

    const QueueFamilyIndices indices = findQueueFamilies(candidate);
    if (!indices.isComplete()) {
        return false;
    }

    if (!checkDeviceExtensionSupport(candidate)) {
        return false;
    }

    const SwapchainSupportDetails swapchainSupport = querySwapchainSupport(candidate);
    if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty()) {
        return false;
    }

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &features12;
    vkGetPhysicalDeviceFeatures2(candidate, &features);

    return features13.dynamicRendering == VK_TRUE
        && features13.synchronization2 == VK_TRUE
        && features12.bufferDeviceAddress == VK_TRUE
        && features12.separateDepthStencilLayouts == VK_TRUE;
}

int VulkanDevice::scoreDevice(VkPhysicalDevice candidate) const
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(candidate, &properties);

    int score = 0;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000;
    }
    score += static_cast<int>(properties.limits.maxImageDimension2D);
    return score;
}

void VulkanDevice::createLogicalDevice()
{
    const uint32_t graphicsFamily = queueFamilies_.graphicsFamily.value();

    // Async compute queue selection: a dedicated compute-only family runs on the
    // GPU's compute ring and overlaps rasterization best; a second queue in the
    // graphics family still lets the driver interleave. Neither is required.
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> familyProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, familyProperties.data());

    asyncComputeQueueFamily_ = UINT32_MAX;
    asyncComputeDedicatedFamily_ = false;
    uint32_t asyncComputeQueueIndex = 0;
    for (uint32_t family = 0; family < queueFamilyCount; ++family) {
        const VkQueueFlags flags = familyProperties[family].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) != 0 && (flags & VK_QUEUE_GRAPHICS_BIT) == 0 &&
            familyProperties[family].queueCount >= 1) {
            asyncComputeQueueFamily_ = family;
            asyncComputeDedicatedFamily_ = true;
            asyncComputeQueueIndex = 0;
            break;
        }
    }
    if (asyncComputeQueueFamily_ == UINT32_MAX && graphicsFamily < queueFamilyCount &&
        familyProperties[graphicsFamily].queueCount >= 2) {
        asyncComputeQueueFamily_ = graphicsFamily;
        asyncComputeQueueIndex = 1;
    }

    std::map<uint32_t, uint32_t> familyQueueCounts;
    familyQueueCounts[graphicsFamily] = 1;
    familyQueueCounts[queueFamilies_.presentFamily.value()] =
        std::max(familyQueueCounts[queueFamilies_.presentFamily.value()], 1u);
    if (asyncComputeQueueFamily_ != UINT32_MAX) {
        familyQueueCounts[asyncComputeQueueFamily_] =
            std::max(familyQueueCounts[asyncComputeQueueFamily_], asyncComputeQueueIndex + 1);
    }

    // The async compute queue gets a lower priority so it never starves the
    // frame-critical graphics queue.
    static constexpr std::array<float, 2> kQueuePriorities = {1.0f, 0.5f};
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.reserve(familyQueueCounts.size());
    for (const auto& [family, count] : familyQueueCounts) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = family;
        queueCreateInfo.queueCount = std::min<uint32_t>(count, static_cast<uint32_t>(kQueuePriorities.size()));
        queueCreateInfo.pQueuePriorities = kQueuePriorities.data();
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceVulkan13Features supported13{};
    supported13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceVulkan12Features supported12{};
    supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    supported12.pNext = &supported13;

    VkPhysicalDeviceFeatures2 supportedFeatures{};
    supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures.pNext = &supported12;
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &supportedFeatures);

    descriptorIndexingEnabled_ = supportsDescriptorIndexing(supported12);
    bufferDeviceAddressEnabled_ = supported12.bufferDeviceAddress == VK_TRUE;
    multiDrawIndirectEnabled_ = supportedFeatures.features.multiDrawIndirect == VK_TRUE;
    drawIndirectFirstInstanceEnabled_ = supportedFeatures.features.drawIndirectFirstInstance == VK_TRUE;

    VkPhysicalDeviceFeatures enabledCore{};
    enabledCore.multiDrawIndirect = multiDrawIndirectEnabled_ ? VK_TRUE : VK_FALSE;
    enabledCore.drawIndirectFirstInstance = drawIndirectFirstInstanceEnabled_ ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan13Features enabled13{};
    enabled13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    enabled13.synchronization2 = VK_TRUE;
    enabled13.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceVulkan12Features enabled12{};
    enabled12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enabled12.pNext = &enabled13;
    enabled12.bufferDeviceAddress = VK_TRUE;
    enabled12.separateDepthStencilLayouts = VK_TRUE;
    enabled12.drawIndirectCount = supported12.drawIndirectCount;

    if (descriptorIndexingEnabled_) {
        enabled12.descriptorIndexing = VK_TRUE;
        enabled12.runtimeDescriptorArray = VK_TRUE;
        enabled12.descriptorBindingPartiallyBound = VK_TRUE;
        enabled12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        // The current bindless heap allocates a fixed-size descriptor array, so
        // variable descriptor count is optional. Enable it when present for
        // future experiments, but do not require it for Milestone 30.
        enabled12.descriptorBindingVariableDescriptorCount =
            supported12.descriptorBindingVariableDescriptorCount;
    }

    const std::vector<const char*> enabledExtensions = deviceExtensionsFor(physicalDevice_);
#ifndef NDEBUG
    logEnabledExtensions("Enabled Vulkan device extensions:", enabledExtensions);
#endif

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &enabled12;
    createInfo.pEnabledFeatures = &enabledCore;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    VK_CHECK(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_));
    volkLoadDevice(device_);

    vkGetDeviceQueue(device_, queueFamilies_.graphicsFamily.value(), 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, queueFamilies_.presentFamily.value(), 0, &presentQueue_);

    if (asyncComputeQueueFamily_ != UINT32_MAX) {
        vkGetDeviceQueue(device_, asyncComputeQueueFamily_, asyncComputeQueueIndex, &asyncComputeQueue_);
        asyncComputeAvailable_ = asyncComputeQueue_ != VK_NULL_HANDLE;
    }
    if (asyncComputeAvailable_) {
        Logger::info(std::string("Async compute queue available (") +
                     (asyncComputeDedicatedFamily_ ? "dedicated compute-only family "
                                                   : "second queue in the graphics family ") +
                     std::to_string(asyncComputeQueueFamily_) + ").");
    } else {
        Logger::info("No async compute queue available; compute passes stay on the graphics queue. "
                     "(On MoltenVK, set MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1 to expose one.)");
    }

    createPipelineCache();

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    maxDrawIndirectCount_ = properties.limits.maxDrawIndirectCount;
    drawIndexedIndirectCountAvailable_ = supported12.drawIndirectCount == VK_TRUE &&
                                         vkCmdDrawIndexedIndirectCount != nullptr &&
                                         maxDrawIndirectCount_ > 0;

    if (descriptorIndexingEnabled_) {
        Logger::info("Descriptor indexing features for bindless material textures are enabled.");
    } else {
        Logger::warn("Descriptor indexing features required for bindless material textures are not fully supported; "
                     "the renderer will use per-material descriptor sets.");
    }

    if (multiDrawIndirectEnabled_ && drawIndirectFirstInstanceEnabled_) {
        Logger::info("Multi-draw indirect and drawIndirectFirstInstance are enabled.");
    } else {
        Logger::warn("Multi-draw indirect object-data array indexing is not fully supported; "
                     "the main pass will use per-draw indirect fallback recording.");
    }

    if (drawIndexedIndirectCountAvailable_) {
        Logger::info("vkCmdDrawIndexedIndirectCount is available. maxDrawIndirectCount=" +
                     std::to_string(maxDrawIndirectCount_) + ".");
    } else {
        Logger::warn("vkCmdDrawIndexedIndirectCount is unavailable; indirect-count drawing will remain disabled.");
    }
}

void VulkanDevice::createPipelineCache()
{
    assert(device_ != VK_NULL_HANDLE && "createPipelineCache() requires a live logical device.");

    // Load any previously saved blob and validate it against this exact GPU +
    // driver before handing it to the driver as initial data. On any mismatch we
    // build an empty cache so the driver is never fed a foreign/stale blob.
    const std::filesystem::path cachePath = pipelineCacheFilePath();
    std::vector<std::byte> blob = readPipelineCacheBlob(cachePath);

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    const bool usable = pipelineCacheHeaderMatches(blob, properties);

    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = usable ? blob.size() : 0;
    cacheInfo.pInitialData = usable ? blob.data() : nullptr;

    VK_CHECK(vkCreatePipelineCache(device_, &cacheInfo, nullptr, &pipelineCache_));
    debug::setObjectName(device_, pipelineCache_, VK_OBJECT_TYPE_PIPELINE_CACHE, "EnginePipelineCache");

    if (usable) {
        Logger::info("Loaded pipeline cache (" + std::to_string(blob.size()) + " bytes) from " +
                     cachePath.string() + ".");
    } else if (!blob.empty()) {
        Logger::warn("Discarded incompatible pipeline cache at " + cachePath.string() +
                     " (GPU/driver mismatch); starting from an empty cache.");
    } else {
        Logger::info("No usable pipeline cache found; starting from an empty cache.");
    }
}

void VulkanDevice::destroyPipelineCache()
{
    if (pipelineCache_ == VK_NULL_HANDLE) {
        return;
    }

    // Persist the accumulated cache before destroying it. A failed save must
    // never abort teardown, so everything here is best-effort.
    try {
        std::size_t dataSize = 0;
        if (vkGetPipelineCacheData(device_, pipelineCache_, &dataSize, nullptr) == VK_SUCCESS && dataSize > 0) {
            std::vector<std::byte> data(dataSize);
            if (vkGetPipelineCacheData(device_, pipelineCache_, &dataSize, data.data()) == VK_SUCCESS) {
                data.resize(dataSize);
                const std::filesystem::path cachePath = pipelineCacheFilePath();
                if (writePipelineCacheBlob(cachePath, data)) {
                    Logger::info("Saved pipeline cache (" + std::to_string(data.size()) + " bytes) to " +
                                 cachePath.string() + ".");
                } else {
                    Logger::warn("Failed to save pipeline cache to disk; the next launch will rebuild it.");
                }
            }
        }
    } catch (const std::exception& error) {
        Logger::warn(std::string("Pipeline cache save skipped: ") + error.what());
    } catch (...) {
        Logger::warn("Pipeline cache save skipped due to an unknown error.");
    }

    vkDestroyPipelineCache(device_, pipelineCache_, nullptr);
    pipelineCache_ = VK_NULL_HANDLE;
}

QueueFamilyIndices VulkanDevice::findQueueFamilies(VkPhysicalDevice candidate) const
{
    QueueFamilyIndices indices{};

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());

    for (uint32_t familyIndex = 0; familyIndex < familyCount; ++familyIndex) {
        const VkQueueFamilyProperties& family = families[familyIndex];
        if ((family.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            indices.graphicsFamily = familyIndex;
        }

        VkBool32 presentSupport = VK_FALSE;
        VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(candidate, familyIndex, surface_, &presentSupport));
        if (presentSupport == VK_TRUE) {
            indices.presentFamily = familyIndex;
        }

        if (indices.isComplete()) {
            break;
        }
    }

    return indices;
}

bool VulkanDevice::checkDeviceExtensionSupport(VkPhysicalDevice candidate) const
{
    const std::vector<VkExtensionProperties> availableExtensions = enumerateDeviceExtensions(candidate);

    std::set<std::string> required(kRequiredDeviceExtensions.begin(), kRequiredDeviceExtensions.end());
    for (const VkExtensionProperties& extension : availableExtensions) {
        required.erase(extension.extensionName);
    }

    return required.empty();
}

SwapchainSupportDetails VulkanDevice::querySwapchainSupport() const
{
    return querySwapchainSupport(physicalDevice_);
}

SwapchainSupportDetails VulkanDevice::querySwapchainSupport(VkPhysicalDevice candidate) const
{
    SwapchainSupportDetails details{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(candidate, surface_, &details.capabilities));

    uint32_t formatCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &formatCount, nullptr));
    if (formatCount != 0) {
        details.formats.resize(formatCount);
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &formatCount, details.formats.data()));
    }

    uint32_t presentModeCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_, &presentModeCount, nullptr));
    if (presentModeCount != 0) {
        details.presentModes.resize(presentModeCount);
        VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_, &presentModeCount, details.presentModes.data()));
    }

    return details;
}

} // namespace ve::rhi
