#include "rhi/VulkanDevice.h"

#include "core/Logger.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>

namespace ve::rhi {

namespace {

constexpr std::array<const char*, 1> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

bool supportsDescriptorIndexing(const VkPhysicalDeviceVulkan12Features& features)
{
    return features.descriptorIndexing == VK_TRUE
        && features.runtimeDescriptorArray == VK_TRUE
        && features.descriptorBindingPartiallyBound == VK_TRUE
        && features.descriptorBindingVariableDescriptorCount == VK_TRUE
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
    if (device_) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    physicalDevice_ = VK_NULL_HANDLE;
    graphicsQueue_ = VK_NULL_HANDLE;
    presentQueue_ = VK_NULL_HANDLE;
    queueFamilies_ = {};
    descriptorIndexingEnabled_ = false;
    bufferDeviceAddressEnabled_ = false;
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
    Logger::info(std::string("Selected GPU: ") + properties.deviceName);
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
    const std::array<uint32_t, 2> familyCandidates = {
        queueFamilies_.graphicsFamily.value(),
        queueFamilies_.presentFamily.value()
    };
    std::set<uint32_t> uniqueFamilies(familyCandidates.begin(), familyCandidates.end());

    const float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    queueCreateInfos.reserve(uniqueFamilies.size());
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = family;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
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

    VkPhysicalDeviceVulkan13Features enabled13{};
    enabled13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    enabled13.synchronization2 = VK_TRUE;
    enabled13.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceVulkan12Features enabled12{};
    enabled12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enabled12.pNext = &enabled13;
    enabled12.bufferDeviceAddress = VK_TRUE;
    enabled12.separateDepthStencilLayouts = VK_TRUE;

    if (descriptorIndexingEnabled_) {
        enabled12.descriptorIndexing = VK_TRUE;
        enabled12.runtimeDescriptorArray = VK_TRUE;
        enabled12.descriptorBindingPartiallyBound = VK_TRUE;
        enabled12.descriptorBindingVariableDescriptorCount = VK_TRUE;
        enabled12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &enabled12;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(kRequiredDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kRequiredDeviceExtensions.data();

    VK_CHECK(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_));
    volkLoadDevice(device_);

    vkGetDeviceQueue(device_, queueFamilies_.graphicsFamily.value(), 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, queueFamilies_.presentFamily.value(), 0, &presentQueue_);

    if (descriptorIndexingEnabled_) {
        Logger::info("Descriptor indexing features are enabled.");
    } else {
        Logger::warn("Descriptor indexing features are not fully supported; bindless texture work will stay optional.");
    }
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
    uint32_t extensionCount = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, nullptr));
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(candidate, nullptr, &extensionCount, availableExtensions.data()));

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
