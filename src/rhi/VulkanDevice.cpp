#include "rhi/VulkanDevice.h"

#include "rhi/DeviceExtensionSelection.h"
#include "rhi/TransferQueueSelection.h"

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
#include <string_view>
#include <utility>
#include <vector>

namespace ve::rhi {

namespace {

constexpr std::array<const char*, 1> kRequiredDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#if defined(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)
constexpr const char* kPortabilitySubsetExtensionName = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
#else
constexpr const char* kPortabilitySubsetExtensionName = "VK_KHR_portability_subset";
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

// What the device will be created with, plus why each optional extension was or
// was not taken. The outcomes are kept so the startup report can explain a path
// being off without the reader having to reason about driver versions.
struct DeviceExtensionPlan {
    std::vector<const char*> enabled;
    std::vector<ExtensionOutcome> optionalOutcomes;
};

DeviceExtensionPlan planDeviceExtensions(VkPhysicalDevice physicalDevice)
{
    DeviceExtensionPlan plan{};
    plan.enabled.assign(kRequiredDeviceExtensions.begin(), kRequiredDeviceExtensions.end());

    // Held by value for the whole function: the string_views below point into it.
    const std::vector<VkExtensionProperties> availableProperties = enumerateDeviceExtensions(physicalDevice);
    std::vector<std::string_view> available;
    available.reserve(availableProperties.size());
    for (const VkExtensionProperties& extension : availableProperties) {
        available.emplace_back(extension.extensionName);
    }

    // The optional table. Requested unconditionally rather than behind
    // #if defined(__APPLE__): the spec says portability_subset must be enabled
    // whenever the device exposes it, which is a property of the driver and not
    // of the host OS, so asking the device is both simpler and more correct than
    // asking the compiler. On a native driver it is absent and the row reports
    // exactly that.
    //
    // A ray query or mesh shader path is added here as one row plus its feature
    // predicate; selectOptionalExtensions already handles dependency chains and
    // is pinned by tests/test_device_extension_selection.cpp.
    static constexpr std::array<const char*, 1> kOptionalExtensionNames{
        kPortabilitySubsetExtensionName,
    };
    const std::array<OptionalExtensionRequest, kOptionalExtensionNames.size()> requests{
        OptionalExtensionRequest{kOptionalExtensionNames[0],
                                 {},
                                 true,
                                 "required by portability-layer drivers such as MoltenVK"},
    };

    plan.optionalOutcomes = selectOptionalExtensions(available, requests);
    // Indexed rather than taking outcome.name.data(): a string_view is not
    // required to be null-terminated, and this is the one place a name reaches a
    // C API. selectOptionalExtensions returns one outcome per request in request
    // order, which tests/test_device_extension_selection.cpp pins.
    for (size_t index = 0; index < plan.optionalOutcomes.size(); ++index) {
        if (plan.optionalOutcomes[index].enabled()) {
            appendUniqueExtension(plan.enabled, kOptionalExtensionNames[index]);
        }
    }
    return plan;
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

void VulkanDevice::initialize(VkInstance instance, VkSurfaceKHR surface, std::filesystem::path shaderDirectory)
{
    instance_ = instance;
    surface_ = surface;
    shaderDirectory_ = std::move(shaderDirectory);

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
    transferQueue_ = VK_NULL_HANDLE;
    transferQueueFamily_ = kNoQueueFamily;
    transferQueueAvailable_ = false;
    queueFamilies_ = {};
    descriptorIndexingEnabled_ = false;
    descriptorUpdateAfterBindEnabled_ = false;
    bufferDeviceAddressEnabled_ = false;
    multiDrawIndirectEnabled_ = false;
    drawIndirectFirstInstanceEnabled_ = false;
    multiviewEnabled_ = false;
    independentBlendEnabled_ = false;
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
        // Needed by the alpha-tested main and shadow fragment shaders; see the
        // enabled13 setup in createLogicalDevice for why `discard` requires it.
        && features13.shaderDemoteToHelperInvocation == VK_TRUE
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

    // Load-time uploads want a DMA family, not just any transfer-capable one.
    // The policy is a pure function so it can be unit tested; see
    // rhi/TransferQueueSelection.h for why there is no graphics-family fallback.
    std::vector<QueueFamilyCapabilities> transferCandidates;
    transferCandidates.reserve(queueFamilyCount);
    for (uint32_t family = 0; family < queueFamilyCount; ++family) {
        const VkQueueFlags flags = familyProperties[family].queueFlags;
        transferCandidates.push_back(QueueFamilyCapabilities{
            (flags & VK_QUEUE_GRAPHICS_BIT) != 0,
            (flags & VK_QUEUE_COMPUTE_BIT) != 0,
            (flags & VK_QUEUE_TRANSFER_BIT) != 0,
            familyProperties[family].queueCount,
        });
    }
    transferQueueFamily_ = selectTransferQueueFamily(transferCandidates);

    std::map<uint32_t, uint32_t> familyQueueCounts;
    familyQueueCounts[graphicsFamily] = 1;
    familyQueueCounts[queueFamilies_.presentFamily.value()] =
        std::max(familyQueueCounts[queueFamilies_.presentFamily.value()], 1u);
    if (asyncComputeQueueFamily_ != UINT32_MAX) {
        familyQueueCounts[asyncComputeQueueFamily_] =
            std::max(familyQueueCounts[asyncComputeQueueFamily_], asyncComputeQueueIndex + 1);
    }
    if (transferQueueFamily_ != kNoQueueFamily) {
        familyQueueCounts[transferQueueFamily_] = std::max(familyQueueCounts[transferQueueFamily_], 1u);
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

    VkPhysicalDeviceVulkan11Features supported11{};
    supported11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    supported11.pNext = &supported13;

    VkPhysicalDeviceVulkan12Features supported12{};
    supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    supported12.pNext = &supported11;

    VkPhysicalDeviceFeatures2 supportedFeatures{};
    supportedFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures.pNext = &supported12;
    vkGetPhysicalDeviceFeatures2(physicalDevice_, &supportedFeatures);

    descriptorIndexingEnabled_ = supportsDescriptorIndexing(supported12);
    descriptorUpdateAfterBindEnabled_ = descriptorIndexingEnabled_ &&
                                        supported12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE;
    bufferDeviceAddressEnabled_ = supported12.bufferDeviceAddress == VK_TRUE;
    multiDrawIndirectEnabled_ = supportedFeatures.features.multiDrawIndirect == VK_TRUE;
    // Per-attachment blend state. The transparent pass wants "over" blending on
    // scene color while velocity and the thin G-buffer are plain overwrites;
    // without this, all attachments must share one blend state.
    independentBlendEnabled_ = supportedFeatures.features.independentBlend == VK_TRUE;
    drawIndirectFirstInstanceEnabled_ = supportedFeatures.features.drawIndirectFirstInstance == VK_TRUE;
    // Draws the whole cascade array in one render pass. Optional: without it the
    // renderer falls back to one pass per cascade, which is what it always did.
    multiviewEnabled_ = supported11.multiview == VK_TRUE;

    VkPhysicalDeviceFeatures enabledCore{};
    enabledCore.multiDrawIndirect = multiDrawIndirectEnabled_ ? VK_TRUE : VK_FALSE;
    enabledCore.drawIndirectFirstInstance = drawIndirectFirstInstanceEnabled_ ? VK_TRUE : VK_FALSE;
    enabledCore.independentBlend = independentBlendEnabled_ ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan13Features enabled13{};
    enabled13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    enabled13.synchronization2 = VK_TRUE;
    enabled13.dynamicRendering = VK_TRUE;
    // SPIR-V 1.6 (what --target-env=vulkan1.3 emits) lowers GLSL `discard` to
    // OpDemoteToHelperInvocation rather than the deprecated OpKill, so every
    // fragment shader with an alpha test needs this bit. Promoted to core in 1.3
    // and required by isDeviceSuitable, so it is always available here.
    enabled13.shaderDemoteToHelperInvocation = VK_TRUE;

    VkPhysicalDeviceVulkan11Features enabled11{};
    enabled11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    enabled11.pNext = &enabled13;
    enabled11.multiview = multiviewEnabled_ ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan12Features enabled12{};
    enabled12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enabled12.pNext = &enabled11;
    enabled12.bufferDeviceAddress = VK_TRUE;
    enabled12.separateDepthStencilLayouts = VK_TRUE;
    enabled12.drawIndirectCount = supported12.drawIndirectCount;
    // Frame pacing. Unconditional, and deliberately not a row in the optional
    // extension table: timelineSemaphore is a mandatory Vulkan 1.2 core feature
    // and isDeviceSuitable already refuses anything below 1.3, so there is no
    // device reaching this line that could answer no. A fallback path here would
    // be unreachable code pretending to be caution.
    enabled12.timelineSemaphore = VK_TRUE;

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
        // Update-after-bind lets the bindless heap be validated against the much
        // larger maxPerStageDescriptorUpdateAfterBind* limits instead of the
        // small non-update-after-bind per-stage limits.
        enabled12.descriptorBindingSampledImageUpdateAfterBind =
            descriptorUpdateAfterBindEnabled_ ? VK_TRUE : VK_FALSE;
    }

    const DeviceExtensionPlan extensionPlan = planDeviceExtensions(physicalDevice_);
    const std::vector<const char*>& enabledExtensions = extensionPlan.enabled;
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
    if (transferQueueFamily_ != kNoQueueFamily) {
        vkGetDeviceQueue(device_, transferQueueFamily_, 0, &transferQueue_);
        transferQueueAvailable_ = transferQueue_ != VK_NULL_HANDLE;
    }
    createPipelineCache();

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    maxDrawIndirectCount_ = properties.limits.maxDrawIndirectCount;
    drawIndexedIndirectCountAvailable_ = supported12.drawIndirectCount == VK_TRUE &&
                                         vkCmdDrawIndexedIndirectCount != nullptr &&
                                         maxDrawIndirectCount_ > 0;

    logCapabilityReport(extensionPlan.optionalOutcomes);
}

void VulkanDevice::logCapabilityReport(std::span<const ExtensionOutcome> optionalExtensions) const
{
    // One table instead of five scattered lines. Every row names the path that is
    // actually live, because "descriptor indexing is not fully supported" answers
    // only half of what a bug report needs; the other half is what the renderer
    // does instead.
    struct Row {
        const char* capability;
        bool active;
        std::string detail;
    };

    const bool multiDrawIndirectActive = multiDrawIndirectEnabled_ && drawIndirectFirstInstanceEnabled_;

    std::vector<Row> rows;
    rows.push_back({"bindless material textures",
                    descriptorIndexingEnabled_,
                    descriptorIndexingEnabled_
                        ? (descriptorUpdateAfterBindEnabled_ ? "descriptor indexing, with update-after-bind"
                                                             : "descriptor indexing, without update-after-bind")
                        : "falling back to per-material descriptor sets"});
    rows.push_back({"multi-draw indirect",
                    multiDrawIndirectActive,
                    multiDrawIndirectActive ? "object data indexed by firstInstance"
                                            : "falling back to per-draw indirect recording"});
    rows.push_back({"indirect draw count",
                    drawIndexedIndirectCountAvailable_,
                    drawIndexedIndirectCountAvailable_
                        ? "vkCmdDrawIndexedIndirectCount, max " + std::to_string(maxDrawIndirectCount_)
                        : "falling back to a CPU-supplied draw count"});
    rows.push_back({"layered cascade shadows",
                    multiviewEnabled_,
                    multiviewEnabled_ ? "multiview, one pass for all cascades"
                                      : "falling back to one pass per cascade"});
    rows.push_back({"independent blend",
                    independentBlendEnabled_,
                    independentBlendEnabled_ ? "per-attachment blend state"
                                             : "falling back to uniform blend state across attachments"});
    rows.push_back({"async compute queue",
                    asyncComputeAvailable_,
                    asyncComputeAvailable_
                        ? (asyncComputeDedicatedFamily_ ? "dedicated compute-only family "
                                                        : "second queue in graphics family ") +
                              std::to_string(asyncComputeQueueFamily_)
                        : "compute passes stay on the graphics queue"});
    rows.push_back({"transfer queue",
                    transferQueueAvailable_,
                    transferQueueAvailable_ ? "dedicated DMA family " + std::to_string(transferQueueFamily_)
                                            : "uploads stay on the graphics queue"});

    size_t labelWidth = 0;
    for (const Row& row : rows) {
        labelWidth = std::max(labelWidth, std::strlen(row.capability));
    }
    for (const ExtensionOutcome& outcome : optionalExtensions) {
        labelWidth = std::max(labelWidth, outcome.name.size());
    }

    std::string message = "Device capability report:";
    const auto appendRow = [&message, labelWidth](std::string_view label, bool active, const std::string& detail) {
        message += "\n  ";
        message += label;
        message.append(labelWidth - label.size(), ' ');
        message += active ? " : on  -- " : " : off -- ";
        message += detail;
    };
    for (const Row& row : rows) {
        appendRow(row.capability, row.active, row.detail);
    }
    for (const ExtensionOutcome& outcome : optionalExtensions) {
        std::string detail = extensionDecisionName(outcome.decision);
        if (!outcome.blockedBy.empty()) {
            detail += " (";
            detail += outcome.blockedBy;
            detail += ")";
        }
        if (!outcome.purpose.empty()) {
            detail += "; ";
            detail += outcome.purpose;
        }
        appendRow(outcome.name, outcome.enabled(), detail);
    }
    Logger::info(message);

    // A single warn keeps a degraded run greppable at one level, which is what
    // the individual warn lines used to provide.
    std::string degraded;
    for (const Row& row : rows) {
        if (!row.active) {
            degraded += degraded.empty() ? "" : ", ";
            degraded += row.capability;
        }
    }
    if (!degraded.empty()) {
        Logger::warn("Running on fallback paths for: " + degraded +
                     ". The capability report above says what each one does instead. "
                     "(On MoltenVK, MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES=1 exposes the extra queues.)");
    }
}

void VulkanDevice::createPipelineCache()
{
    assert(device_ != VK_NULL_HANDLE && "createPipelineCache() requires a live logical device.");

    // Bind the cache to the exact SPIR-V it was built from. The driver header
    // only identifies the GPU and driver, so a blob written before a shader
    // interface change looks valid to the driver; MoltenVK then fails pipeline
    // creation outright instead of dropping the stale entries. Hashing the
    // compiled modules turns that into a cache miss.
    shaderHash_ = hashShaderDirectory(shaderDirectory_);
    if (shaderHash_ == 0) {
        Logger::warn("Could not hash the compiled shader directory (" + shaderDirectory_.string() +
                     "); the pipeline cache will not be invalidated by shader changes.");
    }

    // Load any previously saved blob and validate it against this exact GPU +
    // driver + shader set before handing it to the driver as initial data. On
    // any mismatch we build an empty cache so the driver is never fed a
    // foreign/stale blob.
    const std::filesystem::path cachePath = pipelineCacheFilePath();
    const std::vector<std::byte> blob = readPipelineCacheBlob(cachePath);

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice_, &properties);
    const PipelineCacheContents contents = decodePipelineCacheBlob(blob, properties, shaderHash_);
    const bool usable = contents.status == PipelineCacheStatus::Usable;

    VkPipelineCacheCreateInfo cacheInfo{};
    cacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheInfo.initialDataSize = usable ? contents.driverData.size() : 0;
    cacheInfo.pInitialData = usable ? contents.driverData.data() : nullptr;

    VK_CHECK(vkCreatePipelineCache(device_, &cacheInfo, nullptr, &pipelineCache_));
    debug::setObjectName(device_, pipelineCache_, VK_OBJECT_TYPE_PIPELINE_CACHE, "EnginePipelineCache");

    switch (contents.status) {
    case PipelineCacheStatus::Usable:
        Logger::info("Loaded pipeline cache (" + std::to_string(contents.driverData.size()) + " bytes) from " +
                     cachePath.string() + ".");
        break;
    case PipelineCacheStatus::Missing:
        Logger::info("No usable pipeline cache found; starting from an empty cache.");
        break;
    case PipelineCacheStatus::ShaderMismatch:
        Logger::info("Discarded pipeline cache at " + cachePath.string() +
                     " (compiled shaders changed since it was written); starting from an empty cache.");
        break;
    case PipelineCacheStatus::DeviceMismatch:
        Logger::warn("Discarded incompatible pipeline cache at " + cachePath.string() +
                     " (GPU/driver mismatch); starting from an empty cache.");
        break;
    case PipelineCacheStatus::Malformed:
        Logger::warn("Discarded unrecognized pipeline cache at " + cachePath.string() +
                     " (written by an older engine build, or truncated); starting from an empty cache.");
        break;
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
                // Stamp the digest captured at startup, not a fresh one: these
                // pipelines were compiled from the modules loaded back then, so
                // a shader edited mid-run must still invalidate the blob.
                const std::vector<std::byte> blob = encodePipelineCacheBlob(data, shaderHash_);
                if (writePipelineCacheBlob(cachePath, blob)) {
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
