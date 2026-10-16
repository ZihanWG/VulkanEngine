#pragma once

// Which optional device extensions a run enables, and why each refusal happened.
//
// Split out of VulkanDevice for the same reason as TransferQueueSelection: the
// decision is a pure function of what the device reports and what the renderer
// asks for, so it can be pinned by tests rather than inferred from a log line on
// one machine. The engine requires exactly one device extension (swapchain) and
// has, until now, had no machinery for optional ones at all -- the Apple
// portability subset was an #if in the middle of the list. Anything with a
// raster fallback (ray query for probe capture, mesh shaders for the geometry
// path) needs that machinery before it can be added as a path rather than a
// requirement.
//
// Vulkan-free on purpose. Extensions are names and dependencies are names; the
// caller evaluates whatever VkPhysicalDevice*Features predicate an extension
// needs and passes the answer in as a bool. That keeps this in a translation
// unit the headless tests already link.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ve::rhi {

// One optional extension the renderer would like to use.
//
// All string_views must outlive the returned outcomes -- in practice they are
// the VK_*_EXTENSION_NAME literals, which have static storage duration.
struct OptionalExtensionRequest {
    std::string_view name;
    // Extension names this one requires. Vulkan says an extension's dependencies
    // must be enabled alongside it; enabling a dependent without its dependency
    // is a validation error, not a graceful degradation, so a request whose
    // dependency was refused has to be refused too rather than half-enabled.
    std::span<const std::string_view> dependencies;
    // The caller's answer to "are the features this extension needs supported?".
    // Defaults to true so an extension that needs no feature bits can omit it.
    bool featuresSupported = true;
    // What path this unlocks. Reported, never part of the decision.
    std::string_view purpose;
    // The run asked for this path to stay off -- a startup flag such as
    // --disable-ray-query, which puts the fallback in reach of a device that
    // could take the real path. Refused like any other refusal, so dependents
    // come down with it, but reported under its own reason: "the device cannot"
    // and "this run chose not to" lead to different conclusions.
    bool declined = false;
};

enum class ExtensionDecision : uint8_t {
    Enabled,
    // The device does not expose it.
    ExtensionUnavailable,
    // It exists, but something it depends on is unavailable or was itself refused.
    DependencyUnavailable,
    // It exists, but the caller's feature predicate said no.
    FeaturesUnsupported,
    // It exists, but the request said to leave it off.
    Declined,
};

struct ExtensionOutcome {
    std::string_view name;
    ExtensionDecision decision = ExtensionDecision::ExtensionUnavailable;
    std::string_view purpose;
    // Which dependency blocked it. Empty unless decision is DependencyUnavailable.
    std::string_view blockedBy;

    [[nodiscard]] bool enabled() const
    {
        return decision == ExtensionDecision::Enabled;
    }
};

// Decide every request. One outcome per request, in request order.
//
// Dependency resolution runs to a fixed point rather than assuming the table is
// sorted, so a request may name a dependency that appears after it. Refusing a
// dependency refuses its dependents transitively. Order-independence is a
// property worth having for the same reason TransientMemoryPlan has it: a policy
// whose answer depends on how the table happens to be written is one that
// changes when someone reorders it for readability.
[[nodiscard]] std::vector<ExtensionOutcome>
selectOptionalExtensions(std::span<const std::string_view> available,
                         std::span<const OptionalExtensionRequest> requests);

// Stable spelling for logs and test failure messages.
[[nodiscard]] const char* extensionDecisionName(ExtensionDecision decision);

// --- Ray query ---------------------------------------------------------------
//
// The rows a ray query path needs, as a function rather than a literal in
// VulkanDevice.cpp, so the table the device is actually created from is the one
// the tests exercise. Vulkan's chain: ray query needs acceleration structures,
// which need deferred host operations. Its other dependencies (SPIR-V 1.4,
// descriptor indexing, buffer device address) are Vulkan 1.2 core, and this
// renderer refuses anything below 1.3.
//
// Spelled as literals because this unit is Vulkan-free; VulkanDevice.cpp pins
// each against its VK_*_EXTENSION_NAME macro with a static_assert.
inline constexpr std::string_view kDeferredHostOperationsExtensionName = "VK_KHR_deferred_host_operations";
inline constexpr std::string_view kAccelerationStructureExtensionName = "VK_KHR_acceleration_structure";
inline constexpr std::string_view kRayQueryExtensionName = "VK_KHR_ray_query";

// The two feature bits the path needs, as the device reports them. Both false
// when the extension is absent: the feature struct is only queried for an
// extension the device exposes.
struct RayQueryFeatureSupport {
    // VkPhysicalDeviceAccelerationStructureFeaturesKHR::accelerationStructure
    bool accelerationStructure = false;
    // VkPhysicalDeviceRayQueryFeaturesKHR::rayQuery
    bool rayQuery = false;
};

inline constexpr size_t kRayQueryExtensionRequestCount = 3;

// Every row carries `declined` when the run asked for ray query to stay off.
// All three rather than only the top one, so the device is created exactly as
// one without ray tracing would be -- leaving acceleration structures enabled
// and unused would make the fallback run differ from the real one in a way
// nothing reports.
[[nodiscard]] std::array<OptionalExtensionRequest, kRayQueryExtensionRequestCount>
rayQueryExtensionRequests(RayQueryFeatureSupport features, bool declined);

// Whether the ray query path survived selection. Asks for both extensions by
// name rather than trusting propagation alone, so an outcome list that came
// from some other table cannot report a path whose dependency is missing.
[[nodiscard]] bool rayQueryEnabled(std::span<const ExtensionOutcome> outcomes);

} // namespace ve::rhi
