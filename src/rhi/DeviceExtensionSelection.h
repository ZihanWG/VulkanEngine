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
};

enum class ExtensionDecision : uint8_t {
    Enabled,
    // The device does not expose it.
    ExtensionUnavailable,
    // It exists, but something it depends on is unavailable or was itself refused.
    DependencyUnavailable,
    // It exists, but the caller's feature predicate said no.
    FeaturesUnsupported,
};

struct ExtensionOutcome {
    std::string_view name;
    ExtensionDecision decision = ExtensionDecision::ExtensionUnavailable;
    std::string_view purpose;
    // Which dependency blocked it. Empty unless decision is DependencyUnavailable.
    std::string_view blockedBy;

    [[nodiscard]] bool enabled() const { return decision == ExtensionDecision::Enabled; }
};

// Decide every request. One outcome per request, in request order.
//
// Dependency resolution runs to a fixed point rather than assuming the table is
// sorted, so a request may name a dependency that appears after it. Refusing a
// dependency refuses its dependents transitively. Order-independence is a
// property worth having for the same reason TransientMemoryPlan has it: a policy
// whose answer depends on how the table happens to be written is one that
// changes when someone reorders it for readability.
[[nodiscard]] std::vector<ExtensionOutcome> selectOptionalExtensions(
    std::span<const std::string_view> available,
    std::span<const OptionalExtensionRequest> requests);

// Stable spelling for logs and test failure messages.
[[nodiscard]] const char* extensionDecisionName(ExtensionDecision decision);

} // namespace ve::rhi
