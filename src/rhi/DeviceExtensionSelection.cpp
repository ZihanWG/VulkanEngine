#include "rhi/DeviceExtensionSelection.h"

#include <algorithm>

namespace ve::rhi {

namespace {

[[nodiscard]] bool contains(std::span<const std::string_view> names, std::string_view name)
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

// Index of the request declaring `name`, or requests.size() when the caller is
// not asking for it. A dependency the table never mentions is judged purely on
// device availability; one it does mention has to have survived its own checks.
[[nodiscard]] size_t requestIndexFor(std::span<const OptionalExtensionRequest> requests, std::string_view name)
{
    for (size_t index = 0; index < requests.size(); ++index) {
        if (requests[index].name == name) {
            return index;
        }
    }
    return requests.size();
}

} // namespace

std::vector<ExtensionOutcome> selectOptionalExtensions(std::span<const std::string_view> available,
                                                       std::span<const OptionalExtensionRequest> requests)
{
    std::vector<ExtensionOutcome> outcomes;
    outcomes.reserve(requests.size());

    // First pass judges each request on its own: present on the device, wanted by
    // this run, and the features the caller checked. Dependencies come after,
    // because a dependency may be another request that has not been judged yet.
    //
    // Absence outranks a decline: an extension the device lacks was never a
    // choice, and the report's first job is to say what the device can do. A
    // decline outranks the feature predicate, which the caller may have filled in
    // without the path ever being wanted.
    for (const OptionalExtensionRequest& request : requests) {
        ExtensionOutcome outcome{};
        outcome.name = request.name;
        outcome.purpose = request.purpose;
        if (!contains(available, request.name)) {
            outcome.decision = ExtensionDecision::ExtensionUnavailable;
        } else if (request.declined) {
            outcome.decision = ExtensionDecision::Declined;
        } else if (!request.featuresSupported) {
            outcome.decision = ExtensionDecision::FeaturesUnsupported;
        } else {
            outcome.decision = ExtensionDecision::Enabled;
        }
        outcomes.push_back(outcome);
    }

    // Then withdraw anything whose dependency did not survive, repeating until
    // nothing changes so a chain A->B->C collapses whichever order it is written
    // in. Bounded by the request count: each sweep withdraws at least one
    // request or stops, and a request is never re-enabled.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t index = 0; index < requests.size(); ++index) {
            if (outcomes[index].decision != ExtensionDecision::Enabled) {
                continue;
            }
            for (const std::string_view dependency : requests[index].dependencies) {
                const size_t dependencyRequest = requestIndexFor(requests, dependency);
                const bool satisfied = dependencyRequest < requests.size() ? outcomes[dependencyRequest].enabled()
                                                                           : contains(available, dependency);
                if (!satisfied) {
                    outcomes[index].decision = ExtensionDecision::DependencyUnavailable;
                    outcomes[index].blockedBy = dependency;
                    changed = true;
                    break;
                }
            }
        }
    }

    return outcomes;
}

const char* extensionDecisionName(ExtensionDecision decision)
{
    switch (decision) {
    case ExtensionDecision::Enabled:
        return "enabled";
    case ExtensionDecision::ExtensionUnavailable:
        return "not exposed by the device";
    case ExtensionDecision::DependencyUnavailable:
        return "dependency unavailable";
    case ExtensionDecision::FeaturesUnsupported:
        return "required features unsupported";
    case ExtensionDecision::Declined:
        return "declined by a startup flag";
    }
    return "unknown";
}

std::array<OptionalExtensionRequest, kRayQueryExtensionRequestCount>
rayQueryExtensionRequests(RayQueryFeatureSupport features, bool declined)
{
    // Static storage: OptionalExtensionRequest holds a span into these, and the
    // outcomes outlive this call.
    static constexpr std::array<std::string_view, 1> kAccelerationStructureDependencies{
        kDeferredHostOperationsExtensionName};
    static constexpr std::array<std::string_view, 1> kRayQueryDependencies{kAccelerationStructureExtensionName};

    return {
        OptionalExtensionRequest{
            kDeferredHostOperationsExtensionName, {}, true, "required by VK_KHR_acceleration_structure", declined},
        OptionalExtensionRequest{kAccelerationStructureExtensionName,
                                 kAccelerationStructureDependencies,
                                 features.accelerationStructure,
                                 "bottom- and top-level acceleration structures",
                                 declined},
        OptionalExtensionRequest{kRayQueryExtensionName,
                                 kRayQueryDependencies,
                                 features.rayQuery,
                                 "ray queries from fragment shaders",
                                 declined},
    };
}

bool rayQueryEnabled(std::span<const ExtensionOutcome> outcomes)
{
    bool accelerationStructure = false;
    bool rayQuery = false;
    for (const ExtensionOutcome& outcome : outcomes) {
        if (outcome.name == kAccelerationStructureExtensionName) {
            accelerationStructure = outcome.enabled();
        } else if (outcome.name == kRayQueryExtensionName) {
            rayQuery = outcome.enabled();
        }
    }
    return accelerationStructure && rayQuery;
}

} // namespace ve::rhi
