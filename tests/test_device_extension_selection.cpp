#include "rhi/DeviceExtensionSelection.h"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string_view>
#include <vector>

using ve::rhi::ExtensionDecision;
using ve::rhi::ExtensionOutcome;
using ve::rhi::extensionDecisionName;
using ve::rhi::OptionalExtensionRequest;
using ve::rhi::selectOptionalExtensions;

namespace {

// Real extension names, so a reader can tell at a glance which of these
// relationships Vulkan actually imposes. Ray query needs acceleration
// structure, which needs deferred host operations.
constexpr std::string_view kRayQuery = "VK_KHR_ray_query";
constexpr std::string_view kAccelerationStructure = "VK_KHR_acceleration_structure";
constexpr std::string_view kDeferredHostOperations = "VK_KHR_deferred_host_operations";
constexpr std::string_view kMeshShader = "VK_EXT_mesh_shader";

std::vector<ExtensionOutcome> select(const std::vector<std::string_view>& available,
                                     const std::vector<OptionalExtensionRequest>& requests)
{
    return selectOptionalExtensions(available, requests);
}

const ExtensionOutcome& outcomeFor(const std::vector<ExtensionOutcome>& outcomes, std::string_view name)
{
    for (const ExtensionOutcome& outcome : outcomes) {
        if (outcome.name == name) {
            return outcome;
        }
    }
    FAIL("no outcome for " << name);
    return outcomes.front();
}

} // namespace

TEST_CASE("An empty request table produces no outcomes")
{
    CHECK(select({kRayQuery}, {}).empty());
}

TEST_CASE("An extension the device exposes with its features supported is enabled")
{
    const std::vector<ExtensionOutcome> outcomes =
        select({kMeshShader}, {OptionalExtensionRequest{kMeshShader, {}, true, "mesh shading path"}});

    REQUIRE(outcomes.size() == 1);
    CHECK(outcomes[0].decision == ExtensionDecision::Enabled);
    CHECK(outcomes[0].enabled());
    CHECK(outcomes[0].purpose == "mesh shading path");
    CHECK(outcomes[0].blockedBy.empty());
}

TEST_CASE("An extension the device does not expose is refused")
{
    const std::vector<ExtensionOutcome> outcomes =
        select({kDeferredHostOperations}, {OptionalExtensionRequest{kMeshShader, {}, true, "mesh shading path"}});

    REQUIRE(outcomes.size() == 1);
    CHECK(outcomes[0].decision == ExtensionDecision::ExtensionUnavailable);
    CHECK_FALSE(outcomes[0].enabled());
}

TEST_CASE("An exposed extension whose features are unsupported is refused for that reason")
{
    // The distinction matters in a bug report: "the driver does not have it" and
    // "the driver has it but this GPU cannot do it" send you to different places.
    const std::vector<ExtensionOutcome> outcomes =
        select({kMeshShader}, {OptionalExtensionRequest{kMeshShader, {}, false, "mesh shading path"}});

    REQUIRE(outcomes.size() == 1);
    CHECK(outcomes[0].decision == ExtensionDecision::FeaturesUnsupported);
}

TEST_CASE("Absence outranks an unsupported feature predicate")
{
    // An extension that is not there cannot have had its features evaluated
    // meaningfully, so the report must say it is missing rather than blaming a
    // predicate the caller filled in speculatively.
    const std::vector<ExtensionOutcome> outcomes =
        select({}, {OptionalExtensionRequest{kMeshShader, {}, false, "mesh shading path"}});

    REQUIRE(outcomes.size() == 1);
    CHECK(outcomes[0].decision == ExtensionDecision::ExtensionUnavailable);
}

TEST_CASE("A dependency the device lacks refuses the dependent and names it")
{
    static constexpr std::array<std::string_view, 1> kDeps{kAccelerationStructure};
    const std::vector<ExtensionOutcome> outcomes =
        select({kRayQuery}, {OptionalExtensionRequest{kRayQuery, kDeps, true, "ray-queried probe capture"}});

    REQUIRE(outcomes.size() == 1);
    CHECK(outcomes[0].decision == ExtensionDecision::DependencyUnavailable);
    CHECK(outcomes[0].blockedBy == kAccelerationStructure);
}

TEST_CASE("A dependency that is itself a refused request refuses the dependent")
{
    // Enabling a dependent whose dependency the table refused is a validation
    // error, not a degraded path -- so a refusal has to propagate rather than
    // leaving the dependent enabled because the device happened to expose it.
    static constexpr std::array<std::string_view, 1> kDeps{kAccelerationStructure};
    const std::vector<ExtensionOutcome> outcomes = select(
        {kRayQuery, kAccelerationStructure},
        {
            OptionalExtensionRequest{kAccelerationStructure, {}, false, "BLAS/TLAS"},
            OptionalExtensionRequest{kRayQuery, kDeps, true, "ray-queried probe capture"},
        });

    REQUIRE(outcomes.size() == 2);
    CHECK(outcomeFor(outcomes, kAccelerationStructure).decision == ExtensionDecision::FeaturesUnsupported);
    CHECK(outcomeFor(outcomes, kRayQuery).decision == ExtensionDecision::DependencyUnavailable);
    CHECK(outcomeFor(outcomes, kRayQuery).blockedBy == kAccelerationStructure);
}

TEST_CASE("A refusal collapses the whole dependency chain")
{
    static constexpr std::array<std::string_view, 1> kRayQueryDeps{kAccelerationStructure};
    static constexpr std::array<std::string_view, 1> kAccelDeps{kDeferredHostOperations};

    // Deferred host operations is the one the device lacks; both extensions
    // above it must come down with it.
    const std::vector<ExtensionOutcome> outcomes = select(
        {kRayQuery, kAccelerationStructure},
        {
            OptionalExtensionRequest{kRayQuery, kRayQueryDeps, true, "ray-queried probe capture"},
            OptionalExtensionRequest{kAccelerationStructure, kAccelDeps, true, "BLAS/TLAS"},
            OptionalExtensionRequest{kDeferredHostOperations, {}, true, "parallel AS builds"},
        });

    REQUIRE(outcomes.size() == 3);
    CHECK(outcomeFor(outcomes, kDeferredHostOperations).decision == ExtensionDecision::ExtensionUnavailable);
    CHECK(outcomeFor(outcomes, kAccelerationStructure).decision == ExtensionDecision::DependencyUnavailable);
    CHECK(outcomeFor(outcomes, kRayQuery).decision == ExtensionDecision::DependencyUnavailable);
}

TEST_CASE("The answer does not depend on how the table is ordered")
{
    // A dependent listed before its dependency resolves identically. Without the
    // fixed point this passes only when the table happens to be sorted, and
    // breaks silently the first time someone regroups it for readability.
    static constexpr std::array<std::string_view, 1> kRayQueryDeps{kAccelerationStructure};
    static constexpr std::array<std::string_view, 1> kAccelDeps{kDeferredHostOperations};

    const std::vector<std::string_view> available{kRayQuery, kAccelerationStructure};
    const OptionalExtensionRequest rayQuery{kRayQuery, kRayQueryDeps, true, "ray-queried probe capture"};
    const OptionalExtensionRequest accel{kAccelerationStructure, kAccelDeps, true, "BLAS/TLAS"};
    const OptionalExtensionRequest deferred{kDeferredHostOperations, {}, true, "parallel AS builds"};

    const std::vector<ExtensionOutcome> forward = select(available, {rayQuery, accel, deferred});
    const std::vector<ExtensionOutcome> reversed = select(available, {deferred, accel, rayQuery});

    for (const std::string_view name : {kRayQuery, kAccelerationStructure, kDeferredHostOperations}) {
        CHECK(outcomeFor(forward, name).decision == outcomeFor(reversed, name).decision);
    }
}

TEST_CASE("A dependency the table never requests is judged on device availability alone")
{
    // Dependencies are not required to be requests themselves: swapchain is
    // already a hard requirement elsewhere, so naming it as a dependency should
    // consult the device list rather than demand a redundant table row.
    static constexpr std::array<std::string_view, 1> kDeps{kDeferredHostOperations};

    const std::vector<ExtensionOutcome> present = select(
        {kAccelerationStructure, kDeferredHostOperations},
        {OptionalExtensionRequest{kAccelerationStructure, kDeps, true, "BLAS/TLAS"}});
    CHECK(present[0].decision == ExtensionDecision::Enabled);

    const std::vector<ExtensionOutcome> absent = select(
        {kAccelerationStructure},
        {OptionalExtensionRequest{kAccelerationStructure, kDeps, true, "BLAS/TLAS"}});
    CHECK(absent[0].decision == ExtensionDecision::DependencyUnavailable);
}

TEST_CASE("Outcomes come back in request order")
{
    const std::vector<ExtensionOutcome> outcomes = select({kMeshShader},
                                                          {
                                                              OptionalExtensionRequest{kRayQuery, {}, true, "a"},
                                                              OptionalExtensionRequest{kMeshShader, {}, true, "b"},
                                                          });

    REQUIRE(outcomes.size() == 2);
    CHECK(outcomes[0].name == kRayQuery);
    CHECK(outcomes[1].name == kMeshShader);
}

TEST_CASE("Every decision has a distinct spelling for the startup report")
{
    // The report is the only place a user sees why a path is off, so two
    // decisions rendering identically would make it useless.
    const std::array<ExtensionDecision, 4> decisions{
        ExtensionDecision::Enabled,
        ExtensionDecision::ExtensionUnavailable,
        ExtensionDecision::DependencyUnavailable,
        ExtensionDecision::FeaturesUnsupported,
    };
    for (size_t i = 0; i < decisions.size(); ++i) {
        CHECK(std::string_view(extensionDecisionName(decisions[i])).size() > 0);
        for (size_t j = i + 1; j < decisions.size(); ++j) {
            CHECK(std::string_view(extensionDecisionName(decisions[i])) !=
                  std::string_view(extensionDecisionName(decisions[j])));
        }
    }
}
