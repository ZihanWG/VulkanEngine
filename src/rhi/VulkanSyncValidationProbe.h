#pragma once

// Negative control for synchronization validation.
//
// Turning the layer's synchronization validation on and then rendering a clean
// frame proves nothing on its own: a setting the layer silently ignored, a
// spelling it did not recognise, or a loader that dropped the settings chain all
// look exactly like a frame with no hazards in it. This provokes a hazard the
// layer must report, so a run can state that the check was live rather than
// assume it.
//
// The hazard is deliberate and the errors it produces are this probe's own. The
// caller is expected to treat a self-test run as a self-test run and not as
// evidence about the frame -- see Application's handling of
// --sync-validation-selftest.

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <string>

namespace ve::rhi {

class VulkanContext;
class VulkanCommandContext;

struct SyncValidationProbeResult {
    // The layer reported at least one synchronization hazard while the probe ran.
    // False is a failure of the check, not of the engine: it means nothing is
    // watching the ordering, so any later "clean" result is worthless.
    bool hazardReported = false;

    // What the tally moved by across the probe. Errors that are not sync hazards
    // are reported separately because they would mean the probe itself is
    // malformed -- it must provoke an ordering mistake and nothing else.
    uint64_t syncHazardsObserved = 0;
    uint64_t otherErrorsObserved = 0;

    std::string detail;
};

// Records two overlapping writes to one buffer with no barrier between them,
// submits, and waits. Allocates and destroys everything it uses; leaves no state
// behind beyond the validation messages it deliberately caused.
[[nodiscard]] SyncValidationProbeResult probeSynchronizationValidation(VulkanContext& context,
                                                                       const VulkanCommandContext& commandContext);

} // namespace ve::rhi
