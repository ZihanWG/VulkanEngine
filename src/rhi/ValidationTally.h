#pragma once

// Process-wide tally of Vulkan validation-layer messages by severity.
//
// Exists so a headless run can *fail* on validation output rather than leave it
// for a human to notice in a log. The debug messenger already routes messages
// into the engine logger; this only counts them, and counting is separate from
// the reporting policy: the callback always tallies, and whether a non-zero
// tally ends the process is Application's decision.
//
// Thread-safe: the validation layer may invoke the messenger from any thread
// that makes Vulkan calls.

#include <cstdint>

namespace ve::rhi {

class ValidationTally final {
public:
    static void recordError();
    static void recordWarning();
    // A synchronization-validation finding. Counted separately from, and in
    // addition to, the error it already is.
    //
    // Apart because it answers a different question. An ordinary validation
    // error says a Vulkan call was malformed; a sync hazard says every call was
    // legal and the ordering between them was not, which is the one class of
    // mistake the render graph's inferred barriers can make. Separating them is
    // also what lets the self-test assert it provoked the hazard it meant to
    // rather than some unrelated error.
    static void recordSyncHazard();

    [[nodiscard]] static uint64_t errorCount();
    [[nodiscard]] static uint64_t warningCount();
    [[nodiscard]] static uint64_t syncHazardCount();

    static void reset();
};

} // namespace ve::rhi
