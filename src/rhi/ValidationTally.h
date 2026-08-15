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

    [[nodiscard]] static uint64_t errorCount();
    [[nodiscard]] static uint64_t warningCount();

    static void reset();
};

} // namespace ve::rhi
