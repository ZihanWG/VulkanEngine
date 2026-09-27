#pragma once

// --portability-fallbacks: run a capable GPU on the paths MoltenVK takes.
//
// Three optional capabilities are missing under MoltenVK's default
// configuration and present on the RTX machine and on lavapipe: indirect draw
// count, an async compute queue, and a dedicated transfer queue. Every path the
// renderer takes without them -- fixed-slot indirect commands with zeroed
// tails, compute passes inline on the graphics queue, uploads without an
// ownership transfer -- therefore ran only on the Mac, so a change that broke
// one of them could not fail anywhere else. Forcing the capabilities off puts
// those paths in reach of any GPU and of CI.
//
// Queue families are rewritten rather than the selections overridden, so the
// real selection code runs and reaches its own "none" on the rewritten list,
// exactly as it does on the Mac.

#include "rhi/TransferQueueSelection.h"

#include <span>
#include <vector>

namespace ve::rhi {

// The family list as MoltenVK presents it without
// MVK_CONFIG_SPECIALIZED_QUEUE_FAMILIES: the same number of families, each
// GRAPHICS|COMPUTE|TRANSFER with one queue. No family is compute-only or
// transfer-only, and no family offers a second queue, so neither the async
// compute nor the transfer selection finds anything. Family indices are
// preserved, because the graphics and present families were chosen from the
// real list and are still what the device is created with.
[[nodiscard]] std::vector<QueueFamilyCapabilities>
moltenVkDefaultQueueFamilies(std::span<const QueueFamilyCapabilities> families);

} // namespace ve::rhi
