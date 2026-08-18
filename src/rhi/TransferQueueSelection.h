#pragma once

// Which queue family, if any, should carry load-time texture uploads.
//
// Split out from VulkanDevice so the choice is a pure function of the family
// list rather than something only observable in a log line: on this project's
// primary machine the answer is "none" unless a MoltenVK environment variable is
// set, so the fallback is the case that actually needs pinning.

#include <cstdint>
#include <span>

namespace ve::rhi {

// The subset of VkQueueFamilyProperties this decision depends on. Mirrored as a
// plain struct so the policy is testable without a device.
struct QueueFamilyCapabilities {
    bool graphics = false;
    bool compute = false;
    bool transfer = false;
    uint32_t queueCount = 0;
};

inline constexpr uint32_t kNoQueueFamily = UINT32_MAX;

// A true DMA family: transfer-capable and neither graphics nor compute.
//
// There is deliberately no "second queue in the graphics family" fallback, which
// is what the async compute selection does. A second graphics queue is still the
// graphics ring, so it would buy nothing for uploads while still requiring queue
// family ownership transfers. Returns kNoQueueFamily when there is none, and
// uploads then stay on the graphics queue exactly as before.
[[nodiscard]] uint32_t selectTransferQueueFamily(std::span<const QueueFamilyCapabilities> families);

} // namespace ve::rhi
