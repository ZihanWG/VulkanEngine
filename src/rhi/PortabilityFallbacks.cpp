#include "rhi/PortabilityFallbacks.h"

namespace ve::rhi {

std::vector<QueueFamilyCapabilities> moltenVkDefaultQueueFamilies(std::span<const QueueFamilyCapabilities> families)
{
    return std::vector<QueueFamilyCapabilities>(families.size(), QueueFamilyCapabilities{true, true, true, 1});
}

} // namespace ve::rhi
