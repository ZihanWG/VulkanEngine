#include "rhi/AsyncComputeQueueSelection.h"

namespace ve::rhi {

AsyncComputeQueueChoice selectAsyncComputeQueue(std::span<const QueueFamilyCapabilities> families,
                                                uint32_t graphicsFamily)
{
    for (uint32_t family = 0; family < families.size(); ++family) {
        const QueueFamilyCapabilities& capabilities = families[family];
        if (capabilities.compute && !capabilities.graphics && capabilities.queueCount >= 1) {
            return {family, 0, true};
        }
    }

    if (graphicsFamily < families.size() && families[graphicsFamily].queueCount >= 2) {
        return {graphicsFamily, 1, false};
    }

    return {};
}

} // namespace ve::rhi
