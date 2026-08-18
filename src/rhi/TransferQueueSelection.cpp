#include "rhi/TransferQueueSelection.h"

namespace ve::rhi {

uint32_t selectTransferQueueFamily(std::span<const QueueFamilyCapabilities> families)
{
    for (uint32_t family = 0; family < families.size(); ++family) {
        const QueueFamilyCapabilities& capabilities = families[family];
        if (capabilities.transfer && !capabilities.graphics && !capabilities.compute
            && capabilities.queueCount >= 1) {
            return family;
        }
    }

    return kNoQueueFamily;
}

} // namespace ve::rhi
