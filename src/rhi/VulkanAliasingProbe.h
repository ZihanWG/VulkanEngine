#pragma once

// Phase 1 capability probe for the render graph transient memory allocator.
//
// Answers one question before anything is built on the answer: can two images be
// bound into a single VMA allocation at different offsets on this driver, with
// the validation layer quiet? MoltenVK reaches Vulkan memory through Metal
// placement heaps, and nothing in the engine has exercised that path before.
//
// Throwaway. Once VulkanTransientMemoryPool exists this probe is superseded by
// the pool's own capability check and should be deleted along with its flag.

#include "rhi/VulkanCommon.h"

#include <cstdint>
#include <string>

namespace ve::rhi {

class VulkanContext;
class VulkanCommandContext;

struct AliasingProbeResult {
    bool supported = false;

    // Clears one alias and reads the other back. vmaCreateAliasingImage2
    // returning VK_SUCCESS only proves the API accepted the binding; a driver
    // that quietly handed each image private memory would look identical until a
    // transient resource silently stopped being overwritten.
    //
    // Asymmetric evidence, deliberately: true proves the two images share bytes.
    // False does NOT prove they do not -- reading an alias requires transitioning
    // it from VK_IMAGE_LAYOUT_UNDEFINED, which the spec licenses to discard
    // contents. So false is reported as "not observable", never as a failure.
    bool aliasedWriteObserved = false;

    // Memory the driver wanted for each image, and what the shared allocation
    // ended up costing. Aliasing is only worth anything if these differ.
    VkDeviceSize firstImageBytes = 0;
    VkDeviceSize secondImageBytes = 0;
    VkDeviceSize sharedAllocationBytes = 0;
    VkDeviceSize secondImageOffset = 0;

    // Intersection of both images' memoryTypeBits. An empty intersection is the
    // one failure the real pool must fall back from rather than treat as a bug.
    uint32_t commonMemoryTypeBits = 0;

    std::string detail;
};

// Allocates, binds, and immediately destroys. Records nothing and leaves no
// state behind; safe to call once at startup.
[[nodiscard]] AliasingProbeResult probeImageMemoryAliasing(VulkanContext& context,
                                                          const VulkanCommandContext& commandContext);

} // namespace ve::rhi
