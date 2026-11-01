#pragma once

// Which enumerated physical device the renderer runs on.
//
// Split out from VulkanDevice because one case is real and was invisible: a
// single GPU can be enumerated once per installed driver. An Apple M3 with both
// MoltenVK and KosmicKrisp installed shows up twice, as the same device name with
// identical scores, and the first one enumerated wins. That is still the rule --
// the loader's order is the one the user can steer, through VK_DRIVER_FILES --
// but the caller now learns that the choice was a tie, so it can say so.

#include <cstddef>
#include <span>
#include <vector>

namespace ve::rhi {

// Score of a device that failed the suitability checks. Any suitable device
// scores at least zero.
inline constexpr int kUnsuitableDeviceScore = -1;

struct PhysicalDeviceChoice {
    static constexpr size_t kNone = static_cast<size_t>(-1);

    size_t index = kNone;
    // Every other suitable device with the same score as the chosen one, in
    // enumeration order. Non-empty means the loader's order decided.
    std::vector<size_t> tiedWith;

    [[nodiscard]] bool available() const
    {
        return index != kNone;
    }
};

// The first device with the highest score, ignoring kUnsuitableDeviceScore, plus
// whatever tied with it.
[[nodiscard]] PhysicalDeviceChoice selectPhysicalDevice(std::span<const int> scores);

} // namespace ve::rhi
