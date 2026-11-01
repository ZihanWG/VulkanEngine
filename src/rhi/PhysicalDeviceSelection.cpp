#include "rhi/PhysicalDeviceSelection.h"

namespace ve::rhi {

PhysicalDeviceChoice selectPhysicalDevice(std::span<const int> scores)
{
    PhysicalDeviceChoice choice;
    int bestScore = kUnsuitableDeviceScore;
    for (size_t index = 0; index < scores.size(); ++index) {
        // Strictly greater, so the first of equals is kept.
        if (scores[index] > bestScore) {
            bestScore = scores[index];
            choice.index = index;
        }
    }

    if (!choice.available()) {
        return choice;
    }

    for (size_t index = 0; index < scores.size(); ++index) {
        if (index != choice.index && scores[index] == bestScore) {
            choice.tiedWith.push_back(index);
        }
    }
    return choice;
}

} // namespace ve::rhi
