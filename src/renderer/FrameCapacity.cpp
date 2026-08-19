#include "renderer/FrameCapacity.h"

#include <algorithm>
#include <limits>

namespace ve::renderer {

namespace {

// The drop counters are uint32_t because that is what the debug UI and the log
// line print, but the object count they are derived from is a size_t. Saturating
// keeps a pathological scene from wrapping the counter around to a small number,
// which would report "nothing was dropped" at exactly the moment everything was.
uint32_t addSaturating(uint32_t value, size_t addend)
{
    constexpr size_t kMax = std::numeric_limits<uint32_t>::max();
    const size_t sum = static_cast<size_t>(value) + std::min(addend, kMax);
    return static_cast<uint32_t>(std::min(sum, kMax));
}

} // namespace

FrameCapacityBudget::FrameCapacityBudget(uint32_t maxObjects, uint32_t maxDrawItems)
    : maxObjects_(maxObjects)
    , maxDrawItems_(maxDrawItems)
{
}

size_t FrameCapacityBudget::admitObjects(size_t objectCount)
{
    const size_t admitted = std::min(objectCount, static_cast<size_t>(maxObjects_));
    droppedObjects_ = addSaturating(droppedObjects_, objectCount - admitted);
    return admitted;
}

bool FrameCapacityBudget::admitDrawItem()
{
    if (emittedDrawItems_ >= maxDrawItems_) {
        droppedDrawItems_ = addSaturating(droppedDrawItems_, 1);
        return false;
    }

    ++emittedDrawItems_;
    return true;
}

} // namespace ve::renderer
