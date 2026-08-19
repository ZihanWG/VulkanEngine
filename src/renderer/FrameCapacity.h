#pragma once

#include <cstddef>
#include <cstdint>

namespace ve::renderer {

// Accounting for the two fixed per-frame caps: how many render objects the frame
// sweeps (kMaxFrameObjects) and how many draw items it may emit (kMaxDrawItems).
//
// The caps are not new and neither is the truncation. What is new is that going
// over one is *counted* rather than silently dropped. Both limits used to clamp
// in five separate places with a `std::min` and a bare `return false`, so a scene
// past either cap simply lost geometry -- no log line, no counter, no ImGui
// indicator. The failure looked like missing meshes with no cause, which is the
// worst shape a renderer bug can take.
//
// Pure counting, no Vulkan: it lives in VulkanEngineCore so the arithmetic can be
// tested without a device, same reasoning as renderer/DrawItemBatching.h and
// renderer/OcclusionYield.h.
//
// The frame builder drives this with the same calls it uses to decide what to
// emit -- admitDrawItem() *is* the capacity check -- so the reported counts cannot
// drift away from the loop that produced them. A separate model of "what the loop
// would have done" would be free to disagree with the loop, silently, which is
// the failure this type exists to remove.
class FrameCapacityBudget {
public:
    FrameCapacityBudget() = default;
    FrameCapacityBudget(uint32_t maxObjects, uint32_t maxDrawItems);

    // Clamp a scene's object count to the sweep cap, recording the remainder as
    // dropped. Returns how many objects the caller should visit.
    //
    // Objects past the cap are dropped whole: they are invisible to draw-item
    // building, to CPU culling, to the shadow cascades, and to the GPU cull
    // input, so their draw items are never counted by droppedDrawItems().
    // The two counters answer different questions and neither subsumes
    // the other.
    size_t admitObjects(size_t objectCount);

    // Reserve one draw-item slot. Returns false, and records the loss, once the
    // draw-item cap is full.
    //
    // Callers must keep calling after a false: the count is only a real total if
    // every rejected item is offered. Stopping at the first rejection is what the
    // old `return false` did, and it is why the size of the loss was unknowable.
    bool admitDrawItem();

    [[nodiscard]] uint32_t emittedDrawItems() const { return emittedDrawItems_; }
    [[nodiscard]] uint32_t droppedObjects() const { return droppedObjects_; }
    [[nodiscard]] uint32_t droppedDrawItems() const { return droppedDrawItems_; }
    [[nodiscard]] bool overflowed() const { return droppedObjects_ != 0 || droppedDrawItems_ != 0; }

private:
    uint32_t maxObjects_ = 0;
    uint32_t maxDrawItems_ = 0;
    uint32_t emittedDrawItems_ = 0;
    uint32_t droppedObjects_ = 0;
    uint32_t droppedDrawItems_ = 0;
};

} // namespace ve::renderer
