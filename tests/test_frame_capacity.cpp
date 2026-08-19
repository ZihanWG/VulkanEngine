#include "renderer/FrameCapacity.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

using ve::renderer::FrameCapacityBudget;

namespace {

// Small caps on purpose: the production values are 8192 each, and a test that had
// to build 8192 items to reach a boundary would be testing patience rather than
// arithmetic. The type takes its caps as constructor arguments precisely so the
// boundary can be reached in three calls.
constexpr uint32_t kObjects = 4;
constexpr uint32_t kDrawItems = 3;

FrameCapacityBudget budget()
{
    return FrameCapacityBudget(kObjects, kDrawItems);
}

// Offer `count` draw items and report how many were admitted.
uint32_t admitDrawItems(FrameCapacityBudget& subject, uint32_t count)
{
    uint32_t admitted = 0;
    for (uint32_t index = 0; index < count; ++index) {
        if (subject.admitDrawItem()) {
            ++admitted;
        }
    }
    return admitted;
}

} // namespace

TEST_CASE("A fresh budget has dropped nothing")
{
    FrameCapacityBudget subject = budget();

    CHECK(subject.emittedDrawItems() == 0);
    CHECK(subject.droppedObjects() == 0);
    CHECK(subject.droppedDrawItems() == 0);
    CHECK_FALSE(subject.overflowed());
}

TEST_CASE("A default-constructed budget admits nothing")
{
    // Zero caps, not infinite ones: a budget that was never given its caps must
    // not silently behave as though everything fits.
    FrameCapacityBudget subject;

    CHECK(subject.admitObjects(5) == 0);
    CHECK_FALSE(subject.admitDrawItem());
    CHECK(subject.droppedObjects() == 5);
    CHECK(subject.droppedDrawItems() == 1);
}

TEST_CASE("An object count under the cap is admitted whole and drops nothing")
{
    FrameCapacityBudget subject = budget();

    CHECK(subject.admitObjects(kObjects - 1) == kObjects - 1);
    CHECK(subject.droppedObjects() == 0);
    CHECK_FALSE(subject.overflowed());
}

TEST_CASE("An object count exactly at the cap is admitted whole")
{
    FrameCapacityBudget subject = budget();

    CHECK(subject.admitObjects(kObjects) == kObjects);
    CHECK(subject.droppedObjects() == 0);
    CHECK_FALSE(subject.overflowed());
}

TEST_CASE("Objects past the cap are clamped and counted")
{
    FrameCapacityBudget subject = budget();

    CHECK(subject.admitObjects(kObjects + 7) == kObjects);
    CHECK(subject.droppedObjects() == 7);
    CHECK(subject.overflowed());
}

TEST_CASE("Draw items are admitted up to the cap and refused after it")
{
    FrameCapacityBudget subject = budget();

    CHECK(admitDrawItems(subject, kDrawItems) == kDrawItems);
    CHECK(subject.emittedDrawItems() == kDrawItems);
    CHECK(subject.droppedDrawItems() == 0);
    CHECK_FALSE(subject.overflowed());

    CHECK_FALSE(subject.admitDrawItem());
    CHECK(subject.emittedDrawItems() == kDrawItems);
    CHECK(subject.droppedDrawItems() == 1);
    CHECK(subject.overflowed());
}

TEST_CASE("Every draw item offered after the cap is counted, not just the first")
{
    // This is the whole point of the type. The code it replaced returned on the
    // first rejection, so the size of the loss was unknowable -- "at least one"
    // is not a diagnostic. Refusing must not stop the counting.
    FrameCapacityBudget subject = budget();

    CHECK(admitDrawItems(subject, kDrawItems + 10) == kDrawItems);
    CHECK(subject.emittedDrawItems() == kDrawItems);
    CHECK(subject.droppedDrawItems() == 10);
}

TEST_CASE("The two counters are independent")
{
    // Objects past the object cap are dropped whole, so their draw items are
    // never offered and never appear in the draw-item count. Neither counter
    // subsumes the other and a caller must read both.
    FrameCapacityBudget subject = budget();

    CHECK(subject.admitObjects(kObjects + 2) == kObjects);
    CHECK(subject.droppedObjects() == 2);
    CHECK(subject.droppedDrawItems() == 0);
    CHECK(subject.overflowed());

    CHECK(admitDrawItems(subject, kDrawItems + 5) == kDrawItems);
    CHECK(subject.droppedObjects() == 2);
    CHECK(subject.droppedDrawItems() == 5);
}

TEST_CASE("A zero draw-item cap refuses everything without emitting")
{
    FrameCapacityBudget subject(kObjects, 0);

    CHECK(admitDrawItems(subject, 3) == 0);
    CHECK(subject.emittedDrawItems() == 0);
    CHECK(subject.droppedDrawItems() == 3);
    CHECK(subject.overflowed());
}

TEST_CASE("Repeated object admissions accumulate their drops")
{
    FrameCapacityBudget subject = budget();

    subject.admitObjects(kObjects + 1);
    subject.admitObjects(kObjects + 4);

    CHECK(subject.droppedObjects() == 5);
}

TEST_CASE("A scene larger than the counter saturates instead of wrapping")
{
    // Wrapping would report a small drop count -- or zero -- at exactly the moment
    // everything was dropped, which is the silent-truncation failure this type
    // exists to remove, reintroduced one layer up.
    if constexpr (sizeof(size_t) <= sizeof(uint32_t)) {
        SUCCEED("size_t cannot hold a count past the counter's range on this target.");
        return;
    }

    FrameCapacityBudget subject = budget();
    constexpr size_t kHuge = static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 100;

    CHECK(subject.admitObjects(kHuge) == kObjects);
    CHECK(subject.droppedObjects() == std::numeric_limits<uint32_t>::max());
    CHECK(subject.overflowed());
}
