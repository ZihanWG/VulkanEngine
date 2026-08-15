#include "rhi/ValidationTally.h"

#include <atomic>

namespace ve::rhi {

namespace {

// Relaxed ordering is sufficient: these counters are only read after the run has
// finished and every Vulkan thread has been joined, so no happens-before edge is
// needed between the increments and the read.
std::atomic<uint64_t>& errorCounter()
{
    static std::atomic<uint64_t> counter{0};
    return counter;
}

std::atomic<uint64_t>& warningCounter()
{
    static std::atomic<uint64_t> counter{0};
    return counter;
}

} // namespace

void ValidationTally::recordError()
{
    errorCounter().fetch_add(1, std::memory_order_relaxed);
}

void ValidationTally::recordWarning()
{
    warningCounter().fetch_add(1, std::memory_order_relaxed);
}

uint64_t ValidationTally::errorCount()
{
    return errorCounter().load(std::memory_order_relaxed);
}

uint64_t ValidationTally::warningCount()
{
    return warningCounter().load(std::memory_order_relaxed);
}

void ValidationTally::reset()
{
    errorCounter().store(0, std::memory_order_relaxed);
    warningCounter().store(0, std::memory_order_relaxed);
}

} // namespace ve::rhi
