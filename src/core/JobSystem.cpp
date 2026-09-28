#include "core/JobSystem.h"

#include <algorithm>
#include <exception>
#include <memory>

namespace ve {

JobSystem::JobSystem(std::size_t threadCount)
{
    if (threadCount == 0) {
        const unsigned int hardware = std::thread::hardware_concurrency();
        threadCount = hardware > 1 ? static_cast<std::size_t>(hardware - 1) : 1;
    }

    workers_.reserve(threadCount);
    for (std::size_t i = 0; i < threadCount; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

JobSystem::~JobSystem()
{
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        stop_ = true;
    }
    condition_.notify_all();

    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void JobSystem::workerLoop()
{
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void JobSystem::parallelFor(std::size_t count,
                            std::size_t minChunkSize,
                            const std::function<void(std::size_t, std::size_t)>& body,
                            std::size_t maxHelpers)
{
    if (count == 0) {
        return;
    }

    minChunkSize = std::max<std::size_t>(minChunkSize, 1);
    const std::size_t maxChunks = (count + minChunkSize - 1) / minChunkSize;
    const std::size_t helperCount = std::min({workers_.size(), maxHelpers, maxChunks - 1});
    if (helperCount == 0) {
        body(0, count);
        return;
    }

    // Chunks are claimed from a shared counter rather than assigned up front, and
    // the calling thread claims them too. A worker that wakes late -- or never
    // wakes before the range is exhausted -- just finds nothing left, so the
    // caller never sits waiting on a thread that has not started. With one fixed
    // chunk per worker it did: every call paid the slowest wake-up in the pool.
    //
    // The batch is shared, not on this stack: a helper can still be queued after
    // this call returns, and it must find an exhausted counter, not a dead frame.
    // `body` is only dereferenced under a claimed chunk, and the caller cannot
    // return while any claimed chunk is unfinished, so the reference is safe.
    const std::size_t targetChunks = std::min((helperCount + 1) * kChunksPerParticipant, maxChunks);
    auto batch = std::make_shared<ParallelForBatch>();
    batch->body = &body;
    batch->count = count;
    batch->chunkSize = (count + targetChunks - 1) / targetChunks;
    batch->chunkCount = (count + batch->chunkSize - 1) / batch->chunkSize;

    // Every helper is woken here, once. Recruiting more from inside the batch --
    // each new participant waking a couple more while chunks remained -- was
    // measured and lost on the frame prep loops: every hop adds a wake-up
    // latency, so on a loop of tens of microseconds the chain either arrives
    // after the work is gone or keeps waking threads that find nothing.
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (stop_) {
            throw std::runtime_error("JobSystem::parallelFor called after shutdown");
        }
        for (std::size_t helper = 0; helper < helperCount; ++helper) {
            tasks_.emplace([batch] { batch->runChunks(); });
        }
    }
    if (helperCount == workers_.size()) {
        condition_.notify_all();
    } else {
        for (std::size_t helper = 0; helper < helperCount; ++helper) {
            condition_.notify_one();
        }
    }

    batch->runChunks();
    batch->waitForCompletion();

    if (batch->error) {
        std::rethrow_exception(batch->error);
    }
}

void JobSystem::ParallelForBatch::runChunks()
{
    for (;;) {
        const std::size_t chunk = nextChunk.fetch_add(1, std::memory_order_relaxed);
        if (chunk >= chunkCount) {
            return;
        }
        const std::size_t begin = chunk * chunkSize;
        const std::size_t end = std::min(begin + chunkSize, count);
        try {
            (*body)(begin, end);
        } catch (...) {
            // Which chunk throws first is a race; keeping whichever claims the
            // slot first matches "the first chunk exception" closely enough, and
            // every remaining chunk still runs, as it did before.
            if (!errorClaimed.exchange(true, std::memory_order_relaxed)) {
                error = std::current_exception();
            }
        }
        // Release publishes this chunk's writes (and any stored error) to the
        // caller, which acquires the same counter before reading either.
        if (completedChunks.fetch_add(1, std::memory_order_acq_rel) + 1 == chunkCount) {
            std::lock_guard<std::mutex> lock(doneMutex);
            done = true;
            doneCondition.notify_one();
        }
    }
}

void JobSystem::ParallelForBatch::waitForCompletion()
{
    // By now every chunk is claimed, so what is left is at most one chunk per
    // worker, already running. That is usually microseconds, so spin briefly
    // before paying for a sleep and a wake-up.
    constexpr int kSpinIterations = 256;
    for (int spin = 0; spin < kSpinIterations; ++spin) {
        if (completedChunks.load(std::memory_order_acquire) == chunkCount) {
            return;
        }
        std::this_thread::yield();
    }

    // The last finisher's counter update read every earlier release in order,
    // and it sets `done` under this mutex, so taking it here is enough to see
    // every chunk's writes.
    std::unique_lock<std::mutex> lock(doneMutex);
    doneCondition.wait(lock, [this] { return done; });
}

std::size_t JobSystem::pendingJobs() const
{
    std::lock_guard<std::mutex> lock(queueMutex_);
    return tasks_.size();
}

} // namespace ve
