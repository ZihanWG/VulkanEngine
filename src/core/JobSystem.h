#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ve {

// A small fixed-size worker thread pool. Jobs are submitted with enqueue() and
// run on background threads; the returned std::future delivers the result (or a
// propagated exception). It is intended for CPU-bound work such as decoding
// textures and parsing meshes off the main/render thread.
//
// All Vulkan work must still happen on the thread that owns the device; use this
// only for CPU-side decode/parse and hand the results back to the main thread.
class JobSystem final {
public:
    // threadCount == 0 picks hardware_concurrency() - 1 (at least one worker).
    explicit JobSystem(std::size_t threadCount = 0);
    ~JobSystem();

    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;
    JobSystem(JobSystem&&) = delete;
    JobSystem& operator=(JobSystem&&) = delete;

    // Submit a callable; returns a future for its result. Throws if called after
    // the pool has begun shutting down.
    template <typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    // parallelFor's default helper bound: every worker in the pool.
    static constexpr std::size_t kAllWorkers = static_cast<std::size_t>(-1);

    // Runs body(begin, end) over disjoint chunks of [0, count) and returns once
    // every chunk has completed. Up to maxHelpers workers are woken, once, and
    // they and the calling thread claim chunks from a shared counter, so the
    // caller never waits on a worker that has not started. Chunks are no smaller
    // than minChunkSize, and there are up to kChunksPerParticipant per thread so
    // a slow core holds up at most one small chunk. Chunks never overlap, so
    // per-index writes need no locking; anything else body touches must be safe
    // to share across threads, and body must not assume how many chunks there
    // are or which thread runs which. Ranges smaller than minChunkSize run inline
    // on the calling thread. Every chunk runs even when one throws, and one chunk
    // exception is rethrown.
    //
    // maxHelpers bounds how many workers one call wakes. The default wakes the
    // whole pool, which suits long throughput work (load-time integrations).
    // Short, latency-bound loops want a small bound: every wake-up is a
    // cross-core signal, and waking a whole pool for microseconds of work costs
    // more than the work -- see docs/parallel_frame_prep.md.
    //
    // Must be called from a thread that is NOT a pool worker: a worker calling
    // this would block on chunks that need the (occupied) workers to progress.
    void parallelFor(std::size_t count,
                     std::size_t minChunkSize,
                     const std::function<void(std::size_t begin, std::size_t end)>& body,
                     std::size_t maxHelpers = kAllWorkers);

    [[nodiscard]] std::size_t threadCount() const
    {
        return workers_.size();
    }

    // Approximate number of jobs not yet picked up by a worker. For diagnostics.
    [[nodiscard]] std::size_t pendingJobs() const;

private:
    // Enough chunks per thread that one landing on a slow core, or on a worker
    // that wakes late, leaves the others something to take; few enough that the
    // per-chunk counter traffic stays negligible next to the body.
    static constexpr std::size_t kChunksPerParticipant = 4;

    // One parallelFor call's shared state. Owned by a shared_ptr because helper
    // tasks can outlive the call that queued them (see parallelFor).
    struct ParallelForBatch {
        const std::function<void(std::size_t, std::size_t)>* body = nullptr;
        std::size_t count = 0;
        std::size_t chunkSize = 0;
        std::size_t chunkCount = 0;
        std::atomic<std::size_t> nextChunk{0};
        std::atomic<std::size_t> completedChunks{0};
        std::atomic<bool> errorClaimed{false};
        std::exception_ptr error;
        std::mutex doneMutex;
        std::condition_variable doneCondition;
        bool done = false;

        void runChunks();
        void waitForCompletion();
    };

    void workerLoop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queueMutex_;
    std::condition_variable condition_;
    bool stop_ = false;
};

template <typename F, typename... Args>
auto JobSystem::enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
{
    using ReturnType = std::invoke_result_t<F, Args...>;

    auto task =
        std::make_shared<std::packaged_task<ReturnType()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<ReturnType> result = task->get_future();

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (stop_) {
            throw std::runtime_error("JobSystem::enqueue called after shutdown");
        }
        tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return result;
}

} // namespace ve
