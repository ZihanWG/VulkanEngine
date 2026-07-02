#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
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

    // Runs body(begin, end) over disjoint chunks of [0, count), spreading the
    // chunks across the worker pool while the calling thread executes one chunk
    // itself, and returns once every chunk has completed. Chunks never overlap,
    // so per-index writes need no locking; anything else body touches must be
    // safe to share across threads. Ranges smaller than minChunkSize run inline
    // on the calling thread. The first chunk exception (if any) is rethrown.
    //
    // Must be called from a thread that is NOT a pool worker: a worker calling
    // this would block on chunks that need the (occupied) workers to progress.
    void parallelFor(std::size_t count,
                     std::size_t minChunkSize,
                     const std::function<void(std::size_t begin, std::size_t end)>& body);

    [[nodiscard]] std::size_t threadCount() const { return workers_.size(); }

    // Approximate number of jobs not yet picked up by a worker. For diagnostics.
    [[nodiscard]] std::size_t pendingJobs() const;

private:
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

    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
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
