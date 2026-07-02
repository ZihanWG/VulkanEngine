#include "core/JobSystem.h"

#include <algorithm>
#include <exception>

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
                            const std::function<void(std::size_t, std::size_t)>& body)
{
    if (count == 0) {
        return;
    }

    minChunkSize = std::max<std::size_t>(minChunkSize, 1);
    const std::size_t maxChunks = (count + minChunkSize - 1) / minChunkSize;
    const std::size_t chunkCount = std::min(workers_.size() + 1, maxChunks);
    if (chunkCount <= 1) {
        body(0, count);
        return;
    }

    const std::size_t chunkSize = (count + chunkCount - 1) / chunkCount;
    std::vector<std::future<void>> pending;
    pending.reserve(chunkCount - 1);
    for (std::size_t chunk = 1; chunk < chunkCount; ++chunk) {
        const std::size_t begin = chunk * chunkSize;
        const std::size_t end = std::min(begin + chunkSize, count);
        if (begin >= end) {
            break;
        }
        // Capturing body by reference is safe: this function does not return
        // until every chunk future has completed.
        pending.push_back(enqueue([&body, begin, end] { body(begin, end); }));
    }

    // The calling thread executes the first chunk instead of blocking idle.
    body(0, std::min(chunkSize, count));

    std::exception_ptr firstError;
    for (std::future<void>& chunkFuture : pending) {
        try {
            chunkFuture.get();
        } catch (...) {
            if (!firstError) {
                firstError = std::current_exception();
            }
        }
    }
    if (firstError) {
        std::rethrow_exception(firstError);
    }
}

std::size_t JobSystem::pendingJobs() const
{
    std::lock_guard<std::mutex> lock(queueMutex_);
    return tasks_.size();
}

} // namespace ve
