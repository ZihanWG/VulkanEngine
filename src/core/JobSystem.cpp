#include "core/JobSystem.h"

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

std::size_t JobSystem::pendingJobs() const
{
    std::lock_guard<std::mutex> lock(queueMutex_);
    return tasks_.size();
}

} // namespace ve
