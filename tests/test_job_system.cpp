#include "core/JobSystem.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

using ve::JobSystem;

TEST_CASE("JobSystem runs queued jobs and returns their results", "[jobs]")
{
    JobSystem jobs(4);
    REQUIRE(jobs.threadCount() == 4);

    constexpr int count = 256;
    std::vector<std::future<int>> futures;
    futures.reserve(count);
    for (int i = 0; i < count; ++i) {
        futures.push_back(jobs.enqueue([i] { return i * i; }));
    }

    long long total = 0;
    for (auto& future : futures) {
        total += future.get();
    }

    long long expected = 0;
    for (int i = 0; i < count; ++i) {
        expected += static_cast<long long>(i) * i;
    }
    CHECK(total == expected);
}

TEST_CASE("JobSystem actually distributes work across worker threads", "[jobs]")
{
    JobSystem jobs(4);

    std::mutex mutex;
    std::set<std::thread::id> threadIds;
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 64; ++i) {
        futures.push_back(jobs.enqueue([&] {
            // Keep the worker busy briefly so several threads overlap.
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::lock_guard<std::mutex> lock(mutex);
            threadIds.insert(std::this_thread::get_id());
        }));
    }
    for (auto& future : futures) {
        future.get();
    }

    // With four workers and 64 short jobs, more than one thread must participate.
    CHECK(threadIds.size() > 1);
}

TEST_CASE("JobSystem supports void jobs with side effects", "[jobs]")
{
    JobSystem jobs(2);

    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 1000; ++i) {
        futures.push_back(jobs.enqueue([&counter] { counter.fetch_add(1, std::memory_order_relaxed); }));
    }
    for (auto& future : futures) {
        future.get();
    }

    CHECK(counter.load() == 1000);
}

TEST_CASE("JobSystem propagates exceptions through the future", "[jobs]")
{
    JobSystem jobs(2);

    auto future = jobs.enqueue([]() -> int { throw std::runtime_error("boom"); });
    CHECK_THROWS_AS(future.get(), std::runtime_error);
}

TEST_CASE("JobSystem auto-sizes to at least one worker", "[jobs]")
{
    JobSystem jobs; // default sizing
    CHECK(jobs.threadCount() >= 1);

    auto future = jobs.enqueue([] { return 42; });
    CHECK(future.get() == 42);
}

TEST_CASE("parallelFor covers every index exactly once", "[jobs]")
{
    JobSystem jobs(4);

    // Deliberately not a multiple of any chunk size the pool might pick.
    constexpr std::size_t count = 1013;
    std::vector<int> touches(count, 0);
    jobs.parallelFor(count, 16, [&touches](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            ++touches[i]; // chunks are disjoint, so unsynchronized writes are safe
        }
    });

    CHECK(std::count(touches.begin(), touches.end(), 1) == static_cast<long>(count));
}

TEST_CASE("parallelFor runs small ranges inline on the calling thread", "[jobs]")
{
    JobSystem jobs(4);

    std::mutex mutex;
    std::set<std::thread::id> threadIds;
    jobs.parallelFor(8, 64, [&](std::size_t, std::size_t) {
        std::lock_guard<std::mutex> lock(mutex);
        threadIds.insert(std::this_thread::get_id());
    });

    REQUIRE(threadIds.size() == 1);
    CHECK(*threadIds.begin() == std::this_thread::get_id());
}

TEST_CASE("parallelFor distributes large ranges across threads", "[jobs]")
{
    JobSystem jobs(4);

    std::mutex mutex;
    std::set<std::thread::id> threadIds;
    jobs.parallelFor(4096, 1, [&](std::size_t, std::size_t) {
        // Keep each chunk busy briefly so workers overlap with the caller.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::lock_guard<std::mutex> lock(mutex);
        threadIds.insert(std::this_thread::get_id());
    });

    CHECK(threadIds.size() > 1);
}

TEST_CASE("parallelFor is a no-op for an empty range", "[jobs]")
{
    JobSystem jobs(2);

    bool called = false;
    jobs.parallelFor(0, 1, [&called](std::size_t, std::size_t) { called = true; });
    CHECK_FALSE(called);
}

TEST_CASE("parallelFor rethrows the first chunk exception", "[jobs]")
{
    JobSystem jobs(4);

    CHECK_THROWS_AS(jobs.parallelFor(1024,
                                     1,
                                     [](std::size_t begin, std::size_t) {
                                         if (begin > 0) {
                                             throw std::runtime_error("chunk failure");
                                         }
                                     }),
                    std::runtime_error);
}
