#include <gtest/gtest.h>
#include "concurrency/thread_pool.hpp"

#include <atomic>
#include <chrono>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

using namespace vectorpp;

class ThreadPoolTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: Thread pool creates correct number of workers
TEST_F(ThreadPoolTest, CreatesCorrectNumberOfWorkers) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.size(), 4);
}

// Test: Default constructor uses hardware concurrency
TEST_F(ThreadPoolTest, DefaultConstructorUsesHardwareConcurrency) {
    ThreadPool pool;
    size_t expected = std::thread::hardware_concurrency();
    if (expected == 0) expected = 1;
    EXPECT_EQ(pool.size(), expected);
}

// Test: Zero threads results in at least one worker
TEST_F(ThreadPoolTest, ZeroThreadsCreatesOneWorker) {
    ThreadPool pool(0);
    EXPECT_EQ(pool.size(), 1);
}

// Test: Submit task and get result via future
TEST_F(ThreadPoolTest, SubmitTaskReturnsCorrectResult) {
    ThreadPool pool(2);

    auto future = pool.submit([]() {
        return 42;
    });

    EXPECT_EQ(future.get(), 42);
}

// Test: Submit task with arguments
TEST_F(ThreadPoolTest, SubmitTaskWithArguments) {
    ThreadPool pool(2);

    auto future = pool.submit([](int a, int b) {
        return a + b;
    }, 10, 20);

    EXPECT_EQ(future.get(), 30);
}

// Test: Multiple tasks execute correctly
TEST_F(ThreadPoolTest, MultipleTasks) {
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;

    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.submit([i]() {
            return i * 2;
        }));
    }

    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(futures[i].get(), i * 2);
    }
}

// Test: Tasks execute on different threads
TEST_F(ThreadPoolTest, TasksExecuteOnDifferentThreads) {
    ThreadPool pool(4);
    std::mutex mutex;
    std::set<std::thread::id> thread_ids;
    std::atomic<int> counter{0};

    std::vector<std::future<void>> futures;

    for (int i = 0; i < 20; ++i) {
        futures.push_back(pool.submit([&mutex, &thread_ids, &counter]() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                thread_ids.insert(std::this_thread::get_id());
            }
            ++counter;
            // Small sleep to allow multiple threads to participate
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }));
    }

    // Wait for all tasks
    for (auto& f : futures) {
        f.get();
    }

    EXPECT_EQ(counter.load(), 20);
    // With 4 threads and small delays, we expect multiple threads used
    EXPECT_GT(thread_ids.size(), 1);
}

// Test: Pending tasks count
TEST_F(ThreadPoolTest, PendingTasksCount) {
    ThreadPool pool(1); // Single thread to control execution
    std::mutex block_mutex;
    std::unique_lock<std::mutex> block_lock(block_mutex);

    // Submit a blocking task
    auto blocking_future = pool.submit([&block_mutex]() {
        std::lock_guard<std::mutex> lock(block_mutex);
        return 0;
    });

    // Give the blocking task time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Submit more tasks (they should queue up)
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 5; ++i) {
        futures.push_back(pool.submit([i]() { return i; }));
    }

    // Check pending tasks (should be 5)
    EXPECT_EQ(pool.pending_tasks(), 5);

    // Release the blocking task
    block_lock.unlock();

    // Wait for all tasks to complete
    blocking_future.get();
    for (auto& f : futures) {
        f.get();
    }

    EXPECT_EQ(pool.pending_tasks(), 0);
}

// Test: Graceful shutdown completes pending tasks
TEST_F(ThreadPoolTest, GracefulShutdownCompletesPendingTasks) {
    auto pool = std::make_unique<ThreadPool>(2);
    std::atomic<int> completed{0};

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool->submit([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            ++completed;
        }));
    }

    // Explicitly shutdown (should wait for tasks)
    pool->shutdown();

    // All tasks should have completed
    EXPECT_EQ(completed.load(), 10);
}

// Test: Cannot submit to stopped pool
TEST_F(ThreadPoolTest, CannotSubmitToStoppedPool) {
    auto pool = std::make_unique<ThreadPool>(2);
    pool->shutdown();

    EXPECT_THROW(pool->submit([]() { return 0; }), std::runtime_error);
}

// Test: is_stopped returns correct state
TEST_F(ThreadPoolTest, IsStoppedReturnsCorrectState) {
    auto pool = std::make_unique<ThreadPool>(2);

    EXPECT_FALSE(pool->is_stopped());

    pool->shutdown();

    EXPECT_TRUE(pool->is_stopped());
}

// Test: Multiple shutdown calls are safe
TEST_F(ThreadPoolTest, MultipleShutdownCallsAreSafe) {
    auto pool = std::make_unique<ThreadPool>(2);

    pool->shutdown();
    pool->shutdown();
    pool->shutdown();

    EXPECT_TRUE(pool->is_stopped());
}

// Test: Exception in task doesn't crash pool
TEST_F(ThreadPoolTest, ExceptionInTaskDoesNotCrashPool) {
    ThreadPool pool(2);

    // Submit task that throws
    auto throwing_future = pool.submit([]() -> int {
        throw std::runtime_error("Test exception");
    });

    // Submit normal task after
    auto normal_future = pool.submit([]() {
        return 42;
    });

    // The throwing task should propagate exception through future
    EXPECT_THROW(throwing_future.get(), std::runtime_error);

    // Normal task should still work
    EXPECT_EQ(normal_future.get(), 42);
}

// Test: Large number of tasks
TEST_F(ThreadPoolTest, LargeNumberOfTasks) {
    ThreadPool pool(8);
    const int num_tasks = 1000;

    std::vector<std::future<int>> futures;
    futures.reserve(num_tasks);

    for (int i = 0; i < num_tasks; ++i) {
        futures.push_back(pool.submit([i]() {
            return i * i;
        }));
    }

    for (int i = 0; i < num_tasks; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

// Test: Void return type
TEST_F(ThreadPoolTest, VoidReturnType) {
    ThreadPool pool(2);
    std::atomic<bool> executed{false};

    auto future = pool.submit([&executed]() {
        executed.store(true);
    });

    future.get();
    EXPECT_TRUE(executed.load());
}

// Test: Stress test with concurrent submits
TEST_F(ThreadPoolTest, ConcurrentSubmits) {
    ThreadPool pool(4);
    std::atomic<int> total{0};
    const int tasks_per_thread = 100;
    const int num_submit_threads = 4;

    std::vector<std::thread> submitters;
    std::vector<std::vector<std::future<int>>> all_futures(num_submit_threads);

    for (int t = 0; t < num_submit_threads; ++t) {
        submitters.emplace_back([&pool, &all_futures, t, tasks_per_thread]() {
            for (int i = 0; i < tasks_per_thread; ++i) {
                all_futures[t].push_back(pool.submit([t, i]() {
                    return t * 1000 + i;
                }));
            }
        });
    }

    // Wait for all submitters
    for (auto& submitter : submitters) {
        submitter.join();
    }

    // Verify all results
    int count = 0;
    for (int t = 0; t < num_submit_threads; ++t) {
        for (int i = 0; i < tasks_per_thread; ++i) {
            EXPECT_EQ(all_futures[t][i].get(), t * 1000 + i);
            ++count;
        }
    }

    EXPECT_EQ(count, num_submit_threads * tasks_per_thread);
}
