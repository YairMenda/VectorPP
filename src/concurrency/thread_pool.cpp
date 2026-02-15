#include "concurrency/thread_pool.hpp"

#include <stdexcept>

namespace vectorpp {

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        num_threads = 1; // Ensure at least one worker thread
    }

    workers_.reserve(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // Wait for either a task or shutdown signal
            condition_.wait(lock, [this] {
                return stop_.load() || !tasks_.empty();
            });

            // If stopping and no more tasks, exit the thread
            if (stop_.load() && tasks_.empty()) {
                return;
            }

            // Get the next task
            task = std::move(tasks_.front());
            tasks_.pop();
        }

        // Execute the task outside the lock
        task();
    }
}

size_t ThreadPool::size() const noexcept {
    return workers_.size();
}

size_t ThreadPool::pending_tasks() const {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return tasks_.size();
}

bool ThreadPool::is_stopped() const noexcept {
    return stop_.load();
}

void ThreadPool::shutdown() {
    // Prevent multiple shutdown calls
    bool expected = false;
    if (!shutdown_initiated_.compare_exchange_strong(expected, true)) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_.store(true);
    }

    // Wake up all worker threads
    condition_.notify_all();

    // Wait for all workers to complete
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace vectorpp
