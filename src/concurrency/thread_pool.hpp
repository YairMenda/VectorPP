#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace vectorpp {

/**
 * @brief Custom thread pool implementation using std::thread, std::queue, and condition variables.
 *
 * This thread pool demonstrates concurrency knowledge by implementing:
 * - Worker threads that wait on a condition variable
 * - A task queue protected by a mutex
 * - Graceful shutdown that completes pending tasks
 * - Future-based task submission for result retrieval
 */
class ThreadPool {
public:
    /**
     * @brief Constructs a thread pool with the specified number of worker threads.
     * @param num_threads Number of worker threads to create. Defaults to hardware concurrency.
     */
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());

    /**
     * @brief Destructor. Performs graceful shutdown - completes all pending tasks before destroying.
     */
    ~ThreadPool();

    // Non-copyable and non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /**
     * @brief Submits a task to the thread pool and returns a future for the result.
     * @tparam F Callable type
     * @tparam Args Argument types
     * @param f The function to execute
     * @param args Arguments to pass to the function
     * @return std::future<ReturnType> Future that will contain the result
     * @throws std::runtime_error if the thread pool has been stopped
     */
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    /**
     * @brief Returns the number of worker threads in the pool.
     */
    size_t size() const noexcept;

    /**
     * @brief Returns the number of pending tasks in the queue.
     */
    size_t pending_tasks() const;

    /**
     * @brief Checks if the thread pool has been stopped.
     */
    bool is_stopped() const noexcept;

    /**
     * @brief Initiates graceful shutdown - stops accepting new tasks and waits for pending tasks.
     */
    void shutdown();

private:
    void worker_thread();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;

    std::atomic<bool> stop_{false};
    std::atomic<bool> shutdown_initiated_{false};
};

// Template implementation must be in header
template<typename F, typename... Args>
auto ThreadPool::submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    // Create a packaged task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        if (stop_.load()) {
            throw std::runtime_error("Cannot submit task to stopped thread pool");
        }

        tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return result;
}

} // namespace vectorpp
