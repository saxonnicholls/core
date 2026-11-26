//
//  task_pool_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::deferred_task_pool`: the standard-library reference task pool; single-threaded and deterministic.
/// \ingroup core
///
/// **Overview.** Core has no threads, so its pool runs nothing until asked: `submit` queues a task, and `wait_idle`
/// runs the queue on the caller's thread, in submission order, until it is empty (tasks a task submits run in the
/// same call). `run_one()` runs only the next, so a test can step through work and look between tasks; `pending()`
/// says how many wait. Code written against `task_pool_like` is tested with it and gets the same interleaving on
/// every run. (ts-moveables' `inline_task_pool` runs a task inside `submit`; this one defers it.)
///
/// \snippet interfaces/task_pool_reference_demo.cpp reference
///
/// **Relationships.** Models `task_pool_like` (`concepts/task_pool.hpp`); passes `interfaces/task_pool_conformance.hpp`
/// directly and through `interfaces/task_pool.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** Not thread-safe: submit from the thread that waits. FIFO. A task that throws leaves `wait_idle` (or
/// `run_one`) with its exception, already removed from the queue; the rest stay queued. C++17-clean.
///
/// **Errors.** A task's exception, as it is.
///
/// **Performance.** Not benchmarked: a test double (one `std::deque` push and pop per task).
///
/// **Tests and demos.** `tests/interfaces/task_pool_reference_test.cpp`, `demos/interfaces/task_pool_reference_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<deque>`, `<utility>`, `interfaces/task_pool.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_task_pool_reference_hpp
#define snicholls_core_interfaces_task_pool_reference_hpp

#include <snicholls/core/interfaces/task_pool.hpp>

#include <cstddef>
#include <deque>
#include <utility>

namespace snicholls {

    /// \brief The reference task pool: queues on `submit`, runs on `wait_idle` or `run_one`, on the caller's thread.
    /// \ingroup core
    class deferred_task_pool {
    public:
        /// \brief Queues `t`; an empty task is ignored.
        /// \param t The task.
        void submit(pool_task_function t)
        {
            if (t)
                queue_.push_back(std::move(t));
        }

        /// \brief Runs every queued task, and every task they queue, in order.
        void wait_idle()
        {
            while (run_one()) {
            }
        }

        /// \brief Runs the next queued task.
        /// \return `false` when the queue was empty.
        bool run_one()
        {
            if (queue_.empty())
                return false;
            pool_task_function t = std::move(queue_.front());
            queue_.pop_front();
            ++ran_;
            t();
            return true;
        }

        /// \brief One: the caller's thread.
        /// \return 1.
        [[nodiscard]] std::size_t worker_count() const noexcept { return 1; }
        /// \brief How many tasks wait.
        /// \return The queue's length.
        [[nodiscard]] std::size_t pending() const noexcept { return queue_.size(); }
        /// \brief How many tasks have started.
        /// \return The count.
        [[nodiscard]] std::size_t ran() const noexcept { return ran_; }

    private:
        std::deque<pool_task_function> queue_;
        std::size_t ran_ = 0;
    };

    static_assert(is_task_pool_like_v<deferred_task_pool>, "snicholls::deferred_task_pool must have task_pool's members");
#if SNICHOLLS_HAS_CONCEPTS
    static_assert(task_pool_like<deferred_task_pool>, "snicholls::deferred_task_pool must model snicholls::task_pool_like");
#endif

} // namespace snicholls

#endif // snicholls_core_interfaces_task_pool_reference_hpp
