//
//  task_pool_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_task_pool`: the conformance kit of the task pool contract, C++17-clean.
/// \ingroup core
///
/// **Overview.** Drives any pool with `task_pool_like`'s members (a pool, a `task_pool_interface&`, an
/// `any_task_pool`) through the contract of `concepts/task_pool.hpp` and returns the failures as strings, ts-moveables'
/// kit style, so any test framework (or none) reports them: an empty vector is a pass. C++17-clean, so ts-moveables
/// runs it on its threaded pools under its own build. The checks: `worker_count is at least 1`, `wait_idle on an idle
/// pool returns`, `every task runs exactly once`, `tasks submitted by tasks run before wait_idle returns`, `a lambda
/// is a task`, `wait_idle can be called again`. Counters are atomic, so a threaded pool is checked as it is.
///
/// \snippet interfaces/task_pool_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/task_pool.hpp`, `interfaces/task_pool.hpp`; run on `deferred_task_pool` in every form
/// by `tests/interfaces/task_pool_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** The pool is left idle. Exceptions from the pool propagate (the kit catches none).
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit (about 300 tasks).
///
/// **Tests and demos.** `tests/interfaces/task_pool_conformance_test.cpp` (the kit catches broken pools),
/// `demos/interfaces/task_pool_conformance_demo.cpp`.
///
/// **Dependencies.** `<atomic>`, `<cstddef>`, `<memory>`, `<string>`, `<vector>`, `concepts/task_pool.hpp`.
///
/// **Known issues.** A pool that never runs anything hangs in `wait_idle` rather than failing; run the kit under a
/// test timeout.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_task_pool_conformance_hpp
#define snicholls_core_interfaces_task_pool_conformance_hpp

#include <snicholls/core/concepts/task_pool.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace snicholls {

    /// \brief Checks `pool` against the task pool contract.
    /// \ingroup core
    /// \tparam Pool A type with `task_pool_like`'s members.
    /// \param pool The pool; idle when called, idle on return.
    /// \return Every failed expectation, as text; empty when the pool conforms.
    template <typename Pool>
    std::vector<std::string> check_task_pool(Pool& pool)
    {
        static_assert(is_task_pool_like_v<Pool>, "snicholls::check_task_pool: Pool must have task_pool's members");
        std::vector<std::string> fail;
        const auto expect = [&](bool holds, const char* what) {
            if (!holds)
                fail.emplace_back(what);
        };

        expect(pool.worker_count() >= 1, "worker_count is at least 1");
        pool.wait_idle();   // must return: nothing is pending

        constexpr std::size_t n = 200;
        auto runs = std::make_shared<std::vector<std::atomic<int>>>(n);
        for (std::size_t i = 0; i < n; ++i)
            pool.submit([runs, i] { (*runs)[i].fetch_add(1, std::memory_order_relaxed); });
        pool.wait_idle();
        bool once = true;
        for (const auto& r : *runs)
            once = once && r.load() == 1;
        expect(once, "every task runs exactly once (200 tasks, then wait_idle)");

        auto nested = std::make_shared<std::atomic<int>>(0);
        for (int i = 0; i < 10; ++i)
            pool.submit([&pool, nested] {
                nested->fetch_add(1);
                pool.submit([nested] { nested->fetch_add(100); });
            });
        pool.wait_idle();
        expect(nested->load() == 10 + 1000, "tasks submitted by tasks run before wait_idle returns");

        std::atomic<bool> ran{false};
        pool.submit([&ran] { ran = true; });   // a lambda converts to the task type
        pool.wait_idle();
        expect(ran.load(), "a lambda is a task");

        pool.wait_idle();
        pool.wait_idle();
        expect(pool.worker_count() >= 1, "wait_idle can be called again");
        return fail;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_task_pool_conformance_hpp
