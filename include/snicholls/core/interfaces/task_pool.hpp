//
//  task_pool.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  ts-moveables' interfaces/task_pool.hpp and its task_pool_adapter, moved into core; C++17-clean.
//

/// \file
/// \brief `snicholls::task_pool_interface`: the pure virtual twin of `task_pool_like`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the pool is held by pointer and chosen at run time (a task graph keeps one, a benchmark swaps
/// them), code holds a `task_pool_interface&`. Its members are those of `task_pool_like` (`concepts/task_pool.hpp`,
/// where the contract is documented), and it satisfies the concept and the trait (asserted below), so a member changed
/// on one side only fails to compile. The adapters:
/// - `task_pool_adapter<Impl>`: the interface implemented by forwarding to a model, held by value, or bound to one when
///   `Impl` is a reference (a non-owning view);
/// - `any_task_pool`: a value owning a `std::unique_ptr<task_pool_interface>`, which models the concept, for code
///   that takes pools by value.
///
/// \snippet interfaces/task_pool_demo.cpp virtual
///
/// **Relationships.** `concepts/task_pool.hpp`, `interfaces/task_pool_reference.hpp`,
/// `interfaces/task_pool_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `task_pool_adapter<Impl>` over any model; ts-moveables' pools derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety is the wrapped pool's.
///
/// **Errors.** None.
///
/// **Performance.** One virtual call; not benchmarked here (ts-moveables benchmarks its pools).
///
/// **Tests and demos.** `tests/interfaces/task_pool_test.cpp` (the kit through every form),
/// `demos/interfaces/task_pool_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<functional>`, `<memory>`, `<type_traits>`, `<utility>`, `concepts/task_pool.hpp`.
///
/// **Known issues.** C++17-clean by rule (`snicholls_core_cpp17_check`).
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_task_pool_hpp
#define snicholls_core_interfaces_task_pool_hpp

#include <snicholls/core/concepts/task_pool.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief Somewhere to submit work: the pure virtual twin of `task_pool_like` (ts-moveables' `task_pool`).
    /// \ingroup core
    struct task_pool_interface {
        /// \brief A task.
        using task = pool_task_function;

        /// \brief Destroys the pool.
        virtual ~task_pool_interface() = default;
        /// \brief Enqueues work.
        /// \param t The task.
        virtual void submit(task t) = 0;
        /// \brief Blocks until all submitted work has run.
        virtual void wait_idle() = 0;
        /// \brief The parallelism the algorithms may assume.
        /// \return At least 1.
        [[nodiscard]] virtual std::size_t worker_count() const noexcept = 0;
    };

    /// \brief `task_pool_interface` implemented by forwarding to a model: held by value, or bound to one when `Impl`
    /// is a reference.
    /// \ingroup core
    /// \tparam Impl The model, or a reference to one.
    template <typename Impl>
    class task_pool_adapter final : public task_pool_interface {
        static_assert(is_task_pool_like_v<std::remove_reference_t<Impl>>,
                      "snicholls::task_pool_adapter: Impl does not have task_pool's members "
                      "(submit(task), wait_idle(), worker_count() const noexcept)");

    public:
        /// \brief The model's type.
        using implementation_type = std::remove_reference_t<Impl>;

        /// \brief Constructs (or binds) the model from `args`.
        /// \tparam Args The arguments' types.
        /// \param args The arguments.
        template <typename... Args, std::enable_if_t<std::is_constructible_v<Impl, Args...>, int> = 0>
        explicit task_pool_adapter(Args&&... args) noexcept(std::is_nothrow_constructible_v<Impl, Args...>)
            : impl_(std::forward<Args>(args)...)
        {
        }

        /// \brief The model.
        /// \return It.
        implementation_type& implementation() noexcept { return impl_; }
        /// \brief The model.
        /// \return It.
        const implementation_type& implementation() const noexcept { return impl_; }

        /// \brief Forwards to the model.
        /// \param t The task.
        void submit(task t) override { impl_.submit(std::move(t)); }
        /// \brief Forwards to the model.
        void wait_idle() override { impl_.wait_idle(); }
        /// \brief Forwards to the model.
        /// \return As the model.
        [[nodiscard]] std::size_t worker_count() const noexcept override { return impl_.worker_count(); }

    private:
        Impl impl_;
    };

    /// \brief A pool as a value: owns a `std::unique_ptr<task_pool_interface>` and models the concept.
    /// \ingroup core
    class any_task_pool {
    public:
        /// \brief Takes ownership of `pool`, which must not be null.
        /// \param pool The pool.
        explicit any_task_pool(std::unique_ptr<task_pool_interface> pool) noexcept : p_(std::move(pool)) {}

        /// \brief A pool made from a model, adapted and owned.
        /// \tparam Impl The model's type.
        /// \param impl The model.
        /// \return The value.
        template <typename Impl>
        [[nodiscard]] static any_task_pool of(Impl impl)
        {
            return any_task_pool(std::make_unique<task_pool_adapter<Impl>>(std::move(impl)));
        }

        /// \brief Forwards to the interface.
        /// \param t The task.
        void submit(pool_task_function t) { p_->submit(std::move(t)); }
        /// \brief Forwards to the interface.
        void wait_idle() { p_->wait_idle(); }
        /// \brief Forwards to the interface.
        /// \return As the interface.
        [[nodiscard]] std::size_t worker_count() const noexcept { return p_->worker_count(); }
        /// \brief The interface.
        /// \return It.
        [[nodiscard]] task_pool_interface& get() noexcept { return *p_; }

    private:
        std::unique_ptr<task_pool_interface> p_;
    };

    static_assert(is_task_pool_like_v<task_pool_interface>, "snicholls::task_pool_interface no longer matches is_task_pool_like_v");
    static_assert(is_task_pool_like_v<any_task_pool>, "snicholls::any_task_pool no longer matches is_task_pool_like_v");
    static_assert(is_pool_task_v<task_pool_interface::task> && !is_pool_task_v<int>);
#if SNICHOLLS_HAS_CONCEPTS
    static_assert(task_pool_like<task_pool_interface>, "snicholls::task_pool_interface no longer matches task_pool_like");
    static_assert(task_pool_like<any_task_pool>, "snicholls::any_task_pool no longer matches task_pool_like");
    static_assert(pool_task<task_pool_interface::task> && !pool_task<int>);
#endif

} // namespace snicholls

#endif // snicholls_core_interfaces_task_pool_hpp
