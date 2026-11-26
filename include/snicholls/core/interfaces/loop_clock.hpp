//
//  loop_clock.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  ts-moveables' loop_clock base class (event/clock.hpp), its members unchanged, with the adapters both ways; C++17-clean.
//

/// \file
/// \brief `snicholls::loop_clock_interface`: the virtual twin of `loop_clock_like`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** An event loop holds its clock by pointer (`std::shared_ptr<loop_clock_interface>`), chosen at run
/// time: real time, manual, replay. The interface is ts-moveables' `loop_clock` exactly: `now()` and `block_for()` are
/// pure virtual; `set_waker()` is virtual with a default that keeps the waker, and the protected `wake()` calls it, so
/// a clock that moves on command (a manual or a scaled one) wakes the loop with one call. It satisfies the concept and
/// the trait (asserted below). The adapters:
/// - `loop_clock_adapter<Impl>`: the interface implemented by forwarding to a model, held by value, or bound to one
///   when `Impl` is a reference (a view); `set_waker` is forwarded, so the model wakes the loop itself;
/// - `any_loop_clock`: a value owning a `std::unique_ptr<loop_clock_interface>`, which models the concept.
///
/// \snippet interfaces/loop_clock_demo.cpp virtual
///
/// **Relationships.** `concepts/loop_clock.hpp`, `interfaces/loop_clock_reference.hpp`,
/// `interfaces/loop_clock_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `loop_clock_adapter<Impl>` over any model; ts-moveables' clocks derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call. `set_waker` is not synchronised: the loop installs the waker
/// before the clock is shared, as in ts-moveables.
///
/// **Errors.** None.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/loop_clock_test.cpp` (the kit through every form),
/// `demos/interfaces/loop_clock_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<functional>`, `<memory>`, `<type_traits>`, `<utility>`, `concepts/loop_clock.hpp`.
///
/// **Known issues.** C++17-clean by rule (`snicholls_core_cpp17_check`).
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_loop_clock_hpp
#define snicholls_core_interfaces_loop_clock_hpp

#include <snicholls/core/concepts/loop_clock.hpp>

#include <chrono>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The loop's clock (ts-moveables' `loop_clock`): the virtual twin of `loop_clock_like`.
    /// \ingroup core
    class loop_clock_interface {
    public:
        /// \brief The time point: steady_clock's, so a loop's timer heap is unchanged by the clock.
        using time_point = loop_time_point;
        /// \brief The duration of the timeline.
        using duration = std::chrono::nanoseconds;

        /// \brief Destroys the clock.
        virtual ~loop_clock_interface() = default;
        /// \brief The current instant in the loop's timeline.
        /// \return It.
        [[nodiscard]] virtual time_point now() const noexcept = 0;
        /// \brief Real milliseconds the poller may block while the next deadline is `until` in this timeline.
        /// \param until The deadline.
        /// \return `-1` to block until woken, `0` not to block, or the wait.
        [[nodiscard]] virtual long long block_for(time_point until) const noexcept = 0;
        /// \brief Installs the loop's waker (the default keeps it for `wake()`).
        /// \param wake The waker; empty to remove it.
        virtual void set_waker(std::function<void()> wake) { waker_ = std::move(wake); }

    protected:
        /// \brief Constructs the base.
        loop_clock_interface() = default;
        /// \brief Copies the base and its waker.
        loop_clock_interface(const loop_clock_interface&) = default;
        /// \brief Assigns the base and its waker.
        /// \return `*this`.
        loop_clock_interface& operator=(const loop_clock_interface&) = default;
        /// \brief Calls the waker, if one is installed.
        void wake() const
        {
            if (waker_)
                waker_();
        }

    private:
        std::function<void()> waker_;
    };

    /// \brief `loop_clock_interface` implemented by forwarding to a model: held by value, or bound to one when `Impl`
    /// is a reference.
    /// \ingroup core
    /// \tparam Impl The model, or a reference to one.
    template <typename Impl>
    class loop_clock_adapter final : public loop_clock_interface {
        static_assert(is_loop_clock_like_v<std::remove_reference_t<Impl>>,
                      "snicholls::loop_clock_adapter: Impl does not have loop_clock's members "
                      "(now() const noexcept, block_for(time_point) const noexcept, set_waker(std::function<void()>))");

    public:
        /// \brief The model's type.
        using implementation_type = std::remove_reference_t<Impl>;

        /// \brief Constructs (or binds) the model from `args`.
        /// \tparam Args The arguments' types.
        /// \param args The arguments.
        template <typename... Args, std::enable_if_t<std::is_constructible_v<Impl, Args...>, int> = 0>
        explicit loop_clock_adapter(Args&&... args) noexcept(std::is_nothrow_constructible_v<Impl, Args...>)
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
        /// \return As the model.
        [[nodiscard]] time_point now() const noexcept override { return impl_.now(); }
        /// \brief Forwards to the model.
        /// \param until The deadline.
        /// \return As the model.
        [[nodiscard]] long long block_for(time_point until) const noexcept override { return impl_.block_for(until); }
        /// \brief Forwards to the model, which wakes the loop itself.
        /// \param wake The waker.
        void set_waker(std::function<void()> wake) override { impl_.set_waker(std::move(wake)); }

    private:
        Impl impl_;
    };

    /// \brief A clock as a value: owns a `std::unique_ptr<loop_clock_interface>` and models the concept.
    /// \ingroup core
    class any_loop_clock {
    public:
        /// \brief The time point.
        using time_point = loop_time_point;

        /// \brief Takes ownership of `clock`, which must not be null.
        /// \param clock The clock.
        explicit any_loop_clock(std::unique_ptr<loop_clock_interface> clock) noexcept : p_(std::move(clock)) {}

        /// \brief A clock made from a model, adapted and owned.
        /// \tparam Impl The model's type.
        /// \param impl The model.
        /// \return The value.
        template <typename Impl>
        [[nodiscard]] static any_loop_clock of(Impl impl)
        {
            return any_loop_clock(std::make_unique<loop_clock_adapter<Impl>>(std::move(impl)));
        }

        /// \brief Forwards to the interface.
        /// \return As the interface.
        [[nodiscard]] time_point now() const noexcept { return p_->now(); }
        /// \brief Forwards to the interface.
        /// \param until The deadline.
        /// \return As the interface.
        [[nodiscard]] long long block_for(time_point until) const noexcept { return p_->block_for(until); }
        /// \brief Forwards to the interface.
        /// \param wake The waker.
        void set_waker(std::function<void()> wake) { p_->set_waker(std::move(wake)); }
        /// \brief The interface.
        /// \return It.
        [[nodiscard]] loop_clock_interface& get() noexcept { return *p_; }

    private:
        std::unique_ptr<loop_clock_interface> p_;
    };

    static_assert(is_loop_clock_like_v<loop_clock_interface>, "snicholls::loop_clock_interface no longer matches is_loop_clock_like_v");
    static_assert(is_loop_clock_like_v<any_loop_clock>, "snicholls::any_loop_clock no longer matches is_loop_clock_like_v");
#if SNICHOLLS_HAS_CONCEPTS
    static_assert(loop_clock_like<loop_clock_interface>, "snicholls::loop_clock_interface no longer matches loop_clock_like");
    static_assert(loop_clock_like<any_loop_clock>, "snicholls::any_loop_clock no longer matches loop_clock_like");
#endif

} // namespace snicholls

#endif // snicholls_core_interfaces_loop_clock_hpp
