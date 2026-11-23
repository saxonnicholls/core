//
//  holiday_calendar.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::holiday_calendar_interface`: the pure virtual twin of the `holiday_calendar` concept, and the
/// adapters between them.
/// \ingroup core
///
/// **Overview.** A server holds one calendar per market, loaded at run time, and picks one per trade.
/// `holiday_calendar_interface` is `is_business_day` behind a virtual call; `holiday_calendar_model<T>` adapts any
/// `holiday_calendar` to it, and the interface models the concept, so the adjustment rules of
/// `finance/business_day.hpp` accept either form.
///
/// \snippet interfaces/holiday_calendar_demo.cpp interface
///
/// **Relationships.** `concepts/holiday_calendar.hpp`, `finance/business_day.hpp`,
/// `interfaces/holiday_calendar_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `holiday_calendar_model<T>` for every `holiday_calendar` T.
///
/// **Guarantees.** `is_business_day` is `noexcept`: a calendar answers from data it already holds. The adapter
/// holds its `T` by value and is usable in constant expressions when `T` is.
///
/// **Errors.** None.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/holiday_calendar_test.cpp`, `demos/interfaces/holiday_calendar_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<utility>`, `concepts/holiday_calendar.hpp`.
///
/// **Known issues.** The adapted calendar's `is_business_day` should not throw: the interface's is `noexcept`, so
/// an exception would end the program.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_holiday_calendar_hpp
#define snicholls_core_interfaces_holiday_calendar_hpp

#include <snicholls/core/concepts/holiday_calendar.hpp>

#include <chrono>
#include <utility>

namespace snicholls {

    /// \brief A calendar of business days chosen at run time.
    class holiday_calendar_interface {
    public:
        /// \brief Virtual, so an implementation can be deleted through the interface.
        constexpr virtual ~holiday_calendar_interface() = default;
        /// \brief Whether `day` is a business day.
        /// \param day The day.
        /// \return `true` when the market is open.
        [[nodiscard]] constexpr virtual bool is_business_day(std::chrono::sys_days day) const noexcept = 0;
    };

    /// \brief Any `holiday_calendar` behind `holiday_calendar_interface`.
    /// \tparam T The calendar, held by value.
    template <holiday_calendar T>
    class holiday_calendar_model final : public holiday_calendar_interface {
    public:
        /// \brief Adapt a default-constructed `T`.
        constexpr holiday_calendar_model() = default;
        /// \brief Adapt this value.
        /// \param t The value.
        constexpr explicit holiday_calendar_model(T t) : t_(std::move(t)) {}
        /// \brief Whether the adapted calendar has `day` as a business day.
        /// \param day The day.
        /// \return Its answer.
        [[nodiscard]] constexpr bool is_business_day(std::chrono::sys_days day) const noexcept override
        {
            return static_cast<bool>(t_.is_business_day(day));
        }

    private:
        T t_{};
    };

    static_assert(holiday_calendar<holiday_calendar_interface>,
                  "snicholls::holiday_calendar_interface must model snicholls::holiday_calendar");

} // namespace snicholls

#endif // snicholls_core_interfaces_holiday_calendar_hpp
