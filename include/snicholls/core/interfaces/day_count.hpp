//
//  day_count.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::day_count_interface`: the pure virtual twin of the `day_count_convention` concept, and the
/// adapters between them.
/// \ingroup core
///
/// **Overview.** A trade's terms name its day count as data ("ACT/360"), so code that reads trades chooses the
/// convention at run time. `day_count_interface` is the concept's three operations behind virtual calls;
/// `day_count_model<T>` adapts any `day_count_convention` to it, and the interface models the concept, so a
/// template written for the concept accepts either form.
///
/// \snippet interfaces/day_count_demo.cpp interface
///
/// **Relationships.** `concepts/day_count.hpp`, `finance/day_count.hpp`, `interfaces/day_count_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations.** `day_count_model<T>` for every `day_count_convention` T.
///
/// **Guarantees.** Every operation is `noexcept`; the adapter holds its `T` by value and is usable in constant
/// expressions when `T` is.
///
/// **Errors.** None.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/day_count_test.cpp`, `demos/interfaces/day_count_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<cstdint>`, `<string_view>`, `concepts/day_count.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_day_count_hpp
#define snicholls_core_interfaces_day_count_hpp

#include <snicholls/core/concepts/day_count.hpp>

#include <chrono>
#include <cstdint>
#include <string_view>

namespace snicholls {

    /// \brief A day-count convention chosen at run time.
    class day_count_interface {
    public:
        /// \brief Virtual, so an implementation can be deleted through the interface.
        constexpr virtual ~day_count_interface() = default;
        /// \brief The convention's name.
        /// \return A view valid while the object lives.
        [[nodiscard]] constexpr virtual std::string_view name() const noexcept = 0;
        /// \brief The days the convention counts from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return The count.
        [[nodiscard]] constexpr virtual std::int64_t day_count(std::chrono::sys_days start, std::chrono::sys_days end) const noexcept = 0;
        /// \brief The fraction of a year from `start` to `end`.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return The exact fraction.
        [[nodiscard]] constexpr virtual finance::year_fraction year_fraction(std::chrono::sys_days start,
                                                                             std::chrono::sys_days end) const noexcept = 0;
    };

    /// \brief Any `day_count_convention` behind `day_count_interface`.
    /// \tparam T The convention, held by value.
    template <day_count_convention T>
    class day_count_model final : public day_count_interface {
    public:
        /// \brief Adapt a default-constructed `T`.
        constexpr day_count_model() = default;
        /// \brief Adapt this value.
        /// \param t The value.
        constexpr explicit day_count_model(T t) : t_(t) {}
        /// \brief The adapted convention's name.
        /// \return Its name.
        [[nodiscard]] constexpr std::string_view name() const noexcept override { return t_.name(); }
        /// \brief The adapted convention's day count.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return The count.
        [[nodiscard]] constexpr std::int64_t day_count(std::chrono::sys_days start, std::chrono::sys_days end) const noexcept override
        {
            return t_.day_count(start, end);
        }
        /// \brief The adapted convention's year fraction.
        /// \param start The first day.
        /// \param end   The last day.
        /// \return The exact fraction.
        [[nodiscard]] constexpr finance::year_fraction year_fraction(std::chrono::sys_days start,
                                                                     std::chrono::sys_days end) const noexcept override
        {
            return t_.year_fraction(start, end);
        }

    private:
        T t_{};
    };

    static_assert(day_count_convention<day_count_interface>,
                  "snicholls::day_count_interface must model snicholls::day_count_convention");

} // namespace snicholls

#endif // snicholls_core_interfaces_day_count_hpp
