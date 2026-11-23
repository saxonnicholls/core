//
//  names.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief English month and weekday names, declared once as presentation attributes of `month_of_year` and
/// `day_of_week`.
/// \ingroup core
///
/// **Overview.** "November" and "Mon" are presentation, so they are attributes of a type (CLAUDE.md, "Presentation
/// is a compile-time attribute of the type"), not string tables at call sites. `month_of_year` (January = 1) and
/// `day_of_week` (Monday = 1, as ISO 8601 and `std::chrono::weekday::iso_encoding()` number them) declare, per
/// enumerator, a `label` (the English name) and an `abbreviation` (the three-letter form). Every printer reads
/// them: acout, `print_to`, `stats::pivot`, and `write_enum_json` for imgui and React. For `std::chrono`'s own
/// types, `month_name`, `month_abbreviation`, `weekday_name` and `weekday_abbreviation` read the same attributes.
///
/// \snippet time/names_demo.cpp names
///
/// **Relationships.** `meta/attributes.hpp` (`per_enumerator`, `attributes_of`), `display.hpp` (`label`,
/// `attribute_json`), `time/week_date.hpp`, `time/utc_format.hpp`. Group: \ref core.
///
/// **Implementations.** One.
///
/// **Guarantees.** constexpr, `noexcept`; a name is one table load. The views are of string literals: valid for
/// the whole program.
///
/// **Errors.** None. A `std::chrono::month` or `weekday` that is not `ok()` has the empty name.
///
/// **Performance.** One table load; not benchmarked.
///
/// **Tests and demos.** `tests/time/names_test.cpp`, `demos/time/names_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<chrono>`, `<cstdint>`, `<ostream>`, `<string_view>`, `display.hpp` (which brings
/// `<ostream>` for the JSON export), `meta/attributes.hpp`.
///
/// **Known issues.** The enumerations and their `attributes` are declared in `snicholls::calendar` and brought into
/// `snicholls` by using-declarations, so `snicholls` itself still declares nothing called `attributes`
/// (`meta/attributes.hpp`). English only: other languages are the caller's attributes, or CLDR data in a library
/// above core.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_time_names_hpp
#define snicholls_core_time_names_hpp

#include <snicholls/core/display.hpp>
#include <snicholls/core/meta/attributes.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <string_view>

namespace snicholls {

    /// \brief A presentation attribute: the short form of a name ("Jan", "Mon").
    /// \ingroup core
    struct abbreviation {
        std::string_view text;   ///< The text; a string literal.
    };

    /// \brief `"abbreviation": "..."`.
    template <>
    struct attribute_json<abbreviation> {
        static constexpr std::string_view name = "abbreviation";   ///< The key.
        /// \brief The value. \param os The stream. \param a The abbreviation.
        static void write(std::ostream& os, const abbreviation& a) { detail::display::write_json_string(os, a.text); }
    };

    /// \brief The months and weekdays as enumerations with names; see `snicholls::month_of_year`.
    namespace calendar {

        /// \brief A month of the year, January = 1, with its English name and abbreviation as attributes.
        /// \ingroup core
        enum class month_of_year : std::uint8_t {
            january = 1,   ///< January
            february,      ///< February
            march,         ///< March
            april,         ///< April
            may,           ///< May
            june,          ///< June
            july,          ///< July
            august,        ///< August
            september,     ///< September
            october,       ///< October
            november,      ///< November
            december,      ///< December
        };

        /// \brief A day of the week, Monday = 1 to Sunday = 7 (ISO 8601), with its English name and abbreviation.
        /// \ingroup core
        enum class day_of_week : std::uint8_t {
            monday = 1,   ///< Monday
            tuesday,      ///< Tuesday
            wednesday,    ///< Wednesday
            thursday,     ///< Thursday
            friday,       ///< Friday
            saturday,     ///< Saturday
            sunday,       ///< Sunday
        };

        /// \brief The month's English name and abbreviation.
        /// \param m The month.
        /// \return Its attribute set: `label`, `abbreviation`.
        constexpr const auto& attributes(month_of_year m) noexcept
        {
            return per_enumerator(m, [] { return std::array{
                attrs(label{"January"}, abbreviation{"Jan"}),   attrs(label{"February"}, abbreviation{"Feb"}),
                attrs(label{"March"}, abbreviation{"Mar"}),     attrs(label{"April"}, abbreviation{"Apr"}),
                attrs(label{"May"}, abbreviation{"May"}),       attrs(label{"June"}, abbreviation{"Jun"}),
                attrs(label{"July"}, abbreviation{"Jul"}),      attrs(label{"August"}, abbreviation{"Aug"}),
                attrs(label{"September"}, abbreviation{"Sep"}), attrs(label{"October"}, abbreviation{"Oct"}),
                attrs(label{"November"}, abbreviation{"Nov"}),  attrs(label{"December"}, abbreviation{"Dec"}),
            }; });
        }

        /// \brief The weekday's English name and abbreviation.
        /// \param d The weekday.
        /// \return Its attribute set: `label`, `abbreviation`.
        constexpr const auto& attributes(day_of_week d) noexcept
        {
            return per_enumerator(d, [] { return std::array{
                attrs(label{"Monday"}, abbreviation{"Mon"}),   attrs(label{"Tuesday"}, abbreviation{"Tue"}),
                attrs(label{"Wednesday"}, abbreviation{"Wed"}), attrs(label{"Thursday"}, abbreviation{"Thu"}),
                attrs(label{"Friday"}, abbreviation{"Fri"}),   attrs(label{"Saturday"}, abbreviation{"Sat"}),
                attrs(label{"Sunday"}, abbreviation{"Sun"}),
            }; });
        }

    } // namespace calendar

    using calendar::day_of_week;
    using calendar::month_of_year;

    /// \brief The `month_of_year` of a `std::chrono::month`.
    /// \ingroup core
    /// \pre `m.ok()`.
    /// \param m The month.
    /// \return The same month.
    [[nodiscard]] constexpr month_of_year to_month_of_year(std::chrono::month m) noexcept
    {
        return static_cast<month_of_year>(static_cast<unsigned>(m));
    }

    /// \brief The `day_of_week` of a `std::chrono::weekday`.
    /// \ingroup core
    /// \pre `w.ok()`.
    /// \param w The weekday.
    /// \return The same weekday.
    [[nodiscard]] constexpr day_of_week to_day_of_week(std::chrono::weekday w) noexcept
    {
        return static_cast<day_of_week>(w.iso_encoding());
    }

    /// \brief The English name of a month: "January".
    /// \ingroup core
    /// \param m The month.
    /// \return Its `label`, or empty when `!m.ok()`.
    [[nodiscard]] constexpr std::string_view month_name(std::chrono::month m) noexcept
    {
        return m.ok() ? attributes_of(to_month_of_year(m)).get<label>().text : std::string_view{};
    }

    /// \brief The three-letter English abbreviation of a month: "Jan".
    /// \ingroup core
    /// \param m The month.
    /// \return Its `abbreviation`, or empty when `!m.ok()`.
    [[nodiscard]] constexpr std::string_view month_abbreviation(std::chrono::month m) noexcept
    {
        return m.ok() ? attributes_of(to_month_of_year(m)).get<abbreviation>().text : std::string_view{};
    }

    /// \brief The English name of a weekday: "Monday".
    /// \ingroup core
    /// \param w The weekday.
    /// \return Its `label`, or empty when `!w.ok()`.
    [[nodiscard]] constexpr std::string_view weekday_name(std::chrono::weekday w) noexcept
    {
        return w.ok() ? attributes_of(to_day_of_week(w)).get<label>().text : std::string_view{};
    }

    /// \brief The three-letter English abbreviation of a weekday: "Mon".
    /// \ingroup core
    /// \param w The weekday.
    /// \return Its `abbreviation`, or empty when `!w.ok()`.
    [[nodiscard]] constexpr std::string_view weekday_abbreviation(std::chrono::weekday w) noexcept
    {
        return w.ok() ? attributes_of(to_day_of_week(w)).get<abbreviation>().text : std::string_view{};
    }

} // namespace snicholls

#endif // snicholls_core_time_names_hpp
