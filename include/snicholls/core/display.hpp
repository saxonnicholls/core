//
//  display.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Display attributes: `colour`, `sign_colours` (P&L green and red), `label`, `number_format`, `tooltip`; and their JSON export.
/// \ingroup core
///
/// **Overview.** Colour is how a trader reads a screen that changes every second, so a currency, a greek or a P&L
/// must look the same in the terminal, in imgui and in the browser. These are the attributes every printer
/// understands, declared once with `meta/attributes.hpp`:
///
/// - `colour` (from `colour.hpp`): the value's colour.
/// - `sign_colours`: for a number, green when positive, red when negative, grey at zero or within a dead zone;
///   wins over `colour`.
/// - `threshold_colours`: for a number, a colour per band of ascending cut points (amber from 5 s, red from 15 s);
///   wins over both.
/// - `label`: the text to show instead of the value (an enumerator's display name).
/// - `number_format`: a `std::format` specification for a number (`"{:+.2f}"`), checked by the compiler.
/// - `units` (`"bps"`, `"USD"`, `"s"`), written after a number; `scale` (`{1e6, "m"}`), dividing it first.
/// - `minor_units`: a currency's decimal places, for whatever formats an amount in it.
/// - `tooltip`: the text for a mouse-over, a string literal fixed at compile time; the terminal ignores it.
///
/// `acout` prints a value in its colour, with its label or format; `stats::pivot` does the same for its keys and
/// cells. `write_json` and `write_enum_json` render the same attributes as JSON for imgui and React, from a
/// small program run at build time, so the C++ declaration is the only place a colour or a tooltip is written.
/// A caller's own attribute joins the JSON by specialising `attribute_json`.
///
/// \snippet display_demo.cpp display
///
/// **Relationships.** `meta/attributes.hpp`, `colour.hpp`, `acout.hpp`, `stats/pivot.hpp`. Group: \ref core.
///
/// **Guarantees.** The attributes are literal types, `constexpr` throughout; choosing a colour by sign is one
/// comparison. NaN has no sign colour. `rgba` gives imgui's four floats from a colour, `constexpr`.
///
/// **Errors.** None: the JSON writer sets the stream's state on a failed write, as any `operator<<` does.
///
/// **Performance.** Not benchmarked: printing; the lookup it rests on is benchmarked in `meta/attributes.hpp`.
///
/// **Tests and demos.** `tests/display_test.cpp`, `demos/display_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<charconv>`, `<cmath>`, `<cstddef>`, `<initializer_list>`, `<concepts>`, `<optional>`, `<ostream>`, `<string_view>`,
/// `<type_traits>`, `colour.hpp`, `meta/attributes.hpp`, `meta/enums.hpp`.
///
/// **Known issues.** `std::format` has no thousands separator except through a locale (`L`), so `{:,}` is refused.
///
/// **Status.** New in 0.4.0.

#ifndef snicholls_core_display_hpp
#define snicholls_core_display_hpp

#include <snicholls/core/colour.hpp>
#include <snicholls/core/meta/attributes.hpp>
#include <snicholls/core/meta/enums.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <concepts>
#include <optional>
#include <ostream>
#include <string_view>
#include <type_traits>

namespace snicholls {

    /// \brief The colours of a number by its sign: P&L green when positive, red when negative, grey within the dead zone.
    /// \ingroup core
    struct sign_colours {
        colour positive = colours::green;   ///< Above the dead zone.
        colour negative = colours::red;     ///< Below it.
        colour zero = colours::grey;        ///< Within it: |x| <= dead_zone (zero alone by default).
        double dead_zone = 0;               ///< Half the width of the neutral band: 0.5 makes -0.5 to 0.5 grey.

        /// \brief The colour of `x`.
        /// \param x The number.
        /// \return Its colour; none for NaN.
        [[nodiscard]] constexpr std::optional<colour> of(double x) const noexcept
        {
            if (x > dead_zone)
                return positive;
            if (x < -dead_zone)
                return negative;
            if (x == x)
                return zero;
            return std::nullopt;
        }
    };

    /// \brief Colours by ascending cut points: `threshold_colours{colours::green, {{5, colours::yellow}, {15, colours::red}}}`
    /// is green below 5, yellow from 5, red from 15 (staleness in seconds, a book's age).
    /// \ingroup core
    ///
    /// Wins over `sign_colours` and `colour` for a number. Up to eight cut points, checked ascending by the compiler.
    struct threshold_colours {
        /// \brief A cut point and the colour from it on.
        struct step {
            double from;   ///< The cut point.
            colour tint;   ///< The colour at and above it.
        };
        colour below;                  ///< The colour below the first cut point.
        std::array<step, 8> steps{};   ///< The cut points, ascending; the first `count` are used.
        std::size_t count = 0;         ///< How many.

        /// \brief Check the cut points at compile time.
        /// \param below_first The colour below the first cut point.
        /// \param from        The cut points and their colours, ascending, at most eight.
        consteval threshold_colours(colour below_first, std::initializer_list<step> from) : below(below_first)
        {
            if (from.size() > steps.size())
                throw "snicholls::threshold_colours: at most eight cut points";
            for (const step& st : from) {
                if (count > 0 && !(st.from > steps[count - 1].from))
                    throw "snicholls::threshold_colours: cut points must ascend";
                steps[count++] = st;
            }
        }

        /// \brief The colour of `x`.
        /// \param x The number.
        /// \return Its colour; none for NaN.
        [[nodiscard]] constexpr std::optional<colour> of(double x) const noexcept
        {
            if (x != x)
                return std::nullopt;
            colour c = below;
            for (std::size_t i = 0; i < count; ++i)
                if (x >= steps[i].from)
                    c = steps[i].tint;
            return c;
        }
    };

    /// \brief A unit written after a number, with a space: `units{"bps"}` prints `12.5 bps`.
    /// \ingroup core
    struct units {
        std::string_view text;   ///< The unit; a string literal.
    };

    /// \brief A number divided by `factor` before it is formatted, then `suffix`: `scale{1e6, "m"}` prints 1250000 as `1.25m`.
    /// \ingroup core
    ///
    /// Composes with `number_format` (applied to the scaled number) and `units` (after the suffix): `+1.25m USD`.
    struct scale {
        double factor;             ///< The divisor.
        std::string_view suffix;   ///< Written straight after the number.
    };

    /// \brief A currency's decimal places: `minor_units{2}` for USD, `{0}` for JPY, `{3}` for KWD.
    /// \ingroup core
    ///
    /// Read by whatever formats an amount in that currency; exported to JSON for the front end.
    struct minor_units {
        int digits;   ///< Decimal places.
    };

    /// \brief The text to show instead of the value.
    /// \ingroup core
    struct label {
        std::string_view text;   ///< The text; a string literal.
    };

    /// \brief A `std::format` specification for a number, checked by the compiler: `number_format{"{:+.2f}"}`.
    /// \ingroup core
    ///
    /// The number is formatted as a `double`, so one specification serves integers and floating point alike, and a
    /// specification the compiler accepted cannot fail when printed.
    struct number_format {
        std::string_view spec;   ///< The specification: one replacement field.

        /// \brief Check `s` at compile time: `{` `:` [fill]align, sign, `#`, `0`, width, `.precision`, `L`, one of `aAeEfFgG`, `}`.
        /// \param s The specification; anything else is a compile error.
        consteval number_format(std::string_view s) : spec(s)   // NOLINT(google-explicit-constructor): number_format{"{:+.2f}"}
        {
            const auto digit = [](char c) { return c >= '0' && c <= '9'; };
            const auto align = [](char c) { return c == '<' || c == '>' || c == '^'; };
            if (s.size() < 2 || s.front() != '{' || s.back() != '}')
                throw "snicholls::number_format: not one {...} replacement field";
            std::string_view f = s.substr(1, s.size() - 2);
            if (f.empty())
                return;
            if (f.front() != ':')
                throw "snicholls::number_format: the field must be {} or start {:";
            f.remove_prefix(1);
            const auto take = [&f](auto pred) {
                const bool took = !f.empty() && pred(f.front());
                if (took)
                    f.remove_prefix(1);
                return took;
            };
            if (f.size() >= 2 && align(f[1]) && f[0] != '{' && f[0] != '}')
                f.remove_prefix(2);
            else
                take(align);
            take([](char c) { return c == '+' || c == '-' || c == ' '; });
            take([](char c) { return c == '#'; });
            take([](char c) { return c == '0'; });
            while (take(digit)) {
            }
            if (take([](char c) { return c == '.'; }) && !take(digit))
                throw "snicholls::number_format: a precision needs digits";
            while (take(digit)) {
            }
            take([](char c) { return c == 'L'; });
            take([](char c) { return std::string_view{"aAeEfFgG"}.find(c) != std::string_view::npos; });
            if (!f.empty())
                throw "snicholls::number_format: not a std::format specification for a double";
        }
    };

    /// \brief The text of a mouse-over; a string literal fixed at compile time.
    /// \ingroup core
    struct tooltip {
        std::string_view text;   ///< The text.
    };

    /// \brief A number, or a struct or strong type whose `value` member is one: what the number attributes apply to.
    /// \ingroup core
    template <typename T>
    concept number_like = std::is_arithmetic_v<T> || requires(const T& t) { requires std::is_arithmetic_v<std::remove_cvref_t<decltype(t.value)>>; };

    /// \brief The number in a `number_like` value.
    /// \ingroup core
    /// \param x The number.
    /// \return It.
    template <typename T>
        requires std::is_arithmetic_v<T>
    [[nodiscard]] constexpr T numeric_of(T x) noexcept
    {
        return x;
    }
    /// \brief The number in a `number_like` value.
    /// \ingroup core
    /// \param x A struct or strong type with an arithmetic `value`.
    /// \return Its `value`.
    template <number_like T>
        requires(!std::is_arithmetic_v<T>)
    [[nodiscard]] constexpr auto numeric_of(const T& x) noexcept
    {
        return x.value;
    }

    /// \brief Whether values of type `T` carry an attribute that changes how they print.
    /// \ingroup core
    template <typename T>
    concept has_display_attributes = has_attribute_v<colour, T> || has_attribute_v<sign_colours, T> || has_attribute_v<threshold_colours, T> ||
                                     has_attribute_v<label, T> || has_attribute_v<number_format, T> || has_attribute_v<units, T> ||
                                     has_attribute_v<scale, T>;

    namespace detail::display {
        // Ranked overloads: the highest rank that applies wins, so thresholds beat signs beat a plain colour.
        /// \cond
        template <int N>
        struct rank : rank<N - 1> {};
        template <>
        struct rank<0> {};
        /// \endcond

        template <typename Set, typename T>
            requires(Set::template has<threshold_colours> && number_like<T>)
        constexpr std::optional<colour> colour_of(const Set& s, const T& v, rank<3>) noexcept
        {
            return s.template get<threshold_colours>().of(static_cast<double>(numeric_of(v)));
        }
        template <typename Set, typename T>
            requires(Set::template has<sign_colours> && number_like<T>)
        constexpr std::optional<colour> colour_of(const Set& s, const T& v, rank<2>) noexcept
        {
            return s.template get<sign_colours>().of(static_cast<double>(numeric_of(v)));
        }
        template <typename Set, typename T>
            requires(Set::template has<colour>)
        constexpr std::optional<colour> colour_of(const Set& s, const T&, rank<1>) noexcept
        {
            return s.template get<colour>();
        }
        template <typename Set, typename T>
        constexpr std::optional<colour> colour_of(const Set&, const T&, rank<0>) noexcept
        {
            return std::nullopt;
        }
    } // namespace detail::display

    /// \brief The colour a value prints in: for a number, by `threshold_colours`, else by `sign_colours`; else its `colour`; else none.
    /// \ingroup core
    /// \tparam Set An `attribute_set`.
    /// \tparam T   The value's type.
    /// \param s The attributes.
    /// \param v The value.
    /// \return The colour, or none.
    template <typename Set, typename T>
    [[nodiscard]] constexpr std::optional<colour> display_colour(const Set& s, const T& v) noexcept
    {
        return detail::display::colour_of(s, v, detail::display::rank<3>{});
    }

    /// \brief A colour as imgui's four floats, red, green, blue and alpha in [0, 1].
    /// \ingroup core
    /// \param c The colour.
    /// \return The floats.
    [[nodiscard]] constexpr std::array<float, 4> rgba(colour c) noexcept
    {
        return {static_cast<float>((c.rgb >> 16) & 0xFFu) / 255.0f, static_cast<float>((c.rgb >> 8) & 0xFFu) / 255.0f,
                static_cast<float>(c.rgb & 0xFFu) / 255.0f, 1.0f};
    }

    // ---- JSON, for imgui and React ----------------------------------------------------------------------------------

    /// \brief How an attribute is written as JSON: specialise with `name` and `write(os, a)` to export your own.
    /// \ingroup core
    /// \tparam A The attribute type; one without a specialisation is left out of the JSON.
    template <typename A>
    struct attribute_json;

    namespace detail::display {
        void write_json_string(std::ostream& os, std::string_view s);

        // A number as the shortest text that reads back to it, whatever the stream's precision.
        inline void write_json_number(std::ostream& os, double x)
        {
            char text[32];
            const auto end = std::to_chars(text, text + sizeof text, x).ptr;
            os.write(text, end - text);
        }
    } // namespace detail::display

    /// \brief Whether attribute type `A` is written to JSON.
    /// \ingroup core
    template <typename A>
    concept json_attribute = requires(std::ostream& os, const A& a) {
        { attribute_json<A>::name } -> std::convertible_to<std::string_view>;
        attribute_json<A>::write(os, a);
    };

    namespace detail::display {
        // One attribute as "name": value, or nothing for an attribute without attribute_json.
        template <json_attribute A>
        void write_member(std::ostream& os, const char*& separator, const A& a)
        {
            os << separator;
            write_json_string(os, attribute_json<A>::name);
            os << ": ";
            attribute_json<A>::write(os, a);
            separator = ", ";
        }
        template <typename A>
        void write_member(std::ostream&, const char*&, const A&) noexcept
        {
        }

        inline void write_json_string(std::ostream& os, std::string_view s)
        {
            os << '"';
            for (const char c : s) {
                if (c == '"' || c == '\\')
                    os << '\\' << c;
                else if (c == '\n')
                    os << "\\n";
                else
                    os << c;
            }
            os << '"';
        }
    } // namespace detail::display

    /// \brief `"colour": "#rrggbb"`.
    template <>
    struct attribute_json<colour> {
        static constexpr std::string_view name = "colour";   ///< The key.
        /// \brief The value. \param os The stream. \param c The colour.
        static void write(std::ostream& os, const colour& c) { os << '"' << hex(c).view() << '"'; }
    };
    /// \brief `"sign_colours": {"positive": ..., "negative": ..., "zero": ...}`.
    template <>
    struct attribute_json<sign_colours> {
        static constexpr std::string_view name = "sign_colours";   ///< The key.
        /// \brief The value. \param os The stream. \param s The colours.
        static void write(std::ostream& os, const sign_colours& s)
        {
            os << "{\"positive\": \"" << hex(s.positive).view() << "\", \"negative\": \"" << hex(s.negative).view()
               << "\", \"zero\": \"" << hex(s.zero).view() << "\", \"dead_zone\": ";
            detail::display::write_json_number(os, s.dead_zone);
            os << '}';
        }
    };
    /// \brief `"threshold_colours": {"below": ..., "steps": [{"from": 5, "colour": ...}, ...]}`.
    template <>
    struct attribute_json<threshold_colours> {
        static constexpr std::string_view name = "threshold_colours";   ///< The key.
        /// \brief The value. \param os The stream. \param t The colours.
        static void write(std::ostream& os, const threshold_colours& t)
        {
            os << "{\"below\": \"" << hex(t.below).view() << "\", \"steps\": [";
            for (std::size_t i = 0; i < t.count; ++i)
            {
                os << (i ? ", " : "") << "{\"from\": ";
                detail::display::write_json_number(os, t.steps[i].from);
                os << ", \"colour\": \"" << hex(t.steps[i].tint).view() << "\"}";
            }
            os << "]}";
        }
    };
    /// \brief `"units": "bps"`.
    template <>
    struct attribute_json<units> {
        static constexpr std::string_view name = "units";   ///< The key.
        /// \brief The value. \param os The stream. \param u The units.
        static void write(std::ostream& os, const units& u) { detail::display::write_json_string(os, u.text); }
    };
    /// \brief `"scale": {"factor": 1000000, "suffix": "m"}`.
    template <>
    struct attribute_json<scale> {
        static constexpr std::string_view name = "scale";   ///< The key.
        /// \brief The value. \param os The stream. \param s The scale.
        static void write(std::ostream& os, const scale& s)
        {
            os << "{\"factor\": ";
            detail::display::write_json_number(os, s.factor);
            os << ", \"suffix\": ";
            detail::display::write_json_string(os, s.suffix);
            os << '}';
        }
    };
    /// \brief `"minor_units": 2`.
    template <>
    struct attribute_json<minor_units> {
        static constexpr std::string_view name = "minor_units";   ///< The key.
        /// \brief The value. \param os The stream. \param m The minor units.
        static void write(std::ostream& os, const minor_units& m) { os << m.digits; }
    };
    /// \brief `"label": "..."`.
    template <>
    struct attribute_json<label> {
        static constexpr std::string_view name = "label";   ///< The key.
        /// \brief The value. \param os The stream. \param l The label.
        static void write(std::ostream& os, const label& l) { detail::display::write_json_string(os, l.text); }
    };
    /// \brief `"format": "..."`.
    template <>
    struct attribute_json<number_format> {
        static constexpr std::string_view name = "format";   ///< The key.
        /// \brief The value. \param os The stream. \param f The format.
        static void write(std::ostream& os, const number_format& f) { detail::display::write_json_string(os, f.spec); }
    };
    /// \brief `"tooltip": "..."`.
    template <>
    struct attribute_json<tooltip> {
        static constexpr std::string_view name = "tooltip";   ///< The key.
        /// \brief The value. \param os The stream. \param t The tooltip.
        static void write(std::ostream& os, const tooltip& t) { detail::display::write_json_string(os, t.text); }
    };

    /// \brief An attribute set as a JSON object; attributes without an `attribute_json` are left out.
    /// \ingroup core
    /// \tparam As The attribute types.
    /// \param os The stream.
    /// \param s  The attributes.
    template <typename... As>
    void write_json(std::ostream& os, const attribute_set<As...>& s)
    {
        os << '{';
        [[maybe_unused]] const char* separator = "";   // unused for an empty set
        (detail::display::write_member(os, separator, s.template get<As>()), ...);
        os << '}';
    }

    /// \brief Every enumerator's attributes as one JSON object keyed by enumerator name: a theme file for imgui or React.
    /// \ingroup core
    /// \tparam E A `reflectable_enum` with `attributes`.
    /// \param os The stream.
    template <reflectable_enum E>
        requires has_attributes<E>
    void write_enum_json(std::ostream& os)
    {
        os << '{';
        const char* separator = "";
        for (const E e : enum_values_v<E>) {
            os << separator;
            detail::display::write_json_string(os, enum_name(e));
            os << ": ";
            write_json(os, attributes_of(e));
            separator = ", ";
        }
        os << '}';
    }

} // namespace snicholls

#endif /* snicholls_core_display_hpp */
