//
//  time_zone.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The contract of crtp's time_zone_interface (interfaces/utc/time_zone_interface.hpp): the rules at an instant, a
//  wall-clock time classified, local to UTC with gaps and folds explicit, RFC 3339 text with an offset. Errors are a
//  result, not crtp's time_zone_error. The zone database (IANA tzdata, its loading, find by name) stays in ts-moveables
//  and crtp.
//

/// \file
/// \brief `snicholls::time_zone`: the rules of a time zone, and conversions between UTC and wall-clock time.
/// \ingroup core
///
/// **Overview.** An instant (`utc<D>`) has no zone. A zone turns instants into wall-clock times and back: exchange
/// sessions, cut-offs, statements, display. The data (the IANA database) is run-time data and the operating system's
/// or a library's; core holds the contract and the conversions written over it, so every zone plugs in:
/// - `z.name()`: the zone's name (`"Australia/Sydney"`, `"UTC+10"`);
/// - `z.info(t)` at an instant `std::chrono::sys_seconds`: the `zone_info` in force (offset, daylight saving,
///   abbreviation, and the interval `[begin, end)` the rules hold for);
/// - `z.info(l)` at a wall-clock time `std::chrono::local_seconds`: a `local_info` saying whether it is `unique`,
///   `nonexistent` (in a gap: clocks jumped forward) or `ambiguous` (in a fold: clocks went back), with the rules on
///   either side.
///
/// Written over the concept: `offset_at(z, t)`, `to_local(z, t)`, and `from_local(z, l, options)`, which returns the
/// instant or `time_zone_errc::nonexistent_local_time` / `ambiguous_local_time`, never a silent guess: gaps and folds
/// are resolved only by a policy the caller names (`local_options`). `to_text(t, offset)` writes RFC 3339 with a
/// numeric offset.
///
/// \snippet concepts/time_zone_demo.cpp concept
///
/// **The contract** (`interfaces/time_zone_conformance.hpp` checks it):
/// - `info(t)` holds at `t`: `begin <= t < end`, and the interval after `end` begins at `end`;
/// - a wall-clock time is `unique` exactly when one instant has it, and then `from_local(to_local(t)) == t`;
/// - in a gap `first` holds before the transition and `second` from it; in a fold `first` is the earlier offset's;
/// - an offset is less than a day either way.
///
/// **Relationships.** `time/utc.hpp`, `time/rfc3339.hpp`; the twin `interfaces/time_zone.hpp`, the references
/// `interfaces/time_zone_reference.hpp` (`fixed_offset_zone`, `two_rule_zone`), the kit
/// `interfaces/time_zone_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `fixed_offset_zone` and `two_rule_zone` (a toy daylight-saving zone for tests). Above core:
/// the IANA database adapters of ts-moveables and crtp (`utc_tz`).
///
/// **Guarantees.** Every function here is `constexpr` when the zone's `info` is; none allocates. `from_local` is
/// exact: one or two calls to `info`.
///
/// **Errors.** `from_local`: `nonexistent_local_time` and `ambiguous_local_time` under the `reject` policies (the
/// default). `to_text`: `std::errc::not_supported` in the text's `ec` for an offset with seconds (before about 1900),
/// `std::errc::value_too_large` when the local year leaves 0000 to 9999.
///
/// **Performance.** Not benchmarked: the cost is the zone's `info`.
///
/// **Tests and demos.** `tests/concepts/time_zone_test.cpp`, `demos/concepts/time_zone_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<concepts>`, `<cstdint>`, `<string_view>`, `<type_traits>`,
/// `containers/inplace_string.hpp`, `result.hpp`, `time/rfc3339.hpp`, `time/utc.hpp`.
///
/// **Known issues.** Leap seconds do not exist in `utc`, so none appear in local time. RFC 3339 cannot carry an
/// offset with seconds.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_time_zone_hpp
#define snicholls_core_concepts_time_zone_hpp

#include <snicholls/core/containers/inplace_string.hpp>
#include <snicholls/core/result.hpp>
#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/utc.hpp>

#include <chrono>
#include <concepts>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace snicholls {

    /// \brief Why a wall-clock time has no single instant.
    /// \ingroup core
    enum class time_zone_errc : std::uint8_t {
        nonexistent_local_time = 1,  ///< in a gap (clocks jumped forward), and the policy rejects gaps
        ambiguous_local_time = 2,    ///< in a fold (clocks went back), and the policy rejects folds
    };

    /// \brief The rules of a zone over one interval: the offset, daylight saving and the abbreviation.
    /// \ingroup core
    struct zone_info {
        std::chrono::sys_seconds begin{std::chrono::sys_seconds::min()};  ///< the first instant the rules hold (inclusive)
        std::chrono::sys_seconds end{std::chrono::sys_seconds::max()};    ///< the first instant they no longer hold
        std::chrono::seconds offset{};                                    ///< local time = UTC + offset
        std::chrono::minutes save{};                                      ///< daylight saving; zero in standard time
        inplace_string<15> abbrev{};                                      ///< `"AEST"`, `"AEDT"`, `"+09"`

        /// \brief Whether daylight saving is in force.
        /// \return `save != 0`.
        [[nodiscard]] constexpr bool is_dst() const noexcept { return save != std::chrono::minutes{0}; }

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const zone_info&, const zone_info&) noexcept = default;
    };

    /// \brief How a wall-clock time maps to instants in a zone.
    /// \ingroup core
    enum class local_time_kind : std::uint8_t {
        unique,       ///< exactly one instant
        nonexistent,  ///< in a gap: no instant
        ambiguous,    ///< in a fold: two instants
    };

    /// \brief A wall-clock time classified in a zone (as `std::chrono::local_info`).
    /// \ingroup core
    struct local_info {
        local_time_kind kind{local_time_kind::unique};  ///< unique, in a gap or in a fold
        zone_info first;   ///< unique: the rules; gap: the rules before it; fold: the earlier instant's rules
        zone_info second;  ///< gap: the rules from the transition; fold: the later instant's rules; unique: unspecified

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const local_info&, const local_info&) noexcept = default;
    };

    /// \brief A time zone: `z.name()`, `z.info(sys_seconds)` and `z.info(local_seconds)`.
    /// \ingroup core
    template <typename Z>
    concept time_zone = requires(const Z& z, std::chrono::sys_seconds t, std::chrono::local_seconds l) {
        { z.name() } -> std::same_as<std::string_view>;
        { z.info(t) } -> std::same_as<zone_info>;
        { z.info(l) } -> std::same_as<local_info>;
    };

    /// \brief What to do with a wall-clock time in a fold.
    /// \ingroup core
    enum class fold_policy : std::uint8_t {
        reject,    ///< `ambiguous_local_time`
        earliest,  ///< the first instant (on the earlier, usually daylight, offset)
        latest,    ///< the second instant
    };

    /// \brief What to do with a wall-clock time in a gap.
    /// \ingroup core
    enum class gap_policy : std::uint8_t {
        reject,         ///< `nonexistent_local_time`
        next_valid,     ///< the transition instant (`std::chrono::choose`): 02:30 becomes 03:00
        shift_forward,  ///< the distance kept on the earlier offset (java.time): 02:30 becomes 03:30
    };

    /// \brief How `from_local` resolves gaps and folds. Both reject by default: never a silent guess.
    /// \ingroup core
    struct local_options {
        fold_policy fold{fold_policy::reject};  ///< in a fold
        gap_policy gap{gap_policy::reject};     ///< in a gap
    };

    /// \brief The UTC offset of `zone` at the instant `t`.
    /// \ingroup core
    /// \param zone The zone.
    /// \param t    The instant.
    /// \return Local time minus UTC.
    template <time_zone Z, utc_duration D>
    [[nodiscard]] constexpr std::chrono::seconds offset_at(const Z& zone, const utc<D>& t)
    {
        return zone.info(std::chrono::floor<std::chrono::seconds>(t.sys_time())).offset;
    }

    /// \brief The wall-clock time in `zone` at the instant `t`.
    /// \ingroup core
    /// \param zone The zone.
    /// \param t    The instant.
    /// \return The local time, in the finer of `D` and seconds.
    template <time_zone Z, utc_duration D>
    [[nodiscard]] constexpr auto to_local(const Z& zone, const utc<D>& t)
    {
        using C = std::common_type_t<D, std::chrono::seconds>;
        return std::chrono::local_time<C>(std::chrono::duration_cast<C>(t.time_since_epoch()) + offset_at(zone, t));
    }

    /// \brief The instant of the wall-clock time `local` in `zone`, gaps and folds resolved by `options`.
    /// \ingroup core
    /// \param zone    The zone.
    /// \param local   The wall-clock time.
    /// \param options The policies; both reject by default.
    /// \return The instant, `nonexistent_local_time` or `ambiguous_local_time`.
    template <time_zone Z, utc_duration D>
    [[nodiscard]] constexpr result<utc<D>, time_zone_errc> from_local(const Z& zone, const std::chrono::local_time<D>& local,
                                                                     local_options options = {})
    {
        const local_info li = zone.info(std::chrono::floor<std::chrono::seconds>(local));
        const auto at = [&](std::chrono::seconds offset) {
            return utc<D>(typename utc<D>::time_point(local.time_since_epoch() - std::chrono::duration_cast<D>(offset)));
        };
        switch (li.kind) {
        case local_time_kind::unique: return ok(at(li.first.offset));
        case local_time_kind::nonexistent:
            if (options.gap == gap_policy::next_valid)
                return ok(utc<D>(typename utc<D>::time_point(std::chrono::duration_cast<D>(li.second.begin.time_since_epoch()))));
            if (options.gap == gap_policy::shift_forward)
                return ok(at(li.first.offset));
            return err(time_zone_errc::nonexistent_local_time);
        case local_time_kind::ambiguous:
            if (options.fold == fold_policy::earliest)
                return ok(at(li.first.offset));
            if (options.fold == fold_policy::latest)
                return ok(at(li.second.offset));
            return err(time_zone_errc::ambiguous_local_time);
        }
        return err(time_zone_errc::nonexistent_local_time);
    }

    /// \brief `t` as RFC 3339 local time with a numeric offset: `"2026-11-01T01:30:00-05:00"` (`Z` at offset zero).
    /// \ingroup core
    /// \param t         The instant.
    /// \param offset    The offset, a whole number of minutes, less than a day.
    /// \param precision The fraction digits, 0 to 9.
    /// \return The text; empty, with `ec` set, on failure (see the file's **Errors**).
    template <utc_duration D>
    [[nodiscard]] constexpr rfc3339_text to_text(const utc<D>& t, std::chrono::seconds offset, int precision = utc<D>::fraction_digits) noexcept
    {
        rfc3339_text text;
        const auto s = offset.count();
        if (s % 60 != 0 || s <= -86400 || s >= 86400) {
            text.ec = std::errc::not_supported;
            return text;
        }
        using C = std::common_type_t<D, std::chrono::seconds>;
        const auto ticks = std::chrono::duration_cast<C>(t.time_since_epoch());
        const auto shift = std::chrono::duration_cast<C>(offset);
        if ((shift.count() > 0 && ticks.count() > C::max().count() - shift.count()) ||
            (shift.count() < 0 && ticks.count() < C::min().count() - shift.count())) {
            text.ec = std::errc::value_too_large;
            return text;
        }
        const utc<C> local_as_utc(typename utc<C>::time_point(ticks + shift));
        text = to_text(local_as_utc, precision);
        if (!text || s == 0)
            return text;
        // replace the trailing 'Z' with +HH:MM
        const auto minutes = static_cast<unsigned>((s < 0 ? -s : s) / 60);
        char* p = text.buffer.data() + text.size - 1;
        p[0] = s < 0 ? '-' : '+';
        p[1] = static_cast<char>('0' + minutes / 600);
        p[2] = static_cast<char>('0' + minutes / 60 % 10);
        p[3] = ':';
        p[4] = static_cast<char>('0' + minutes % 60 / 10);
        p[5] = static_cast<char>('0' + minutes % 10);
        text.size = static_cast<std::uint8_t>(text.size + 5);
        return text;
    }

} // namespace snicholls

#endif // snicholls_core_concepts_time_zone_hpp
