//
//  time_zone_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::fixed_offset_zone` and `snicholls::two_rule_zone`: the standard-library reference `time_zone`s.
/// \ingroup core
///
/// **Overview.** Two zones with no database behind them, for tests, examples and the kit:
/// - `fixed_offset_zone`: one offset forever (`UTC`, `UTC+10`, an exchange that never changes its clocks);
/// - `two_rule_zone`: a toy daylight-saving zone with one rule into daylight time and one out of it, each "the nth
///   (or last) weekday of a month at a wall-clock time", in either hemisphere. Real zones change their rules over the
///   years; this one never does, which is what makes it a fixed, readable test zone (and never a substitute for the
///   IANA database).
///
/// \snippet interfaces/time_zone_reference_demo.cpp reference
///
/// **Relationships.** Model `time_zone` (`concepts/time_zone.hpp`); pass `interfaces/time_zone_conformance.hpp`
/// directly and through `interfaces/time_zone.hpp`. Group: \ref core.
///
/// **Implementations / variants.** The two above. `sydney_like_zone()` and `new_york_like_zone()` are ready-made
/// `two_rule_zone`s with today's rules of those cities (not their history).
///
/// **Guarantees.** Immutable values, `constexpr`, no allocation; safe to share between threads.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked: six transitions are computed per `info` call.
///
/// **Tests and demos.** `tests/interfaces/time_zone_reference_test.cpp`,
/// `demos/interfaces/time_zone_reference_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<chrono>`, `<string_view>`, `interfaces/time_zone.hpp`.
///
/// **Known issues.** Years outside `std::chrono::year`'s range (about +-32767) are not supported by `two_rule_zone`.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_time_zone_reference_hpp
#define snicholls_core_interfaces_time_zone_reference_hpp

#include <snicholls/core/interfaces/time_zone.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <string_view>

namespace snicholls {

    /// \brief A zone with one offset forever.
    /// \ingroup core
    class fixed_offset_zone {
    public:
        /// \brief UTC.
        constexpr fixed_offset_zone() noexcept = default;
        /// \brief A zone called `name` at `offset`.
        /// \param name   The name.
        /// \param offset Local time minus UTC.
        /// \param abbrev The abbreviation.
        constexpr fixed_offset_zone(inplace_string<32> name, std::chrono::seconds offset, inplace_string<15> abbrev) noexcept
            : name_(name), info_{std::chrono::sys_seconds::min(), std::chrono::sys_seconds::max(), offset, {}, abbrev}
        {
        }

        /// \brief The name.
        /// \return It.
        [[nodiscard]] constexpr std::string_view name() const noexcept { return name_.view(); }
        /// \brief The rules, which hold forever.
        /// \return The one `zone_info`.
        [[nodiscard]] constexpr zone_info info(std::chrono::sys_seconds) const noexcept { return info_; }
        /// \brief Every wall-clock time is unique.
        /// \return `unique` with the one `zone_info`.
        [[nodiscard]] constexpr local_info info(std::chrono::local_seconds) const noexcept
        {
            return {local_time_kind::unique, info_, info_};
        }

    private:
        inplace_string<32> name_{"UTC"};
        zone_info info_{std::chrono::sys_seconds::min(), std::chrono::sys_seconds::max(), {}, {}, inplace_string<15>{"UTC"}};
    };

    /// \brief When a `two_rule_zone` changes its clocks: the nth weekday of a month (5: the last) at a wall-clock time.
    /// \ingroup core
    struct dst_rule {
        std::chrono::month month{std::chrono::October};     ///< the month
        std::chrono::weekday weekday{std::chrono::Sunday};  ///< the weekday
        unsigned nth{1};                                    ///< 1 to 4, or 5 for the last
        std::chrono::seconds at{std::chrono::hours{2}};     ///< the wall-clock time, on the clock in force before it

        /// \brief Equality, member by member.
        friend constexpr bool operator==(const dst_rule&, const dst_rule&) noexcept = default;
    };

    /// \brief A toy daylight-saving zone: standard time, and daylight time between two yearly rules.
    /// \ingroup core
    class two_rule_zone {
    public:
        /// \brief The zone.
        /// \param name         The name.
        /// \param standard     The standard offset.
        /// \param std_abbrev   The standard abbreviation.
        /// \param save         Daylight saving (positive).
        /// \param dst_abbrev   The daylight abbreviation.
        /// \param into_dst     When daylight time begins (`at` in standard time).
        /// \param out_of_dst   When it ends (`at` in daylight time).
        constexpr two_rule_zone(inplace_string<32> name, std::chrono::seconds standard, inplace_string<15> std_abbrev,
                                std::chrono::minutes save, inplace_string<15> dst_abbrev, dst_rule into_dst,
                                dst_rule out_of_dst) noexcept
            : name_(name), std_(standard), save_(save), std_abbrev_(std_abbrev), dst_abbrev_(dst_abbrev), into_(into_dst),
              out_(out_of_dst)
        {
        }

        /// \brief The name.
        /// \return It.
        [[nodiscard]] constexpr std::string_view name() const noexcept { return name_.view(); }

        /// \brief The rules at the instant `t`, with the transitions on either side.
        /// \param t The instant.
        /// \return The rules.
        [[nodiscard]] constexpr zone_info info(std::chrono::sys_seconds t) const noexcept
        {
            const auto local_day = std::chrono::floor<std::chrono::days>(t + std_);
            const int y = static_cast<int>(std::chrono::year_month_day(local_day).year());
            std::array<transition, 6> ts{};
            for (int i = 0; i < 3; ++i) {
                ts[2 * i] = {when(into_, y - 1 + i, std_), true};
                ts[2 * i + 1] = {when(out_, y - 1 + i, std_ + save_), false};
            }
            std::sort(ts.begin(), ts.end(), [](const transition& a, const transition& b) { return a.at < b.at; });
            std::size_t k = 0;
            while (k + 1 < ts.size() && ts[k + 1].at <= t)
                ++k;
            const bool dst = ts[k].into_dst;
            return {ts[k].at, k + 1 < ts.size() ? ts[k + 1].at : std::chrono::sys_seconds::max(), dst ? std_ + save_ : std_,
                    dst ? save_ : std::chrono::minutes{0}, dst ? dst_abbrev_ : std_abbrev_};
        }

        /// \brief Classifies the wall-clock time `l`.
        /// \param l The wall-clock time.
        /// \return Unique, in the gap when the clocks go forward, or in the fold when they go back.
        [[nodiscard]] constexpr local_info info(std::chrono::local_seconds l) const noexcept
        {
            const std::chrono::seconds dst_offset = std_ + save_;
            const zone_info as_std = info(std::chrono::sys_seconds(l.time_since_epoch() - std_));
            const zone_info as_dst = info(std::chrono::sys_seconds(l.time_since_epoch() - dst_offset));
            const bool std_ok = as_std.offset == std_;
            const bool dst_ok = as_dst.offset == dst_offset;
            if (std_ok && dst_ok)   // the earlier instant is the daylight one
                return {local_time_kind::ambiguous, as_dst, as_std};
            if (std_ok)
                return {local_time_kind::unique, as_std, as_std};
            if (dst_ok)
                return {local_time_kind::unique, as_dst, as_dst};
            // in the gap: interpreted on the standard offset it is already daylight time, and the other way round
            return {local_time_kind::nonexistent, as_dst, as_std};
        }

    private:
        struct transition {
            std::chrono::sys_seconds at{};
            bool into_dst = false;
        };

        // The instant of `r` in year `y`, on the clock at `offset`.
        static constexpr std::chrono::sys_seconds when(const dst_rule& r, int y, std::chrono::seconds offset) noexcept
        {
            const std::chrono::year year{y};
            const std::chrono::sys_days day = r.nth >= 5 ? std::chrono::sys_days(year / r.month / r.weekday[std::chrono::last])
                                                         : std::chrono::sys_days(year / r.month / r.weekday[r.nth]);
            return std::chrono::sys_seconds(day) + r.at - offset;
        }

        inplace_string<32> name_;
        std::chrono::seconds std_;
        std::chrono::minutes save_;
        inplace_string<15> std_abbrev_;
        inplace_string<15> dst_abbrev_;
        dst_rule into_;
        dst_rule out_;
    };

    /// \brief Sydney's rules since 2008: AEST +10:00, AEDT +11:00 from the first Sunday of October at 02:00 to the first
    /// Sunday of April at 03:00 (daylight time).
    /// \ingroup core
    /// \return The zone.
    [[nodiscard]] constexpr two_rule_zone sydney_like_zone() noexcept
    {
        using namespace std::chrono;
        return {"Sydney-like", hours{10}, "AEST", minutes{60}, "AEDT", {October, Sunday, 1, hours{2}}, {April, Sunday, 1, hours{3}}};
    }

    /// \brief New York's rules since 2007: EST -05:00, EDT -04:00 from the second Sunday of March at 02:00 to the first
    /// Sunday of November at 02:00 (daylight time).
    /// \ingroup core
    /// \return The zone.
    [[nodiscard]] constexpr two_rule_zone new_york_like_zone() noexcept
    {
        using namespace std::chrono;
        return {"New-York-like", hours{-5}, "EST", minutes{60}, "EDT", {March, Sunday, 2, hours{2}}, {November, Sunday, 1, hours{2}}};
    }

    static_assert(time_zone<fixed_offset_zone>, "snicholls::fixed_offset_zone must model snicholls::time_zone");
    static_assert(time_zone<two_rule_zone>, "snicholls::two_rule_zone must model snicholls::time_zone");

} // namespace snicholls

#endif // snicholls_core_interfaces_time_zone_reference_hpp
