//
//  time_zone_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_time_zone_conformance`: the conformance kit of the `time_zone` contract.
/// \ingroup core
///
/// **Overview.** Drives any `time_zone`, in either form, through the contract of `concepts/time_zone.hpp` over a window
/// of instants, and reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a
/// default-constructible type with
/// - `auto& zone()`: the implementation (a zone, a `time_zone_interface&`, a `time_zone_ref`);
/// - `std::chrono::sys_seconds from()` and `to()`: the window to scan (a few years is plenty).
///
/// A fresh probe is made for every check. The checks, each at every hour of the window and at every transition in it:
/// `the rules hold at the instant`, `intervals follow one another`, `a unique wall-clock time round-trips`, `a gap is
/// nonexistent and resolves by policy`, `a fold is ambiguous and resolves by policy`, `the name is stable`. A zone
/// without transitions in the window passes the gap and fold checks vacuously.
///
/// \snippet interfaces/time_zone_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/time_zone.hpp`, `interfaces/time_zone.hpp`, `interfaces/capability_conformance.hpp`;
/// run on the references in every form by `tests/interfaces/time_zone_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit (about nine calls to `info` per hour of the window).
///
/// **Tests and demos.** `tests/interfaces/time_zone_conformance_test.cpp` (the kit catches broken zones),
/// `demos/interfaces/time_zone_conformance_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<string>`, `<type_traits>`, `<utility>`, `concepts/time_zone.hpp`,
/// `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** Transitions are found by hourly sampling: two within one hour are seen as one.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_time_zone_conformance_hpp
#define snicholls_core_interfaces_time_zone_conformance_hpp

#include <snicholls/core/concepts/time_zone.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <chrono>
#include <string>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept time_zone_probe = std::is_default_constructible_v<Probe> &&
            time_zone<std::remove_reference_t<decltype(std::declval<Probe&>().zone())>> && requires(Probe& p) {
                { p.from() } -> std::same_as<std::chrono::sys_seconds>;
                { p.to() } -> std::same_as<std::chrono::sys_seconds>;
            };

        // Calls f(before, after) for every change of rules found by hourly sampling of [from, to).
        template <typename Z, typename F>
        void for_each_transition(const Z& z, std::chrono::sys_seconds from, std::chrono::sys_seconds to, F&& f)
        {
            for (auto t = from; t < to; t += std::chrono::hours{1}) {
                const zone_info here = z.info(t);
                if (here.end < to && here.end > t) {
                    const zone_info next = z.info(here.end);
                    if (next.offset != here.offset)
                        f(here, next);
                    t = here.end - std::chrono::hours{1};
                }
            }
        }

        constexpr std::chrono::local_seconds local_of(std::chrono::sys_seconds t, std::chrono::seconds offset) noexcept
        {
            return std::chrono::local_seconds(t.time_since_epoch() + offset);
        }
    } // namespace detail

    /// \brief Runs every check of the `time_zone` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::time_zone_probe<Probe>
    [[nodiscard]] capability_conformance_report run_time_zone_conformance()
    {
        using namespace std::chrono;
        capability_conformance_report report;

        detail::capability_check<Probe>(report, "the rules hold at the instant", [](Probe& p, const auto& expect) {
            const auto& z = p.zone();
            bool holds = true, bounded = true;
            for (auto t = p.from(); t < p.to(); t += hours{1}) {
                const zone_info i = z.info(t);
                holds = holds && i.begin <= t && t < i.end;
                bounded = bounded && i.offset > -hours{24} && i.offset < hours{24} && !i.abbrev.empty();
            }
            expect(holds, "begin <= t < end at every hour");
            expect(bounded, "offsets are under a day and every interval has an abbreviation");
        });

        detail::capability_check<Probe>(report, "intervals follow one another", [](Probe& p, const auto& expect) {
            const auto& z = p.zone();
            bool follows = true;
            for (auto t = p.from(); t < p.to(); t += hours{1}) {
                const zone_info i = z.info(t);
                if (i.end < p.to())
                    follows = follows && z.info(i.end).begin == i.end && z.info(i.end - seconds{1}) == i;
            }
            expect(follows, "the next interval begins where one ends");
        });

        detail::capability_check<Probe>(report, "a unique wall-clock time round-trips", [](Probe& p, const auto& expect) {
            const auto& z = p.zone();
            bool round_trips = true, classified = true;
            for (auto t = p.from(); t < p.to(); t += hours{1}) {
                const utc_s u{t};
                const auto l = to_local(z, u);
                const local_info li = z.info(floor<seconds>(l));
                if (li.kind == local_time_kind::unique) {
                    const auto back = from_local(z, l);
                    round_trips = round_trips && back && *back == u;
                } else {
                    // a time that is not unique must still be one of its candidates
                    classified = classified && li.kind == local_time_kind::ambiguous &&
                                 (li.first.offset == offset_at(z, u) || li.second.offset == offset_at(z, u));
                }
            }
            expect(round_trips, "from_local(to_local(t)) == t whenever the local time is unique");
            expect(classified, "a local time reached from an instant is never nonexistent");
        });

        detail::capability_check<Probe>(report, "a gap is nonexistent and resolves by policy", [](Probe& p, const auto& expect) {
            const auto& z = p.zone();
            detail::for_each_transition(z, p.from(), p.to(), [&](const zone_info& before, const zone_info& after) {
                if (after.offset <= before.offset)
                    return;
                const auto width = after.offset - before.offset;
                const local_seconds in_gap = detail::local_of(after.begin, before.offset) + width / 2;
                const local_info li = z.info(in_gap);
                expect(li.kind == local_time_kind::nonexistent, "a time in the gap is nonexistent");
                expect(li.first.offset == before.offset && li.second.offset == after.offset, "first before, second after");
                const auto rejected = from_local(z, in_gap);
                expect(rejected.is_err() && rejected.unwrap_err() == time_zone_errc::nonexistent_local_time, "rejected by default");
                const auto next = from_local(z, in_gap, {fold_policy::reject, gap_policy::next_valid});
                expect(next && next->sys_time() == after.begin, "next_valid is the transition");
                const auto shifted = from_local(z, in_gap, {fold_policy::reject, gap_policy::shift_forward});
                expect(shifted && shifted->sys_time() == sys_seconds(in_gap.time_since_epoch() - before.offset),
                       "shift_forward keeps the earlier offset");
            });
        });

        detail::capability_check<Probe>(report, "a fold is ambiguous and resolves by policy", [](Probe& p, const auto& expect) {
            const auto& z = p.zone();
            detail::for_each_transition(z, p.from(), p.to(), [&](const zone_info& before, const zone_info& after) {
                if (after.offset >= before.offset)
                    return;
                const auto width = before.offset - after.offset;
                const local_seconds in_fold = detail::local_of(after.begin, after.offset) + width / 2;
                const local_info li = z.info(in_fold);
                expect(li.kind == local_time_kind::ambiguous, "a time in the fold is ambiguous");
                expect(li.first.offset == before.offset && li.second.offset == after.offset, "first is the earlier offset's");
                const auto rejected = from_local(z, in_fold);
                expect(rejected.is_err() && rejected.unwrap_err() == time_zone_errc::ambiguous_local_time, "rejected by default");
                const auto early = from_local(z, in_fold, {fold_policy::earliest, gap_policy::reject});
                const auto late = from_local(z, in_fold, {fold_policy::latest, gap_policy::reject});
                expect(early && late && *early < *late && *late - *early == width, "earliest and latest are the width apart");
            });
        });

        detail::capability_check<Probe>(report, "the name is stable", [](Probe& p, const auto& expect) {
            const auto& z = p.zone();
            const std::string first{z.name()};
            expect(!first.empty(), "the zone has a name");
            expect(z.name() == first, "the name does not change");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_time_zone_conformance_hpp
