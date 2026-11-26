//
//  uuid_generator_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_uuid_generator_conformance`: the conformance kit of the `uuid_generator` contract.
/// \ingroup core
///
/// **Overview.** Drives any `uuid_generator`, in either form, through the contract of `concepts/uuid_generator.hpp`
/// and reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type
/// with
/// - `auto& generator()`: the implementation (a generator, a `uuid_generator_interface&`, a `uuid_generator_ref`);
/// - `std::int64_t clock_ms()`: the Unix millisecond its clock reads now;
/// - `void clock_steps_back()`: the clock goes back (a second, say), as a system clock may;
/// - `void randomness_fails()`: the next draws of its random source fail.
///
/// A fresh probe is made for every check. The checks: `v4 is version 4 and distinct`, `v7 is version 7 at the
/// clock's millisecond`, `v7 strictly increases within a millisecond`, `v7 never goes back with the clock`, `a failed
/// random source is unavailable`.
///
/// \snippet interfaces/uuid_generator_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/uuid_generator.hpp`, `interfaces/uuid_generator.hpp`,
/// `interfaces/capability_conformance.hpp`; run on `basic_uuid_generator` in every form by
/// `tests/interfaces/uuid_generator_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit (it makes about 10,000 identifiers).
///
/// **Tests and demos.** `tests/interfaces/uuid_generator_conformance_test.cpp` (the kit catches broken generators),
/// `demos/interfaces/uuid_generator_conformance_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cstdint>`, `<type_traits>`, `<utility>`, `<vector>`,
/// `concepts/uuid_generator.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** Randomness is not tested for quality, only for distinctness.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_uuid_generator_conformance_hpp
#define snicholls_core_interfaces_uuid_generator_conformance_hpp

#include <snicholls/core/concepts/uuid_generator.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept uuid_generator_probe = std::is_default_constructible_v<Probe> &&
            uuid_generator<std::remove_reference_t<decltype(std::declval<Probe&>().generator())>> && requires(Probe& p) {
                { p.clock_ms() } -> std::convertible_to<std::int64_t>;
                p.clock_steps_back();
                p.randomness_fails();
            };

        constexpr std::int64_t uuid_ms(const uuid& u) noexcept
        {
            const auto t = u.timestamp();
            return t ? static_cast<std::int64_t>(t->time_since_epoch().count()) : -1;
        }
    } // namespace detail

    /// \brief Runs every check of the `uuid_generator` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::uuid_generator_probe<Probe>
    [[nodiscard]] capability_conformance_report run_uuid_generator_conformance()
    {
        capability_conformance_report report;

        detail::capability_check<Probe>(report, "v4 is version 4 and distinct", [](Probe& p, const auto& expect) {
            std::vector<uuid> seen;
            bool shape = true;
            for (int i = 0; i < 1000; ++i) {
                const auto r = p.generator().v4();
                expect(r.is_ok(), "v4 succeeds");
                if (!r)
                    return;
                shape = shape && r->version() == 4 && r->variant() == uuid_variant::rfc9562;
                seen.push_back(*r);
            }
            expect(shape, "every v4 has version 4 and the RFC 9562 variant");
            std::sort(seen.begin(), seen.end());
            expect(std::adjacent_find(seen.begin(), seen.end()) == seen.end(), "no two of 1000 are equal");
        });

        detail::capability_check<Probe>(report, "v7 is version 7 at the clock's millisecond", [](Probe& p, const auto& expect) {
            const std::int64_t before = static_cast<std::int64_t>(p.clock_ms());
            const auto r = p.generator().v7();
            const std::int64_t after = static_cast<std::int64_t>(p.clock_ms());
            expect(r.is_ok(), "v7 succeeds");
            if (!r)
                return;
            expect(r->version() == 7 && r->variant() == uuid_variant::rfc9562, "version 7 and the RFC 9562 variant");
            const std::int64_t ms = detail::uuid_ms(*r);
            expect(ms >= before && ms <= after, "the first v7 carries the clock's millisecond");
        });

        detail::capability_check<Probe>(report, "v7 strictly increases within a millisecond", [](Probe& p, const auto& expect) {
            auto last = p.generator().v7();
            expect(last.is_ok(), "v7 succeeds");
            bool increasing = last.is_ok();
            for (int i = 0; i < 5000 && increasing; ++i) {   // more than a 12-bit counter holds
                auto next = p.generator().v7();
                increasing = next.is_ok() && *last < *next && next->version() == 7;
                last = next;
            }
            expect(increasing, "5000 in a row strictly increase");
        });

        detail::capability_check<Probe>(report, "v7 never goes back with the clock", [](Probe& p, const auto& expect) {
            const auto first = p.generator().v7();
            p.clock_steps_back();
            const auto second = p.generator().v7();
            expect(first.is_ok() && second.is_ok(), "v7 succeeds");
            if (first.is_ok() && second.is_ok())
                expect(*first < *second, "after the clock steps back, v7 still increases");
        });

        detail::capability_check<Probe>(report, "a failed random source is unavailable", [](Probe& p, const auto& expect) {
            p.randomness_fails();
            const auto a = p.generator().v4();
            const auto b = p.generator().v7();
            expect(a.is_err() && a.unwrap_err().code == capability_errc::unavailable, "v4 is unavailable");
            expect(b.is_err() && b.unwrap_err().code == capability_errc::unavailable, "v7 is unavailable");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_uuid_generator_conformance_hpp
