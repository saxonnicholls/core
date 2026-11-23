//
//  key_store_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_key_store_conformance`: the conformance kit of the `key_store` contract.
/// \ingroup core
///
/// **Overview.** Drives any `key_store`, in either form, through the contract of `concepts/key_store.hpp` and reports
/// which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type with
/// - `auto& store()`: the implementation (a store, a `key_store_interface&`, a `key_store_ref`);
/// - `key_spec supported()`: a spec the store can create;
/// - `std::optional<key_spec> unsupported()`: a spec it cannot, or `std::nullopt` when it holds every algorithm;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `create` fails with `code`, as
///   though the KMS had answered `response`.
///
/// The checks: `create makes version one active`, `rotate retires the previous version and list shows both`, `a
/// retired key cannot rotate`, `destroy removes every version and the id is never reused`, `an unknown id is not
/// found everywhere`, `an algorithm it cannot hold is unsupported`, `a label too long is invalid`, `a provider failure
/// never leaks its response`.
///
/// \snippet interfaces/key_store_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/key_store.hpp`, `interfaces/key_store.hpp`, `interfaces/capability_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/key_store_conformance_test.cpp`,
/// `demos/interfaces/key_store_conformance_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<optional>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `concepts/key_store.hpp`, `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_key_store_conformance_hpp
#define snicholls_core_interfaces_key_store_conformance_hpp

#include <snicholls/core/concepts/key_store.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept key_store_probe = std::is_default_constructible_v<Probe> &&
            key_store<std::remove_reference_t<decltype(std::declval<Probe&>().store())>> &&
            requires(Probe& p, capability_errc c, std::string_view r) {
                { p.supported() } -> std::same_as<key_spec>;
                { p.unsupported() } -> std::same_as<std::optional<key_spec>>;
                p.provider_fails(c, r);
            };
    } // namespace detail

    /// \brief Runs every check of the `key_store` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::key_store_probe<Probe>
    [[nodiscard]] capability_conformance_report run_key_store_conformance()
    {
        capability_conformance_report report;

        detail::capability_check<Probe>(report, "create makes version one active", [](Probe& p, const auto& expect) {
            auto& k = p.store();
            const key_spec spec = p.supported();
            const auto made = k.create(spec);
            expect(made.is_ok(), "a supported spec is created");
            if (!made.is_ok())
                return;
            expect(made->version == 1 && made->state == key_state::active, "version 1, active");
            expect(made->algorithm == spec.algorithm && made->label == spec.label, "the algorithm and label are kept");
            const auto info = k.info(made->key_id.view());
            expect(info.is_ok() && *info == *made, "info returns the current version");
        });

        detail::capability_check<Probe>(report, "rotate retires the previous version and list shows both", [](Probe& p, const auto& expect) {
            auto& k = p.store();
            const auto made = k.create(p.supported());
            expect(made.is_ok(), "a supported spec is created");
            if (!made.is_ok())
                return;
            const auto next = k.rotate(made->key_id.view());
            expect(next.is_ok() && next->version == 2 && next->state == key_state::active, "rotate makes version 2 active");
            const auto all = k.list();
            expect(all.is_ok(), "list answers");
            if (!all.is_ok())
                return;
            std::size_t mine = 0;
            for (const key_info& i : *all)
                if (i.key_id == made->key_id) {
                    ++mine;
                    if (i.version == 1)
                        expect(i.state == key_state::retired, "version 1 is retired");
                    expect(mine == i.version, "versions are listed in order");
                }
            expect(mine == 2, "list shows both versions");
        });

        detail::capability_check<Probe>(report, "a retired key cannot rotate", [](Probe& p, const auto& expect) {
            auto& k = p.store();
            const auto made = k.create(p.supported());
            expect(made.is_ok(), "a supported spec is created");
            if (!made.is_ok())
                return;
            expect(k.retire(made->key_id.view()).is_ok(), "retire succeeds");
            const auto info = k.info(made->key_id.view());
            expect(info.is_ok() && info->state == key_state::retired, "the key is retired");
            const auto r = k.rotate(made->key_id.view());
            expect(r.is_err() && r.unwrap_err().code == capability_errc::conflict, "rotating a retired key is a conflict");
        });

        detail::capability_check<Probe>(report, "destroy removes every version and the id is never reused", [](Probe& p, const auto& expect) {
            auto& k = p.store();
            const auto made = k.create(p.supported());
            expect(made.is_ok(), "a supported spec is created");
            if (!made.is_ok())
                return;
            expect(k.destroy(made->key_id.view()).is_ok(), "destroy succeeds");
            const auto info = k.info(made->key_id.view());
            expect(info.is_err() && info.unwrap_err().code == capability_errc::not_found, "a destroyed key is not found");
            const auto all = k.list();
            if (all.is_ok())
                for (const key_info& i : *all)
                    expect(i.key_id != made->key_id, "list no longer shows it");
            const auto again = k.create(p.supported());
            expect(again.is_ok() && again->key_id != made->key_id, "a new key never reuses the id");
        });

        detail::capability_check<Probe>(report, "an unknown id is not found everywhere", [](Probe& p, const auto& expect) {
            auto& k = p.store();
            const auto is_not_found = [](const auto& r) { return r.is_err() && r.unwrap_err().code == capability_errc::not_found; };
            expect(is_not_found(k.info("no-such-key")), "info");
            expect(is_not_found(k.rotate("no-such-key")), "rotate");
            expect(is_not_found(k.retire("no-such-key")), "retire");
            expect(is_not_found(k.destroy("no-such-key")), "destroy");
        });

        detail::capability_check<Probe>(report, "an algorithm it cannot hold is unsupported", [](Probe& p, const auto& expect) {
            const std::optional<key_spec> spec = p.unsupported();
            if (!spec)
                return;
            const auto r = p.store().create(*spec);
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unsupported, "create is unsupported");
        });

        detail::capability_check<Probe>(report, "a label too long is invalid", [](Probe& p, const auto& expect) {
            const std::string long_label(short_text::capacity() + 1, 'l');
            key_spec spec = p.supported();
            spec.label = long_label;
            const auto r = p.store().create(spec);
            expect(r.is_err() && r.unwrap_err().code == capability_errc::invalid_argument, "a 256-character label is invalid_argument");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [](Probe& p, const auto& expect) {
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto r = p.store().create(p.supported());
            expect(r.is_err() && r.unwrap_err().code == capability_errc::unavailable, "the provider's failure is returned");
            if (r.is_err())
                expect(!capability_error_leaks(r.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_key_store_conformance_hpp
