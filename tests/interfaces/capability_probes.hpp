//
//  capability_probes.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The probes the capability kits run the reference implementations through, in every form: the reference itself
//  (the concept), the reference behind the virtual interface (a borrowing adapter), and a handle on that interface.
//  Shared by tests/interfaces/<capability>_test.cpp and <capability>_conformance_test.cpp.
//

#ifndef snicholls_core_tests_interfaces_capability_probes_hpp
#define snicholls_core_tests_interfaces_capability_probes_hpp

#include <snicholls/core/interfaces/capability_conformance.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>
#include <snicholls/core/interfaces/key_store_reference.hpp>
#include <snicholls/core/interfaces/message_sender_reference.hpp>
#include <snicholls/core/interfaces/onramp_provider_reference.hpp>
#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace snicholls::test {

    /// Which form a probe hands the kit.
    enum class form { concept_model, interface, ref };

    /// An implementation in its three forms; not copyable, because the adapter and the handle point into it.
    template <typename Impl, typename Interface, template <typename> class Adapter, typename Ref, form F>
    struct forms {
        Impl impl;
        Adapter<Impl&> adapter{impl};
        Ref handle{adapter};

        explicit forms(Impl i) : impl(std::move(i)) {}
        forms(const forms&) = delete;
        forms& operator=(const forms&) = delete;

        auto& get()
        {
            if constexpr (F == form::concept_model)
                return impl;
            else if constexpr (F == form::interface)
                return static_cast<Interface&>(adapter);
            else
                return handle;
        }
    };

    inline std::string describe(const capability_conformance_report& report)
    {
        std::string out = std::to_string(report.checks) + " checks";
        for (const auto& f : report.failures)
            out += "\n  " + std::string(f.check) + ": " + std::string(f.expectation);
        return out;
    }

    inline const utc_s probe_time = utc_s::from_unix_seconds(1'800'000'000);
    inline constexpr asset aud = checked_asset("AUD", 2);
    inline constexpr asset usdc = checked_asset("USDC", 6);

    template <form F>
    struct message_sender_probe {
        forms<memory_message_sender, message_sender_interface, message_sender_adapter, message_sender_ref, F> f{memory_message_sender{}};
        auto& sender() { return f.get(); }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    template <form F>
    struct key_store_probe {
        forms<memory_key_store, key_store_interface, key_store_adapter, key_store_ref, F> f{memory_key_store::deterministic()};
        auto& store() { return f.get(); }
        static key_spec supported() { return {signature_algorithm::hs256, "jwt-test"}; }
        static std::optional<key_spec> unsupported() { return key_spec{signature_algorithm::eddsa, "jwt-test"}; }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    template <form F>
    struct signer_probe {
        forms<memory_signer, signer_interface, signer_adapter, signer_ref, F> f{memory_signer::deterministic()};
        auto& signer() { return f.get(); }
        std::string active_key() { return std::string(f.impl.create({signature_algorithm::hs256, "jwt"})->key_id.view()); }
        std::string retired_key()
        {
            std::string id = active_key();
            static_cast<void>(f.impl.retire(id));
            return id;
        }
        bool verify(const signature& s, std::span<const std::byte> payload) { return f.impl.verify(s, payload); }
        static bool deterministic() { return true; }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    inline memory_payment_provider make_payment_provider()
    {
        return memory_payment_provider{{aud}, secret<std::string>{"whsec_test_only"}, [] { return probe_time; }};
    }

    template <form F>
    struct payment_provider_probe {
        forms<memory_payment_provider, payment_provider_interface, payment_provider_adapter, payment_provider_ref, F> f{
            make_payment_provider()};
        auto& provider() { return f.get(); }
        static money amount() { return *parse_money(aud, "125.50"); }
        static asset unsupported_currency() { return checked_asset("XYZ", 2); }
        void payer_pays(const collection& c)
        {
            if (c.next.kind == next_step_kind::bank_transfer)
                static_cast<void>(f.impl.receive_credit(c.pay.amount, c.next.bank->reference.view(), "A Payer"));
            else
                static_cast<void>(f.impl.complete(c.pay.payment_id.view()));
        }
        owned_delivery deliver(const webhook_event& e) { return f.impl.signed_webhook(e); }
        owned_delivery deliver_late(const webhook_event& e) { return f.impl.signed_webhook(e, probe_time - std::chrono::minutes{6}); }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    inline memory_onramp_provider make_onramp_provider()
    {
        memory_onramp_provider v{{onramp_asset{usdc, {8453}, true, "USD"}}, [] { return probe_time; }};
        v.set_price(*rate::parse(usdc, aud, "1.53"));
        v.deposit(*parse_money(aud, "10000"));
        return v;
    }

    template <form F>
    struct onramp_provider_probe {
        forms<memory_onramp_provider, onramp_provider_interface, onramp_provider_adapter, onramp_provider_ref, F> f{
            make_onramp_provider()};
        auto& venue() { return f.get(); }
        static asset unit() { return usdc; }
        static asset quote_currency() { return aud; }
        static money ticket() { return *parse_money(aud, "153"); }
        static money too_much() { return *parse_money(aud, "1000000"); }
        static std::uint64_t chain() { return 8453; }
        static std::string address() { return "0x52908400098527886E0F7030069857D2E4169EE7"; }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    inline route_table make_routes()
    {
        route_table t;
        static_cast<void>(t.add(http_method::get, "/health", [](const http_request&) { return respond(200, "ok"); }));
        static_cast<void>(t.add(http_method::get, "/boom", [](const http_request&) -> http_response {
            throw std::runtime_error("db password=" + std::string(capability_redaction_marker));
        }));
        return t;
    }

    inline http_request request(http_method m, std::string path)
    {
        http_request r;
        r.method = m;
        r.path = std::move(path);
        return r;
    }

    template <form F>
    struct http_handler_probe {
        forms<route_table, http_handler_interface, http_handler_adapter, http_handler_ref, F> f{make_routes()};
        auto& handler() { return f.get(); }
        static http_request known() { return request(http_method::get, "/health"); }
        static http_request unknown() { return request(http_method::get, "/nowhere"); }
        static http_request throwing() { return request(http_method::get, "/boom"); }
    };

} // namespace snicholls::test

#endif // snicholls_core_tests_interfaces_capability_probes_hpp
