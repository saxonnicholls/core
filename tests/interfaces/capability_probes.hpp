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
#include <snicholls/core/interfaces/https_client_conformance.hpp>
#include <snicholls/core/interfaces/https_client_reference.hpp>
#include <snicholls/core/interfaces/key_store_reference.hpp>
#include <snicholls/core/interfaces/message_sender_reference.hpp>
#include <snicholls/core/interfaces/onramp_provider_reference.hpp>
#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/interfaces/remote_files_reference.hpp>
#include <snicholls/core/interfaces/role_store_reference.hpp>
#include <snicholls/core/interfaces/signature_verifier_conformance.hpp>
#include <snicholls/core/interfaces/signature_verifier_reference.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>
#include <snicholls/core/interfaces/token_verifier_reference.hpp>
#include <snicholls/core/interfaces/wallet_verifier_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
    struct remote_files_probe {
        forms<memory_remote_files, remote_files_interface, remote_files_adapter, remote_files_ref, F> f{memory_remote_files{true, 1000}};
        auto& files() { return f.get(); }
        void seed(std::string_view name, std::span<const std::byte> bytes, utc_s modified) { static_cast<void>(f.impl.put(name, bytes, modified)); }
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

    inline constexpr uuid probe_party = uuid::from_u64(0x0190'0000'0000'7000ULL, 0x8000'0000'0000'0001ULL);

    struct issued_tokens {
        memory_token_verifier verifier = memory_token_verifier::deterministic(probe_time);
        std::string valid;
        std::string expired;
        issued_tokens()
        {
            token_claims c;
            c.subject = probe_party;
            c.expires_at = probe_time + std::chrono::hours{1};
            valid = verifier.issue(c);
            c.expires_at = probe_time;
            expired = verifier.issue(c);
        }
    };

    template <form F>
    struct token_verifier_probe {
        issued_tokens tokens;
        forms<memory_token_verifier, token_verifier_interface, token_verifier_adapter, token_verifier_ref, F> f{std::move(tokens.verifier)};
        auto& verifier() { return f.get(); }
        std::string valid_token() { return tokens.valid; }
        std::string expired_token() { return tokens.expired; }
        static uuid subject() { return probe_party; }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    struct hs256_keys {
        memory_key_store store = memory_key_store::deterministic(probe_time);
        std::string id = std::string(store.create({signature_algorithm::hs256, "jwt"})->key_id.view());

        signed_sample sign(std::span<const std::byte> message)
        {
            const auto s = store.sign({id, message, payload_kind::message});
            return {signature_algorithm::hs256, id, s->key_version, {}, s->bytes};
        }
    };

    template <form F>
    struct signature_verifier_probe {
        hs256_keys keys;
        forms<memory_signature_verifier, signature_verifier_interface, signature_verifier_adapter, signature_verifier_ref, F> f{
            memory_signature_verifier{keys.store}};
        auto& verifier() { return f.get(); }
        signed_sample sign(std::span<const std::byte> message) { return keys.sign(message); }
        static signature_algorithm unsupported() { return signature_algorithm::eddsa; }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    inline constexpr std::string_view probe_wallet = "0x52908400098527886e0f7030069857d2e4169ee7";
    inline constexpr std::string_view probe_other_wallet = "0x8617e340b3d01fa5f11f306f4090fd50e238070d";

    inline memory_wallet_verifier make_wallet_verifier()
    {
        memory_wallet_verifier v;
        static_cast<void>(v.add_wallet(probe_wallet, "test-only wallet key 1"));
        static_cast<void>(v.add_wallet(probe_other_wallet, "test-only wallet key 2"));
        return v;
    }

    template <form F>
    struct wallet_verifier_probe {
        forms<memory_wallet_verifier, wallet_verifier_interface, wallet_verifier_adapter, wallet_verifier_ref, F> f{make_wallet_verifier()};
        auto& verifier() { return f.get(); }
        static std::string signer_address() { return std::string(probe_wallet); }
        static std::string other_address() { return std::string(probe_other_wallet); }
        std::vector<std::byte> sign(std::string_view message) { return f.impl.sign(probe_wallet, message); }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    template <form F>
    struct role_store_probe {
        forms<memory_role_store, role_store_interface, role_store_adapter, role_store_ref, F> f{memory_role_store{}};
        auto& store() { return f.get(); }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

    // A server in memory: answers what it is told and remembers what it got, as a socket server would see it.
    struct scripted_server {
        https_response next{200, {}, ""};
        https_observed_request last;

        capability_result<https_response> operator()(const https_request& r)
        {
            last.method = std::string(to_string(r.method));
            const auto target = parse_https_url(r.url)->target;
            last.target = target.empty() ? "/" : std::string(target);
            last.headers = r.headers;
            if (!r.authorization.reveal().empty())
                last.headers.push_back({"Authorization", r.authorization.reveal()});
            if (!r.idempotency_key.empty())
                last.headers.push_back({"Idempotency-Key", std::string(r.idempotency_key.view())});
            last.body = r.body;
            return ok(next);
        }
    };

    inline memory_https_client routed_to(const std::shared_ptr<scripted_server>& server)
    {
        memory_https_client c;
        c.route("api.example.test", [server](const https_request& r) { return (*server)(r); });
        return c;
    }

    template <form F>
    struct https_client_probe {
        std::shared_ptr<scripted_server> server = std::make_shared<scripted_server>();
        forms<memory_https_client, https_client_interface, https_client_adapter, https_client_ref, F> f{routed_to(server)};
        auto& client() { return f.get(); }
        static std::string url(std::string_view path) { return "https://api.example.test" + std::string(path); }
        static std::string unreachable_url() { return "https://unreachable.example.test/"; }
        void answer(https_response r) { server->next = std::move(r); }
        https_observed_request seen() { return server->last; }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

} // namespace snicholls::test

#endif // snicholls_core_tests_interfaces_capability_probes_hpp
