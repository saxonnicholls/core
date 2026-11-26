//
//  https_client_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_https_client_conformance`: the conformance kit of the `https_client` contract.
/// \ingroup core
///
/// **Overview.** Drives any `https_client`, in either form, through the contract of `concepts/https_client.hpp` and
/// reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type
/// with
/// - `auto& client()`: the implementation (a client, an `https_client_interface&`, an `https_client_ref`);
/// - `std::string url(std::string_view path_and_query)`: an `https://` URL the probe's server answers;
/// - `std::string unreachable_url()`: an `https://` URL nothing answers;
/// - `void answer(https_response r)`: what the server answers next;
/// - `https_observed_request seen()`: what the server last received, as it arrived;
/// - `void provider_fails(capability_errc code, std::string_view response)`: the next `send` fails with `code`, as
///   though the TLS layer had answered `response`.
///
/// The checks: `a GET returns status, headers and body`, `an error status is a response, not an error`, `a POST
/// carries its body, idempotency key and authorization`, `only https URLs and clean headers are sent`, `an unreachable
/// host is unavailable`, `the credential never reaches an error`, `a provider failure never leaks its response`.
///
/// \snippet interfaces/https_client_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/https_client.hpp`, `interfaces/https_client.hpp`, `interfaces/capability_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations / variants.** One kit for every form. A TLS client's probe runs a local server.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/https_client_conformance_test.cpp`,
/// `demos/interfaces/https_client_conformance_demo.cpp`.
///
/// **Dependencies.** `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `<vector>`, `concepts/https_client.hpp`,
/// `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** Timeouts are not exercised: a kit that waits would slow every test run.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_https_client_conformance_hpp
#define snicholls_core_interfaces_https_client_conformance_hpp

#include <snicholls/core/concepts/https_client.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief What a probe's server received, as it arrived.
    /// \ingroup core
    struct https_observed_request {
        std::string method;                ///< `"GET"`, `"POST"`, ...
        std::string target;                ///< the path and query
        std::vector<http_header> headers;  ///< as received, the credential and the idempotency key among them
        std::string body;                  ///< as received
    };

    namespace detail {
        template <typename Probe>
        concept https_client_probe = std::is_default_constructible_v<Probe> &&
            https_client<std::remove_reference_t<decltype(std::declval<Probe&>().client())>> &&
            requires(Probe& p, std::string_view path, https_response a, capability_errc c, std::string_view r) {
                { p.url(path) } -> std::convertible_to<std::string>;
                { p.unreachable_url() } -> std::convertible_to<std::string>;
                p.answer(a);
                { p.seen() } -> std::convertible_to<https_observed_request>;
                p.provider_fails(c, r);
            };

        inline std::string_view header_or_empty(const std::vector<http_header>& h, std::string_view name) noexcept
        {
            return find_header(h, name).value_or(std::string_view{});
        }
    } // namespace detail

    /// \brief Runs every check of the `https_client` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::https_client_probe<Probe>
    [[nodiscard]] capability_conformance_report run_https_client_conformance()
    {
        capability_conformance_report report;
        using detail::header_or_empty;

        detail::capability_check<Probe>(report, "a GET returns status, headers and body", [](Probe& p, const auto& expect) {
            p.answer({200, {{"Content-Type", "application/json"}, {"Request-Id", "req_1"}}, R"({"ok":true})"});
            https_request r;
            r.url = p.url("/v1/thing?expand[]=x");
            const auto got = p.client().send(r);
            expect(got.is_ok(), "the request is answered");
            if (!got.is_ok())
                return;
            expect(got->status == 200 && got->ok() && got->body == R"({"ok":true})", "the status and the body");
            expect(header_or_empty(got->headers, "request-id") == "req_1", "the headers, found ignoring case");
            const https_observed_request seen = p.seen();
            expect(seen.method == "GET" && seen.target == "/v1/thing?expand[]=x", "the server saw the method and the target");
        });

        detail::capability_check<Probe>(report, "an error status is a response, not an error", [](Probe& p, const auto& expect) {
            p.answer({503, {{"Retry-After", "2"}}, "busy"});
            https_request r;
            r.url = p.url("/busy");
            const auto got = p.client().send(r);
            expect(got.is_ok() && got->status == 503 && !got->ok(), "a 503 is a response");
            if (got.is_ok())
                expect(header_or_empty(got->headers, "Retry-After") == "2", "with its headers");
        });

        detail::capability_check<Probe>(report, "a POST carries its body, idempotency key and authorization", [](Probe& p, const auto& expect) {
            p.answer({200, {}, "{}"});
            https_request r;
            r.method = http_method::post;
            r.url = p.url("/v1/refunds");
            r.headers = {{"Content-Type", "application/x-www-form-urlencoded"}};
            r.authorization = secret<std::string>{"Bearer kit_test_only"};
            r.body = "payment_intent=pi_1&amount=100";
            r.idempotency_key = "kit-key-1";
            expect(p.client().send(r).is_ok(), "the request is answered");
            const https_observed_request seen = p.seen();
            expect(seen.method == "POST" && seen.body == r.body, "the method and the body arrive");
            expect(header_or_empty(seen.headers, "Idempotency-Key") == "kit-key-1", "the idempotency key arrives as a header");
            expect(header_or_empty(seen.headers, "Authorization") == "Bearer kit_test_only", "the credential arrives as Authorization");
            expect(header_or_empty(seen.headers, "Content-Type") == "application/x-www-form-urlencoded", "the headers arrive");
        });

        detail::capability_check<Probe>(report, "only https URLs and clean headers are sent", [](Probe& p, const auto& expect) {
            const auto refused = [&](https_request r) {
                const auto got = p.client().send(r);
                return got.is_err() && got.unwrap_err().code == capability_errc::invalid_argument;
            };
            https_request plain;
            plain.url = "http://example.test/";
            expect(refused(plain), "http:// is invalid_argument");
            https_request other;
            other.url = "ftp://example.test/";
            expect(refused(other), "ftp:// is invalid_argument");
            https_request user;
            user.url = "https://user:pass@example.test/";
            expect(refused(user), "user information in the URL is invalid_argument");
            https_request split;
            split.url = p.url("/x");
            split.headers = {{"X-Evil", "a\r\nInjected: yes"}};
            expect(refused(split), "a value with CR LF is invalid_argument");
            https_request bad_name;
            bad_name.url = p.url("/x");
            bad_name.headers = {{"Bad:Name", "v"}};
            expect(refused(bad_name), "a name with a colon is invalid_argument");
            https_request smuggled;
            smuggled.url = p.url("/x");
            smuggled.authorization = secret<std::string>{"Bearer a\r\nX-Injected: 1"};
            expect(refused(smuggled), "a credential with CR LF is invalid_argument");
        });

        detail::capability_check<Probe>(report, "an unreachable host is unavailable", [](Probe& p, const auto& expect) {
            https_request r;
            r.url = p.unreachable_url();
            const auto got = p.client().send(r);
            expect(got.is_err() && (got.unwrap_err().code == capability_errc::unavailable ||
                                    got.unwrap_err().code == capability_errc::timeout),
                   "unavailable (or timeout)");
        });

        detail::capability_check<Probe>(report, "the credential never reaches an error", [](Probe& p, const auto& expect) {
            https_request r;
            r.url = p.unreachable_url();
            r.authorization = secret<std::string>{"Bearer " + std::string(capability_redaction_marker)};
            r.body = std::string(capability_redaction_marker);
            r.sensitive_body = true;
            const auto got = p.client().send(r);
            expect(got.is_err(), "the unreachable host fails");
            if (got.is_err())
                expect(!capability_error_leaks(got.unwrap_err(), capability_redaction_marker),
                       "neither the credential nor the sensitive body reaches the error");
        });

        detail::capability_check<Probe>(report, "a provider failure never leaks its response", [](Probe& p, const auto& expect) {
            https_request r;
            r.url = p.url("/v1/thing");
            p.provider_fails(capability_errc::unavailable, capability_provider_response);
            const auto got = p.client().send(r);
            expect(got.is_err() && got.unwrap_err().code == capability_errc::unavailable, "the provider's failure is returned");
            if (got.is_err())
                expect(!capability_error_leaks(got.unwrap_err(), capability_redaction_marker),
                       "neither provider_code nor message carries the provider's response");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_https_client_conformance_hpp
