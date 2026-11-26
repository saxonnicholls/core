//
//  http_client_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_http_client_conformance`: the conformance kit of the `http_client` contract.
/// \ingroup core
///
/// **Overview.** Drives any `http_client`, in either form, through the contract of `concepts/http_client.hpp` and
/// reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type
/// with
/// - `auto& client()`: the implementation (a client, an `http_client_interface&`, an `http_client_ref`), talking to a
///   server that answers `GET /echo` with 200 and the request's body and `X-Echo` header echoed back, and every
///   unknown path with 404;
/// - `void transport_fails(capability_errc code, std::string_view response)`: the next send gets no response.
///
/// A fresh probe is made for every check. The checks: `a request reaches the server and its response returns`, `an
/// error status is a response`, `HEAD returns no body`, `a malformed request is refused`, `a transport failure leaks
/// nothing`.
///
/// \snippet interfaces/http_client_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/http_client.hpp`, `interfaces/http_client.hpp`, `interfaces/capability_conformance.hpp`;
/// run on `loopback_http_client` in every form by `tests/interfaces/http_client_test.cpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/http_client_conformance_test.cpp` (the kit catches broken clients),
/// `demos/interfaces/http_client_conformance_demo.cpp`.
///
/// **Dependencies.** `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `concepts/http_client.hpp`,
/// `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_http_client_conformance_hpp
#define snicholls_core_interfaces_http_client_conformance_hpp

#include <snicholls/core/concepts/http_client.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept http_client_probe = std::is_default_constructible_v<Probe> &&
            http_client<std::remove_reference_t<decltype(std::declval<Probe&>().client())>> &&
            requires(Probe& p, capability_errc c, std::string_view r) { p.transport_fails(c, r); };

        inline http_request kit_echo(http_method m = http_method::get)
        {
            http_request r;
            r.method = m;
            r.path = "/echo";
            r.headers.push_back({"X-Echo", "header value"});
            r.body = std::string("binary\0body\xff", 12);
            return r;
        }
    } // namespace detail

    /// \brief Runs every check of the `http_client` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::http_client_probe<Probe>
    [[nodiscard]] capability_conformance_report run_http_client_conformance()
    {
        capability_conformance_report report;

        detail::capability_check<Probe>(report, "a request reaches the server and its response returns", [](Probe& p, const auto& expect) {
            const http_request r = detail::kit_echo();
            const auto got = p.client().send(r);
            expect(got.is_ok(), "the request is sent");
            if (!got)
                return;
            expect(got->status == 200, "the server's status");
            expect(got->body == r.body, "the body, byte for byte");
            expect(find_header(got->headers, "x-echo") == std::optional<std::string_view>{"header value"}, "the header, either way");
        });

        detail::capability_check<Probe>(report, "an error status is a response", [](Probe& p, const auto& expect) {
            const auto got = http_get(p.client(), "/no/such/path?x=1");
            expect(got.is_ok() && got->status == 404, "a 404 is a response, not an error");
        });

        detail::capability_check<Probe>(report, "HEAD returns no body", [](Probe& p, const auto& expect) {
            const auto got = p.client().send(detail::kit_echo(http_method::head));
            expect(got.is_ok() && got->body.empty(), "HEAD has no body");
        });

        detail::capability_check<Probe>(report, "a malformed request is refused", [](Probe& p, const auto& expect) {
            http_request no_path = detail::kit_echo();
            no_path.path = "echo";
            http_request split = detail::kit_echo();
            split.headers.push_back({"X-Bad", "a\r\nInjected: yes"});
            for (const http_request* r : {&no_path, &split}) {
                const auto got = p.client().send(*r);
                expect(got.is_err() && got.unwrap_err().code == capability_errc::invalid_argument, "invalid_argument");
            }
        });

        detail::capability_check<Probe>(report, "a transport failure leaks nothing", [](Probe& p, const auto& expect) {
            p.transport_fails(capability_errc::unavailable, capability_provider_response);
            const auto got = p.client().send(detail::kit_echo());
            expect(got.is_err() && got.unwrap_err().code == capability_errc::unavailable, "no response is an error");
            if (got.is_err())
                expect(!capability_error_leaks(got.unwrap_err(), capability_redaction_marker), "nothing received is kept");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_http_client_conformance_hpp
