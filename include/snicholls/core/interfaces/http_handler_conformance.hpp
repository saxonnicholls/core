//
//  http_handler_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::run_http_handler_conformance`: the conformance kit of the `http_handler` contract.
/// \ingroup core
///
/// **Overview.** Drives any `http_handler`, in either form, through the contract of `concepts/http_handler.hpp` and
/// reports which check failed (`interfaces/capability_conformance.hpp`). The probe is a default-constructible type with
/// - `auto& handler()`: the implementation (a handler, an `http_handler_interface&`, an `http_handler_ref`);
/// - `http_request known()`: a `GET` it answers 2xx with a body;
/// - `http_request unknown()`: a request for a path it does not serve;
/// - `http_request throwing()`: a request whose handler throws an exception carrying
///   `capability_redaction_marker` in its message.
///
/// The checks: `a known route answers 2xx`, `an unknown path is 404`, `HEAD is GET without a body`, `any request gets
/// a status in range`, `a throwing handler answers 500 without detail`.
///
/// \snippet interfaces/http_handler_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/http_handler.hpp`, `interfaces/http_handler.hpp`,
/// `interfaces/capability_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** As `interfaces/capability_conformance.hpp`: no exception escapes a check.
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/http_handler_conformance_test.cpp`,
/// `demos/interfaces/http_handler_conformance_demo.cpp`.
///
/// **Dependencies.** `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `concepts/http_handler.hpp`,
/// `interfaces/capability_conformance.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_interfaces_http_handler_conformance_hpp
#define snicholls_core_interfaces_http_handler_conformance_hpp

#include <snicholls/core/concepts/http_handler.hpp>
#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename Probe>
        concept http_handler_probe = std::is_default_constructible_v<Probe> &&
            http_handler<std::remove_reference_t<decltype(std::declval<Probe&>().handler())>> && requires(Probe& p) {
                { p.known() } -> std::same_as<http_request>;
                { p.unknown() } -> std::same_as<http_request>;
                { p.throwing() } -> std::same_as<http_request>;
            };
    } // namespace detail

    /// \brief Runs every check of the `http_handler` contract, each on a fresh `Probe`.
    /// \ingroup core
    /// \tparam Probe The probe (see the file's overview).
    /// \return The report.
    template <typename Probe>
        requires detail::http_handler_probe<Probe>
    [[nodiscard]] capability_conformance_report run_http_handler_conformance()
    {
        capability_conformance_report report;

        detail::capability_check<Probe>(report, "a known route answers 2xx", [](Probe& p, const auto& expect) {
            const http_response r = p.handler().handle(p.known());
            expect(r.status >= 200 && r.status < 300, "the status is 2xx");
            expect(!r.body.empty(), "there is a body");
        });

        detail::capability_check<Probe>(report, "an unknown path is 404", [](Probe& p, const auto& expect) {
            expect(p.handler().handle(p.unknown()).status == 404, "the status is 404");
        });

        detail::capability_check<Probe>(report, "HEAD is GET without a body", [](Probe& p, const auto& expect) {
            http_request head = p.known();
            head.method = http_method::head;
            const http_response got = p.handler().handle(p.known());
            const http_response r = p.handler().handle(head);
            expect(r.status == got.status, "HEAD answers GET's status");
            expect(r.body.empty(), "HEAD has no body");
        });

        detail::capability_check<Probe>(report, "any request gets a status in range", [](Probe& p, const auto& expect) {
            http_request odd;
            odd.method = http_method::patch;
            odd.path = "";
            odd.body = std::string("\0\xff", 2);
            const int status = p.handler().handle(odd).status;
            expect(status >= 100 && status <= 599, "the status is from 100 to 599");
        });

        detail::capability_check<Probe>(report, "a throwing handler answers 500 without detail", [](Probe& p, const auto& expect) {
            const http_response r = p.handler().handle(p.throwing());
            expect(r.status == 500, "the status is 500");
            expect(r.body.find(capability_redaction_marker) == std::string::npos, "the body carries nothing of the exception");
            for (const http_header& h : r.headers)
                expect(h.value.find(capability_redaction_marker) == std::string::npos, "no header carries the exception");
        });

        return report;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_http_handler_conformance_hpp
