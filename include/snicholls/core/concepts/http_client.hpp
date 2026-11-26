//
//  http_client.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The contract of crtp's http_client_interface (interfaces/http_interface.hpp: send(request) -> response, net_error
//  when there is no response), over core's http_request and http_response, with "no response" a capability_error
//  instead of an exception. The transports (cpp-httplib, ts-moveables' client) stay above core.
//

/// \file
/// \brief `snicholls::http_client`: anything that sends an HTTP request and returns the response.
/// \ingroup core
///
/// **Overview.** The client half of HTTP, beside `http_handler` (`concepts/http_handler.hpp`): `c.send(request)`
/// returns the server's `http_response` (any status, a 404 or a 503 included, is a response) or a `capability_error`
/// when there is no response at all: `unavailable` (DNS, connect, TLS, a reset), `timeout`, or `invalid_argument` for
/// a request that cannot be sent (no path, a header with a line break). Code that calls a provider's API takes an
/// `http_client` and is tested against a handler through `loopback_http_client` (`interfaces/http_client_reference.hpp`)
/// with no network at all. `http_get(c, target)` builds a GET from `path?query`.
///
/// \snippet concepts/http_client_demo.cpp concept
///
/// **The contract** (`interfaces/http_client_conformance.hpp` checks it): a request reaches the server and its
/// response comes back as it is; an error status is a response, never an error; no response is an error that carries
/// nothing of what the transport received (`capability_error.hpp`'s redaction rule); a malformed request is refused
/// before anything is sent; `HEAD` comes back without a body.
///
/// **Relationships.** `concepts/http_handler.hpp` (the values), `capability_error.hpp`; the twin
/// `interfaces/http_client.hpp`, the reference `interfaces/http_client_reference.hpp`, the kit
/// `interfaces/http_client_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `loopback_http_client<H>` (over any handler, in memory). Above core: crtp's cpp-httplib client
/// and ts-moveables' client.
///
/// **Guarantees.** `validate_request` is `noexcept` and allocation-free. Thread safety is the implementation's.
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Not benchmarked: a client is I/O bound.
///
/// **Tests and demos.** `tests/concepts/http_client_test.cpp`, `demos/concepts/http_client_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<string>`, `<string_view>`, `<utility>`, `<vector>`, `capability_error.hpp`,
/// `concepts/http_handler.hpp`.
///
/// **Known issues.** Streaming bodies and WebSocket clients are not in the contract (crtp's
/// `websocket_client_interface` stays crtp's).
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_http_client_hpp
#define snicholls_core_concepts_http_client_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/concepts/http_handler.hpp>

#include <concepts>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The checks every client applies before sending.
    /// \ingroup core
    /// \param r The request.
    /// \return Success, or `invalid_argument` naming what is wrong (never its value).
    [[nodiscard]] inline capability_result<void> validate_request(const http_request& r) noexcept
    {
        if (r.path.empty() || r.path.front() != '/')
            return fail(capability_errc::invalid_argument, {}, "a path starts with /");
        const auto breaks = [](std::string_view s) { return s.find_first_of("\r\n") != std::string_view::npos; };
        if (breaks(r.path) || breaks(r.query))
            return fail(capability_errc::invalid_argument, {}, "a line break in the target");
        for (const http_header& h : r.headers)
            if (h.name.empty() || breaks(h.name) || breaks(h.value) || h.name.find(':') != std::string::npos)
                return fail(capability_errc::invalid_argument, {}, "a header that cannot be sent");
        return ok();
    }

    /// \brief An HTTP client: `c.send(request)`.
    /// \ingroup core
    template <typename C>
    concept http_client = requires(C& c, const http_request& r) {
        { c.send(r) } -> std::same_as<capability_result<http_response>>;
    };

    /// \brief A GET of `target` (`path` or `path?query`) with `headers`.
    /// \ingroup core
    /// \param c       The client.
    /// \param target  The target.
    /// \param headers The headers.
    /// \return As `send`.
    template <http_client C>
    capability_result<http_response> http_get(C& c, std::string_view target, std::vector<http_header> headers = {})
    {
        http_request r;
        const auto q = target.find('?');
        r.path = std::string(target.substr(0, q));
        if (q != std::string_view::npos)
            r.query = std::string(target.substr(q + 1));
        r.headers = std::move(headers);
        return c.send(r);
    }

} // namespace snicholls

#endif // snicholls_core_concepts_http_client_hpp
