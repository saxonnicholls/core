//
//  http_handler.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::http_handler`: anything that answers an HTTP request with a response, with no transport in sight.
/// \ingroup core
///
/// **Overview.** Everything a server answers (routes, middleware, webhooks, health) is a function of an
/// `http_request` to an `http_response`. The transport (sockets, TLS, HTTP/1.1 and HTTP/2 framing) is the server's,
/// above core, and is not in this contract, so handlers are tested without a network and composed freely: a router
/// is a handler of handlers, a middleware wraps one. This is the concept half of a pair; the pure virtual half is
/// `http_handler_interface` (`interfaces/http_handler.hpp`), with `http_handler_adapter<H>` and `http_handler_ref`.
///
/// **The contract** (`interfaces/http_handler_conformance.hpp` checks it): `handle` always returns a response with a
/// status from 100 to 599; failure is a response (4xx, 5xx), not an exception; a handler that throws anyway is
/// answered 500 with no detail (the redaction rule of `capability_error.hpp` for HTTP: nothing of the exception
/// reaches the response). `HEAD` is answered as `GET` without a body. A request's body is the raw bytes, so a
/// webhook's signature can be checked over exactly what was sent.
///
/// \snippet concepts/http_handler_demo.cpp concept
///
/// **Relationships.** `http_header.hpp`; the twin `interfaces/http_handler.hpp`, the reference
/// `interfaces/http_handler_reference.hpp`, the kit `interfaces/http_handler_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `route_table` (`interfaces/http_handler_reference.hpp`, an exact-path router). Above core:
/// the servers' HTTP server.
///
/// **Guarantees.** `to_string(http_method)` and `parse_http_method` are `constexpr` and `noexcept`. A handler shared
/// across connections must be safe to call concurrently; that is the implementation's.
///
/// **Errors.** None: errors are responses.
///
/// **Performance.** `benchmarks/interfaces/http_handler_benchmark.cpp` (concept, virtual and hand-written dispatch);
/// not yet quoted from a quiet machine: `make benchmarks` records them in BENCHMARKS.md.
///
/// **Tests and demos.** `tests/concepts/http_handler_test.cpp`, `demos/concepts/http_handler_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<string>`, `<string_view>`, `<utility>`, `<vector>`,
/// `http_header.hpp`.
///
/// **Known issues.** The request and response own their strings: an HTTP message is a payload, not an identifier,
/// and routers above core build on owned requests. Streaming bodies and WebSocket upgrades are not in the contract.
///
/// **Helpers** (from crtp's `http_interface.hpp`, 0.27.0): `path_param` (a pattern's captures, in
/// `http_request::path_params`), `request_target`, `is_success`, `match_route`; headers by name are `find_header`
/// (`http_header.hpp`). A client of this contract is `concepts/http_client.hpp`.
///
/// **Status.** New in 0.23.0; the helpers and `path_params` in 0.27.0. Experimental until two real implementations
/// pass the kit.

#ifndef snicholls_core_concepts_http_handler_hpp
#define snicholls_core_concepts_http_handler_hpp

#include <snicholls/core/http_header.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief An HTTP method.
    /// \ingroup core
    enum class http_method : std::uint8_t {
        get = 1,      ///< GET
        head = 2,     ///< HEAD
        post = 3,     ///< POST
        put = 4,      ///< PUT
        patch = 5,    ///< PATCH
        del = 6,      ///< DELETE
        options = 7,  ///< OPTIONS
    };

    /// \brief The method's token on the wire (RFC 9110), not the enumerator's name.
    /// \ingroup core
    /// \param m The method.
    /// \return `"GET"`, `"HEAD"`, ..., or `"UNKNOWN"`.
    [[nodiscard]] constexpr std::string_view to_string(http_method m) noexcept
    {
        switch (m) {
        case http_method::get: return "GET";
        case http_method::head: return "HEAD";
        case http_method::post: return "POST";
        case http_method::put: return "PUT";
        case http_method::patch: return "PATCH";
        case http_method::del: return "DELETE";
        case http_method::options: return "OPTIONS";
        }
        return "UNKNOWN";
    }

    /// \brief Parses a method token, case-sensitively as RFC 9110 requires.
    /// \ingroup core
    /// \param s The token.
    /// \return The method, or `std::nullopt`.
    [[nodiscard]] constexpr std::optional<http_method> parse_http_method(std::string_view s) noexcept
    {
        for (const auto m : {http_method::get, http_method::head, http_method::post, http_method::put, http_method::patch,
                             http_method::del, http_method::options})
            if (to_string(m) == s)
                return m;
        return std::nullopt;
    }

    /// \brief One request.
    /// \ingroup core
    struct http_request {
        http_method method{http_method::get};  ///< the method
        std::string path;                      ///< the percent-decoded path, from `/`
        std::string query;                     ///< the raw query, without `?`
        std::vector<http_header> headers;      ///< every header, any case
        std::string body;                      ///< the raw body bytes
        std::string remote_address;            ///< the peer, after trusted proxies
        std::string request_id;                ///< ours, for logs and the response
        std::vector<http_header> path_params;  ///< what a pattern's `:name` segments captured (`route_table::add_pattern`)

        /// \brief Equality.
        friend bool operator==(const http_request&, const http_request&) = default;
    };

    /// \brief One response.
    /// \ingroup core
    struct http_response {
        int status{200};                   ///< 100 to 599
        std::vector<http_header> headers;  ///< the headers
        std::string body;                  ///< the body

        /// \brief Equality.
        friend bool operator==(const http_response&, const http_response&) = default;
    };

    /// \brief A response with a status, and a body with its content type.
    /// \ingroup core
    /// \param status       The status.
    /// \param body         The body; no `Content-Type` when empty.
    /// \param content_type Its type.
    /// \return The response.
    [[nodiscard]] inline http_response respond(int status, std::string body = {},
                                               std::string_view content_type = "text/plain; charset=utf-8")
    {
        http_response r;
        r.status = status;
        if (!body.empty())
            r.headers.push_back({"Content-Type", std::string{content_type}});
        r.body = std::move(body);
        return r;
    }

    /// \brief A path parameter a route pattern captured: `path_param(r, "id")` for `/orders/:id`.
    /// \ingroup core
    /// \param r    The request.
    /// \param name The parameter's name, without the colon (case-sensitive).
    /// \return A view of its value (valid while `r` is), or `std::nullopt`.
    [[nodiscard]] inline std::optional<std::string_view> path_param(const http_request& r, std::string_view name) noexcept
    {
        for (const http_header& p : r.path_params)
            if (p.name == name)
                return std::string_view{p.value};
        return std::nullopt;
    }

    /// \brief The request's target as sent on the wire: `path?query`, or the path alone.
    /// \ingroup core
    /// \param r The request.
    /// \return The target.
    [[nodiscard]] inline std::string request_target(const http_request& r)
    {
        return r.query.empty() ? r.path : r.path + '?' + r.query;
    }

    /// \brief Whether a response is a success (2xx).
    /// \ingroup core
    /// \param r The response.
    /// \return `200 <= status < 300`.
    [[nodiscard]] constexpr bool is_success(const http_response& r) noexcept { return r.status >= 200 && r.status < 300; }

    /// \brief Matches `path` against a route pattern whose `:name` segments capture: `/orders/:id` matches
    /// `/orders/42` with `id` = `42`. Segments compare exactly; a capture is never empty and never spans a `/`.
    /// \ingroup core
    /// \param pattern  The pattern, from `/`.
    /// \param path     The request's path.
    /// \param captured Where the captures are appended (in pattern order) when it matches; untouched otherwise.
    /// \return Whether it matches.
    [[nodiscard]] inline bool match_route(std::string_view pattern, std::string_view path, std::vector<http_header>& captured)
    {
        std::vector<http_header> found;
        std::size_t i = 0, j = 0;
        while (i < pattern.size() && j < path.size()) {
            if (pattern[i] != '/' || path[j] != '/')
                return false;
            ++i;
            ++j;
            const std::size_t pe = std::min(pattern.find('/', i), pattern.size());
            const std::size_t qe = std::min(path.find('/', j), path.size());
            const std::string_view seg = pattern.substr(i, pe - i);
            const std::string_view got = path.substr(j, qe - j);
            if (!seg.empty() && seg.front() == ':') {
                if (got.empty())
                    return false;
                found.push_back({std::string(seg.substr(1)), std::string(got)});
            } else if (seg != got) {
                return false;
            }
            i = pe;
            j = qe;
        }
        if (i != pattern.size() || j != path.size())
            return false;
        captured.insert(captured.end(), found.begin(), found.end());
        return true;
    }

    /// \brief An HTTP handler: `h.handle(request)`.
    /// \ingroup core
    template <typename H>
    concept http_handler = requires(H& h, const http_request& r) {
        { h.handle(r) } -> std::same_as<http_response>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_http_handler_hpp
