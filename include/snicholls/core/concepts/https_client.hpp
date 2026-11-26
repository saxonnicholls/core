//
//  https_client.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::https_client`: one request out to a provider's API, one response back, held as data.
/// \ingroup core
///
/// **Overview.** Every provider adapter (Stripe, Wise, OKX, Xero) talks to its API through an `https_client`, so the
/// adapter's logic is tested against a fixture server or a recorded day with no network, and the transport (TLS,
/// timeouts, retries) is written once. The request and the response are values: a method (`http_method`, shared with
/// `concepts/http_handler.hpp`), a URL, headers (`http_header.hpp`), a body, timeouts and a reference to a client
/// certificate. Core does no I/O: the transport is ts-moveables' and servers'. This is the concept half of a pair; the
/// pure virtual half is `https_client_interface` (`interfaces/https_client.hpp`), with `https_client_adapter<C>` and
/// `https_client_ref`.
///
/// **The contract** (`interfaces/https_client_conformance.hpp` checks it):
/// - `send` returns the provider's answer whatever its status: a 4xx or a 5xx is a response, not an error. An error is
///   a transport failure only: `capability_errc::unavailable` (no connection, TLS refused, the connection dropped),
///   `capability_errc::timeout` (no complete answer in time; the outcome is unknown), `capability_errc::invalid_argument`
///   (a URL that `parse_https_url` refuses, or a header `is_sendable` refuses: no request is sent).
/// - **Credentials travel only in `authorization`** (a `snicholls::secret`, sent as the `Authorization` header). A body
///   that carries a secret (a signed challenge, a token exchange) sets `sensitive_body`. An implementation never logs or
///   records `authorization` or a sensitive body, and neither ever reaches an error.
/// - `idempotency_key`, when set, is sent as the `Idempotency-Key` header; a retry layer retries a request that changes
///   state (`is_idempotent` false) only when it carries one.
/// - `client_certificate`, when set, names a client certificate the transport holds (mutual TLS, a bank's API); the
///   request never carries the certificate or its key. An implementation without that certificate is `unsupported`.
/// - `connect_timeout` and `timeout` of zero mean the implementation's defaults; `timeout` bounds the whole exchange.
/// - Errors follow the redaction rule of `capability_error.hpp`: no response, URL or credential in them.
///
/// \snippet concepts/https_client_demo.cpp concept
///
/// **Relationships.** `concepts/http_handler.hpp` (`http_method`), `http_header.hpp`, `secret.hpp`,
/// `capability_error.hpp`, `capability_ids.hpp`; the twin `interfaces/https_client.hpp`, the reference
/// `interfaces/https_client_reference.hpp`, the kit `interfaces/https_client_conformance.hpp`. Group: \ref core.
///
/// **Implementations.**
/// - `memory_https_client` (`interfaces/https_client_reference.hpp`): an in-memory router, by host, to handlers or to
///   any `http_handler`; with `cassette`, `recording_https_client` and `replay_https_client` for recorded days.
///   Standard library only.
/// - Above core: servers' `tls_https_client` (TLS through ts-moveables and OpenSSL) and `retrying_https_client`
///   (back-off honouring `Retry-After`, over any model).
///
/// **Guarantees.** `is_idempotent`, `is_sendable` and `parse_https_url` are `constexpr`, `noexcept` and
/// allocation-free. `https_url` views the URL it was parsed from. Thread safety is the implementation's.
///
/// **Errors.** `capability_error`, as the contract says; `parse_https_url` returns `invalid_argument`.
///
/// **Performance.** Dominated by the network; the reference by its handler.
///
/// **Tests and demos.** `tests/concepts/https_client_test.cpp`, `demos/concepts/https_client_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<concepts>`, `<cstdint>`, `<string>`, `<string_view>`, `<vector>`,
/// `capability_error.hpp`, `capability_ids.hpp`, `concepts/http_handler.hpp`, `http_header.hpp`, `secret.hpp`.
///
/// **Known issues.** Bodies are whole strings: no streaming. IPv6 literal hosts (`[::1]`) are refused.
///
/// **Status.** New in 0.27.0, lifted from servers. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_https_client_hpp
#define snicholls_core_concepts_https_client_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/concepts/http_handler.hpp>
#include <snicholls/core/http_header.hpp>
#include <snicholls/core/secret.hpp>

#include <chrono>
#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief Whether sending the same request twice has the effect of sending it once (RFC 9110 section 9.2.2).
    /// \ingroup core
    /// \param m The method.
    /// \return `true` for GET, HEAD, PUT, DELETE and OPTIONS; `false` for POST and PATCH.
    [[nodiscard]] constexpr bool is_idempotent(http_method m) noexcept { return m != http_method::post && m != http_method::patch; }

    /// \brief Whether a header can be sent as it is: a name of RFC 9110 token characters, a value with no CR, LF or NUL.
    /// \ingroup core
    /// \param h The header.
    /// \return `true` when sending it cannot split or smuggle a request.
    [[nodiscard]] constexpr bool is_sendable(const http_header& h) noexcept
    {
        if (h.name.empty())
            return false;
        for (const char c : h.name)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                  std::string_view{"!#$%&'*+-.^_`|~"}.find(c) != std::string_view::npos))
                return false;
        return h.value.find_first_of(std::string_view{"\r\n\0", 3}) == std::string::npos;
    }

    /// \brief An `https://` URL in its parts, viewing the text it was parsed from.
    /// \ingroup core
    struct https_url {
        std::string_view host;       ///< the host name, as written
        std::uint16_t port{443};     ///< the port; 443 when not given
        std::string_view target{};   ///< the path and query, from `/` or `?`; empty means `/`
    };

    /// \brief Parses `https://host[:port][/path][?query]`.
    /// \ingroup core
    /// \param url The URL, already percent-encoded.
    /// \return Its parts, or `invalid_argument` for another scheme, user information, an empty or bracketed host, a bad
    ///         port, or a fragment.
    [[nodiscard]] constexpr capability_result<https_url> parse_https_url(std::string_view url) noexcept
    {
        constexpr std::string_view scheme = "https://";
        if (!url.starts_with(scheme) || url.find('#') != std::string_view::npos)
            return fail(capability_errc::invalid_argument, {}, "not an https URL");
        std::string_view rest = url.substr(scheme.size());
        const auto end = rest.find_first_of("/?");
        std::string_view authority = rest.substr(0, end);
        https_url out;
        out.target = end == std::string_view::npos ? std::string_view{} : rest.substr(end);
        if (const auto colon = authority.find(':'); colon != std::string_view::npos) {
            const auto digits = authority.substr(colon + 1);
            if (digits.empty() || digits.size() > 5)
                return fail(capability_errc::invalid_argument, {}, "not a port");
            std::uint32_t port = 0;
            for (const char c : digits) {
                if (c < '0' || c > '9')
                    return fail(capability_errc::invalid_argument, {}, "not a port");
                port = port * 10 + static_cast<std::uint32_t>(c - '0');
            }
            if (port == 0 || port > 65535)
                return fail(capability_errc::invalid_argument, {}, "not a port");
            out.port = static_cast<std::uint16_t>(port);
            authority = authority.substr(0, colon);
        }
        if (authority.empty())
            return fail(capability_errc::invalid_argument, {}, "no host");
        for (const char c : authority)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '.'))
                return fail(capability_errc::invalid_argument, {}, "not a host name");
        out.host = authority;
        return ok(out);
    }

    /// \brief One request.
    /// \ingroup core
    struct https_request {
        http_method method{http_method::get};       ///< the method
        std::string url;                            ///< `https://host[:port]/path?query`, already percent-encoded
        std::vector<http_header> headers;           ///< never a credential: that goes in `authorization`
        secret<std::string> authorization{};        ///< the whole `Authorization` value; empty for none
        std::string body;                           ///< the body (a form or JSON)
        bool sensitive_body{false};                 ///< the body carries a secret: never logged or recorded
        client_id idempotency_key{};                ///< sent as `Idempotency-Key` when not empty
        provider_id client_certificate{};           ///< the id of a client certificate the transport holds; empty for none
        std::chrono::milliseconds connect_timeout{0};  ///< zero: the client's default
        std::chrono::milliseconds timeout{0};          ///< the whole exchange; zero: the client's default

        /// \brief Equality, member by member.
        friend bool operator==(const https_request&, const https_request&) = default;
    };

    /// \brief One response, whatever its status.
    /// \ingroup core
    struct https_response {
        int status{0};                     ///< the HTTP status
        std::vector<http_header> headers;  ///< every header, as received
        std::string body;                  ///< the body

        /// \brief Whether the status is 2xx.
        /// \return `200 <= status < 300`.
        [[nodiscard]] constexpr bool ok() const noexcept { return status >= 200 && status < 300; }

        /// \brief Equality, member by member.
        friend bool operator==(const https_response&, const https_response&) = default;
    };

    /// \brief An HTTPS client: `c.send(request)`.
    /// \ingroup core
    template <typename C>
    concept https_client = requires(C& c, const https_request& r) {
        { c.send(r) } -> std::same_as<capability_result<https_response>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_https_client_hpp
