//
//  http_client_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::loopback_http_client`: the standard-library reference `http_client`, which sends to a handler in
/// memory.
/// \ingroup core
///
/// **Overview.** Code that calls an API through `http_client` is tested against a fake of that API written as an
/// `http_handler` (a `route_table`, say): `loopback_http_client<H>` validates the request as a transport would, hands it
/// to the handler and returns its response, `HEAD` without a body; nothing touches the network. `fail_next(code,
/// response)` makes the next send fail as though the transport had failed after receiving `response`, which is
/// dropped (its first word is kept as the provider code when it is one), to test a caller's retry path. `sent()`
/// counts the requests that reached the handler.
///
/// \snippet interfaces/http_client_reference_demo.cpp reference
///
/// **Relationships.** Models `http_client` (`concepts/http_client.hpp`); passes `interfaces/http_client_conformance.hpp`
/// directly and through `interfaces/http_client.hpp`; any `http_handler`, such as `route_table`
/// (`interfaces/http_handler_reference.hpp`). Group: \ref core.
///
/// **Implementations / variants.** `loopback_http_client<H>`, owning (`H`) or borrowing (`H&`) the handler.
///
/// **Guarantees.** Not thread-safe unless the handler is. Errors follow the redaction rule.
///
/// **Errors.** `invalid_argument` from `validate_request`; whatever `fail_next` armed.
///
/// **Performance.** Not benchmarked: a test double (one request copy).
///
/// **Tests and demos.** `tests/interfaces/http_client_reference_test.cpp`, `demos/interfaces/http_client_reference_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<optional>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `interfaces/http_client.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_http_client_reference_hpp
#define snicholls_core_interfaces_http_client_reference_hpp

#include <snicholls/core/interfaces/http_client.hpp>

#include <cstddef>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The reference client: sends to a handler in memory.
    /// \ingroup core
    /// \tparam H An `http_handler`, or a reference to one.
    template <typename H>
        requires http_handler<std::remove_reference_t<H>>
    class loopback_http_client {
    public:
        /// \brief A client of `handler`.
        /// \param handler The handler (owned, or referred to when `H` is a reference).
        explicit loopback_http_client(H handler) noexcept(std::is_nothrow_constructible_v<H, H&&>) : handler_(std::forward<H>(handler)) {}

        /// \brief Sends `r` to the handler.
        /// \param r The request.
        /// \return The handler's response, or the error.
        capability_result<http_response> send(const http_request& r)
        {
            if (auto valid = validate_request(r); !valid)
                return err(valid.unwrap_err());
            if (injected_) {
                const capability_error e = *injected_;
                injected_.reset();
                return err(e);
            }
            ++sent_;
            http_response out = handler_.handle(r);
            if (r.method == http_method::head)
                out.body.clear();
            return ok(std::move(out));
        }

        /// \brief Makes the next valid send fail (once), as though the transport failed after receiving `response`.
        /// \param code     The error's code.
        /// \param response What the transport received: its first word becomes the provider code when it is one, and
        ///                 nothing else of it is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

        /// \brief How many requests reached the handler.
        /// \return The count.
        [[nodiscard]] std::size_t sent() const noexcept { return sent_; }
        /// \brief The handler.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<H>& handler() noexcept { return handler_; }

    private:
        H handler_;
        std::optional<capability_error> injected_;
        std::size_t sent_ = 0;
    };

    /// \brief Deduces an owning client: `loopback_http_client(std::move(routes))`.
    /// \ingroup core
    template <typename H>
    loopback_http_client(H) -> loopback_http_client<H>;

} // namespace snicholls

#endif // snicholls_core_interfaces_http_client_reference_hpp
