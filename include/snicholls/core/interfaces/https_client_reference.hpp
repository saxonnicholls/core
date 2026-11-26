//
//  https_client_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::memory_https_client`, the standard-library reference `https_client`: an in-memory router by
/// host; and recorded days (`cassette`, `recording_https_client`, `replay_https_client`).
/// \ingroup core
///
/// **Overview.** Three models of `https_client`, none of which touches a network:
/// - `memory_https_client`: routes each request by its URL's host to a handler (`https_request` in, a response or a
///   transport error out) set with `route`, or to any `http_handler` (`concepts/http_handler.hpp`, such as core's
///   `route_table`) set with `serve`; a host with no route is `unavailable`, as an unreachable one is. It makes the
///   checks a real client makes first (`parse_https_url`, `is_sendable`) and keeps what it sent, already redacted as a
///   cassette records it, so a test can check the requests. A provider's fixture server is such a handler; it may answer
///   with a transport error to script a dropped connection or a timeout. `fail_next(code, response)` makes the next
///   `send` fail as though the TLS layer had answered `response`.
/// - `recording_https_client<C>`: forwards to any model and appends each answered exchange to a `cassette`.
/// - `replay_https_client`: answers from a `cassette`, in order: each request takes the first unused exchange with the
///   same method, URL, idempotency key and body; nothing matching is `capability_errc::not_found`.
///
/// A `cassette` is a recorded day. `save` writes it as text and `load` reads it back, never with the `Authorization`
/// value (it is not recorded at all, only that one was sent) and never with a sensitive body (recorded as
/// `cassette::redacted`, which replay matches against any body). Response `Set-Cookie` headers are dropped. The text,
/// one exchange after another, bodies length-prefixed so any bytes survive:
///
///     > POST https://api.stripe.com/v1/refunds
///     > a
///     > h Idempotency-Key: r-1
///     > b 23
///     payment_intent=pi_1&...
///     < 200
///     < h Request-Id: req_1
///     < b 42
///     {...}
///
/// \snippet interfaces/https_client_reference_demo.cpp reference
///
/// **Relationships.** Models `https_client` (`concepts/https_client.hpp`); passes its kit in every form; `serve`
/// bridges to `http_handler` (`concepts/http_handler.hpp`), so a product's routes can be driven through a client. Group:
/// \ref core.
///
/// **Implementations / variants.** The router, the recorder and the replay.
///
/// **Guarantees.** Not thread-safe. Deterministic. Nothing recorded carries a credential or a sensitive body; errors
/// carry fixed messages only.
///
/// **Errors.** As the contract; `load` returns `invalid_argument` on malformed text.
///
/// **Performance.** Not benchmarked: linear in the routes and in the cassette.
///
/// **Tests and demos.** `tests/interfaces/https_client_reference_test.cpp`,
/// `demos/interfaces/https_client_reference_demo.cpp`.
///
/// **Dependencies.** `<charconv>`, `<cstddef>`, `<functional>`, `<optional>`, `<string>`, `<string_view>`,
/// `<system_error>`, `<type_traits>`, `<utility>`, `<vector>`, `concepts/http_handler.hpp`,
/// `interfaces/https_client.hpp`.
///
/// **Known issues.** A path handed to an `http_handler` by `serve` is not percent-decoded. The client certificate is not
/// recorded.
///
/// **Status.** New in 0.27.0, lifted from servers.

#ifndef snicholls_core_interfaces_https_client_reference_hpp
#define snicholls_core_interfaces_https_client_reference_hpp

#include <snicholls/core/concepts/http_handler.hpp>
#include <snicholls/core/interfaces/https_client.hpp>

#include <charconv>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief One request and its answer, as recorded: never the credential, never a sensitive body.
    /// \ingroup core
    struct recorded_exchange {
        http_method method{http_method::get};      ///< the method
        std::string url;                           ///< the URL
        std::vector<http_header> request_headers;  ///< the request's headers, `Idempotency-Key` among them when set
        std::string request_body;                  ///< the body, or `cassette::redacted`
        bool had_authorization{false};             ///< whether a credential was sent; its value never is recorded
        https_response response;                   ///< the answer

        /// \brief The idempotency key that was sent, or empty.
        /// \return A view of it, valid while this exchange is.
        [[nodiscard]] std::string_view idempotency_key() const noexcept
        {
            return find_header(request_headers, "Idempotency-Key").value_or(std::string_view{});
        }

        /// \brief Equality, member by member.
        friend bool operator==(const recorded_exchange&, const recorded_exchange&) = default;
    };

    /// \brief A recorded day: exchanges in order, with their text form.
    /// \ingroup core
    struct cassette {
        /// \brief What a sensitive body is recorded as.
        static constexpr std::string_view redacted = "<redacted>";

        std::vector<recorded_exchange> exchanges;  ///< in the order they happened

        /// \brief The recordable form of a request and its answer: the credential dropped, a sensitive body redacted.
        /// \param r        The request.
        /// \param response Its answer.
        /// \return The exchange.
        [[nodiscard]] static recorded_exchange record(const https_request& r, const https_response& response)
        {
            recorded_exchange x;
            x.method = r.method;
            x.url = r.url;
            for (const auto& h : r.headers)
                if (!ascii_iequals(h.name, "Authorization"))
                    x.request_headers.push_back(h);
            if (!r.idempotency_key.empty())
                x.request_headers.push_back({"Idempotency-Key", std::string(r.idempotency_key.view())});
            x.request_body = r.sensitive_body ? std::string(redacted) : r.body;
            x.had_authorization = !r.authorization.reveal().empty();
            x.response.status = response.status;
            x.response.body = response.body;
            for (const auto& h : response.headers)
                if (!ascii_iequals(h.name, "Set-Cookie"))
                    x.response.headers.push_back(h);
            return x;
        }

        /// \brief The text form.
        /// \return The text.
        [[nodiscard]] std::string save() const
        {
            std::string out;
            for (const auto& x : exchanges) {
                out += "> " + std::string(to_string(x.method)) + " " + x.url + "\n";
                if (x.had_authorization)
                    out += "> a\n";
                for (const auto& h : x.request_headers)
                    out += "> h " + h.name + ": " + h.value + "\n";
                out += "> b " + std::to_string(x.request_body.size()) + "\n" + x.request_body + "\n";
                out += "< " + std::to_string(x.response.status) + "\n";
                for (const auto& h : x.response.headers)
                    out += "< h " + h.name + ": " + h.value + "\n";
                out += "< b " + std::to_string(x.response.body.size()) + "\n" + x.response.body + "\n";
            }
            return out;
        }

        /// \brief Reads the text form.
        /// \param text What `save` wrote.
        /// \return The cassette, or `invalid_argument`.
        [[nodiscard]] static capability_result<cassette> load(std::string_view text)
        {
            constexpr capability_message malformed = "not a cassette";
            cassette c;
            std::size_t at = 0;
            const auto line = [&](std::string_view& out) {
                const auto nl = text.find('\n', at);
                if (at >= text.size() || nl == std::string_view::npos)
                    return false;
                out = text.substr(at, nl - at);
                at = nl + 1;
                return true;
            };
            const auto bytes = [&](std::string_view count, std::string& out) {
                std::size_t n = 0;
                const auto [end, ec] = std::from_chars(count.data(), count.data() + count.size(), n);
                if (ec != std::errc{} || end != count.data() + count.size() || n > text.size() - at ||
                    text.size() - at - n < 1 || text[at + n] != '\n')
                    return false;
                out.assign(text.substr(at, n));
                at += n + 1;
                return true;
            };
            const auto header = [](std::string_view l, std::vector<http_header>& out) {
                const auto colon = l.find(": ", 4);
                if (colon == std::string_view::npos)
                    return false;
                out.push_back({std::string(l.substr(4, colon - 4)), std::string(l.substr(colon + 2))});
                return true;
            };
            std::string_view l;
            while (line(l)) {
                recorded_exchange x;
                const auto space = l.find(' ', 2);
                const auto method = l.starts_with("> ") && space != std::string_view::npos
                                        ? parse_http_method(l.substr(2, space - 2)) : std::nullopt;
                if (!method)
                    return fail(capability_errc::invalid_argument, {}, malformed);
                x.method = *method;
                x.url = std::string(l.substr(space + 1));
                bool have_body = false;
                while (!have_body && line(l)) {
                    if (l == "> a")
                        x.had_authorization = true;
                    else if (!(l.starts_with("> h ") && header(l, x.request_headers)) &&
                             !(l.starts_with("> b ") && (have_body = bytes(l.substr(4), x.request_body))))
                        return fail(capability_errc::invalid_argument, {}, malformed);
                }
                if (!have_body || !line(l) || !l.starts_with("< "))
                    return fail(capability_errc::invalid_argument, {}, malformed);
                const auto status = l.substr(2);
                const auto [end, ec] = std::from_chars(status.data(), status.data() + status.size(), x.response.status);
                if (ec != std::errc{} || end != status.data() + status.size())
                    return fail(capability_errc::invalid_argument, {}, malformed);
                bool have_response_body = false;
                while (!have_response_body && line(l))
                    if (!(l.starts_with("< h ") && header(l, x.response.headers)) &&
                        !(l.starts_with("< b ") && (have_response_body = bytes(l.substr(4), x.response.body))))
                        return fail(capability_errc::invalid_argument, {}, malformed);
                if (!have_response_body)
                    return fail(capability_errc::invalid_argument, {}, malformed);
                c.exchanges.push_back(std::move(x));
            }
            return ok(std::move(c));
        }

        /// \brief Equality.
        friend bool operator==(const cassette&, const cassette&) = default;
    };

    /// \brief The reference client: an in-memory router by host, keeping what it sent.
    /// \ingroup core
    class memory_https_client {
    public:
        /// \brief A server: a request in, a response or a transport error to simulate out (a dropped connection is
        /// `unavailable`, no answer in time `timeout`). A fixture server, or a lambda in a test.
        using handler = std::function<capability_result<https_response>(const https_request&)>;

        /// \brief A client with no routes: every host is unreachable until routed.
        memory_https_client() = default;
        /// \brief A client whose every host `h` answers.
        /// \param every_host The handler.
        explicit memory_https_client(handler every_host) : fallback_(std::move(every_host)) {}

        /// \brief Routes `host` (compared ignoring ASCII case) to `h`, replacing any route it had.
        /// \param host The host name, without a port.
        /// \param h    The handler.
        /// \return `*this`.
        memory_https_client& route(std::string_view host, handler h)
        {
            for (auto& r : routes_)
                if (ascii_iequals(r.host, host)) {
                    r.answer = std::move(h);
                    return *this;
                }
            routes_.push_back({std::string(host), std::move(h)});
            return *this;
        }

        /// \brief Routes `host` to an `http_handler`, which must outlive the client: the request arrives as an
        /// `http_request` (the path and query split, `Authorization` and `Idempotency-Key` as headers).
        /// \tparam H The handler type.
        /// \param host The host name, without a port.
        /// \param h    The handler (borrowed).
        /// \return `*this`.
        template <typename H>
            requires http_handler<H>
        memory_https_client& serve(std::string_view host, H& h)
        {
            return route(host, [p = &h](const https_request& r) -> capability_result<https_response> {
                const auto url = parse_https_url(r.url);
                const std::string_view target = url->target.empty() ? std::string_view{"/"} : url->target;
                const auto q = target.find('?');
                http_request in;
                in.method = r.method;
                in.path = target.substr(0, q);
                if (in.path.empty())
                    in.path = "/";
                in.query = q == std::string_view::npos ? std::string{} : std::string(target.substr(q + 1));
                in.headers = r.headers;
                if (!r.authorization.reveal().empty())
                    in.headers.push_back({"Authorization", r.authorization.reveal()});
                if (!r.idempotency_key.empty())
                    in.headers.push_back({"Idempotency-Key", std::string(r.idempotency_key.view())});
                in.body = r.body;
                http_response out = p->handle(in);
                return ok(https_response{out.status, std::move(out.headers), std::move(out.body)});
            });
        }

        /// \brief Sends one request to its host's handler, after the checks a real client makes.
        /// \param r The request.
        /// \return The handler's answer; `invalid_argument` for a URL or header a real client refuses; `unavailable` for
        ///         a host with no route; or the armed failure.
        capability_result<https_response> send(const https_request& r)
        {
            if (injected_) {
                const capability_error e = *injected_;
                injected_.reset();
                ++failures_;
                return err(e);
            }
            const auto url = parse_https_url(r.url);
            if (url.is_err())
                return err(url.unwrap_err());
            for (const auto& h : r.headers)
                if (!is_sendable(h))
                    return fail(capability_errc::invalid_argument, {}, "a header that would split the request");
            if (!is_sendable({"Authorization", r.authorization.reveal()}) ||
                !is_sendable({"Idempotency-Key", std::string(r.idempotency_key.view())}))
                return fail(capability_errc::invalid_argument, {}, "a header that would split the request");
            const handler* answer = fallback_ ? &fallback_ : nullptr;
            for (const auto& route : routes_)
                if (ascii_iequals(route.host, url->host))
                    answer = &route.answer;
            if (!answer) {
                ++failures_;
                return fail(capability_errc::unavailable, {}, "no route to the host");
            }
            auto response = (*answer)(r);
            if (response.is_ok())
                seen_.exchanges.push_back(cassette::record(r, *response));
            else
                ++failures_;
            return response;
        }

        /// \brief What was sent and answered, redacted as a cassette records it.
        /// \return The exchanges.
        [[nodiscard]] const cassette& seen() const noexcept { return seen_; }

        /// \brief How many requests failed at the transport (no route, a handler's error, an armed failure).
        /// \return The count.
        [[nodiscard]] std::size_t transport_failures() const noexcept { return failures_; }

        /// \brief Makes the next `send` fail (once) as though the TLS layer had answered `response`.
        /// \param code     The error's code.
        /// \param response The response: its first word becomes the provider code when it is one; nothing else is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

    private:
        struct host_route {
            std::string host;
            handler answer;
        };

        std::vector<host_route> routes_;
        handler fallback_;
        cassette seen_;
        std::size_t failures_{0};
        std::optional<capability_error> injected_;
    };

    static_assert(https_client<memory_https_client>, "snicholls::memory_https_client must model snicholls::https_client");

    /// \brief Forwards to any client and records each exchange that got an answer.
    /// \ingroup core
    /// \tparam C The client, or a reference to one.
    template <typename C>
        requires https_client<std::remove_reference_t<C>>
    class recording_https_client {
    public:
        /// \brief Records what `inner` answers.
        /// \param inner The client (a value, or for a reference `C`, borrowed).
        explicit recording_https_client(C inner) : inner_(std::forward<C>(inner)) {}

        /// \brief Forwards and records.
        /// \param r The request.
        /// \return The inner client's answer.
        capability_result<https_response> send(const https_request& r)
        {
            auto answer = inner_.send(r);
            if (answer.is_ok())
                tape_.exchanges.push_back(cassette::record(r, *answer));
            return answer;
        }

        /// \brief The recording so far.
        /// \return The cassette.
        [[nodiscard]] const cassette& tape() const noexcept { return tape_; }

    private:
        C inner_;
        cassette tape_;
    };

    /// \brief Answers from a recorded day.
    /// \ingroup core
    class replay_https_client {
    public:
        /// \brief Replays `c`.
        /// \param c The cassette.
        explicit replay_https_client(cassette c) : tape_(std::move(c)), used_(tape_.exchanges.size(), false) {}

        /// \brief The first unused exchange with this method, URL, idempotency key and body.
        /// \param r The request.
        /// \return The recorded answer, or `not_found`.
        capability_result<https_response> send(const https_request& r)
        {
            for (std::size_t i = 0; i < tape_.exchanges.size(); ++i) {
                const auto& x = tape_.exchanges[i];
                if (used_[i] || x.method != r.method || x.url != r.url || x.idempotency_key() != r.idempotency_key.view())
                    continue;
                if (x.request_body != cassette::redacted && x.request_body != r.body)
                    continue;
                used_[i] = true;
                return ok(x.response);
            }
            return fail(capability_errc::not_found, {}, "no such exchange recorded");
        }

        /// \brief How many recorded exchanges were not asked for.
        /// \return The count.
        [[nodiscard]] std::size_t unused() const noexcept
        {
            std::size_t n = 0;
            for (const bool u : used_)
                n += u ? 0U : 1U;
            return n;
        }

    private:
        cassette tape_;
        std::vector<bool> used_;
    };

    static_assert(https_client<replay_https_client>, "snicholls::replay_https_client must model snicholls::https_client");
    static_assert(https_client<recording_https_client<memory_https_client>>,
                  "snicholls::recording_https_client must model snicholls::https_client");

} // namespace snicholls

#endif // snicholls_core_interfaces_https_client_reference_hpp
