//
//  memory_webhook.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Internal: the signed webhooks of core's in-memory reference providers (`memory_payment_provider`,
/// `memory_security_provider`), written once.
/// \ingroup core
///
/// A `Memory-Signature: t=<unix seconds>,v1=<hex>` header, the tag HMAC-SHA256 under the webhook secret over
/// `<t>.<raw body>` (Stripe's scheme), several `v1` accepted during a secret rotation, a 300-second window; the body
/// `key=value` lines of a `webhook_event`'s fields, its kind by the enumerator's name. Not public API; tested in
/// `tests/detail/memory_webhook_test.cpp` and through both references' kits.

#ifndef snicholls_core_detail_memory_webhook_hpp
#define snicholls_core_detail_memory_webhook_hpp

#include <snicholls/core/concepts/payment_provider.hpp>
#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/secret.hpp>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace snicholls::detail {

    // The signing scheme of the reference providers.
    struct memory_webhook {
        static constexpr std::string_view header = "Memory-Signature";
        static constexpr std::chrono::seconds tolerance{300};

        static digest_hex<32> tag(const secret<std::string>& key, std::string_view t, std::string_view body)
        {
            hmac<sha256> m{key.reveal()};
            m.update(t);
            m.update(std::string_view{"."});
            m.update(body);
            return m.finish().hex();
        }

        static bool parse_int(std::string_view s, std::int64_t& out) noexcept
        {
            const auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
            return ec == std::errc{} && end == s.data() + s.size();
        }

        // A delivery of `e`, signed at `signed_at` and received at `received_at`.
        static owned_delivery sign(const secret<std::string>& key, const webhook_event& e, utc_s signed_at, utc_s received_at)
        {
            owned_delivery d;
            d.body = format(e);
            const std::string t = std::to_string(signed_at.unix_seconds());
            d.headers.push_back({std::string(header), "t=" + t + ",v1=" + std::string(tag(key, t, d.body).view())});
            d.received_at = received_at;
            return d;
        }

        // Authenticates (constant time) and parses; fails closed.
        static capability_result<webhook_event> verify(const secret<std::string>& key, const webhook_delivery& w)
        {
            const auto sig = find_header(w.headers, header);
            if (!sig)
                return fail(capability_errc::signature_invalid, {}, "no signature header");
            std::string_view t;
            std::vector<std::string_view> tags;
            for (std::string_view rest = *sig; !rest.empty();) {
                const auto comma = rest.find(',');
                const std::string_view part = rest.substr(0, comma);
                rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
                if (part.starts_with("t="))
                    t = part.substr(2);
                else if (part.starts_with("v1="))
                    tags.push_back(part.substr(3));
            }
            std::int64_t seconds = 0;
            if (t.empty() || tags.empty() || !parse_int(t, seconds))
                return fail(capability_errc::signature_invalid, {}, "a malformed signature header");
            const auto expected = tag(key, t, w.body);
            bool matched = false;
            for (const std::string_view candidate : tags)
                matched = constant_time_equal(expected.view(), candidate) || matched;
            if (!matched)
                return fail(capability_errc::signature_invalid, {}, "no signature matches");
            const auto sent = utc_s::from_unix_seconds(seconds);
            if (w.received_at - sent > tolerance || sent - w.received_at > tolerance)
                return fail(capability_errc::stale, {}, "outside the 300-second window");
            return parse(w.body);
        }

        static std::string format(const webhook_event& e)
        {
            std::string b;
            const auto line = [&b](std::string_view k, std::string_view v) {
                b += k;
                b += '=';
                b += v;
                b += '\n';
            };
            line("source", e.source.view());
            line("provider_account", e.provider_account.view());
            line("id", e.event_id.view());
            line("fact_id", e.fact_id.view());
            line("kind", enum_name(e.kind));
            line("type", e.type.view());
            line("payment_id", e.payment_id.view());
            line("our_reference", e.our_reference.view());
            line("reference", e.reference.view());
            line("occurred_at", std::to_string(e.occurred_at.unix_seconds()));
            line("needs_lookup", e.needs_lookup ? "1" : "0");
            line("is_test", e.is_test ? "1" : "0");
            return b;
        }

        static capability_result<webhook_event> parse(std::string_view body)
        {
            webhook_event e;
            bool have_id = false, have_kind = false, fits = true;
            const auto set = [&fits](auto& field, std::string_view v) {
                auto r = std::remove_reference_t<decltype(field)>::make(v);
                if (r)
                    field = *r;
                else
                    fits = false;
            };
            for (std::string_view rest = body; !rest.empty();) {
                const auto nl = rest.find('\n');
                const std::string_view line = rest.substr(0, nl);
                rest = nl == std::string_view::npos ? std::string_view{} : rest.substr(nl + 1);
                const auto eq = line.find('=');
                if (eq == std::string_view::npos)
                    return fail(capability_errc::invalid_argument, {}, "a webhook line without =");
                const std::string_view k = line.substr(0, eq), v = line.substr(eq + 1);
                if (k == "id") {
                    set(e.event_id, v);
                    have_id = !v.empty();
                } else if (k == "kind") {
                    const auto kind = enum_cast<event_kind>(v);
                    if (!kind)
                        return fail(capability_errc::invalid_argument, {}, "an unknown event kind");
                    e.kind = *kind;
                    have_kind = true;
                } else if (k == "occurred_at") {
                    std::int64_t s = 0;
                    if (!parse_int(v, s))
                        return fail(capability_errc::invalid_argument, {}, "occurred_at is not unix seconds");
                    e.occurred_at = utc_s::from_unix_seconds(s);
                } else if (k == "needs_lookup") {
                    e.needs_lookup = v == "1";
                } else if (k == "is_test") {
                    e.is_test = v == "1";
                } else if (k == "source") {
                    set(e.source, v);
                } else if (k == "provider_account") {
                    set(e.provider_account, v);
                } else if (k == "fact_id") {
                    set(e.fact_id, v);
                } else if (k == "type") {
                    set(e.type, v);
                } else if (k == "payment_id") {
                    set(e.payment_id, v);
                } else if (k == "our_reference") {
                    set(e.our_reference, v);
                } else if (k == "reference") {
                    set(e.reference, v);
                } else {
                    return fail(capability_errc::invalid_argument, {}, "an unknown webhook field");
                }
            }
            if (!have_id || !have_kind || !fits)
                return fail(capability_errc::invalid_argument, {}, "a webhook without id or kind, or a field too long");
            return ok(e);
        }
    };

} // namespace snicholls::detail

#endif // snicholls_core_detail_memory_webhook_hpp
