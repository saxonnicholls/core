//
//  role_store_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::kv_role_store<S>`, the reference `role_store` over any `key_value_store`, and
/// `memory_role_store`, it over core's `reference_key_value_store`.
/// \ingroup core
///
/// **Overview.** Grants and their history as rows of a key-value store, so the same code runs over the in-memory
/// reference in a test and over any store a product persists to. Each `grant` and `revoke` is one atomic batch: the
/// current grant (put or erased), the event, and the next sequence number. The rows (keys length-prefixed, so a
/// tenant's rows are one prefix and never another tenant's):
///
/// | Key | Value |
/// |---|---|
/// | `g` tenant-length tenant party(16 bytes) role | the grant |
/// | `e` tenant-length tenant sequence(8 bytes, big-endian) | the event, in order within the tenant |
/// | `s` | the next sequence number |
///
/// Values are length-prefixed fields with times as big-endian Unix seconds. `S` is the store by value (owned) or a
/// reference to one (borrowed). `fail_next(code, response)` makes the next call fail as though the backend had
/// answered `response`; a failure of the store itself is `unavailable` (or `conflict`, `internal`) with a fixed message.
///
/// \snippet interfaces/role_store_reference_demo.cpp reference
///
/// **Relationships.** Models `role_store` (`concepts/role_store.hpp`); passes its kit in every form; stores through
/// `kv_writable` (`concepts/key_value_store.hpp`), by default `reference_key_value_store`
/// (`interfaces/key_value_store_reference.hpp`). Group: \ref core.
///
/// **Implementations / variants.** `kv_role_store<S>` over any store; `memory_role_store`.
///
/// **Guarantees.** Strong: a call that fails changes nothing (one batch per write). Not thread-safe: synchronisation is
/// the caller's (or ts-moveables'). Rows a product wrote before a restart are read back by a new store over the same
/// backend.
///
/// **Errors.** As the contract; `invalid_argument` for a confirmation, reason or id that does not fit its bound;
/// `internal` ("a stored grant is malformed") for a row it cannot decode.
///
/// **Performance.** Not benchmarked: a prefix scan per read, one batch per write.
///
/// **Tests and demos.** `tests/interfaces/role_store_reference_test.cpp`,
/// `demos/interfaces/role_store_reference_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<optional>`, `<string>`, `<string_view>`, `<type_traits>`,
/// `<utility>`, `<vector>`, `concepts/key_value_store.hpp`, `interfaces/key_value_store_reference.hpp`,
/// `interfaces/role_store.hpp`.
///
/// **Known issues.** `members` and `history` scan the whole tenant.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_role_store_reference_hpp
#define snicholls_core_interfaces_role_store_reference_hpp

#include <snicholls/core/concepts/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/role_store.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    namespace detail {
        // The rows of kv_role_store: keys and values as length-prefixed bytes.
        struct role_rows {
            static void u8(std::string& o, std::size_t n) { o.push_back(static_cast<char>(static_cast<unsigned char>(n))); }
            static void text(std::string& o, std::string_view t)
            {
                u8(o, t.size());
                o.append(t);
            }
            static void id(std::string& o, const uuid& u)
            {
                for (const auto b : u.bytes)
                    o.push_back(static_cast<char>(b));
            }
            static void u64(std::string& o, std::uint64_t v)
            {
                for (int i = 7; i >= 0; --i)
                    o.push_back(static_cast<char>(static_cast<unsigned char>(v >> (8 * i))));
            }
            static void time(std::string& o, utc_s t) { u64(o, static_cast<std::uint64_t>(t.unix_seconds())); }

            static std::string tenant_prefix(char kind, std::string_view tenant)
            {
                std::string k(1, kind);
                text(k, tenant);
                return k;
            }
            static std::string grant_key(std::string_view tenant, const uuid& party, std::string_view role)
            {
                std::string k = tenant_prefix('g', tenant);
                id(k, party);
                k.append(role);
                return k;
            }
            static void grant(std::string& o, const role_grant& g)
            {
                id(o, g.subject);
                text(o, g.tenant.view());
                text(o, g.role.view());
                id(o, g.granted_by);
                text(o, g.reason.view());
                time(o, g.granted_at);
                u8(o, g.expires_at ? 1 : 0);
                time(o, g.expires_at.value_or(utc_s{}));
            }
            static std::string event(const role_event& e)
            {
                std::string o;
                u8(o, e.granted ? 1 : 0);
                grant(o, e.grant);
                text(o, e.confirmation.view());
                return o;
            }

            // Reads what the writers wrote; any shortfall or overflow clears `ok`.
            struct reader {
                std::string_view in;
                bool ok = true;

                std::string_view take(std::size_t n)
                {
                    if (!ok || in.size() < n) {
                        ok = false;
                        return {};
                    }
                    const auto out = in.substr(0, n);
                    in.remove_prefix(n);
                    return out;
                }
                std::size_t u8() { const auto b = take(1); return ok ? static_cast<unsigned char>(b[0]) : 0; }
                template <typename Text>
                Text text()
                {
                    const auto t = take(u8());
                    auto r = Text::make(t);
                    ok = ok && r.is_ok();
                    return ok ? *r : Text{};
                }
                uuid id()
                {
                    uuid u;
                    const auto b = take(16);
                    for (std::size_t i = 0; ok && i < 16; ++i)
                        u.bytes[i] = static_cast<std::uint8_t>(b[i]);
                    return u;
                }
                std::uint64_t u64()
                {
                    std::uint64_t v = 0;
                    for (const char c : take(8))
                        v = (v << 8) | static_cast<unsigned char>(c);
                    return v;
                }
                utc_s time() { return utc_s::from_unix_seconds(static_cast<std::int64_t>(u64())); }
                role_grant grant()
                {
                    role_grant g;
                    g.subject = id();
                    g.tenant = text<tenant_id>();
                    g.role = text<role_id>();
                    g.granted_by = id();
                    g.reason = text<short_text>();
                    g.granted_at = time();
                    const bool expires = u8() != 0;
                    const utc_s at = time();
                    if (expires)
                        g.expires_at = at;
                    return g;
                }
                role_event event()
                {
                    role_event e;
                    e.granted = u8() != 0;
                    e.grant = grant();
                    e.confirmation = text<short_text>();
                    ok = ok && in.empty();
                    return e;
                }
            };
        };
    } // namespace detail

    /// \brief The reference role store over a key-value store `S` (a value to own, or `T&` to borrow).
    /// \ingroup core
    /// \tparam S The store, or a reference to one.
    template <typename S>
        requires kv_writable<std::remove_reference_t<S>>
    class kv_role_store {
    public:
        /// \brief A store over a default-constructed backend.
        kv_role_store() requires std::is_default_constructible_v<S> : kv_() {}
        /// \brief A store over `backend` (owned, or for a reference `S`, borrowed).
        /// \param backend The key-value store.
        explicit kv_role_store(S backend) noexcept(std::is_nothrow_constructible_v<S, S&&>) : kv_(std::forward<S>(backend)) {}

        /// \brief The party's active grants in the tenant.
        /// \param party  The party.
        /// \param tenant The tenant.
        /// \param now    The time.
        /// \return The grants, or the error.
        [[nodiscard]] capability_result<std::vector<role_grant>> grants_of(const uuid& party, std::string_view tenant, utc_s now) const
        {
            if (tenant.size() > tenant_id::capacity())
                return ok(std::vector<role_grant>{});
            std::string prefix = detail::role_rows::tenant_prefix('g', tenant);
            detail::role_rows::id(prefix, party);
            return active(prefix, now);
        }

        /// \brief Every active grant in the tenant.
        /// \param tenant The tenant.
        /// \param now    The time.
        /// \return The grants, or the error.
        [[nodiscard]] capability_result<std::vector<role_grant>> members(std::string_view tenant, utc_s now) const
        {
            if (tenant.size() > tenant_id::capacity())
                return ok(std::vector<role_grant>{});
            return active(detail::role_rows::tenant_prefix('g', tenant), now);
        }

        /// \brief Records a grant, replacing one of the same party, tenant and role, and appends its event.
        /// \param g            The grant.
        /// \param confirmation The sentence the actor confirmed, if any (at most 255 characters).
        /// \return Success, or `invalid_argument`, or the store's error.
        capability_result<void> grant(const role_grant& g, std::string_view confirmation)
        {
            if (auto e = take_injected())
                return err(*e);
            const auto c = short_text::make(confirmation);
            if (!c)
                return fail(capability_errc::invalid_argument, {}, "a confirmation longer than 255 characters");
            std::string value;
            detail::role_rows::grant(value, g);
            return append(detail::role_rows::grant_key(g.tenant.view(), g.subject, g.role.view()), value, {true, g, *c});
        }

        /// \brief Records a revoke and appends its event.
        /// \param party  The party.
        /// \param tenant The tenant.
        /// \param role   The role.
        /// \param actor  Who revoked it.
        /// \param reason Why (at most 255 characters).
        /// \param now    When.
        /// \return Success, or `not_found` (nothing appended), `invalid_argument`, or the store's error.
        capability_result<void> revoke(const uuid& party, std::string_view tenant, std::string_view role, const uuid& actor,
                                       std::string_view reason, utc_s now)
        {
            if (auto e = take_injected())
                return err(*e);
            const auto why = short_text::make(reason);
            if (!why)
                return fail(capability_errc::invalid_argument, {}, "a reason longer than 255 characters");
            if (tenant.size() > tenant_id::capacity() || role.size() > role_id::capacity())
                return fail(capability_errc::not_found, {}, "no such grant");
            const std::string key = detail::role_rows::grant_key(tenant, party, role);
            const auto row = kv_get(kv_, key);
            if (row.is_err())
                return err(backend_error(row.unwrap_err()));
            if (!*row)
                return fail(capability_errc::not_found, {}, "no such grant");
            detail::role_rows::reader in{**row};
            role_grant g = in.grant();
            if (!in.ok || !in.in.empty())
                return fail(capability_errc::internal, {}, "a stored grant is malformed");
            g.granted_by = actor;
            g.reason = *why;
            g.granted_at = now;
            return append(key, std::nullopt, {false, g, {}});
        }

        /// \brief The tenant's history, oldest first.
        /// \param tenant The tenant.
        /// \return The events, or the error.
        [[nodiscard]] capability_result<std::vector<role_event>> history(std::string_view tenant) const
        {
            if (auto e = take_injected())
                return err(*e);
            std::vector<role_event> out;
            if (tenant.size() > tenant_id::capacity())
                return ok(std::move(out));
            bool malformed = false;
            const auto scanned = kv_scan_prefix(kv_, detail::role_rows::tenant_prefix('e', tenant), [&](kv_bytes, kv_bytes v) {
                detail::role_rows::reader in{as_text(v)};
                out.push_back(in.event());
                malformed = malformed || !in.ok;
            });
            if (scanned.is_err())
                return err(backend_error(scanned.unwrap_err()));
            if (malformed)
                return fail(capability_errc::internal, {}, "a stored event is malformed");
            return ok(std::move(out));
        }

        /// \brief Makes the next call fail (once) as though the backend had answered `response`.
        /// \param code     The error's code.
        /// \param response The response: its first word becomes the provider code when it is one; nothing else is kept.
        void fail_next(capability_errc code, std::string_view response = {}) noexcept
        {
            injected_ = provider_error(code, response.substr(0, response.find(' ')));
        }

        /// \brief The key-value store underneath.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<S>& backend() noexcept { return kv_; }

    private:
        static std::string_view as_text(kv_bytes b) noexcept { return {reinterpret_cast<const char*>(b.data()), b.size()}; }

        static capability_error backend_error(kv_errc e) noexcept
        {
            if (e == kv_errc::conflict)
                return {capability_errc::conflict, {}, "the key-value store refused the write"};
            if (e == kv_errc::corrupt || e == kv_errc::invalid_argument || e == kv_errc::limit)
                return {capability_errc::internal, {}, "the key-value store failed"};
            return {capability_errc::unavailable, {}, "the key-value store is unavailable"};
        }

        std::optional<capability_error> take_injected() const noexcept
        {
            auto e = injected_;
            injected_.reset();
            return e;
        }

        capability_result<std::vector<role_grant>> active(const std::string& prefix, utc_s now) const
        {
            if (auto e = take_injected())
                return err(*e);
            std::vector<role_grant> out;
            bool malformed = false;
            const auto scanned = kv_scan_prefix(kv_, prefix, [&](kv_bytes, kv_bytes v) {
                detail::role_rows::reader in{as_text(v)};
                const role_grant g = in.grant();
                malformed = malformed || !in.ok || !in.in.empty();
                if (g.active(now))
                    out.push_back(g);
            });
            if (scanned.is_err())
                return err(backend_error(scanned.unwrap_err()));
            if (malformed)
                return fail(capability_errc::internal, {}, "a stored grant is malformed");
            return ok(std::move(out));
        }

        // One batch: the grant row (put, or erased when `value` is empty), the event, the next sequence number.
        capability_result<void> append(const std::string& key, std::optional<std::string_view> value, const role_event& e)
        {
            const auto next = kv_get(kv_, std::string_view{"s"});
            if (next.is_err())
                return err(backend_error(next.unwrap_err()));
            std::uint64_t seq = 0;
            if (*next) {
                detail::role_rows::reader in{**next};
                seq = in.u64();
                if (!in.ok)
                    return fail(capability_errc::internal, {}, "the stored sequence is malformed");
            }
            std::string event_key = detail::role_rows::tenant_prefix('e', e.grant.tenant.view());
            detail::role_rows::u64(event_key, seq);
            std::string next_seq;
            detail::role_rows::u64(next_seq, seq + 1);
            kv_write_batch batch;
            if (value)
                batch.put(key, *value);
            else
                batch.erase(key);
            batch.put(event_key, detail::role_rows::event(e));
            batch.put(std::string_view{"s"}, next_seq);
            const auto applied = kv_.apply(batch);
            if (applied.is_err())
                return err(backend_error(applied.unwrap_err()));
            return ok();
        }

        S kv_;
        mutable std::optional<capability_error> injected_;
    };

    /// \brief The reference role store in memory: `kv_role_store` over core's `reference_key_value_store`.
    /// \ingroup core
    using memory_role_store = kv_role_store<reference_key_value_store>;

    static_assert(role_store<memory_role_store>, "snicholls::memory_role_store must model snicholls::role_store");
    static_assert(role_store<kv_role_store<reference_key_value_store&>>, "a borrowing kv_role_store models snicholls::role_store");

} // namespace snicholls

#endif // snicholls_core_interfaces_role_store_reference_hpp
