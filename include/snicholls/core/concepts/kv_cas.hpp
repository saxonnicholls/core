//
//  kv_cas.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The compare-and-swap of servers' interim key_value_store (interfaces/key_value_store.hpp, compare_and_swap), the
//  one generic operation of the four duplicate key-value contracts that core's lacked, as a capability of its own like
//  kv_health and kv_backup.
//

/// \file
/// \brief The compare-and-swap capability of a store, static form: `kv_cas<S>`.
/// \ingroup core
///
/// **Overview.** `s.compare_and_swap(key, expected, desired)` replaces the value of `key` with `desired` only when
/// it is exactly `expected`, atomically: what a lease, a lock with a fencing token, an idempotency record or an
/// optimistic update needs, and what `insert` (absent only) and `apply` (no conditions) cannot express. Either side
/// may be `std::nullopt`: `expected` absent means "only if the key is absent", `desired` absent means "erase it". It
/// returns `true` when it swapped and `false` when the value was something else, in which case nothing changed.
/// `kv_compare_and_swap(store, key, expected, desired)` takes any one-byte sequences.
///
/// \snippet concepts/kv_cas_demo.cpp cas
///
/// **Relationships.** `concepts/key_value_store.hpp` (`kv_bytes`, `kv_errc`); the virtual twin `kv_cas_interface`
/// and the kit `check_kv_cas` (`interfaces/kv_cas.hpp`); `reference_key_value_store` implements it. Group: \ref core.
///
/// **Implementations / variants.** `reference_key_value_store`; above core, a SQL store (`UPDATE ... WHERE v = ?`),
/// RocksDB (a transaction) or Redis (`WATCH`/`MULTI`).
///
/// **Guarantees.** Atomic with respect to every other write of the store; a swap that changes the store advances its
/// `sequence()`, a refused one does not.
///
/// **Errors.** The store's `kv_errc` (`unavailable`, `io`, `read_only`, ...); a mismatch is `false`, not an error.
///
/// **Performance.** Not benchmarked: one map lookup in the reference.
///
/// **Tests and demos.** `tests/concepts/kv_cas_test.cpp`, `demos/concepts/kv_cas_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<optional>`, `concepts/key_value_store.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental.

#ifndef snicholls_core_concepts_kv_cas_hpp
#define snicholls_core_concepts_kv_cas_hpp

#include <snicholls/core/concepts/key_value_store.hpp>

#include <concepts>
#include <optional>

namespace snicholls {

    /// \brief A store with compare-and-swap: `s.compare_and_swap(key, expected, desired) -> result<bool, kv_errc>`.
    /// \ingroup core
    /// \tparam S the store type.
    template <typename S>
    concept kv_cas = requires(S& s, kv_bytes key, std::optional<kv_bytes> expected, std::optional<kv_bytes> desired) {
        { s.compare_and_swap(key, expected, desired) } -> std::same_as<result<bool, kv_errc>>;
    };

    /// \brief `compare_and_swap` over any one-byte sequences (`std::string_view`, literals, byte vectors).
    /// \ingroup core
    /// \param store    The store.
    /// \param key      The key.
    /// \param expected The value it must have, or `std::nullopt` for "absent".
    /// \param desired  The value to set, or `std::nullopt` to erase.
    /// \return Whether it swapped, or the error.
    template <kv_cas S, kv_byte_source K>
    result<bool, kv_errc> kv_compare_and_swap(S& store, const K& key, std::optional<kv_bytes> expected, std::optional<kv_bytes> desired)
    {
        return store.compare_and_swap(kv_bytes_of(key), expected, desired);
    }
}

#endif /* snicholls_core_concepts_kv_cas_hpp */
