//
//  kv_health.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The health check of a key-value store (asoMonoRepo1's ping() and diagnostics()), as a capability of its own,
//  with the diagnostics as plain data instead of JSON.
//

/// \file
/// \brief The health capability of a store, static form: `kv_health<S>` (`ping`, `diagnostics`) and `kv_diagnostics`.
/// \ingroup core
///
/// **Overview.** A supervisor, a readiness probe or an operator's console asks a store two things: can it serve
/// requests now (`ping()`), and what does it hold (`diagnostics()`, a `kv_diagnostics` struct). This is separate
/// from the store contract, so a store that cannot answer cheaply need not pretend to, and a caller that needs
/// health asks for exactly that capability.
///
/// \snippet concepts/kv_health_demo.cpp health
///
/// **Relationships.** The virtual twin is `kv_health_interface` (`interfaces/kv_health.hpp`), with adapters both
/// ways; the reference implementation is `reference_key_value_store`; the kit is `run_kv_health_conformance`
/// (`interfaces/key_value_store_conformance.hpp`). Errors are `kv_errc` (`concepts/key_value_store.hpp`). A
/// diagnostics struct is turned into JSON, a log line or a dashboard by the layer above, never here. Group: \ref core.
///
/// **Implementations / variants.** `reference_key_value_store`; the backends above core.
///
/// **Guarantees.** `ping()` succeeds exactly when the store can serve requests; it never changes the store.
/// `diagnostics().sequence` is the store's `sequence()` at the time of the call; `keys` and `bytes`, when present,
/// are exact for that moment, and absent when the backend cannot count them cheaply. No exception crosses either.
///
/// **Errors.** `ping()` and `diagnostics()` return the `kv_errc` that stops the store serving (`unavailable`,
/// `io`, ...).
///
/// **Examples.** See the overview.
///
/// **Performance.** Not benchmarked: one call per probe.
///
/// **Tests and demos.** `tests/concepts/kv_health_test.cpp`, `demos/concepts/kv_health_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstdint>`, `<optional>`, `<string_view>`, and
/// `concepts/key_value_store.hpp`.
///
/// **Known issues.** `backend` is a view of a static string (the implementation's name), never of a buffer.
///
/// **Status.** New in 0.22.0. Experimental.

#ifndef snicholls_core_concepts_kv_health_hpp
#define snicholls_core_concepts_kv_health_hpp

#include <snicholls/core/concepts/key_value_store.hpp>

#include <concepts>
#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief What a store reports about itself: plain data, for the layer above to print or export.
    /// \ingroup core
    struct kv_diagnostics {
        std::string_view backend{};          ///< the implementation's name, a static string (`"reference"`)
        std::uint64_t sequence = 0;          ///< the store's `sequence()` at the time of the call
        std::optional<std::uint64_t> keys{};  ///< the number of keys, when the backend knows it cheaply
        std::optional<std::uint64_t> bytes{}; ///< the bytes stored (keys plus values, or on disk), when known
        bool persistent = false;             ///< whether the data survives the process
        bool read_only = false;              ///< whether writes are refused (`kv_errc::read_only`)
    };

    /// \brief A store that reports its health: `s.ping() -> result<void, kv_errc>`, `s.diagnostics() -> result<kv_diagnostics, kv_errc>`.
    /// \ingroup core
    /// \tparam S the store type.
    template <typename S>
    concept kv_health = requires(const S& s) {
        { s.ping() } -> std::same_as<result<void, kv_errc>>;
        { s.diagnostics() } -> std::same_as<result<kv_diagnostics, kv_errc>>;
    };
}

#endif /* snicholls_core_concepts_kv_health_hpp */
