//
//  kv_backup.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The checkpoint and consistency check of asoMonoRepo1's BackupInterface, as a capability of its own, with results
//  as plain data instead of JSON. Its named snapshots, JSON serialisation, WAL positions and backup policy are not
//  taken: see the file documentation.
//

/// \file
/// \brief The backup capability of a store, static form: `kv_backup<S>` (`checkpoint`, `verify`) and its reports.
/// \ingroup core
///
/// **Overview.** Two operations a store with durable data offers to the process that guards it:
/// `checkpoint(destination)` makes a consistent copy of the whole store at `destination` while writes go on
/// (RocksDB's `Checkpoint`, SQLite's `VACUUM INTO`, a file copy), and `verify()` checks the stored data
/// (checksums, structure, order) and reports what it found. A run that must be reproduced against a state that
/// has since changed records the checkpoint's `sequence` beside the store's `kv_digest_v1`.
///
/// \snippet concepts/kv_backup_demo.cpp backup
///
/// What a backup is *for* stays above core: schedules and retention (asoMonoRepo1's `BackupPolicy`), uploading a
/// checkpoint elsewhere, restoring one (opening a store at that destination is a backend's constructor). Logical
/// point-in-time views are `snapshot()` of the store contract. Dumping to JSON is a scan plus a codec, above core.
///
/// **Relationships.** The virtual twin is `kv_backup_interface` (`interfaces/kv_backup.hpp`), with adapters both
/// ways; the reference implementation is `reference_key_value_store`, which keeps checkpoints in memory; the kit is
/// `run_kv_backup_conformance` (`interfaces/key_value_store_conformance.hpp`). Group: \ref core.
///
/// **Implementations / variants.** `reference_key_value_store`; the backends above core.
///
/// **Guarantees.** A checkpoint is consistent: it holds the state of one moment, whose sequence it reports.
/// Checkpointing changes nothing in the store. A destination that already holds a checkpoint is refused, never
/// overwritten. `verify()` changes nothing. No exception crosses either.
///
/// **Errors.** `kv_errc::invalid_argument` for an empty destination or one that exists; `io`, `unavailable` and
/// the rest for storage failures. A `verify()` that ran and found damage is a success whose report has
/// `problems > 0`; an error means the check itself could not run.
///
/// **Examples.** See the overview.
///
/// **Performance.** Not benchmarked: backend-specific, from O(1) (hard links) to a full copy.
///
/// **Tests and demos.** `tests/concepts/kv_backup_test.cpp`, `demos/concepts/kv_backup_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstdint>`, `<optional>`, `<string_view>`, and
/// `concepts/key_value_store.hpp`.
///
/// **Known issues.** The destination is text whose meaning is the backend's (a directory, a file, a name).
///
/// **Status.** New in 0.22.0. Experimental.

#ifndef snicholls_core_concepts_kv_backup_hpp
#define snicholls_core_concepts_kv_backup_hpp

#include <snicholls/core/concepts/key_value_store.hpp>

#include <concepts>
#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls {

    /// \brief What a checkpoint captured.
    /// \ingroup core
    struct kv_checkpoint_info {
        std::uint64_t sequence = 0;          ///< the store's `sequence()` at the moment the checkpoint holds
        std::optional<std::uint64_t> bytes{}; ///< its size, when the backend knows it
    };

    /// \brief What a consistency check found.
    /// \ingroup core
    struct kv_verify_report {
        std::uint64_t keys_checked = 0;  ///< how many entries were checked
        std::uint64_t problems = 0;      ///< how many failed a check

        /// \brief Whether nothing failed.
        /// \return `problems == 0`.
        [[nodiscard]] constexpr bool ok() const noexcept { return problems == 0; }
    };

    /// \brief A store that checkpoints and verifies: `s.checkpoint(destination)`, `s.verify()` (const).
    /// \ingroup core
    ///
    /// - `s.checkpoint(std::string_view destination) -> result<kv_checkpoint_info, kv_errc>`;
    /// - `s.verify() -> result<kv_verify_report, kv_errc>`, on a const store.
    /// \tparam S the store type.
    template <typename S>
    concept kv_backup = requires(S& s, const S& cs, std::string_view destination) {
        { s.checkpoint(destination) } -> std::same_as<result<kv_checkpoint_info, kv_errc>>;
        { cs.verify() } -> std::same_as<result<kv_verify_report, kv_errc>>;
    };
}

#endif /* snicholls_core_concepts_kv_backup_hpp */
