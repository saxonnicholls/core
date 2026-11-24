//
//  persist.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `fsm::save` and `fsm::load`: a machine's states through any `key_value_store`, versioned, with
/// compare-and-set so that a stale writer is refused.
/// \ingroup core
///
/// \overview A payment's machine must survive the process that runs it. `save(store, key, machine, version)` writes
/// every region's state (and every sub-machine's) with the next version; `load(store, key, machine)` sets the machine
/// back and answers the version it read. A writer that loaded version `n` and saves while another has already saved
/// `n + 1` gets `fsm_errc::stale` and nothing is written: it reloads and retries, as with any optimistic lock.
///
/// \snippet fsm/persist_demo.cpp save_load
///
/// The record is text, one line each, readable in any store browser:
///
///     snicholls.fsm.v1
///     machine lifecycle
///     version 3
///     regions 1
///     3 settled
///
/// Each region is a number and a name: the enumerator's value where the machine has a state enumeration (so a record
/// stays valid when enumerators are appended, or renamed), the state's name where it has not (so it stays valid when
/// states are added or reordered).
///
/// The compare-and-set is built from the contract's own atomic primitives, so every store has it: a save is one batch
/// of `insert(key + "/v/" + version)` (which fails if that version exists) and `put(key)`. The versioned keys are the
/// machine's history, one record per save; `load_version` reads one back.
///
/// Use it wherever a machine's state must outlive a process. Not a replacement for storing the state on the object
/// itself (`current()` is one byte); this is for the machine, and for an audit trail of its states.
///
/// \relations `fsm/state_machine.hpp` (`snapshot`, `restore_snapshot`, `wire_state`), `concepts/key_value_store.hpp`
/// (`kv_readable`, `kv_writable`, `kv_errc`), `interfaces/key_value_store_reference.hpp` (the reference store, in the
/// tests and demos). Group: \ref core.
///
/// \impls One format, `snicholls.fsm.v1`. A new format gets a new name; this one never changes meaning.
///
/// \contracts
/// - `save` is all or nothing (one `apply`): on any error nothing is written.
/// - `load` changes the machine only when it succeeds (strong guarantee); no entry or exit action runs.
/// - Versions start at 1; `save(..., 0)` creates. Passing a version other than the one last loaded or saved is a
///   precondition violation (it could create a gap in the history).
/// - No exception crosses these functions except `std::bad_alloc` from building the record and the batch.
///
/// \errors `fsm_errc::stale` (that version exists: another writer saved first), `not_found` (nothing under the key),
/// `corrupt` (the record is not this format), `shape_mismatch` (a different number of regions), `unknown_state` (a saved
/// state is not a state of the table), `store_unavailable` (`kv_errc::unavailable`), `store_failed` (any other
/// `kv_errc`).
///
/// \examples The payment demo, `fsm/lifecycle_payment_demo.cpp`, crashes a process and restores the payment, and shows
/// a stale writer refused.
///
/// \perf Not benchmarked: a save is one store batch and a record of a few dozen bytes; the store's cost dominates.
///
/// \testing `tests/fsm/persist_test.cpp` (round trips, the property that a restored machine behaves as the one never
/// saved, every error); `demos/fsm/persist_demo.cpp`.
///
/// \deps `<array>`, `<charconv>`, `<cstddef>`, `<cstdint>`, `<string>`, `<string_view>`, `<system_error>`,
/// and core's `fsm/state_machine.hpp`, `concepts/key_value_store.hpp` (which brings `hash/sha2.hpp`).
///
/// \gotchas A machine's history keys sit under its key: a prefix scan of `key` sees them (`key/v/...`).
///
/// \status Experimental, new in 0.24.0.

#ifndef snicholls_core_fsm_persist_hpp
#define snicholls_core_fsm_persist_hpp

#include <snicholls/core/concepts/key_value_store.hpp>
#include <snicholls/core/fsm/state_machine.hpp>

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <system_error>

namespace snicholls::fsm {

    /// \brief The name of the record format, its first line.
    /// \ingroup core
    inline constexpr std::string_view persist_format = "snicholls.fsm.v1";

    namespace detail {
        inline fsm_errc from_kv(kv_errc e) noexcept
        {
            return e == kv_errc::unavailable ? fsm_errc::store_unavailable : fsm_errc::store_failed;
        }

        inline void append_number(std::string& out, std::uint64_t v)
        {
            std::array<char, 24> digits{};
            const auto r = std::to_chars(digits.data(), digits.data() + digits.size(), v);
            out.append(digits.data(), r.ptr);
        }
        inline void append_number(std::string& out, std::int64_t v)
        {
            std::array<char, 24> digits{};
            const auto r = std::to_chars(digits.data(), digits.data() + digits.size(), v);
            out.append(digits.data(), r.ptr);
        }

        // One line of the record; empty and advances to the end when there is none.
        inline std::string_view next_line(std::string_view& text) noexcept
        {
            const std::size_t end = text.find('\n');
            const std::string_view line = text.substr(0, end);
            text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
            return line;
        }

        template <class T>
        bool parse_number(std::string_view text, T& out) noexcept
        {
            const auto r = std::from_chars(text.data(), text.data() + text.size(), out);
            return r.ec == std::errc{} && r.ptr == text.data() + text.size();
        }

        inline bool parse_field(std::string_view line, std::string_view name, std::string_view& value) noexcept
        {
            if (line.size() <= name.size() || line.substr(0, name.size()) != name || line[name.size()] != ' ')
                return false;
            value = line.substr(name.size() + 1);
            return true;
        }
    } // namespace detail

    /// \brief The key under which version `version` of the record at `key` is kept: `key + "/v/"` and 20 digits.
    /// \ingroup core
    /// \param key     the machine's key.
    /// \param version the version.
    /// \return the versioned key.
    [[nodiscard]] inline std::string version_key(std::string_view key, std::uint64_t version)
    {
        std::string out(key);
        out += "/v/";
        std::array<char, 20> digits{};
        for (auto i = digits.size(); i-- > 0; version /= 10)
            digits[i] = static_cast<char>('0' + version % 10);
        out.append(digits.data(), digits.size());
        return out;
    }

    /// \brief The record of a machine's states at a version, in the `snicholls.fsm.v1` format.
    /// \ingroup core
    /// \param machine the machine.
    /// \param version the version to write into the record.
    /// \return the record.
    /// \throws std::bad_alloc
    template <class SM>
    [[nodiscard]] std::string to_record(const SM& machine, std::uint64_t version)
    {
        std::string out(persist_format);
        out += "\nmachine ";
        out += detail::short_name(type_name<typename SM::definition>());
        out += "\nversion ";
        detail::append_number(out, version);
        out += "\nregions ";
        detail::append_number(out, static_cast<std::uint64_t>(SM::slot_count));
        for (const auto& slot : machine.snapshot()) {
            out += '\n';
            detail::append_number(out, slot.value);
            out += ' ';
            out += slot.name;
        }
        out += '\n';
        return out;
    }

    /// \brief Sets a machine from a record, as `restore_snapshot` does.
    /// \ingroup core
    /// \param record  the record.
    /// \param machine the machine; unchanged on error.
    /// \return the record's version, or `fsm_errc::corrupt`, `shape_mismatch` or `unknown_state`.
    template <class SM>
    result<std::uint64_t, fsm_errc> from_record(std::string_view record, SM& machine) noexcept
    {
        std::string_view rest = record;
        std::string_view field;
        std::uint64_t version = 0;
        std::size_t regions = 0;
        if (detail::next_line(rest) != persist_format || !detail::parse_field(detail::next_line(rest), "machine", field) ||
            !detail::parse_field(detail::next_line(rest), "version", field) || !detail::parse_number(field, version) ||
            !detail::parse_field(detail::next_line(rest), "regions", field) || !detail::parse_number(field, regions))
            return err(fsm_errc::corrupt);
        if (regions != SM::slot_count)
            return err(fsm_errc::shape_mismatch);
        std::array<wire_state, SM::slot_count> slots{};
        for (auto& slot : slots) {
            const std::string_view line = detail::next_line(rest);
            const std::size_t space = line.find(' ');
            if (space == std::string_view::npos || !detail::parse_number(line.substr(0, space), slot.value))
                return err(fsm_errc::corrupt);
            slot.name = line.substr(space + 1);
        }
        if (!rest.empty())
            return err(fsm_errc::corrupt);
        const auto restored = machine.restore_snapshot(slots);
        if (restored.is_err())
            return err(restored.unwrap_err_unchecked());
        return ok(version);
    }

    /// \brief Saves a machine's states under `key` as version `expected_version + 1`, unless another writer saved that
    /// version first.
    /// \ingroup core
    /// \param store            the store.
    /// \param key              the machine's key.
    /// \param machine          the machine.
    /// \param expected_version the version this writer last loaded or saved; 0 to create.
    /// \return the version written, or `fsm_errc::stale`, `store_unavailable`, `store_failed`; nothing written on error.
    /// \throws std::bad_alloc
    template <kv_writable Store, kv_byte_source K, class SM>
    result<std::uint64_t, fsm_errc> save(Store& store, const K& key, const SM& machine, std::uint64_t expected_version)
    {
        const std::uint64_t version = expected_version + 1;
        const kv_bytes key_bytes = kv_bytes_of(key);
        const std::string_view key_text(reinterpret_cast<const char*>(key_bytes.data()), key_bytes.size());
        const std::string record = to_record(machine, version);
        kv_write_batch batch;
        batch.insert(version_key(key_text, version), record);
        batch.put(key_text, record);
        const auto applied = store.apply(batch);
        if (applied.is_err())
            return err(applied.unwrap_err_unchecked() == kv_errc::conflict ? fsm_errc::stale
                                                                          : detail::from_kv(applied.unwrap_err_unchecked()));
        return ok(version);
    }

    /// \brief Loads a machine's states from `key`.
    /// \ingroup core
    /// \param store   the store.
    /// \param key     the machine's key.
    /// \param machine the machine; unchanged on error, and no action runs.
    /// \return the version loaded, or `fsm_errc::not_found`, `corrupt`, `shape_mismatch`, `unknown_state`,
    /// `store_unavailable`, `store_failed`.
    /// \throws std::bad_alloc
    template <kv_readable Store, kv_byte_source K, class SM>
    result<std::uint64_t, fsm_errc> load(const Store& store, const K& key, SM& machine)
    {
        std::string record;
        const auto found = kv_get(store, key, record);
        if (found.is_err())
            return err(detail::from_kv(found.unwrap_err_unchecked()));
        if (!*found)
            return err(fsm_errc::not_found);
        return from_record(record, machine);
    }

    /// \brief Loads the states a machine had at an earlier version, from its history.
    /// \ingroup core
    /// \param store   the store.
    /// \param key     the machine's key.
    /// \param version the version.
    /// \param machine the machine; unchanged on error.
    /// \return the version loaded, or the errors of `load`.
    /// \throws std::bad_alloc
    template <kv_readable Store, class SM>
    result<std::uint64_t, fsm_errc> load_version(const Store& store, std::string_view key, std::uint64_t version, SM& machine)
    {
        return load(store, version_key(key, version), machine);
    }

} // namespace snicholls::fsm

#endif /* snicholls_core_fsm_persist_hpp */
