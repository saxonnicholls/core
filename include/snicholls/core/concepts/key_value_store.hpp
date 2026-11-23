//
//  key_value_store.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The contract of crtp's key_value_store_interface (get / read, put, insert, contains, erase, erase of a range,
//  scans, prefix scans, atomic batches), with errors returned through snicholls::result instead of thrown, ranges
//  that view their bounds instead of owning them, and the snapshot, sequence and digest of crtp's KV_PLAN D20.
//

/// \file
/// \brief The key-value store contract, static form: `key_value_store<S>`, its ranges, batches, errors and digest.
/// \ingroup core
///
/// **Overview.** An ordered map of byte strings to byte strings, as a concept. Storage is chosen by the
/// composition root (in memory, a file, SQL, RocksDB); code written against `key_value_store<S>` runs on any of
/// them, and inlines on the in-memory one. Keys and values are `std::span<const std::byte>` (`kv_bytes`); the
/// free functions `kv_get`, `kv_put`, `kv_insert`, `kv_erase`, `kv_contains`, `kv_scan`, `kv_scan_prefix` and
/// `kv_count` accept any one-byte sequence as well (`std::string_view`, `std::string`, literals, byte vectors).
///
/// \snippet concepts/key_value_store_demo.cpp contract
///
/// The contract is three concepts, each the one before plus operations:
/// - `kv_readable<S>`: `read` (zero-copy, a visitor sees the value), `contains`, `scan` (a visitor sees each
///   entry of a `kv_range` in key order and may stop), `sequence`;
/// - `kv_writable<S>`: plus `put`, `insert` (only when absent: the primitive a ledger appends with), `erase`,
///   `erase_range`, `apply` (a batch of `kv_write`s, all or nothing);
/// - `key_value_store<S>`: plus `snapshot()`, a read-only view of one consistent state.
///
/// `kv_digest_v1(store, range)` is the named, stable digest of a range (KV_PLAN D20): a run records
/// (sequence, digest) and a later run verifies it reads the same bytes on any backend.
///
/// Use it for keyed persistence with ordered scans. Not for relational queries (an SQL layer), nor for typed
/// objects (a codec layered on top, as crtp's `object_store`).
///
/// **Relationships.** The virtual twin is `key_value_store_interface` (`interfaces/key_value_store.hpp`), with
/// adapters both ways; the reference implementation and test fake is `reference_key_value_store`
/// (`interfaces/key_value_store_reference.hpp`); the conformance kit is `run_kv_conformance`
/// (`interfaces/key_value_store_conformance.hpp`). Separate capabilities: `kv_health` (`concepts/kv_health.hpp`),
/// `kv_backup` (`concepts/kv_backup.hpp`). Errors are `snicholls::result<V, kv_errc>` (`result.hpp`); visitors
/// are `snicholls::function_ref` (`function_ref.hpp`); the digest is SHA-256 (`hash/sha2.hpp`). Group: \ref core.
///
/// **Implementations / variants.** In core, `reference_key_value_store` (a `std::map` with a byte-wise
/// comparator). Above core: crtp's in-memory, JSON, SQL, RocksDB and base-encode-decode backends; the database
/// backends of the products. Each documents its own complexity and thread safety.
///
/// **Guarantees.** Every implementation promises, and the conformance kit checks:
/// - **Order:** keys are ordered by unsigned byte-wise comparison, on every backend, with no custom comparator.
///   Empty keys, empty values, embedded NULs and bytes >= 0x80 are all valid.
/// - **Missing is not an error:** `read`, `contains`, `insert` and `erase` answer `false` for an absent key; a
///   storage failure is an error (`kv_errc`), never `false` or an empty value.
/// - **Atomicity:** every call is atomic; `scan` sees one consistent state; `apply` is all or nothing, and a batch
///   that fails (an `insert` whose key is present, a storage failure) changes nothing, the sequence included.
/// - **Sequence:** `sequence()` never decreases, never moves on a read or on a call that changed nothing, and
///   increases with every call that changed the store.
/// - **Snapshot:** reads of a snapshot see the state at the moment it was taken, whatever is written later.
/// - **Exceptions:** no exception of the backend crosses the contract: failures are `kv_errc` values. The only
///   exceptions a call lets out are the caller's own visitor's, unchanged, and the store stays usable after one.
/// - **Lifetime:** bytes passed to a visitor are valid only during the call; a visitor must not call back into the
///   store (copy out what you need).
/// - `kv_range`, `kv_write` and the comparisons here are `constexpr` and `noexcept`. Nothing in the contract
///   forces an allocation: reads and scans hand out views to a visitor.
///
/// **Errors.** `kv_errc`: `unavailable`, `io`, `corrupt`, `read_only`, `conflict`, `limit`, `invalid_argument`,
/// `out_of_memory`, `backend`. Nothing here throws, except `kv_write_batch`, which may throw `std::bad_alloc`
/// while it is being built.
///
/// **Examples.** \snippet concepts/key_value_store_demo.cpp scan
///
/// **Performance.** The contract costs nothing on the static path; the virtual path's cost on the in-memory
/// reference is measured in `benchmarks/interfaces/key_value_store_benchmark.cpp` (BENCHMARKS.md).
///
/// **Tests and demos.** `tests/concepts/key_value_store_test.cpp`, `demos/concepts/key_value_store_demo.cpp`;
/// the kit through both forms in `tests/interfaces/key_value_store_test.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<memory>`,
/// `<optional>`, `<span>`, `<string>`, `<string_view>`, `<type_traits>`, `<utility>`, `<vector>`, and core's
/// `function_ref.hpp`, `result.hpp`, `reflect/string_access.hpp`, `hash/sha2.hpp` (for the digest).
///
/// **Known issues.** `kv_digest_v1` is named `crtp.kv.digest.v1` on the wire, the name agreed with its first
/// consumers before it moved into core; the name's meaning never changes, and a new definition gets a new name.
/// Thread safety is a property of each implementation: the reference is not synchronised.
///
/// **Status.** New in 0.22.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_key_value_store_hpp
#define snicholls_core_concepts_key_value_store_hpp

#include <snicholls/core/function_ref.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/reflect/string_access.hpp>
#include <snicholls/core/result.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief A key or a value: read-only bytes, viewed, never owned.
    /// \ingroup core
    using kv_bytes = std::span<const std::byte>;

    /// \brief Why a key-value store operation failed. A missing key is never one of these.
    /// \ingroup core
    enum class kv_errc : std::uint8_t {
        unavailable = 1,   ///< the backend cannot be reached: not opened, closed, connection lost
        io,                ///< reading or writing the medium failed
        corrupt,           ///< stored data failed a check
        read_only,         ///< a write to a store opened read-only
        conflict,          ///< a batch's `insert` found its key present, or a concurrent writer won; nothing applied
        limit,             ///< a key, value or batch the backend cannot represent (size, encoding)
        invalid_argument,  ///< an argument the operation rejects (a checkpoint destination that exists)
        out_of_memory,     ///< the backend could not allocate
        backend            ///< any other failure the backend reported, translated (a third-party exception)
    };

    /// \brief Any argument the convenience functions take as a key or a value: a one-byte sequence (`byte_viewable`).
    /// \ingroup core
    /// \tparam S the type.
    template <typename S>
    concept kv_byte_source = byte_viewable<S>;

    /// \brief The bytes of `s`, zero-copy: a string's code units, a byte sequence's bytes (a literal's length is N - 1).
    /// \ingroup core
    /// \tparam S a `kv_byte_source`.
    /// \param s the key or value.
    /// \return a view of its bytes.
    template <kv_byte_source S>
    [[nodiscard]] constexpr kv_bytes kv_bytes_of(const S& s) noexcept
    {
        return as_bytes_view(s);
    }

    /// \brief Unsigned byte-wise order, the order of every store: `a < b`.
    /// \ingroup core
    /// \param a the left key.
    /// \param b the right key.
    /// \return whether `a` sorts before `b`.
    [[nodiscard]] constexpr bool kv_less(kv_bytes a, kv_bytes b) noexcept
    {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    }

    /// \brief Byte-wise equality of two keys or values.
    /// \ingroup core
    /// \param a the left bytes.
    /// \param b the right bytes.
    /// \return whether they are the same bytes.
    [[nodiscard]] constexpr bool kv_equal(kv_bytes a, kv_bytes b) noexcept
    {
        return std::equal(a.begin(), a.end(), b.begin(), b.end());
    }

    /// \brief Whether `key` starts with `prefix` (every key starts with the empty prefix).
    /// \ingroup core
    /// \param key the key.
    /// \param prefix the prefix.
    /// \return whether the first bytes of `key` are `prefix`.
    [[nodiscard]] constexpr bool kv_starts_with(kv_bytes key, kv_bytes prefix) noexcept
    {
        return key.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), key.begin());
    }

    /// \brief The keys a scan or a range erase covers: a lower bound, an optional exclusive upper bound, a prefix.
    /// \ingroup core
    ///
    /// Views its bounds (keep them alive while the range is used); allocates nothing, so a prefix range has no
    /// computed upper bound: a backend that needs one computes it. Built by `all()`, `from()`, `between()` and
    /// `prefixed()`; `after(key)` continues a range after a key already seen, which is how a scan is chunked.
    /// \invariant `lower()` starts with `prefix()` or sorts after it, so an ascending walk from `lower()` stops at
    /// the first key that `beyond()` reports.
    class kv_range {
    public:
        /// \brief Every key.
        /// \return the unbounded range.
        [[nodiscard]] static constexpr kv_range all() noexcept { return kv_range{}; }

        /// \brief Keys `>= first`.
        /// \param first the inclusive lower bound.
        /// \return the range.
        [[nodiscard]] static constexpr kv_range from(kv_bytes first) noexcept
        {
            kv_range r;
            r.lower_ = first;
            return r;
        }

        /// \brief Keys `>= first` and `< last`; empty when `last <= first`.
        /// \param first the inclusive lower bound.
        /// \param last the exclusive upper bound.
        /// \return the range.
        [[nodiscard]] static constexpr kv_range between(kv_bytes first, kv_bytes last) noexcept
        {
            kv_range r;
            r.lower_ = first;
            r.upper_ = last;
            r.bounded_ = true;
            return r;
        }

        /// \brief Keys that start with `prefix`: exactly its keys, never a substring match.
        /// \param prefix the prefix; empty means every key.
        /// \return the range.
        [[nodiscard]] static constexpr kv_range prefixed(kv_bytes prefix) noexcept
        {
            kv_range r;
            r.lower_ = prefix;
            r.prefix_ = prefix;
            return r;
        }

        /// \brief The rest of this range after `key`, exclusive: resumes a scan stopped at `key`.
        /// \param key a key of this range (the last one a chunk saw).
        /// \return the range of the keys of `*this` that sort after `key`.
        /// \pre `contains(key)`.
        [[nodiscard]] constexpr kv_range after(kv_bytes key) const noexcept
        {
            kv_range r = *this;
            r.lower_ = key;
            r.lower_open_ = true;
            return r;
        }

        /// \brief The lower bound (empty for `all()`).
        /// \return a view of it.
        [[nodiscard]] constexpr kv_bytes lower() const noexcept { return lower_; }
        /// \brief Whether the lower bound is exclusive (a range made by `after()`).
        /// \return `true` when `lower()` itself is outside the range.
        [[nodiscard]] constexpr bool lower_open() const noexcept { return lower_open_; }
        /// \brief Whether the range has an exclusive upper bound.
        /// \return `true` for `between()` and what `after()` makes of it.
        [[nodiscard]] constexpr bool bounded() const noexcept { return bounded_; }
        /// \brief The exclusive upper bound; meaningful when `bounded()`.
        /// \return a view of it.
        [[nodiscard]] constexpr kv_bytes upper() const noexcept { return upper_; }
        /// \brief The prefix every key of the range starts with (empty: no constraint).
        /// \return a view of it.
        [[nodiscard]] constexpr kv_bytes prefix() const noexcept { return prefix_; }

        /// \brief Whether `key` lies in the range.
        /// \param key the key.
        /// \return `true` when it is at or after the lower bound, before the upper and starts with the prefix.
        [[nodiscard]] constexpr bool contains(kv_bytes key) const noexcept
        {
            if (lower_open_ ? !kv_less(lower_, key) : kv_less(key, lower_))
                return false;
            return !beyond(key);
        }

        /// \brief For a key at or after `lower()`: whether it, and every key after it, is outside the range.
        /// \param key a key that sorts at or after `lower()`.
        /// \return `true` when an ascending walk can stop at `key`.
        [[nodiscard]] constexpr bool beyond(kv_bytes key) const noexcept
        {
            return !kv_starts_with(key, prefix_) || (bounded_ && !kv_less(key, upper_));
        }

        /// \brief True when the bounds alone prove the range empty (`upper() <= lower()`).
        /// \return whether no key can be in the range.
        [[nodiscard]] constexpr bool empty() const noexcept { return bounded_ && !kv_less(lower_, upper_); }

    private:
        kv_bytes lower_{};
        kv_bytes upper_{};
        kv_bytes prefix_{};
        bool lower_open_ = false;
        bool bounded_ = false;
    };

    /// \brief The kind of one write of a batch.
    /// \ingroup core
    enum class kv_op : std::uint8_t {
        put,     ///< insert or overwrite
        insert,  ///< insert only if absent; a present key fails the whole batch with `kv_errc::conflict`
        erase    ///< remove; an absent key is ignored
    };

    /// \brief One write of a batch: an operation, a key, a value (empty for an erase). Views its bytes.
    /// \ingroup core
    struct kv_write {
        kv_op op = kv_op::put;  ///< the operation
        kv_bytes key{};         ///< the key
        kv_bytes value{};       ///< the value; ignored for an erase

        /// \brief An insert-or-overwrite.
        /// \param key the key.
        /// \param value the value.
        /// \return the write.
        [[nodiscard]] static constexpr kv_write put(kv_bytes key, kv_bytes value) noexcept { return {kv_op::put, key, value}; }
        /// \brief An insert of an absent key; the batch fails if the key is present when this write is reached.
        /// \param key the key.
        /// \param value the value.
        /// \return the write.
        [[nodiscard]] static constexpr kv_write insert(kv_bytes key, kv_bytes value) noexcept { return {kv_op::insert, key, value}; }
        /// \brief A removal.
        /// \param key the key.
        /// \return the write.
        [[nodiscard]] static constexpr kv_write erase(kv_bytes key) noexcept { return {kv_op::erase, key, {}}; }
    };

    /// \brief An owning list of writes, built from temporaries and kept; passes to `apply()` as `std::span<const kv_write>`.
    /// \ingroup core
    ///
    /// Copies each key and value once, into storage that never moves, so the views it hands out stay valid until it
    /// is cleared or destroyed. A plain value type: move-only, not synchronised. For a batch whose bytes already
    /// live elsewhere, a `std::array<kv_write, N>` needs no allocation at all.
    class kv_write_batch {
    public:
        /// \brief Appends an insert-or-overwrite.
        /// \tparam K a `kv_byte_source`.
        /// \tparam V a `kv_byte_source`.
        /// \param key the key (copied).
        /// \param value the value (copied).
        /// \return `*this`.
        /// \throws std::bad_alloc; the batch is unchanged.
        template <kv_byte_source K, kv_byte_source V>
        kv_write_batch& put(const K& key, const V& value) { return add(kv_op::put, kv_bytes_of(key), kv_bytes_of(value)); }

        /// \brief Appends an insert of an absent key (a present key fails the whole batch).
        /// \tparam K a `kv_byte_source`.
        /// \tparam V a `kv_byte_source`.
        /// \param key the key (copied).
        /// \param value the value (copied).
        /// \return `*this`.
        /// \throws std::bad_alloc; the batch is unchanged.
        template <kv_byte_source K, kv_byte_source V>
        kv_write_batch& insert(const K& key, const V& value) { return add(kv_op::insert, kv_bytes_of(key), kv_bytes_of(value)); }

        /// \brief Appends a removal.
        /// \tparam K a `kv_byte_source`.
        /// \param key the key (copied).
        /// \return `*this`.
        /// \throws std::bad_alloc; the batch is unchanged.
        template <kv_byte_source K>
        kv_write_batch& erase(const K& key) { return add(kv_op::erase, kv_bytes_of(key), {}); }

        /// \brief The writes, in the order they apply.
        /// \return a view, valid until the batch is next modified.
        [[nodiscard]] std::span<const kv_write> writes() const noexcept { return writes_; }
        /// \brief The writes, so a batch passes straight to `apply()`.
        /// \return `writes()`.
        operator std::span<const kv_write>() const noexcept { return writes_; }   // NOLINT(google-explicit-constructor)
        /// \brief The number of writes.
        /// \return the count.
        [[nodiscard]] std::size_t size() const noexcept { return writes_.size(); }
        /// \brief Whether there are no writes (applying it changes nothing).
        /// \return `size() == 0`.
        [[nodiscard]] bool empty() const noexcept { return writes_.empty(); }
        /// \brief Removes every write.
        void clear() noexcept
        {
            writes_.clear();
            storage_.clear();
        }

    private:
        kv_write_batch& add(kv_op op, kv_bytes key, kv_bytes value)
        {
            auto block = std::make_unique<std::byte[]>(key.size() + value.size() + 1);   // + 1: never a zero-length array
            std::copy(key.begin(), key.end(), block.get());
            std::copy(value.begin(), value.end(), block.get() + key.size());
            writes_.reserve(writes_.size() + 1);
            storage_.reserve(storage_.size() + 1);
            writes_.push_back({op, kv_bytes(block.get(), key.size()), kv_bytes(block.get() + key.size(), value.size())});
            storage_.push_back(std::move(block));   // cannot throw: room reserved
            return *this;
        }

        std::vector<kv_write> writes_;
        std::vector<std::unique_ptr<std::byte[]>> storage_;
    };

    /// \brief The visitor of `read()`: `fn(value)`, the stored bytes valid only during the call.
    /// \ingroup core
    using kv_read_visitor = function_ref<void(kv_bytes)>;

    /// \brief The visitor of `scan()`: `fn(key, value) -> continue?`, both valid only during the call.
    /// \ingroup core
    using kv_scan_visitor = function_ref<bool(kv_bytes, kv_bytes)>;

    /// \brief The read operations: `read`, `contains`, `scan`, `sequence`, on a const store, returning `result`.
    /// \ingroup core
    ///
    /// - `s.read(key, fn) -> result<bool, kv_errc>`: calls `fn(value)` and answers `true`, or `false` without calling.
    /// - `s.contains(key) -> result<bool, kv_errc>`.
    /// - `s.scan(range, fn) -> result<std::size_t, kv_errc>`: `fn(key, value)` in ascending key order until it
    ///   answers `false`; the count of entries passed to `fn`, the one that stopped it included.
    /// - `s.sequence() -> result<std::uint64_t, kv_errc>`: see the file's guarantees.
    /// \tparam S the store type.
    template <typename S>
    concept kv_readable = requires(const S& s, kv_bytes key, const kv_range& range, kv_read_visitor rv, kv_scan_visitor sv) {
        { s.read(key, rv) } -> std::same_as<result<bool, kv_errc>>;
        { s.contains(key) } -> std::same_as<result<bool, kv_errc>>;
        { s.scan(range, sv) } -> std::same_as<result<std::size_t, kv_errc>>;
        { s.sequence() } -> std::same_as<result<std::uint64_t, kv_errc>>;
    };

    /// \brief `kv_readable` plus the writes: `put`, `insert`, `erase`, `erase_range`, `apply`.
    /// \ingroup core
    ///
    /// - `s.put(key, value) -> result<void, kv_errc>`: insert or overwrite.
    /// - `s.insert(key, value) -> result<bool, kv_errc>`: `true` if inserted; `false` if the key existed, which is
    ///   then left exactly as it was (value and sequence).
    /// - `s.erase(key) -> result<bool, kv_errc>`: whether the key existed.
    /// - `s.erase_range(range) -> result<std::size_t, kv_errc>`: how many keys were removed.
    /// - `s.apply(writes) -> result<void, kv_errc>`: every write in order, all or nothing.
    /// \tparam S the store type.
    template <typename S>
    concept kv_writable = kv_readable<S> &&
        requires(S& s, kv_bytes key, kv_bytes value, const kv_range& range, std::span<const kv_write> batch) {
            { s.put(key, value) } -> std::same_as<result<void, kv_errc>>;
            { s.insert(key, value) } -> std::same_as<result<bool, kv_errc>>;
            { s.erase(key) } -> std::same_as<result<bool, kv_errc>>;
            { s.erase_range(range) } -> std::same_as<result<std::size_t, kv_errc>>;
            { s.apply(batch) } -> std::same_as<result<void, kv_errc>>;
        };

    namespace detail {
        // A snapshot handle: a readable store by value, or something that dereferences to one (std::unique_ptr).
        template <typename P>
        concept kv_snapshot_handle = kv_readable<P> || requires(const P& p) {
            requires kv_readable<std::remove_cvref_t<decltype(*p)>>;
        };

        template <typename R>
        inline constexpr bool kv_snapshot_result_v = false;
        template <typename V>
        inline constexpr bool kv_snapshot_result_v<result<V, kv_errc>> = kv_snapshot_handle<V>;

        // The store a handle designates: the object itself, or what it points to.
        template <typename T>
            requires kv_readable<std::remove_cvref_t<T>>
        constexpr T& kv_target(T& t) noexcept
        {
            return t;
        }
        template <typename P>
            requires(!kv_readable<std::remove_cvref_t<P>>) && requires(P& p) { *p; }
        constexpr auto& kv_target(P& p) noexcept
        {
            return *p;
        }
    }

    /// \brief The full contract: `kv_writable` plus `s.snapshot() -> result<H, kv_errc>`, a read-only consistent view.
    /// \ingroup core
    ///
    /// `H` is a `kv_readable` value, or a pointer-like handle to one (`std::unique_ptr<kv_reader_interface>` in the
    /// virtual form). Its reads, `sequence()` included, see the state when it was taken, for as long as it lives.
    /// \tparam S the store type.
    template <typename S>
    concept key_value_store = kv_writable<S> && requires(const S& s) {
        requires detail::kv_snapshot_result_v<decltype(s.snapshot())>;
    };

    /// \brief Calls `fn(value)` with the value of `key`, zero-copy.
    /// \ingroup core
    /// \tparam S a `kv_readable` store.
    /// \tparam K a `kv_byte_source`.
    /// \tparam F callable as `void(kv_bytes)`.
    /// \param store the store.
    /// \param key the key.
    /// \param fn the visitor; must not call back into the store.
    /// \return whether the key was found, or the storage error.
    /// \throws whatever `fn` throws, unchanged.
    template <kv_readable S, kv_byte_source K, typename F>
        requires std::is_invocable_v<F&, kv_bytes>
    result<bool, kv_errc> kv_read(const S& store, const K& key, F&& fn)
    {
        return store.read(kv_bytes_of(key), kv_read_visitor(fn));
    }

    /// \brief Copies the value of `key` into `out` (a `std::string` or `std::vector<std::byte>`), reusing its capacity.
    /// \ingroup core
    /// \tparam S a `kv_readable` store.
    /// \tparam K a `kv_byte_source`.
    /// \tparam Out `std::string` or `std::vector<std::byte>`.
    /// \param store the store.
    /// \param key the key.
    /// \param out receives the value; unchanged when the key is absent.
    /// \return whether the key was found, or the storage error.
    /// \throws std::bad_alloc when `out` cannot grow.
    template <kv_readable S, kv_byte_source K, typename Out>
        requires std::same_as<Out, std::string> || std::same_as<Out, std::vector<std::byte>>
    result<bool, kv_errc> kv_get(const S& store, const K& key, Out& out)
    {
        return store.read(kv_bytes_of(key), [&](kv_bytes v) {
            using unit = typename Out::value_type;
            const auto* p = reinterpret_cast<const unit*>(v.data());
            out.assign(p, p + v.size());
        });
    }

    /// \brief A copy of the value of `key` as text, or `std::nullopt` when absent.
    /// \ingroup core
    /// \tparam S a `kv_readable` store.
    /// \tparam K a `kv_byte_source`.
    /// \param store the store.
    /// \param key the key.
    /// \return the value, `std::nullopt`, or the storage error.
    /// \throws std::bad_alloc when the copy cannot be made.
    template <kv_readable S, kv_byte_source K>
    result<std::optional<std::string>, kv_errc> kv_get(const S& store, const K& key)
    {
        std::string value;
        auto found = kv_get(store, key, value);
        if (found.is_err())
            return err(found.unwrap_err_unchecked());
        return ok(*found ? std::optional<std::string>(std::move(value)) : std::nullopt);
    }

    /// \brief Whether `key` is present.
    /// \ingroup core
    /// \tparam S a `kv_readable` store.
    /// \tparam K a `kv_byte_source`.
    /// \param store the store.
    /// \param key the key.
    /// \return whether it is present, or the storage error.
    template <kv_readable S, kv_byte_source K>
    result<bool, kv_errc> kv_contains(const S& store, const K& key)
    {
        return store.contains(kv_bytes_of(key));
    }

    /// \brief Inserts or overwrites `key`.
    /// \ingroup core
    /// \tparam S a `kv_writable` store.
    /// \tparam K a `kv_byte_source`.
    /// \tparam V a `kv_byte_source`.
    /// \param store the store.
    /// \param key the key.
    /// \param value the value.
    /// \return nothing, or the storage error.
    template <kv_writable S, kv_byte_source K, kv_byte_source V>
    result<void, kv_errc> kv_put(S& store, const K& key, const V& value)
    {
        return store.put(kv_bytes_of(key), kv_bytes_of(value));
    }

    /// \brief Inserts `key` only when absent; a present key is left exactly as it was.
    /// \ingroup core
    /// \tparam S a `kv_writable` store.
    /// \tparam K a `kv_byte_source`.
    /// \tparam V a `kv_byte_source`.
    /// \param store the store.
    /// \param key the key.
    /// \param value the value.
    /// \return `true` if inserted, `false` if the key existed, or the storage error.
    template <kv_writable S, kv_byte_source K, kv_byte_source V>
    result<bool, kv_errc> kv_insert(S& store, const K& key, const V& value)
    {
        return store.insert(kv_bytes_of(key), kv_bytes_of(value));
    }

    /// \brief Removes `key`.
    /// \ingroup core
    /// \tparam S a `kv_writable` store.
    /// \tparam K a `kv_byte_source`.
    /// \param store the store.
    /// \param key the key.
    /// \return whether it existed, or the storage error.
    template <kv_writable S, kv_byte_source K>
    result<bool, kv_errc> kv_erase(S& store, const K& key)
    {
        return store.erase(kv_bytes_of(key));
    }

    /// \brief Visits `range` in ascending key order with `fn(key, value)`, which returns `void` or `bool` (false stops).
    /// \ingroup core
    /// \tparam S a `kv_readable` store.
    /// \tparam F callable as `void(kv_bytes, kv_bytes)` or `bool(kv_bytes, kv_bytes)`.
    /// \param store the store.
    /// \param range the keys to visit.
    /// \param fn the visitor; must not call back into the store.
    /// \return the number of entries passed to `fn`, the one that stopped it included, or the storage error.
    /// \throws whatever `fn` throws, unchanged.
    template <kv_readable S, typename F>
        requires std::is_invocable_v<F&, kv_bytes, kv_bytes>
    result<std::size_t, kv_errc> kv_scan(const S& store, const kv_range& range, F&& fn)
    {
        using answer = std::invoke_result_t<F&, kv_bytes, kv_bytes>;
        static_assert(std::is_void_v<answer> || std::is_convertible_v<answer, bool>,
                      "snicholls::kv_scan: the visitor must return void or bool");
        if constexpr (std::is_void_v<answer>)
            return store.scan(range, [&](kv_bytes k, kv_bytes v) {
                fn(k, v);
                return true;
            });
        else
            return store.scan(range, kv_scan_visitor(fn));
    }

    /// \brief `kv_scan(store, kv_range::prefixed(prefix), fn)`: exactly the keys that start with `prefix`.
    /// \ingroup core
    /// \tparam S a `kv_readable` store.
    /// \tparam K a `kv_byte_source`.
    /// \tparam F as for `kv_scan`.
    /// \param store the store.
    /// \param prefix the prefix.
    /// \param fn the visitor.
    /// \return the number of entries passed to `fn`, or the storage error.
    /// \throws whatever `fn` throws, unchanged.
    template <kv_readable S, kv_byte_source K, typename F>
        requires std::is_invocable_v<F&, kv_bytes, kv_bytes>
    result<std::size_t, kv_errc> kv_scan_prefix(const S& store, const K& prefix, F&& fn)
    {
        return kv_scan(store, kv_range::prefixed(kv_bytes_of(prefix)), fn);
    }

    /// \brief The number of keys in `range`: a scan, O(n) on every backend.
    /// \ingroup core
    /// \tparam S a `kv_readable` store.
    /// \param store the store.
    /// \param range the keys to count.
    /// \return the count, or the storage error.
    template <kv_readable S>
    result<std::size_t, kv_errc> kv_count(const S& store, const kv_range& range = kv_range::all())
    {
        return store.scan(range, [](kv_bytes, kv_bytes) { return true; });
    }

    /// \brief The domain tag that starts every `kv_digest_v1` stream, its terminating NUL included (18 bytes).
    /// \ingroup core
    inline constexpr std::string_view kv_digest_v1_tag{"crtp.kv.digest.v1\0", 18};

    /// \brief The digest of a range, `crtp.kv.digest.v1`: SHA-256 over its pairs in key order. A stable wire contract.
    /// \ingroup core
    ///
    /// The stream is `kv_digest_v1_tag`, then for each pair in ascending key order
    /// `u64_le(size(key)) || key || u64_le(size(value)) || value`, then `u64_le(pair count)`, so an empty range has
    /// a defined digest. The range itself is not hashed: record it beside the digest. Take it from a snapshot to
    /// pair it with that snapshot's `sequence()`.
    /// \tparam S a `kv_readable` store.
    /// \param store the store or snapshot.
    /// \param range the keys to digest.
    /// \return the 32 bytes, or the storage error.
    template <kv_readable S>
    result<digest<32>, kv_errc> kv_digest_v1(const S& store, const kv_range& range = kv_range::all())
    {
        const auto u64_le = [](std::uint64_t n) {
            std::array<std::byte, 8> out{};
            for (std::size_t i = 0; i < 8; ++i)
                out[i] = static_cast<std::byte>(n >> (8 * i));
            return out;
        };
        sha256 h;
        h.update(kv_digest_v1_tag);
        auto pairs = store.scan(range, [&](kv_bytes k, kv_bytes v) {
            h.update(u64_le(k.size())).update(k).update(u64_le(v.size())).update(v);
            return true;
        });
        if (pairs.is_err())
            return err(pairs.unwrap_err_unchecked());
        h.update(u64_le(*pairs));
        return ok(h.finish());
    }
}

#endif /* snicholls_core_concepts_key_value_store_hpp */
