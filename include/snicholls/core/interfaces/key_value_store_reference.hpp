//
//  key_value_store_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The reference key-value store: crtp's in_memory_key_value_store reduced to the standard library (a std::map
//  with a byte-wise comparator), with the copy-on-write snapshots of crtp's KV_PLAN D20, in-memory checkpoints,
//  and the fault injection a test fake needs.
//

/// \file
/// \brief `reference_key_value_store`: the std-only in-memory store, which tests use as their fake.
/// \ingroup core
///
/// **Overview.** A `std::map` of byte vectors under unsigned byte-wise order, modelling every key-value capability
/// in core: `key_value_store` (`concepts/key_value_store.hpp`), `kv_health` and `kv_backup`. It is the contract's
/// executable definition, the store a unit test hands to the code under test, and the baseline of the benchmark.
/// As a fake it can fail on purpose: `inject_failure(e, n)` makes the write `n` writes from now fail with `e`
/// (inside a batch: the batch fails at that write and changes nothing), and `set_unavailable(true)` makes every
/// operation answer `kv_errc::unavailable`, as a lost connection would.
///
/// \snippet interfaces/key_value_store_reference_demo.cpp fake
///
/// Not for production data: nothing persists, and it is not synchronised.
///
/// **Relationships.** Models the concepts directly (static dispatch); behind the virtual interfaces through
/// `key_value_store_adapter`, `kv_health_adapter`, `kv_backup_adapter`. Its snapshots are
/// `reference_kv_snapshot`. Passes `run_kv_conformance`, `run_kv_health_conformance` and
/// `run_kv_backup_conformance` (`interfaces/key_value_store_conformance.hpp`). Group: \ref core.
///
/// **Implementations / variants.** This is the one implementation in core.
///
/// **Guarantees.**
/// - Everything the contract promises (the kit checks it).
/// - **Strong exception safety, and no exception out:** a call that fails changes nothing, `std::bad_alloc`
///   included, which it reports as `kv_errc::out_of_memory`. Only a visitor's own exception leaves a call.
/// - **Copy-on-write:** a snapshot, a checkpoint and a copy of the store share the data; the first write after one
///   copies the map (O(n)), so taking them is O(1) and reading them is free.
/// - **Complexity:** `read`, `contains`, `put`, `insert`, `erase` O(log n) unless a snapshot is shared; `scan`
///   O(log n + k); `apply` O(n + b log n) (it builds the next state and swaps it in); `diagnostics` and `verify` O(n).
/// - **Thread safety:** none; like a standard container, one thread at a time (snapshots included, since they share
///   the data and its reference count with the store).
/// - **Sequence:** starts at 0 and rises by one per call that changed the store.
///
/// **Errors.** `kv_errc::unavailable` while `set_unavailable(true)`; the injected error; `kv_errc::conflict` for a
/// batch `insert` of a present key; `kv_errc::invalid_argument` for an empty or used checkpoint name;
/// `kv_errc::out_of_memory`.
///
/// **Examples.** See the overview, and `demos/interfaces/key_value_store_reference_demo.cpp`.
///
/// **Performance.** See `benchmarks/interfaces/key_value_store_benchmark.cpp` and BENCHMARKS.md.
///
/// **Tests and demos.** `tests/interfaces/key_value_store_reference_test.cpp` (the fake's controls, sharing,
/// checkpoints), the kit in `tests/interfaces/key_value_store_test.cpp`,
/// `demos/interfaces/key_value_store_reference_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<cstddef>`, `<cstdint>`, `<iterator>`, `<map>`, `<memory>`, `<new>`, `<optional>`,
/// `<span>`, `<string>`, `<string_view>`, `<utility>`, `<vector>`, and the three capability concepts.
///
/// **Known issues.** `apply` copies the map, so a fake holding many entries pays O(n) per batch.
///
/// **Status.** New in 0.22.0. Experimental.

#ifndef snicholls_core_interfaces_key_value_store_reference_hpp
#define snicholls_core_interfaces_key_value_store_reference_hpp

#include <snicholls/core/concepts/key_value_store.hpp>
#include <snicholls/core/concepts/kv_backup.hpp>
#include <snicholls/core/concepts/kv_health.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace snicholls {

    namespace detail {
        struct kv_bytes_order {
            using is_transparent = void;
            constexpr bool operator()(kv_bytes a, kv_bytes b) const noexcept { return kv_less(a, b); }
        };

        using kv_reference_map = std::map<std::vector<std::byte>, std::vector<std::byte>, kv_bytes_order>;

        inline std::vector<std::byte> kv_copy(kv_bytes b) { return {b.begin(), b.end()}; }

        template <typename F>
        bool kv_reference_read(const kv_reference_map& m, kv_bytes key, F& fn)
        {
            const auto it = m.find(key);
            if (it == m.end())
                return false;
            fn(kv_bytes(it->second));
            return true;
        }

        inline kv_reference_map::const_iterator kv_reference_start(const kv_reference_map& m, const kv_range& r)
        {
            return r.lower_open() ? m.upper_bound(r.lower()) : m.lower_bound(r.lower());
        }

        template <typename F>
        std::size_t kv_reference_scan(const kv_reference_map& m, const kv_range& r, F& fn)
        {
            if (r.empty())
                return 0;
            std::size_t n = 0;
            for (auto it = kv_reference_start(m, r); it != m.end() && !r.beyond(it->first); ++it) {
                ++n;
                if (!fn(kv_bytes(it->first), kv_bytes(it->second)))
                    break;
            }
            return n;
        }
    }

    class reference_key_value_store;

    /// \brief A read-only view of a `reference_key_value_store` at one moment. Models `kv_readable`.
    /// \ingroup core
    ///
    /// Shares the data with the store (copy-on-write), so it costs O(1) to take and keeps its state whatever the
    /// store does later, the store's destruction included.
    class reference_kv_snapshot {
    public:
        /// \brief See `kv_readable`: calls `fn(value)` if `key` is present.
        /// \tparam F callable as `void(kv_bytes)`.
        /// \param key the key.
        /// \param fn the visitor.
        /// \return whether the key was found.
        template <typename F>
        result<bool, kv_errc> read(kv_bytes key, F&& fn) const
        {
            return ok(detail::kv_reference_read(*data_, key, fn));
        }
        /// \brief See `kv_readable`.
        /// \param key the key.
        /// \return whether it is present.
        result<bool, kv_errc> contains(kv_bytes key) const { return ok(data_->find(key) != data_->end()); }
        /// \brief See `kv_readable`.
        /// \tparam F callable as `bool(kv_bytes, kv_bytes)`.
        /// \param range the keys.
        /// \param fn the visitor.
        /// \return the number of entries passed to `fn`.
        template <typename F>
        result<std::size_t, kv_errc> scan(const kv_range& range, F&& fn) const
        {
            return ok(detail::kv_reference_scan(*data_, range, fn));
        }
        /// \brief The store's sequence when the snapshot was taken.
        /// \return it.
        result<std::uint64_t, kv_errc> sequence() const { return ok(sequence_); }

    private:
        friend class reference_key_value_store;
        reference_kv_snapshot(std::shared_ptr<const detail::kv_reference_map> data, std::uint64_t sequence) noexcept
            : data_(std::move(data)), sequence_(sequence)
        {
        }

        std::shared_ptr<const detail::kv_reference_map> data_;
        std::uint64_t sequence_;
    };

    /// \brief The std-only in-memory key-value store. Models `key_value_store`, `kv_health` and `kv_backup`.
    /// \ingroup core
    ///
    /// See the file documentation for its guarantees and its controls as a test fake.
    class reference_key_value_store {
    public:
        /// \brief An empty store, sequence 0.
        /// \throws std::bad_alloc
        reference_key_value_store() : data_(std::make_shared<detail::kv_reference_map>()) {}

        // --- reads ------------------------------------------------------------------------------------------------

        /// \brief See `kv_readable`: calls `fn(value)` if `key` is present.
        /// \tparam F callable as `void(kv_bytes)`.
        /// \param key the key.
        /// \param fn the visitor.
        /// \return whether the key was found, or `kv_errc::unavailable`.
        template <typename F>
        result<bool, kv_errc> read(kv_bytes key, F&& fn) const
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
            return ok(detail::kv_reference_read(*data_, key, fn));
        }

        /// \brief See `kv_readable`.
        /// \param key the key.
        /// \return whether it is present, or `kv_errc::unavailable`.
        result<bool, kv_errc> contains(kv_bytes key) const
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
            return ok(data_->find(key) != data_->end());
        }

        /// \brief See `kv_readable`.
        /// \tparam F callable as `bool(kv_bytes, kv_bytes)`.
        /// \param range the keys.
        /// \param fn the visitor.
        /// \return the number of entries passed to `fn`, or `kv_errc::unavailable`.
        template <typename F>
        result<std::size_t, kv_errc> scan(const kv_range& range, F&& fn) const
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
            return ok(detail::kv_reference_scan(*data_, range, fn));
        }

        /// \brief See `kv_readable`: rises by one per call that changed the store.
        /// \return the sequence, or `kv_errc::unavailable`.
        result<std::uint64_t, kv_errc> sequence() const
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
            return ok(sequence_);
        }

        /// \brief A view of the current state, O(1): it shares the data until the store next writes.
        /// \return the snapshot, or `kv_errc::unavailable`.
        result<reference_kv_snapshot, kv_errc> snapshot() const
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
            return ok(reference_kv_snapshot(data_, sequence_));
        }

        // --- writes -----------------------------------------------------------------------------------------------

        /// \brief See `kv_writable`.
        /// \param key the key.
        /// \param value the value.
        /// \return nothing, or the error (nothing changed).
        result<void, kv_errc> put(kv_bytes key, kv_bytes value)
        {
            return write([&]() -> result<void, kv_errc> {
                if (const auto e = tick())
                    return err(*e);
                auto copy = detail::kv_copy(value);
                auto& m = writable();
                if (const auto it = m.find(key); it != m.end())
                    it->second.swap(copy);
                else
                    m.emplace(detail::kv_copy(key), std::move(copy));
                ++sequence_;
                return ok();
            });
        }

        /// \brief See `kv_writable`: inserts only when absent; a present key is left exactly as it was.
        /// \param key the key.
        /// \param value the value.
        /// \return whether it was inserted, or the error (nothing changed).
        result<bool, kv_errc> insert(kv_bytes key, kv_bytes value)
        {
            return write([&]() -> result<bool, kv_errc> {
                if (const auto e = tick())
                    return err(*e);
                if (data_->find(key) != data_->end())
                    return ok(false);
                writable().emplace(detail::kv_copy(key), detail::kv_copy(value));
                ++sequence_;
                return ok(true);
            });
        }

        /// \brief See `kv_writable`.
        /// \param key the key.
        /// \return whether it existed, or the error (nothing changed).
        result<bool, kv_errc> erase(kv_bytes key)
        {
            return write([&]() -> result<bool, kv_errc> {
                if (const auto e = tick())
                    return err(*e);
                if (data_->find(key) == data_->end())
                    return ok(false);
                auto& m = writable();
                m.erase(m.find(key));
                ++sequence_;
                return ok(true);
            });
        }

        /// \brief See `kv_writable`.
        /// \param range the keys.
        /// \return how many were removed, or the error (nothing changed).
        result<std::size_t, kv_errc> erase_range(const kv_range& range)
        {
            return write([&]() -> result<std::size_t, kv_errc> {
                if (const auto e = tick())
                    return err(*e);
                if (range.empty())
                    return ok(std::size_t{0});
                const auto first = detail::kv_reference_start(*data_, range);
                if (first == data_->end() || range.beyond(first->first))
                    return ok(std::size_t{0});
                auto& m = writable();
                std::size_t n = 0;
                for (auto it = detail::kv_reference_start(m, range); it != m.end() && !range.beyond(it->first); ++n)
                    it = m.erase(it);
                ++sequence_;
                return ok(n);
            });
        }

        /// \brief See `kv_writable`: every write in order, all or nothing. Each write counts toward an injected failure.
        /// \param batch the writes.
        /// \return nothing, or the error (`kv_errc::conflict` for an `insert` of a present key); nothing changed.
        result<void, kv_errc> apply(std::span<const kv_write> batch)
        {
            return write([&]() -> result<void, kv_errc> {
                if (batch.empty())
                    return ok();
                auto next = std::make_shared<detail::kv_reference_map>(*data_);
                for (const kv_write& w : batch) {
                    if (const auto e = tick())
                        return err(*e);
                    switch (w.op) {
                    case kv_op::put:
                        next->insert_or_assign(detail::kv_copy(w.key), detail::kv_copy(w.value));
                        break;
                    case kv_op::insert:
                        if (next->find(w.key) != next->end())
                            return err(kv_errc::conflict);
                        next->emplace(detail::kv_copy(w.key), detail::kv_copy(w.value));
                        break;
                    case kv_op::erase:
                        if (const auto it = next->find(w.key); it != next->end())
                            next->erase(it);
                        break;
                    }
                }
                data_ = std::move(next);
                ++sequence_;
                return ok();
            });
        }

        // --- health -----------------------------------------------------------------------------------------------

        /// \brief See `kv_health`.
        /// \return nothing, or `kv_errc::unavailable`.
        result<void, kv_errc> ping() const
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
            return ok();
        }

        /// \brief See `kv_health`: backend `"reference"`, exact key and byte counts, not persistent.
        /// \return the diagnostics, or `kv_errc::unavailable`.
        result<kv_diagnostics, kv_errc> diagnostics() const
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
            std::uint64_t bytes = 0;
            for (const auto& [k, v] : *data_)
                bytes += k.size() + v.size();
            return ok(kv_diagnostics{"reference", sequence_, data_->size(), bytes, false, false});
        }

        // --- backup -----------------------------------------------------------------------------------------------

        /// \brief See `kv_backup`: keeps the current state in memory under the name `destination`, O(1).
        /// \param destination the checkpoint's name; not empty, not used before.
        /// \return the sequence and size captured, or the error (nothing changed).
        result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view destination)
        {
            return write([&]() -> result<kv_checkpoint_info, kv_errc> {
                if (const auto e = tick())
                    return err(*e);
                if (destination.empty() || checkpoints_.find(destination) != checkpoints_.end())
                    return err(kv_errc::invalid_argument);
                std::uint64_t bytes = 0;
                for (const auto& [k, v] : *data_)
                    bytes += k.size() + v.size();
                checkpoints_.emplace(std::string(destination), saved{data_, sequence_});
                return ok(kv_checkpoint_info{sequence_, bytes});
            });
        }

        /// \brief See `kv_backup`: checks that the keys are strictly ascending.
        /// \return the report, or `kv_errc::unavailable`.
        result<kv_verify_report, kv_errc> verify() const
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
            kv_verify_report report{data_->size(), 0};
            const detail::kv_bytes_order less;
            for (auto it = data_->begin(); it != data_->end() && std::next(it) != data_->end(); ++it)
                report.problems += !less(it->first, std::next(it)->first);   // strictly ascending, no duplicates
            return ok(report);
        }

        /// \brief A writable store holding what the checkpoint `name` captured, its sequence included.
        /// \param name the checkpoint's name.
        /// \return the store, or `std::nullopt` when there is no such checkpoint.
        /// \throws std::bad_alloc
        [[nodiscard]] std::optional<reference_key_value_store> open_checkpoint(std::string_view name) const
        {
            const auto it = checkpoints_.find(name);
            if (it == checkpoints_.end())
                return std::nullopt;
            reference_key_value_store opened;
            opened.data_ = it->second.data;
            opened.sequence_ = it->second.sequence;
            return opened;
        }

        // --- the fake's controls ----------------------------------------------------------------------------------

        /// \brief Makes the write `after` writes from now fail with `error`, and the call containing it change nothing.
        ///
        /// Every `put`, `insert`, `erase`, `erase_range` and `checkpoint` is one write; a batch counts each of its
        /// writes. `inject_failure(e)` fails the next write call; `inject_failure(e, 2)` fails a batch at its
        /// third write. One-shot: it is spent by the failure. A later call replaces it.
        /// \param error the error to fail with.
        /// \param after how many writes succeed first.
        void inject_failure(kv_errc error, std::size_t after = 0) noexcept
        {
            fault_ = error;
            fault_after_ = after;
        }

        /// \brief Every operation answers `kv_errc::unavailable` while `unavailable` is true, as a lost connection would.
        /// \param unavailable whether the store is unreachable.
        void set_unavailable(bool unavailable) noexcept { unavailable_ = unavailable; }

    private:
        struct saved {
            std::shared_ptr<detail::kv_reference_map> data;
            std::uint64_t sequence;
        };

        // The map, copied first if a snapshot, a checkpoint or a copy of the store shares it.
        detail::kv_reference_map& writable()
        {
            if (data_.use_count() != 1)
                data_ = std::make_shared<detail::kv_reference_map>(*data_);
            return *data_;
        }

        // An injected failure due now, spending it; otherwise counts one write toward it.
        std::optional<kv_errc> tick() noexcept
        {
            if (!fault_)
                return std::nullopt;
            if (fault_after_ > 0) {
                --fault_after_;
                return std::nullopt;
            }
            return std::exchange(fault_, std::nullopt);
        }

        // A write: refused while unavailable; std::bad_alloc becomes an error (nothing has changed when it is thrown).
        template <typename F>
        auto write(F&& f) -> decltype(f())
        {
            if (unavailable_)
                return err(kv_errc::unavailable);
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            try {
                return f();
            } catch (const std::bad_alloc&) {
                return err(kv_errc::out_of_memory);
            }
#else
            return f();
#endif
        }

        std::shared_ptr<detail::kv_reference_map> data_;
        std::uint64_t sequence_ = 0;
        std::optional<kv_errc> fault_{};
        std::size_t fault_after_ = 0;
        bool unavailable_ = false;
        std::map<std::string, saved, std::less<>> checkpoints_;
    };
}

#endif /* snicholls_core_interfaces_key_value_store_reference_hpp */
