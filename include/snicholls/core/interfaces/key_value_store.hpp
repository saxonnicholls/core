//
//  key_value_store.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The virtual twin of concepts/key_value_store.hpp: the same operations as pure virtual functions, for a store
//  chosen at run time, and the adapters both ways. The surface is crtp's key_value_store_interface; the do_*
//  non-virtual-interface split is not needed, because the conveniences are free functions written once for both
//  forms.
//

/// \file
/// \brief The key-value store contract, virtual form: `key_value_store_interface`, `kv_reader_interface` and the adapters.
/// \ingroup core
///
/// **Overview.** When the store is chosen at run time (a configuration file names the backend, a plugin provides
/// it), code holds a `key_value_store_interface&`. Its pure virtual functions are exactly the operations of the
/// concept `key_value_store` (`concepts/key_value_store.hpp`), so:
/// - the interface itself models the concept: every function template written against the concept, the
///   conveniences `kv_get`, `kv_scan_prefix`, `kv_digest_v1` and the conformance kit included, takes it as it is;
/// - `key_value_store_adapter<S>` puts any model of the concept behind the interface, owning it (`S`) or referring
///   to it (`S&`); `kv_reader_adapter<R>` does the same for a read-only view, and is what an adapted store's
///   `snapshot()` returns.
///
/// \snippet interfaces/key_value_store_demo.cpp composition_root
///
/// Use the concept by default (static dispatch, inlined on in-memory stores); use the interface where the choice
/// is made at run time or a compilation firewall is wanted. For an I/O-bound backend the virtual call is free; on
/// the in-memory reference it is measured (Performance).
///
/// **Relationships.** The concept, the ranges, batches, errors and conveniences: `concepts/key_value_store.hpp`.
/// The reference implementation: `interfaces/key_value_store_reference.hpp`. The kit:
/// `interfaces/key_value_store_conformance.hpp`. The other capabilities' interfaces: `kv_health_interface`
/// (`interfaces/kv_health.hpp`), `kv_backup_interface` (`interfaces/kv_backup.hpp`); a backend implements as many
/// as it offers, and a caller holding one asks for another with `dynamic_cast`. Group: \ref core.
///
/// **Implementations / variants.** `key_value_store_adapter<S>` over any model; above core, crtp's backends
/// derive from the interface directly.
///
/// **Guarantees.** Those of the concept, which every implementation keeps. The adapters add one: **no exception
/// of the adapted store crosses them.** An exception from the model becomes `kv_errc::out_of_memory`
/// (`std::bad_alloc`) or `kv_errc::backend` (anything else); an exception thrown by the caller's own visitor
/// propagates unchanged. Adapters are as thread-safe as the store they adapt.
///
/// **Errors.** As the concept; see the adapters' translation above.
///
/// **Examples.** See the overview, and `demos/interfaces/key_value_store_demo.cpp`.
///
/// **Performance.** `benchmarks/interfaces/key_value_store_benchmark.cpp` compares a hand-written `std::map`, the
/// concept path and the virtual path on `reference_key_value_store`: point reads, overwrites and a 100-key prefix
/// scan. Numbers in BENCHMARKS.md (Release only).
///
/// **Tests and demos.** `tests/interfaces/key_value_store_test.cpp` (the kit through both forms, the adapters'
/// exception translation), `demos/interfaces/key_value_store_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<memory>`, `<new>`, `<span>`, `<type_traits>`, `<utility>`, and
/// `concepts/key_value_store.hpp`.
///
/// **Known issues.** A visitor that swallows an exception of the store and throws another is indistinguishable
/// from a visitor's own exception, and propagates.
///
/// **Status.** New in 0.22.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_key_value_store_hpp
#define snicholls_core_interfaces_key_value_store_hpp

#include <snicholls/core/concepts/key_value_store.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The read operations of a store, behind a virtual interface: `read`, `contains`, `scan`, `sequence`.
    /// \ingroup core
    ///
    /// What `key_value_store_interface::snapshot()` returns: a read-only view, so a write to a snapshot is a compile
    /// error rather than a run-time one. Models `kv_readable`.
    class kv_reader_interface {
    public:
        /// \brief Destroys the reader.
        virtual ~kv_reader_interface() = default;

        /// \brief Calls `fn(value)` with the value of `key` and answers `true`; `false`, without calling, if absent.
        /// \param key the key.
        /// \param fn the visitor; the bytes are valid only during the call.
        /// \return whether the key was found, or the storage error.
        /// \throws whatever `fn` throws.
        virtual result<bool, kv_errc> read(kv_bytes key, kv_read_visitor fn) const = 0;

        /// \brief Whether `key` is present.
        /// \param key the key.
        /// \return the answer, or the storage error.
        virtual result<bool, kv_errc> contains(kv_bytes key) const = 0;

        /// \brief Visits `range` in ascending key order until `fn` answers `false`.
        /// \param range the keys.
        /// \param fn the visitor; the bytes are valid only during the call.
        /// \return the number of entries passed to `fn`, the one that stopped it included, or the storage error.
        /// \throws whatever `fn` throws.
        virtual result<std::size_t, kv_errc> scan(const kv_range& range, kv_scan_visitor fn) const = 0;

        /// \brief The store's sequence: never decreases, increases with every call that changed the store.
        /// \return the sequence, or the storage error.
        virtual result<std::uint64_t, kv_errc> sequence() const = 0;

    protected:
        /// \brief Constructs the base.
        kv_reader_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        kv_reader_interface(const kv_reader_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        kv_reader_interface& operator=(const kv_reader_interface&) = default;
    };

    /// \brief An ordered map of byte strings to byte strings, behind a virtual interface. Models `key_value_store`.
    /// \ingroup core
    ///
    /// The operations, their guarantees and errors are the concept's (`concepts/key_value_store.hpp`).
    class key_value_store_interface : public kv_reader_interface {
    public:
        /// \brief Inserts or overwrites `key`.
        /// \param key the key.
        /// \param value the value.
        /// \return nothing, or the storage error.
        virtual result<void, kv_errc> put(kv_bytes key, kv_bytes value) = 0;

        /// \brief Inserts `key` only when absent.
        /// \param key the key.
        /// \param value the value.
        /// \return `true` if inserted; `false` if the key existed (left exactly as it was); or the storage error.
        virtual result<bool, kv_errc> insert(kv_bytes key, kv_bytes value) = 0;

        /// \brief Removes `key`.
        /// \param key the key.
        /// \return whether it existed, or the storage error.
        virtual result<bool, kv_errc> erase(kv_bytes key) = 0;

        /// \brief Removes every key of `range`.
        /// \param range the keys.
        /// \return how many were removed, or the storage error.
        virtual result<std::size_t, kv_errc> erase_range(const kv_range& range) = 0;

        /// \brief Applies `batch` in order, all or nothing.
        /// \param batch the writes; an empty batch changes nothing.
        /// \return nothing, or the error that stopped it (`kv_errc::conflict` for an `insert` of a present key),
        /// in which case nothing was applied.
        virtual result<void, kv_errc> apply(std::span<const kv_write> batch) = 0;

        /// \brief A read-only view of the current state, unaffected by later writes.
        /// \return the view, or the storage error.
        virtual result<std::unique_ptr<kv_reader_interface>, kv_errc> snapshot() const = 0;
    };

    namespace detail {
        // Runs f(), turning an exception of the store into an error and letting the caller's visitor's through.
        // visitor_threw is set by kv_visit when the visitor throws.
        template <typename R, typename F>
        R kv_guard(const bool& visitor_threw, F&& f)
        {
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            try {
                return f();
            } catch (const std::bad_alloc&) {
                if (visitor_threw)
                    throw;
                return err(kv_errc::out_of_memory);
            } catch (...) {
                if (visitor_threw)
                    throw;
                return err(kv_errc::backend);
            }
#else
            static_cast<void>(visitor_threw);
            return f();
#endif
        }

        // Calls the caller's visitor, recording that an exception came from it.
        template <typename F, typename... A>
        decltype(auto) kv_visit(bool& visitor_threw, F& fn, A... args)
        {
#if SNICHOLLS_RESULT_HAS_EXCEPTIONS
            try {
                return fn(args...);
            } catch (...) {
                visitor_threw = true;
                throw;
            }
#else
            static_cast<void>(visitor_threw);
            return fn(args...);
#endif
        }
    }

    /// \brief Any `kv_readable` view (or a pointer-like handle to one) behind `kv_reader_interface`.
    /// \ingroup core
    /// \tparam R the view: a `kv_readable` type, a reference to one, or a handle such as `std::unique_ptr`.
    template <typename R>
        requires detail::kv_snapshot_handle<std::remove_cvref_t<R>>
    class kv_reader_adapter final : public kv_reader_interface {
    public:
        /// \brief Takes (or, for a reference `R`, refers to) the view.
        /// \param reader the view.
        explicit kv_reader_adapter(R reader) : reader_(std::forward<R>(reader)) {}

        /// \brief The adapted view.
        /// \return it.
        [[nodiscard]] const auto& get() const noexcept { return detail::kv_target(reader_); }

        /// \brief See `kv_reader_interface::read`.
        /// \param key the key.
        /// \param fn the visitor.
        /// \return as the adapted view, its exceptions translated.
        result<bool, kv_errc> read(kv_bytes key, kv_read_visitor fn) const override
        {
            bool threw = false;
            return detail::kv_guard<result<bool, kv_errc>>(threw, [&] {
                return get().read(key, [&](kv_bytes v) { detail::kv_visit(threw, fn, v); });
            });
        }
        /// \brief See `kv_reader_interface::contains`.
        /// \param key the key.
        /// \return as the adapted view, its exceptions translated.
        result<bool, kv_errc> contains(kv_bytes key) const override
        {
            return detail::kv_guard<result<bool, kv_errc>>(false, [&] { return get().contains(key); });
        }
        /// \brief See `kv_reader_interface::scan`.
        /// \param range the keys.
        /// \param fn the visitor.
        /// \return as the adapted view, its exceptions translated.
        result<std::size_t, kv_errc> scan(const kv_range& range, kv_scan_visitor fn) const override
        {
            bool threw = false;
            return detail::kv_guard<result<std::size_t, kv_errc>>(threw, [&] {
                return get().scan(range, [&](kv_bytes k, kv_bytes v) -> bool { return detail::kv_visit(threw, fn, k, v); });
            });
        }
        /// \brief See `kv_reader_interface::sequence`.
        /// \return as the adapted view, its exceptions translated.
        result<std::uint64_t, kv_errc> sequence() const override
        {
            return detail::kv_guard<result<std::uint64_t, kv_errc>>(false, [&] { return get().sequence(); });
        }

    private:
        R reader_;
    };

    namespace detail {
        // A snapshot handed out by a model, as the interface's snapshot type: passed through when it already is one,
        // wrapped in a kv_reader_adapter otherwise.
        template <typename V>
            requires std::is_convertible_v<V, std::unique_ptr<kv_reader_interface>>
        std::unique_ptr<kv_reader_interface> kv_as_reader(V&& v)
        {
            return std::forward<V>(v);
        }
        template <typename V>
            requires(!std::is_convertible_v<V, std::unique_ptr<kv_reader_interface>>)
        std::unique_ptr<kv_reader_interface> kv_as_reader(V&& v)
        {
            return std::make_unique<kv_reader_adapter<std::remove_cvref_t<V>>>(std::forward<V>(v));
        }
    }

    /// \brief Any model of `key_value_store` behind `key_value_store_interface`: owned (`S`) or referred to (`S&`).
    /// \ingroup core
    ///
    /// No exception of the model crosses it (see the file's guarantees).
    /// \tparam S the model, or a reference to one.
    template <typename S>
        requires key_value_store<std::remove_cvref_t<S>>
    class key_value_store_adapter final : public key_value_store_interface {
    public:
        /// \brief Takes (or, for a reference `S`, refers to) the store.
        /// \param store the store.
        explicit key_value_store_adapter(S store) : store_(std::forward<S>(store)) {}

        /// \brief The adapted store.
        /// \return it.
        [[nodiscard]] std::remove_reference_t<S>& get() noexcept { return store_; }
        /// \brief The adapted store.
        /// \return it.
        [[nodiscard]] const std::remove_reference_t<S>& get() const noexcept { return store_; }

        /// \brief See `kv_reader_interface::read`.
        /// \param key the key.
        /// \param fn the visitor.
        /// \return as the model, its exceptions translated.
        result<bool, kv_errc> read(kv_bytes key, kv_read_visitor fn) const override
        {
            bool threw = false;
            return detail::kv_guard<result<bool, kv_errc>>(threw, [&] {
                return get().read(key, [&](kv_bytes v) { detail::kv_visit(threw, fn, v); });
            });
        }
        /// \brief See `kv_reader_interface::contains`.
        /// \param key the key.
        /// \return as the model, its exceptions translated.
        result<bool, kv_errc> contains(kv_bytes key) const override
        {
            return detail::kv_guard<result<bool, kv_errc>>(false, [&] { return get().contains(key); });
        }
        /// \brief See `kv_reader_interface::scan`.
        /// \param range the keys.
        /// \param fn the visitor.
        /// \return as the model, its exceptions translated.
        result<std::size_t, kv_errc> scan(const kv_range& range, kv_scan_visitor fn) const override
        {
            bool threw = false;
            return detail::kv_guard<result<std::size_t, kv_errc>>(threw, [&] {
                return get().scan(range, [&](kv_bytes k, kv_bytes v) -> bool { return detail::kv_visit(threw, fn, k, v); });
            });
        }
        /// \brief See `kv_reader_interface::sequence`.
        /// \return as the model, its exceptions translated.
        result<std::uint64_t, kv_errc> sequence() const override
        {
            return detail::kv_guard<result<std::uint64_t, kv_errc>>(false, [&] { return get().sequence(); });
        }
        /// \brief See `key_value_store_interface::put`.
        /// \param key the key.
        /// \param value the value.
        /// \return as the model, its exceptions translated.
        result<void, kv_errc> put(kv_bytes key, kv_bytes value) override
        {
            return detail::kv_guard<result<void, kv_errc>>(false, [&] { return get().put(key, value); });
        }
        /// \brief See `key_value_store_interface::insert`.
        /// \param key the key.
        /// \param value the value.
        /// \return as the model, its exceptions translated.
        result<bool, kv_errc> insert(kv_bytes key, kv_bytes value) override
        {
            return detail::kv_guard<result<bool, kv_errc>>(false, [&] { return get().insert(key, value); });
        }
        /// \brief See `key_value_store_interface::erase`.
        /// \param key the key.
        /// \return as the model, its exceptions translated.
        result<bool, kv_errc> erase(kv_bytes key) override
        {
            return detail::kv_guard<result<bool, kv_errc>>(false, [&] { return get().erase(key); });
        }
        /// \brief See `key_value_store_interface::erase_range`.
        /// \param range the keys.
        /// \return as the model, its exceptions translated.
        result<std::size_t, kv_errc> erase_range(const kv_range& range) override
        {
            return detail::kv_guard<result<std::size_t, kv_errc>>(false, [&] { return get().erase_range(range); });
        }
        /// \brief See `key_value_store_interface::apply`.
        /// \param batch the writes.
        /// \return as the model, its exceptions translated.
        result<void, kv_errc> apply(std::span<const kv_write> batch) override
        {
            return detail::kv_guard<result<void, kv_errc>>(false, [&] { return get().apply(batch); });
        }
        /// \brief See `key_value_store_interface::snapshot`: the model's snapshot, behind `kv_reader_interface`.
        /// \return the view, or the error.
        result<std::unique_ptr<kv_reader_interface>, kv_errc> snapshot() const override
        {
            using answer = result<std::unique_ptr<kv_reader_interface>, kv_errc>;
            return detail::kv_guard<answer>(false, [&]() -> answer {
                auto snap = get().snapshot();
                if (snap.is_err())
                    return err(snap.unwrap_err_unchecked());
                return ok(detail::kv_as_reader(std::move(snap).unwrap_unchecked()));
            });
        }

    private:
        S store_;
    };

    /// \brief Deduces an owning adapter: `key_value_store_adapter(std::move(store))`.
    /// \ingroup core
    template <typename S>
    key_value_store_adapter(S) -> key_value_store_adapter<S>;
}

#endif /* snicholls_core_interfaces_key_value_store_hpp */
