//
//  kv_cas.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The compare-and-swap capability, virtual form: `kv_cas_interface`, its adapter, and the kit `check_kv_cas`.
/// \ingroup core
///
/// **Overview.** `kv_cas_interface` is `kv_cas` (`concepts/kv_cas.hpp`) behind a virtual function, for a store
/// chosen at run time; it models the concept. `kv_cas_adapter<S>` puts any model behind it, owned (`S`) or referred
/// to (`S&`), translating an exception into `kv_errc` as the store adapters do. `check_kv_cas(store)` is the
/// capability's kit: it drives a store that is also `kv_readable` through the contract and names the first check
/// that fails.
///
/// \snippet interfaces/kv_cas_demo.cpp virtual
///
/// **Relationships.** `concepts/kv_cas.hpp`, `interfaces/key_value_store.hpp`, `reference_key_value_store`
/// (`interfaces/key_value_store_reference.hpp`). Group: \ref core.
///
/// **Implementations / variants.** `kv_cas_adapter<S>` over any model.
///
/// **Guarantees.** One virtual call; no exception crosses the adapter but `std::bad_alloc`.
///
/// **Errors.** As the model's; `check_kv_cas` returns the failed check's name.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/kv_cas_test.cpp` (the kit through both forms),
/// `demos/interfaces/kv_cas_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<optional>`, `<string_view>`, `<type_traits>`, `<utility>`,
/// `concepts/kv_cas.hpp`, `interfaces/key_value_store.hpp`.
///
/// **Known issues.** The kit writes and erases the keys `kit/cas/*`.
///
/// **Status.** New in 0.27.0. Experimental.

#ifndef snicholls_core_interfaces_kv_cas_hpp
#define snicholls_core_interfaces_kv_cas_hpp

#include <snicholls/core/concepts/kv_cas.hpp>
#include <snicholls/core/interfaces/key_value_store.hpp>

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief A store's compare-and-swap, behind a virtual interface. Models `kv_cas`.
    /// \ingroup core
    class kv_cas_interface {
    public:
        /// \brief Destroys the object.
        virtual ~kv_cas_interface() = default;

        /// \brief Replaces `key`'s value with `desired` only when it is `expected`, atomically.
        /// \param key      The key.
        /// \param expected The value it must have, or `std::nullopt` for "absent".
        /// \param desired  The value to set, or `std::nullopt` to erase.
        /// \return Whether it swapped (nothing changed when not), or the error.
        virtual result<bool, kv_errc> compare_and_swap(kv_bytes key, std::optional<kv_bytes> expected, std::optional<kv_bytes> desired) = 0;

    protected:
        /// \brief Constructs the base.
        kv_cas_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        kv_cas_interface(const kv_cas_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        kv_cas_interface& operator=(const kv_cas_interface&) = default;
    };

    static_assert(kv_cas<kv_cas_interface>, "snicholls::kv_cas_interface must model snicholls::kv_cas");

    /// \brief Any model of `kv_cas` behind `kv_cas_interface`: owned (`S`) or referred to (`S&`).
    /// \ingroup core
    /// \tparam S the model, or a reference to one.
    template <typename S>
        requires kv_cas<std::remove_reference_t<S>>
    class kv_cas_adapter final : public kv_cas_interface {
    public:
        /// \brief Takes (or, for a reference `S`, refers to) the model.
        /// \param model the model.
        explicit kv_cas_adapter(S model) : model_(std::forward<S>(model)) {}

        /// \brief The adapted model.
        /// \return it.
        [[nodiscard]] std::remove_reference_t<S>& get() noexcept { return model_; }

        /// \brief See `kv_cas_interface::compare_and_swap`.
        /// \param key      The key.
        /// \param expected The expected value.
        /// \param desired  The new value.
        /// \return As the model, its exceptions translated.
        result<bool, kv_errc> compare_and_swap(kv_bytes key, std::optional<kv_bytes> expected, std::optional<kv_bytes> desired) override
        {
            return detail::kv_guard<result<bool, kv_errc>>(false, [&] { return model_.compare_and_swap(key, expected, desired); });
        }

    private:
        S model_;
    };

    /// \brief Deduces an owning adapter.
    /// \ingroup core
    template <typename S>
    kv_cas_adapter(S) -> kv_cas_adapter<S>;

    /// \brief The `kv_cas` kit: drives `store` (also `kv_readable`, to read back) through the contract.
    /// \ingroup core
    /// \param store The store; the keys `kit/cas/*` are written and erased.
    /// \return `std::nullopt` when it conforms, or the first check that failed.
    template <typename S>
        requires kv_cas<S> && kv_readable<S>
    [[nodiscard]] std::optional<std::string_view> check_kv_cas(S& store)
    {
        const auto bytes = [](std::string_view s) { return kv_bytes(reinterpret_cast<const std::byte*>(s.data()), s.size()); };
        const kv_bytes key = bytes("kit/cas/lease");
        const auto value = [&]() -> std::optional<std::string> {
            std::optional<std::string> out;
            const auto found = store.read(key, [&](kv_bytes v) { out = std::string(reinterpret_cast<const char*>(v.data()), v.size()); });
            return found && *found ? out : std::nullopt;
        };
        static_cast<void>(store.compare_and_swap(key, value() ? std::optional<kv_bytes>(bytes(*value())) : std::nullopt, std::nullopt));
        if (value())
            return "check_kv_cas: the kit's key could not be cleared";
        const auto created = store.compare_and_swap(key, std::nullopt, bytes("owner-a"));
        if (!created || !*created || value() != "owner-a")
            return "kv_cas: expected-absent on an absent key swaps";
        const auto again = store.compare_and_swap(key, std::nullopt, bytes("owner-b"));
        if (!again || *again || value() != "owner-a")
            return "kv_cas: expected-absent on a present key refuses and changes nothing";
        const auto wrong = store.compare_and_swap(key, bytes("owner-x"), bytes("owner-b"));
        if (!wrong || *wrong || value() != "owner-a")
            return "kv_cas: a wrong expected value refuses and changes nothing";
        const auto moved = store.compare_and_swap(key, bytes("owner-a"), bytes("owner-b"));
        if (!moved || !*moved || value() != "owner-b")
            return "kv_cas: the right expected value swaps";
        const auto prefix = store.compare_and_swap(key, bytes("owner"), std::nullopt);
        if (!prefix || *prefix || value() != "owner-b")
            return "kv_cas: a prefix of the value is not the value";
        const auto erased = store.compare_and_swap(key, bytes("owner-b"), std::nullopt);
        if (!erased || !*erased || value())
            return "kv_cas: a desired nullopt erases";
        const auto gone = store.compare_and_swap(key, bytes("owner-b"), bytes("owner-c"));
        if (!gone || *gone || value())
            return "kv_cas: an expected value on an absent key refuses";
        return std::nullopt;
    }
}

#endif /* snicholls_core_interfaces_kv_cas_hpp */
