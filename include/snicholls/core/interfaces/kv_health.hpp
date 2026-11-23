//
//  kv_health.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The virtual twin of concepts/kv_health.hpp, with the adapters both ways.
//

/// \file
/// \brief The health capability, virtual form: `kv_health_interface` and `kv_health_adapter`.
/// \ingroup core
///
/// **Overview.** `ping()` and `diagnostics()` as pure virtual functions, for a store chosen at run time. The
/// interface models the concept `kv_health` (`concepts/kv_health.hpp`); `kv_health_adapter<S>` puts any model of
/// the concept behind the interface, owning it or referring to it (`S&`).
///
/// \snippet interfaces/kv_health_demo.cpp probe
///
/// **Relationships.** The concept and `kv_diagnostics`: `concepts/kv_health.hpp`. The store contract:
/// `key_value_store_interface` (`interfaces/key_value_store.hpp`); a backend derives from both, and a caller
/// holding the store asks for health with `dynamic_cast<const kv_health_interface*>`. Group: \ref core.
///
/// **Implementations / variants.** `kv_health_adapter<S>`; backends above core derive directly.
///
/// **Guarantees.** Those of the concept. The adapter lets no exception of the model through:
/// `std::bad_alloc` becomes `kv_errc::out_of_memory`, anything else `kv_errc::backend`.
///
/// **Errors.** As the concept.
///
/// **Examples.** See the overview.
///
/// **Performance.** Not benchmarked: one call per probe.
///
/// **Tests and demos.** `tests/interfaces/kv_health_test.cpp`, `demos/interfaces/kv_health_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, and `concepts/kv_health.hpp`, `interfaces/key_value_store.hpp`
/// (for the exception translation).
///
/// **Known issues.** None known.
///
/// **Status.** New in 0.22.0. Experimental.

#ifndef snicholls_core_interfaces_kv_health_hpp
#define snicholls_core_interfaces_kv_health_hpp

#include <snicholls/core/concepts/kv_health.hpp>
#include <snicholls/core/interfaces/key_value_store.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief A store's health, behind a virtual interface. Models `kv_health`.
    /// \ingroup core
    class kv_health_interface {
    public:
        /// \brief Destroys the object.
        virtual ~kv_health_interface() = default;

        /// \brief Whether the store can serve requests now; changes nothing.
        /// \return nothing, or the error that stops it serving.
        virtual result<void, kv_errc> ping() const = 0;

        /// \brief What the store holds, as plain data.
        /// \return the diagnostics, or the error.
        virtual result<kv_diagnostics, kv_errc> diagnostics() const = 0;

    protected:
        /// \brief Constructs the base.
        kv_health_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        kv_health_interface(const kv_health_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        kv_health_interface& operator=(const kv_health_interface&) = default;
    };

    /// \brief Any model of `kv_health` behind `kv_health_interface`: owned (`S`) or referred to (`S&`).
    /// \ingroup core
    /// \tparam S the model, or a reference to one.
    template <typename S>
        requires kv_health<std::remove_cvref_t<S>>
    class kv_health_adapter final : public kv_health_interface {
    public:
        /// \brief Takes (or, for a reference `S`, refers to) the model.
        /// \param model the model.
        explicit kv_health_adapter(S model) : model_(std::forward<S>(model)) {}

        /// \brief The adapted model.
        /// \return it.
        [[nodiscard]] const std::remove_reference_t<S>& get() const noexcept { return model_; }

        /// \brief See `kv_health_interface::ping`.
        /// \return as the model, its exceptions translated.
        result<void, kv_errc> ping() const override
        {
            return detail::kv_guard<result<void, kv_errc>>(false, [&] { return get().ping(); });
        }
        /// \brief See `kv_health_interface::diagnostics`.
        /// \return as the model, its exceptions translated.
        result<kv_diagnostics, kv_errc> diagnostics() const override
        {
            return detail::kv_guard<result<kv_diagnostics, kv_errc>>(false, [&] { return get().diagnostics(); });
        }

    private:
        S model_;
    };

    /// \brief Deduces an owning adapter.
    /// \ingroup core
    template <typename S>
    kv_health_adapter(S) -> kv_health_adapter<S>;
}

#endif /* snicholls_core_interfaces_kv_health_hpp */
