//
//  kv_backup.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The virtual twin of concepts/kv_backup.hpp, with the adapters both ways.
//

/// \file
/// \brief The backup capability, virtual form: `kv_backup_interface` and `kv_backup_adapter`.
/// \ingroup core
///
/// **Overview.** `checkpoint(destination)` and `verify()` as pure virtual functions, for a store chosen at run
/// time. The interface models the concept `kv_backup` (`concepts/kv_backup.hpp`); `kv_backup_adapter<S>` puts any
/// model of the concept behind the interface, owning it or referring to it (`S&`).
///
/// \snippet interfaces/kv_backup_demo.cpp guard
///
/// **Relationships.** The concept and its reports: `concepts/kv_backup.hpp`. The store contract:
/// `key_value_store_interface` (`interfaces/key_value_store.hpp`); a backend derives from both. Group: \ref core.
///
/// **Implementations / variants.** `kv_backup_adapter<S>`; backends above core derive directly.
///
/// **Guarantees.** Those of the concept. The adapter lets no exception of the model through:
/// `std::bad_alloc` becomes `kv_errc::out_of_memory`, anything else `kv_errc::backend`.
///
/// **Errors.** As the concept.
///
/// **Examples.** See the overview.
///
/// **Performance.** Not benchmarked: backend-specific.
///
/// **Tests and demos.** `tests/interfaces/kv_backup_test.cpp`, `demos/interfaces/kv_backup_demo.cpp`.
///
/// **Dependencies.** `<string_view>`, `<type_traits>`, `<utility>`, and `concepts/kv_backup.hpp`,
/// `interfaces/key_value_store.hpp` (for the exception translation).
///
/// **Known issues.** None known.
///
/// **Status.** New in 0.22.0. Experimental.

#ifndef snicholls_core_interfaces_kv_backup_hpp
#define snicholls_core_interfaces_kv_backup_hpp

#include <snicholls/core/concepts/kv_backup.hpp>
#include <snicholls/core/interfaces/key_value_store.hpp>

#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief A store's checkpoint and consistency check, behind a virtual interface. Models `kv_backup`.
    /// \ingroup core
    class kv_backup_interface {
    public:
        /// \brief Destroys the object.
        virtual ~kv_backup_interface() = default;

        /// \brief Makes a consistent copy of the store at `destination`.
        /// \param destination where, in the backend's terms; must not hold a checkpoint already.
        /// \return what was captured, or the error (`kv_errc::invalid_argument` for an existing destination).
        virtual result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view destination) = 0;

        /// \brief Checks the stored data.
        /// \return what the check found, or the error that stopped it running.
        virtual result<kv_verify_report, kv_errc> verify() const = 0;

    protected:
        /// \brief Constructs the base.
        kv_backup_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        kv_backup_interface(const kv_backup_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        kv_backup_interface& operator=(const kv_backup_interface&) = default;
    };

    /// \brief Any model of `kv_backup` behind `kv_backup_interface`: owned (`S`) or referred to (`S&`).
    /// \ingroup core
    /// \tparam S the model, or a reference to one.
    template <typename S>
        requires kv_backup<std::remove_cvref_t<S>>
    class kv_backup_adapter final : public kv_backup_interface {
    public:
        /// \brief Takes (or, for a reference `S`, refers to) the model.
        /// \param model the model.
        explicit kv_backup_adapter(S model) : model_(std::forward<S>(model)) {}

        /// \brief The adapted model.
        /// \return it.
        [[nodiscard]] std::remove_reference_t<S>& get() noexcept { return model_; }
        /// \brief The adapted model.
        /// \return it.
        [[nodiscard]] const std::remove_reference_t<S>& get() const noexcept { return model_; }

        /// \brief See `kv_backup_interface::checkpoint`.
        /// \param destination where.
        /// \return as the model, its exceptions translated.
        result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view destination) override
        {
            return detail::kv_guard<result<kv_checkpoint_info, kv_errc>>(false, [&] { return get().checkpoint(destination); });
        }
        /// \brief See `kv_backup_interface::verify`.
        /// \return as the model, its exceptions translated.
        result<kv_verify_report, kv_errc> verify() const override
        {
            return detail::kv_guard<result<kv_verify_report, kv_errc>>(false, [&] { return get().verify(); });
        }

    private:
        S model_;
    };

    /// \brief Deduces an owning adapter.
    /// \ingroup core
    template <typename S>
    kv_backup_adapter(S) -> kv_backup_adapter<S>;
}

#endif /* snicholls_core_interfaces_kv_backup_hpp */
