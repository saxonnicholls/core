//
//  role_store.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::role_store_interface`: the pure virtual twin of `role_store`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the store is chosen by configuration (Postgres in production, the reference in a test), code
/// holds a `role_store_interface&`. Its pure virtual functions are the operations of the concept `role_store`
/// (`concepts/role_store.hpp`, where the contract and `role_grant` are documented), so the interface models the concept
/// itself. `role_store_adapter<S>` puts any model behind the interface, owning it (`S`) or referring to it (`S&`);
/// `role_store_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/role_store_demo.cpp virtual
///
/// **Relationships.** `concepts/role_store.hpp`, `interfaces/role_store_reference.hpp`,
/// `interfaces/role_store_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `role_store_adapter<S>` over any model; an implementation above core may derive from
/// the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; the store's own cost dominates.
///
/// **Tests and demos.** `tests/interfaces/role_store_test.cpp`, `demos/interfaces/role_store_demo.cpp`.
///
/// **Dependencies.** `<string_view>`, `<type_traits>`, `<utility>`, `<vector>`, `concepts/role_store.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_role_store_hpp
#define snicholls_core_interfaces_role_store_hpp

#include <snicholls/core/concepts/role_store.hpp>

#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The pure virtual twin of `role_store`.
    /// \ingroup core
    class role_store_interface {
    public:
        /// \brief Destroys a store.
        virtual ~role_store_interface() = default;
        /// \brief The party's active grants in the tenant.
        /// \param party  The party.
        /// \param tenant The tenant.
        /// \param now    The time.
        /// \return The grants, or the error.
        [[nodiscard]] virtual capability_result<std::vector<role_grant>> grants_of(const uuid& party, std::string_view tenant,
                                                                                   utc_s now) const = 0;
        /// \brief Every active grant in the tenant.
        /// \param tenant The tenant.
        /// \param now    The time.
        /// \return The grants, or the error.
        [[nodiscard]] virtual capability_result<std::vector<role_grant>> members(std::string_view tenant, utc_s now) const = 0;
        /// \brief Records a grant, replacing one of the same party, tenant and role.
        /// \param g            The grant.
        /// \param confirmation The sentence the actor confirmed, if any.
        /// \return Success, or the error.
        virtual capability_result<void> grant(const role_grant& g, std::string_view confirmation) = 0;
        /// \brief Records a revoke.
        /// \param party  The party.
        /// \param tenant The tenant.
        /// \param role   The role.
        /// \param actor  Who revoked it.
        /// \param reason Why.
        /// \param now    When.
        /// \return Success, or `not_found`, or the error.
        virtual capability_result<void> revoke(const uuid& party, std::string_view tenant, std::string_view role, const uuid& actor,
                                               std::string_view reason, utc_s now) = 0;
        /// \brief The tenant's history, oldest first.
        /// \param tenant The tenant.
        /// \return The events, or the error.
        [[nodiscard]] virtual capability_result<std::vector<role_event>> history(std::string_view tenant) const = 0;

    protected:
        /// \brief Constructs the base.
        role_store_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        role_store_interface(const role_store_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        role_store_interface& operator=(const role_store_interface&) = default;
    };

    static_assert(role_store<role_store_interface>, "snicholls::role_store_interface must model snicholls::role_store");

    /// \brief Any model of `role_store` behind the interface: owned (`S`) or referred to (`S&`).
    /// \ingroup core
    /// \tparam S The model, or a reference to one.
    template <typename S>
        requires role_store<std::remove_reference_t<S>>
    class role_store_adapter final : public role_store_interface {
    public:
        /// \brief Takes (or, for a reference `S`, refers to) the implementation.
        /// \param impl The implementation.
        explicit role_store_adapter(S impl) noexcept(std::is_nothrow_constructible_v<S, S&&>) : impl_(std::forward<S>(impl)) {}
        /// \brief The party's active grants in the tenant. Forwards to the model.
        /// \param party  The party.
        /// \param tenant The tenant.
        /// \param now    The time.
        /// \return The grants, or the error.
        [[nodiscard]] capability_result<std::vector<role_grant>> grants_of(const uuid& party, std::string_view tenant,
                                                                           utc_s now) const override
        {
            return impl_.grants_of(party, tenant, now);
        }
        /// \brief Every active grant in the tenant. Forwards to the model.
        /// \param tenant The tenant.
        /// \param now    The time.
        /// \return The grants, or the error.
        [[nodiscard]] capability_result<std::vector<role_grant>> members(std::string_view tenant, utc_s now) const override
        {
            return impl_.members(tenant, now);
        }
        /// \brief Records a grant. Forwards to the model.
        /// \param g            The grant.
        /// \param confirmation The sentence the actor confirmed, if any.
        /// \return Success, or the error.
        capability_result<void> grant(const role_grant& g, std::string_view confirmation) override { return impl_.grant(g, confirmation); }
        /// \brief Records a revoke. Forwards to the model.
        /// \param party  The party.
        /// \param tenant The tenant.
        /// \param role   The role.
        /// \param actor  Who revoked it.
        /// \param reason Why.
        /// \param now    When.
        /// \return Success, or the error.
        capability_result<void> revoke(const uuid& party, std::string_view tenant, std::string_view role, const uuid& actor,
                                       std::string_view reason, utc_s now) override
        {
            return impl_.revoke(party, tenant, role, actor, reason, now);
        }
        /// \brief The tenant's history. Forwards to the model.
        /// \param tenant The tenant.
        /// \return The events, or the error.
        [[nodiscard]] capability_result<std::vector<role_event>> history(std::string_view tenant) const override
        {
            return impl_.history(tenant);
        }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<S>& get() noexcept { return impl_; }

    private:
        S impl_;
    };

    /// \brief Deduces an owning adapter: `role_store_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename S>
    role_store_adapter(S) -> role_store_adapter<S>;

    /// \brief A copyable, non-owning handle on a `role_store_interface`; it models the concept.
    /// \ingroup core
    class role_store_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit role_store_ref(role_store_interface& impl) noexcept : p_(&impl) {}
        /// \brief The party's active grants in the tenant. Forwards to the interface.
        /// \param party  The party.
        /// \param tenant The tenant.
        /// \param now    The time.
        /// \return The grants, or the error.
        [[nodiscard]] capability_result<std::vector<role_grant>> grants_of(const uuid& party, std::string_view tenant, utc_s now) const
        {
            return p_->grants_of(party, tenant, now);
        }
        /// \brief Every active grant in the tenant. Forwards to the interface.
        /// \param tenant The tenant.
        /// \param now    The time.
        /// \return The grants, or the error.
        [[nodiscard]] capability_result<std::vector<role_grant>> members(std::string_view tenant, utc_s now) const
        {
            return p_->members(tenant, now);
        }
        /// \brief Records a grant. Forwards to the interface.
        /// \param g            The grant.
        /// \param confirmation The sentence the actor confirmed, if any.
        /// \return Success, or the error.
        capability_result<void> grant(const role_grant& g, std::string_view confirmation) { return p_->grant(g, confirmation); }
        /// \brief Records a revoke. Forwards to the interface.
        /// \param party  The party.
        /// \param tenant The tenant.
        /// \param role   The role.
        /// \param actor  Who revoked it.
        /// \param reason Why.
        /// \param now    When.
        /// \return Success, or the error.
        capability_result<void> revoke(const uuid& party, std::string_view tenant, std::string_view role, const uuid& actor,
                                       std::string_view reason, utc_s now)
        {
            return p_->revoke(party, tenant, role, actor, reason, now);
        }
        /// \brief The tenant's history. Forwards to the interface.
        /// \param tenant The tenant.
        /// \return The events, or the error.
        [[nodiscard]] capability_result<std::vector<role_event>> history(std::string_view tenant) const { return p_->history(tenant); }

    private:
        role_store_interface* p_;
    };

    static_assert(role_store<role_store_ref>, "snicholls::role_store_ref must model snicholls::role_store");

} // namespace snicholls

#endif // snicholls_core_interfaces_role_store_hpp
