//
//  security_provider.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::security_provider_interface`: the pure virtual twin of `security_provider`, and the adapters both
/// ways.
/// \ingroup core
///
/// **Overview.** When the broker, custodian or registry is chosen per account, code holds a
/// `security_provider_interface&`. Its pure virtual functions (`capabilities`, `verify_webhook`, `resolve`, `statement`,
/// `holdings`) are the operations of the concept `security_provider` (`concepts/security_provider.hpp`, where the
/// contract and its value types are documented), so the interface models the concept itself.
/// `security_provider_adapter<P>` puts any model behind the interface, owning it (`P`) or referring to it (`P&`);
/// `security_provider_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/security_provider_demo.cpp virtual
///
/// **Relationships.** `concepts/security_provider.hpp`, `interfaces/security_provider_reference.hpp`,
/// `interfaces/security_provider_conformance.hpp`; the sibling `interfaces/payment_provider.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `security_provider_adapter<P>` over any model; an implementation above core may
/// derive from the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call. Not benchmarked: the provider's own I/O dominates.
///
/// **Tests and demos.** `tests/interfaces/security_provider_test.cpp` (the kit through every form, the adapters
/// forward), `demos/interfaces/security_provider_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<type_traits>`, `<utility>`, `<vector>`, `concepts/security_provider.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.26.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_security_provider_hpp
#define snicholls_core_interfaces_security_provider_hpp

#include <snicholls/core/concepts/security_provider.hpp>

#include <chrono>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The pure virtual twin of `security_provider`.
    /// \ingroup core
    class security_provider_interface {
    public:
        /// \brief Destroys a security provider.
        virtual ~security_provider_interface() = default;
        /// \brief What the provider reports.
        /// \return Its capabilities.
        [[nodiscard]] virtual security_capabilities capabilities() const = 0;
        /// \brief Authenticates and parses a webhook delivery; fails closed; no I/O.
        /// \param w The delivery.
        /// \return The event, or the error.
        [[nodiscard]] virtual capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const = 0;
        /// \brief The authoritative facts behind a verified event.
        /// \param e A verified event.
        /// \return The facts, or the error.
        virtual capability_result<std::vector<security_fact>> resolve(const webhook_event& e) = 0;
        /// \brief A page of statement lines, as facts.
        /// \param q The page.
        /// \return The page, or the error.
        virtual capability_result<security_statement_page> statement(const security_statement_query& q) = 0;
        /// \brief The positions held for us at the end of a day.
        /// \param as_of The day.
        /// \return The holdings, or the error.
        virtual capability_result<std::vector<security_holding>> holdings(std::chrono::sys_days as_of) = 0;

    protected:
        /// \brief Constructs the base.
        security_provider_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        security_provider_interface(const security_provider_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        security_provider_interface& operator=(const security_provider_interface&) = default;
    };

    static_assert(security_provider<security_provider_interface>,
                  "snicholls::security_provider_interface must model snicholls::security_provider");

    /// \brief Any model of `security_provider` behind the interface: owned (`P`) or referred to (`P&`).
    /// \ingroup core
    /// \tparam P The model, or a reference to one.
    template <typename P>
        requires security_provider<std::remove_reference_t<P>>
    class security_provider_adapter final : public security_provider_interface {
    public:
        /// \brief Takes (or, for a reference `P`, refers to) the implementation.
        /// \param impl The implementation.
        explicit security_provider_adapter(P impl) noexcept(std::is_nothrow_constructible_v<P, P&&>) : impl_(std::forward<P>(impl)) {}
        /// \brief What the provider reports. Forwards to the model.
        /// \return Its capabilities.
        [[nodiscard]] security_capabilities capabilities() const override { return impl_.capabilities(); }
        /// \brief Authenticates and parses a webhook delivery. Forwards to the model.
        /// \param w The delivery.
        /// \return The event, or the error.
        [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const override { return impl_.verify_webhook(w); }
        /// \brief The authoritative facts behind a verified event. Forwards to the model.
        /// \param e A verified event.
        /// \return The facts, or the error.
        capability_result<std::vector<security_fact>> resolve(const webhook_event& e) override { return impl_.resolve(e); }
        /// \brief A page of statement lines, as facts. Forwards to the model.
        /// \param q The page.
        /// \return The page, or the error.
        capability_result<security_statement_page> statement(const security_statement_query& q) override { return impl_.statement(q); }
        /// \brief The positions held for us at the end of a day. Forwards to the model.
        /// \param as_of The day.
        /// \return The holdings, or the error.
        capability_result<std::vector<security_holding>> holdings(std::chrono::sys_days as_of) override { return impl_.holdings(as_of); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<P>& get() noexcept { return impl_; }

    private:
        P impl_;
    };

    /// \brief Deduces an owning adapter: `security_provider_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename P>
    security_provider_adapter(P) -> security_provider_adapter<P>;

    /// \brief A copyable, non-owning handle on a `security_provider_interface`; it models the concept.
    /// \ingroup core
    class security_provider_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit security_provider_ref(security_provider_interface& impl) noexcept : p_(&impl) {}
        /// \brief What the provider reports. Forwards to the interface.
        /// \return Its capabilities.
        [[nodiscard]] security_capabilities capabilities() const { return p_->capabilities(); }
        /// \brief Authenticates and parses a webhook delivery. Forwards to the interface.
        /// \param w The delivery.
        /// \return The event, or the error.
        [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const { return p_->verify_webhook(w); }
        /// \brief The authoritative facts behind a verified event. Forwards to the interface.
        /// \param e A verified event.
        /// \return The facts, or the error.
        capability_result<std::vector<security_fact>> resolve(const webhook_event& e) { return p_->resolve(e); }
        /// \brief A page of statement lines, as facts. Forwards to the interface.
        /// \param q The page.
        /// \return The page, or the error.
        capability_result<security_statement_page> statement(const security_statement_query& q) { return p_->statement(q); }
        /// \brief The positions held for us at the end of a day. Forwards to the interface.
        /// \param as_of The day.
        /// \return The holdings, or the error.
        capability_result<std::vector<security_holding>> holdings(std::chrono::sys_days as_of) { return p_->holdings(as_of); }

    private:
        security_provider_interface* p_;
    };

    static_assert(security_provider<security_provider_ref>, "snicholls::security_provider_ref must model snicholls::security_provider");

} // namespace snicholls

#endif // snicholls_core_interfaces_security_provider_hpp
