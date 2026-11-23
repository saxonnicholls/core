//
//  payment_provider.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::payment_provider_interface`: the pure virtual twin of `payment_provider`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When Stripe or Wise is chosen per account, code holds a `payment_provider_interface&`. Its pure virtual functions (`capabilities`, `collect`, `payout`, `refund`, `quote`, `status`, `balances`, `credits`, `verify_webhook`, `resolve`, `statement`) are the
/// operations of the concept `payment_provider` (`concepts/payment_provider.hpp`, where the contract and its value types are documented),
/// so the interface models the concept itself. `payment_provider_adapter<P>` puts any model behind the interface, owning it (`P`) or
/// referring to it (`P&`); `payment_provider_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/payment_provider_demo.cpp virtual
///
/// **Relationships.** `concepts/payment_provider.hpp`, `interfaces/payment_provider_reference.hpp`, `interfaces/payment_provider_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations / variants.** `payment_provider_adapter<P>` over any model; an implementation above core may derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; `verify_webhook`, the hot path, is measured on the reference in `benchmarks/interfaces/payment_provider_reference_benchmark.cpp`.
///
/// **Tests and demos.** `tests/interfaces/payment_provider_test.cpp` (the kit through every form, the adapters forward),
/// `demos/interfaces/payment_provider_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/payment_provider.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_payment_provider_hpp
#define snicholls_core_interfaces_payment_provider_hpp

#include <snicholls/core/concepts/payment_provider.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `payment_provider`.
    /// \ingroup core
    class payment_provider_interface {
    public:
        /// \brief Destroys a payment provider.
        virtual ~payment_provider_interface() = default;
        /// \brief What the provider does.
        /// \return Its capabilities.
        [[nodiscard]] virtual provider_capabilities capabilities() const = 0;
        /// \brief Starts a collection.
        /// \param c The request.
        /// \return The payment and the payer's next step, or the error.
        virtual capability_result<collection> collect(const collection_request& c) = 0;
        /// \brief Pays a registered beneficiary.
        /// \param o The request.
        /// \return The payout, or the error.
        virtual capability_result<payment> payout(const payout_request& o) = 0;
        /// \brief Refunds a collection, in whole or in part.
        /// \param r The request.
        /// \return The refund, or the error.
        virtual capability_result<payment> refund(const refund_request& r) = 0;
        /// \brief Quotes a currency conversion.
        /// \param q The request.
        /// \return The quote, or the error.
        virtual capability_result<fx_quote> quote(const quote_request& q) = 0;
        /// \brief The current state of a movement.
        /// \param payment_id The movement's id.
        /// \return The movement, or the error.
        virtual capability_result<payment> status(std::string_view payment_id) = 0;
        /// \brief The balance in each currency that has one.
        /// \return The balances, or the error.
        virtual capability_result<std::vector<money>> balances() = 0;
        /// \brief The credits received in a window.
        /// \param q The window.
        /// \return The credits, or the error.
        virtual capability_result<std::vector<credit>> credits(const credit_query& q) = 0;
        /// \brief Authenticates and parses a webhook delivery; fails closed; no I/O.
        /// \param w The delivery.
        /// \return The event, or the error.
        [[nodiscard]] virtual capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const = 0;
        /// \brief The authoritative facts behind a verified event.
        /// \param e A verified event.
        /// \return The facts, or the error.
        virtual capability_result<std::vector<payment_fact>> resolve(const webhook_event& e) = 0;
        /// \brief A page of statement lines, as facts.
        /// \param q The page.
        /// \return The page, or the error.
        virtual capability_result<statement_page> statement(const statement_query& q) = 0;

    protected:
        /// \brief Constructs the base.
        payment_provider_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        payment_provider_interface(const payment_provider_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        payment_provider_interface& operator=(const payment_provider_interface&) = default;
    };

    static_assert(payment_provider<payment_provider_interface>, "snicholls::payment_provider_interface must model snicholls::payment_provider");

    /// \brief Any model of `payment_provider` behind the interface: owned (`P`) or referred to (`P&`).
    /// \ingroup core
    /// \tparam P The model, or a reference to one.
    template <typename P>
        requires payment_provider<std::remove_reference_t<P>>
    class payment_provider_adapter final : public payment_provider_interface {
    public:
        /// \brief Takes (or, for a reference `P`, refers to) the implementation.
        /// \param impl The implementation.
        explicit payment_provider_adapter(P impl) noexcept(std::is_nothrow_constructible_v<P, P&&>) : impl_(std::forward<P>(impl)) {}
        /// \brief What the provider does. Forwards to the model.
        /// \return Its capabilities.
        [[nodiscard]] provider_capabilities capabilities() const override { return impl_.capabilities(); }
        /// \brief Starts a collection. Forwards to the model.
        /// \param c The request.
        /// \return The payment and the payer's next step, or the error.
        capability_result<collection> collect(const collection_request& c) override { return impl_.collect(c); }
        /// \brief Pays a registered beneficiary. Forwards to the model.
        /// \param o The request.
        /// \return The payout, or the error.
        capability_result<payment> payout(const payout_request& o) override { return impl_.payout(o); }
        /// \brief Refunds a collection, in whole or in part. Forwards to the model.
        /// \param r The request.
        /// \return The refund, or the error.
        capability_result<payment> refund(const refund_request& r) override { return impl_.refund(r); }
        /// \brief Quotes a currency conversion. Forwards to the model.
        /// \param q The request.
        /// \return The quote, or the error.
        capability_result<fx_quote> quote(const quote_request& q) override { return impl_.quote(q); }
        /// \brief The current state of a movement. Forwards to the model.
        /// \param payment_id The movement's id.
        /// \return The movement, or the error.
        capability_result<payment> status(std::string_view payment_id) override { return impl_.status(payment_id); }
        /// \brief The balance in each currency that has one. Forwards to the model.
        /// \return The balances, or the error.
        capability_result<std::vector<money>> balances() override { return impl_.balances(); }
        /// \brief The credits received in a window. Forwards to the model.
        /// \param q The window.
        /// \return The credits, or the error.
        capability_result<std::vector<credit>> credits(const credit_query& q) override { return impl_.credits(q); }
        /// \brief Authenticates and parses a webhook delivery; fails closed; no I/O. Forwards to the model.
        /// \param w The delivery.
        /// \return The event, or the error.
        [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const override { return impl_.verify_webhook(w); }
        /// \brief The authoritative facts behind a verified event. Forwards to the model.
        /// \param e A verified event.
        /// \return The facts, or the error.
        capability_result<std::vector<payment_fact>> resolve(const webhook_event& e) override { return impl_.resolve(e); }
        /// \brief A page of statement lines, as facts. Forwards to the model.
        /// \param q The page.
        /// \return The page, or the error.
        capability_result<statement_page> statement(const statement_query& q) override { return impl_.statement(q); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<P>& get() noexcept { return impl_; }

    private:
        P impl_;
    };

    /// \brief Deduces an owning adapter: `payment_provider_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename P>
    payment_provider_adapter(P) -> payment_provider_adapter<P>;

    /// \brief A copyable, non-owning handle on a `payment_provider_interface`; it models the concept.
    /// \ingroup core
    class payment_provider_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit payment_provider_ref(payment_provider_interface& impl) noexcept : p_(&impl) {}
        /// \brief What the provider does. Forwards to the interface.
        /// \return Its capabilities.
        [[nodiscard]] provider_capabilities capabilities() const { return p_->capabilities(); }
        /// \brief Starts a collection. Forwards to the interface.
        /// \param c The request.
        /// \return The payment and the payer's next step, or the error.
        capability_result<collection> collect(const collection_request& c) { return p_->collect(c); }
        /// \brief Pays a registered beneficiary. Forwards to the interface.
        /// \param o The request.
        /// \return The payout, or the error.
        capability_result<payment> payout(const payout_request& o) { return p_->payout(o); }
        /// \brief Refunds a collection, in whole or in part. Forwards to the interface.
        /// \param r The request.
        /// \return The refund, or the error.
        capability_result<payment> refund(const refund_request& r) { return p_->refund(r); }
        /// \brief Quotes a currency conversion. Forwards to the interface.
        /// \param q The request.
        /// \return The quote, or the error.
        capability_result<fx_quote> quote(const quote_request& q) { return p_->quote(q); }
        /// \brief The current state of a movement. Forwards to the interface.
        /// \param payment_id The movement's id.
        /// \return The movement, or the error.
        capability_result<payment> status(std::string_view payment_id) { return p_->status(payment_id); }
        /// \brief The balance in each currency that has one. Forwards to the interface.
        /// \return The balances, or the error.
        capability_result<std::vector<money>> balances() { return p_->balances(); }
        /// \brief The credits received in a window. Forwards to the interface.
        /// \param q The window.
        /// \return The credits, or the error.
        capability_result<std::vector<credit>> credits(const credit_query& q) { return p_->credits(q); }
        /// \brief Authenticates and parses a webhook delivery; fails closed; no I/O. Forwards to the interface.
        /// \param w The delivery.
        /// \return The event, or the error.
        [[nodiscard]] capability_result<webhook_event> verify_webhook(const webhook_delivery& w) const { return p_->verify_webhook(w); }
        /// \brief The authoritative facts behind a verified event. Forwards to the interface.
        /// \param e A verified event.
        /// \return The facts, or the error.
        capability_result<std::vector<payment_fact>> resolve(const webhook_event& e) { return p_->resolve(e); }
        /// \brief A page of statement lines, as facts. Forwards to the interface.
        /// \param q The page.
        /// \return The page, or the error.
        capability_result<statement_page> statement(const statement_query& q) { return p_->statement(q); }

    private:
        payment_provider_interface* p_;
    };

    static_assert(payment_provider<payment_provider_ref>, "snicholls::payment_provider_ref must model snicholls::payment_provider");

} // namespace snicholls

#endif // snicholls_core_interfaces_payment_provider_hpp
