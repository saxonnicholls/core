//
//  onramp_provider.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::onramp_provider_interface`: the pure virtual twin of `onramp_provider`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the venue is chosen by configuration, code holds a `onramp_provider_interface&`. Its pure virtual functions (`assets`, `quote`, `buy`, `order_status`, `withdraw`, `withdrawal_status`, `balances`, `statement`) are the
/// operations of the concept `onramp_provider` (`concepts/onramp_provider.hpp`, where the contract and its value types are documented),
/// so the interface models the concept itself. `onramp_provider_adapter<O>` puts any model behind the interface, owning it (`O`) or
/// referring to it (`O&`); `onramp_provider_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/onramp_provider_demo.cpp virtual
///
/// **Relationships.** `concepts/onramp_provider.hpp`, `interfaces/onramp_provider_reference.hpp`, `interfaces/onramp_provider_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations / variants.** `onramp_provider_adapter<O>` over any model; an implementation above core may derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; not benchmarked (a venue is I/O bound).
///
/// **Tests and demos.** `tests/interfaces/onramp_provider_test.cpp` (the kit through every form, the adapters forward),
/// `demos/interfaces/onramp_provider_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/onramp_provider.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_onramp_provider_hpp
#define snicholls_core_interfaces_onramp_provider_hpp

#include <snicholls/core/concepts/onramp_provider.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `onramp_provider`.
    /// \ingroup core
    class onramp_provider_interface {
    public:
        /// \brief Destroys an on-ramp provider.
        virtual ~onramp_provider_interface() = default;
        /// \brief What the venue delivers.
        /// \return The assets.
        [[nodiscard]] virtual std::vector<onramp_asset> assets() const = 0;
        /// \brief A price for a ticket.
        /// \param q The request.
        /// \return The quote, or the error.
        virtual capability_result<onramp_quote> quote(const onramp_quote_request& q) = 0;
        /// \brief Buys, idempotently on the client order id.
        /// \param b The request.
        /// \return The order, or the error.
        virtual capability_result<order> buy(const buy_request& b) = 0;
        /// \brief An order's state.
        /// \param order_id The venue's order id.
        /// \return The order, or the error.
        virtual capability_result<order> order_status(std::string_view order_id) = 0;
        /// \brief Withdraws to an address, idempotently on the client reference.
        /// \param w The request.
        /// \return The withdrawal, or the error.
        virtual capability_result<withdrawal> withdraw(const withdraw_request& w) = 0;
        /// \brief A withdrawal's state.
        /// \param withdrawal_id The venue's withdrawal id.
        /// \return The withdrawal, or the error.
        virtual capability_result<withdrawal> withdrawal_status(std::string_view withdrawal_id) = 0;
        /// \brief The balances at the venue.
        /// \return The balances, or the error.
        virtual capability_result<std::vector<venue_balance>> balances() = 0;
        /// \brief A page of the venue's facts for the ledger.
        /// \param q The page.
        /// \return The page, or the error.
        virtual capability_result<onramp_statement_page> statement(const onramp_statement_query& q) = 0;

    protected:
        /// \brief Constructs the base.
        onramp_provider_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        onramp_provider_interface(const onramp_provider_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        onramp_provider_interface& operator=(const onramp_provider_interface&) = default;
    };

    static_assert(onramp_provider<onramp_provider_interface>, "snicholls::onramp_provider_interface must model snicholls::onramp_provider");

    /// \brief Any model of `onramp_provider` behind the interface: owned (`O`) or referred to (`O&`).
    /// \ingroup core
    /// \tparam O The model, or a reference to one.
    template <typename O>
        requires onramp_provider<std::remove_reference_t<O>>
    class onramp_provider_adapter final : public onramp_provider_interface {
    public:
        /// \brief Takes (or, for a reference `O`, refers to) the implementation.
        /// \param impl The implementation.
        explicit onramp_provider_adapter(O impl) noexcept(std::is_nothrow_constructible_v<O, O&&>) : impl_(std::forward<O>(impl)) {}
        /// \brief What the venue delivers. Forwards to the model.
        /// \return The assets.
        [[nodiscard]] std::vector<onramp_asset> assets() const override { return impl_.assets(); }
        /// \brief A price for a ticket. Forwards to the model.
        /// \param q The request.
        /// \return The quote, or the error.
        capability_result<onramp_quote> quote(const onramp_quote_request& q) override { return impl_.quote(q); }
        /// \brief Buys, idempotently on the client order id. Forwards to the model.
        /// \param b The request.
        /// \return The order, or the error.
        capability_result<order> buy(const buy_request& b) override { return impl_.buy(b); }
        /// \brief An order's state. Forwards to the model.
        /// \param order_id The venue's order id.
        /// \return The order, or the error.
        capability_result<order> order_status(std::string_view order_id) override { return impl_.order_status(order_id); }
        /// \brief Withdraws to an address, idempotently on the client reference. Forwards to the model.
        /// \param w The request.
        /// \return The withdrawal, or the error.
        capability_result<withdrawal> withdraw(const withdraw_request& w) override { return impl_.withdraw(w); }
        /// \brief A withdrawal's state. Forwards to the model.
        /// \param withdrawal_id The venue's withdrawal id.
        /// \return The withdrawal, or the error.
        capability_result<withdrawal> withdrawal_status(std::string_view withdrawal_id) override { return impl_.withdrawal_status(withdrawal_id); }
        /// \brief The balances at the venue. Forwards to the model.
        /// \return The balances, or the error.
        capability_result<std::vector<venue_balance>> balances() override { return impl_.balances(); }
        /// \brief A page of the venue's facts for the ledger. Forwards to the model.
        /// \param q The page.
        /// \return The page, or the error.
        capability_result<onramp_statement_page> statement(const onramp_statement_query& q) override { return impl_.statement(q); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<O>& get() noexcept { return impl_; }

    private:
        O impl_;
    };

    /// \brief Deduces an owning adapter: `onramp_provider_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename O>
    onramp_provider_adapter(O) -> onramp_provider_adapter<O>;

    /// \brief A copyable, non-owning handle on a `onramp_provider_interface`; it models the concept.
    /// \ingroup core
    class onramp_provider_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit onramp_provider_ref(onramp_provider_interface& impl) noexcept : p_(&impl) {}
        /// \brief What the venue delivers. Forwards to the interface.
        /// \return The assets.
        [[nodiscard]] std::vector<onramp_asset> assets() const { return p_->assets(); }
        /// \brief A price for a ticket. Forwards to the interface.
        /// \param q The request.
        /// \return The quote, or the error.
        capability_result<onramp_quote> quote(const onramp_quote_request& q) { return p_->quote(q); }
        /// \brief Buys, idempotently on the client order id. Forwards to the interface.
        /// \param b The request.
        /// \return The order, or the error.
        capability_result<order> buy(const buy_request& b) { return p_->buy(b); }
        /// \brief An order's state. Forwards to the interface.
        /// \param order_id The venue's order id.
        /// \return The order, or the error.
        capability_result<order> order_status(std::string_view order_id) { return p_->order_status(order_id); }
        /// \brief Withdraws to an address, idempotently on the client reference. Forwards to the interface.
        /// \param w The request.
        /// \return The withdrawal, or the error.
        capability_result<withdrawal> withdraw(const withdraw_request& w) { return p_->withdraw(w); }
        /// \brief A withdrawal's state. Forwards to the interface.
        /// \param withdrawal_id The venue's withdrawal id.
        /// \return The withdrawal, or the error.
        capability_result<withdrawal> withdrawal_status(std::string_view withdrawal_id) { return p_->withdrawal_status(withdrawal_id); }
        /// \brief The balances at the venue. Forwards to the interface.
        /// \return The balances, or the error.
        capability_result<std::vector<venue_balance>> balances() { return p_->balances(); }
        /// \brief A page of the venue's facts for the ledger. Forwards to the interface.
        /// \param q The page.
        /// \return The page, or the error.
        capability_result<onramp_statement_page> statement(const onramp_statement_query& q) { return p_->statement(q); }

    private:
        onramp_provider_interface* p_;
    };

    static_assert(onramp_provider<onramp_provider_ref>, "snicholls::onramp_provider_ref must model snicholls::onramp_provider");

} // namespace snicholls

#endif // snicholls_core_interfaces_onramp_provider_hpp
