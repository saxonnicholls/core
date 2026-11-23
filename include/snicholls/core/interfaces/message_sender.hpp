//
//  message_sender.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::message_sender_interface`: the pure virtual twin of `message_sender`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the sender is chosen at run time (configuration names SES or SMTP), code holds a
/// `message_sender_interface&`. Its pure virtual functions are the operations of the concept `message_sender`
/// (`concepts/message_sender.hpp`, where the contract and its value types are documented), so the interface models
/// the concept itself. `message_sender_adapter<S>` puts any model behind the interface, owning it (`S`) or referring
/// to it (`S&`); `message_sender_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/message_sender_demo.cpp virtual
///
/// **Relationships.** `concepts/message_sender.hpp`, `interfaces/message_sender_reference.hpp`,
/// `interfaces/message_sender_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `message_sender_adapter<S>` over any model; a sender above core may derive from
/// the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety is the wrapped
/// implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; not benchmarked (a sender is I/O bound).
///
/// **Tests and demos.** `tests/interfaces/message_sender_test.cpp` (the kit through every form),
/// `demos/interfaces/message_sender_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/message_sender.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_message_sender_hpp
#define snicholls_core_interfaces_message_sender_hpp

#include <snicholls/core/concepts/message_sender.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `message_sender`.
    /// \ingroup core
    class message_sender_interface {
    public:
        /// \brief Destroys the sender.
        virtual ~message_sender_interface() = default;
        /// \brief Sends one message.
        /// \param m The message.
        /// \return The receipt, or the error.
        virtual capability_result<send_receipt> send(const message& m) = 0;
        /// \brief Whether the channel is supported.
        /// \param c The channel.
        /// \return `true` when `send` may succeed on it.
        [[nodiscard]] virtual bool supports(channel c) const = 0;

    protected:
        /// \brief Constructs the base.
        message_sender_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        message_sender_interface(const message_sender_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        message_sender_interface& operator=(const message_sender_interface&) = default;
    };

    static_assert(message_sender<message_sender_interface>, "snicholls::message_sender_interface must model snicholls::message_sender");

    /// \brief Any model of `message_sender` behind the interface: owned (`S`) or referred to (`S&`).
    /// \ingroup core
    /// \tparam S The model, or a reference to one.
    template <typename S>
        requires message_sender<std::remove_reference_t<S>>
    class message_sender_adapter final : public message_sender_interface {
    public:
        /// \brief Takes (or, for a reference `S`, refers to) the sender.
        /// \param impl The sender.
        explicit message_sender_adapter(S impl) noexcept(std::is_nothrow_constructible_v<S, S&&>) : impl_(std::forward<S>(impl)) {}
        /// \brief Sends through the model.
        /// \param m The message.
        /// \return As the model.
        capability_result<send_receipt> send(const message& m) override { return impl_.send(m); }
        /// \brief Asks the model.
        /// \param c The channel.
        /// \return As the model.
        [[nodiscard]] bool supports(channel c) const override { return impl_.supports(c); }
        /// \brief The adapted sender.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<S>& get() noexcept { return impl_; }

    private:
        S impl_;
    };

    /// \brief Deduces an owning adapter: `message_sender_adapter(std::move(sender))`.
    /// \ingroup core
    template <typename S>
    message_sender_adapter(S) -> message_sender_adapter<S>;

    /// \brief A copyable, non-owning handle on a `message_sender_interface`; it models the concept.
    /// \ingroup core
    class message_sender_ref {
    public:
        /// \brief Refers to `sender`, which must outlive the handle.
        /// \param sender The sender.
        explicit message_sender_ref(message_sender_interface& sender) noexcept : p_(&sender) {}
        /// \brief Sends through the interface.
        /// \param m The message.
        /// \return As the interface.
        capability_result<send_receipt> send(const message& m) { return p_->send(m); }
        /// \brief Asks the interface.
        /// \param c The channel.
        /// \return As the interface.
        [[nodiscard]] bool supports(channel c) const { return p_->supports(c); }

    private:
        message_sender_interface* p_;
    };

    static_assert(message_sender<message_sender_ref>, "snicholls::message_sender_ref must model snicholls::message_sender");

} // namespace snicholls

#endif // snicholls_core_interfaces_message_sender_hpp
