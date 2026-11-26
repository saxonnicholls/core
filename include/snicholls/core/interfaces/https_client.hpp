//
//  https_client.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::https_client_interface`: the pure virtual twin of `https_client`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the transport is chosen at run time (TLS in production, the in-memory router or a replayed day in
/// a test), code holds an `https_client_interface&`. Its pure virtual `send` is the operation of the concept
/// `https_client` (`concepts/https_client.hpp`, where the contract and the request and response are documented), so
/// the interface models the concept itself. `https_client_adapter<C>` puts any model behind the interface, owning it
/// (`C`) or referring to it (`C&`); `https_client_ref` is a copyable, non-owning handle on the interface that models
/// the concept.
///
/// \snippet interfaces/https_client_demo.cpp virtual
///
/// **Relationships.** `concepts/https_client.hpp`, `interfaces/https_client_reference.hpp`,
/// `interfaces/https_client_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `https_client_adapter<C>` over any model; an implementation above core may derive
/// from the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; the network dominates.
///
/// **Tests and demos.** `tests/interfaces/https_client_test.cpp`, `demos/interfaces/https_client_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/https_client.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_https_client_hpp
#define snicholls_core_interfaces_https_client_hpp

#include <snicholls/core/concepts/https_client.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `https_client`.
    /// \ingroup core
    class https_client_interface {
    public:
        /// \brief Destroys a client.
        virtual ~https_client_interface() = default;
        /// \brief Sends one request.
        /// \param r The request.
        /// \return The response, whatever its status, or a transport error.
        virtual capability_result<https_response> send(const https_request& r) = 0;

    protected:
        /// \brief Constructs the base.
        https_client_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        https_client_interface(const https_client_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        https_client_interface& operator=(const https_client_interface&) = default;
    };

    static_assert(https_client<https_client_interface>, "snicholls::https_client_interface must model snicholls::https_client");

    /// \brief Any model of `https_client` behind the interface: owned (`C`) or referred to (`C&`).
    /// \ingroup core
    /// \tparam C The model, or a reference to one.
    template <typename C>
        requires https_client<std::remove_reference_t<C>>
    class https_client_adapter final : public https_client_interface {
    public:
        /// \brief Takes (or, for a reference `C`, refers to) the implementation.
        /// \param impl The implementation.
        explicit https_client_adapter(C impl) noexcept(std::is_nothrow_constructible_v<C, C&&>) : impl_(std::forward<C>(impl)) {}
        /// \brief Sends one request. Forwards to the model.
        /// \param r The request.
        /// \return The response, or the error.
        capability_result<https_response> send(const https_request& r) override { return impl_.send(r); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<C>& get() noexcept { return impl_; }

    private:
        C impl_;
    };

    /// \brief Deduces an owning adapter: `https_client_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename C>
    https_client_adapter(C) -> https_client_adapter<C>;

    /// \brief A copyable, non-owning handle on an `https_client_interface`; it models the concept.
    /// \ingroup core
    class https_client_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit https_client_ref(https_client_interface& impl) noexcept : p_(&impl) {}
        /// \brief Sends one request. Forwards to the interface.
        /// \param r The request.
        /// \return The response, or the error.
        capability_result<https_response> send(const https_request& r) { return p_->send(r); }

    private:
        https_client_interface* p_;
    };

    static_assert(https_client<https_client_ref>, "snicholls::https_client_ref must model snicholls::https_client");

} // namespace snicholls

#endif // snicholls_core_interfaces_https_client_hpp
