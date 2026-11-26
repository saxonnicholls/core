//
//  http_client.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::http_client_interface`: the pure virtual twin of `http_client`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the client is chosen at run time (a real transport in production, a loopback in tests), code
/// holds an `http_client_interface&` (crtp's `http_client_interface`). Its pure virtual function is the operation of
/// the concept `http_client` (`concepts/http_client.hpp`, where the contract is documented), so the interface models
/// the concept. `http_client_adapter<C>` puts any model behind the interface, owning it (`C`) or referring to it
/// (`C&`); `http_client_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/http_client_demo.cpp virtual
///
/// **Relationships.** `concepts/http_client.hpp`, `interfaces/http_client_reference.hpp`,
/// `interfaces/http_client_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `http_client_adapter<C>` over any model.
///
/// **Guarantees.** The adapters add one virtual call; thread safety is the wrapped client's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/http_client_test.cpp`, `demos/interfaces/http_client_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/http_client.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_http_client_hpp
#define snicholls_core_interfaces_http_client_hpp

#include <snicholls/core/concepts/http_client.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `http_client`.
    /// \ingroup core
    class http_client_interface {
    public:
        /// \brief Destroys the client.
        virtual ~http_client_interface() = default;
        /// \brief Sends a request and waits for the response.
        /// \param r The request.
        /// \return The response (any status), or the error when there is none.
        virtual capability_result<http_response> send(const http_request& r) = 0;

    protected:
        /// \brief Constructs the base.
        http_client_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        http_client_interface(const http_client_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        http_client_interface& operator=(const http_client_interface&) = default;
    };

    static_assert(http_client<http_client_interface>, "snicholls::http_client_interface must model snicholls::http_client");

    /// \brief Any model of `http_client` behind the interface: owned (`C`) or referred to (`C&`).
    /// \ingroup core
    /// \tparam C The model, or a reference to one.
    template <typename C>
        requires http_client<std::remove_reference_t<C>>
    class http_client_adapter final : public http_client_interface {
    public:
        /// \brief Takes (or, for a reference `C`, refers to) the client.
        /// \param impl The client.
        explicit http_client_adapter(C impl) noexcept(std::is_nothrow_constructible_v<C, C&&>) : impl_(std::forward<C>(impl)) {}
        /// \brief Sends through the model.
        /// \param r The request.
        /// \return As the model.
        capability_result<http_response> send(const http_request& r) override { return impl_.send(r); }
        /// \brief The adapted client.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<C>& get() noexcept { return impl_; }

    private:
        C impl_;
    };

    /// \brief Deduces an owning adapter: `http_client_adapter(std::move(client))`.
    /// \ingroup core
    template <typename C>
    http_client_adapter(C) -> http_client_adapter<C>;

    /// \brief A copyable, non-owning handle on an `http_client_interface`; it models the concept.
    /// \ingroup core
    class http_client_ref {
    public:
        /// \brief Refers to `c`, which must outlive the handle.
        /// \param c The client.
        explicit http_client_ref(http_client_interface& c) noexcept : p_(&c) {}
        /// \brief Sends through the interface.
        /// \param r The request.
        /// \return As the interface.
        capability_result<http_response> send(const http_request& r) { return p_->send(r); }

    private:
        http_client_interface* p_;
    };

    static_assert(http_client<http_client_ref>, "snicholls::http_client_ref must model snicholls::http_client");

} // namespace snicholls

#endif // snicholls_core_interfaces_http_client_hpp
