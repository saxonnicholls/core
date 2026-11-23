//
//  http_handler.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::http_handler_interface`: the pure virtual twin of `http_handler`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When routes are assembled at run time, code holds a `http_handler_interface&`. Its pure virtual functions (`handle`) are the
/// operations of the concept `http_handler` (`concepts/http_handler.hpp`, where the contract and its value types are documented),
/// so the interface models the concept itself. `http_handler_adapter<H>` puts any model behind the interface, owning it (`H`) or
/// referring to it (`H&`); `http_handler_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/http_handler_demo.cpp virtual
///
/// **Relationships.** `concepts/http_handler.hpp`, `interfaces/http_handler_reference.hpp`, `interfaces/http_handler_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations / variants.** `http_handler_adapter<H>` over any model; an implementation above core may derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call, measured against the concept and a hand-written dispatch in `benchmarks/interfaces/http_handler_benchmark.cpp`.
///
/// **Tests and demos.** `tests/interfaces/http_handler_test.cpp` (the kit through every form, the adapters forward),
/// `demos/interfaces/http_handler_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/http_handler.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_http_handler_hpp
#define snicholls_core_interfaces_http_handler_hpp

#include <snicholls/core/concepts/http_handler.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `http_handler`.
    /// \ingroup core
    class http_handler_interface {
    public:
        /// \brief Destroys an HTTP handler.
        virtual ~http_handler_interface() = default;
        /// \brief Answers one request.
        /// \param r The request.
        /// \return The response.
        virtual http_response handle(const http_request& r) = 0;

    protected:
        /// \brief Constructs the base.
        http_handler_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        http_handler_interface(const http_handler_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        http_handler_interface& operator=(const http_handler_interface&) = default;
    };

    static_assert(http_handler<http_handler_interface>, "snicholls::http_handler_interface must model snicholls::http_handler");

    /// \brief Any model of `http_handler` behind the interface: owned (`H`) or referred to (`H&`).
    /// \ingroup core
    /// \tparam H The model, or a reference to one.
    template <typename H>
        requires http_handler<std::remove_reference_t<H>>
    class http_handler_adapter final : public http_handler_interface {
    public:
        /// \brief Takes (or, for a reference `H`, refers to) the implementation.
        /// \param impl The implementation.
        explicit http_handler_adapter(H impl) noexcept(std::is_nothrow_constructible_v<H, H&&>) : impl_(std::forward<H>(impl)) {}
        /// \brief Answers one request. Forwards to the model.
        /// \param r The request.
        /// \return The response.
        http_response handle(const http_request& r) override { return impl_.handle(r); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<H>& get() noexcept { return impl_; }

    private:
        H impl_;
    };

    /// \brief Deduces an owning adapter: `http_handler_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename H>
    http_handler_adapter(H) -> http_handler_adapter<H>;

    /// \brief A copyable, non-owning handle on a `http_handler_interface`; it models the concept.
    /// \ingroup core
    class http_handler_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit http_handler_ref(http_handler_interface& impl) noexcept : p_(&impl) {}
        /// \brief Answers one request. Forwards to the interface.
        /// \param r The request.
        /// \return The response.
        http_response handle(const http_request& r) { return p_->handle(r); }

    private:
        http_handler_interface* p_;
    };

    static_assert(http_handler<http_handler_ref>, "snicholls::http_handler_ref must model snicholls::http_handler");

} // namespace snicholls

#endif // snicholls_core_interfaces_http_handler_hpp
