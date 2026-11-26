//
//  token_verifier.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::token_verifier_interface`: the pure virtual twin of `token_verifier`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the verifier is chosen by configuration (a JWT verifier in production, the reference in a
/// test), code holds a `token_verifier_interface&`. Its pure virtual `verify` is the operation of the concept
/// `token_verifier` (`concepts/token_verifier.hpp`, where the contract and `token_claims` are documented), so the
/// interface models the concept itself. `token_verifier_adapter<V>` puts any model behind the interface, owning it
/// (`V`) or referring to it (`V&`); `token_verifier_ref` is a copyable, non-owning handle on the interface that models
/// the concept.
///
/// \snippet interfaces/token_verifier_demo.cpp virtual
///
/// **Relationships.** `concepts/token_verifier.hpp`, `interfaces/token_verifier_reference.hpp`,
/// `interfaces/token_verifier_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `token_verifier_adapter<V>` over any model; an implementation above core may derive
/// from the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; the verifier's own cost dominates.
///
/// **Tests and demos.** `tests/interfaces/token_verifier_test.cpp` (the kit through every form, the adapters forward),
/// `demos/interfaces/token_verifier_demo.cpp`.
///
/// **Dependencies.** `<string_view>`, `<type_traits>`, `<utility>`, `concepts/token_verifier.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_token_verifier_hpp
#define snicholls_core_interfaces_token_verifier_hpp

#include <snicholls/core/concepts/token_verifier.hpp>

#include <string_view>
#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `token_verifier`.
    /// \ingroup core
    class token_verifier_interface {
    public:
        /// \brief Destroys a verifier.
        virtual ~token_verifier_interface() = default;
        /// \brief Verifies a presented token.
        /// \param token The token.
        /// \return The caller, or `unauthorized`, `stale` or another `capability_error`.
        [[nodiscard]] virtual capability_result<token_claims> verify(std::string_view token) const = 0;

    protected:
        /// \brief Constructs the base.
        token_verifier_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        token_verifier_interface(const token_verifier_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        token_verifier_interface& operator=(const token_verifier_interface&) = default;
    };

    static_assert(token_verifier<token_verifier_interface>,
                  "snicholls::token_verifier_interface must model snicholls::token_verifier");

    /// \brief Any model of `token_verifier` behind the interface: owned (`V`) or referred to (`V&`).
    /// \ingroup core
    /// \tparam V The model, or a reference to one.
    template <typename V>
        requires token_verifier<std::remove_reference_t<V>>
    class token_verifier_adapter final : public token_verifier_interface {
    public:
        /// \brief Takes (or, for a reference `V`, refers to) the implementation.
        /// \param impl The implementation.
        explicit token_verifier_adapter(V impl) noexcept(std::is_nothrow_constructible_v<V, V&&>) : impl_(std::forward<V>(impl)) {}
        /// \brief Verifies a presented token. Forwards to the model.
        /// \param token The token.
        /// \return The caller, or the error.
        [[nodiscard]] capability_result<token_claims> verify(std::string_view token) const override { return impl_.verify(token); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<V>& get() noexcept { return impl_; }

    private:
        V impl_;
    };

    /// \brief Deduces an owning adapter: `token_verifier_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename V>
    token_verifier_adapter(V) -> token_verifier_adapter<V>;

    /// \brief A copyable, non-owning handle on a `token_verifier_interface`; it models the concept.
    /// \ingroup core
    class token_verifier_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit token_verifier_ref(const token_verifier_interface& impl) noexcept : p_(&impl) {}
        /// \brief Verifies a presented token. Forwards to the interface.
        /// \param token The token.
        /// \return The caller, or the error.
        [[nodiscard]] capability_result<token_claims> verify(std::string_view token) const { return p_->verify(token); }

    private:
        const token_verifier_interface* p_;
    };

    static_assert(token_verifier<token_verifier_ref>, "snicholls::token_verifier_ref must model snicholls::token_verifier");

} // namespace snicholls

#endif // snicholls_core_interfaces_token_verifier_hpp
