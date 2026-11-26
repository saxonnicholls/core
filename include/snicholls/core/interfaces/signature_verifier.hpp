//
//  signature_verifier.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signature_verifier_interface`: the pure virtual twin of `signature_verifier`, and the adapters
/// both ways.
/// \ingroup core
///
/// **Overview.** When the verifier is chosen at run time (crypto's in production, the HS256 reference in a test), code
/// holds a `signature_verifier_interface&`. Its pure virtual `verify` is the operation of the concept
/// `signature_verifier` (`concepts/signature_verifier.hpp`, where the contract and `verify_request` are documented), so
/// the interface models the concept itself. `signature_verifier_adapter<V>` puts any model behind the interface, owning
/// it (`V`) or referring to it (`V&`); `signature_verifier_ref` is a copyable, non-owning handle on the interface that
/// models the concept.
///
/// \snippet interfaces/signature_verifier_demo.cpp virtual
///
/// **Relationships.** `concepts/signature_verifier.hpp`, `interfaces/signature_verifier_reference.hpp`,
/// `interfaces/signature_verifier_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `signature_verifier_adapter<V>` over any model; an implementation above core may
/// derive from the interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; the verification's own cost dominates.
///
/// **Tests and demos.** `tests/interfaces/signature_verifier_test.cpp`, `demos/interfaces/signature_verifier_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/signature_verifier.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_signature_verifier_hpp
#define snicholls_core_interfaces_signature_verifier_hpp

#include <snicholls/core/concepts/signature_verifier.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `signature_verifier`.
    /// \ingroup core
    class signature_verifier_interface {
    public:
        /// \brief Destroys a verifier.
        virtual ~signature_verifier_interface() = default;
        /// \brief Checks one signature.
        /// \param r The request.
        /// \return Success, or `signature_invalid`, `unsupported`, `invalid_argument`, `not_found`.
        [[nodiscard]] virtual capability_result<void> verify(const verify_request& r) const = 0;

    protected:
        /// \brief Constructs the base.
        signature_verifier_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        signature_verifier_interface(const signature_verifier_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        signature_verifier_interface& operator=(const signature_verifier_interface&) = default;
    };

    static_assert(signature_verifier<signature_verifier_interface>,
                  "snicholls::signature_verifier_interface must model snicholls::signature_verifier");

    /// \brief Any model of `signature_verifier` behind the interface: owned (`V`) or referred to (`V&`).
    /// \ingroup core
    /// \tparam V The model, or a reference to one.
    template <typename V>
        requires signature_verifier<std::remove_reference_t<V>>
    class signature_verifier_adapter final : public signature_verifier_interface {
    public:
        /// \brief Takes (or, for a reference `V`, refers to) the implementation.
        /// \param impl The implementation.
        explicit signature_verifier_adapter(V impl) noexcept(std::is_nothrow_constructible_v<V, V&&>) : impl_(std::forward<V>(impl)) {}
        /// \brief Checks one signature. Forwards to the model.
        /// \param r The request.
        /// \return Success, or the error.
        [[nodiscard]] capability_result<void> verify(const verify_request& r) const override { return impl_.verify(r); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<V>& get() noexcept { return impl_; }

    private:
        V impl_;
    };

    /// \brief Deduces an owning adapter: `signature_verifier_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename V>
    signature_verifier_adapter(V) -> signature_verifier_adapter<V>;

    /// \brief A copyable, non-owning handle on a `signature_verifier_interface`; it models the concept.
    /// \ingroup core
    class signature_verifier_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit signature_verifier_ref(const signature_verifier_interface& impl) noexcept : p_(&impl) {}
        /// \brief Checks one signature. Forwards to the interface.
        /// \param r The request.
        /// \return Success, or the error.
        [[nodiscard]] capability_result<void> verify(const verify_request& r) const { return p_->verify(r); }

    private:
        const signature_verifier_interface* p_;
    };

    static_assert(signature_verifier<signature_verifier_ref>,
                  "snicholls::signature_verifier_ref must model snicholls::signature_verifier");

} // namespace snicholls

#endif // snicholls_core_interfaces_signature_verifier_hpp
