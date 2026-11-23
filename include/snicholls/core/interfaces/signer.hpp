//
//  signer.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::signer_interface`: the pure virtual twin of `signer`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the keys live in a KMS chosen by configuration, code holds a `signer_interface&`. Its pure virtual functions (`sign`, `public_key_of`) are the
/// operations of the concept `signer` (`concepts/signer.hpp`, where the contract and its value types are documented),
/// so the interface models the concept itself. `signer_adapter<S>` puts any model behind the interface, owning it (`S`) or
/// referring to it (`S&`); `signer_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/signer_demo.cpp virtual
///
/// **Relationships.** `concepts/signer.hpp`, `interfaces/signer_reference.hpp`, `interfaces/signer_conformance.hpp`.
/// Group: \ref core.
///
/// **Implementations / variants.** `signer_adapter<S>` over any model; an implementation above core may derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety and exception safety are the
/// wrapped implementation's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; the signer's own cost dominates (`benchmarks/interfaces/key_store_reference_benchmark.cpp`).
///
/// **Tests and demos.** `tests/interfaces/signer_test.cpp` (the kit through every form, the adapters forward),
/// `demos/interfaces/signer_demo.cpp`.
///
/// **Dependencies.** `<type_traits>`, `<utility>`, `concepts/signer.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_signer_hpp
#define snicholls_core_interfaces_signer_hpp

#include <snicholls/core/concepts/signer.hpp>

#include <type_traits>
#include <utility>

namespace snicholls {

    /// \brief The pure virtual twin of `signer`.
    /// \ingroup core
    class signer_interface {
    public:
        /// \brief Destroys a signer.
        virtual ~signer_interface() = default;
        /// \brief Signs with the key's current version.
        /// \param r The request.
        /// \return The signature, or the error.
        virtual capability_result<signature> sign(const sign_request& r) = 0;
        /// \brief The public key of a version.
        /// \param k The key and version.
        /// \return The key, or the error.
        [[nodiscard]] virtual capability_result<public_key> public_key_of(const key_ref& k) const = 0;

    protected:
        /// \brief Constructs the base.
        signer_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        signer_interface(const signer_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        signer_interface& operator=(const signer_interface&) = default;
    };

    static_assert(signer<signer_interface>, "snicholls::signer_interface must model snicholls::signer");

    /// \brief Any model of `signer` behind the interface: owned (`S`) or referred to (`S&`).
    /// \ingroup core
    /// \tparam S The model, or a reference to one.
    template <typename S>
        requires signer<std::remove_reference_t<S>>
    class signer_adapter final : public signer_interface {
    public:
        /// \brief Takes (or, for a reference `S`, refers to) the implementation.
        /// \param impl The implementation.
        explicit signer_adapter(S impl) noexcept(std::is_nothrow_constructible_v<S, S&&>) : impl_(std::forward<S>(impl)) {}
        /// \brief Signs with the key's current version. Forwards to the model.
        /// \param r The request.
        /// \return The signature, or the error.
        capability_result<signature> sign(const sign_request& r) override { return impl_.sign(r); }
        /// \brief The public key of a version. Forwards to the model.
        /// \param k The key and version.
        /// \return The key, or the error.
        [[nodiscard]] capability_result<public_key> public_key_of(const key_ref& k) const override { return impl_.public_key_of(k); }
        /// \brief The adapted implementation.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<S>& get() noexcept { return impl_; }

    private:
        S impl_;
    };

    /// \brief Deduces an owning adapter: `signer_adapter(std::move(impl))`.
    /// \ingroup core
    template <typename S>
    signer_adapter(S) -> signer_adapter<S>;

    /// \brief A copyable, non-owning handle on a `signer_interface`; it models the concept.
    /// \ingroup core
    class signer_ref {
    public:
        /// \brief Refers to `impl`, which must outlive the handle.
        /// \param impl The implementation.
        explicit signer_ref(signer_interface& impl) noexcept : p_(&impl) {}
        /// \brief Signs with the key's current version. Forwards to the interface.
        /// \param r The request.
        /// \return The signature, or the error.
        capability_result<signature> sign(const sign_request& r) { return p_->sign(r); }
        /// \brief The public key of a version. Forwards to the interface.
        /// \param k The key and version.
        /// \return The key, or the error.
        [[nodiscard]] capability_result<public_key> public_key_of(const key_ref& k) const { return p_->public_key_of(k); }

    private:
        signer_interface* p_;
    };

    static_assert(signer<signer_ref>, "snicholls::signer_ref must model snicholls::signer");

} // namespace snicholls

#endif // snicholls_core_interfaces_signer_hpp
