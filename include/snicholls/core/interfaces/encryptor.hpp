//
//  encryptor.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::encryptor_interface`: the pure virtual twin of `encryptor`, and the adapters both ways.
/// \ingroup core
///
/// **Overview.** When the cipher is chosen at run time (by configuration, by the key's algorithm), code holds an
/// `encryptor_interface&` (base-encode-decode's `Encryptor`). Its pure virtual functions are the operations of the
/// concept `encryptor` (`concepts/encryptor.hpp`, where the contract and the envelope are documented), so the interface
/// models the concept. `encryptor_adapter<E>` puts any model behind the interface, owning it (`E`) or referring to it
/// (`E&`); `encryptor_ref` is a copyable, non-owning handle on the interface that models the concept.
///
/// \snippet interfaces/encryptor_demo.cpp virtual
///
/// **Relationships.** `concepts/encryptor.hpp`, `interfaces/encryptor_reference.hpp`,
/// `interfaces/encryptor_conformance.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `encryptor_adapter<E>` over any model; crypto's ciphers may derive from the
/// interface directly.
///
/// **Guarantees.** The adapters add one virtual call and nothing else; thread safety is the wrapped encryptor's.
///
/// **Errors.** As the contract.
///
/// **Performance.** One virtual call; not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/encryptor_test.cpp` (the kit through every form),
/// `demos/interfaces/encryptor_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<span>`, `<string_view>`, `<type_traits>`, `<utility>`, `<vector>`,
/// `concepts/encryptor.hpp`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_interfaces_encryptor_hpp
#define snicholls_core_interfaces_encryptor_hpp

#include <snicholls/core/concepts/encryptor.hpp>

#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls {

    /// \brief The pure virtual twin of `encryptor`.
    /// \ingroup core
    class encryptor_interface {
    public:
        /// \brief Destroys the encryptor.
        virtual ~encryptor_interface() = default;
        /// \brief Seals `plaintext` under the key `key_id`, authenticating `aad`.
        /// \param key_id    The key.
        /// \param plaintext The bytes to seal.
        /// \param aad       The associated data.
        /// \return The sealed envelope, or the error.
        virtual capability_result<std::vector<std::byte>> encrypt(std::string_view key_id, std::span<const std::byte> plaintext,
                                                                  std::span<const std::byte> aad) = 0;
        /// \brief Opens an envelope, if it is authentic with `aad`.
        /// \param sealed The envelope.
        /// \param aad    The associated data it was sealed with.
        /// \return The plaintext, or the error.
        [[nodiscard]] virtual capability_result<std::vector<std::byte>> decrypt(std::span<const std::byte> sealed,
                                                                                std::span<const std::byte> aad) const = 0;
        /// \brief The algorithm, e.g. `"AES-256-GCM"`, `"XChaCha20-Poly1305"`.
        /// \return Its name.
        [[nodiscard]] virtual std::string_view algorithm() const = 0;

    protected:
        /// \brief Constructs the base.
        encryptor_interface() = default;
        /// \brief Copies the base (there is nothing to copy).
        encryptor_interface(const encryptor_interface&) = default;
        /// \brief Assigns the base (there is nothing to assign).
        /// \return `*this`.
        encryptor_interface& operator=(const encryptor_interface&) = default;
    };

    static_assert(encryptor<encryptor_interface>, "snicholls::encryptor_interface must model snicholls::encryptor");

    /// \brief Any model of `encryptor` behind the interface: owned (`E`) or referred to (`E&`).
    /// \ingroup core
    /// \tparam E The model, or a reference to one.
    template <typename E>
        requires encryptor<std::remove_reference_t<E>>
    class encryptor_adapter final : public encryptor_interface {
    public:
        /// \brief Takes (or, for a reference `E`, refers to) the encryptor.
        /// \param impl The encryptor.
        explicit encryptor_adapter(E impl) noexcept(std::is_nothrow_constructible_v<E, E&&>) : impl_(std::forward<E>(impl)) {}
        /// \brief Seals through the model.
        /// \param key_id    The key.
        /// \param plaintext The bytes.
        /// \param aad       The associated data.
        /// \return As the model.
        capability_result<std::vector<std::byte>> encrypt(std::string_view key_id, std::span<const std::byte> plaintext,
                                                          std::span<const std::byte> aad) override
        {
            return impl_.encrypt(key_id, plaintext, aad);
        }
        /// \brief Opens through the model.
        /// \param sealed The envelope.
        /// \param aad    The associated data.
        /// \return As the model.
        [[nodiscard]] capability_result<std::vector<std::byte>> decrypt(std::span<const std::byte> sealed,
                                                                        std::span<const std::byte> aad) const override
        {
            return impl_.decrypt(sealed, aad);
        }
        /// \brief Asks the model.
        /// \return As the model.
        [[nodiscard]] std::string_view algorithm() const override { return impl_.algorithm(); }
        /// \brief The adapted encryptor.
        /// \return It.
        [[nodiscard]] std::remove_reference_t<E>& get() noexcept { return impl_; }

    private:
        E impl_;
    };

    /// \brief Deduces an owning adapter: `encryptor_adapter(std::move(e))`.
    /// \ingroup core
    template <typename E>
    encryptor_adapter(E) -> encryptor_adapter<E>;

    /// \brief A copyable, non-owning handle on an `encryptor_interface`; it models the concept.
    /// \ingroup core
    class encryptor_ref {
    public:
        /// \brief Refers to `e`, which must outlive the handle.
        /// \param e The encryptor.
        explicit encryptor_ref(encryptor_interface& e) noexcept : p_(&e) {}
        /// \brief Seals through the interface.
        /// \param key_id    The key.
        /// \param plaintext The bytes.
        /// \param aad       The associated data.
        /// \return As the interface.
        capability_result<std::vector<std::byte>> encrypt(std::string_view key_id, std::span<const std::byte> plaintext,
                                                          std::span<const std::byte> aad)
        {
            return p_->encrypt(key_id, plaintext, aad);
        }
        /// \brief Opens through the interface.
        /// \param sealed The envelope.
        /// \param aad    The associated data.
        /// \return As the interface.
        [[nodiscard]] capability_result<std::vector<std::byte>> decrypt(std::span<const std::byte> sealed,
                                                                        std::span<const std::byte> aad) const
        {
            return p_->decrypt(sealed, aad);
        }
        /// \brief Asks the interface.
        /// \return As the interface.
        [[nodiscard]] std::string_view algorithm() const { return p_->algorithm(); }

    private:
        encryptor_interface* p_;
    };

    static_assert(encryptor<encryptor_ref>, "snicholls::encryptor_ref must model snicholls::encryptor");

} // namespace snicholls

#endif // snicholls_core_interfaces_encryptor_hpp
