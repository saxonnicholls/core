//
//  encryptor_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::insecure_test_encryptor`: a test double of the `encryptor` contract. **It is not encryption.**
/// \ingroup core
///
/// **Overview.** **Not encryption: a test double. Never use it to protect anything.** It exists so that code written
/// against `encryptor` (an encrypted store, a sealed column, a secrets file) can be tested without a cipher library,
/// and so the kit has something to run on: it honours every rule of the contract (the envelope and its key id,
/// associated data, a fresh nonce per call, tamper detection, rotation, the error codes and the redaction rule) with a
/// construction nobody has analysed: an XOR keystream of SHA-256 blocks over the key, the nonce and a counter, and a
/// truncated SHA-256 of everything as the tag. Its nonces count up from 1, so its output is deterministic.
/// `algorithm()` says what it is: `"TEST-ONLY-XOR-SHA256 (not encryption)"`. Real encryptors are crypto's
/// (ChaCha20-Poly1305, AES-256-GCM over vetted libraries).
///
/// \snippet interfaces/encryptor_reference_demo.cpp reference
///
/// **Relationships.** Models `encryptor` (`concepts/encryptor.hpp`); passes `interfaces/encryptor_conformance.hpp`
/// directly and through `interfaces/encryptor.hpp`. `hash/sha2.hpp`, `hash/secure.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One.
///
/// **Guarantees.** Not thread-safe. Keys are 32 bytes, wiped when the encryptor is destroyed; errors are literals,
/// with no provider code. Envelope: header (`sealed_header`), a 12-byte nonce, the ciphertext, a 16-byte tag.
///
/// **Errors.** As the contract; `add_key` refuses a key id that is not one, or material that is not 32 bytes
/// (`invalid_argument`).
///
/// **Performance.** Not benchmarked: a test double.
///
/// **Tests and demos.** `tests/interfaces/encryptor_reference_test.cpp`, `demos/interfaces/encryptor_reference_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<functional>`, `<map>`, `<span>`, `<string>`,
/// `<string_view>`, `<vector>`, `interfaces/encryptor.hpp`, `hash/secure.hpp`, `hash/sha2.hpp`.
///
/// **Known issues.** Insecure by design: predictable nonces, an unvetted keystream and a tag open to length extension.
///
/// **Status.** New in 0.27.0. A test double, permanently.

#ifndef snicholls_core_interfaces_encryptor_reference_hpp
#define snicholls_core_interfaces_encryptor_reference_hpp

#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/interfaces/encryptor.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief **Not encryption.** A test double of `encryptor`: XOR with a SHA-256 keystream and a SHA-256 tag.
    /// \ingroup core
    class insecure_test_encryptor {
    public:
        /// \brief The key size.
        static constexpr std::size_t key_size = 32;
        /// \brief The nonce size.
        static constexpr std::size_t nonce_size = 12;
        /// \brief The tag size.
        static constexpr std::size_t tag_size = 16;

        /// \brief An encryptor with no keys.
        insecure_test_encryptor() = default;
        /// \brief Copies the keys.
        insecure_test_encryptor(const insecure_test_encryptor&) = default;
        /// \brief Moves the keys.
        insecure_test_encryptor(insecure_test_encryptor&&) = default;
        /// \brief Copies the keys.
        /// \return `*this`.
        insecure_test_encryptor& operator=(const insecure_test_encryptor&) = default;
        /// \brief Moves the keys.
        /// \return `*this`.
        insecure_test_encryptor& operator=(insecure_test_encryptor&&) = default;
        /// \brief Wipes the keys.
        ~insecure_test_encryptor()
        {
            for (auto& [id, k] : keys_)
                secure_wipe(std::span<std::byte>(k));
        }

        /// \brief Holds `material` under `key_id` (replacing a key of that id).
        /// \param key_id   The key id.
        /// \param material 32 bytes.
        /// \return Success, or `invalid_argument`.
        capability_result<void> add_key(std::string_view key_id, std::span<const std::byte> material)
        {
            if (!is_key_id(key_id))
                return fail(capability_errc::invalid_argument, {}, "not a key id");
            if (material.size() != key_size)
                return fail(capability_errc::invalid_argument, {}, "a key is 32 bytes");
            std::array<std::byte, key_size> k{};
            for (std::size_t i = 0; i < key_size; ++i)
                k[i] = material[i];
            keys_[std::string(key_id)] = k;
            return ok();
        }

        /// \brief Drops the key `key_id`: what it sealed no longer opens.
        /// \param key_id The key id.
        /// \return Whether it was held.
        bool remove_key(std::string_view key_id)
        {
            const auto it = keys_.find(key_id);
            if (it == keys_.end())
                return false;
            secure_wipe(std::span<std::byte>(it->second));
            keys_.erase(it);
            return true;
        }

        /// \brief Seals `plaintext` under `key_id`, authenticating `aad`.
        /// \param key_id    The key.
        /// \param plaintext The bytes.
        /// \param aad       The associated data.
        /// \return The envelope, `invalid_argument` or `not_found`.
        capability_result<std::vector<std::byte>> encrypt(std::string_view key_id, std::span<const std::byte> plaintext,
                                                          std::span<const std::byte> aad)
        {
            if (!is_key_id(key_id))
                return fail(capability_errc::invalid_argument, {}, "not a key id");
            const auto it = keys_.find(key_id);
            if (it == keys_.end())
                return fail(capability_errc::not_found, {}, "no such key");
            std::vector<std::byte> out = sealed_header(key_id);
            const std::size_t header = out.size();
            ++counter_;
            for (std::size_t i = 0; i < nonce_size; ++i)
                out.push_back(i < 8 ? static_cast<std::byte>(counter_ >> (56 - 8 * i)) : std::byte{0});
            const std::span<const std::byte> nonce(out.data() + header, nonce_size);
            const std::array<std::byte, nonce_size> n = copy_nonce(nonce);
            out.resize(out.size() + plaintext.size());
            xor_stream(it->second, n, plaintext, std::span<std::byte>(out.data() + header + nonce_size, plaintext.size()));
            const auto t = tag(it->second, std::span<const std::byte>(out.data(), out.size()), aad);
            out.insert(out.end(), t.begin(), t.end());
            return ok(std::move(out));
        }

        /// \brief Opens an envelope, if it is authentic with `aad`.
        /// \param sealed The envelope.
        /// \param aad    The associated data.
        /// \return The plaintext, `invalid_argument`, `not_found` or `signature_invalid`.
        [[nodiscard]] capability_result<std::vector<std::byte>> decrypt(std::span<const std::byte> sealed,
                                                                        std::span<const std::byte> aad) const
        {
            const auto id = sealed_key_id(sealed);
            if (!id)
                return err(id.unwrap_err());
            const std::size_t header = 2 + id->size();
            if (sealed.size() < header + nonce_size + tag_size)
                return fail(capability_errc::invalid_argument, {}, "the envelope is too short");
            const auto it = keys_.find(*id);
            if (it == keys_.end())
                return fail(capability_errc::not_found, {}, "no such key");
            const auto body = sealed.first(sealed.size() - tag_size);
            const auto expected = tag(it->second, body, aad);
            if (!constant_time_equal(std::span<const std::byte>(expected), sealed.last(tag_size)))
                return fail(capability_errc::signature_invalid, {}, "the envelope is not authentic");
            const auto n = copy_nonce(sealed.subspan(header, nonce_size));
            const auto ciphertext = body.subspan(header + nonce_size);
            std::vector<std::byte> plain(ciphertext.size());
            xor_stream(it->second, n, ciphertext, plain);
            return ok(std::move(plain));
        }

        /// \brief What it is.
        /// \return `"TEST-ONLY-XOR-SHA256 (not encryption)"`.
        [[nodiscard]] std::string_view algorithm() const noexcept { return "TEST-ONLY-XOR-SHA256 (not encryption)"; }

    private:
        using key_type = std::array<std::byte, key_size>;

        static std::array<std::byte, nonce_size> copy_nonce(std::span<const std::byte> s) noexcept
        {
            std::array<std::byte, nonce_size> n{};
            for (std::size_t i = 0; i < nonce_size; ++i)
                n[i] = s[i];
            return n;
        }

        static void xor_stream(const key_type& key, const std::array<std::byte, nonce_size>& nonce,
                               std::span<const std::byte> in, std::span<std::byte> out) noexcept
        {
            for (std::size_t block = 0; block * 32 < in.size(); ++block) {
                std::array<std::byte, 8> counter{};
                for (std::size_t i = 0; i < 8; ++i)
                    counter[i] = static_cast<std::byte>(static_cast<std::uint64_t>(block) >> (56 - 8 * i));
                sha256 h;
                h.update(std::span<const std::byte>(key)).update(std::span<const std::byte>(nonce)).update(std::span<const std::byte>(counter));
                auto ks = h.finish();
                for (std::size_t i = 0; i < 32 && block * 32 + i < in.size(); ++i)
                    out[block * 32 + i] = in[block * 32 + i] ^ ks.bytes[i];
                secure_wipe(ks);
            }
        }

        static std::array<std::byte, tag_size> tag(const key_type& key, std::span<const std::byte> body, std::span<const std::byte> aad) noexcept
        {
            std::array<std::byte, 8> aad_length{};
            for (std::size_t i = 0; i < 8; ++i)
                aad_length[i] = static_cast<std::byte>(static_cast<std::uint64_t>(aad.size()) >> (56 - 8 * i));
            sha256 h;
            h.update(std::string_view{"tag"}).update(std::span<const std::byte>(key)).update(std::span<const std::byte>(aad_length));
            h.update(aad).update(body);
            const auto d = h.finish();
            std::array<std::byte, tag_size> t{};
            for (std::size_t i = 0; i < tag_size; ++i)
                t[i] = d.bytes[i];
            return t;
        }

        std::map<std::string, key_type, std::less<>> keys_;
        std::uint64_t counter_ = 0;
    };

    static_assert(encryptor<insecure_test_encryptor>, "snicholls::insecure_test_encryptor must model snicholls::encryptor");

} // namespace snicholls

#endif // snicholls_core_interfaces_encryptor_reference_hpp
