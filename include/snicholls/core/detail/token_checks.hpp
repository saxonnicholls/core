//
//  token_checks.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Internals of the token capability: the canonical address forms (EIP-55 for EVM chains, base58check for TRON) and the
//  shape of a token's asset code. Shared by basic_token's compile-time checks and the conformance kit. Not public API:
//  address encoding, decoding and derivation belong to the crypto library; core only recognises the canonical text.
//

#ifndef snicholls_core_detail_token_checks_hpp
#define snicholls_core_detail_token_checks_hpp

#include <snicholls/core/concepts/chain.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/hash/sha3.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace snicholls::detail {

    constexpr int token_hex_value(char c) noexcept
    {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        return -1;
    }

    // "0x" and 40 hex digits, in EIP-55's mixed case: a letter is a capital exactly when the matching nibble of
    // keccak256(the lower-case hex digits) is 8 or more.
    constexpr std::optional<std::string_view> check_evm_address(std::string_view a) noexcept
    {
        if (a.size() != 42 || a[0] != '0' || a[1] != 'x')
            return "token: contract() is not 0x and 40 hex digits";
        std::array<char, 40> lower{};
        for (std::size_t i = 0; i < 40; ++i) {
            const char c = a[i + 2];
            if (token_hex_value(c) < 0)
                return "token: contract() is not 0x and 40 hex digits";
            lower[i] = (c >= 'A' && c <= 'F') ? static_cast<char>(c - 'A' + 'a') : c;
        }
        const auto h = keccak256::hash(std::string_view{lower.data(), lower.size()});
        for (std::size_t i = 0; i < 40; ++i) {
            const char c = a[i + 2];
            if (c >= '0' && c <= '9')
                continue;
            const auto byte = static_cast<unsigned>(h.bytes[i / 2]);
            const unsigned nibble = i % 2 == 0 ? byte >> 4 : byte & 0xfu;
            if ((c >= 'A' && c <= 'F') != (nibble >= 8))
                return "token: contract() fails its EIP-55 checksum";
        }
        return std::nullopt;
    }

    constexpr int token_base58_value(char c) noexcept
    {
        constexpr std::string_view alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        for (std::size_t i = 0; i < alphabet.size(); ++i)
            if (alphabet[i] == c)
                return static_cast<int>(i);
        return -1;
    }

    // TRON's address text: base58check of 25 bytes, 0x41 and 20 address bytes and the first four bytes of
    // sha256(sha256(those 21)); 34 characters beginning with 'T'.
    constexpr std::optional<std::string_view> check_tron_address(std::string_view a) noexcept
    {
        if (a.size() != 34 || a[0] != 'T')
            return "token: contract() is not a TRON address (T and 33 more base58 characters)";
        std::array<std::uint8_t, 25> bytes{};   // big-endian
        for (const char c : a) {
            const int v = token_base58_value(c);
            if (v < 0)
                return "token: contract() is not a TRON address (T and 33 more base58 characters)";
            unsigned carry = static_cast<unsigned>(v);
            for (std::size_t i = bytes.size(); i-- > 0;) {
                carry += 58u * bytes[i];
                bytes[i] = static_cast<std::uint8_t>(carry & 0xffu);
                carry >>= 8;
            }   // no overflow: 34 digits beginning 'T' (51) stay below 51 * 58^33 + 58^33 < 2^200
        }
        if (bytes[0] != 0x41)
            return "token: contract() is not a TRON address (T and 33 more base58 characters)";
        std::array<std::uint8_t, 21> payload{};
        for (std::size_t i = 0; i < payload.size(); ++i)
            payload[i] = bytes[i];
        const auto check = sha256::hash(sha256::hash(payload));
        for (std::size_t i = 0; i < 4; ++i)
            if (static_cast<std::uint8_t>(check.bytes[i]) != bytes[21 + i])
                return "token: contract() fails its base58check checksum";
        return std::nullopt;
    }

    // The contract address in the canonical form of a chain of this kind. Kinds with no rule here need a non-empty one.
    constexpr std::optional<std::string_view> check_token_address(chain_kind kind, std::string_view a) noexcept
    {
        if (kind == chain_kind::evm)
            return check_evm_address(a);
        if (kind == chain_kind::tron)
            return check_tron_address(a);
        if (a.empty())
            return "token: contract() is empty";
        return std::nullopt;
    }

    // The number of decimal digits of n.
    constexpr std::size_t token_digits(std::uint64_t n) noexcept
    {
        std::size_t d = 1;
        for (; n >= 10; n /= 10)
            ++d;
        return d;
    }

    // Whether code is symbol, or symbol ':' and the decimal chain id (when there is one), with no leading zeros.
    constexpr bool is_token_code(std::string_view code, std::string_view symbol, std::optional<std::uint64_t> id) noexcept
    {
        if (code == symbol)
            return true;
        if (!id || code.size() != symbol.size() + 1 + token_digits(*id) || code.substr(0, symbol.size()) != symbol
            || code[symbol.size()] != ':')
            return false;
        std::uint64_t n = *id;
        for (std::size_t i = code.size(); i-- > symbol.size() + 1; n /= 10)
            if (code[i] != static_cast<char>('0' + n % 10))
                return false;
        return true;
    }

} // namespace snicholls::detail

#endif // snicholls_core_detail_token_checks_hpp
