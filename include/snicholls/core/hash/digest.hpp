//
//  digest.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `digest<N>`, the value every hash in core returns, and the concepts the hashes share.
/// \ingroup core
///
/// **Overview.** A digest is `N` bytes and nothing else: `digest<32>` is what `sha256::hash`, `keccak256::hash`
/// and `blake3::hash` return. It is an aggregate over `std::array<std::byte, N>`, trivially copyable and structural
/// (so a digest can be a template argument), compared with `==` and `<=>`, written as lowercase hexadecimal by
/// `hex()` without allocating, and read back by `from_hex`. Every operation is `constexpr`, so a digest computed by
/// the compiler can be checked in a `static_assert` against its published hexadecimal:
///
/// \snippet hash/digest_demo.cpp digest
///
/// The concepts are the hashes' vocabulary. `hash_input<S>` is any sequence of one-byte code units under core's
/// string rules (`reflect/string_traits.hpp`): `std::string`, `std::string_view`, a string literal (its length is
/// N - 1, never `strlen`), a `const char*` (a null pointer is empty), `std::span<const std::byte>`,
/// `std::vector<std::uint8_t>`, `std::array<std::byte, N>`, `char8_t` text. Wider text (`char16_t`, `wchar_t`) is
/// not an input: its bytes depend on the host's byte order. A `digest<N>` is accepted too, so a digest can be
/// hashed or used as a key directly. `hash_function<H>` is the interface every fixed-length hash meets
/// (`update`, `finish`, `hash`, `block_size`, `digest_size`), which `hmac<H>`, `hkdf_*` and `pbkdf2_hmac` are
/// written against. `md_hash<Policy>` is the class behind SHA-1, every SHA-2 and RIPEMD-160; use it through those
/// names.
///
/// **Relationships.** Returned by `hash/sha1.hpp`, `hash/sha2.hpp`, `hash/sha3.hpp`, `hash/ripemd160.hpp`,
/// `hash/blake2.hpp` and `hash/blake3.hpp`; consumed by `hash/hmac.hpp` and `hash/kdf.hpp`. Compare digests that
/// authenticate (a MAC, a password hash) with `constant_time_equal` (`hash/secure.hpp`), not `==`, which stops at
/// the first difference. The CRCs (`hash/crc.hpp`) return integers, as zlib does. Group: \ref core.
///
/// **Guarantees.** `digest<N>` is trivially copyable, standard layout, exactly `N` bytes with alignment 1 (static
/// assertions in the tests). Every member is `constexpr` and `noexcept`. `hex()` writes `2N` lowercase digits and
/// a terminating `'\0'` into a value of its own; nothing allocates.
///
/// **Errors.** `from_hex` returns `std::nullopt` for text that is not exactly `2N` hexadecimal digits (after an
/// optional `0x` / `0X`); nothing throws.
///
/// **Performance.** Not benchmarked: a digest is an array of bytes, and `hex()` is a table lookup per nibble.
///
/// **Tests and demos.** `tests/hash/digest_test.cpp`, `demos/hash/digest_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<compare>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<ostream>`,
/// `<span>`, `<string_view>`, `<type_traits>`, and `reflect/string_access.hpp`, `reflect/string_traits.hpp`,
/// `detail/hash_engine.hpp` (the Merkle-Damgard engine behind `md_hash`).
///
/// **Known issues.** `==` on digests is not constant-time; see Relationships.
///
/// **Status.** New in 0.17.0. Experimental.

#ifndef snicholls_core_hash_digest_hpp
#define snicholls_core_hash_digest_hpp

#include <snicholls/core/detail/hash_engine.hpp>
#include <snicholls/core/reflect/string_access.hpp>
#include <snicholls/core/reflect/string_traits.hpp>

#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <string_view>
#include <type_traits>

namespace snicholls {

/// \brief A digest's hexadecimal text: `2 * Bytes` lowercase digits and a `'\0'`, held by value.
/// \ingroup core
///
/// There is no implicit conversion to `std::string_view`: `std::string_view v = d.hex();` would view a temporary
/// that is gone at the semicolon. Keep the text (`auto text = d.hex();`) and take `text.view()`.
/// \tparam Bytes The digest's length in bytes.
template <std::size_t Bytes>
struct digest_hex
{
    /// \brief The digits and the terminator. Public for aggregate use; read it through `view()`.
    char chars[2 * Bytes + 1]{};

    /// \brief The digits.
    /// \return A view of the `2 * Bytes` digits.
    [[nodiscard]] constexpr std::string_view view() const noexcept { return {chars, 2 * Bytes}; }
    /// \brief The digits as a C string.
    /// \return A pointer to the null-terminated digits.
    [[nodiscard]] constexpr const char* c_str() const noexcept { return chars; }
    /// \brief The number of digits.
    /// \return `2 * Bytes`.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return 2 * Bytes; }
};

/// \brief `N` bytes produced by a hash: compared, ordered, printed and parsed as hexadecimal, all `constexpr`.
/// \ingroup core
///
/// An aggregate: `digest<4>{}` is four zero bytes, `digest<4>{{std::byte{1}, ...}}` names them. The bytes are in
/// the order the algorithm's specification writes them, which is the order of the hexadecimal.
/// \tparam N The length in bytes.
template <std::size_t N>
struct digest
{
    /// \brief The bytes. Public, so a digest is structural and an aggregate.
    std::array<std::byte, N> bytes{};

    /// \brief The length in bytes.
    /// \return `N`.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
    /// \brief The first byte.
    /// \return A pointer to the bytes.
    [[nodiscard]] constexpr const std::byte* data() const noexcept { return bytes.data(); }
    /// \brief The first byte, writable.
    /// \return A pointer to the bytes.
    [[nodiscard]] constexpr std::byte* data() noexcept { return bytes.data(); }
    /// \brief Iteration over the bytes.
    /// \return An iterator to the first byte.
    [[nodiscard]] constexpr auto begin() const noexcept { return bytes.begin(); }
    /// \brief Iteration over the bytes.
    /// \return An iterator past the last byte.
    [[nodiscard]] constexpr auto end() const noexcept { return bytes.end(); }
    /// \brief One byte.
    /// \param i The index, `< N`.
    /// \return The byte.
    [[nodiscard]] constexpr std::byte operator[](std::size_t i) const noexcept { return bytes[i]; }
    /// \brief One byte, writable.
    /// \param i The index, `< N`.
    /// \return The byte.
    [[nodiscard]] constexpr std::byte& operator[](std::size_t i) noexcept { return bytes[i]; }
    /// \brief The bytes as a fixed-extent span.
    /// \return A span of the `N` bytes.
    [[nodiscard]] constexpr std::span<const std::byte, N> span() const noexcept { return std::span<const std::byte, N>(bytes); }

    /// \brief The lowercase hexadecimal of the bytes, `2N` digits, without allocating.
    /// \return The text, by value.
    [[nodiscard]] constexpr digest_hex<N> hex() const noexcept
    {
        constexpr char digits[] = "0123456789abcdef";
        digest_hex<N> out{};
        for (std::size_t i = 0; i < N; ++i) {
            const auto b = static_cast<unsigned>(bytes[i]);
            out.chars[2 * i] = digits[b >> 4];
            out.chars[2 * i + 1] = digits[b & 0xFu];
        }
        return out;
    }

    /// \brief The first `M` bytes, as a digest of their own (a truncated hash, an Ethereum address from Keccak-256).
    /// \tparam M The length kept, `<= N`.
    /// \return The prefix.
    template <std::size_t M>
        requires(M <= N)
    [[nodiscard]] constexpr digest<M> first() const noexcept
    {
        digest<M> out{};
        for (std::size_t i = 0; i < M; ++i)
            out.bytes[i] = bytes[i];
        return out;
    }

    /// \brief The digest written as hexadecimal: exactly `2N` digits of either case, after an optional `0x` or `0X`.
    /// \param text The text.
    /// \return The digest, or `std::nullopt` for any other text.
    [[nodiscard]] static constexpr std::optional<digest> from_hex(std::string_view text) noexcept
    {
        if (text.size() >= 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X'))
            text.remove_prefix(2);
        if (text.size() != 2 * N)
            return std::nullopt;
        digest out{};
        for (std::size_t i = 0; i < N; ++i) {
            const int hi = nibble(text[2 * i]), lo = nibble(text[2 * i + 1]);
            if (hi < 0 || lo < 0)
                return std::nullopt;
            out.bytes[i] = static_cast<std::byte>((hi << 4) | lo);
        }
        return out;
    }

    /// \brief Equality of every byte. Not constant-time: use `constant_time_equal` for a secret.
    /// \return Whether the bytes are equal.
    friend constexpr bool operator==(const digest&, const digest&) noexcept = default;
    /// \brief Lexicographic order of the bytes, for sorted containers.
    /// \return The ordering.
    friend constexpr std::strong_ordering operator<=>(const digest& a, const digest& b) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
            if (a.bytes[i] != b.bytes[i])
                return static_cast<unsigned>(a.bytes[i]) <=> static_cast<unsigned>(b.bytes[i]);
        return std::strong_ordering::equal;
    }

    /// \brief Writes the lowercase hexadecimal: how `acout`, `print` and `print_to` show a digest.
    /// \param os The stream.
    void print(std::ostream& os) const { os << hex().view(); }

private:
    static constexpr int nibble(char c) noexcept
    {
        return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
    }
};

/// \brief Whether `T` is a `digest<N>`.
/// \ingroup core
template <typename T>
inline constexpr bool is_digest_v = false;

/// \brief A `digest<N>` is a digest.
/// \ingroup core
template <std::size_t N>
inline constexpr bool is_digest_v<digest<N>> = true;

/// \brief What a hash accepts: a sequence of one-byte code units under core's string rules, or a `digest<N>`.
/// \ingroup core
///
/// `std::string`, `std::string_view`, a string literal (length N - 1), `const char*`, `char8_t` text,
/// `std::span<const std::byte>`, `std::vector<std::uint8_t>`, `std::array<std::byte, N>`, a `digest<N>`.
/// Not wider text, whose bytes depend on the host's byte order.
template <typename S>
concept hash_input = (code_unit_sequence<S> && sizeof(code_unit_t<S>) == 1) || is_digest_v<std::remove_cvref_t<S>>;

namespace detail {

    // The code units and length of a hash_input, without copying: the string_access rules for sequences, the
    // bytes for a digest. Overloads, not a chain of if constexpr.
    template <typename S>
        requires(code_unit_sequence<S> && sizeof(code_unit_t<S>) == 1)
    constexpr const code_unit_t<S>* input_data(const S& s) noexcept
    {
        return data_of(s);
    }
    template <typename S>
        requires(code_unit_sequence<S> && sizeof(code_unit_t<S>) == 1)
    constexpr std::size_t input_size(const S& s) noexcept
    {
        return length_of(s);
    }
    template <std::size_t N>
    constexpr const std::byte* input_data(const digest<N>& d) noexcept
    {
        return d.bytes.data();
    }
    template <std::size_t N>
    constexpr std::size_t input_size(const digest<N>&) noexcept
    {
        return N;
    }

    // update(input) for every hasher, forwarding the units to Derived::absorb(const U*, std::size_t).
    template <typename Derived>
    class hash_updates
    {
    public:
        template <hash_input S>
        constexpr Derived& update(const S& input) noexcept
        {
            Derived& self = static_cast<Derived&>(*this);
            self.absorb(input_data(input), input_size(input));
            return self;
        }
    };

    // hash(input): one-shot, for a hasher that is default-constructible.
    template <typename Derived>
    class one_shot_hash
    {
    public:
        template <hash_input S>
        [[nodiscard]] static constexpr auto hash(const S& input) noexcept
        {
            Derived h;
            h.update(input);
            return h.finish();
        }
    };

} // namespace detail

/// \brief A fixed-length hash: incremental `update` / `finish`, a one-shot `hash`, `block_size` and `digest_size`.
/// \ingroup core
///
/// SHA-1, every SHA-2 and SHA-3, Keccak-256, RIPEMD-160, BLAKE2b, BLAKE2s and BLAKE3 meet it; `hmac<H>`,
/// `hkdf_extract<H>`, `hkdf_expand<H>` and `pbkdf2_hmac<H>` take any `H` that does.
template <typename H>
concept hash_function = std::default_initializable<H> && std::copyable<H> && requires(H h, std::string_view s) {
    { H::block_size } -> std::convertible_to<std::size_t>;
    { H::digest_size } -> std::convertible_to<std::size_t>;
    h.update(s);
    { h.finish() } -> std::same_as<digest<H::digest_size>>;
};

/// \brief The digest a hash function returns: `digest<H::digest_size>`.
/// \ingroup core
template <hash_function H>
using digest_of_t = digest<H::digest_size>;

/// \brief A Merkle-Damgard hash over a policy: the class behind `sha1`, every SHA-2 and `ripemd160`.
/// \ingroup core
///
/// Use it through its aliases. `update(input)` absorbs any `hash_input` and returns the hasher, so calls chain;
/// `finish()` pads, compresses the last block and returns the digest; `hash(input)` is both at once.
/// \tparam Policy The algorithm (internal).
template <typename Policy>
class md_hash : public detail::hash_updates<md_hash<Policy>>, public detail::one_shot_hash<md_hash<Policy>>
{
public:
    /// \brief The block size in bytes (64 or 128): what `hmac` pads the key to.
    static constexpr std::size_t block_size = Policy::block_size;
    /// \brief The digest size in bytes.
    static constexpr std::size_t digest_size = Policy::digest_size;
    /// \brief The digest type.
    using digest_type = digest<digest_size>;

    /// \brief A hasher with nothing absorbed.
    constexpr md_hash() noexcept = default;

    /// \brief Pads, compresses the last block and returns the digest. The hasher is not used afterwards.
    /// \return The digest.
    [[nodiscard]] constexpr digest_type finish() noexcept
    {
        digest_type out{};
        engine_.finish(out.bytes.data());
        return out;
    }

private:
    friend detail::hash_updates<md_hash>;
    template <typename U>
    constexpr void absorb(const U* p, std::size_t n) noexcept
    {
        engine_.absorb(p, n);
    }

    detail::md_engine<Policy> engine_{};
};

} // namespace snicholls

#endif /* snicholls_core_hash_digest_hpp */
