//
//  uuid.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::uuid`: an RFC 9562 UUID value.
/// \ingroup core
///
/// snicholls::uuid - an RFC 9562 UUID: exactly 16 bytes in network (big-endian) order, nothing else.
///
/// \code
///   snicholls::uuid id = make_uuid();                         // v7: a generator needs OS randomness and lives in the consumer
///   snicholls::to_text(id).view();                             // "0190a5c4-7e1b-7c3a-9f12-3b4c5d6e7f80", no allocation
///   snicholls::to_string(id);  std::cout << id;                // the same text, allocating / streaming
///   snicholls::parse_uuid("0190A5C4-7E1B-7C3A-9F12-3B4C5D6E7F80");      // std::optional<uuid> (hex is case-insensitive)
///   snicholls::parse_uuid("{0190a5c4-...}", snicholls::uuid_parse_options::lenient());   // braces, urn:uuid:, 32 hex digits
///   using namespace snicholls::literals;  constexpr auto ns = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"_uuid;
///   id.version();  id.variant();  id.timestamp();  id.is_nil();  snicholls::uuid::max();
///   snicholls::make_uuid_v5(snicholls::uuid_namespace::dns, "www.example.com");   // name-based, deterministic, constexpr
/// \endcode
///
/// Semantics:
/// - An aggregate holding std::array\<std::uint8_t, 16>: sizeof 16, alignof 1 (adds no padding to an object),
///   trivially copyable, standard layout, structural (usable as a template argument). Value-initialised = nil.
/// - Every 128-bit value is a valid snicholls::uuid (nil, max, any version or variant); generators and composers set
///   the version and variant bits, the value type never checks them.
/// - == and <=> are constexpr and compare the bytes in order (strong_ordering): the RFC 9562 sort order, so v7
///   UUIDs sort by creation time. std::hash\<snicholls::uuid> is constexpr and noexcept (not portable, like every
///   std::hash).
/// - Text is the canonical 8-4-4-4-12 form, lowercase on output. Input hex is case-insensitive (RFC 9562 §4);
///   braces, the urn:uuid: prefix and the 32-digit form are accepted only when asked for (uuid_parse_options).
///   Text I/O is to_chars / from_chars style: no allocation, no locale, no exceptions, constexpr.

#ifndef snicholls_core_uuid_hpp
#define snicholls_core_uuid_hpp

#include <array>
#include <charconv>
#include <chrono>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

#include <snicholls/core/detail/sha1.hpp>
#include <snicholls/core/reflect/string_access.hpp>
#include <snicholls/core/reflect/string_traits.hpp>

namespace snicholls {

/// \brief The variant field (the top bits of byte 8), RFC 9562 §4.1.
/// \ingroup core
enum class uuid_variant : std::uint8_t
{
    ncs,   ///< 0xxx: reserved, NCS backward compatibility (includes nil)
    rfc9562,   ///< 10xx: every UUID this library generates (also RFC 4122)
    microsoft,   ///< 110x: reserved, Microsoft backward compatibility
    future,   ///< 111x: reserved for future definition (includes max)
};

/// \brief An RFC 9562 UUID: exactly 16 bytes in network (big-endian) order, nothing else.
/// \ingroup core
///
/// \overview
/// See the file description above.
///
/// \relations
/// - Composers here: `snicholls::make_uuid_v4`, `snicholls::make_uuid_v7` (from given bits), `snicholls::make_uuid_v5`; text `snicholls::to_chars`, `snicholls::parse_uuid`.
/// - Generators (OS randomness), JSON and SQL mappings belong in the consuming library.
/// - Group: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - Trivially copyable, 16 bytes, ordered bytewise (v7 sorts by time).
///
/// \errors
/// None.
///
/// \examples
/// \snippet uuid_demo.cpp demo
///
/// \perf
/// Release, 5 repetitions, median, 2026-10-05 (macOS / Ubuntu): `make_uuid_v5` 343 / 270 ns; `snicholls::to_chars` 9.3 / 9.4 ns (`snprintf` 586 / 496). Not benchmarked in core.
///
/// \testing
/// - Tests: tests/uuid_test.cpp.
/// - Demo: demos/uuid_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only .
///
/// \gotchas
/// None known.
///
/// \status
/// Stable, pre-1.0.
struct uuid
{
    /// \brief The bytes.
    std::array<std::uint8_t, 16> bytes{};

    /// \brief 00000000-0000-0000-0000-000000000000 and ffffffff-ffff-ffff-ffff-ffffffffffff.
    ///
    /// (RFC 9562 §5.9, §5.10).
    /// \return the result.
    [[nodiscard]] static constexpr uuid nil() noexcept { return {}; }
    /// \brief 00000000-0000-0000-0000-000000000000 and ffffffff-ffff-ffff-ffff-ffffffffffff.
    ///
    /// (RFC 9562 §5.9, §5.10).
    /// \return the result.
    [[nodiscard]] static constexpr uuid max() noexcept
    {
        uuid u;
        u.bytes.fill(0xFF);
        return u;
    }

    /// \brief From / to two big-endian 64-bit halves: bytes 0-7 and 8-15.
    /// \param high the high.
    /// \param low the low.
    /// \return the result.
    [[nodiscard]] static constexpr uuid from_u64(std::uint64_t high, std::uint64_t low) noexcept
    {
        uuid u;
        for (std::size_t i = 0; i < 8; ++i) {
            u.bytes[i] = static_cast<std::uint8_t>(high >> (56 - 8 * i));
            u.bytes[8 + i] = static_cast<std::uint8_t>(low >> (56 - 8 * i));
        }
        return u;
    }

    /// \brief The high.
    /// \return the high.
    [[nodiscard]] constexpr std::uint64_t high() const noexcept { return load(0); }
    /// \brief The low.
    /// \return the low.
    [[nodiscard]] constexpr std::uint64_t low() const noexcept { return load(8); }

    /// \brief Whether the value is nil.
    /// \return `true` if the value is nil.
    [[nodiscard]] constexpr bool is_nil() const noexcept { return (high() | low()) == 0; }
    /// \brief Whether the value is max.
    /// \return `true` if the value is max.
    [[nodiscard]] constexpr bool is_max() const noexcept { return (high() & low()) == ~std::uint64_t{0}; }

    /// \brief The version field (the top 4 bits of byte 6): 1..8 for RFC 9562 UUIDs.
    ///
    /// Meaningful only when variant() == uuid_variant::rfc9562 (nil reports 0, max 15).
    /// \return the version field (the top 4 bits of byte 6): 1..8 for RFC 9562 UUIDs.
    [[nodiscard]] constexpr unsigned version() const noexcept { return bytes[6] >> 4U; }

    /// \brief The variant.
    /// \return the variant.
    [[nodiscard]] constexpr uuid_variant variant() const noexcept
    {
        const unsigned b = bytes[8];
        if ((b & 0x80U) == 0) return uuid_variant::ncs;
        if ((b & 0x40U) == 0) return uuid_variant::rfc9562;
        if ((b & 0x20U) == 0) return uuid_variant::microsoft;
        return uuid_variant::future;
    }

    /// \brief The creation time of a version 7 UUID (its 48-bit Unix millisecond field).
    ///
    /// Std::nullopt for any other version or variant.
    ///
    /// (Versions 1 and 6 also carry a time; it is not decoded here.)
    /// \return the creation time of a version 7 UUID (its 48-bit Unix millisecond field).
    [[nodiscard]] constexpr std::optional<std::chrono::sys_time<std::chrono::milliseconds>> timestamp() const noexcept
    {
        if (version() != 7 || variant() != uuid_variant::rfc9562) return std::nullopt;
        const auto ms = static_cast<std::chrono::milliseconds::rep>(high() >> 16);
        return std::chrono::sys_time<std::chrono::milliseconds>(std::chrono::milliseconds(ms));
    }

    /// \brief The 16 bytes as std::byte (not constexpr).
    /// \return the 16 bytes as std::byte (not constexpr).
    [[nodiscard]] std::span<const std::byte, 16> as_bytes() const noexcept
    {
        return std::span<const std::byte, 16>(reinterpret_cast<const std::byte*>(bytes.data()), 16);
    }

    /// \brief Equality.
    /// \param a the left operand.
    /// \param b the right operand.
    /// \return the result.
    friend constexpr bool operator==(const uuid& a, const uuid& b) noexcept
    {
        return a.high() == b.high() && a.low() == b.low();
    }

    /// \brief Three-way comparison.
    /// \param a the left operand.
    /// \param b the right operand.
    /// \return the result.
    friend constexpr std::strong_ordering operator<=>(const uuid& a, const uuid& b) noexcept
    {
        if (const auto c = a.high() <=> b.high(); c != 0) return c;
        return a.low() <=> b.low();
    }

private:
    // Big-endian load; compilers turn this into one load + byte swap.
    constexpr std::uint64_t load(std::size_t at) const noexcept
    {
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < 8; ++i)
            v = (v << 8) | bytes[at + i];
        return v;
    }
};

namespace detail {

// The 64-bit xmx finaliser snicholls::hash_combine uses (a bijection: full avalanche, no collisions of its own).
constexpr std::uint64_t uuid_mix(std::uint64_t x) noexcept
{
    constexpr std::uint64_t m = 0xe9846af9b1a615dULL;
    x ^= x >> 32;
    x *= m;
    x ^= x >> 32;
    x *= m;
    x ^= x >> 28;
    return x;
}

} // namespace detail

/// \brief A well-mixed std::size_t of all 128 bits (what std::hash\<snicholls::uuid> returns).
///
/// Constexpr. Not portable across builds or architectures (size_t), like every std::hash.
/// \ingroup core
/// \param u the u.
/// \return a well-mixed std::size_t of all 128 bits (what std::hash\<snicholls::uuid> returns).
[[nodiscard]] constexpr std::size_t hash_value(const uuid& u) noexcept
{
    return static_cast<std::size_t>(detail::uuid_mix(u.high() ^ detail::uuid_mix(u.low() + 0x9E3779B97F4A7C15ULL)));
}

// ---- composers: fixed bits in, an RFC 9562 UUID out (constexpr, no randomness) ---------------------------

/// \brief Version 4 from 16 random bytes.
///
/// Sets the version (4) and variant (10) bits, keeps the other 122 bits.
/// \ingroup core
/// \param random the random.
/// \return the result.
[[nodiscard]] constexpr uuid make_uuid_v4(const std::array<std::uint8_t, 16>& random) noexcept
{
    uuid u{random};
    u.bytes[6] = static_cast<std::uint8_t>((u.bytes[6] & 0x0FU) | 0x40U);
    u.bytes[8] = static_cast<std::uint8_t>((u.bytes[8] & 0x3FU) | 0x80U);
    return u;
}

/// \brief Version 7 from its fields (RFC 9562 §5.7).
///
/// Unix_ts_ms (48 bits), rand_a (12 bits), rand_b (62 bits).
///
/// Higher bits of each argument are ignored.
/// \ingroup core
/// \param unix_ts_ms the unix ts ms.
/// \param rand_a the rand a.
/// \param rand_b the rand b.
/// \return the result.
[[nodiscard]] constexpr uuid make_uuid_v7(std::uint64_t unix_ts_ms, std::uint16_t rand_a, std::uint64_t rand_b) noexcept
{
    const std::uint64_t high = ((unix_ts_ms & 0xFFFF'FFFF'FFFFULL) << 16) | 0x7000U | (rand_a & 0x0FFFU);
    const std::uint64_t low = 0x8000'0000'0000'0000ULL | (rand_b & 0x3FFF'FFFF'FFFF'FFFFULL);
    return uuid::from_u64(high, low);
}

namespace detail {

template <typename Name>
constexpr uuid make_uuid_v5_from(const uuid& name_space, Name name) noexcept
{
    sha1 h;
    h.update(std::span<const std::uint8_t>(name_space.bytes));
    h.update(name);
    const sha1::digest_type digest = h.finish();
    uuid u;
    for (std::size_t i = 0; i < 16; ++i)
        u.bytes[i] = digest[i];   // the first 128 of 160 bits
    u.bytes[6] = static_cast<std::uint8_t>((u.bytes[6] & 0x0FU) | 0x50U);
    u.bytes[8] = static_cast<std::uint8_t>((u.bytes[8] & 0x3FU) | 0x80U);
    return u;
}

} // namespace detail

/// \brief Version 5 (name-based, SHA-1, RFC 9562 §5.5).
///
/// The same namespace and name always give the same UUID, on every platform and at compile time.
///
/// The name is hashed exactly as given (UTF-8 text, or raw bytes).
/// \ingroup core
/// \param name_space the name space.
/// \param name the name.
/// \return the result.
[[nodiscard]] constexpr uuid make_uuid_v5(const uuid& name_space, std::string_view name) noexcept
{
    return detail::make_uuid_v5_from(name_space, name);
}

/// \brief The make uuid v5.
/// \param name_space the name space.
/// \param name the name.
/// \return the make uuid v5.
[[nodiscard]] constexpr uuid make_uuid_v5(const uuid& name_space, std::span<const std::byte> name) noexcept
{
    return detail::make_uuid_v5_from(name_space, name);
}

// ---- text ----------------------------------------------------------------------------------------------

/// \brief Characters in the canonical form "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
/// \ingroup core
inline constexpr std::size_t uuid_text_size = 36;

/// \brief Writes the canonical lowercase form (36 chars, no terminator) to [first, last).
///
/// errc::value_too_large (and ptr == last) if it does not fit.
/// \ingroup core
/// \param first the first.
/// \param last the last.
/// \param u the u.
/// \return the result.
constexpr std::to_chars_result to_chars(char* first, char* last, const uuid& u) noexcept
{
    if (last - first < static_cast<std::ptrdiff_t>(uuid_text_size)) return {last, std::errc::value_too_large};
    constexpr const char* digits = "0123456789abcdef";
    char* p = first;
    for (std::size_t i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) *p++ = '-';
        *p++ = digits[u.bytes[i] >> 4U];
        *p++ = digits[u.bytes[i] & 0x0FU];
    }
    return {p, std::errc{}};
}

/// \brief The canonical text in a small value (no allocation): view(), c_str(), implicit std::string_view.
/// \ingroup core
struct uuid_text
{
    /// \brief The chars.
    std::array<char, uuid_text_size + 1> chars{};

    /// \brief The view.
    /// \return the view.
    [[nodiscard]] constexpr std::string_view view() const noexcept { return {chars.data(), uuid_text_size}; }
    /// \brief The c str.
    /// \return the c str.
    [[nodiscard]] constexpr const char* c_str() const noexcept { return chars.data(); }
    /// \brief The number of elements.
    /// \return the number of elements.
    [[nodiscard]] static constexpr std::size_t size() noexcept { return uuid_text_size; }
    /// \brief The text as a view.
    /// \return the view.
    constexpr operator std::string_view() const noexcept { return view(); }
};

/// \brief Converts to text.
/// \param u the u.
/// \return the result.
[[nodiscard]] constexpr uuid_text to_text(const uuid& u) noexcept
{
    uuid_text t;
    (void)to_chars(t.chars.data(), t.chars.data() + uuid_text_size, u);
    return t;
}

/// \brief The text form.
/// \param u the u.
/// \return the text form.
[[nodiscard]] inline std::string to_string(const uuid& u) { return std::string(to_text(u).view()); }

/// \brief Writes the value to a stream.
/// \param os the output stream.
/// \param u the u.
/// \return the result.
inline std::ostream& operator<<(std::ostream& os, const uuid& u)
{
    const uuid_text t = to_text(u);
    return os.write(t.c_str(), static_cast<std::streamsize>(uuid_text_size));
}

/// \brief Which non-canonical spellings from_chars / parse_uuid accept.
///
/// The canonical 8-4-4-4-12 form (any hex case) is always accepted.
/// \ingroup core
struct uuid_parse_options
{
    bool braces = false;   ///< {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}  (Microsoft GUID registry form)
    bool urn = false;   ///< urn:uuid:xxxxxxxx-...  (RFC 9562 §4; the prefix is case-insensitive)
    bool compact = false;   ///< 32 hex digits, no hyphens

    /// \brief The lenient.
    /// \return the lenient.
    [[nodiscard]] static constexpr uuid_parse_options lenient() noexcept { return {true, true, true}; }
};

namespace detail {

// -1 for a non-hex character.
inline constexpr std::array<std::int8_t, 256> uuid_hex_values = [] {
    std::array<std::int8_t, 256> t{};
    for (auto& v : t)
        v = -1;
    for (int c = 0; c < 10; ++c)
        t[static_cast<std::size_t>('0' + c)] = static_cast<std::int8_t>(c);
    for (int c = 0; c < 6; ++c) {
        t[static_cast<std::size_t>('a' + c)] = static_cast<std::int8_t>(10 + c);
        t[static_cast<std::size_t>('A' + c)] = static_cast<std::int8_t>(10 + c);
    }
    return t;
}();

// Offsets of the 16 byte pairs in the canonical and compact forms.
inline constexpr std::array<std::uint8_t, 16> uuid_canonical_offsets{0,  2,  4,  6,  9,  11, 14, 16,
                                                                     19, 21, 24, 26, 28, 30, 32, 34};

// Decodes 16 hex pairs at p + offsets (canonical) or p + 2i (compact). false on any non-hex character.
template <bool Canonical>
constexpr bool decode_uuid_hex(const char* p, uuid& out) noexcept
{
    int bad = 0;
    uuid u;
    for (std::size_t i = 0; i < 16; ++i) {
        const std::size_t at = Canonical ? uuid_canonical_offsets[i] : 2 * i;
        const int hi = uuid_hex_values[static_cast<unsigned char>(p[at])];
        const int lo = uuid_hex_values[static_cast<unsigned char>(p[at + 1])];
        bad |= hi | lo;   // negative (sign bit set) if either is not hex
        u.bytes[i] = static_cast<std::uint8_t>(((hi & 0x0F) << 4) | (lo & 0x0F));
    }
    if (bad < 0) return false;
    out = u;
    return true;
}

constexpr const char* parse_canonical_uuid(const char* first, const char* last, uuid& out) noexcept
{
    if (last - first < static_cast<std::ptrdiff_t>(uuid_text_size)) return nullptr;
    if (first[8] != '-' || first[13] != '-' || first[18] != '-' || first[23] != '-') return nullptr;
    return decode_uuid_hex<true>(first, out) ? first + uuid_text_size : nullptr;
}

constexpr bool starts_with_urn_uuid(const char* first, const char* last) noexcept
{
    constexpr std::string_view prefix = "urn:uuid:";
    if (last - first < static_cast<std::ptrdiff_t>(prefix.size())) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        char c = first[i];
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if (c != prefix[i]) return false;
    }
    return true;
}

template <typename S>
concept uuid_char_string = string_like<S> && std::same_as<code_unit_t<S>, char>;

} // namespace detail

/// \brief Parses a UUID at the start of [first, last), like std::from_chars.
///
/// On success ptr is one past the UUID's last character (trailing text is left alone); on failure ec == errc::invalid_argument, ptr == first and out is unchanged.
///
/// Leading whitespace, signs and "0x" are never accepted.
/// \ingroup core
/// \param first the first.
/// \param last the last.
/// \param out receives the result.
/// \param options the options.
/// \return the result.
constexpr std::from_chars_result from_chars(const char* first, const char* last, uuid& out,
                                             uuid_parse_options options = {}) noexcept
{
    const std::from_chars_result fail{first, std::errc::invalid_argument};
    if (first == last) return fail;
    uuid u;
    if (*first == '{') {
        if (!options.braces) return fail;
        const char* end = detail::parse_canonical_uuid(first + 1, last, u);
        if (end == nullptr || end == last || *end != '}') return fail;
        out = u;
        return {end + 1, std::errc{}};
    }
    if (*first == 'u' || *first == 'U') {
        if (!options.urn || !detail::starts_with_urn_uuid(first, last)) return fail;
        const char* end = detail::parse_canonical_uuid(first + 9, last, u);
        if (end == nullptr) return fail;
        out = u;
        return {end, std::errc{}};
    }
    if (const char* end = detail::parse_canonical_uuid(first, last, u); end != nullptr) {
        out = u;
        return {end, std::errc{}};
    }
    if (options.compact && last - first >= 32 && detail::decode_uuid_hex<false>(first, u)) {
        out = u;
        return {first + 32, std::errc{}};
    }
    return fail;
}

/// \brief The whole string as a UUID (see from_chars); std::nullopt if it is not one or anything follows it.
///
/// Accepts any char string (std::string, std::string_view, const char*, literals, snicholls::fixed_string, ...).
/// \ingroup core
/// \param text the text.
/// \param options the options.
/// \return the whole string as a UUID (see from_chars); std::nullopt if it is not one or anything follows it.
template <detail::uuid_char_string S>
[[nodiscard]] constexpr std::optional<uuid> parse_uuid(const S& text, uuid_parse_options options = {}) noexcept
{
    const auto sv = to_string_view(text);
    const char* const last = sv.data() + sv.size();
    uuid u;
    const auto [end, ec] = from_chars(sv.data(), last, u, options);
    if (ec != std::errc{} || end != last) return std::nullopt;
    return u;
}

// The RFC 9562 §6.6 namespace IDs for version 5 (and 3) UUIDs.
namespace uuid_namespace {

/// \brief The dns.
/// \return the dns.
inline constexpr uuid dns = uuid::from_u64(0x6ba7b8109dad11d1ULL, 0x80b400c04fd430c8ULL);
/// \brief The url.
/// \return the url.
inline constexpr uuid url = uuid::from_u64(0x6ba7b8119dad11d1ULL, 0x80b400c04fd430c8ULL);
/// \brief The oid.
/// \return the oid.
inline constexpr uuid oid = uuid::from_u64(0x6ba7b8129dad11d1ULL, 0x80b400c04fd430c8ULL);
/// \brief The x500.
/// \return the x500.
inline constexpr uuid x500 = uuid::from_u64(0x6ba7b8149dad11d1ULL, 0x80b400c04fd430c8ULL);

} // namespace uuid_namespace

namespace literals {

/// \brief "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"_uuid.
///
/// A snicholls::uuid checked at compile time (canonical form, any case).
/// \ingroup core
/// \param text the text.
/// \param length the length.
/// \return the result.
consteval uuid operator""_uuid(const char* text, std::size_t length)
{
    const auto r = parse_uuid(std::string_view(text, length));
    if (!r) throw "snicholls::operator\"\"_uuid: not a canonical 8-4-4-4-12 UUID";
    return *r;
}

} // namespace literals

} // namespace snicholls

// The hash<.
/// \cond
template <>
struct std::hash<snicholls::uuid>
{
    /// \brief Invokes it.
    /// \param u the u.
    /// \return the operator.
    constexpr std::size_t operator()(const snicholls::uuid& u) const noexcept { return snicholls::hash_value(u); }
};
/// \endcond

#endif /* snicholls_core_uuid_hpp */
