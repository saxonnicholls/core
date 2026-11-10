//
//  uuid_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/uuid.hpp>.
//

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/uuid.hpp>

namespace {

using snicholls::uuid;
using snicholls::uuid_parse_options;
using snicholls::uuid_variant;
using namespace snicholls::literals;
using namespace std::string_view_literals;

// ---- the type ------------------------------------------------------------------------------------------

TEST(UuidType, IsSixteenTriviallyCopyableBytesWithNoAlignment)
{
    static_assert(sizeof(uuid) == 16);
    static_assert(alignof(uuid) == 1);   // never adds padding to an object
    static_assert(std::is_trivially_copyable_v<uuid>);
    static_assert(std::is_standard_layout_v<uuid>);
    static_assert(std::is_aggregate_v<uuid>);
    static_assert(std::is_nothrow_default_constructible_v<uuid>);
    SUCCEED();
}

template <uuid U>
constexpr unsigned version_of = U.version();

TEST(UuidType, IsStructuralSoItCanBeATemplateArgument)
{
    static_assert(version_of<"f81d4fae-7dec-11d0-a765-00a0c91e6bf6"_uuid> == 1);
    SUCCEED();
}

TEST(UuidType, ValueInitialisedIsNil)
{
    constexpr uuid u{};
    static_assert(u.is_nil());
    static_assert(u == uuid::nil());
    static_assert(!u.is_max());
    EXPECT_EQ(snicholls::to_text(u).view(), "00000000-0000-0000-0000-000000000000");
}

TEST(UuidType, NilAndMaxAreTheRfcSpecialValues)
{
    static_assert(uuid::nil().is_nil() && !uuid::nil().is_max());
    static_assert(uuid::max().is_max() && !uuid::max().is_nil());
    EXPECT_EQ(snicholls::to_string(uuid::max()), "ffffffff-ffff-ffff-ffff-ffffffffffff");
    EXPECT_EQ(uuid::nil().version(), 0U);
    EXPECT_EQ(uuid::max().version(), 15U);
    EXPECT_EQ(uuid::nil().variant(), uuid_variant::ncs);
    EXPECT_EQ(uuid::max().variant(), uuid_variant::future);
}

TEST(UuidType, HalvesAreBigEndian)
{
    constexpr uuid u = uuid::from_u64(0x0011223344556677ULL, 0x8899aabbccddeeffULL);
    static_assert(u.bytes[0] == 0x00 && u.bytes[7] == 0x77 && u.bytes[8] == 0x88 && u.bytes[15] == 0xff);
    static_assert(u.high() == 0x0011223344556677ULL && u.low() == 0x8899aabbccddeeffULL);
    EXPECT_EQ(snicholls::to_string(u), "00112233-4455-6677-8899-aabbccddeeff");
}

TEST(UuidType, AsBytesViewsTheSameMemory)
{
    const uuid u = "00112233-4455-6677-8899-aabbccddeeff"_uuid;
    const auto b = u.as_bytes();
    static_assert(decltype(b)::extent == 16);
    EXPECT_EQ(static_cast<const void*>(b.data()), static_cast<const void*>(u.bytes.data()));
    EXPECT_EQ(b[15], std::byte{0xff});
}

// ---- version and variant -------------------------------------------------------------------------------

TEST(UuidVersion, ReadsTheVersionNibble)
{
    static_assert("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"_uuid.version() == 1);
    static_assert("919108f7-52d1-4320-9bac-f847db4148a8"_uuid.version() == 4);
    static_assert("2ed6657d-e927-568b-95e1-2665a8aea6a2"_uuid.version() == 5);
    static_assert("1ec9414c-232a-6b00-b3c8-9f6bdeced846"_uuid.version() == 6);
    static_assert("017f22e2-79b0-7cc3-98c4-dc0c0c07398f"_uuid.version() == 7);
    static_assert("320c3d4d-cc00-875b-8ec9-32d5f69181c0"_uuid.version() == 8);
    SUCCEED();
}

TEST(UuidVariant, EveryVariantPattern)
{
    for (unsigned b = 0; b < 256; ++b) {
        uuid u;
        u.bytes[8] = static_cast<std::uint8_t>(b);
        const uuid_variant expected = b < 0x80   ? uuid_variant::ncs
                                      : b < 0xC0 ? uuid_variant::rfc9562
                                      : b < 0xE0 ? uuid_variant::microsoft
                                                 : uuid_variant::future;
        ASSERT_EQ(u.variant(), expected) << b;
    }
}

TEST(UuidTimestamp, VersionSevenCarriesItsMillisecond)
{
    using namespace std::chrono;
    // RFC 9562 A.6: 0x017F22E279B0 = Tuesday, February 22, 2022 2:22:22.00 PM GMT-05:00.
    constexpr uuid u = "017f22e2-79b0-7cc3-98c4-dc0c0c07398f"_uuid;
    static_assert(u.timestamp().has_value());
    EXPECT_EQ(u.timestamp()->time_since_epoch(), milliseconds(0x017F22E279B0));
    EXPECT_EQ(*u.timestamp(), sys_days{2022y / February / 22} + 19h + 22min + 22s);
}

TEST(UuidTimestamp, OtherVersionsAndVariantsHaveNone)
{
    static_assert(!"919108f7-52d1-4320-9bac-f847db4148a8"_uuid.timestamp());   // v4
    static_assert(!uuid::nil().timestamp());
    static_assert(!uuid::max().timestamp());
    static_assert(!"017f22e2-79b0-7cc3-c8c4-dc0c0c07398f"_uuid.timestamp());   // v7 nibble, Microsoft variant
    SUCCEED();
}

// ---- comparison and hashing ----------------------------------------------------------------------------

TEST(UuidCompare, BytewiseLexicographicStrongOrdering)
{
    static_assert(std::is_same_v<decltype(uuid{} <=> uuid{}), std::strong_ordering>);
    static_assert(noexcept(uuid{} <=> uuid{}) && noexcept(uuid{} == uuid{}));
    static_assert(uuid::nil() < uuid::max());
    static_assert(uuid::from_u64(1, 0) > uuid::from_u64(0, ~0ULL));   // high half dominates
    static_assert(uuid::from_u64(5, 1) < uuid::from_u64(5, 2));

    // Agrees with comparing the byte arrays, and with comparing the canonical text, on random values.
    std::mt19937_64 rng(9562);
    std::vector<uuid> v;
    for (int i = 0; i < 2000; ++i) {
        uuid u = uuid::from_u64(rng(), rng());
        u.bytes[i % 16] = static_cast<std::uint8_t>(i % 3);   // force shared prefixes
        v.push_back(u);
    }
    for (std::size_t i = 1; i < v.size(); ++i) {
        const uuid& a = v[i - 1];
        const uuid& b = v[i];
        ASSERT_EQ(a <=> b, a.bytes <=> b.bytes);
        ASSERT_EQ(a <=> b, snicholls::to_string(a) <=> snicholls::to_string(b));
        ASSERT_EQ(a == b, a.bytes == b.bytes);
    }
}

TEST(UuidHash, StdHashIsConstexprNoexceptAndSpreads)
{
    static_assert(std::hash<uuid>{}(uuid::nil()) == snicholls::hash_value(uuid::nil()));
    static_assert(noexcept(std::hash<uuid>{}(uuid{})));
    // Structured inputs (single bits, small counters in either half) must not collide.
    std::set<uuid> inputs;
    for (unsigned bit = 0; bit < 128; ++bit) {
        uuid u;
        u.bytes[bit / 8] = static_cast<std::uint8_t>(0x80U >> (bit % 8));
        inputs.insert(u);
    }
    for (std::uint64_t i = 0; i < 10000; ++i) {
        inputs.insert(uuid::from_u64(0, i));
        inputs.insert(uuid::from_u64(i, 0));
    }
    std::unordered_set<std::size_t> hashes;
    for (const uuid& u : inputs)
        hashes.insert(std::hash<uuid>{}(u));
    EXPECT_EQ(hashes.size(), inputs.size());
    const std::unordered_set<uuid> set(inputs.begin(), inputs.end());
    EXPECT_EQ(set.size(), inputs.size());
}

// ---- composers -----------------------------------------------------------------------------------------

TEST(UuidComposeV4, RfcAppendixA3)
{
    // RFC 9562 A.3: 919108f7-52d1-4320-9bac-f847db4148a8, built from its random bits with other ver/var bits.
    constexpr std::array<std::uint8_t, 16> random{0x91, 0x91, 0x08, 0xf7, 0x52, 0xd1, 0xF3, 0x20,
                                                  0x5b, 0xac, 0xf8, 0x47, 0xdb, 0x41, 0x48, 0xa8};
    constexpr uuid u = snicholls::make_uuid_v4(random);
    static_assert(u == "919108f7-52d1-4320-9bac-f847db4148a8"_uuid);
    static_assert(u.version() == 4 && u.variant() == uuid_variant::rfc9562);
    SUCCEED();
}

TEST(UuidComposeV4, KeepsExactly122Bits)
{
    std::array<std::uint8_t, 16> ones{};
    ones.fill(0xFF);
    const uuid a = snicholls::make_uuid_v4(ones);
    EXPECT_EQ(snicholls::to_string(a), "ffffffff-ffff-4fff-bfff-ffffffffffff");
    const uuid b = snicholls::make_uuid_v4(std::array<std::uint8_t, 16>{});
    EXPECT_EQ(snicholls::to_string(b), "00000000-0000-4000-8000-000000000000");
}

TEST(UuidComposeV7, RfcAppendixA6)
{
    constexpr uuid u = snicholls::make_uuid_v7(0x017F22E279B0ULL, 0xCC3, 0x18C4DC0C0C07398FULL);
    static_assert(u == "017F22E2-79B0-7CC3-98C4-DC0C0C07398F"_uuid);
    static_assert(u.version() == 7 && u.variant() == uuid_variant::rfc9562);
    SUCCEED();
}

TEST(UuidComposeV7, IgnoresBitsBeyondEachField)
{
    constexpr uuid u = snicholls::make_uuid_v7(~0ULL, 0xFFFF, ~0ULL);
    EXPECT_EQ(snicholls::to_string(u), "ffffffff-ffff-7fff-bfff-ffffffffffff");
}

TEST(UuidComposeV5, RfcAppendixA4AndIndependentReference)
{
    // RFC 9562 A.4.
    static_assert(snicholls::make_uuid_v5(snicholls::uuid_namespace::dns, "www.example.com") ==
                  "2ed6657d-e927-568b-95e1-2665a8aea6a2"_uuid);
    // Python 3 uuid.uuid5 (independent implementation) for the other namespaces, an empty and a UTF-8 name.
    EXPECT_EQ(snicholls::make_uuid_v5(snicholls::uuid_namespace::url, "https://example.com/a"),
              "6639460f-3425-5329-8097-a58f06127860"_uuid);
    EXPECT_EQ(snicholls::make_uuid_v5(snicholls::uuid_namespace::oid, "1.3.6.1"), "1447fa61-5277-5fef-a9b3-fbc6e44f4af3"_uuid);
    EXPECT_EQ(snicholls::make_uuid_v5(snicholls::uuid_namespace::x500, "cn=John"), "1713550e-4d56-5817-bce4-d5dac105f99d"_uuid);
    EXPECT_EQ(snicholls::make_uuid_v5(snicholls::uuid_namespace::dns, ""), "4ebd0208-8328-5d69-8c44-ec50939c0967"_uuid);
    EXPECT_EQ(snicholls::make_uuid_v5(snicholls::uuid_namespace::dns, "h\xc3\xa9llo"), "5ca283f8-ff14-58cb-ba86-bf52120a3414"_uuid);
}

TEST(UuidComposeV5, BytesNameEqualsTextName)
{
    const std::string_view text = "www.example.com";
    const auto bytes = std::as_bytes(std::span(text.data(), text.size()));
    EXPECT_EQ(snicholls::make_uuid_v5(snicholls::uuid_namespace::dns, bytes),
              snicholls::make_uuid_v5(snicholls::uuid_namespace::dns, text));
    static_assert(noexcept(snicholls::make_uuid_v5(uuid{}, "x"sv)));
}

TEST(UuidNamespaces, AreTheRfcValues)
{
    EXPECT_EQ(snicholls::to_string(snicholls::uuid_namespace::dns), "6ba7b810-9dad-11d1-80b4-00c04fd430c8");
    EXPECT_EQ(snicholls::to_string(snicholls::uuid_namespace::url), "6ba7b811-9dad-11d1-80b4-00c04fd430c8");
    EXPECT_EQ(snicholls::to_string(snicholls::uuid_namespace::oid), "6ba7b812-9dad-11d1-80b4-00c04fd430c8");
    EXPECT_EQ(snicholls::to_string(snicholls::uuid_namespace::x500), "6ba7b814-9dad-11d1-80b4-00c04fd430c8");
}

// ---- formatting ----------------------------------------------------------------------------------------

constexpr std::string_view sample = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6";

TEST(UuidToChars, WritesCanonicalLowercase)
{
    constexpr uuid u = "F81D4FAE-7DEC-11D0-A765-00A0C91E6BF6"_uuid;
    char buf[40] = {};
    const auto [end, ec] = snicholls::to_chars(buf, buf + sizeof buf, u);
    EXPECT_EQ(ec, std::errc{});
    EXPECT_EQ(end, buf + 36);
    EXPECT_EQ(std::string_view(buf, 36), sample);
    EXPECT_EQ(buf[36], '\0');   // nothing written past the text
}

TEST(UuidToChars, TooSmallBufferIsValueTooLarge)
{
    char buf[35];
    const auto [end, ec] = snicholls::to_chars(buf, buf + sizeof buf, uuid{});
    EXPECT_EQ(ec, std::errc::value_too_large);
    EXPECT_EQ(end, buf + sizeof buf);
    char exact[36];
    EXPECT_EQ(snicholls::to_chars(exact, exact + 36, uuid{}).ec, std::errc{});
}

TEST(UuidToChars, ConstexprAndNoexcept)
{
    constexpr auto t = snicholls::to_text("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"_uuid);
    static_assert(t.view() == sample);
    static_assert(t.c_str()[36] == '\0');
    static_assert(snicholls::uuid_text::size() == 36);
    static_assert(noexcept(snicholls::to_text(uuid{})));
    const std::string_view implicit = t;
    EXPECT_EQ(implicit, sample);
}

TEST(UuidToString, AllocatingAndStreamingAgree)
{
    const uuid u = "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"_uuid;
    EXPECT_EQ(snicholls::to_string(u), sample);
    std::ostringstream os;
    os << u << '|' << uuid::nil();
    EXPECT_EQ(os.str(), std::string(sample) + "|00000000-0000-0000-0000-000000000000");
}

// ---- parsing -------------------------------------------------------------------------------------------

TEST(UuidParse, CanonicalAnyCaseRoundTrips)
{
    static_assert(snicholls::parse_uuid(sample).has_value());
    EXPECT_EQ(snicholls::parse_uuid("F81D4FAE-7DEC-11D0-A765-00A0C91E6BF6"sv), snicholls::parse_uuid(sample));
    EXPECT_EQ(snicholls::parse_uuid("f81D4fAe-7dEc-11d0-A765-00a0C91e6Bf6"sv), snicholls::parse_uuid(sample));
    // Every byte value at every position survives format -> parse.
    for (unsigned v = 0; v < 256; ++v)
        for (std::size_t i = 0; i < 16; ++i) {
            uuid u = uuid::from_u64(0x0123456789abcdefULL, 0xfedcba9876543210ULL);
            u.bytes[i] = static_cast<std::uint8_t>(v);
            ASSERT_EQ(snicholls::parse_uuid(snicholls::to_text(u).view()), u);
        }
}

TEST(UuidParse, AcceptsEveryCharStringType)
{
    const std::string s(sample);
    const char* c = s.c_str();
    char arr[37];
    std::copy(s.begin(), s.end(), arr);
    arr[36] = '\0';
    const auto expected = snicholls::parse_uuid(sample);
    ASSERT_TRUE(expected);
    EXPECT_EQ(snicholls::parse_uuid(s), expected);
    EXPECT_EQ(snicholls::parse_uuid(c), expected);
    EXPECT_EQ(snicholls::parse_uuid(arr), expected);
    EXPECT_EQ(snicholls::parse_uuid("f81d4fae-7dec-11d0-a765-00a0c91e6bf6"), expected);
    EXPECT_EQ(snicholls::parse_uuid(static_cast<const char*>(nullptr)), std::nullopt);   // a null C string is empty
}

TEST(UuidParse, LenientFormsOnlyWhenEnabled)
{
    const auto expected = snicholls::parse_uuid(sample);
    const std::string braces = "{" + std::string(sample) + "}";
    const std::string urn = "urn:uuid:" + std::string(sample);
    const std::string urn_upper = "URN:UUID:" + std::string(sample);
    const std::string compact = "f81d4fae7dec11d0a76500a0c91e6bf6";

    for (const std::string& s : {braces, urn, urn_upper, compact})
        EXPECT_EQ(snicholls::parse_uuid(s), std::nullopt) << s;   // strict by default

    const auto lenient = uuid_parse_options::lenient();
    for (const std::string& s : {braces, urn, urn_upper, compact})
        EXPECT_EQ(snicholls::parse_uuid(s, lenient), expected) << s;

    EXPECT_EQ(snicholls::parse_uuid(braces, uuid_parse_options{.braces = true}), expected);
    EXPECT_EQ(snicholls::parse_uuid(urn, uuid_parse_options{.braces = true}), std::nullopt);
    EXPECT_EQ(snicholls::parse_uuid(urn, uuid_parse_options{.urn = true}), expected);
    EXPECT_EQ(snicholls::parse_uuid(compact, uuid_parse_options{.urn = true}), std::nullopt);
    EXPECT_EQ(snicholls::parse_uuid(compact, uuid_parse_options{.compact = true}), expected);
    EXPECT_EQ(snicholls::parse_uuid(sample, uuid_parse_options{}), expected);
}

// Every class of malformed input, rejected strictly and leniently.
TEST(UuidParse, RejectsEveryMalformedClass)
{
    const std::string good(sample);
    std::vector<std::string> bad = {
        "",                                                // empty
        "f81d4fae-7dec-11d0-a765-00a0c91e6bf",             // one short
        "f81d4fae-7dec-11d0-a765",                         // truncated
        "f81d4fae-7dec-11d0-a765-00a0c91e6bf6a",           // one long (trailing hex)
        "f81d4fae-7dec-11d0-a765-00a0c91e6bf6 ",           // trailing space
        " f81d4fae-7dec-11d0-a765-00a0c91e6bf6",           // leading space
        "\tf81d4fae-7dec-11d0-a765-00a0c91e6bf6",          // leading tab
        "+f81d4fae-7dec-11d0-a765-00a0c91e6bf6",           // sign
        "0xf81d4fae-7dec-11d0-a765-00a0c91e6bf6",          // hex prefix
        "f81d4fae7dec-11d0-a765-00a0c91e6bf6-",            // hyphen moved
        "f81d4fae-7dec11d0-a765-00a0c91e6bf6",             // hyphen missing
        "f81d4fa-e7dec-11d0-a765-00a0c91e6bf6",            // hyphen early
        "f81d4fae_7dec_11d0_a765_00a0c91e6bf6",            // wrong separator
        "f81d4fae-7dec-11d0-a765-00a0c91e6bf6}",           // stray closing brace
        "{f81d4fae-7dec-11d0-a765-00a0c91e6bf6",           // unbalanced brace
        "{f81d4fae-7dec-11d0-a765-00a0c91e6bf6)",          // wrong closing bracket
        "(f81d4fae-7dec-11d0-a765-00a0c91e6bf6)",          // parentheses
        "{{f81d4fae-7dec-11d0-a765-00a0c91e6bf6}}",        // double braces
        "{f81d4fae7dec11d0a76500a0c91e6bf6}",              // braces around the compact form
        "urn:uid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6",    // wrong URN namespace
        "urn:uuid:",                                       // URN with nothing
        "urn:uuid:f81d4fae7dec11d0a76500a0c91e6bf6",       // URN around the compact form
        "uuid:f81d4fae-7dec-11d0-a765-00a0c91e6bf6",       // partial prefix
        "f81d4fae7dec11d0a76500a0c91e6bf",                 // compact one short
        "f81d4fae7dec11d0a76500a0c91e6bf6a",               // compact one long
        "f81d4fae7dec11d0a76500a0c91e6bfg",                // compact non-hex
        "f81d4fae-7dec-11d0-a765-00a0c91e6bf6f81d4fae",    // concatenated
        "f81d4fae-7dec-11d0-a765-00a0c91e6bf\xc3\xa9",     // non-ASCII (UTF-8) byte
    };
    // A non-hex character at each of the 32 hex positions (several kinds of non-hex).
    constexpr std::array<char, 7> non_hex{'g', 'G', 'z', ' ', '-', ':', '\x7f'};
    for (std::size_t pos = 0; pos < good.size(); ++pos) {
        if (good[pos] == '-') {
            std::string s = good;
            s[pos] = '0';   // hyphen replaced by a digit
            bad.push_back(s);
            continue;
        }
        for (const char c : non_hex) {
            std::string s = good;
            s[pos] = c;
            bad.push_back(s);
        }
    }
    // An embedded NUL is not the end of the string.
    std::string nul = good;
    nul[10] = '\0';
    bad.push_back(nul);

    for (const std::string& s : bad) {
        ASSERT_EQ(snicholls::parse_uuid(s), std::nullopt) << '"' << s << '"';
        ASSERT_EQ(snicholls::parse_uuid(s, uuid_parse_options::lenient()), std::nullopt) << '"' << s << '"';
    }
}

TEST(UuidFromChars, ParsesAPrefixLikeStdFromChars)
{
    const std::string text = std::string(sample) + ",tail";
    uuid u;
    const auto [end, ec] = snicholls::from_chars(text.data(), text.data() + text.size(), u);
    EXPECT_EQ(ec, std::errc{});
    EXPECT_EQ(end, text.data() + 36);
    EXPECT_EQ(u, *snicholls::parse_uuid(sample));

    // Braces and URN report the end past their decoration.
    const std::string braced = "{" + std::string(sample) + "}x";
    const auto b = snicholls::from_chars(braced.data(), braced.data() + braced.size(), u, uuid_parse_options::lenient());
    EXPECT_EQ(b.ptr, braced.data() + 38);
    const std::string urn = "urn:uuid:" + std::string(sample) + "x";
    const auto r = snicholls::from_chars(urn.data(), urn.data() + urn.size(), u, uuid_parse_options::lenient());
    EXPECT_EQ(r.ptr, urn.data() + 45);
}

TEST(UuidFromChars, FailureLeavesOutputUnchangedAndPointsAtFirst)
{
    const uuid before = uuid::max();
    uuid u = before;
    const std::string text = "f81d4fae-7dec-11d0-a765-00a0c91e6bfX";
    const auto [end, ec] = snicholls::from_chars(text.data(), text.data() + text.size(), u);
    EXPECT_EQ(ec, std::errc::invalid_argument);
    EXPECT_EQ(end, text.data());
    EXPECT_EQ(u, before);
    const auto empty = snicholls::from_chars(text.data(), text.data(), u);
    EXPECT_EQ(empty.ec, std::errc::invalid_argument);
    EXPECT_EQ(u, before);
}

TEST(UuidParse, ConstexprAndNoexcept)
{
    static_assert(snicholls::parse_uuid(sample).value() == "f81d4fae-7dec-11d0-a765-00a0c91e6bf6"_uuid);
    static_assert(!snicholls::parse_uuid("not a uuid"sv).has_value());
    static_assert(noexcept(snicholls::parse_uuid(sample)));
    SUCCEED();
}

} // namespace
