//
//  digest_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/digest.hpp>: the layout contract, hexadecimal both ways, ordering, prefixes,
//  printing, the hash_input concept and md_hash's interface.
//

#include <snicholls/core/hash/digest.hpp>
#include <snicholls/core/hash/sha2.hpp>

#include <gtest/gtest.h>

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using namespace snicholls;

// Layout contract: exactly N bytes, alignment 1, trivially copyable, standard layout, structural
static_assert(sizeof(digest<32>) == 32 && alignof(digest<32>) == 1 && sizeof(digest<1>) == 1);
static_assert(std::is_trivially_copyable_v<digest<20>> && std::is_standard_layout_v<digest<20>> && std::is_aggregate_v<digest<20>>);
template <digest<2> D>
constexpr std::size_t first_byte = static_cast<std::size_t>(D[0]);
static_assert(first_byte<digest<2>{{std::byte{7}, std::byte{0}}}> == 7);   // usable as a template argument

constexpr auto abc = digest<3>{{std::byte{0xab}, std::byte{0x0c}, std::byte{0xff}}};
static_assert(abc.hex().view() == "ab0cff" && abc.hex().size() == 6 && abc.size() == 3);
static_assert(digest<3>::from_hex("ab0cff") == abc && digest<3>::from_hex("AB0CFF") == abc && digest<3>::from_hex("0xab0cFF") == abc);
static_assert(!digest<3>::from_hex("ab0cf") && !digest<3>::from_hex("ab0cffee") && !digest<3>::from_hex("ab0cfg") && !digest<3>::from_hex(""));
static_assert(digest<0>::from_hex("").has_value() && digest<0>{}.hex().view().empty());
static_assert(abc.first<2>().hex().view() == "ab0c" && abc.first<0>().size() == 0);
static_assert((digest<1>{{std::byte{1}}} <=> digest<1>{{std::byte{2}}}) == std::strong_ordering::less);
static_assert((digest<2>{{std::byte{0xff}, std::byte{0}}} <=> digest<2>{{std::byte{1}, std::byte{0xff}}}) == std::strong_ordering::greater);
static_assert(abc.span().size() == 3 && abc[2] == std::byte{0xff});

// hash_input: one-byte sequences and digests; not wider text, not signed char, not single units
static_assert(hash_input<std::string> && hash_input<std::string_view> && hash_input<const char*> && hash_input<char[4]>);
static_assert(hash_input<std::vector<std::uint8_t>> && hash_input<std::span<const std::byte>> && hash_input<std::array<std::byte, 8>>);
static_assert(hash_input<std::u8string_view> && hash_input<digest<32>> && hash_input<const digest<32>&>);
static_assert(!hash_input<std::u16string> && !hash_input<std::wstring_view> && !hash_input<char> && !hash_input<int>);
static_assert(!hash_input<std::vector<signed char>> && !hash_input<std::vector<int>>);

// The digest type of a hash function
static_assert(std::is_same_v<digest_of_t<sha256>, digest<32>> && std::is_same_v<sha256::digest_type, digest<32>>);
static_assert(noexcept(abc.hex()) && noexcept(digest<3>::from_hex(std::string_view{})));

TEST(DigestHex, RoundTripsAtRunTime)
{
    const auto d = sha256::hash("abc");
    const std::string text(d.hex().view());
    EXPECT_EQ(text.size(), 64u);
    EXPECT_EQ(digest<32>::from_hex(text), d);
    EXPECT_STREQ(d.hex().c_str(), text.c_str());
    const auto kept = d.hex();   // the text is a value: keep it, then view it
    EXPECT_EQ(kept.view().substr(0, 8), "ba7816bf");
    static_assert(!std::is_convertible_v<digest_hex<32>, std::string_view>);   // no view of a temporary by accident
}

TEST(DigestOrder, SortsLexicographicallyByByte)
{
    std::set<digest<2>> s{digest<2>::from_hex("ff00").value(), digest<2>::from_hex("00ff").value(), digest<2>::from_hex("0100").value()};
    std::string order;
    for (const auto& d : s)
        order += std::string(d.hex().view()) + " ";
    EXPECT_EQ(order, "00ff 0100 ff00 ");
}

TEST(DigestPrint, WritesLowercaseHexadecimal)
{
    std::ostringstream os;
    abc.print(os);
    EXPECT_EQ(os.str(), "ab0cff");
}

TEST(DigestBytes, AreWritableAndIterable)
{
    digest<4> d{};
    d[1] = std::byte{0x10};
    *(d.data() + 3) = std::byte{0x01};
    std::size_t sum = 0;
    for (const std::byte b : d)
        sum += static_cast<std::size_t>(b);
    EXPECT_EQ(sum, 0x11u);
    EXPECT_EQ(d.hex().view(), "00100001");
}

} // namespace
