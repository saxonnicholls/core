//
//  uint256_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The arithmetic is constexpr, so the identities and the values pinned from Python's integers are
//  static_asserts; the renderings, the hash and the generated cross-checks run at run time.
//

#include <snicholls/core/uint256.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using snicholls::uint256;

// Layout: four limbs and nothing else
static_assert(sizeof(uint256) == 32);
static_assert(std::is_trivially_copyable_v<uint256> && std::is_standard_layout_v<uint256>);

constexpr uint256 two64 = uint256::from_limbs(0, 1);
constexpr uint256 two128 = uint256::from_limbs(0, 0, 1);
constexpr uint256 two255 = uint256::from_limbs(0, 0, 0, std::uint64_t{1} << 63);
constexpr uint256 all64 = uint256{~std::uint64_t{0}};

// Addition and subtraction wrap like unsigned; the checked forms say so
static_assert(all64 + 1 == two64);
static_assert(two64 - 1 == all64);
static_assert(uint256::max() + 1 == uint256{});
static_assert(uint256{} - 1 == uint256::max());
static_assert(uint256::checked_add(uint256::max() - 1, 1) == uint256::max());
static_assert(!uint256::checked_add(uint256::max(), 1));
static_assert(uint256::checked_sub(two64, 1) == all64);
static_assert(!uint256::checked_sub(0, 1));

// Multiplication: (2^128 - 1)^2 = 2^256 - 2^129 + 1, and the checked form
constexpr uint256 two128m1 = uint256::from_limbs(~std::uint64_t{0}, ~std::uint64_t{0});
static_assert(two128m1 * two128m1 == uint256::from_limbs(1, 0, ~std::uint64_t{0} - 1, ~std::uint64_t{0}));
static_assert(!uint256::checked_mul(two128, two128));
static_assert(uint256::checked_mul(uint256::from_limbs(0, std::uint64_t{1} << 63), two128) == two255);   // 2^127 * 2^128
static_assert(uint256{6} * 7 == 42);

// Values pinned from Python: a, d, a / d, a % d and a * d mod 2^256
constexpr uint256 a = uint256::from_limbs(0x8796a5b4c3d2e1f0ull, 0x0f1e2d3c4b5a6978ull, 0x0123456789abcdefull, 0xdeadbeefcafebabeull);
constexpr uint256 d = uint256::from_limbs(0x0123456789abcdefull, 0x0123456789abcdefull);
constexpr uint256 q = uint256::from_limbs(0x4983b55344abaa00ull, 0xb6b4d0c169e2d87aull, 0xc3ull);
constexpr uint256 r = uint256::from_limbs(0x9b2a8c55a26d2bf0ull, 0x003c8d656b47a046ull);
static_assert(a / d == q && a % d == r);
static_assert(a.divmod(d) == std::pair{q, r});
static_assert(q * d + r == a);
static_assert(a * d == uint256::from_limbs(0x196fb4e90c1e1f10ull, 0x703efca944cf48b2ull, 0x33865d2f74a435a1ull, 0x5b6eeacb026b800eull));

// Division: short, long, dividend smaller than divisor, by self, by zero checked
static_assert(uint256{100} / 7 == 14 && uint256{100} % 7 == 2);
static_assert(two64 / 2 == uint256{std::uint64_t{1} << 63});
static_assert((two255 + 7) / two64 == uint256::from_limbs(0, 0, std::uint64_t{1} << 63) && (two255 + 7) % two64 == 7);
static_assert(uint256{5} / two64 == 0 && uint256{5} % two64 == 5);
static_assert(a / a == 1 && a % a == 0);
static_assert(uint256::max() / uint256::max() == 1);
static_assert(uint256::max() / 1 == uint256::max());
static_assert(!uint256::checked_div(1, 0) && !uint256::checked_divmod(1, 0));
static_assert(uint256::checked_div(a, d) == q);

// Powers of ten: 10^18 fits a limb, 10^20 does not, 10^77 is the last that fits
static_assert(uint256::pow10(0) == 1);
static_assert(uint256::pow10(18) == 1'000'000'000'000'000'000ull);
static_assert(uint256::pow10(20) == uint256::from_limbs(0x6bc75e2d63100000ull, 5));
static_assert(uint256::pow10(77).bit_width() == 256);
static_assert(uint256::pow10(77) > two255);

// Shifts and bits
static_assert(((uint256{1} << 200) >> 200) == 1);
static_assert((uint256{1} << 64) == two64 && (two64 >> 64) == 1);
static_assert((uint256{3} << 63) == uint256::from_limbs(std::uint64_t{1} << 63, 1));
static_assert((uint256{1} << 256) == 0 && (uint256::max() >> 256) == 0);
static_assert((uint256::max() >> 255) == 1);
static_assert(~uint256{} == uint256::max());
static_assert((uint256::max() & two64) == two64 && (uint256{5} | 2) == 7 && (uint256{5} ^ 1) == 4);
static_assert(uint256{}.bit_width() == 0 && uint256{1}.bit_width() == 1 && two64.bit_width() == 65 && uint256::max().bit_width() == 256);
static_assert(two64.bit(64) && !two64.bit(0) && !two64.bit(255));

// Observers and comparison
static_assert(uint256{5}.to_uint64() == 5 && !two64.to_uint64());
static_assert((two64 + 3).low64() == 3);
static_assert(two64 > all64 && uint256::max() > two64 && (two64 <=> two64) == std::strong_ordering::equal);
static_assert(uint256{} .is_zero() && !uint256{1}.is_zero() && !static_cast<bool>(uint256{}) && static_cast<bool>(two64));
static_assert(uint256::from_limbs(1, 2, 3, 4).limb(2) == 3 && uint256::from_limbs(1, 2, 3, 4).limbs()[3] == 4);

// Parsing never throws: decimal
static_assert(uint256::from_decimal("0") == 0);
static_assert(uint256::from_decimal("00012") == 12);
static_assert(uint256::from_decimal("115792089237316195423570985008687907853269984665640564039457584007913129639935") == uint256::max());
static_assert(!uint256::from_decimal("115792089237316195423570985008687907853269984665640564039457584007913129639936"));   // 2^256
static_assert(!uint256::from_decimal("") && !uint256::from_decimal("12a") && !uint256::from_decimal("-1") && !uint256::from_decimal(" 1"));

// Parsing never throws: hexadecimal
static_assert(uint256::from_hex("0xff") == 255 && uint256::from_hex("FF") == 255 && uint256::from_hex("0X0a") == 10);
static_assert(uint256::from_hex("0x0") == 0 && uint256::from_hex("0") == 0);
static_assert(!uint256::from_hex("0x") && !uint256::from_hex("") && !uint256::from_hex("0xg") && !uint256::from_hex("x1"));
static_assert(uint256::from_hex("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff") == uint256::max());
static_assert(!uint256::from_hex("10000000000000000000000000000000000000000000000000000000000000000"));   // 65 nibbles
static_assert(uint256::from_hex("0x00000000000000000000000000000000000000000000000000000000000000000000001") == 1);   // leading zeros

constexpr std::string_view one_digit{"1"};   // a literal's conversion to string_view is not noexcept; the function is
static_assert(noexcept(a + d) && noexcept(a * d) && noexcept(a << 1) && noexcept(uint256::checked_add(a, d)) &&
              noexcept(uint256::from_decimal(one_digit)) && noexcept(uint256::from_hex(one_digit)) && noexcept(a.bit_width()));

TEST(Uint256Text, RendersDecimalExactly)
{
    EXPECT_EQ(uint256{}.to_string(), "0");
    EXPECT_EQ(uint256{42}.to_string(), "42");
    EXPECT_EQ(two64.to_string(), "18446744073709551616");
    EXPECT_EQ(two128.to_string(), "340282366920938463463374607431768211456");
    EXPECT_EQ(two255.to_string(), "57896044618658097711785492504343953926634992332820282019728792003956564819968");
    EXPECT_EQ(uint256::max().to_string(), "115792089237316195423570985008687907853269984665640564039457584007913129639935");
    EXPECT_EQ(uint256::pow10(20).to_string(), "100000000000000000000");
    EXPECT_EQ(uint256::pow10(27).to_string(), "1" + std::string(27, '0'));   // a chunk boundary of nine digits
}

// Units: a token's decimals, exactly
static_assert(uint256::parse_units("10000.5", 6) == uint256{10'000'500'000ULL});   // USDT, USDC
static_assert(uint256::parse_units("1", 18) == uint256::pow10(18));                 // ETH
static_assert(uint256::parse_units("0.00100000", 6) == uint256{1000});              // Binance's eight places, zeros past six
static_assert(!uint256::parse_units("0.0000001", 6));                               // a seventh place would be rounded: refused
static_assert(uint256::parse_units(".5", 6) == uint256{500000} && uint256::parse_units("7.", 6) == uint256{7000000});
static_assert(uint256::parse_units("42", 0) == uint256{42} && !uint256::parse_units("4.2", 0));
static_assert(!uint256::parse_units("", 6) && !uint256::parse_units(".", 6) && !uint256::parse_units("1.2.3", 6));
static_assert(!uint256::parse_units("-1", 6) && !uint256::parse_units("1,000", 6));
static_assert(!uint256::parse_units("1", 77 + 1) && uint256::parse_units("1", 77).has_value());
static_assert(!uint256::parse_units("1000", 75));                                   // 10^78 overflows
static_assert(noexcept(uint256::parse_units(std::string_view{}, 6)));

TEST(Uint256Units, FormatsEveryDigitOrTrimmed)
{
    using digits = uint256::fraction_digits;
    EXPECT_EQ(uint256{10'000'500'000ULL}.format_units(6), "10000.500000");
    EXPECT_EQ(uint256{10'000'500'000ULL}.format_units(6, digits::trimmed), "10000.5");
    EXPECT_EQ(uint256{10'000'000'000ULL}.format_units(6, digits::trimmed), "10000");
    EXPECT_EQ(uint256{1000}.format_units(6), "0.001000");
    EXPECT_EQ(uint256{1000}.format_units(8), "0.00001000");
    EXPECT_EQ(uint256{}.format_units(6), "0.000000");
    EXPECT_EQ(uint256{}.format_units(6, digits::trimmed), "0");
    EXPECT_EQ(uint256{42}.format_units(0), "42");
    EXPECT_EQ(uint256::max().format_units(18), "115792089237316195423570985008687907853269984665640564039457.584007913129639935");
}

TEST(Uint256Units, RoundTripThroughText)
{
    for (const char* text : {"0.000001", "1.5", "123456789.123456", "0"})
        EXPECT_EQ(uint256::parse_units(text, 6)->format_units(6, uint256::fraction_digits::trimmed), text);
    const auto wei = uint256::parse_units("1.000000000000000001", 18);
    ASSERT_TRUE(wei.has_value());
    EXPECT_EQ(wei->format_units(18), "1.000000000000000001");
}

TEST(Uint256Text, PrintsInDecimalAsAPrintable)
{
    std::ostringstream os;
    two64.print(os);
    EXPECT_EQ(os.str(), "18446744073709551616");
    os.str("");
    uint256{}.print(os);
    EXPECT_EQ(os.str(), "0");
}

TEST(Uint256Text, RendersHexadecimalMinimalAndAsAnAbiWord)
{
    EXPECT_EQ(uint256{}.to_hex(), "0x0");
    EXPECT_EQ(uint256{255}.to_hex(), "0xff");
    EXPECT_EQ(two64.to_hex(), "0x10000000000000000");
    EXPECT_EQ(uint256::max().to_hex(), "0x" + std::string(64, 'f'));
    EXPECT_EQ(uint256{1}.to_abi_word(), std::string(63, '0') + "1");
    EXPECT_EQ(uint256{1}.to_abi_word().size(), 64u);
    EXPECT_EQ(a.to_abi_word(), "deadbeefcafebabe0123456789abcdef0f1e2d3c4b5a69788796a5b4c3d2e1f0");
}

TEST(Uint256Text, RoundTripsThroughBothRenderings)
{
    const std::vector<uint256> values{uint256{}, uint256{1}, all64, two64, two128m1, a, d, q, r, two255, uint256::max(), uint256::pow10(77)};
    for (const auto& v : values) {
        EXPECT_EQ(uint256::from_decimal(v.to_string()), v) << v.to_string();
        EXPECT_EQ(uint256::from_hex(v.to_hex()), v) << v.to_hex();
        EXPECT_EQ(uint256::from_hex(v.to_abi_word()), v) << v.to_abi_word();
    }
}

std::vector<uint256> generated(std::size_t n)
{
    std::vector<uint256> out;
    std::uint64_t x = 0x9e3779b97f4a7c15ull;
    auto next = [&] { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (std::size_t i = 0; i < n; ++i) {
        // every width: 1 to 4 limbs, with some limbs zero
        const auto l0 = next(), l1 = i % 4 >= 1 ? next() : 0, l2 = i % 4 >= 2 ? next() : 0, l3 = i % 4 >= 3 ? next() : 0;
        out.push_back(uint256::from_limbs(l0, i % 5 == 0 ? 0 : l1, l2, i % 7 == 0 ? 0 : l3));
    }
    return out;
}

TEST(Uint256Arithmetic, DivisionAndMultiplicationAgreeOnGeneratedValues)
{
    const auto values = generated(200);
    for (const auto& n : values)
        for (const auto& m : values) {
            if (m.is_zero())
                continue;
            const auto [quotient, remainder] = n.divmod(m);
            EXPECT_LT(remainder, m);
            EXPECT_EQ(quotient * m + remainder, n) << n.to_hex() << " / " << m.to_hex();
            if (const auto product = uint256::checked_mul(n, m)) {
                EXPECT_EQ(*product / m, n);   // a product that fits divides back exactly
            }
        }
}

TEST(Uint256Arithmetic, IncrementsAndDecrementsWrap)
{
    uint256 v = uint256::max();
    EXPECT_EQ(v++, uint256::max());
    EXPECT_EQ(v, uint256{});
    EXPECT_EQ(--v, uint256::max());
    EXPECT_EQ(++v, uint256{});
    EXPECT_EQ(v--, uint256{});
    EXPECT_EQ(v, uint256::max());
}

TEST(Uint256Hash, EqualValuesHashEqualAndKeyAnUnorderedSet)
{
    std::unordered_set<uint256> seen{a, d, q, r, two64};
    EXPECT_EQ(seen.size(), 5u);
    EXPECT_TRUE(seen.contains(uint256::from_limbs(0, 1)));
    EXPECT_FALSE(seen.contains(uint256{1}));
    EXPECT_EQ(std::hash<uint256>{}(a), std::hash<uint256>{}(uint256::from_decimal(a.to_string()).value()));
    EXPECT_NE(std::hash<uint256>{}(uint256{1}), std::hash<uint256>{}(two64));   // the limb position counts
}

} // namespace
