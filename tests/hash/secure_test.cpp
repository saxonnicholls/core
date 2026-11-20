//
//  secure_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/hash/secure.hpp>: secure_wipe zeroes exactly the span (bytes, words, a digest) at run
//  time and in a constant expression; constant_time_equal agrees with == on every carrier and length.
//

#include <snicholls/core/hash/secure.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace snicholls;

constexpr bool wipes_at_compile_time()
{
    std::array<std::uint32_t, 4> w{1, 2, 3, 4};
    secure_wipe(std::span<std::uint32_t>(w).subspan(1, 2));
    digest<3> d{{std::byte{1}, std::byte{2}, std::byte{3}}};
    secure_wipe(d);
    return w == std::array<std::uint32_t, 4>{1, 0, 0, 4} && d == digest<3>{};
}
static_assert(wipes_at_compile_time());
static_assert(constant_time_equal("abc", "abc") && !constant_time_equal("abc", "abd") && !constant_time_equal("abc", "ab"));
static_assert(constant_time_equal("", ""));
static_assert(noexcept(constant_time_equal("a", "b")) && noexcept(secure_wipe(std::span<std::byte>{})));

TEST(SecureWipe, ZeroesExactlyTheSpanAtRunTime)
{
    std::vector<std::byte> bytes(64, std::byte{0xff});
    secure_wipe(std::span<std::byte>(bytes).subspan(8, 48));
    for (std::size_t i = 0; i < bytes.size(); ++i)
        EXPECT_EQ(bytes[i], (i >= 8 && i < 56) ? std::byte{0} : std::byte{0xff}) << i;

    std::string password = "correct horse battery staple";
    secure_wipe(std::span<char>(password.data(), password.size()));
    EXPECT_EQ(password, std::string(28, '\0'));

    std::array<std::uint64_t, 3> words{~0ull, ~0ull, ~0ull};
    secure_wipe(std::span<std::uint64_t, 3>(words));
    EXPECT_EQ(words, (std::array<std::uint64_t, 3>{}));

    auto d = digest<4>::from_hex("deadbeef").value();
    secure_wipe(d);
    EXPECT_EQ(d, digest<4>{});

    secure_wipe(std::span<std::byte>{});   // an empty span is fine
}

TEST(SecureCompare, AgreesWithEqualityOnEveryCarrierAndPosition)
{
    const std::vector<std::uint8_t> a{1, 2, 3, 4, 5};
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto b = a;
        b[i] ^= 0x80;
        EXPECT_FALSE(constant_time_equal(a, b)) << i;
    }
    EXPECT_TRUE(constant_time_equal(a, std::vector<std::uint8_t>{1, 2, 3, 4, 5}));
    const std::array<std::byte, 5> same{std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}};
    EXPECT_TRUE(constant_time_equal(a, same));                       // different carriers of the same bytes
    EXPECT_FALSE(constant_time_equal(a, std::span<const std::byte>(same).first(4)));   // a length differs
    const auto d = digest<2>::from_hex("0102").value();
    EXPECT_TRUE(constant_time_equal(d, std::span<const std::uint8_t>(a).first(2)));
    EXPECT_TRUE(constant_time_equal(std::string("x"), "x"));
}

} // namespace
