//
//  random_source_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/random_source_reference.hpp>.
//

#include <snicholls/core/interfaces/random_source_reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <system_error>
#include <type_traits>

namespace {

using namespace snicholls;

static_assert(std::is_trivially_copyable_v<test_random_source>);

// fill is constexpr: a test key drawn at compile time, and the stream's first byte checked there
constexpr std::array<std::byte, 16> compile_time_key(std::uint64_t seed)
{
    std::array<std::byte, 16> key{};
    test_random_source s{seed};
    (void)s.fill(key.data(), key.size());
    return key;
}
static_assert(compile_time_key(0)[0] == std::byte{0xaf}, "SplitMix64's first word for seed 0 is 0xe220a8397b1dcdaf");
static_assert(compile_time_key(1) != compile_time_key(2));
static_assert(!random_failed(test_random_source{5}.fill(nullptr, 0)));

TEST(TestRandomSource, TheSameSeedGivesTheSameStreamWhateverTheChunking)
{
    test_random_source whole{42}, pieces{42};
    std::array<std::byte, 41> a{}, b{};
    EXPECT_EQ(whole.fill(a.data(), a.size()), std::errc{});
    EXPECT_EQ(pieces.fill(b.data(), 3), std::errc{});
    EXPECT_EQ(pieces.fill(b.data() + 3, 30), std::errc{});
    EXPECT_EQ(pieces.fill(b.data() + 33, 8), std::errc{});
    EXPECT_EQ(a, b);
}

TEST(TestRandomSource, ItsFirstWordIsSplitMix64)
{
    // SplitMix64's first output from state seed + golden gamma: the published reference value for seed 0
    test_random_source s{0};
    std::array<std::byte, 8> w{};
    ASSERT_EQ(s.fill(w.data(), w.size()), std::errc{});
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < 8; ++i)
        v |= std::uint64_t(std::to_integer<unsigned>(w[i])) << (8 * i);
    EXPECT_EQ(v, 0xe220a8397b1dcdafULL);
}

TEST(TestRandomSource, DifferentSeedsDiffer)
{
    test_random_source a{1}, b{2};
    std::array<std::byte, 32> x{}, y{};
    ASSERT_EQ(a.fill(x.data(), x.size()), std::errc{});
    ASSERT_EQ(b.fill(y.data(), y.size()), std::errc{});
    EXPECT_NE(x, y);
}

TEST(UnavailableRandomSource, AlwaysFailsWithItsCode)
{
    unavailable_random_source s;
    std::array<std::byte, 4> out{};
    EXPECT_EQ(s.fill(out.data(), out.size()), std::errc::resource_unavailable_try_again);
    unavailable_random_source custom{std::errc::function_not_supported};
    EXPECT_EQ(custom.fill(out.data(), out.size()), std::errc::function_not_supported);
}

} // namespace
