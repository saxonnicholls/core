//
//  uniform_int_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/uniform_int.hpp>: each library's algorithm against the real
//  std::uniform_int_distribution where the test is built with that library, committed sequences from both
//  libraries everywhere, bounds, the default (core) draw, and the layout and noexcept contract.
//

#include <snicholls/core/stats/uniform_int.hpp>

#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

namespace {

namespace stats = snicholls::stats;
using stats::draw;
using stats::uniform_int;

// The library this test is built with, if core reproduces it: libc++ anywhere, libstdc++ on a 64-bit target.
#if defined(_LIBCPP_VERSION)
constexpr bool native_known = true;
constexpr draw native = draw::libcxx;
#elif defined(__GLIBCXX__) && defined(__SIZEOF_INT128__)
constexpr bool native_known = true;
constexpr draw native = draw::libstdcxx;
#else
constexpr bool native_known = false;
constexpr draw native = draw::core;
#endif

constexpr int digest_draws = 10000;

// FNV-1a over the draws, each as 64 bits: one number standing for 10,000 draws.
template <typename E, typename Dist>
std::uint64_t digest(E g, Dist d)
{
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < digest_draws; ++i) {
        h ^= static_cast<std::uint64_t>(d(g));
        h *= 0x100000001b3ull;
    }
    return h;
}

// The digests were taken from the real libraries: libstdc++ under GCC 13, GCC 15 and Clang 18 on Ubuntu,
// libc++ under Apple Clang 17, each matching std::uniform_int_distribution draw for draw.
// With Native, the draws are compared with std::uniform_int_distribution's instead.
template <bool Native>
struct check {
    template <typename E, typename T>
    static void expect_digests(typename E::result_type seed, T lo, T hi, std::uint64_t libstdcxx, std::uint64_t libcxx)
    {
        const E g(seed);
        SCOPED_TRACE(testing::Message() << "[" << lo << ", " << hi << "], seed " << seed);
        if constexpr (!Native) {
            EXPECT_EQ(digest(g, uniform_int<T, draw::libstdcxx>(lo, hi)), libstdcxx);
            EXPECT_EQ(digest(g, uniform_int<T, draw::libcxx>(lo, hi)), libcxx);
        }
        else if constexpr (native_known) {
            E mine = g, theirs = g;
            uniform_int<T, native> ours(lo, hi);
            std::uniform_int_distribution<T> std_dist(lo, hi);
            for (int i = 0; i < 2000; ++i) {
                const T a = ours(mine), b = std_dist(theirs);
                ASSERT_EQ(a, b) << "draw " << i;
                ASSERT_TRUE(lo <= a && a <= hi);
            }
            EXPECT_TRUE(mine == theirs);   // the same number of generator calls
        }
    }

    static void every_case()
    {
        expect_digests<std::mt19937, int>(1, 0, 9, 0x7e76cf372762c400ull, 0x8d0031c63fa769feull);
        expect_digests<std::mt19937, int>(1, -5, 5, 0x27f89d5e2f9181d3ull, 0x5e46be16b9ea93b7ull);
        expect_digests<std::mt19937, int>(1, 7, 7, 0x3e843bfe02a637f5ull, 0x3e843bfe02a637f5ull);
        expect_digests<std::mt19937, int>(1, INT_MIN, INT_MAX, 0x144e9bee50d1f0e9ull, 0xd1b8ffe150d1f0e9ull);
        expect_digests<std::mt19937, unsigned>(1, 0u, UINT_MAX, 0xb515cc2550d1f0e9ull, 0xb515cc2550d1f0e9ull);
        expect_digests<std::mt19937, unsigned>(1, 0u, 2147483648u, 0x4b5c2451feee4adeull, 0x5924b26a9c94c386ull);
        expect_digests<std::mt19937, short>(1, -100, 100, 0x64feae510c9605f2ull, 0xd6efc9c015736c2cull);
        expect_digests<std::mt19937, unsigned long long>(1, 0ull, 1099511627776ull, 0xd4dc2f113ffaecbdull, 0x023f6823291dc0a2ull);
        expect_digests<std::mt19937, unsigned long long>(1, 0ull, ULLONG_MAX, 0x485faaa6a30cbe2bull, 0x485faaa6a30cbe2bull);
        expect_digests<std::mt19937, long long>(1, -1099511627776ll, 1099511627776ll, 0x8fdc4cf297e12f10ull, 0xf641753f3c687822ull);
        expect_digests<std::mt19937, std::size_t>(1, std::size_t{0}, std::size_t{1000}, 0x637696296b65f873ull, 0x48a5ece28355b0b9ull);
        expect_digests<std::mt19937, std::size_t>(1, std::size_t{0}, std::size_t{4294967295u}, 0xb515cc2550d1f0e9ull, 0xb515cc2550d1f0e9ull);
        expect_digests<std::mt19937_64, int>(2026, 0, 9, 0xaa69c7dce36df38dull, 0x5d74087ba09a82f0ull);
        expect_digests<std::mt19937_64, unsigned long long>(2026, 0ull, 1000000000000000000ull, 0xfd5271d675831cbbull, 0x5a138d055b40c879ull);
        expect_digests<std::mt19937_64, unsigned long long>(2026, 0ull, ULLONG_MAX, 0x27590a30654dcedcull, 0x27590a30654dcedcull);
        expect_digests<std::mt19937_64, long long>(2026, LLONG_MIN, 0ll, 0x517894bc2a3f86b3ull, 0x28b82562b51ed070ull);
        expect_digests<std::mt19937_64, unsigned>(2026, 0u, UINT_MAX, 0xc2aaff8e68cfb2c9ull, 0xe81c3cc8654dcedcull);
        expect_digests<std::minstd_rand, int>(2026, 0, 9, 0x94b058f391bf8ea8ull, 0xc3c97301f6516995ull);
        expect_digests<std::minstd_rand, int>(2026, 0, 2147483645, 0xd09fa44c75d881f8ull, 0x0ea35335a5cf4f41ull);
        expect_digests<std::minstd_rand, unsigned long long>(2026, 0ull, 1099511627776ull, 0x984aad884c81f7faull, 0x15176ba1b57a671aull);
        expect_digests<std::minstd_rand, int>(2026, -1000, 1000000, 0xc7391cb80401299cull, 0xadea1637d9635d60ull);
        expect_digests<std::ranlux24_base, int>(2026, 0, 9, 0xdfb0ff8afb7db375ull, 0x4acacde6e4e245a5ull);
        expect_digests<std::ranlux24_base, unsigned>(2026, 0u, UINT_MAX, 0x30e9ee62f588d777ull, 0xdbade174157ed777ull);
        expect_digests<std::ranlux24_base, long long>(2026, -1000000000000ll, 1000000000000ll, 0x0c26ff201e5b4d29ull, 0x16c2f10a63e37570ull);
        expect_digests<snicholls::stats::wyrand, int>(2026, 0, 9, 0x57748bdc21214643ull, 0x2d4535dea64f8e9bull);
        expect_digests<snicholls::stats::wyrand, unsigned long long>(2026, 0ull, ULLONG_MAX, 0xf96ef549ba3f57c7ull, 0xf96ef549ba3f57c7ull);
        expect_digests<snicholls::stats::wyrand, long long>(2026, -3ll, 3ll, 0x46b21f3d554defb7ull, 0x6c6e70336a05ee05ull);
    }
};

TEST(UniformIntLibraries, ReproduceBothStandardLibrariesOnEveryPlatform)
{
    check<false>::every_case();
}

TEST(UniformIntLibraries, MatchTheStandardDistributionOfTheLibraryBuiltWith)
{
    if (!native_known)
        GTEST_SKIP() << "built with a standard library core does not reproduce (MSVC's, or a 32-bit libstdc++)";
    check<true>::every_case();   // draw for draw against std::uniform_int_distribution
}

TEST(UniformIntLibraries, PickLlamaCppsFirstHellaSwagTasks)
{
    // mt19937 seeded 1 drawing indices in [0, 99], as llama.cpp's task selection does.
    std::mt19937 g1(1), g2(1);
    uniform_int<std::size_t, draw::libstdcxx> linux_build(0, 99);
    uniform_int<std::size_t, draw::libcxx> mac_build(0, 99);
    const std::array<std::size_t, 6> expect_linux{41, 99, 72, 93, 0, 12}, expect_mac{37, 12, 72, 9, 75, 5};
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(linux_build(g1), expect_linux[i]);
        EXPECT_EQ(mac_build(g2), expect_mac[i]);
    }
}

TEST(UniformIntCore, IsUniformIndexOffsetByTheLowerBound)
{
    stats::wyrand a(2026), b(2026);
    uniform_int<int> d(-3, 3);
    for (int i = 0; i < 1000; ++i)
        EXPECT_EQ(d(a), -3 + static_cast<int>(stats::uniform_index(b, 7)));
    uniform_int<unsigned long long> full;   // the whole range: the 64 bits themselves
    EXPECT_EQ(full(a), stats::random_u64(b));
}

TEST(UniformIntCore, DrawsFromMt19937AndCoversEveryValueEvenly)
{
    std::mt19937 g(1);
    uniform_int<short> d(-2, 2);
    std::array<int, 5> counts{};
    for (int i = 0; i < 50000; ++i) {
        const short v = d(g);
        ASSERT_TRUE(v >= -2 && v <= 2);
        ++counts[static_cast<std::size_t>(v + 2)];
    }
    for (int c : counts)
        EXPECT_NEAR(c, 10000, 400);
}

TEST(UniformIntBounds, EqualBoundsGiveTheBoundAndDefaultBoundsAreTheTypesRange)
{
    std::mt19937 g(7);
    EXPECT_EQ((uniform_int<int, draw::libstdcxx>(5, 5)(g)), 5);
    EXPECT_EQ((uniform_int<int, draw::libcxx>(5, 5)(g)), 5);
    EXPECT_EQ((uniform_int<int>(5, 5)(g)), 5);
    constexpr uniform_int<long long, draw::libcxx> d;
    EXPECT_EQ(d.a(), 0);
    EXPECT_EQ(d.min(), 0);
    EXPECT_EQ(d.b(), LLONG_MAX);
    EXPECT_EQ(d.max(), LLONG_MAX);
    EXPECT_TRUE((uniform_int<int>(1, 2) == uniform_int<int>(1, 2)));
    EXPECT_FALSE((uniform_int<int>(1, 2) == uniform_int<int>(1, 3)));
    uniform_int<int> r(1, 2);
    r.reset();
    EXPECT_EQ(r.b(), 2);
}

// Compile time: every algorithm runs in a constant expression over a constexpr generator.
template <draw D>
consteval int sum_of_draws()
{
    stats::wyrand g(2026);
    uniform_int<int, D> d(1, 6);
    int s = 0;
    for (int i = 0; i < 100; ++i) {
        const int v = d(g);
        if (v < 1 || v > 6)
            return -1;
        s += v;
    }
    return s;
}
static_assert(sum_of_draws<draw::core>() > 100 && sum_of_draws<draw::libstdcxx>() > 100 && sum_of_draws<draw::libcxx>() > 100);

// The layout and noexcept contract: two Ts, trivially copyable, noexcept exactly when the generator is.
static_assert(sizeof(uniform_int<int>) == 2 * sizeof(int) && std::is_trivially_copyable_v<uniform_int<long long, draw::libcxx>>);
static_assert(noexcept(std::declval<const uniform_int<int>&>()(std::declval<stats::wyrand&>())));
static_assert(noexcept(std::declval<const uniform_int<int, draw::libcxx>&>()(std::declval<stats::wyrand&>())));
static_assert(stats::standard_int_type<unsigned long> && !stats::standard_int_type<char> && !stats::standard_int_type<bool> &&
              !stats::standard_int_type<signed char>);

// draw::core needs 32 or 64 full bits; the library algorithms take any generator.
template <typename D, typename G>
concept draws_from = requires(const D d, G g) { d(g); };
static_assert(!draws_from<uniform_int<int>, std::minstd_rand> && draws_from<uniform_int<int, draw::libcxx>, std::minstd_rand> &&
              draws_from<uniform_int<int, draw::libstdcxx>, std::ranlux24_base>);

} // namespace
