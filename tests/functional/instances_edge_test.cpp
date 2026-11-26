//
//  instances_edge_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Edge cases of <snicholls/core/functional/instances.hpp>: empty and single carriers, nesting (an optional of a
//  tuple of monoids, an array of arrays), the list monad's lengths, Either's first error, and the noexcept claims.
//

#include <snicholls/core/functional/instances.hpp>

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace fn = snicholls::fn;
using snicholls::err;
using snicholls::ok;
using snicholls::result;

enum class e { a, b };
using R = result<int, e>;
constexpr auto dbl = [](int x) noexcept { return 2 * x; };

// ---- empty and single ----------------------------------------------------------------------------------------------

static_assert(fn::fmap(dbl, std::array<int, 0>{}) == std::array<int, 0>{});
static_assert(fn::mbind(std::array<int, 0>{}, [](int x) { return std::array{x, x}; }).empty());
static_assert(fn::mbind(std::array{1, 2}, [](int) { return std::array<int, 0>{}; }).empty());   // N * 0
static_assert(fn::lift_a2(std::plus<>{}, std::array<int, 0>{}, std::array{1, 2}).empty());
static_assert(fn::fmap(dbl, std::tuple<>{}) == std::tuple<>{});
static_assert(fn::fmap(dbl, std::tuple{1}) == std::tuple{2});
static_assert(fn::combine(std::tuple<>{}, std::tuple<>{}) == std::tuple<>{});
static_assert(fn::combine(std::array<fn::sum_of<int>, 0>{}, std::array<fn::sum_of<int>, 0>{}).empty());
static_assert(fn::mempty<std::array<fn::product_of<int>, 2>>() == std::array{fn::product_of<int>{1}, fn::product_of<int>{1}});

// ---- nesting -------------------------------------------------------------------------------------------------------

using stats = std::tuple<fn::sum_of<int>, fn::min_of<int>, fn::max_of<int>>;
constexpr stats one(int x) { return {{x}, {x}, {x}}; }
static_assert(fn::combine(std::optional<stats>{one(3)}, std::optional<stats>{one(-2)}) ==
              std::optional<stats>{stats{{1}, {-2}, {3}}});
static_assert(fn::combine(std::optional<stats>{}, std::optional<stats>{one(4)}) == std::optional<stats>{one(4)});
static_assert(fn::mempty<std::optional<stats>>() == std::nullopt);
static_assert(fn::fmap([](const std::array<int, 2>& a) { return a[0] + a[1]; }, std::array{std::array{1, 2}, std::array{3, 4}}) ==
              std::array{3, 7});
static_assert(fn::join(std::array{std::array{1, 2}, std::array{3, 4}}) == std::array{1, 2, 3, 4});
static_assert(fn::fmap([](std::optional<int> o) { return o.has_value(); }, std::optional<std::optional<int>>{std::optional<int>{}}) ==
              std::optional<bool>{false});

// ---- Either: first error, first success ---------------------------------------------------------------------------

static_assert(fn::mbind(R(err(e::a)), [](int) { return R(err(e::b)); }).unwrap_err_unchecked() == e::a);
static_assert(fn::lift_a2(std::plus<>{}, R(err(e::b)), R(err(e::a))).unwrap_err_unchecked() == e::b);
static_assert(fn::alt(R(ok(1)), R(ok(2))).unwrap_unchecked() == 1);
static_assert(fn::combine(R(err(e::a)), R(err(e::b))).unwrap_err_unchecked() == e::b);   // no success: the last error
static_assert(fn::bimap(dbl, [](e x) { return x == e::a ? 'a' : 'b'; }, R(err(e::b))).unwrap_err_unchecked() == 'b');
static_assert(fn::join(result<R, e>(ok(R(err(e::b))))).unwrap_err_unchecked() == e::b);

// ---- noexcept ------------------------------------------------------------------------------------------------------

constexpr std::optional<int> o{};
constexpr std::array<int, 2> arr{1, 2};
static_assert(noexcept(fn::fmap(dbl, o)) && noexcept(fn::fmap(dbl, arr)));
static_assert(!noexcept(fn::fmap([](int x) { return x; }, arr)));
static_assert(!noexcept(fn::combine(std::string(), std::string())));   // allocation is never noexcept
static_assert(noexcept(fn::combine(fn::sum_of<int>{}, fn::sum_of<int>{})));

TEST(FunctionalInstancesEdge, EmptyVectorsAndStrings)
{
    EXPECT_TRUE(fn::fmap(dbl, std::vector<int>{}).empty());
    EXPECT_TRUE(fn::mconcat(std::vector<std::string>{}).empty());
    EXPECT_EQ(fn::mconcat(std::vector<std::string>{"a", "", "b"}), "ab");
    EXPECT_EQ(fn::combine(std::vector<int>{}, std::vector<int>{4}), std::vector<int>{4});
}

TEST(FunctionalInstancesEdge, AnOptionalOfStringsCombinesTheValues)
{
    EXPECT_EQ(fn::combine(std::optional<std::string>("EUR"), std::optional<std::string>("USD")), std::optional<std::string>("EURUSD"));
    EXPECT_EQ(fn::combine(std::optional<std::string>(), std::optional<std::string>("USD")), std::optional<std::string>("USD"));
}

} // namespace
