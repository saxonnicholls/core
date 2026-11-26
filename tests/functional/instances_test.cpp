//
//  instances_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/instances.hpp>: every instance of the standard library's and core's types, at compile time and at run time.
//

#include <snicholls/core/containers/inplace_vector.hpp>
#include <snicholls/core/functional/instances.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
namespace fn = snicholls::fn;
using snicholls::result; using snicholls::ok; using snicholls::err;
enum class e { bad };
using R = result<int, e>;
constexpr auto dbl = [](int x) { return 2 * x; };
constexpr auto half = [](int x) { return x % 2 ? std::optional<int>{} : std::optional<int>{x / 2}; };
static_assert(fn::fmap(dbl, std::optional<int>{3}) == std::optional<int>{6});
static_assert(fn::fmap(dbl, R(ok(3))).unwrap_unchecked() == 6);
static_assert(fn::fmap(dbl, std::array{1, 2}) == std::array{2, 4});
static_assert(fn::fmap(dbl, std::tuple{1, 2}) == std::tuple{2, 4});
static_assert(fn::pure<std::optional<int>>(3) == std::optional<int>{3});
static_assert(fn::ap(std::optional{dbl}, std::optional<int>{3}) == std::optional<int>{6});
static_assert(fn::lift_a2([](int a, int b) { return a - b; }, std::optional<int>{5}, std::optional<int>{3}) == std::optional<int>{2});
static_assert(fn::lift_a2([](int a, int b) { return a * 10 + b; }, std::array{1, 2}, std::array{3, 4}) == std::array{13, 14, 23, 24});
static_assert(fn::mbind(std::optional<int>{4}, half) == std::optional<int>{2});
static_assert(fn::join(std::optional<std::optional<int>>{std::optional<int>{1}}) == std::optional<int>{1});
static_assert(fn::mbind(std::array{1, 2}, [](int x) { return std::array{x, x * 10}; }) == std::array{1, 10, 2, 20});
static_assert(fn::kleisli(half, half)(8) == std::optional<int>{2});
static_assert(fn::alt(std::optional<int>{}, std::optional<int>{2}) == std::optional<int>{2});
static_assert(fn::mzero<std::optional<int>> == std::nullopt);
static_assert(fn::combine(fn::sum_of<int>{2}, fn::sum_of<int>{3}).value == 5);
static_assert(fn::mconcat(std::array{fn::max_of<int>{2}, fn::max_of<int>{7}}).value == 7);
static_assert(fn::mconcat(std::array<fn::max_of<int>, 0>{}).value == std::numeric_limits<int>::lowest());
static_assert(fn::sconcat(std::tuple{fn::sum_of<int>{1}, fn::all_of{true}}, std::tuple{fn::sum_of<int>{2}, fn::all_of{false}}) == std::tuple{fn::sum_of<int>{3}, fn::all_of{false}});
static_assert(fn::mempty<std::tuple<fn::sum_of<int>, fn::product_of<int>>>() == std::tuple{fn::sum_of<int>{0}, fn::product_of<int>{1}});
static_assert(fn::combine(std::string("ab"), std::string("c")) == "abc");
static_assert(fn::combine(fn::endo{dbl}, fn::endo{[](int x) { return x + 1; }})(3) == 8);
static_assert(fn::bimap(dbl, dbl, std::pair{1, 2}) == std::pair{2, 4});
static_assert(fn::first(dbl, std::pair{1, 2}) == std::pair{2, 2});
static_assert(fn::bimap(dbl, [](e) { return 1; }, R(err(e::bad))).unwrap_err_unchecked() == 1);
static_assert(fn::dimap([](int x) { return x + 1; }, dbl, dbl)(1) == 8);
static_assert(fn::replace(7, std::optional<int>{1}) == std::optional<int>{7});
static_assert(fn::mthen(std::optional<int>{1}, std::optional<char>{'a'}) == std::optional<char>{'a'});
static_assert(fn::semigroup<R> && !fn::monoid<R> && fn::monad<std::array<int, 2>> && !fn::functor<int>);
static_assert(fn::alt(R(err(e::bad)), R(ok(1))).unwrap_unchecked() == 1);
static_assert(fn::ap(fn::pure<R>(dbl), R(ok(2))).unwrap_unchecked() == 4);

static_assert(!fn::ap(std::optional{dbl}, std::optional<int>{}).has_value());
static_assert(fn::ap(std::array{dbl, dbl}, std::array{1, 2}) == std::array{2, 4, 2, 4});
static_assert(fn::pure<std::array<int, 1>>(5) == std::array{5});
static_assert(fn::ap(fn::pure<R>(dbl), R(err(e::bad))).unwrap_err_unchecked() == e::bad);
static_assert(fn::combine(std::optional<fn::sum_of<int>>{{1}}, std::optional<fn::sum_of<int>>{}) == std::optional<fn::sum_of<int>>{{1}});
static_assert(fn::combine(R(err(e::bad)), R(ok(2))).unwrap_unchecked() == 2);   // Either's semigroup: the first success

TEST(FunctionalInstances, AnRvalueOptionalMovesItsValueIntoFmap)
{
    auto r = fn::fmap([](std::unique_ptr<int> p) { return *p + 1; }, std::optional<std::unique_ptr<int>>(std::make_unique<int>(1)));
    EXPECT_EQ(r, std::optional<int>(2));
}

// inplace_vector: same size and capacity, at compile time, moving out of an rvalue
using small = snicholls::containers::inplace_vector<int, 4>;
static_assert(fn::fmap([](int x) { return x * 2; }, small{1, 2, 3}) == small{2, 4, 6});
static_assert(fn::fmap([](int x) { return x * 0.5; }, small{1}) == snicholls::containers::inplace_vector<double, 4>{0.5});
static_assert(fn::fmap([](int x) { return x; }, small{}).empty() && decltype(fn::fmap(dbl, small{}))::capacity() == 4);
static_assert(fn::functor<small> && !fn::monoid<small>);
static_assert(noexcept(fn::fmap([](int x) noexcept { return x; }, small{})) && !noexcept(fn::fmap(dbl, small{})));

TEST(FunctionalInstances, InplaceVectorsMapElementByElement)
{
    snicholls::containers::inplace_vector<std::string, 3> words{"a", "bc"};
    const auto sizes = fn::fmap([](std::string w) { return w.size(); }, std::move(words));   // moved into f
    EXPECT_EQ(sizes.size(), 2u);
    EXPECT_EQ(sizes[0], 1u);
    EXPECT_EQ(sizes[1], 2u);
}

TEST(FunctionalInstances, VectorsMapAndConcatenate)
{
    EXPECT_EQ(fn::fmap([](int x) { return x * 2; }, std::vector<int>{1, 2}), (std::vector<int>{2, 4}));
    EXPECT_EQ(fn::mconcat(std::vector<std::vector<int>>{{1}, {2, 3}}), (std::vector<int>{1, 2, 3}));
}

} // namespace
