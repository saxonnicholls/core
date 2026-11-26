//
//  monad_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/monad.hpp>: fmap, and_then, value_or, lift, sequence and traverse
//  over std::optional, snicholls::result and std::variant, at compile time and at run time.
//

#include <snicholls/core/functional/curry.hpp>
#include <snicholls/core/functional/monad.hpp>

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace {

namespace fn = snicholls::fn;
using snicholls::err;
using snicholls::ok;
using snicholls::result;

enum class failure { stale, crossed };
using quote = result<int, failure>;

constexpr auto dbl = [](int x) noexcept { return x * 2; };
constexpr auto add = [](int a, int b) { return a + b; };
constexpr auto half = [](int x) { return x % 2 == 0 ? std::optional<int>(x / 2) : std::nullopt; };
constexpr auto checked_half = [](int x) -> quote { return x % 2 == 0 ? quote(ok(x / 2)) : quote(err(failure::crossed)); };

// ---- fmap ----------------------------------------------------------------------------------------------------------

static_assert(fn::fmap(dbl, std::optional<int>{4}) == std::optional<int>{8});
static_assert(!fn::fmap(dbl, std::optional<int>{}).has_value());
static_assert(std::is_same_v<decltype(fn::fmap([](int) { return 1.5; }, std::optional<int>{})), std::optional<double>>);
static_assert(fn::fmap(dbl, quote(ok(4))).unwrap_unchecked() == 8);
static_assert(fn::fmap(dbl, quote(err(failure::stale))).unwrap_err_unchecked() == failure::stale);
static_assert(*fn::curry<2>(fn::fmap)(dbl)(std::optional<int>{1}) == 2);  // fmap is a value too

// ---- and_then and value_or -----------------------------------------------------------------------------------------

static_assert(*fn::and_then(half, std::optional<int>{4}) == 2);
static_assert(!fn::and_then(half, std::optional<int>{3}).has_value());
static_assert(!fn::and_then(half, std::optional<int>{}).has_value());
static_assert(fn::and_then(checked_half, quote(ok(4))).unwrap_unchecked() == 2);
static_assert(fn::and_then(checked_half, quote(ok(3))).unwrap_err_unchecked() == failure::crossed);
static_assert(fn::value_or(std::optional<int>{}, 7) == 7);
static_assert(fn::value_or(std::optional<int>{1}, 7) == 1);
static_assert(fn::value_or(quote(err(failure::stale)), 7) == 7);
static_assert(fn::value_or(quote(ok(1)), 7) == 1);

// ---- lift ----------------------------------------------------------------------------------------------------------

static_assert(*fn::lift(add)(std::optional<int>{1}, std::optional<int>{2}) == 3);
static_assert(!fn::lift(add)(std::optional<int>{1}, std::optional<int>{}).has_value());
static_assert(*fn::lift(dbl)(std::optional<int>{4}) == 8);
static_assert(fn::lift(add)(quote(ok(1)), quote(ok(2))).unwrap_unchecked() == 3);
static_assert(fn::lift(add)(quote(ok(1)), quote(err(failure::crossed))).unwrap_err_unchecked() == failure::crossed);
static_assert(fn::lift(add)(quote(err(failure::stale)), quote(err(failure::crossed))).unwrap_err_unchecked() ==
              failure::stale);   // the first error, left to right
static_assert(!std::is_invocable_v<decltype(fn::lift(add)), std::optional<int>, quote>);           // one kind
static_assert(!std::is_invocable_v<decltype(fn::lift(add)), quote, result<int, int>>);             // one error type
static_assert(!std::is_invocable_v<decltype(fn::lift(add))>);

// ---- sequence and traverse ---------------------------------------------------------------------------------------

static_assert(*fn::sequence(std::tuple{std::optional<int>{1}, std::optional<double>{2.5}}) == std::tuple{1, 2.5});
static_assert(!fn::sequence(std::tuple{std::optional<int>{1}, std::optional<double>{}}).has_value());
static_assert(*fn::sequence(std::array{std::optional<int>{1}, std::optional<int>{2}}) == std::array{1, 2});   // an array stays an array
static_assert(fn::sequence(std::pair{quote(ok(1)), result<char, failure>(ok('a'))}).unwrap_unchecked() == std::tuple{1, 'a'});
static_assert(fn::sequence(std::tuple{quote(ok(1)), quote(err(failure::stale))}).unwrap_err_unchecked() == failure::stale);
static_assert(*fn::traverse(half, std::tuple{2, 4}) == std::tuple{1, 2});
static_assert(*fn::traverse(half, std::array{2, 4}) == std::array{1, 2});
static_assert(*fn::map_m(half, std::array{2, 4}) == std::array{1, 2} && *fn::for_m(std::array{2, 4}, half) == std::array{1, 2});
static_assert(!fn::traverse(half, std::array{2, 3}).has_value());
static_assert(fn::traverse(checked_half, std::array{4, 8}).unwrap_unchecked() == std::array{2, 4});
static_assert(!std::is_invocable_v<decltype(fn::sequence), std::tuple<>>);

// ---- the Control.Monad and Data.Maybe helpers ----------------------------------------------------------------------

constexpr auto safe_div = [](int acc, int x) { return x == 0 ? std::optional<int>{} : std::optional<int>{acc / x}; };
static_assert(fn::fold_m(safe_div, 100, std::array{2, 5}) == std::optional<int>{10});
static_assert(!fn::fold_m(safe_div, 100, std::array{2, 0, 5}).has_value());     // stops at the first failure
static_assert(fn::fold_m([](int acc, int x) { return quote(ok(acc + x)); }, 0, std::array{1, 2, 3}).unwrap_unchecked() == 6);
static_assert(*fn::zip_with_m([](int a, int b) { return std::optional<int>{a * b}; }, std::array{1, 2}, std::array{3, 4}) ==
              std::array{3, 8});
static_assert(*fn::replicate_m<3>(std::optional<int>{7}) == std::array{7, 7, 7});
static_assert(!fn::replicate_m<2>(std::optional<int>{}).has_value());
static_assert(fn::when_m(false, std::optional<std::monostate>{}).has_value());    // pure(unit)
static_assert(!fn::when_m(true, std::optional<std::monostate>{}).has_value());
static_assert(!fn::unless_m(false, std::optional<std::monostate>{}).has_value());
static_assert(fn::maybe(0, dbl, std::optional<int>{4}) == 8 && fn::maybe(0, dbl, std::optional<int>{}) == 0);
static_assert(fn::either([](failure) { return -1; }, dbl, quote(ok(4))) == 8);
static_assert(fn::either([](failure) { return -1; }, dbl, quote(err(failure::stale))) == -1);
static_assert(fn::from_maybe(5, std::optional<int>{}) == 5 && fn::from_maybe(5, std::optional<int>{1}) == 1);
static_assert(fn::map_maybe(half, std::array{1, 2, 3, 4}).size() == 2 && fn::map_maybe(half, std::array{1, 2, 3, 4})[1] == 2);
static_assert(fn::cat_maybes(std::array{std::optional<int>{1}, std::optional<int>{}, std::optional<int>{3}}).size() == 2);
static_assert(fn::sequence_a(std::pair{std::optional<int>{1}, std::optional<char>{'x'}}) == std::optional{std::tuple{1, 'x'}});

// ---- noexcept ------------------------------------------------------------------------------------------------------

static_assert(noexcept(fn::fmap(dbl, std::optional<int>{})));
static_assert(!noexcept(fn::fmap([](int x) { return x; }, std::optional<int>{})));
static_assert(noexcept(fn::lift(dbl)(std::optional<int>{})));
static_assert(!noexcept(fn::lift(add)(std::optional<int>{}, std::optional<int>{})));

TEST(FunctionalMonad, AnRvalueOptionalIsMovedIntoTheFunction)
{
    std::optional<std::unique_ptr<int>> p(std::make_unique<int>(3));
    auto r = fn::fmap([](std::unique_ptr<int> q) { return *q * 2; }, std::move(p));
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 6);
}

TEST(FunctionalMonad, AnLvalueOptionalIsSeenByConstReferenceAndKept)
{
    const std::optional<std::string> name("EURUSD");
    const auto size = fn::fmap([](const std::string& s) { return s.size(); }, name);
    EXPECT_EQ(size, std::optional<std::size_t>(6));
    EXPECT_EQ(*name, "EURUSD");
}

TEST(FunctionalMonad, LiftEvaluatesNothingWhenAnArgumentIsMissing)
{
    int calls = 0;
    auto counted = fn::lift([&](int a, int b) { ++calls; return a + b; });
    EXPECT_FALSE(counted(std::optional<int>{1}, std::optional<int>{}).has_value());
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(counted(std::optional<int>{1}, std::optional<int>{2}), std::optional<int>(3));
    EXPECT_EQ(calls, 1);
}

TEST(FunctionalMonad, ResultsCarryAStringErrorThrough)
{
    using parsed = result<int, std::string>;
    const auto both = fn::lift(add)(parsed(ok(1)), parsed(err(std::string("bad bid"))));
    ASSERT_TRUE(both.is_err());
    EXPECT_EQ(both.unwrap_err_unchecked(), "bad bid");
}

TEST(FunctionalMonad, TraverseCallsEveryElement)
{
    int calls = 0;
    const auto r = fn::traverse([&](int x) { ++calls; return half(x); }, std::tuple{3, 4, 6});
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(calls, 3);
}

} // namespace
