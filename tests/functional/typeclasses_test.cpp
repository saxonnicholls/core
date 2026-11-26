//
//  typeclasses_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/typeclasses.hpp>: the dispatch, the concepts, the monoid wrappers, the
//  profunctor functions, and an instance of a type of the test's own. The laws are in typeclasses_laws_test.cpp.
//

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/typeclasses.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

namespace {

namespace fn = snicholls::fn;

constexpr auto dbl = [](int x) noexcept { return 2 * x; };
constexpr auto inc = [](int x) { return x + 1; };

// A carrier of the test's own: a value with a count of how many steps made it.
template <class T>
struct counted {
    T value;
    int steps;
    friend constexpr bool operator==(const counted&, const counted&) = default;
};

} // namespace

template <class T>
struct snicholls::fn::functor_instance<counted<T>> {
    static constexpr bool defined = true;
    template <class F, class M>
    static constexpr auto fmap(F&& f, M&& m)
    {
        return counted<std::remove_cvref_t<std::invoke_result_t<F, T>>>{std::invoke(f, m.value), m.steps + 1};
    }
};

namespace {

// ---- the concepts ---------------------------------------------------------------------------------------------------

static_assert(fn::functor<std::optional<int>> && fn::applicative<std::optional<int>> && fn::monad<std::optional<int>>);
static_assert(fn::functor<counted<int>> && !fn::applicative<counted<int>>);
static_assert(!fn::functor<int> && !fn::semigroup<int> && !fn::monoid<double>);   // a number's monoid is chosen by a wrapper
static_assert(fn::monoid<fn::sum_of<int>> && fn::monoid<std::string> && fn::semigroup<fn::min_of<std::string>>);
static_assert(!fn::monoid<fn::min_of<std::string>>);   // no largest string
static_assert(fn::alternative<std::optional<int>> && fn::bifunctor<std::pair<int, int>>);

// ---- the dispatch -----------------------------------------------------------------------------------------------------

static_assert(fn::fmap(dbl, counted<int>{3, 0}) == counted<int>{6, 1});
static_assert(fn::fmap(inc, fn::fmap(dbl, counted<int>{3, 0})) == counted<int>{7, 2});
static_assert(fn::replace('x', std::optional<int>{1}) == std::optional<char>{'x'});
static_assert(fn::pure<std::optional<int>>(3) == std::optional<int>{3});
static_assert(fn::mreturn<std::optional<double>>(3) == std::optional<int>{3});
static_assert(fn::lift_a2([](int a, int b) { return a - b; }, std::optional<int>{5}, std::optional<int>{3}) == std::optional<int>{2});
static_assert(fn::mthen(std::optional<int>{1}, std::optional<char>{'a'}) == std::optional<char>{'a'});
static_assert(!fn::mthen(std::optional<int>{}, std::optional<char>{'a'}).has_value());
constexpr auto half = [](int x) { return x % 2 ? std::optional<int>{} : std::optional<int>{x / 2}; };
static_assert(fn::kleisli(half, half)(8) == std::optional<int>{2});
static_assert(fn::kleisli_left(half, half)(8) == std::optional<int>{2});
static_assert(!fn::kleisli(half, half)(6).has_value());
static_assert(fn::and_then(half, std::optional<int>{4}) == std::optional<int>{2});
static_assert(fn::mzero<std::optional<int>> == std::nullopt);
static_assert(!std::is_invocable_v<decltype(fn::fmap), decltype(dbl), int>);    // no instance: not invocable
static_assert(noexcept(fn::fmap(dbl, std::optional<int>{})));
static_assert(!noexcept(fn::fmap(inc, std::optional<int>{})));

// ---- the monoid wrappers ----------------------------------------------------------------------------------------------

static_assert(fn::combine(fn::sum_of<int>{2}, fn::sum_of<int>{3}) == fn::sum_of<int>{5});
static_assert(fn::sconcat(fn::product_of<int>{2}, fn::product_of<int>{3}, fn::product_of<int>{4}) == fn::product_of<int>{24});
static_assert(fn::mconcat(std::array{fn::min_of<int>{4}, fn::min_of<int>{2}}) == fn::min_of<int>{2});
static_assert(fn::mconcat(std::array<fn::max_of<int>, 0>{}) == fn::max_of<int>{std::numeric_limits<int>::lowest()});
static_assert(fn::mconcat(std::array{fn::first_of<int>{}, fn::first_of<int>{2}, fn::first_of<int>{3}}) == fn::first_of<int>{2});
static_assert(fn::mconcat(std::array{fn::last_of<int>{1}, fn::last_of<int>{3}, fn::last_of<int>{}}) == fn::last_of<int>{3});
static_assert(fn::mconcat(std::array{fn::all_of{true}, fn::all_of{false}}) == fn::all_of{false});
static_assert(fn::mconcat(std::array{fn::any_of{false}, fn::any_of{true}}) == fn::any_of{true});
static_assert(fn::mempty<fn::all_of>() == fn::all_of{true});
static_assert(fn::combine(fn::endo{dbl}, fn::endo{inc})(3) == 8);                 // dbl . inc
static_assert(fn::combine(fn::mempty<fn::endo<fn::detail::identity_fn>>(), fn::endo{inc})(3) == 4);

// ---- bifunctor and profunctor -----------------------------------------------------------------------------------------

static_assert(fn::bimap(dbl, inc, std::pair{1, 2}) == std::pair{2, 3});
static_assert(fn::first(dbl, std::pair{1, 2}) == std::pair{2, 2});
static_assert(fn::second(dbl, std::tuple{1, 2}) == std::tuple{1, 4});
static_assert(fn::dimap(inc, dbl, dbl)(1) == 8);                                  // dbl(dbl(inc(1)))
static_assert(fn::lmap(inc, dbl)(1) == 4 && fn::contramap(inc, dbl)(1) == 4);
static_assert(fn::rmap(inc, dbl)(1) == 3);

TEST(FunctionalTypeclasses, StringsAndVectorsConcatenate)
{
    EXPECT_EQ(fn::sconcat(std::string("bid"), std::string("/"), std::string("ask")), "bid/ask");
    EXPECT_EQ(fn::mconcat(std::array{std::string("a"), std::string("b")}), "ab");
    EXPECT_EQ(fn::combine(std::vector<int>{1}, std::vector<int>{2, 3}), (std::vector<int>{1, 2, 3}));
    EXPECT_TRUE(fn::mempty<std::string>().empty());
}

TEST(FunctionalTypeclasses, FirstOfKeepsTheEarliestPresentQuote)
{
    const auto best = fn::mconcat(std::array{fn::first_of<std::string>{}, fn::first_of<std::string>{"LMAX"},
                                             fn::first_of<std::string>{"EBS"}});
    ASSERT_TRUE(best.value.has_value());
    EXPECT_EQ(*best.value, "LMAX");
}

} // namespace
