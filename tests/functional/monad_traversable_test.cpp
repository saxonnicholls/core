//
//  monad_traversable_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Edge cases and laws for <snicholls/core/functional/monad.hpp>: empty and single traversals, the Traversable laws
//  (identity, naturality, traverse == sequence . fmap), fold_m and zip_with_m short-circuiting, the lifts of every
//  arity, and the eliminators.
//

#include <snicholls/core/functional/monad.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

namespace fn = snicholls::fn;
using snicholls::err;
using snicholls::ok;
using snicholls::result;

enum class bad { zero, odd, negative };
using res = result<int, bad>;

constexpr auto half = [](int x) { return x % 2 == 0 ? std::optional<int>(x / 2) : std::nullopt; };
constexpr auto checked_half = [](int x) { return x % 2 == 0 ? res(ok(x / 2)) : res(err(bad::odd)); };

// ---- empty and single ------------------------------------------------------------------------------------------------

// An empty sequence has no carrier to name, so it is not sequenced (as an empty tuple is not).
static_assert(!std::is_invocable_v<decltype(fn::sequence), std::array<std::optional<int>, 0>>);
static_assert(!std::is_invocable_v<decltype(fn::traverse), decltype(half), std::array<int, 0>>);
static_assert(fn::traverse(half, std::array{8}) == std::optional<std::array<int, 1>>(std::array{4}));
static_assert(fn::sequence(std::tuple{std::optional<int>{1}}) == std::optional<std::tuple<int>>(std::tuple{1}));
static_assert(fn::fold_m([](int, int) { return std::optional<int>(); }, 7, std::array<int, 0>{}) == std::optional<int>(7));
static_assert(fn::replicate_m<1>(res(ok(3))).unwrap_unchecked() == std::array{3});
static_assert(fn::cat_maybes(std::array<std::optional<int>, 3>{}).empty());
static_assert(fn::map_maybe(half, std::array<int, 0>{}).empty());
static_assert(fn::map_maybe(half, std::array{2, 4, 6}).size() == 3);

// ---- the Traversable laws ------------------------------------------------------------------------------------------

// Identity: traverse pure == pure.
static_assert(fn::traverse([](int x) { return std::optional<int>(x); }, std::array{1, 2, 3}) ==
              std::optional<std::array<int, 3>>(std::array{1, 2, 3}));
// traverse f == sequence . fmap f.
static_assert(fn::traverse(half, std::array{2, 4, 6}) == fn::sequence(fn::fmap(half, std::array{2, 4, 6})));
static_assert(fn::traverse(half, std::array{2, 3, 6}) == fn::sequence(fn::fmap(half, std::array{2, 3, 6})));
// Naturality, with t = an optional to a result (ok or a fixed error): t . traverse f == traverse (t . f).
constexpr auto to_res = [](const auto& o) {
    using A = typename std::remove_cvref_t<decltype(o)>::value_type;
    return o.has_value() ? result<A, bad>(ok(*o)) : result<A, bad>(err(bad::odd));
};
static_assert(to_res(fn::traverse(half, std::array{2, 4})).unwrap_unchecked() ==
              fn::traverse(fn::compose(to_res, half), std::array{2, 4}).unwrap_unchecked());
static_assert(to_res(fn::traverse(half, std::array{2, 5})).unwrap_err_unchecked() ==
              fn::traverse(fn::compose(to_res, half), std::array{2, 5}).unwrap_err_unchecked());
// sequence . sequence over a tuple of arrays leaves the values where they were.
static_assert(fn::sequence(std::pair{res(ok(1)), res(ok(2))}).unwrap_unchecked() == std::tuple{1, 2});

// ---- short-circuiting -------------------------------------------------------------------------------------------------

constexpr auto safe_div = [](int acc, int x) { return x == 0 ? res(err(bad::zero)) : res(ok(acc / x)); };
static_assert(fn::fold_m(safe_div, 1000, std::array{10, 5, 2}).unwrap_unchecked() == 10);
static_assert(fn::fold_m(safe_div, 1000, std::array{10, 0, 2}).unwrap_err_unchecked() == bad::zero);
static_assert(fn::zip_with_m([](int a, int b) { return b == 0 ? std::optional<int>() : std::optional<int>(a / b); },
                             std::array{4, 9}, std::array{2, 0}) == std::nullopt);
static_assert(fn::zip_with_m([](int a, int b) { return res(ok(a - b)); }, std::array{4, 9}, std::array{1, 2}).unwrap_unchecked() ==
              std::array{3, 7});
static_assert(fn::traverse(checked_half, std::tuple{4, 3, 5}).unwrap_err_unchecked() == bad::odd);   // the first error
using unit_res = result<std::monostate, bad>;
static_assert(fn::when_m(true, unit_res(err(bad::negative))).unwrap_err_unchecked() == bad::negative);
static_assert(fn::when_m(false, unit_res(err(bad::negative))).is_ok());
static_assert(fn::unless_m(true, unit_res(err(bad::negative))).is_ok());

// ---- lift of every arity, and the eliminators ----------------------------------------------------------------------

constexpr auto mid3 = [](int a, int b, int c) { return a + b + c; };
static_assert(fn::lift(mid3)(std::optional<int>{1}, std::optional<int>{2}, std::optional<int>{3}) == std::optional<int>{6});
static_assert(fn::lift(mid3)(std::optional<int>{1}, std::optional<int>{}, std::optional<int>{3}) == std::nullopt);
static_assert(fn::lift(mid3)(res(ok(1)), res(err(bad::odd)), res(err(bad::zero))).unwrap_err_unchecked() == bad::odd);
static_assert(fn::lift(mid3)(res(ok(1)), res(ok(2)), res(err(bad::zero))).unwrap_err_unchecked() == bad::zero);
static_assert(fn::lift(mid3)(res(ok(1)), res(ok(2)), res(ok(3))).unwrap_unchecked() == 6);
static_assert(fn::traverse(checked_half, std::array{4, 6, 8, 10}).unwrap_unchecked() == std::array{2, 3, 4, 5});
static_assert(fn::traverse(checked_half, std::array{4, 6, 7, 10}).unwrap_err_unchecked() == bad::odd);
static_assert(fn::maybe(std::string_view("none"), [](int) { return std::string_view("some"); }, std::optional<int>{1}) == "some");
static_assert(fn::either([](bad b) { return static_cast<int>(b); }, [](int v) { return v; }, res(err(bad::negative))) == 2);
static_assert(noexcept(fn::maybe(0, [](int x) noexcept { return x; }, std::optional<int>{})));
static_assert(fn::from_maybe(0, fn::mbind(std::optional<int>{8}, half)) == 4);
static_assert(fn::value_or(fn::mbind(res(ok(3)), checked_half), -1) == -1);

TEST(FunctionalMonadTraversable, FoldMStopsCallingAfterTheFirstFailure)
{
    int calls = 0;
    const auto r = fn::fold_m([&](int acc, int x) { ++calls; return x < 0 ? res(err(bad::negative)) : res(ok(acc + x)); }, 0,
                              std::vector<int>{1, -1, 2, 3});
    EXPECT_TRUE(r.is_err());
    EXPECT_EQ(calls, 2);
}

TEST(FunctionalMonadTraversable, SequenceMovesStringsOutOfATemporaryTuple)
{
    const auto r = fn::sequence(std::tuple{std::optional<std::string>("bid"), std::optional<std::string>("ask")});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(std::get<0>(*r) + "/" + std::get<1>(*r), "bid/ask");
}

TEST(FunctionalMonadTraversable, CatMaybesKeepsOrder)
{
    const auto v = fn::cat_maybes(std::array{std::optional<int>{3}, std::optional<int>{}, std::optional<int>{1}});
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0], 3);
    EXPECT_EQ(v[1], 1);
}

} // namespace
