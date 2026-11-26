//
//  lazy_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/lazy.hpp>: the thunk and the stream operations, at compile time.
//

#include <snicholls/core/functional/lazy.hpp>

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
namespace fn = snicholls::fn; namespace st = snicholls::fn::stream;
constexpr auto odd3 = st::filter([](long x) { return x % 2 == 1; }, st::iterate([](long x) { return x * 3; }, 1L));
static_assert(st::take<4>(odd3) == std::array{1L, 3L, 9L, 27L});
constexpr auto nat = st::iterate([](int x) { return x + 1; }, 0);
static_assert(st::take<3>(st::map([](int x) { return x * x; }, nat)) == std::array{0, 1, 4});
static_assert(st::take<3>(st::zip_with([](int a, int b) { return a + b; }, nat, st::repeat(10))) == std::array{10, 11, 12});
static_assert(st::take<4>(st::scanl([](int a, int b) { return a + b; }, 0, nat)) == std::array{0, 0, 1, 3});
static_assert(st::take<5>(st::cycle(std::array{1, 2})) == std::array{1, 2, 1, 2, 1});
static_assert(st::head(st::drop(5, nat)) == 5 && st::nth(nat, 7) == 7 && st::head(st::tail(nat)) == 1);
static_assert(st::head(st::drop_while([](int x) { return x < 4; }, nat)) == 4);
static_assert(st::take_while<10>([](int x) { return x < 3; }, nat).size() == 3);
static_assert(st::take<2>(st::zip(nat, nat))[1] == std::pair{1, 1});
constexpr int forced() { int calls = 0; fn::lazy l{[&] { ++calls; return 42; }}; int a = l.force(); int b = l.force(); return a + b + calls; }
static_assert(forced() == 85);
static_assert(std::is_trivially_copyable_v<decltype(nat)>);

TEST(FunctionalLazy, AThunkRunsOnceAndOnlyWhenForced)
{
    int calls = 0;
    const fn::lazy value{[&] { ++calls; return 7; }};
    EXPECT_FALSE(value.forced());
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(value.force() + value.force(), 14);
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(value.forced());
}

TEST(FunctionalLazy, AThrowingThunkKeepsNothing)
{
    int calls = 0;
    const fn::lazy bad{[&]() -> int { if (++calls == 1) throw std::runtime_error("feed down"); return 3; }};
    EXPECT_THROW(static_cast<void>(bad.force()), std::runtime_error);
    EXPECT_FALSE(bad.forced());
    EXPECT_EQ(bad.force(), 3);
}

TEST(FunctionalLazy, TakingFromAStreamLeavesItUnchanged)
{
    const auto evens = st::filter([](int x) { return x % 2 == 0; }, nat);
    EXPECT_EQ(st::take<3>(evens), (std::array{0, 2, 4}));
    EXPECT_EQ(st::take<3>(evens), (std::array{0, 2, 4}));
}

} // namespace
