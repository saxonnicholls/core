//
//  function_ref_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/function_ref.hpp>.
//

#include <string>
#include <string_view>
#include <type_traits>

#include <gtest/gtest.h>

#include <snicholls/core/function_ref.hpp>

namespace {

using snicholls::function_ref;

static_assert(std::is_trivially_copyable_v<function_ref<void()>>);
static_assert(sizeof(function_ref<int(int)>) == 2 * sizeof(void*));
static_assert(!std::is_constructible_v<function_ref<int(int)>, int>);
static_assert(!std::is_constructible_v<function_ref<void(std::string)>, int (*)(int)>);

int call_twice(function_ref<int(int)> f) { return f(f(1)); }

TEST(FunctionRef, CallsTheReferencedLambdaWithoutCopyingIt)
{
    int calls = 0;
    auto add = [&](int x) {
        ++calls;
        return x + 10;
    };
    EXPECT_EQ(call_twice(add), 21);
    EXPECT_EQ(calls, 2);
}

TEST(FunctionRef, MutableStateIsSharedWithTheOriginal)
{
    struct counter
    {
        int n = 0;
        void operator()() { ++n; }
    } c;
    const function_ref<void()> f = c;
    f();
    f();
    EXPECT_EQ(c.n, 2);
}

TEST(FunctionRef, VoidSignatureDiscardsResultsAndConstCallablesWork)
{
    const auto returns_int = [](std::string_view s) { return static_cast<int>(s.size()); };
    const function_ref<void(std::string_view)> f = returns_int;
    f("abc");
    const function_ref<long(std::string_view)> g = returns_int;   // converting result
    EXPECT_EQ(g("abcd"), 4L);
}

TEST(FunctionRef, ForwardsArgumentsAndExceptions)
{
    std::string moved_into;
    const auto take = [&](std::string s) { moved_into = std::move(s); };
    const function_ref<void(std::string)> f = take;
    f(std::string(100, 'x'));
    EXPECT_EQ(moved_into.size(), 100u);
    const auto thrower = [](int) -> int { throw 7; };
    EXPECT_THROW(call_twice(thrower), int);
}

TEST(FunctionRef, CopiesReferToTheSameCallable)
{
    int n = 0;
    auto inc = [&] { ++n; };
    const function_ref<void()> a = inc;
    const function_ref<void()> b = a;   // copy, not a reference to a
    b();
    EXPECT_EQ(n, 1);
}

} // namespace
