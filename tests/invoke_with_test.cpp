//
//  invoke_with_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/invoke_with.hpp>: invoke_any over any_source, invoke_with over a
//  source of the test's own, and the typed result.
//

#include <snicholls/core/invoke_with.hpp>

#include <gtest/gtest.h>

#include <any>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

const auto handlers = snicholls::overloaded{
    [](int x) { return 2 * x; },
    [](double d) { return d / 2; },
    [](const std::string& s) { return s + s; },
    [](int a, int b) { return a + b; },
};

using outcome = snicholls::result<std::any, snicholls::overload_mismatch>;
static_assert(std::is_same_v<decltype(snicholls::invoke_any(handlers, {1})), outcome>);
static_assert(std::is_same_v<decltype(snicholls::invoke_any(handlers, std::span<const std::any>{})), outcome>);
static_assert(std::is_same_v<decltype(snicholls::invoke_any<double>(handlers, {1})),
                             snicholls::result<double, snicholls::overload_mismatch>>);

// The source concept: any_source is one; a plain container is not (no get<T>(i))
static_assert(snicholls::argument_source<snicholls::any_source>);
static_assert(!snicholls::argument_source<std::vector<std::any>>);
static_assert(!snicholls::argument_source<int>);

TEST(InvokeAnyMatch, CallsTheCallableWhoseParameterTypesTheArgumentsHold)
{
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(handlers, {21}).unwrap()), 42);
    EXPECT_EQ(std::any_cast<double>(snicholls::invoke_any(handlers, {3.0}).unwrap()), 1.5);
    EXPECT_EQ(std::any_cast<std::string>(snicholls::invoke_any(handlers, {std::string{"ab"}}).unwrap()), "abab");
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(handlers, {1, 2}).unwrap()), 3);
}

TEST(InvokeAnyMatch, DynamicTypeMustBeExactlyTheParameterType)
{
    // A short is not an int, and a string literal is a const char*, not a std::string.
    EXPECT_EQ(snicholls::invoke_any(handlers, {short{1}}).unwrap_err(), snicholls::overload_mismatch::types);
    EXPECT_EQ(snicholls::invoke_any(handlers, {"ab"}).unwrap_err(), snicholls::overload_mismatch::types);
    EXPECT_EQ(snicholls::invoke_any(handlers, {1, 2.0}).unwrap_err(), snicholls::overload_mismatch::types);
}

TEST(InvokeAnyMatch, ArityIsCheckedBeforeTypes)
{
    EXPECT_EQ(snicholls::invoke_any(handlers, {1, 2, 3}).unwrap_err(), snicholls::overload_mismatch::arity);
    EXPECT_EQ(snicholls::invoke_any(handlers, {}).unwrap_err(), snicholls::overload_mismatch::arity);
}

TEST(InvokeAnyMatch, FirstWrittenWinsAmongCallablesWithTheSameParameterTypes)
{
    const auto f = snicholls::overloaded{
        [](int) { return 1; },
        [](const int&) { return 2; },   // the same once references are stripped
    };
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(f, {0}).unwrap()), 1);
}

TEST(InvokeAnyMatch, NullaryCallableTakesAnEmptyArgumentList)
{
    const auto f = snicholls::overloaded{[] { return 7; }, [](int) { return 0; }};
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(f, {}).unwrap()), 7);
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(f, std::vector<std::any>{}).unwrap()), 7);
}

TEST(InvokeAnyResult, VoidComesBackAsAnEmptyAny)
{
    int hits = 0;
    const auto f = snicholls::overloaded{[&hits](int x) { hits += x; }};
    const auto r = snicholls::invoke_any(f, {5});
    ASSERT_TRUE(r.is_ok());
    EXPECT_FALSE(r.unwrap().has_value());
    EXPECT_EQ(hits, 5);
}

TEST(InvokeAnyResult, TypedResultConvertsWhatEveryCallableReturns)
{
    // R = double: int and double results both convert; the string one is never reached here.
    const auto numeric = snicholls::overloaded{[](int x) { return 2 * x; }, [](double d) { return d / 2; }};
    EXPECT_EQ(snicholls::invoke_any<double>(numeric, {21}).unwrap(), 42.0);
    EXPECT_EQ(snicholls::invoke_any<double>(numeric, {3.0}).unwrap(), 1.5);
    EXPECT_EQ(snicholls::invoke_any<long>(numeric, {21}).unwrap(), 42L);

    // A void callable gives R{}.
    const auto quiet = snicholls::overloaded{[](int) {}};
    EXPECT_EQ(snicholls::invoke_any<int>(quiet, {1}).unwrap(), 0);
    EXPECT_EQ(snicholls::invoke_any<std::string>(quiet, {1}).unwrap(), "");
}

TEST(InvokeAnyResult, ErrorTextNamesTheMismatch)
{
    EXPECT_EQ(snicholls::error_text(snicholls::overload_mismatch::arity), "no overload takes this many arguments");
    EXPECT_EQ(snicholls::error_text(snicholls::overload_mismatch::types), "no overload takes arguments of these types");
    try {
        (void)snicholls::invoke_any(handlers, {1, 2, 3}).unwrap();
        FAIL() << "unwrap of a miss must throw";
    } catch (const snicholls::bad_result_access<snicholls::overload_mismatch>& e) {
        EXPECT_EQ(e.error(), snicholls::overload_mismatch::arity);
        EXPECT_NE(std::string{e.what()}.find("no overload takes this many arguments"), std::string::npos);
    }
}

TEST(InvokeAnyArguments, ConstReferenceParametersBindTheObjectInsideTheAny)
{
    const auto f = snicholls::overloaded{[](const std::string& s) { return &s; }};
    const std::vector<std::any> args{std::string{"x"}};
    const auto* inside = std::any_cast<std::string>(&args[0]);
    EXPECT_EQ(std::any_cast<const std::string*>(snicholls::invoke_any(f, args).unwrap()), inside);
}

TEST(InvokeAnyArguments, RvalueReferenceParametersGetACopyAndTheArgumentsStayIntact)
{
    const auto f = snicholls::overloaded{[](std::string&& s) { std::string taken = std::move(s); return taken; }};
    const std::vector<std::any> args{std::string{"keep"}};
    EXPECT_EQ(std::any_cast<std::string>(snicholls::invoke_any(f, args).unwrap()), "keep");
    EXPECT_EQ(std::any_cast<const std::string&>(args[0]), "keep");
}

TEST(InvokeAnyArguments, EveryContiguousRangeOfAnyIsAccepted)
{
    std::vector<std::any> vec{1, 2};
    const std::array<std::any, 2> arr{3, 4};
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(handlers, vec).unwrap()), 3);
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(handlers, arr).unwrap()), 7);
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(handlers, std::span<const std::any>{vec}.first(1)).unwrap()), 2);
}

int free_twice(int n) { return 2 * n; }

TEST(InvokeAnySet, ASingleCallableIsASetOfOne)
{
    const auto lambda = [](int n) { return n + 1; };
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(lambda, {1}).unwrap()), 2);
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(&free_twice, {2}).unwrap()), 4);
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(free_twice, {3}).unwrap()), 6);
    const std::function<int(int)> fn = free_twice;
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(fn, {4}).unwrap()), 8);
    EXPECT_EQ(snicholls::invoke_any(lambda, {2.0}).unwrap_err(), snicholls::overload_mismatch::types);
}

TEST(InvokeAnySet, FunctionPointerElementsAndNestedSetsAreRead)
{
    const auto f = snicholls::overloaded{
        snicholls::overloaded{free_twice, [](double d) { return -d; }},
        snicholls::first_match{[](const std::string& s) { return s.size(); }},
    };
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(f, {5}).unwrap()), 10);
    EXPECT_EQ(std::any_cast<double>(snicholls::invoke_any(f, {1.5}).unwrap()), -1.5);
    EXPECT_EQ(std::any_cast<std::size_t>(snicholls::invoke_any(f, {std::string{"abc"}}).unwrap()), 3u);
}

TEST(InvokeAnySet, FirstMatchWithConcreteCallablesDispatchesTheSameWay)
{
    const auto f = snicholls::first_match{[](int) { return 1; }, [](double) { return 2; }};
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(f, {0}).unwrap()), 1);
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(f, {0.0}).unwrap()), 2);
    EXPECT_EQ(f(0.0), 1);   // the compile-time call converts; the run-time one matches the exact type
}

TEST(InvokeAnySet, RvalueSetAndMutableState)
{
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(snicholls::overloaded{[](int x) { return -x; }}, {1}).unwrap()), -1);

    auto counter = snicholls::overloaded{[n = 0](int) mutable { return ++n; }};
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(counter, {0}).unwrap()), 1);
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(counter, {0}).unwrap()), 2);
}

TEST(InvokeAnySet, ExceptionsFromTheCallablePropagate)
{
    const auto f = snicholls::overloaded{
        [](int) -> int { throw std::runtime_error{"boom"}; },
        [](double) { return 0; },
    };
    EXPECT_THROW((void)snicholls::invoke_any(f, {1}), std::runtime_error);
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(f, {1.0}).unwrap()), 0);
}

TEST(InvokeAnySet, AResultThatIsItselfAnAnyIsReturnedAsIs)
{
    const auto f = snicholls::overloaded{[](int x) { return std::any{x * 3}; }};
    EXPECT_EQ(std::any_cast<int>(snicholls::invoke_any(f, {2}).unwrap()), 6);
}

// ---- a source of the test's own: values, not pointers -------------------------------------------

// Argument i is the integer i (as any integral type) or the string "i"; anything else is refused.
struct counting_source {
    std::size_t count;

    [[nodiscard]] std::size_t size() const noexcept { return count; }

    template <typename T>
    [[nodiscard]] std::optional<T> get(std::size_t i) const
    {
        return snicholls::first_match{
            [i](std::integral auto* tag) { return std::optional<std::remove_pointer_t<decltype(tag)>>{static_cast<std::remove_pointer_t<decltype(tag)>>(i)}; },
            [i](std::string*) { return std::optional<std::string>{std::to_string(i)}; },
            [](auto* tag) { return std::optional<std::remove_pointer_t<decltype(tag)>>{}; },
        }(static_cast<T*>(nullptr));
    }
};
static_assert(snicholls::argument_source<counting_source>);

TEST(InvokeWithSource, AValueSourceFillsParametersItAgreesTo)
{
    const auto f = snicholls::overloaded{
        [](int a, long b, std::string c) { return std::to_string(a) + "," + std::to_string(b) + "," + c; },
        [](double) { return std::string{"never: the source refuses doubles"}; },
        [](unsigned char only) { return std::string{"byte "} + std::to_string(only); },
    };
    EXPECT_EQ(std::any_cast<std::string>(snicholls::invoke_with(f, counting_source{3}).unwrap()), "0,1,2");
    EXPECT_EQ(std::any_cast<std::string>(snicholls::invoke_with(f, counting_source{1}).unwrap()), "byte 0");
    EXPECT_EQ(snicholls::invoke_with(f, counting_source{2}).unwrap_err(), snicholls::overload_mismatch::arity);

    const auto doubles = snicholls::overloaded{[](double) { return 1; }};
    EXPECT_EQ(snicholls::invoke_with(doubles, counting_source{1}).unwrap_err(), snicholls::overload_mismatch::types);
}

TEST(InvokeWithSource, OwnedValuesAreMovedIntoByValueAndRvalueParameters)
{
    const auto f = snicholls::overloaded{
        [](std::string&& s) { return "moved " + s; },
    };
    EXPECT_EQ(std::any_cast<std::string>(snicholls::invoke_with(f, counting_source{1}).unwrap()), "moved 0");
    const auto g = snicholls::overloaded{[](const std::string& s) { return "borrowed " + s; }};
    EXPECT_EQ(std::any_cast<std::string>(snicholls::invoke_with(g, counting_source{1}).unwrap()), "borrowed 0");
    const auto h = snicholls::overloaded{[](std::string s) { return "owned " + s; }};
    EXPECT_EQ(std::any_cast<std::string>(snicholls::invoke_with(h, counting_source{1}).unwrap()), "owned 0");
}

TEST(InvokeWithSource, TypedResultWorksWithAnySource)
{
    const auto f = snicholls::overloaded{[](int a, int b) { return a + b; }};
    EXPECT_EQ(snicholls::invoke_with<long>(f, counting_source{2}).unwrap(), 1L);
}

} // namespace
