//
//  result_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/result.hpp>.
//

// snicholls::result<V, E> (include/snicholls/core/result.hpp): the Rust API, layout and zero-overhead guarantees, noexcept and
// constexpr claims, move-only and void results, rethrow and try_invoke, SNICHOLLS_TRY, comparisons and hashing.
// The SNICHOLLS_RESULT_THROW=0 policy is tested in result_terminate_test.cpp; traced errors in traced_test.cpp.

#include <snicholls/core/result.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

enum class parse_error : std::uint8_t
{
    empty = 1,
    not_a_number = 2,
};

using int_result = snicholls::result<int, parse_error>;

template <class R>
concept iterable_result = requires(R&& r) { static_cast<R&&>(r).iter(); };

constexpr int_result parse_digit(char c) noexcept
{
    if (c == '\0')
        return snicholls::err(parse_error::empty);
    if (c < '0' || c > '9')
        return snicholls::err(parse_error::not_a_number);
    return snicholls::ok(c - '0');
}

// ---- hand-written equivalents (the zero-overhead yardstick) -----------------------------------------------------

// Only its size and alignment are used. MSVC warns (C4624) that a union with a non-trivial member deletes the
// destructor, which is the point of a hand-written union and harmless here.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(disable : 4624)
#endif
template <class V, class E>
struct hand_tagged
{
    union
    {
        V v;
        E e;
    };
    bool ok;
};

template <class E>
struct hand_void
{
    E e;
    bool ok;
};

// An index with an invalid value: the opt-in niche.
struct slot
{
    std::int32_t index;
    friend constexpr bool operator==(slot, slot) noexcept = default;
};

struct not_found
{
};

// Move-only, with an observable moved-from state.
struct handle
{
    std::unique_ptr<int> p;
    explicit handle(int v) : p(std::make_unique<int>(v)) {}
};

// Counts copies and moves.
struct counted
{
    static inline int copies = 0;
    static inline int moves = 0;
    int v = 0;
    explicit counted(int x) noexcept : v(x) {}
    counted(const counted& o) noexcept : v(o.v) { ++copies; }
    counted(counted&& o) noexcept : v(o.v) { ++moves; }
    counted& operator=(const counted&) = default;
    counted& operator=(counted&&) = default;
    static void reset() noexcept { copies = moves = 0; }
};

// A copy that throws, to test assignment's strong guarantee.
struct throws_on_copy
{
    int v = 0;
    explicit throws_on_copy(int x) : v(x) {}
    throws_on_copy(const throws_on_copy& o) : v(o.v)
    {
        if (o.v < 0)
            throw std::runtime_error("copy");
    }
    throws_on_copy(throws_on_copy&&) noexcept = default;
    throws_on_copy& operator=(const throws_on_copy&) = default;
    throws_on_copy& operator=(throws_on_copy&&) noexcept = default;
};

struct my_error
{
    int code = 0;
};

struct my_exception : std::runtime_error
{
    int code;
    explicit my_exception(int c) : std::runtime_error("my_exception"), code(c) {}
};

} // namespace

template <>
struct snicholls::result_niche<slot>
{
    static constexpr slot value{-1};
};

template <>
struct snicholls::error_exception<my_error>
{
    [[noreturn]] static void raise(const my_error& e) { throw my_exception(e.code); }
};

namespace {

// ---- layout: the zero-overhead contract --------------------------------------------------------------------------

static_assert(sizeof(snicholls::result<int, parse_error>) == sizeof(hand_tagged<int, parse_error>));
static_assert(alignof(snicholls::result<int, parse_error>) == alignof(hand_tagged<int, parse_error>));
static_assert(sizeof(snicholls::result<double, std::error_code>) == sizeof(hand_tagged<double, std::error_code>));
static_assert(sizeof(snicholls::result<std::string, int>) == sizeof(hand_tagged<std::string, int>));
static_assert(sizeof(snicholls::result<std::int64_t, std::int64_t>) == sizeof(hand_tagged<std::int64_t, std::int64_t>));
static_assert(sizeof(snicholls::result<void, parse_error>) == sizeof(hand_void<parse_error>));
static_assert(sizeof(snicholls::result<void, std::error_code>) == sizeof(hand_void<std::error_code>));
// The niche: an empty error costs no byte at all.
static_assert(sizeof(snicholls::result<slot, not_found>) == sizeof(slot));
static_assert(std::is_trivially_copyable_v<snicholls::result<slot, not_found>>);

// Trivial exactly when V and E are.
static_assert(std::is_trivially_copyable_v<snicholls::result<int, parse_error>>);
static_assert(std::is_trivially_destructible_v<snicholls::result<int, parse_error>>);
static_assert(std::is_trivially_copy_constructible_v<snicholls::result<int, parse_error>>);
static_assert(std::is_trivially_move_constructible_v<snicholls::result<int, parse_error>>);
static_assert(std::is_trivially_copy_assignable_v<snicholls::result<int, parse_error>>);
static_assert(std::is_trivially_move_assignable_v<snicholls::result<int, parse_error>>);
static_assert(std::is_trivially_copyable_v<snicholls::result<void, std::error_code>>);
static_assert(!std::is_trivially_destructible_v<snicholls::result<std::string, int>>);
static_assert(!std::is_trivially_copy_constructible_v<snicholls::result<int, std::string>>);
static_assert(!std::is_polymorphic_v<snicholls::result<int, parse_error>>);
static_assert(!std::is_default_constructible_v<snicholls::result<int, parse_error>>);   // Rust: no default

// Returned in registers like the hand-written struct, not through a hidden pointer. Clang derives
// __is_trivially_relocatable from "can be passed in registers"; declaring any non-trivial copy/move constructor
// (even one whose constraint fails) breaks it, which is why result.hpp layers its constructors.
#if defined(__clang__) && defined(__has_builtin)
#if __has_builtin(__is_trivially_relocatable)
#pragma clang diagnostic push
#if __has_warning("-Wdeprecated-builtins")
#pragma clang diagnostic ignored "-Wdeprecated-builtins"
#endif
static_assert(__is_trivially_relocatable(snicholls::result<int, parse_error>));
static_assert(__is_trivially_relocatable(snicholls::result<void, std::error_code>));
static_assert(__is_trivially_relocatable(snicholls::result<slot, not_found>));
#pragma clang diagnostic pop
#endif
#endif

// Move-only follows its members.
static_assert(!std::is_copy_constructible_v<snicholls::result<handle, int>>);
static_assert(std::is_nothrow_move_constructible_v<snicholls::result<handle, int>>);
static_assert(std::is_move_assignable_v<snicholls::result<handle, int>>);

// ---- noexcept is exact -------------------------------------------------------------------------------------------

static_assert(noexcept(std::declval<int_result&>().is_ok()));
static_assert(noexcept(std::declval<int_result&>().unwrap_or(0)));
static_assert(noexcept(std::declval<int_result&>().map([](int v) noexcept { return v + 1; })));
static_assert(!noexcept(std::declval<int_result&>().map([](int v) { return v + 1; })));
static_assert(noexcept(std::declval<int_result&>().map_err([](parse_error) noexcept { return 1; })));
static_assert(noexcept(std::declval<int_result&>().and_then([](int v) noexcept { return int_result(snicholls::ok(v)); })));
static_assert(noexcept(std::declval<int_result&>().or_else([](parse_error) noexcept { return int_result(snicholls::ok(0)); })));
static_assert(noexcept(std::declval<int_result&>().map_or(0, [](int v) noexcept { return v; })));
static_assert(!noexcept(std::declval<int_result&>().unwrap()));                 // throws bad_result_access
static_assert(!noexcept(std::declval<int_result&>().expect(std::declval<std::string_view>())));
static_assert(!noexcept(std::declval<snicholls::result<std::string, int>&>().unwrap_or_default()));   // copies a string
static_assert(noexcept(std::declval<snicholls::result<std::string, int>&&>().unwrap_or_default()));   // moves it
static_assert(noexcept(std::declval<snicholls::result<std::string, int>&>().ok()) == false);
static_assert(noexcept(std::declval<snicholls::result<std::string, int>&&>().ok()));
static_assert(noexcept(std::declval<int_result&>() == std::declval<int_result&>()));
static_assert(noexcept(std::hash<int_result>{}(std::declval<const int_result&>())));
static_assert(noexcept(snicholls::try_invoke([] { return 1; })));
static_assert(noexcept(snicholls::ok(1)) && !noexcept(snicholls::ok(std::declval<const std::string&>())));

// ---- constexpr -------------------------------------------------------------------------------------------------------

constexpr int constexpr_pipeline()
{
    int_result r = parse_digit('4');
    r = r.map([](int v) { return v * 10; }).and_then([](int v) { return int_result(snicholls::ok(v + 2)); });
    int_result e = parse_digit('x');
    e = snicholls::ok(1);   // error -> value in a constant expression
    e = snicholls::err(parse_error::empty);
    return r.unwrap_or(-1) + (e.is_err_and([](parse_error p) { return p == parse_error::empty; }) ? 100 : 0);
}
static_assert(constexpr_pipeline() == 142);
static_assert(parse_digit('7') == snicholls::ok(7));
static_assert(parse_digit('?') == snicholls::err(parse_error::not_a_number));
static_assert(parse_digit('1') < parse_digit('?'));   // Rust: every Ok orders before every Err

// ---- construction and observers ----------------------------------------------------------------------------------

TEST(ResultConstruction, OkAndErrTagsAndInPlace)
{
    int_result a = snicholls::ok(3);
    int_result b = snicholls::err(parse_error::empty);
    snicholls::result<std::string, int> c(std::in_place, 3, 'x');
    snicholls::result<std::string, std::string> d(snicholls::err_in_place, "bad");
    EXPECT_TRUE(a.is_ok());
    EXPECT_FALSE(a.is_err());
    EXPECT_TRUE(static_cast<bool>(a));
    EXPECT_TRUE(b.is_err());
    EXPECT_FALSE(static_cast<bool>(b));
    EXPECT_EQ(*c, "xxx");
    EXPECT_EQ(c->size(), 3u);
    EXPECT_EQ(d.unwrap_err(), "bad");
}

TEST(ResultConstruction, SameValueAndErrorTypeIsNeverAmbiguous)
{
    snicholls::result<std::string, std::string> ok_text = snicholls::ok(std::string("value"));
    snicholls::result<std::string, std::string> err_text = snicholls::err(std::string("error"));
    EXPECT_TRUE(ok_text.is_ok());
    EXPECT_TRUE(err_text.is_err());
}

TEST(ResultConstruction, ConvertsBetweenResultsOfConvertibleTypes)
{
    const int_result narrow = snicholls::ok(5);
    snicholls::result<long, parse_error> wide = narrow;
    EXPECT_EQ(*wide, 5L);
    snicholls::result<std::string, std::string> from_literals = snicholls::result<const char*, const char*>(snicholls::err("e"));
    EXPECT_EQ(from_literals.unwrap_err(), "e");
    snicholls::result<std::unique_ptr<const int>, int> moved = snicholls::result<std::unique_ptr<int>, int>(snicholls::ok(std::make_unique<int>(2)));
    EXPECT_EQ(**moved, 2);
}

TEST(ResultConstruction, PredicatesOnTheValueAndTheError)
{
    EXPECT_TRUE(parse_digit('8').is_ok_and([](int v) { return v == 8; }));
    EXPECT_FALSE(parse_digit('x').is_ok_and([](int) { return true; }));
    EXPECT_TRUE(parse_digit('x').is_err_and([](parse_error e) { return e == parse_error::not_a_number; }));
    EXPECT_FALSE(parse_digit('8').is_err_and([](parse_error) { return true; }));
}

TEST(ResultConstruction, OkAndErrAsOptional)
{
    EXPECT_EQ(parse_digit('3').ok(), std::optional<int>(3));
    EXPECT_EQ(parse_digit('x').ok(), std::nullopt);
    EXPECT_EQ(parse_digit('x').err(), std::optional<parse_error>(parse_error::not_a_number));
    EXPECT_EQ(parse_digit('3').err(), std::nullopt);
    snicholls::result<handle, int> h = snicholls::ok(handle(9));
    std::optional<handle> taken = std::move(h).ok();   // move-only: only the rvalue overload exists
    ASSERT_TRUE(taken.has_value());
    EXPECT_EQ(*taken->p, 9);
}

TEST(ResultConstruction, IterIsARangeOfZeroOrOneValue)
{
    int sum = 0;
    const auto six = parse_digit('6');
    const auto bad = parse_digit('x');
    for (int v : six.iter())
        sum += v;
    for (int v : bad.iter())
        sum += v;
    EXPECT_EQ(sum, 6);
    // A span into a temporary would dangle: iter() on an rvalue is deleted.
    static_assert(iterable_result<int_result&> && iterable_result<const int_result&>);
    static_assert(!iterable_result<int_result> && !iterable_result<const int_result&&>);
    auto r = parse_digit('2');
    for (int& v : r.iter_mut())
        v = 20;
    EXPECT_EQ(*r, 20);
}

// ---- unwrap family -------------------------------------------------------------------------------------------------

TEST(ResultUnwrap, SuccessReturnsTheValueAndMovesFromAnRvalue)
{
    EXPECT_EQ(parse_digit('5').unwrap(), 5);
    EXPECT_EQ(parse_digit('5').expect("a digit"), 5);
    snicholls::result<handle, int> h = snicholls::ok(handle(4));
    handle out = std::move(h).unwrap();
    EXPECT_EQ(*out.p, 4);
    counted::reset();
    snicholls::result<counted, int> c(std::in_place, 1);
    counted moved = std::move(c).unwrap();
    EXPECT_EQ(counted::copies, 0);
    EXPECT_EQ(counted::moves, 1);
    EXPECT_EQ(moved.v, 1);
}

TEST(ResultUnwrap, ErrorThrowsBadResultAccessCarryingTheError)
{
    try
    {
        (void)parse_digit('x').unwrap();
        FAIL() << "unwrap() of an error must throw";
    }
    catch (const snicholls::bad_result_access<parse_error>& e)
    {
        EXPECT_EQ(e.error(), parse_error::not_a_number);
        EXPECT_STREQ(e.what(), "called unwrap() on an err: error 2");
    }
    try
    {
        (void)snicholls::result<int, std::string>(snicholls::err("disk full")).expect("saving the book");
        FAIL();
    }
    catch (const snicholls::bad_result_access<std::string>& e)
    {
        EXPECT_STREQ(e.what(), "saving the book: disk full");
        EXPECT_EQ(e.error(), "disk full");
    }
    // bad_result_access is a std::logic_error, so existing logic_error handlers catch it.
    EXPECT_THROW((void)parse_digit('x').unwrap(), std::logic_error);
    // A move-only error cannot be in an exception: the base is thrown.
    EXPECT_THROW(((void)snicholls::result<int, std::unique_ptr<int>>(snicholls::err(std::unique_ptr<int>())).unwrap()),
                 snicholls::bad_result_access<void>);
}

TEST(ResultUnwrap, UnwrapErrAndExpectErr)
{
    EXPECT_EQ(parse_digit('x').unwrap_err(), parse_error::not_a_number);
    EXPECT_EQ(parse_digit('x').expect_err("not a digit"), parse_error::not_a_number);
    EXPECT_THROW((void)parse_digit('1').unwrap_err(), snicholls::bad_result_access<void>);
    try
    {
        (void)parse_digit('1').expect_err("should have failed");
        FAIL();
    }
    catch (const snicholls::bad_result_access<void>& e)
    {
        EXPECT_STREQ(e.what(), "should have failed");
    }
}

TEST(ResultUnwrap, FallbacksOnError)
{
    EXPECT_EQ(parse_digit('x').unwrap_or(-1), -1);
    EXPECT_EQ(parse_digit('3').unwrap_or(-1), 3);
    EXPECT_EQ(parse_digit('x').unwrap_or_else([](parse_error e) { return -static_cast<int>(e); }), -2);
    EXPECT_EQ(parse_digit('x').unwrap_or_default(), 0);
    EXPECT_EQ((snicholls::result<std::string, int>(snicholls::ok(std::string("v"))).unwrap_or_default()), "v");
    EXPECT_EQ((snicholls::result<std::string, int>(snicholls::err(1)).unwrap_or("fallback")), "fallback");
}

// ---- monadic operations ----------------------------------------------------------------------------------------

TEST(ResultCombinators, MapAndMapErr)
{
    EXPECT_EQ(parse_digit('3').map([](int v) { return v * 2.5; }), snicholls::ok(7.5));
    EXPECT_EQ(parse_digit('x').map([](int v) { return v * 2.5; }).unwrap_err(), parse_error::not_a_number);
    EXPECT_EQ(parse_digit('x').map_err([](parse_error e) { return std::to_string(static_cast<int>(e)); }).unwrap_err(),
              "2");
    EXPECT_EQ(parse_digit('3').map_err([](parse_error) { return 0.0; }), snicholls::ok(3));
    // map to void
    int seen = 0;
    snicholls::result<void, parse_error> v = parse_digit('4').map([&](int x) { seen = x; });
    EXPECT_TRUE(v.is_ok());
    EXPECT_EQ(seen, 4);
}

TEST(ResultCombinators, MapOrAndMapOrElse)
{
    EXPECT_EQ(parse_digit('3').map_or(0, [](int v) { return v + 1; }), 4);
    EXPECT_EQ(parse_digit('x').map_or(0, [](int v) { return v + 1; }), 0);
    auto describe = [](const int_result& r) {
        return r.map_or_else([](parse_error e) { return std::string("error ") + std::to_string(static_cast<int>(e)); },
                             [](int v) { return std::string("value ") + std::to_string(v); });
    };
    EXPECT_EQ(describe(parse_digit('9')), "value 9");
    EXPECT_EQ(describe(parse_digit('?')), "error 2");
}

TEST(ResultCombinators, AndThenOrElseAndOr)
{
    auto half = [](int v) -> int_result {
        if (v % 2 != 0)
            return snicholls::err(parse_error::not_a_number);
        return snicholls::ok(v / 2);
    };
    EXPECT_EQ(parse_digit('8').and_then(half), snicholls::ok(4));
    EXPECT_EQ(parse_digit('7').and_then(half), snicholls::err(parse_error::not_a_number));
    EXPECT_EQ(parse_digit('x').and_then(half), snicholls::err(parse_error::not_a_number));
    auto recover = [](parse_error) -> snicholls::result<int, std::string> { return snicholls::ok(0); };
    EXPECT_EQ(parse_digit('x').or_else(recover), snicholls::ok(0));
    EXPECT_EQ(parse_digit('5').or_else(recover), snicholls::ok(5));
    EXPECT_EQ((parse_digit('5').and_(snicholls::result<std::string, parse_error>(snicholls::ok(std::string("next"))))),
              snicholls::ok(std::string("next")));
    EXPECT_EQ((parse_digit('x').and_(snicholls::result<std::string, parse_error>(snicholls::ok(std::string("next"))))),
              snicholls::err(parse_error::not_a_number));
    EXPECT_EQ((parse_digit('x').or_(snicholls::result<int, std::string>(snicholls::ok(1)))), snicholls::ok(1));
    EXPECT_EQ((parse_digit('3').or_(snicholls::result<int, std::string>(snicholls::ok(1)))), snicholls::ok(3));
}

TEST(ResultCombinators, MapMovesOutOfAnRvalueWithoutCopies)
{
    counted::reset();
    snicholls::result<counted, int> c(std::in_place, 5);
    auto doubled = std::move(c).map([](counted&& x) { return x.v * 2; });
    EXPECT_EQ(*doubled, 10);
    EXPECT_EQ(counted::copies, 0);
    snicholls::result<handle, int> h = snicholls::ok(handle(3));
    auto got = std::move(h).and_then([](handle&& x) -> snicholls::result<handle, int> { return snicholls::ok(std::move(x)); });
    EXPECT_EQ(*got->p, 3);
}

TEST(ResultCombinators, InspectAndInspectErrReturnTheResult)
{
    int value = 0;
    int error = 0;
    auto r = parse_digit('3')
                 .inspect([&](const int& v) { value = v; })
                 .inspect_err([&](const parse_error&) { error = 1; });
    EXPECT_EQ(r, snicholls::ok(3));
    EXPECT_EQ(value, 3);
    EXPECT_EQ(error, 0);
    auto e = parse_digit('x');
    e.inspect([&](const int&) { value = -1; }).inspect_err([&](const parse_error& p) { error = static_cast<int>(p); });
    EXPECT_EQ(value, 3);
    EXPECT_EQ(error, 2);
}

TEST(ResultCombinators, FlattenAndTranspose)
{
    using nested = snicholls::result<int_result, parse_error>;
    EXPECT_EQ(nested(snicholls::ok(parse_digit('2'))).flatten(), snicholls::ok(2));
    EXPECT_EQ(nested(snicholls::ok(parse_digit('x'))).flatten(), snicholls::err(parse_error::not_a_number));
    EXPECT_EQ(nested(snicholls::err(parse_error::empty)).flatten(), snicholls::err(parse_error::empty));

    using maybe = snicholls::result<std::optional<int>, parse_error>;
    EXPECT_EQ(maybe(snicholls::ok(std::optional<int>(4))).transpose(), std::optional<int_result>(snicholls::ok(4)));
    EXPECT_EQ(maybe(snicholls::ok(std::optional<int>())).transpose(), std::nullopt);
    EXPECT_EQ(maybe(snicholls::err(parse_error::empty)).transpose(),
              std::optional<int_result>(int_result(snicholls::err(parse_error::empty))));
}

// ---- assignment ------------------------------------------------------------------------------------------------------

TEST(ResultAssignment, SwitchesStatesAndKeepsTheOldStateWhenACopyThrows)
{
    snicholls::result<std::string, int> r = snicholls::ok(std::string("a"));
    r = snicholls::err(3);
    EXPECT_EQ(r.unwrap_err(), 3);
    r = snicholls::ok(std::string("b"));
    EXPECT_EQ(*r, "b");
    snicholls::result<std::string, int> other = snicholls::err(7);
    r = other;
    EXPECT_EQ(r.unwrap_err(), 7);

    snicholls::result<throws_on_copy, std::string> t = snicholls::err(std::string("old"));
    const snicholls::result<throws_on_copy, std::string> bad = snicholls::ok(throws_on_copy(-1));
    EXPECT_THROW(t = bad, std::runtime_error);
    ASSERT_TRUE(t.is_err());   // strong guarantee: unchanged
    EXPECT_EQ(t.unwrap_err(), "old");
}

TEST(ResultAssignment, SwapExchangesStates)
{
    snicholls::result<std::string, int> a = snicholls::ok(std::string("x"));
    snicholls::result<std::string, int> b = snicholls::err(1);
    swap(a, b);
    EXPECT_EQ(a.unwrap_err(), 1);
    EXPECT_EQ(*b, "x");
}

// ---- void --------------------------------------------------------------------------------------------------------

snicholls::result<void, std::string> check_positive(int v)
{
    if (v <= 0)
        return snicholls::err(std::string("not positive"));
    return snicholls::ok();
}

TEST(ResultVoid, TheRustUnitResult)
{
    EXPECT_TRUE(check_positive(1).is_ok());
    EXPECT_EQ(check_positive(1), snicholls::ok());
    EXPECT_EQ(check_positive(0), snicholls::err(std::string("not positive")));
    check_positive(1).unwrap();
    check_positive(1).expect("positive");
    EXPECT_THROW(check_positive(0).unwrap(), snicholls::bad_result_access<std::string>);
    EXPECT_EQ(check_positive(0).unwrap_err(), "not positive");
    EXPECT_EQ(check_positive(5).map([] { return 42; }), snicholls::ok(42));
    EXPECT_EQ((check_positive(5).and_then([]() -> snicholls::result<int, std::string> { return snicholls::ok(1); })), snicholls::ok(1));
    EXPECT_EQ(check_positive(0).map_err([](const std::string& s) { return s.size(); }), snicholls::err(std::size_t{12}));
    EXPECT_TRUE(check_positive(1).is_ok_and([] { return true; }));
    EXPECT_EQ(check_positive(0).map_or(0, [] { return 1; }), 0);
    int calls = 0;
    (void)check_positive(1).inspect([&] { ++calls; });
    EXPECT_EQ(calls, 1);
    snicholls::result<void, std::string> r = snicholls::err(std::string("x"));
    r = snicholls::ok_t<void>{};
    EXPECT_TRUE(r.is_ok());
    EXPECT_EQ((std::hash<snicholls::result<void, int>>{}(snicholls::ok())), (std::hash<snicholls::result<void, int>>{}(snicholls::ok())));
}

// ---- propagation: SNICHOLLS_TRY --------------------------------------------------------------------------------------

snicholls::result<int, parse_error> two_digits(const char* s)
{
    SNICHOLLS_TRY_ASSIGN(const int tens, parse_digit(s[0]));
    SNICHOLLS_TRY_ASSIGN(const int units, parse_digit(s[1]));
    return snicholls::ok(tens * 10 + units);
}

snicholls::result<void, parse_error> all_digits(const char* s)
{
    for (; *s; ++s)
        SNICHOLLS_TRY_VOID(parse_digit(*s));
    return snicholls::ok();
}

#if SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS
snicholls::result<long, std::error_code> sum_digits(const char* a, const char* b)
{
    // Different error type: converts through snicholls::error_conversion (here: parse_error -> std::error_code would not
    // convert, so map it first).
    auto to_code = [](parse_error e) { return std::error_code(static_cast<int>(e), std::generic_category()); };
    const long x = SNICHOLLS_TRY(two_digits(a).map_err(to_code));
    const long y = SNICHOLLS_TRY(two_digits(b).map_err(to_code));
    return snicholls::ok(x + y);
}

snicholls::result<std::string, int> take_handle(snicholls::result<handle, int> h)
{
    handle got = SNICHOLLS_TRY(std::move(h));   // an rvalue: moved out
    return snicholls::ok(std::to_string(*got.p));
}
#endif

TEST(ResultTry, PortableFormsPropagateTheFirstError)
{
    EXPECT_EQ(two_digits("42"), snicholls::ok(42));
    EXPECT_EQ(two_digits("4x"), snicholls::err(parse_error::not_a_number));
    EXPECT_EQ(two_digits("\0" "1"), snicholls::err(parse_error::empty));
    EXPECT_TRUE(all_digits("123").is_ok());
    EXPECT_EQ(all_digits("1a3"), snicholls::err(parse_error::not_a_number));
}

#if SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS
TEST(ResultTry, StatementExpressionFormConvertsAndMoves)
{
    EXPECT_EQ(sum_digits("12", "30"), snicholls::ok(42L));
    EXPECT_EQ(sum_digits("12", "3x").unwrap_err().value(), 2);
    EXPECT_EQ(take_handle(snicholls::ok(handle(8))), snicholls::ok(std::string("8")));
    EXPECT_EQ(take_handle(snicholls::err(5)), snicholls::err(5));
}
#endif

// ---- exceptions ------------------------------------------------------------------------------------------------

TEST(ResultExceptions, TryInvokeCatchesIntoExceptionPtr)
{
    auto ok = snicholls::try_invoke([](int a, int b) { return a + b; }, 2, 3);
    EXPECT_EQ(*ok, 5);
    auto bad = snicholls::try_invoke([]() -> int { throw std::out_of_range("index 9"); });
    ASSERT_TRUE(bad.is_err());
    EXPECT_EQ(snicholls::error_text(bad.unwrap_err()), "index 9");
    auto v = snicholls::try_invoke([] {});
    EXPECT_TRUE(v.is_ok());
    auto ref = snicholls::try_invoke([]() -> const std::string& {
        static const std::string s = "kept";
        return s;
    });
    static_assert(std::is_same_v<decltype(ref), snicholls::result<std::string, std::exception_ptr>>);
    EXPECT_EQ(*ref, "kept");
}

TEST(ResultExceptions, RethrowThrowsTheOriginalException)
{
    auto bad = snicholls::try_invoke([]() -> int { throw std::out_of_range("index 9"); });
    EXPECT_THROW(bad.rethrow(), std::out_of_range);
    EXPECT_THROW(bad.rethrow_if_err(), std::out_of_range);
    EXPECT_THROW((void)bad.value_or_throw(), std::out_of_range);
    EXPECT_NO_THROW(snicholls::try_invoke([] { return 1; }).rethrow_if_err());
    EXPECT_EQ(snicholls::try_invoke([] { return 1; }).value_or_throw(), 1);
    EXPECT_THROW(snicholls::try_invoke([] { return 1; }).rethrow(), snicholls::bad_result_access<void>);
}

TEST(ResultExceptions, ErrorTypesMapToExceptions)
{
    // A std::exception error is thrown as itself.
    snicholls::result<int, std::invalid_argument> ex = snicholls::err(std::invalid_argument("bad arg"));
    EXPECT_THROW(ex.rethrow(), std::invalid_argument);
    // std::error_code -> std::system_error.
    snicholls::result<int, std::error_code> code = snicholls::err(std::make_error_code(std::errc::timed_out));
    try
    {
        code.rethrow();
    }
    catch (const std::system_error& e)
    {
        EXPECT_EQ(e.code(), std::make_error_code(std::errc::timed_out));
    }
    // The customisation point.
    snicholls::result<int, my_error> custom = snicholls::err(my_error{77});
    try
    {
        custom.rethrow();
    }
    catch (const my_exception& e)
    {
        EXPECT_EQ(e.code, 77);
    }
    // Anything else: bad_result_access<E>.
    EXPECT_THROW(parse_digit('x').rethrow(), snicholls::bad_result_access<parse_error>);
    // A null exception_ptr.
    snicholls::result<int, std::exception_ptr> null = snicholls::err(std::exception_ptr());
    EXPECT_THROW(null.rethrow(), snicholls::bad_result_access<void>);
    EXPECT_EQ(snicholls::error_text(std::exception_ptr()), "null exception_ptr");
}

TEST(ResultExceptions, ResultFromExceptionsMapsTheException)
{
    auto map = [](std::exception_ptr p) -> std::string {
        try
        {
            std::rethrow_exception(p);
        }
        catch (const std::out_of_range&)
        {
            return "range";
        }
        catch (...)
        {
            return "other";
        }
    };
    auto r = snicholls::result_from_exceptions([]() -> int { throw std::out_of_range("x"); }, map);
    static_assert(std::is_same_v<decltype(r), snicholls::result<int, std::string>>);
    EXPECT_EQ(r, snicholls::err(std::string("range")));
    EXPECT_EQ(snicholls::result_from_exceptions([] { return 3; }, map), snicholls::ok(3));
    EXPECT_EQ(snicholls::result_from_exceptions([]() -> int { throw 1; }, map), snicholls::err(std::string("other")));
    EXPECT_TRUE(snicholls::result_from_exceptions([] {}, map).is_ok());
    EXPECT_TRUE(snicholls::result_from_exceptions([]() -> int { throw 2; }).is_err());
}

TEST(ResultExceptions, ErrorTextCoversTheCommonErrorTypes)
{
    EXPECT_EQ(snicholls::error_text(std::string("s")), "s");
    EXPECT_EQ(snicholls::error_text("literal"), "literal");
    EXPECT_EQ(snicholls::error_text(std::runtime_error("what")), "what");
    EXPECT_EQ(snicholls::error_text(std::make_error_code(std::errc::invalid_argument)),
              std::make_error_code(std::errc::invalid_argument).message());
    EXPECT_EQ(snicholls::error_text(42), "42");
    EXPECT_EQ(snicholls::error_text(parse_error::empty), "error 1");
    EXPECT_EQ(snicholls::error_text(my_error{}), "error");
    EXPECT_EQ(snicholls::error_text(std::make_exception_ptr(1)), "unknown exception");
}

// ---- comparison and hashing --------------------------------------------------------------------------------------

TEST(ResultCompare, EqualityOrderingAndHash)
{
    EXPECT_EQ(parse_digit('1'), parse_digit('1'));
    EXPECT_NE(parse_digit('1'), parse_digit('2'));
    EXPECT_NE(parse_digit('1'), parse_digit('x'));
    EXPECT_EQ(parse_digit('1'), (snicholls::result<long, parse_error>(snicholls::ok(1L))));   // across types
    EXPECT_LT(parse_digit('9'), parse_digit('x'));                                 // Ok < Err
    EXPECT_LT(parse_digit('1'), parse_digit('2'));
    EXPECT_LT(parse_digit('\0'), parse_digit('x'));                                // by error
    static_assert(std::is_same_v<decltype(parse_digit('1') <=> parse_digit('2')), std::strong_ordering>);
    static_assert(std::is_same_v<decltype(std::declval<snicholls::result<double, int>>() <=>
                                          std::declval<snicholls::result<double, int>>()),
                                 std::partial_ordering>);
    std::unordered_set<int_result> set{parse_digit('1'), parse_digit('1'), parse_digit('x'), parse_digit('2')};
    EXPECT_EQ(set.size(), 3u);
    // Value 1 and error 1 hash differently (the state is part of the hash).
    EXPECT_NE((std::hash<snicholls::result<int, int>>{}(snicholls::ok(1))), (std::hash<snicholls::result<int, int>>{}(snicholls::err(1))));
    // Disabled like the standard's when an element has no hash.
    static_assert(!std::is_default_constructible_v<std::hash<snicholls::result<my_error, int>>>);
}

// ---- niche -------------------------------------------------------------------------------------------------------

TEST(ResultNiche, AnEmptyErrorLivesInTheInvalidValue)
{
    snicholls::result<slot, not_found> found = snicholls::ok(slot{3});
    snicholls::result<slot, not_found> missing = snicholls::err(not_found{});
    EXPECT_TRUE(found.is_ok());
    EXPECT_EQ(found->index, 3);
    EXPECT_TRUE(missing.is_err());
    found = snicholls::err(not_found{});
    EXPECT_TRUE(found.is_err());
    found = snicholls::ok(slot{0});
    EXPECT_EQ(found.unwrap().index, 0);
    EXPECT_EQ(missing.map([](slot s) { return s.index; }).unwrap_or(-7), -7);
}

TEST(ResultNicheDeathTest, OkOfTheNicheValueTerminates)
{
    using niche_result = snicholls::result<slot, not_found>;
    EXPECT_DEATH(
        {
            niche_result r = snicholls::ok(slot{-1});
            (void)r;
        },
        "niche value");
}

// ---- the log hook (snicholls::error_sink's shape), with GoogleMock -----------------------------------------------------

struct mock_log_sink
{
    MOCK_METHOD(void, log_error, (int level, const parse_error& e, snicholls::source_location where));
};

TEST(ResultLogErr, CallsTheSinkOnlyForAnErrorAndReturnsTheResult)
{
    mock_log_sink sink;
    EXPECT_CALL(sink, log_error(3, parse_error::not_a_number, testing::_)).Times(1);
    auto e = parse_digit('x').log_err(sink, 3);
    EXPECT_EQ(e, snicholls::err(parse_error::not_a_number));
    auto o = parse_digit('1').log_err(sink, 3);   // no call
    EXPECT_EQ(o, snicholls::ok(1));
}


// Nested and same-line TRYs name their temporaries uniquely: this block is compiled with -Wshadow as an error.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wshadow"
#endif
snicholls::result<int, parse_error> add_digits(char a, char b)
{
#if SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS
    const int nested = SNICHOLLS_TRY(snicholls::result<int, parse_error>(snicholls::ok(SNICHOLLS_TRY(parse_digit(a)))));
#else
    SNICHOLLS_TRY_ASSIGN(const int nested, parse_digit(a));
#endif
    SNICHOLLS_TRY_ASSIGN(const int x, parse_digit(b)); SNICHOLLS_TRY_ASSIGN(const int y, parse_digit(b));
    SNICHOLLS_TRY_VOID(parse_digit(a)); SNICHOLLS_TRY_VOID(parse_digit(b));
    return snicholls::ok(nested + x + y);
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

TEST(ResultReview, NestedAndSameLineTriesDoNotShadow)
{
    EXPECT_EQ(add_digits('1', '2'), snicholls::ok(5));
    EXPECT_EQ(add_digits('x', '2'), snicholls::err(parse_error::not_a_number));
    EXPECT_EQ(add_digits('1', 'y'), snicholls::err(parse_error::not_a_number));
}

// ---- review regressions (fix/pre-0.1.0-review) -----------------------------------------------------------------------

// Counts live instances; a constructor from int throws after the old object would have been destroyed. Reassigning a
// result in the same state used to destroy, then construct in place: when the construction threw, the result still
// said "value" and its destructor destroyed the old object a second time (ASan: double free; here: live goes negative).
struct live_counted
{
    static inline int live = 0;
    std::string s;
    explicit live_counted(std::string v) : s(std::move(v)) { ++live; }
    live_counted(int) : s(std::string(100, 'x')) { throw std::runtime_error("no"); }
    live_counted(const live_counted& o) : s(o.s) { ++live; }
    live_counted(live_counted&& o) noexcept : s(std::move(o.s)) { ++live; }
    live_counted& operator=(const live_counted&) = default;
    live_counted& operator=(live_counted&&) noexcept = default;
    ~live_counted() { --live; }
};

TEST(ResultReview, SameStateReassignmentThatThrowsKeepsTheOldValue)
{
    live_counted::live = 0;
    {
        snicholls::result<live_counted, int> r = snicholls::ok(live_counted(std::string(50, 'y')));
        EXPECT_THROW(r = snicholls::ok(42), std::runtime_error);
        ASSERT_TRUE(r.is_ok());
        EXPECT_EQ(r.unwrap().s, std::string(50, 'y'));   // strong guarantee: the old value is intact
        EXPECT_EQ(live_counted::live, 1);
    }
    EXPECT_EQ(live_counted::live, 0);   // destroyed exactly once
    {
        snicholls::result<int, live_counted> r = snicholls::err(live_counted(std::string(50, 'z')));
        EXPECT_THROW(r = snicholls::err(42), std::runtime_error);
        ASSERT_TRUE(r.is_err());
        EXPECT_EQ(r.unwrap_err().s, std::string(50, 'z'));
        EXPECT_EQ(live_counted::live, 1);
    }
    EXPECT_EQ(live_counted::live, 0);
}

snicholls::result<std::vector<int>, std::string> make_values() { return snicholls::ok(std::vector<int>(100, 7)); }
snicholls::result<int, std::vector<int>> make_errors() { return snicholls::err(std::vector<int>(100, 3)); }

TEST(ResultReview, RvalueAccessReturnsByValueSoNothingDangles)
{
    using values = snicholls::result<std::vector<int>, std::string>;
    static_assert(std::is_same_v<decltype(std::declval<values&&>().unwrap()), std::vector<int>>);
    static_assert(std::is_same_v<decltype(std::declval<values&&>().expect("")), std::vector<int>>);
    static_assert(std::is_same_v<decltype(std::declval<values&&>().value_or_throw()), std::vector<int>>);
    static_assert(std::is_same_v<decltype(std::declval<values&&>().unwrap_unchecked()), std::vector<int>>);
    static_assert(std::is_same_v<decltype(std::declval<values&&>().unwrap_or_terminate()), std::vector<int>>);
    static_assert(std::is_same_v<decltype(*std::declval<values&&>()), std::vector<int>>);
    static_assert(std::is_same_v<decltype(*std::declval<const values&&>()), std::vector<int>>);
    static_assert(std::is_same_v<decltype(std::declval<values&&>().unwrap_err()), std::string>);
    static_assert(std::is_same_v<decltype(std::declval<values&&>().expect_err("")), std::string>);
    static_assert(std::is_same_v<decltype(std::declval<values&&>().unwrap_err_unchecked()), std::string>);
    // lvalues still give references
    static_assert(std::is_same_v<decltype(std::declval<values&>().unwrap()), std::vector<int>&>);
    static_assert(std::is_same_v<decltype(std::declval<const values&>().unwrap_err()), const std::string&>);
    // noexcept follows the move
    static_assert(noexcept(std::declval<values&&>().unwrap_unchecked()));
    static_assert(noexcept(*std::declval<values&&>()));
    static_assert(!noexcept(*std::declval<const values&&>()));   // a std::vector copy may throw

    // These dangled before (heap-use-after-free under ASan): the temporary result died at the end of the full
    // expression that initialised the range or the reference.
    long sum = 0;
    for (int x : make_values().unwrap())
        sum += x;
    EXPECT_EQ(sum, 700);
    const auto& kept = make_values().unwrap();
    EXPECT_EQ(kept.size(), 100u);
    const auto& kept_err = make_errors().unwrap_err();
    EXPECT_EQ(kept_err[99], 3);
}

TEST(ResultReview, VoidResultsCompareEqual)
{
    using unit = snicholls::result<void, int>;
    const unit a = snicholls::ok();
    const unit b = snicholls::ok();
    const unit c = snicholls::err(1);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(c == unit(snicholls::err(1)));
    static_assert(noexcept(a == b));
}

TEST(ResultReview, UnwrapOrElseIsNoexceptOnlyWhenTheConversionIs)
{
    using text = snicholls::result<std::string, int>;
    const auto fallback = [](int) noexcept { return "fallback"; };   // const char* -> std::string may throw
    static_assert(!std::is_nothrow_constructible_v<std::string, const char*>);
    static_assert(!noexcept(std::declval<text&&>().unwrap_or_else(fallback)));
    static_assert(!noexcept(std::declval<const text&>().unwrap_or_else(fallback)));
    const auto same = [](int) noexcept { return 7; };
    static_assert(noexcept(std::declval<snicholls::result<int, int>&&>().unwrap_or_else(same)));
    EXPECT_EQ(text(snicholls::err(1)).unwrap_or_else(fallback), "fallback");
}

TEST(ResultReview, UnwrapOrTerminateReturnsTheValue)
{
    snicholls::result<std::string, int> r = snicholls::ok(std::string("v"));
    static_assert(noexcept(r.unwrap_or_terminate()));
    EXPECT_EQ(r.unwrap_or_terminate(), "v");
    EXPECT_EQ(std::as_const(r).unwrap_or_terminate(), "v");
    EXPECT_EQ(std::move(r).unwrap_or_terminate(), "v");
    snicholls::result<void, int> u = snicholls::ok();
    u.unwrap_or_terminate();
}

TEST(ResultReviewDeathTest, UnwrapOrTerminateTerminatesUnderTheThrowingPolicyToo)
{
    static_assert(SNICHOLLS_RESULT_THROW == 1);   // this TU throws on unwrap(); unwrap_or_terminate still terminates
    using text = snicholls::result<int, std::string>;
    EXPECT_DEATH((void)text(snicholls::err(std::string("disk full"))).unwrap_or_terminate(),
                 "called unwrap_or_terminate\\(\\) on an err: disk full");
    EXPECT_DEATH((snicholls::result<void, std::string>(snicholls::err(std::string("x"))).unwrap_or_terminate()),
                 "on an err: x");
}

} // namespace
