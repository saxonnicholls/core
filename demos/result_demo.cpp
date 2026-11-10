//
//  result_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/result.hpp>.
//

// Demo: snicholls::result<V, E> (snicholls/core/result.hpp) - Rust's Result in C++: ok/err, the combinators, SNICHOLLS_TRY (Rust's
// `?`), unwrap and its panic, the bridge to and from exceptions, void results and the niche layout.
//
//   ./result_demo        prints each step; exits 1 if any result is not what this file documents.

#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

#include <snicholls/core/result.hpp>

namespace {

bool all_ok = true;

void check(bool cond, std::string_view what)
{
    std::cout << (cond ? "  ok   " : "  FAIL ") << what << '\n';
    all_ok = all_ok && cond;
}

//! [parse]
enum class price_error : std::uint8_t
{
    empty = 1,
    not_a_number = 2,
    negative = 3,
};

// A function that can fail returns snicholls::result: ok(value) or err(error). No allocation, no exception.
snicholls::result<double, price_error> parse_price(std::string_view text) noexcept
{
    if (text.empty())
        return snicholls::err(price_error::empty);
    double value = 0;
    double scale = 0;
    for (const char c : text)
    {
        if (c == '.' && scale == 0)
            scale = 1;
        else if (c >= '0' && c <= '9')
        {
            value = value * 10 + (c - '0');
            scale *= 10;
        }
        else
            return snicholls::err(c == '-' ? price_error::negative : price_error::not_a_number);
    }
    return snicholls::ok(scale > 0 ? value / scale : value);
}
//! [parse]

//! [try]
// SNICHOLLS_TRY is Rust's `?`: the value, or return the error from this function.
snicholls::result<double, price_error> mid_price(std::string_view bid, std::string_view ask)
{
#if SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS
    const double b = SNICHOLLS_TRY(parse_price(bid));
    const double a = SNICHOLLS_TRY(parse_price(ask));
#else
    SNICHOLLS_TRY_ASSIGN(const double b, parse_price(bid));   // the portable statement form
    SNICHOLLS_TRY_ASSIGN(const double a, parse_price(ask));
#endif
    return snicholls::ok((b + a) / 2);
}

// result<void, E> is Rust's Result<(), E>; SNICHOLLS_TRY_VOID discards the value.
snicholls::result<void, price_error> validate_quote(std::string_view bid, std::string_view ask)
{
    SNICHOLLS_TRY_VOID(parse_price(bid));
    SNICHOLLS_TRY_VOID(parse_price(ask));
    return snicholls::ok();
}
//! [try]

// An index whose -1 is never valid: with an empty error, the result is just the index (the niche).
struct level_index
{
    std::int32_t value;
    friend constexpr bool operator==(level_index, level_index) noexcept = default;
};
struct no_level
{
};

} // namespace

template <>
struct snicholls::result_niche<level_index>
{
    static constexpr level_index value{-1};
};

int main()
{
    std::cout << "snicholls::result<V, E>: Rust's Result, zero overhead\n";

    // Observers and unwrap family.
    const auto good = parse_price("101.25");
    const auto bad = parse_price("10x");
    check(good.is_ok() && *good == 101.25, "parse_price(\"101.25\") is ok(101.25)");
    check(bad.is_err() && bad.unwrap_err() == price_error::not_a_number, "parse_price(\"10x\") is err(not_a_number)");
    check(good.is_ok_and([](double p) { return p > 100; }), "is_ok_and");
    check(bad.is_err_and([](price_error e) { return e == price_error::not_a_number; }), "is_err_and");
    check(bad.unwrap_or(0.0) == 0.0 && good.unwrap_or(0.0) == 101.25, "unwrap_or");
    check(bad.unwrap_or_else([](price_error e) { return -static_cast<double>(e); }) == -2.0, "unwrap_or_else");
    check(bad.unwrap_or_default() == 0.0, "unwrap_or_default");
    check(good.ok() == 101.25 && !bad.ok() && bad.err() == price_error::not_a_number, "ok() / err() as std::optional");

    // Combinators, Rust's names (and_ / or_ because `and` / `or` are C++ keywords).
    const auto ticks = good.map([](double p) { return static_cast<long>(p * 4); });
    check(ticks == snicholls::ok(405L), "map");
    check(bad.map_err([](price_error e) { return static_cast<int>(e) * 100; }) == snicholls::err(200), "map_err");
    check(good.map_or(0, [](double p) { return static_cast<int>(p); }) == 101, "map_or");
    check(bad.map_or_else([](price_error) { return std::string("n/a"); }, [](double p) { return std::to_string(p); }) ==
              "n/a",
          "map_or_else");
    auto positive = [](double p) -> snicholls::result<double, price_error> {
        return p > 0 ? snicholls::result<double, price_error>(snicholls::ok(p)) : snicholls::err(price_error::negative);
    };
    check(parse_price("0").and_then(positive) == snicholls::err(price_error::negative), "and_then");
    check(bad.or_else([](price_error) -> snicholls::result<double, std::string> { return snicholls::ok(1.0); }) == snicholls::ok(1.0),
          "or_else");
    check(good.and_(parse_price("7")) == snicholls::ok(7.0) && bad.and_(parse_price("7")) == bad, "and_");
    check(bad.or_(parse_price("7")) == snicholls::ok(7.0), "or_");
    double seen = 0;
    (void)good.inspect([&](double p) { seen = p; }).inspect_err([&](price_error) { seen = -1; });
    check(seen == 101.25, "inspect / inspect_err");
    double total = 0;
    for (const double p : good.iter())
        total += p;
    for (const double p : bad.iter())
        total += p;
    check(total == 101.25, "iter: a range of zero or one value");
    check(snicholls::result<snicholls::result<double, price_error>, price_error>(snicholls::ok(good)).flatten() == good, "flatten");
    check(snicholls::result<std::optional<int>, price_error>(snicholls::ok(std::optional<int>())).transpose() == std::nullopt,
          "transpose");

    // Propagation.
    check(mid_price("100", "102") == snicholls::ok(101.0), "SNICHOLLS_TRY: mid_price(100, 102) == 101");
    check(mid_price("100", "-2") == snicholls::err(price_error::negative), "SNICHOLLS_TRY returns the first error");
    check(validate_quote("1", "2").is_ok() && validate_quote("", "2") == snicholls::err(price_error::empty),
          "result<void, E> and SNICHOLLS_TRY_VOID");

    //! [terminate]
    // Per call, not per type: unwrap() follows SNICHOLLS_RESULT_THROW, value_or_throw() always throws, and
    // unwrap_or_terminate() always terminates - for a call site where an error is a bug, never an exception.
    const double startup_price = parse_price("99.5").unwrap_or_terminate();
    // A temporary's value comes out by value, so this neither dangles nor copies the string:
    const std::string label = snicholls::result<std::string, price_error>(snicholls::ok(std::string("bid"))).unwrap();
    //! [terminate]
    check(startup_price == 99.5 && label == "bid", "unwrap_or_terminate() of a success, unwrap() of a temporary");

    //! [rethrow]
    // unwrap() on an error throws snicholls::bad_result_access<E> (or terminates with SNICHOLLS_RESULT_THROW=0).
    try
    {
        (void)parse_price("abc").expect("price from the feed");
        check(false, "expect() of an error throws");
    }
    catch (const snicholls::bad_result_access<price_error>& e)
    {
        check(e.error() == price_error::not_a_number, std::string("expect() threw: ") + e.what());
    }

    // Exceptions in, results out: try_invoke catches into std::exception_ptr ...
    auto risky = snicholls::try_invoke([] { return std::stoi("not a number"); });
    check(risky.is_err() && snicholls::error_text(risky.unwrap_err()).find("stoi") != std::string::npos, "try_invoke caught std::invalid_argument");
    // ... and rethrow() throws the original exception again.
    try
    {
        risky.rethrow();
    }
    catch (const std::invalid_argument&)
    {
        check(true, "rethrow() threw the original std::invalid_argument");
    }
    // An exception mapped to your own error type.
    auto mapped = snicholls::result_from_exceptions([] { return std::stoi("99999999999999999999"); },
                                               [](std::exception_ptr) { return std::make_error_code(std::errc::result_out_of_range); });
    check(mapped == snicholls::err(std::make_error_code(std::errc::result_out_of_range)), "result_from_exceptions with a map");
    //! [rethrow]
    try
    {
        mapped.rethrow_if_err();
    }
    catch (const std::system_error& e)
    {
        check(e.code() == std::errc::result_out_of_range, "an std::error_code error rethrows as std::system_error");
    }
    check(snicholls::try_invoke([] { return 2; }).value_or_throw() == 2, "value_or_throw on a success");
    mid_price("1", "2").ignore_err();   // discards explicitly

    // Layout: a union plus a bool, or nothing extra at all with a niche.
    static_assert(sizeof(snicholls::result<double, price_error>) == 16);
    static_assert(std::is_trivially_copyable_v<snicholls::result<double, price_error>>);
    static_assert(sizeof(snicholls::result<level_index, no_level>) == sizeof(level_index));
    snicholls::result<level_index, no_level> lookup = snicholls::err(no_level{});
    check(lookup.is_err(), "niche: an empty error stored in the invalid value");
    lookup = snicholls::ok(level_index{3});
    check(lookup->value == 3, "niche: and a value");
    check(parse_price("1") < parse_price("x"), "ordering: every ok before every err (Rust)");
    check(std::hash<snicholls::result<double, price_error>>{}(good) == std::hash<snicholls::result<double, price_error>>{}(good),
          "std::hash");

    std::cout << (all_ok ? "result_demo: all checks passed\n" : "result_demo: FAILED\n");
    return all_ok ? 0 : 1;
}
