//
//  decimal_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/decimal.hpp>.
//

// Demo: exact decimals for strikes and ticks, one canonical text each.
// Exits non-zero if a result differs from what is documented.

#include <array>
#include <cstdio>
#include <string_view>

#include <snicholls/core/decimal.hpp>

int main()
{
    using snicholls::decimal;
    using snicholls::decimal_syntax;
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("  %-62s %s\n", what, ok ? "ok" : "FAILED");
        failures += !ok;
    };

    std::printf("canonical text\n");
    //! [decimal]
    const auto strike = decimal::parse("72.5");                          // canonical text only: no "72.50", no "+"
    const auto tick = decimal::parse("0.0000001");
    const auto plain = decimal::parse("72.50", decimal_syntax::plain);   // venue input may carry trailing zeros
    static_assert(*decimal::parse("6930") == *decimal::from_parts(69300, -1));   // normalised: equal numbers are equal
    //! [decimal]
    check(strike && strike->text().view() == "72.5" && strike->to_string() == "72.5", "72.5 round-trips");
    check(tick && tick->mantissa() == 1 && tick->exponent() == -7, "0.0000001 is 1 x 10^-7");
    check(plain == strike, "plain \"72.50\" equals canonical \"72.5\"");
    check(!decimal::parse("72.50") && !decimal::parse("+1") && !decimal::parse("-0") && !decimal::parse("1e3"),
          "non-canonical spellings are rejected");
    check(!decimal::parse("1234567890123456789") && !decimal::from_parts(1, 19), "beyond 18 digits / exponent 18: rejected");
    check(decimal::from_integer(-5)->is_negative() && decimal{}.is_zero(), "sign and zero");

    std::printf("exact order and output\n");
    check(*decimal::parse("0.1") < *decimal::parse("0.10000000000000001"), "ordering is exact (no double)");
    check(*decimal::parse("-5") < *decimal::parse("-4.99"), "negative ordering");
    check(strike->to_double() == 72.5, "to_double for display and maths");
    std::array<char, 8> buffer{};
    const auto n = decimal::parse("-5.25")->format(buffer);
    check(n && std::string_view(buffer.data(), *n) == "-5.25", "format into a caller's buffer");
    std::array<char, 2> small{};
    check(!decimal::parse("-5.25")->format(small), "a too-small buffer is reported");

    std::printf("%s\n", failures ? "FAILED" : "all ok");
    return failures ? 1 : 0;
}
