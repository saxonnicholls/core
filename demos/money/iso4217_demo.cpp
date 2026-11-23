//
//  iso4217_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/money/iso4217.hpp>.
//

// ISO 4217 as types and as a table
#include <cstdio>
#include <snicholls/core/display.hpp>
#include <snicholls/core/meta/attributes.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <cstdint>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    using namespace snicholls;

    using namespace snicholls::literals;
    //! [iso4217]
    constexpr asset aud = "AUD"_iso4217;                        // a compile error for an unknown code
    static_assert(currency::AUD::numeric() == 36);
    const iso4217_entry* jpy = find_iso4217("JPY");              // a code that arrives at run time
    const iso4217_entry* by_number = find_iso4217(std::uint16_t{978});
    //! [iso4217]
    //! [metadata]
    static_assert(currency::CHF::cash_scale() == 2 && currency::CHF::cash_rounding() == 5);   // cash: 0.05
    static_assert(currency::CHF::countries()[1] == "LI");                                       // CH and LI
    constexpr std::string_view symbol = attribute_or(currency::AUD{}, label{""}).text;        // "A$", by ADL
    //! [metadata]
    std::printf("ISO 4217, list one of %.*s\n", static_cast<int>(iso4217_published.size()), iso4217_published.data());
    check(aud.scale() == 2, "AUD has 2 minor units");
    check(jpy && jpy->scale == 0, "JPY has none");
    check(by_number && by_number->code == "EUR", "978 is EUR");
    check(!iso4217_asset("XAU") && find_iso4217("XAU"), "XAU is ISO but has no scale");
    check(find_iso4217("ANG") && find_iso4217("ANG")->withdrawn == "2025-03", "historic codes still resolve");
    check(!iso4217_entries().empty(), "the table is available");
    check(symbol == "A$" && find_iso4217("EUR")->symbol == "€", "CLDR's root-locale symbols");
    check(find_iso4217("SEK")->cash_scale == 0 && find_iso4217("SEK")->cash_rounding == 1, "SEK cash is whole kronor");
    check(find_iso4217("AUD")->cash_scale == -1, "AUD has no separate cash rule");
    check(find_iso4217("EUR")->countries.size() > 20, "the euro's countries");
    std::printf("CLDR %.*s\n", static_cast<int>(iso4217_cldr_version.size()), iso4217_cldr_version.data());
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
