//
//  country_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/country.hpp>.
//

// ISO 3166-1 as types and as a table
#include <cstdio>
#include <cstdint>
#include <snicholls/core/country.hpp>

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

    //! [country]
    static_assert(country::AU::alpha3() == "AUS" && country::AU::calling_code() == 61);
    static_assert(country::CH::currencies()[0] == "CHF");
    constexpr country_entry nz = "NZ"_country;                    // a compile error for an unknown code
    const country_entry* gb = find_country("GBR");                // alpha-3 that arrives at run time
    const country_entry* by_number = find_country(std::uint16_t{392});
    //! [country]
    std::printf("ISO 3166-1 from Unicode CLDR %.*s: %zu countries\n", static_cast<int>(iso3166_cldr_version.size()),
                iso3166_cldr_version.data(), iso3166_entries().size());
    check(nz.alpha3 == "NZL" && nz.calling_code == 64, "NZ is NZL, +64");
    check(gb && gb->alpha2 == "GB" && gb->currencies[0] == "GBP", "GBR is GB and uses GBP");
    check(by_number && by_number->name == "Japan", "392 is Japan");
    check(find_country("AU") == find_country(std::uint16_t{36}), "alpha-2 and numeric find the same row");
    check(!find_country("ZZ") && !find_country("A") && !find_country(std::uint16_t{999}), "unknown codes find nothing");
    check(country::AQ::calling_code() == 0 && country::AQ::currencies().empty(), "Antarctica has neither");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
