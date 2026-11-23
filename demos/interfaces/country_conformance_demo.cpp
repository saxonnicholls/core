//
//  country_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/country_conformance.hpp>.
//

// check_country: the conformance kit
#include <cstdint>
#include <cstdio>
#include <snicholls/core/country.hpp>
#include <snicholls/core/interfaces/country.hpp>
#include <snicholls/core/interfaces/country_conformance.hpp>
#include <span>
#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// A caller's record whose numeric code is wrong.
struct stale_record {
    static constexpr std::string_view name() noexcept { return "Australia"; }
    static constexpr std::string_view alpha2() noexcept { return "AU"; }
    static constexpr std::string_view alpha3() noexcept { return "AUS"; }
    static constexpr std::uint16_t numeric() noexcept { return 37; }
    static constexpr std::span<const std::string_view> currencies() noexcept { return {}; }
    static constexpr std::uint16_t calling_code() noexcept { return 61; }
};
} // namespace

int main()
{
    using namespace snicholls;

    //! [conformance]
    const auto type_form = check_country(country::JP{});
    const country_model<country::JP> model;
    const auto virtual_form = check_country(static_cast<const country_interface&>(model));
    const auto stale = check_country(stale_record{});   // "country: alpha3() or numeric() disagrees with ISO 3166-1"
    //! [conformance]
    std::puts("country conformance");
    check(!type_form && !virtual_form, "both forms conform");
    check(stale && *stale == "country: alpha3() or numeric() disagrees with ISO 3166-1", "a wrong numeric code is named");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
