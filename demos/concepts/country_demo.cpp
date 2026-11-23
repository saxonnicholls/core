//
//  country_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/country.hpp>.
//

// country_like: generic code over any country
#include <cstdio>
#include <snicholls/core/concepts/country.hpp>
#include <snicholls/core/country.hpp>
#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [concepts]
template <snicholls::country_like C>
constexpr bool uses(const C& c, std::string_view currency)
{
    for (const auto code : c.currencies())
        if (code == currency)
            return true;
    return false;
}
static_assert(uses(snicholls::country::PA{}, "USD") && !uses(snicholls::country::AU{}, "USD"));
//! [concepts]
} // namespace

int main()
{
    using namespace snicholls;
    std::puts("country concepts");
    check(country_like<country::FR>, "a generated type models country_like");
    check(uses(country::FR{}, "EUR"), "France uses the euro");
    check(!country_like<int>, "int does not");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
