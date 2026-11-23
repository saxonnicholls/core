//
//  country_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/country.hpp>.
//

// country_interface: a country chosen at run time
#include <cstdio>
#include <memory>
#include <snicholls/core/country.hpp>
#include <snicholls/core/interfaces/country.hpp>
#include <vector>

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

    //! [interface]
    std::vector<std::unique_ptr<country_interface>> shipping;
    shipping.push_back(std::make_unique<country_model<country::AU>>());
    shipping.push_back(std::make_unique<country_model<country::DE>>());
    //! [interface]
    std::puts("country_interface");
    for (const auto& c : shipping)
        std::printf("  %.*s %.*s +%u\n", static_cast<int>(c->alpha2().size()), c->alpha2().data(),
                    static_cast<int>(c->name().size()), c->name().data(), static_cast<unsigned>(c->calling_code()));
    check(shipping[0]->alpha3() == "AUS" && shipping[0]->numeric() == 36, "Australia's codes");
    check(shipping[1]->currencies()[0] == "EUR" && shipping[1]->calling_code() == 49, "Germany: EUR, +49");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
