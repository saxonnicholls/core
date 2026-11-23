//
//  asset_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/asset.hpp>.
//

// assets behind a virtual interface
#include <cstdio>
#include <snicholls/core/interfaces/asset.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/native.hpp>
#include <memory>
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
    std::vector<std::unique_ptr<asset_interface>> held;
    held.push_back(std::make_unique<asset_model<native::ETH>>());
    held.push_back(std::make_unique<asset_model<asset>>(*make_asset("USDC:56", 18)));
    held.push_back(std::make_unique<currency_model<currency::AUD>>());
    //! [interface]
    std::puts("asset_interface");
    check(held[0]->code() == "ETH" && held[0]->scale() == 18, "ETH");
    check(held[1]->code() == "USDC:56", "a registry token");
    check(*to_asset(*held[2]) == asset_v<currency::AUD>, "back to a value");
    const currency_model<currency::EUR> eur;
    check(eur.numeric() == 978 && eur.name() == "Euro", "a currency's numeric code and name");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
