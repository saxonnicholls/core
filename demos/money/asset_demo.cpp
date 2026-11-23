//
//  asset_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/money/asset.hpp>.
//

// assets as types and as 14-byte values
#include <cstdio>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/native.hpp>
#include <string_view>
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

    //! [assets]
    struct eeau {   // a product's token, declared as a type
        static constexpr std::string_view code() noexcept { return "EEAU"; }
        static constexpr std::uint8_t scale() noexcept { return 18; }
    };
    constexpr asset aud = asset_v<currency::AUD>;       // checked by the compiler
    constexpr asset token = asset_v<eeau>;
    constexpr asset gold = asset_v<with_scale<currency::XAU, 18>>;   // ISO gives XAU no minor units
    const auto usdc_bsc = make_asset("USDC:56", 18);     // at run time, from data
    //! [assets]
    std::puts("Assets");
    check(aud.code() == "AUD" && aud.scale() == 2, "AUD at 2 places");
    check(token.code() == "EEAU" && token.scale() == 18, "a product token at 18 places");
    check(gold.scale() == 18, "gold given a scale");
    check(usdc_bsc && usdc_bsc->code() == "USDC:56", "a registry code at run time");
    check(!make_asset("usdc", 6) && !make_asset("AUD", 40), "malformed codes and scales are refused");
    check(asset_v<native::ETH> != *make_asset("ETH", 6), "equality compares the scale too");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
