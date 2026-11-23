//
//  asset_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/asset.hpp>.
//

// asset_like and currency_like
#include <cstdio>
#include <snicholls/core/concepts/asset.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/money/iso4217.hpp>
#include <snicholls/core/money/native.hpp>

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

    //! [concepts]
    static_assert(currency_like<currency::EUR>);
    static_assert(asset_like<native::ETH> && !currency_like<native::ETH>);
    static_assert(asset_like<asset>);                 // the value models it too
    static_assert(!asset_like<currency::XAU>);        // until a scale is chosen
    //! [concepts]
    std::puts("asset concepts");
    check(asset_like<with_scale<currency::XAU, 18>>, "with_scale makes XAU an asset");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
