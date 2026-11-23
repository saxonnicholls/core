//
//  asset_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/asset_conformance.hpp>.
//

// the asset conformance kit
#include <cstdio>
#include <snicholls/core/interfaces/asset.hpp>
#include <snicholls/core/interfaces/asset_conformance.hpp>
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

    //! [conformance]
    const auto eth = check_asset(native::ETH{});
    const auto aud = check_currency(currency::AUD{});
    const currency_model<currency::JPY> jpy;
    const auto virtual_form = check_currency(static_cast<const currency_interface&>(jpy));
    //! [conformance]
    std::puts("asset conformance");
    check(!eth && !aud && !virtual_form, "every form conforms");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
