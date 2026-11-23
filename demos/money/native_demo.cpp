//
//  native_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/money/native.hpp>.
//

// the native assets of the major chains
#include <cstdio>
#include <snicholls/core/money/asset.hpp>
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

    //! [native]
    constexpr asset eth = asset_v<native::ETH>;
    static_assert(native::ETH::slip44() == 60 && native::BTC::scale() == 8);
    //! [native]
    std::puts("Native assets");
    check(eth.scale() == 18, "ETH has 18 decimals (wei)");
    check(native::SOL::scale() == 9 && native::TRX::scale() == 6, "SOL 9, TRX 6");
    check(native::BTC::name() == "Bitcoin", "names");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
