//
//  chains_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/chain/chains.hpp>.
//

// the chains as types
#include <cstdio>
#include <snicholls/core/chain/chains.hpp>

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

    //! [chains]
    static_assert(chain::ethereum::chain_id() == 1 && chain::base::chain_id() == 8453);
    using native = decltype(chain::bsc::native_asset());     // BNB
    static_assert(native::scale() == 18);
    //! [chains]
    std::puts("chains");
    check(chain::arbitrum::chain_id() == 42161, "Arbitrum One");
    check(chain::tron::kind() == chain_kind::tron, "TRON is its own kind");
    check(chain::sepolia::chain_id() == 11155111, "Sepolia, a test network as a type");
    check(chain::solana::name() == "Solana", "Solana");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
