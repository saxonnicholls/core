//
//  chain_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/chain.hpp>.
//

// chain_like and evm_chain_like
#include <cstdio>
#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/concepts/chain.hpp>

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
    static_assert(evm_chain_like<chain::base>);
    static_assert(chain_like<chain::bitcoin> && !evm_chain_like<chain::bitcoin>);
    //! [concepts]
    std::puts("chain concepts");
    check(chain::base::chain_id() == 8453, "Base is 8453");
    check(chain::bitcoin::kind() == chain_kind::bitcoin, "Bitcoin's kind");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
