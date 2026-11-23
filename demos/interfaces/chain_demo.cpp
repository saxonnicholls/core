//
//  chain_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/chain.hpp>.
//

// chains behind a virtual interface
#include <cstdio>
#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/interfaces/chain.hpp>
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
    std::vector<std::unique_ptr<chain_interface>> networks;
    networks.push_back(std::make_unique<chain_model<chain::ethereum>>());
    networks.push_back(std::make_unique<chain_model<chain::solana>>());
    //! [interface]
    std::puts("chain_interface");
    check(networks[0]->chain_id() == 1u && networks[0]->native_asset() == asset_v<native::ETH>, "Ethereum");
    check(!networks[1]->chain_id() && networks[1]->kind() == chain_kind::solana, "Solana has no EIP-155 id");
    check(networks[0]->name() == "Ethereum", "name");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
