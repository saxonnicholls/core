//
//  chain_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/chain_conformance.hpp>.
//

// the chain conformance kit
#include <cstdio>
#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/interfaces/chain.hpp>
#include <snicholls/core/interfaces/chain_conformance.hpp>

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
    const auto type_form = check_chain(chain::arbitrum{});
    const chain_model<chain::arbitrum> model;
    const auto virtual_form = check_chain(static_cast<const chain_interface&>(model));
    //! [conformance]
    std::puts("chain conformance");
    check(!type_form && !virtual_form, "both forms conform");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
