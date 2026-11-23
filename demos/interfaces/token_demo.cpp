//
//  token_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/token.hpp>.
//

// tokens chosen at run time
#include <cstdio>
#include <memory>
#include <snicholls/core/interfaces/token.hpp>
#include <snicholls/core/token/tokens.hpp>
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
    std::vector<std::unique_ptr<token_interface>> watched;
    watched.push_back(std::make_unique<token_model<token::ethereum::USDT>>());
    watched.push_back(std::make_unique<token_model<token::tron::USDT>>());
    for (const auto& t : watched)
        std::printf("  watch %.*s on chain %llu\n", static_cast<int>(t->contract().size()), t->contract().data(),
                    static_cast<unsigned long long>(*t->chain().chain_id()));
    //! [interface]
    std::puts("token_interface");
    check(watched[0]->code() == "USDT:1" && watched[0]->scale() == 6, "USDT on Ethereum");
    check(watched[1]->chain().kind() == chain_kind::tron, "USDT on TRON");
    check(watched[1]->symbol() == "USDT", "the same symbol");
    check(to_asset(*watched[1]) == asset_v<token::tron::USDT>, "the asset value");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
