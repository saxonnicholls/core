//
//  tokens_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/token/tokens.hpp>.
//

// the major tokens as types
#include <cstdio>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/token/tokens.hpp>

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

    //! [tokens]
    static_assert(token::ethereum::USDC::code() == "USDC:1" && token::ethereum::USDC::scale() == 6);
    static_assert(token::bsc::USDC::scale() == 18);                       // not 6 on BNB Smart Chain
    static_assert(asset_v<token::base::USDC> != asset_v<token::ethereum::USDC>);
    // every one of them, to seed a run-time registry
    int count = 0;
    meta::for_each_type<token::all>([&]<typename T>() {
        std::printf("  %-16.*s %2u  %.*s\n", static_cast<int>(T::code().size()), T::code().data(), unsigned{T::scale()},
                    static_cast<int>(T::contract().size()), T::contract().data());
        ++count;
    });
    //! [tokens]
    std::puts("tokens");
    check(count == 19, "19 tokens");
    check(token::tron::USDT::code() == "USDT:728126428", "USDT on TRON");
    check(token::tron::USDT::contract() == "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t", "its base58check address");
    check(token::ethereum::WBTC::scale() == 8, "WBTC has 8 decimals");
    // who stands behind each one on its chain: USDC on BNB Smart Chain is Binance's, not Circle's
    check(token::bsc::USDC::kind() == token::issuance::pegged, "USDC:56 is pegged");
    std::printf("  %.*s\n", static_cast<int>(token::bsc::USDC::issuer().size()), token::bsc::USDC::issuer().data());
    check(token::ethereum::USDC::kind() == token::issuance::native && token::ethereum::USDC::issuer() == "Circle",
          "USDC:1 is Circle's own");
    check(token::base::USDT::kind() == token::issuance::bridged, "USDT:8453 is bridged");
    check(token::ethereum::WETH::kind() == token::issuance::wrapped, "WETH is wrapped");
    check(token::ethereum::USDC::source().starts_with("https://developers.circle.com/"), "its source is Circle's page");
    check(token::ethereum::USDC::name() == "USD Coin", "its name is the contract's");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
