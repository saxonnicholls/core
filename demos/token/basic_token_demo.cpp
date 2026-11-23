//
//  basic_token_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/token/basic_token.hpp>.
//

// a product declares its own token in one line
#include <cstdio>
#include <snicholls/core/chain/chains.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/token/basic_token.hpp>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

//! [declare]
// A product's token on Base (EIP-55's example address standing in). A typo in the address's case is a compile error.
struct SAX : snicholls::basic_token<snicholls::chain::base, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", 18, "SAX"> {};
static_assert(SAX::code() == "SAX:8453");

// A single-supply issuer token, one asset on every chain, keeps its bare symbol.
struct GLD : snicholls::basic_token<snicholls::chain::arbitrum, "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359", 6, "GLD"> {
    static constexpr std::string_view code() noexcept { return "GLD"; }
};
//! [declare]

int main()
{
    using namespace snicholls;
    std::puts("basic_token");
    check(SAX::scale() == 18 && SAX::symbol() == "SAX", "decimals and symbol");
    check(SAX::chain().chain_id() == 8453, "the chain is a type");
    check(SAX::contract() == "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed", "the contract as written");
    check(asset_v<SAX>.code() == "SAX:8453", "an asset like any other");
    check(GLD::code() == "GLD" && asset_v<GLD>.scale() == 6, "a bare-symbol code");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
