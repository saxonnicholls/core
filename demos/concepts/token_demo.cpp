//
//  token_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/token.hpp>.
//

// a function written once over token_like
#include <cstdio>
#include <snicholls/core/concepts/token.hpp>
#include <snicholls/core/token/tokens.hpp>

#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

//! [concepts]
// Where to send a deposit of this token: the chain's identifier and the contract to watch.
template <snicholls::token_like T>
constexpr bool is_on_chain(const T& t, std::uint64_t id, std::string_view contract)
{
    return t.chain().chain_id() == id && t.contract() == contract;
}
static_assert(snicholls::token_like<snicholls::token::base::USDC>);
static_assert(!snicholls::token_like<snicholls::native::ETH>);   // a native asset has no contract
//! [concepts]

int main()
{
    using namespace snicholls;
    std::puts("token_like");
    check(is_on_chain(token::base::USDC{}, 8453, "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"), "USDC on Base");
    check(!is_on_chain(token::ethereum::USDC{}, 8453, "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"),
          "USDC on Ethereum is another asset");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
