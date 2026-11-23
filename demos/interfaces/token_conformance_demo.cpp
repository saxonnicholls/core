//
//  token_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/token_conformance.hpp>.
//

// the token conformance kit
#include <cstdio>
#include <cstdint>
#include <snicholls/core/interfaces/token.hpp>
#include <snicholls/core/interfaces/token_conformance.hpp>
#include <snicholls/core/token/tokens.hpp>
#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// a registry row with its address copied in lower case
struct lowered {
    static constexpr std::string_view code() noexcept { return "USDC:1"; }
    static constexpr std::uint8_t scale() noexcept { return 6; }
    static constexpr snicholls::chain::ethereum chain() noexcept { return {}; }
    static constexpr std::string_view contract() noexcept { return "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"; }
    static constexpr std::string_view symbol() noexcept { return "USDC"; }
};
} // namespace

int main()
{
    using namespace snicholls;

    //! [conformance]
    const auto type_form = check_token(token::avalanche::USDT{});
    const token_model<token::avalanche::USDT> model;
    const auto virtual_form = check_token(static_cast<const token_interface&>(model));
    const auto broken = check_token(lowered{});   // "token: contract() fails its EIP-55 checksum"
    //! [conformance]
    std::puts("token conformance");
    check(!type_form && !virtual_form, "both forms conform");
    check(broken && *broken == "token: contract() fails its EIP-55 checksum", "a lower-cased address is caught");
    if (broken)
        std::printf("  %.*s\n", static_cast<int>(broken->size()), broken->data());
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
