//
//  uint256_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/uint256.hpp>: token amounts in base units, the
//  arithmetic a ledger does on them, the human rendering at the leaf, and the
//  refusals that replace the exceptions. Exits non-zero if any value is not the
//  one documented.
//

#include <snicholls/core/uint256.hpp>

#include <cstdio>
#include <optional>
#include <sstream>
#include <string>

int main()
{
    std::printf("snicholls-core uint256 demo\n");
    bool ok = true;
    using snicholls::uint256;

    //! [amounts]
    // 10,000 USDT on an 18-decimal chain is 1e22 base units: no uint64 holds it, and that
    // is the deposit that once overflowed std::stoull after the bill had been issued.
    const std::optional<uint256> deposit = uint256::from_decimal("10000000000000000000000");
    const std::optional<uint256> fee = uint256::from_decimal("2500000000000000000");   // 2.5 tokens
    const uint256 one_token = uint256::pow10(18);

    // The ledger works in base units, exactly.
    const uint256 net = *deposit - *fee;
    const auto [whole, fraction] = net.divmod(one_token);

    // Human units only at the leaf, from the amount and its decimals together.
    const std::string human = net.format_units(18);   // 9997.500000000000000000, exactly

    // USDT has 6 decimals, and Binance quotes eight places: parse_units takes both, and rounds nothing.
    const std::optional<uint256> withdrawal = uint256::parse_units("250.12000000", 6);   // 250,120,000 base units

    // The ABI word for the call that moves it.
    const std::string word = net.to_abi_word();   // 64 hex digits, zero-padded

    // Overflow and bad input are refused, never wrapped and never thrown.
    const std::optional<uint256> too_much = uint256::checked_mul(uint256::max(), 2);
    const std::optional<uint256> not_a_number = uint256::from_decimal("10,000");
    //! [amounts]

    std::printf("  deposit   %s base units\n", deposit->to_string().c_str());
    std::printf("  net       %s = %s tokens\n", net.to_string().c_str(), human.c_str());
    std::printf("  abi word  %s\n", word.c_str());
    std::printf("  hex       %s\n", net.to_hex().c_str());
    std::printf("  max * 2   %s\n", too_much ? "fits" : "refused");
    std::printf("  \"10,000\"  %s\n", not_a_number ? "parsed" : "refused");

    ok &= deposit.has_value() && fee.has_value();
    ok &= human == "9997.500000000000000000";
    ok &= withdrawal == uint256{250'120'000} && withdrawal->format_units(6, uint256::fraction_digits::trimmed) == "250.12";
    ok &= whole == 9997 && fraction == uint256::from_decimal("500000000000000000");
    ok &= word.size() == 64 && uint256::from_hex(word) == net && uint256::from_hex(net.to_hex()) == net;
    ok &= !too_much && !not_a_number;
    ok &= net.to_string() == "9997500000000000000000";

    // Wrapping arithmetic, like the built-in unsigned types, when that is what is wanted
    const uint256 wrapped = uint256::max() + 1;
    std::printf("  max + 1   %s (wraps)\n", wrapped.to_string().c_str());
    ok &= wrapped.is_zero();

    // A uint256 prints itself in decimal: print(std::ostream&) const, so acout and
    // print_operators find it without this header knowing about them.
    std::ostringstream shown;
    net.print(shown);
    ok &= shown.str() == "9997500000000000000000";

    std::printf(ok ? "uint256 demo: ok\n" : "uint256 demo: FAILED\n");
    return ok ? 0 : 1;
}
