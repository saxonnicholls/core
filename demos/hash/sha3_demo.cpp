//
//  sha3_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/sha3.hpp>: Ethereum's function selectors and event topics as compile-time
//  constants, an address from a public key, SHA-3, and SHAKE's extendable output. Exits non-zero if any value is
//  not the one documented.
//

#include <snicholls/core/hash/sha3.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>

namespace {

//! [keccak]
// A function selector is the first four bytes of Keccak-256 of the signature: a constant, so a dispatch table
// over calldata is a switch
constexpr std::uint32_t selector(std::string_view signature) noexcept
{
    const auto h = snicholls::keccak256::hash(signature);
    return (static_cast<std::uint32_t>(h[0]) << 24) | (static_cast<std::uint32_t>(h[1]) << 16) |
           (static_cast<std::uint32_t>(h[2]) << 8) | static_cast<std::uint32_t>(h[3]);
}

const char* method(std::uint32_t calldata_selector) noexcept
{
    switch (calldata_selector) {
    case selector("transfer(address,uint256)"): return "transfer";
    case selector("approve(address,uint256)"): return "approve";
    case selector("balanceOf(address)"): return "balanceOf";
    default: return "unknown";
    }
}

// Keccak-256 is not SHA3-256: the padding differs, and so does every digest
static_assert(snicholls::keccak256::hash("") != snicholls::sha3_256::hash(""));
//! [keccak]

} // namespace

int main()
{
    std::printf("snicholls-core sha3 demo\n");
    bool ok = true;

    std::printf("  selector(transfer) = %08x -> %s\n", selector("transfer(address,uint256)"), method(0xa9059cbbu));
    ok &= selector("transfer(address,uint256)") == 0xa9059cbbu && std::string_view(method(0x095ea7b3u)) == "approve";
    ok &= std::string_view(method(0x70a08231u)) == "balanceOf" && std::string_view(method(0)) == "unknown";

    // An event topic is the whole hash
    const auto topic = snicholls::keccak256::hash("Transfer(address,address,uint256)");
    std::printf("  Transfer topic     = %s\n", topic.hex().c_str());
    ok &= topic.hex().view() == "ddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef";

    // SHA-3 proper
    std::printf("  sha3-256(abc)      = %s\n", snicholls::sha3_256::hash("abc").hex().c_str());
    ok &= snicholls::sha3_256::hash("abc").hex().view() == "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532";
    ok &= snicholls::sha3_224::hash("abc").hex().view() == "e642824c3f8cf24ad09234ee7d3c766fc9a3a5168d0c94ad73b46fdf";
    ok &= snicholls::sha3_384::hash("").hex().view().starts_with("0c63a75b845e4f7d");
    ok &= snicholls::sha3_512::hash("").hex().view().starts_with("a69f73cca23a9ac5");

    // SHAKE: any length, from any number of squeezes; a shorter output is a prefix of a longer one
    snicholls::shake256 xof;
    xof.update("seed for a mask");
    std::array<std::byte, 40> stream{};
    xof.squeeze(std::span<std::byte>(stream).first(10));
    xof.squeeze(std::span<std::byte>(stream).subspan(10));
    const auto first32 = snicholls::shake256::hash<32>("seed for a mask");
    bool prefix = true;
    for (std::size_t i = 0; i < 32; ++i)
        prefix &= stream[i] == first32[i];
    std::printf("  shake256 stream is one stream however it is squeezed: %s\n", prefix ? "yes" : "no");
    ok &= prefix && snicholls::shake128::hash<16>("").hex().view() == "7f9c2ba4e88f827d616045507605853e";
    snicholls::shake128 x128;
    x128.update("");
    ok &= x128.finish<16>() == snicholls::shake128::hash<16>("");

    // The permutation itself, for a construction of your own
    std::array<std::uint64_t, 25> state{};
    snicholls::keccak_f1600(state);
    ok &= state[0] == 0xf1258f7940e1dde7ull;

    std::printf(ok ? "sha3 demo: ok\n" : "sha3 demo: FAILED\n");
    return ok ? 0 : 1;
}
