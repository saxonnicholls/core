//
//  wallet_verifier_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/wallet_verifier.hpp>: a SIWE sign-in through the concept, the address form and
//  the EIP-191 hash. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/wallet_verifier.hpp>
#include <snicholls/core/interfaces/wallet_verifier_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <span>
#include <string_view>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

//! [concept]
// Sign-In with Ethereum's last step: any verifier, the address the message names, the message, the signature.
template <wallet_verifier V>
capability_result<void> sign_in(const V& v, std::uint64_t chain, std::string_view address, std::string_view siwe_message,
                                std::span<const std::byte> signature)
{
    if (!is_lowercase_evm_address(address))
        return fail(capability_errc::invalid_argument, {}, "not a lowercase EVM address");
    return v.verify({chain, address, siwe_message, signature});
}
//! [concept]
} // namespace

int main()
{
    std::printf("snicholls-core concepts/wallet_verifier.hpp demo\n");
    constexpr std::string_view wallet = "0x52908400098527886e0f7030069857d2e4169ee7";
    constexpr std::string_view message = "example.test wants you to sign in with your Ethereum account:\n...";
    memory_wallet_verifier v;   // test wallets only
    static_cast<void>(v.add_wallet(wallet, "test-only key"));
    const auto signature = v.sign(wallet, message);

    check(sign_in(v, 1, wallet, message, signature).is_ok(), "the wallet signed the message");
    check(sign_in(v, 1, "0x52908400098527886E0F7030069857D2E4169EE7", message, signature).unwrap_err().code ==
              capability_errc::invalid_argument,
          "a checksummed address is lowercased first");
    constexpr auto h = personal_sign_hash("hello world");   // at compile time
    std::printf("  personal_sign_hash(\"hello world\") = 0x%s\n", h.hex().c_str());
    check(h.hex().view() == "d9eba16ed0ecae432b71fe008c98cc872bb4cc214d3220a36f365326cf807d68", "the EIP-191 hash");
    return failures == 0 ? 0 : 1;
}
