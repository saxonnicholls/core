//
//  wallet_verifier_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/wallet_verifier_reference.hpp>: the test scheme's shape, what it binds (the
//  wallet, the message), the address form, and an armed failure returned once.
//

#include <snicholls/core/interfaces/wallet_verifier_reference.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

constexpr std::string_view alice = "0x52908400098527886e0f7030069857d2e4169ee7";
constexpr std::string_view bob = "0x8617e340b3d01fa5f11f306f4090fd50e238070d";

TEST(MemoryWalletVerifier, ATestSignatureHasAnEoaSignaturesShape)
{
    memory_wallet_verifier v;
    ASSERT_TRUE(v.add_wallet(alice, "k1").is_ok());
    const auto sig = v.sign(alice, "sign in");
    ASSERT_EQ(sig.size(), memory_wallet_verifier::signature_bytes);
    EXPECT_EQ(sig.back(), std::byte{0x1b}) << "v = 27";
    const auto mac = hmac<sha256>::mac(std::string_view{"k1"}, personal_sign_hash("sign in"));
    EXPECT_TRUE(std::equal(mac.bytes.begin(), mac.bytes.end(), sig.begin())) << "r is the HMAC of the EIP-191 hash";
    EXPECT_EQ(v.sign(alice, "sign in"), sig) << "deterministic";
    EXPECT_TRUE(v.sign(bob, "sign in").empty()) << "an unknown wallet signs nothing";
}

TEST(MemoryWalletVerifier, ASignatureBindsTheWalletAndTheMessage)
{
    memory_wallet_verifier v;
    ASSERT_TRUE(v.add_wallet(alice, "k1").is_ok());
    ASSERT_TRUE(v.add_wallet(bob, "k2").is_ok());
    const auto sig = v.sign(alice, "sign in");
    EXPECT_TRUE(v.verify({1, alice, "sign in", sig}).is_ok());
    EXPECT_EQ(v.verify({1, bob, "sign in", sig}).unwrap_err().code, capability_errc::signature_invalid);
    EXPECT_EQ(v.verify({1, alice, "sign in!", sig}).unwrap_err().code, capability_errc::signature_invalid);
    EXPECT_EQ(v.verify({1, "0x0000000000000000000000000000000000000001", "sign in", sig}).unwrap_err().code,
              capability_errc::signature_invalid) << "an unknown wallet";
    ASSERT_TRUE(v.add_wallet(alice, "k3").is_ok());
    EXPECT_EQ(v.verify({1, alice, "sign in", sig}).unwrap_err().code, capability_errc::signature_invalid) << "a new key";
}

TEST(MemoryWalletVerifier, RefusesAnAddressNotInTheContractsForm)
{
    memory_wallet_verifier v;
    EXPECT_EQ(v.add_wallet("0x52908400098527886E0F7030069857D2E4169EE7", "k").unwrap_err().code, capability_errc::invalid_argument);
    EXPECT_EQ(v.verify({1, "alice.eth", "m", {}}).unwrap_err().code, capability_errc::invalid_argument);
}

TEST(MemoryWalletVerifier, AnArmedFailureIsReturnedOnce)
{
    memory_wallet_verifier v;
    ASSERT_TRUE(v.add_wallet(alice, "k1").is_ok());
    const auto sig = v.sign(alice, "m");
    v.fail_next(capability_errc::unavailable, "rpc_unreachable {\"node\":\"never kept\"}");
    const auto e = v.verify({8453, alice, "m", sig}).unwrap_err();
    EXPECT_EQ(e.code, capability_errc::unavailable);
    EXPECT_EQ(e.provider_code.view(), "rpc_unreachable");
    EXPECT_TRUE(v.verify({8453, alice, "m", sig}).is_ok());
}

} // namespace
