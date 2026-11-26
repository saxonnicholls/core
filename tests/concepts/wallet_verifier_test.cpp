//
//  wallet_verifier_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/wallet_verifier.hpp>: the models and the non-models, the address form, and
//  personal_sign_hash against the published EIP-191 vectors (ethers' hashMessage), checked again with an independent
//  Keccak written from the specification.
//

#include <snicholls/core/concepts/wallet_verifier.hpp>
#include <snicholls/core/interfaces/wallet_verifier_reference.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(wallet_verifier<memory_wallet_verifier>);
static_assert(wallet_verifier<wallet_verifier_interface>);
static_assert(wallet_verifier<wallet_verifier_adapter<memory_wallet_verifier>>);
static_assert(wallet_verifier<wallet_verifier_adapter<memory_wallet_verifier&>>);
static_assert(wallet_verifier<wallet_verifier_ref>);

struct verify_not_const {
    capability_result<void> verify(const wallet_signature_request&);
};
struct takes_a_message {
    capability_result<void> verify(std::string_view) const;
};
static_assert(!wallet_verifier<verify_not_const>);
static_assert(!wallet_verifier<takes_a_message>);
static_assert(!wallet_verifier<int>);
static_assert(std::is_trivially_copyable_v<wallet_signature_request>);

static_assert(is_lowercase_evm_address("0x52908400098527886e0f7030069857d2e4169ee7"));
static_assert(!is_lowercase_evm_address("0x52908400098527886E0F7030069857D2E4169EE7"), "a checksummed address is lowercased first");
static_assert(!is_lowercase_evm_address("52908400098527886e0f7030069857d2e4169ee700"));
static_assert(!is_lowercase_evm_address("0x52908400098527886e0f7030069857d2e4169ee"));
static_assert(!is_lowercase_evm_address("0x52908400098527886e0f7030069857d2e4169eg7"));
static_assert(!is_lowercase_evm_address(""));

static_assert(personal_sign_hash("hello world").hex().view() == "d9eba16ed0ecae432b71fe008c98cc872bb4cc214d3220a36f365326cf807d68");
static_assert(personal_sign_hash("hello").hex().view() == "50b2c43fd39106bafbba0da34fc430e1f91e3c96ea2acee2bc34119f92b37750");
static_assert(personal_sign_hash("").hex().view() == "5f35dce98ba4fba25530a026ed80b2cecdaa31091ba4958b99b52ea1d068adad");

TEST(WalletVerifierConcept, TheHashFramesTheLengthInDecimal)
{
    const std::string long_message(1234, 'a');
    keccak256 h;
    h.update(std::string_view{"\x19" "Ethereum Signed Message:\n1234"});
    h.update(long_message);
    EXPECT_EQ(personal_sign_hash(long_message), h.finish());
    EXPECT_NE(personal_sign_hash("ab"), personal_sign_hash("ba"));
}

} // namespace
