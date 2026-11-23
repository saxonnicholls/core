//
//  token_checks_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/detail/token_checks.hpp>: EIP-55 against the EIP's own examples, TRON base58check against
//  published addresses, and the token code's shape.
//

#include <snicholls/core/detail/token_checks.hpp>

#include <gtest/gtest.h>

#include <string>

namespace {

using namespace snicholls;
using detail::check_evm_address;
using detail::check_token_address;
using detail::check_tron_address;
using detail::is_token_code;

// EIP-55's test cases: all capitals, all lower case, and mixed (the checksum happens to give each form)
static_assert(!check_evm_address("0x52908400098527886E0F7030069857D2E4169EE7"));
static_assert(!check_evm_address("0x8617E340B3D01FA5F11F306F4090FD50E238070D"));
static_assert(!check_evm_address("0xde709f2102306220921060314715629080e2fb77"));
static_assert(!check_evm_address("0x27b1fdb04752bbc536007a920d24acb045561c26"));
static_assert(!check_evm_address("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"));
static_assert(!check_evm_address("0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359"));
static_assert(!check_evm_address("0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB"));
static_assert(!check_evm_address("0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb"));
static_assert(*check_evm_address("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeD") ==
              "token: contract() fails its EIP-55 checksum");
static_assert(*check_evm_address("0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed") ==
              "token: contract() fails its EIP-55 checksum");

static_assert(!check_tron_address("TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"));   // Tether's USDT
static_assert(!check_tron_address("T9yD14Nj9j7xAB4dbGeiX9h8unkKHxuWwb"));   // the zero address

static_assert(is_token_code("USDC:1", "USDC", 1));
static_assert(is_token_code("USDT:728126428", "USDT", 728126428));
static_assert(is_token_code("SAX", "SAX", 8453));
static_assert(is_token_code("SAX", "SAX", std::nullopt));
static_assert(!is_token_code("USDC:01", "USDC", 1));
static_assert(!is_token_code("USDC:2", "USDC", 1));
static_assert(!is_token_code("USDC-1", "USDC", 1));
static_assert(!is_token_code("USDC:1", "USDC", std::nullopt));
static_assert(!is_token_code("USDT:1", "USDC", 1));
static_assert(detail::token_digits(0) == 1 && detail::token_digits(9) == 1 && detail::token_digits(10) == 2);
static_assert(detail::token_digits(18446744073709551615u) == 20);

TEST(TokenChecksEvm, RejectsTheWrongShape)
{
    EXPECT_EQ(*check_evm_address(""), "token: contract() is not 0x and 40 hex digits");
    EXPECT_EQ(*check_evm_address("5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed00"), "token: contract() is not 0x and 40 hex digits");
    EXPECT_EQ(*check_evm_address("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAe"), "token: contract() is not 0x and 40 hex digits");
    EXPECT_EQ(*check_evm_address("0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeg"), "token: contract() is not 0x and 40 hex digits");
    EXPECT_EQ(*check_evm_address("0X5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"), "token: contract() is not 0x and 40 hex digits");
}

TEST(TokenChecksEvm, EveryCaseFlipOfALetterBreaksTheChecksum)
{
    const std::string good = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48";
    ASSERT_FALSE(check_evm_address(good));
    for (std::size_t i = 2; i < good.size(); ++i) {
        std::string bad = good;
        const char c = bad[i];
        if (c >= 'a' && c <= 'f') {
            bad[i] = static_cast<char>(c - 'a' + 'A');
        } else if (c >= 'A' && c <= 'F') {
            bad[i] = static_cast<char>(c - 'A' + 'a');
        } else {
            continue;
        }
        EXPECT_EQ(*check_evm_address(bad), "token: contract() fails its EIP-55 checksum") << bad;
    }
}

TEST(TokenChecksTron, RejectsTheWrongShapeAndChecksum)
{
    const std::string_view shape = "token: contract() is not a TRON address (T and 33 more base58 characters)";
    EXPECT_EQ(*check_tron_address(""), shape);
    EXPECT_EQ(*check_tron_address("TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6"), shape);      // 33 characters
    EXPECT_EQ(*check_tron_address("AR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"), shape);     // not T
    EXPECT_EQ(*check_tron_address("TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj60"), shape);     // '0' is not base58
    EXPECT_EQ(*check_tron_address("TzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzZ"), shape);     // decodes to a first byte other than 0x41
    EXPECT_EQ(*check_tron_address("TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6u"),
              "token: contract() fails its base58check checksum");
    EXPECT_EQ(*check_tron_address("TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj7t"),
              "token: contract() fails its base58check checksum");
}

TEST(TokenChecksAddress, DispatchesOnTheChainKind)
{
    EXPECT_FALSE(check_token_address(chain_kind::evm, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"));
    EXPECT_TRUE(check_token_address(chain_kind::evm, "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"));
    EXPECT_FALSE(check_token_address(chain_kind::tron, "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"));
    EXPECT_TRUE(check_token_address(chain_kind::tron, "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"));
    EXPECT_FALSE(check_token_address(chain_kind::solana, "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"));
    EXPECT_EQ(*check_token_address(chain_kind::other, ""), "token: contract() is empty");
}

} // namespace
