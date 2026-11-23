//
//  capability_error_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/capability_error.hpp>: the codes, retryability, the provider code's character set, the literal-only message and the layout.
//

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/meta/enums.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(is_retryable(capability_errc::unavailable) && is_retryable(capability_errc::timeout) &&
              is_retryable(capability_errc::rate_limited));
static_assert(!is_retryable(capability_errc::rejected) && !is_retryable(capability_errc::internal) &&
              !is_retryable(capability_errc::stale) && !is_retryable(capability_errc::unsupported));
static_assert(noexcept(is_retryable(capability_errc::timeout)));
static_assert(enum_name(capability_errc::insufficient_funds) == "insufficient_funds");
static_assert(enum_count_v<capability_errc> == 13);
static_assert(enum_cast<capability_errc>("signature_invalid") == capability_errc::signature_invalid);

static_assert(std::is_trivially_copyable_v<capability_error>);
static_assert(std::is_standard_layout_v<capability_error>);
static_assert(sizeof(capability_error) == 99);
static_assert(alignof(capability_error) == 1);
static_assert(std::is_trivially_copyable_v<capability_result<int>>);

// provider codes: a literal checked by the compiler, text checked at run time
static_assert(provider_code{"card_declined"} == std::string_view{"card_declined"});
static_assert(provider_code::is_code("50011") && provider_code::is_code("a.b-c:d_E"));
static_assert(!provider_code::is_code("") && !provider_code::is_code("two words") && !provider_code::is_code("{\"json\":1}"));
static_assert(!provider_code::is_code("abcdefghijklmnopqrstuvwxyz0123456"));   // 33 characters
static_assert(provider_code::is_code("abcdefghijklmnopqrstuvwxyz012345"));     // 32
static_assert(!provider_code::make("has space").is_ok());
static_assert(provider_code::make("ok").is_ok());
static_assert(capability_message{"fixed text"}.view() == "fixed text");
static_assert(!std::is_constructible_v<capability_message, std::string_view>, "no run-time message: the redaction rule");
static_assert(!std::is_constructible_v<capability_message, const std::string&>);
static_assert(!std::is_constructible_v<provider_code, std::string_view>, "run-time text goes through make");

constexpr capability_result<int> refuse() { return fail(capability_errc::rejected, "card_declined", "the card was declined"); }
static_assert(refuse().is_err());
static_assert(refuse().unwrap_err().provider_code == std::string_view{"card_declined"});

TEST(CapabilityError, FailCarriesTheCodeTheProviderCodeAndTheMessage)
{
    const capability_result<int> r = fail(capability_errc::rate_limited, "429", "slow down");
    ASSERT_TRUE(r.is_err());
    const capability_error& e = r.unwrap_err();
    EXPECT_EQ(e.code, capability_errc::rate_limited);
    EXPECT_EQ(e.provider_code, std::string_view{"429"});
    EXPECT_EQ(e.message, std::string_view{"slow down"});
    EXPECT_TRUE(e.retryable());
    EXPECT_EQ(e, (capability_error{capability_errc::rate_limited, "429", "slow down"}));
    const capability_result<void> bare = fail(capability_errc::not_found);
    EXPECT_TRUE(bare.unwrap_err().provider_code.empty());
    EXPECT_TRUE(bare.unwrap_err().message.empty());
}

TEST(CapabilityError, AProviderErrorKeepsACodeAndDropsAnythingElse)
{
    const capability_error kept = provider_error(capability_errc::rejected, "insufficient_funds");
    EXPECT_EQ(kept.provider_code, std::string_view{"insufficient_funds"});
    const capability_error dropped = provider_error(capability_errc::rejected, "{\"message\":\"declined for someone@example.test\"}");
    EXPECT_EQ(dropped.code, capability_errc::rejected);
    EXPECT_TRUE(dropped.provider_code.empty());
    const std::string long_code(33, 'x');
    EXPECT_TRUE(provider_error(capability_errc::internal, long_code).provider_code.empty()) << "refused, never truncated";
}

TEST(CapabilityError, MakeRefusesTextThatIsNotACode)
{
    EXPECT_EQ(provider_code::make("a b").unwrap_err(), capability_errc::invalid_argument);
    EXPECT_EQ(provider_code::make(std::string(33, 'x')).unwrap_err(), capability_errc::invalid_argument);
    EXPECT_EQ(provider_code::make("card_declined")->view(), "card_declined");
}

} // namespace
