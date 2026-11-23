//
//  capability_error_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/capability_error.hpp>: a capability's error, its retry rule, the provider code and the redaction rule. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/meta/enums.hpp>

#include <cstdio>
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

// A provider call that fails the way a card decline does.
capability_result<int> charge(bool declined)
{
    if (declined)
        return fail(capability_errc::rejected, "card_declined", "the card was declined");
    return ok(42);
}
} // namespace

int main()
{
    std::printf("snicholls-core capability_error.hpp demo\n");
    //! [capability_error]
    const auto r = charge(true);
    const capability_error& e = r.unwrap_err();
    // a caller branches on the code; only unavailable, timeout and rate_limited are retried
    const bool retry = e.retryable();
    const std::string_view name = enum_name(e.code);   // "rejected": names come from the enumeration itself
    // a provider's raw response never becomes the error: only a short code survives
    const capability_error from_response = provider_error(capability_errc::unavailable, "{\"message\":\"down for you@example.test\"}");
    const auto code = provider_code::make("insufficient_funds");   // checked at run time
    //! [capability_error]
    check(!retry && name == "rejected", "a decline is rejected and not retried");
    check(e.provider_code == std::string_view{"card_declined"} && e.message == std::string_view{"the card was declined"},
          "the provider code and the fixed message travel with the code");
    check(from_response.provider_code.empty() && is_retryable(from_response.code), "a response body is dropped, never kept");
    check(code.is_ok() && !provider_code::make("two words").is_ok(), "make refuses free text");
    check(charge(false).is_ok() && *charge(false) == 42, "success is a value");
    check(provider_code::is_code("50011") && !provider_code{}.view().size(), "an OKX code is a code; the default is empty");
    check(capability_message{"fixed"}.view() == "fixed" && capability_message{}.empty(), "a message is a literal");
    return failures == 0 ? 0 : 1;
}
