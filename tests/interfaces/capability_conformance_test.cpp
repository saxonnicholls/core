//
//  capability_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/capability_conformance.hpp>: the report, the leak check and the check runner.
//

#include <snicholls/core/interfaces/capability_conformance.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string_view>

using namespace snicholls;

namespace {

static_assert(capability_provider_response.find(capability_redaction_marker) != std::string_view::npos);
static_assert(!provider_code::is_code(capability_provider_response), "the response as a whole is never a code");

struct probe {};

TEST(CapabilityConformance, TheLeakCheckLooksInBothFields)
{
    // at run time: GCC 15 does not evaluate string_view::find over a temporary's array in a constant expression
    const capability_error in_code{capability_errc::internal, "REDACTION-MARKER-c0re", {}};
    const capability_error in_message{capability_errc::internal, {}, "REDACTION-MARKER-c0re in a message"};
    const capability_error clean{capability_errc::internal, "provider_error", "fixed"};
    EXPECT_TRUE(capability_error_leaks(in_code, capability_redaction_marker));
    EXPECT_TRUE(capability_error_leaks(in_message, capability_redaction_marker));
    EXPECT_FALSE(capability_error_leaks(clean, capability_redaction_marker));
}

TEST(CapabilityConformance, AReportNamesEveryFailedExpectation)
{
    capability_conformance_report report;
    detail::capability_check<probe>(report, "passes", [](probe&, const auto& expect) { expect(true, "fine"); });
    detail::capability_check<probe>(report, "fails", [](probe&, const auto& expect) {
        expect(false, "first");
        expect(false, "second");
    });
    EXPECT_EQ(report.checks, 2u);
    EXPECT_FALSE(report.ok());
    EXPECT_TRUE(report.failed("fails"));
    EXPECT_FALSE(report.failed("passes"));
    ASSERT_EQ(report.failures.size(), 2u);
    EXPECT_EQ(report.failures[1].expectation, "second");
}

TEST(CapabilityConformance, AnEscapingExceptionIsAFailureNotACrash)
{
    capability_conformance_report report;
    detail::capability_check<probe>(report, "throws", [](probe&, const auto&) { throw std::runtime_error("boom"); });
    EXPECT_TRUE(report.failed("throws"));
    EXPECT_EQ(report.failures[0].expectation, "no exception escapes");
}

} // namespace
