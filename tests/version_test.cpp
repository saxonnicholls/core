//
//  version_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/version.hpp>. The macros, the constexpr functions
//  and the string are three views of one number and must agree, and the
//  compile-time claims are made at compile time.
//

#include <snicholls/core/version.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace {

// Everything here is usable in a constant expression, and says so
static_assert(snicholls::core::version_major() == SNICHOLLS_CORE_VERSION_MAJOR);
static_assert(snicholls::core::version_minor() == SNICHOLLS_CORE_VERSION_MINOR);
static_assert(snicholls::core::version_patch() == SNICHOLLS_CORE_VERSION_PATCH);
static_assert(snicholls::core::version() == SNICHOLLS_CORE_VERSION);
static_assert(snicholls::core::version_string() == std::string_view{SNICHOLLS_CORE_VERSION_STRING});
static_assert(noexcept(snicholls::core::version()));
static_assert(noexcept(snicholls::core::version_string()));

// The preprocessor comparison orders versions the way a person does
static_assert(SNICHOLLS_CORE_VERSION_CHECK(1, 0, 0) > SNICHOLLS_CORE_VERSION_CHECK(0, 99, 99));
static_assert(SNICHOLLS_CORE_VERSION_CHECK(0, 2, 0) > SNICHOLLS_CORE_VERSION_CHECK(0, 1, 99));
static_assert(SNICHOLLS_CORE_VERSION_CHECK(0, 1, 1) > SNICHOLLS_CORE_VERSION_CHECK(0, 1, 0));
static_assert(SNICHOLLS_CORE_VERSION >= SNICHOLLS_CORE_VERSION_CHECK(0, 1, 0));

// The consumer-side guard compiles when the minimum is met, at namespace scope
SNICHOLLS_CORE_REQUIRE_VERSION(0, 1, 0);
SNICHOLLS_CORE_REQUIRE_VERSION(SNICHOLLS_CORE_VERSION_MAJOR, SNICHOLLS_CORE_VERSION_MINOR,
                               SNICHOLLS_CORE_VERSION_PATCH);

TEST(CoreVersion, StringIsMajorDotMinorDotPatch)
{
    const std::string expected = std::to_string(SNICHOLLS_CORE_VERSION_MAJOR) + "."
                               + std::to_string(SNICHOLLS_CORE_VERSION_MINOR) + "."
                               + std::to_string(SNICHOLLS_CORE_VERSION_PATCH);
    EXPECT_EQ(std::string{snicholls::core::version_string()}, expected);
    EXPECT_EQ(std::string{SNICHOLLS_CORE_VERSION_STRING}, expected);
}

TEST(CoreVersion, NumberEncodesTwoDigitsPerComponent)
{
    EXPECT_EQ(snicholls::core::version(),
              SNICHOLLS_CORE_VERSION_MAJOR * 10000 + SNICHOLLS_CORE_VERSION_MINOR * 100
                  + SNICHOLLS_CORE_VERSION_PATCH);
    EXPECT_EQ(SNICHOLLS_CORE_VERSION_CHECK(1, 2, 3), 10203);
    EXPECT_EQ(SNICHOLLS_CORE_VERSION_CHECK(0, 0, 0), 0);
}

TEST(CoreVersion, ComponentsStayWithinTwoDigits)
{
    // The encoding has room for 0 to 99 in minor and patch. If a release ever
    // needs more, change SNICHOLLS_CORE_VERSION_CHECK and this test together.
    EXPECT_LT(SNICHOLLS_CORE_VERSION_MINOR, 100);
    EXPECT_LT(SNICHOLLS_CORE_VERSION_PATCH, 100);
}

TEST(CoreVersion, RequireVersionCompilesInsideAFunctionToo)
{
    SNICHOLLS_CORE_REQUIRE_VERSION(0, 1, 0);
    SUCCEED();
}

} // namespace
