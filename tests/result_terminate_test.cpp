//
//  result_terminate_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/result.hpp>.
//

// snicholls::result under SNICHOLLS_RESULT_THROW=0: unwrap / expect on an error print a message and terminate instead of
// throwing, and are noexcept. Built in the same test binary as result_test.cpp (which uses the default policy): the
// two policies live in different inline namespaces, so this is not an ODR violation.

#define SNICHOLLS_RESULT_THROW 0
#include <snicholls/core/result.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <type_traits>

namespace {

using text_result = snicholls::result<int, std::string>;

static_assert(std::is_same_v<text_result, snicholls::result_terminates::result<int, std::string>>);
static_assert(noexcept(std::declval<text_result&>().unwrap()));
static_assert(noexcept(std::declval<text_result&>().expect(std::declval<std::string_view>())));
static_assert(noexcept(std::declval<text_result&>().unwrap_err()));

TEST(ResultTerminatePolicy, SuccessPathIsUnchanged)
{
    text_result r = snicholls::ok(4);
    EXPECT_EQ(r.unwrap(), 4);
    EXPECT_EQ(r.expect("four"), 4);
    text_result e = snicholls::err(std::string("no"));
    EXPECT_EQ(e.unwrap_err(), "no");
}

TEST(ResultTerminatePolicyDeathTest, UnwrapOfAnErrorTerminatesWithTheMessage)
{
    EXPECT_DEATH((void)text_result(snicholls::err(std::string("disk full"))).unwrap(), "called unwrap\\(\\) on an err: disk full");
    EXPECT_DEATH((void)text_result(snicholls::err(std::string("disk full"))).expect("saving"), "saving: disk full");
    EXPECT_DEATH((void)text_result(snicholls::ok(1)).unwrap_err(), "called unwrap_err\\(\\) on an ok");
    EXPECT_DEATH((snicholls::result<void, std::string>(snicholls::err(std::string("x"))).unwrap()), "on an err: x");
}

} // namespace
