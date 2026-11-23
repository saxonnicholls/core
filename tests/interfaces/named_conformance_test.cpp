//
//  named_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/named_conformance.hpp>: the kit catches each broken rule.
//

#include <snicholls/core/interfaces/named.hpp>
#include <snicholls/core/interfaces/named_conformance.hpp>

#include <gtest/gtest.h>

namespace {

using namespace snicholls;

struct good {
    static constexpr std::string_view name() noexcept { return "good"; }
};
struct empty_name {
    static constexpr std::string_view name() noexcept { return ""; }
};
struct unstable {
    mutable int calls = 0;
    std::string_view name() const noexcept { return ++calls == 1 ? "first" : "second"; }
};

static_assert(!check_named(good{}));
static_assert(check_named(empty_name{}).has_value());

TEST(NamedConformance, NamesTheFailedCheck)
{
    EXPECT_FALSE(check_named(good{}));
    EXPECT_EQ(*check_named(empty_name{}), "named: name() is empty");
    EXPECT_EQ(*check_named(unstable{}), "named: name() changes between calls");
}

} // namespace
