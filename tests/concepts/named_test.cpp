//
//  named_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/named.hpp>: who models the concept, and who does not, at compile time.
//

#include <snicholls/core/concepts/named.hpp>
#include <snicholls/core/interfaces/named.hpp>

#include <gtest/gtest.h>

#include <string>

namespace {

using namespace snicholls;

struct static_name {
    static constexpr std::string_view name() noexcept { return "static"; }
};
struct member_name {
    std::string n;
    std::string_view name() const { return n; }
};
struct no_name {};
struct wrong_type {
    static constexpr int name() noexcept { return 1; }
};
struct non_const {
    std::string_view name() { return "x"; }
};

static_assert(has_name<static_name>);
static_assert(has_name<member_name>);
static_assert(has_name<named_interface>);
static_assert(has_name<named_model<static_name>>);
static_assert(!has_name<no_name>);
static_assert(!has_name<wrong_type>);
static_assert(!has_name<non_const>, "name() must be callable on a const object");
static_assert(!has_name<int>);

TEST(NamedConcept, ModelsAnswerTheirName)
{
    EXPECT_EQ(static_name{}.name(), "static");
    EXPECT_EQ(member_name{"run time"}.name(), "run time");
}

} // namespace
