//
//  role_store_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/role_store.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/role_store.hpp>
#include <snicholls/core/interfaces/role_store_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(role_store<role_store_interface>);
static_assert(role_store<role_store_ref>);
static_assert(std::is_abstract_v<role_store_interface>);

TEST(RoleStoreInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_role_store_conformance<test::role_store_probe<test::form::concept_model>>();
    const auto virtual_form = run_role_store_conformance<test::role_store_probe<test::form::interface>>();
    const auto handle = run_role_store_conformance<test::role_store_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 6u);
    EXPECT_EQ(virtual_form.checks, 6u);
    EXPECT_EQ(handle.checks, 6u);
}

struct counting_store {
    mutable int calls = 0;
    capability_result<std::vector<role_grant>> grants_of(const uuid&, std::string_view, utc_s) const { ++calls; return ok(std::vector<role_grant>{}); }
    capability_result<std::vector<role_grant>> members(std::string_view, utc_s) const { ++calls; return ok(std::vector<role_grant>{}); }
    capability_result<void> grant(const role_grant&, std::string_view) { ++calls; return ok(); }
    capability_result<void> revoke(const uuid&, std::string_view, std::string_view, const uuid&, std::string_view, utc_s)
    {
        ++calls;
        return fail(capability_errc::not_found);
    }
    capability_result<std::vector<role_event>> history(std::string_view) const { ++calls; return fail(capability_errc::unavailable); }
};
static_assert(role_store<role_store_adapter<counting_store>>);

TEST(RoleStoreInterface, TheAdaptersForwardEveryOperation)
{
    counting_store impl;
    role_store_adapter<counting_store&> borrowed{impl};
    role_store_ref s{borrowed};
    EXPECT_TRUE(s.grants_of({}, "t", {}).is_ok());
    EXPECT_TRUE(s.members("t", {}).is_ok());
    EXPECT_TRUE(s.grant({}, "").is_ok());
    EXPECT_EQ(s.revoke({}, "t", "r", {}, "", {}).unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(s.history("t").unwrap_err().code, capability_errc::unavailable);
    EXPECT_EQ(impl.calls, 5);
    EXPECT_EQ(&borrowed.get(), &impl);
    role_store_adapter owned{memory_role_store{}};
    EXPECT_TRUE(owned.history("t")->empty());
}

} // namespace
