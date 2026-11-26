//
//  role_store_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/role_store.hpp>: the models and the non-models, a grant's activity window, the
//  value types' layout.
//

#include <snicholls/core/concepts/role_store.hpp>
#include <snicholls/core/interfaces/role_store_reference.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <optional>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(role_store<memory_role_store>);
static_assert(role_store<kv_role_store<reference_key_value_store&>>);
static_assert(role_store<role_store_interface>);
static_assert(role_store<role_store_adapter<memory_role_store>>);
static_assert(role_store<role_store_adapter<memory_role_store&>>);
static_assert(role_store<role_store_ref>);

struct reads_not_const {
    capability_result<std::vector<role_grant>> grants_of(const uuid&, std::string_view, utc_s);
    capability_result<std::vector<role_grant>> members(std::string_view, utc_s);
    capability_result<void> grant(const role_grant&, std::string_view);
    capability_result<void> revoke(const uuid&, std::string_view, std::string_view, const uuid&, std::string_view, utc_s);
    capability_result<std::vector<role_event>> history(std::string_view);
};
struct reads_without_errors {
    std::vector<role_grant> grants_of(const uuid&, std::string_view, utc_s) const;
    std::vector<role_grant> members(std::string_view, utc_s) const;
    capability_result<void> grant(const role_grant&, std::string_view);
    capability_result<void> revoke(const uuid&, std::string_view, std::string_view, const uuid&, std::string_view, utc_s);
    std::vector<role_event> history(std::string_view) const;
};
static_assert(!role_store<reads_not_const>);
static_assert(!role_store<reads_without_errors>, "a failed read is an error, never an empty answer");
static_assert(!role_store<int>);

static_assert(std::is_trivially_copyable_v<role_grant> && std::is_trivially_copyable_v<role_event>);
static_assert(platform_tenant == "platform");

constexpr utc_s t0 = utc_s::from_unix_seconds(1000);
static_assert(role_grant{}.active(t0), "no expiry: always active");
static_assert(role_grant{.expires_at = t0 + std::chrono::seconds{1}}.active(t0));
static_assert(!role_grant{.expires_at = t0}.active(t0), "expiry at the instant: inactive");

TEST(RoleStoreConcept, GrantsAndEventsCompareByEveryMember)
{
    role_grant a{uuid::from_u64(1, 1), "acme", "admin", uuid::from_u64(2, 2), "why", t0, std::nullopt};
    role_grant b = a;
    EXPECT_EQ(a, b);
    b.role = "member";
    EXPECT_NE(a, b);
    role_event e{true, a, "confirmed"};
    role_event f = e;
    EXPECT_EQ(e, f);
    f.granted = false;
    EXPECT_NE(e, f);
}

} // namespace
