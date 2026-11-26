//
//  role_store_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/role_store_reference.hpp>: grants persist through the key-value store (a new
//  role store over the same backend reads them back), a failing backend changes nothing and is reported without
//  detail, bounds are refused rather than cut, malformed rows are internal errors, an armed failure is returned once.
//

#include <snicholls/core/interfaces/role_store_reference.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <optional>
#include <string>

using namespace snicholls;

namespace {

const uuid alice = uuid::from_u64(0xa11ce, 1);
const uuid bob = uuid::from_u64(0xb0b, 2);
const utc_s t0 = utc_s::from_unix_seconds(1'800'000'000);

role_grant grant_of(const uuid& who, std::string_view tenant, std::string_view role)
{
    return role_grant{who, *tenant_id::make(tenant), *role_id::make(role), alice, "because", t0, std::nullopt};
}

TEST(MemoryRoleStore, GrantsAreRowsOfTheKeyValueStore)
{
    reference_key_value_store kv;
    {
        kv_role_store<reference_key_value_store&> roles{kv};
        auto g = grant_of(bob, "acme", "auditor");
        g.expires_at = t0 + std::chrono::hours{1};
        ASSERT_TRUE(roles.grant(g, "Grant auditor to bob").is_ok());
        ASSERT_TRUE(roles.grant(grant_of(bob, "acme", "member"), "").is_ok());
        EXPECT_EQ(&roles.backend(), &kv);
    }
    EXPECT_EQ(*kv_count(kv), 2u + 2u + 1u) << "two grants, two events, the sequence";
    const kv_role_store<reference_key_value_store&> again{kv};
    const auto g = again.grants_of(bob, "acme", t0);
    ASSERT_TRUE(g.is_ok());
    ASSERT_EQ(g->size(), 2u);
    EXPECT_EQ((*g)[0].role.view(), "auditor") << "ordered by role within a party";
    EXPECT_EQ((*g)[0].expires_at, t0 + std::chrono::hours{1});
    EXPECT_EQ(again.history("acme")->front().confirmation.view(), "Grant auditor to bob");
    kv_role_store<reference_key_value_store&> writer{kv};
    ASSERT_TRUE(writer.revoke(bob, "acme", "member", alice, "left", t0).is_ok());
    EXPECT_EQ(writer.history("acme")->size(), 3u) << "the sequence continues where the first store left it";
}

TEST(MemoryRoleStore, AFailingBackendChangesNothingAndSaysNoMore)
{
    memory_role_store roles;
    ASSERT_TRUE(roles.grant(grant_of(bob, "acme", "member"), "").is_ok());
    roles.backend().inject_failure(kv_errc::io, 1);   // the batch fails at its second write
    const auto r = roles.grant(grant_of(alice, "acme", "admin"), "");
    ASSERT_TRUE(r.is_err());
    EXPECT_EQ(r.unwrap_err().code, capability_errc::unavailable);
    EXPECT_TRUE(r.unwrap_err().provider_code.empty());
    EXPECT_TRUE(roles.grants_of(alice, "acme", t0)->empty()) << "all or nothing";
    EXPECT_EQ(roles.history("acme")->size(), 1u);
    roles.backend().set_unavailable(true);
    EXPECT_EQ(roles.members("acme", t0).unwrap_err().code, capability_errc::unavailable) << "an error, never an empty answer";
    EXPECT_EQ(roles.history("acme").unwrap_err().code, capability_errc::unavailable);
    EXPECT_EQ(roles.revoke(bob, "acme", "member", alice, "", t0).unwrap_err().code, capability_errc::unavailable);
    roles.backend().set_unavailable(false);
    roles.backend().inject_failure(kv_errc::conflict);
    EXPECT_EQ(roles.revoke(bob, "acme", "member", alice, "", t0).unwrap_err().code, capability_errc::conflict);
    roles.backend().inject_failure(kv_errc::corrupt);
    EXPECT_EQ(roles.grant(grant_of(bob, "acme", "member"), "").unwrap_err().code, capability_errc::internal);
}

TEST(MemoryRoleStore, BoundsAreRefusedNotCut)
{
    memory_role_store roles;
    EXPECT_EQ(roles.grant(grant_of(bob, "acme", "member"), std::string(256, 'c')).unwrap_err().code, capability_errc::invalid_argument);
    ASSERT_TRUE(roles.grant(grant_of(bob, "acme", "member"), std::string(255, 'c')).is_ok());
    EXPECT_EQ(roles.revoke(bob, "acme", "member", alice, std::string(256, 'r'), t0).unwrap_err().code, capability_errc::invalid_argument);
    const std::string long_tenant(65, 't');
    EXPECT_TRUE(roles.grants_of(bob, long_tenant, t0)->empty());
    EXPECT_TRUE(roles.members(long_tenant, t0)->empty());
    EXPECT_TRUE(roles.history(long_tenant)->empty());
    EXPECT_EQ(roles.revoke(bob, long_tenant, "member", alice, "", t0).unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(roles.revoke(bob, "acme", std::string(65, 'r'), alice, "", t0).unwrap_err().code, capability_errc::not_found);
}

TEST(MemoryRoleStore, AMalformedRowIsAnInternalError)
{
    memory_role_store roles;
    ASSERT_TRUE(roles.grant(grant_of(bob, "acme", "member"), "").is_ok());
    std::string key = "g";
    key.push_back(4);
    key += "acme";
    for (const auto b : bob.bytes)
        key.push_back(static_cast<char>(b));
    key += "member";
    ASSERT_TRUE(kv_put(roles.backend(), key, std::string_view{"short"}).is_ok());
    EXPECT_EQ(roles.grants_of(bob, "acme", t0).unwrap_err().code, capability_errc::internal);
    EXPECT_EQ(roles.revoke(bob, "acme", "member", alice, "", t0).unwrap_err().code, capability_errc::internal);
    std::string event_key = "e";
    event_key.push_back(4);
    event_key += "acme";
    event_key += std::string(8, '\x7f');
    ASSERT_TRUE(kv_put(roles.backend(), event_key, std::string_view{"x"}).is_ok());
    EXPECT_EQ(roles.history("acme").unwrap_err().code, capability_errc::internal);
    ASSERT_TRUE(kv_put(roles.backend(), std::string_view{"s"}, std::string_view{"x"}).is_ok());
    EXPECT_EQ(roles.grant(grant_of(alice, "globex", "member"), "").unwrap_err().code, capability_errc::internal);
}

TEST(MemoryRoleStore, AnArmedFailureIsReturnedOnce)
{
    memory_role_store roles;
    roles.fail_next(capability_errc::timeout, "pg_timeout {\"query\":\"never kept\"}");
    const auto e = roles.grant(grant_of(bob, "acme", "member"), "").unwrap_err();
    EXPECT_EQ(e.code, capability_errc::timeout);
    EXPECT_EQ(e.provider_code.view(), "pg_timeout");
    EXPECT_TRUE(roles.grant(grant_of(bob, "acme", "member"), "").is_ok());
    roles.fail_next(capability_errc::unavailable);
    EXPECT_EQ(roles.history("acme").unwrap_err().code, capability_errc::unavailable);
    roles.fail_next(capability_errc::unavailable);
    EXPECT_EQ(roles.revoke(bob, "acme", "member", alice, "", t0).unwrap_err().code, capability_errc::unavailable);
    EXPECT_TRUE(roles.revoke(bob, "acme", "member", alice, "", t0).is_ok());
}

} // namespace
