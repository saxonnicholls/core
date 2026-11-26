//
//  role_store_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/role_store_conformance.hpp>: the kit passes the reference and catches broken
//  implementations: one that leaks the database's response, one that ignores the tenant, one that keeps expired
//  grants.
//

#include <snicholls/core/interfaces/role_store_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

enum class defect { leaks, ignores_the_tenant, keeps_expired };

template <defect D>
struct broken_store {
    memory_role_store inner;
    bool armed = false;
    capability_result<std::vector<role_grant>> grants_of(const uuid& p, std::string_view t, utc_s now) const
    {
        if (D == defect::keeps_expired)
            now = utc_s{};
        if (D != defect::ignores_the_tenant)
            return inner.grants_of(p, t, now);
        std::vector<role_grant> all;
        for (const std::string_view tenant : {std::string_view{"acme"}, platform_tenant}) {
            const auto some = inner.grants_of(p, tenant, now);
            all.insert(all.end(), some->begin(), some->end());
        }
        return ok(std::move(all));
    }
    capability_result<std::vector<role_grant>> members(std::string_view t, utc_s now) const
    {
        return inner.members(t, D == defect::keeps_expired ? utc_s{} : now);
    }
    capability_result<void> grant(const role_grant& g, std::string_view c)
    {
        if (D == defect::leaks && armed) {
            armed = false;
            return err(capability_error{capability_errc::unavailable, *provider_code::make(capability_redaction_marker), {}});
        }
        return inner.grant(g, c);
    }
    capability_result<void> revoke(const uuid& p, std::string_view t, std::string_view r, const uuid& a, std::string_view why, utc_s now)
    {
        return inner.revoke(p, t, r, a, why, now);
    }
    capability_result<std::vector<role_event>> history(std::string_view t) const { return inner.history(t); }
};

template <defect D>
struct broken_probe {
    broken_store<D> s;
    auto& store() { return s; }
    void provider_fails(capability_errc c, std::string_view r)
    {
        s.armed = true;
        if (D != defect::leaks)   // the leaking store answers for itself
            s.inner.fail_next(c, r);
    }
};

TEST(RoleStoreConformance, TheReferencePasses)
{
    const auto r = run_role_store_conformance<test::role_store_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
    EXPECT_EQ(r.checks, 6u);
}

TEST(RoleStoreConformance, ALeakIsCaught)
{
    const auto r = run_role_store_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(RoleStoreConformance, AStaffRoleShowingInACustomersTenantIsCaught)
{
    const auto r = run_role_store_conformance<broken_probe<defect::ignores_the_tenant>>();
    EXPECT_TRUE(r.failed("a grant is active in its tenant only")) << test::describe(r);
}

TEST(RoleStoreConformance, KeepingAnExpiredGrantIsCaught)
{
    const auto r = run_role_store_conformance<broken_probe<defect::keeps_expired>>();
    EXPECT_TRUE(r.failed("an expired grant is inactive")) << test::describe(r);
}

} // namespace
