//
//  token_verifier_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/token_verifier_conformance.hpp>: the kit passes the reference and catches
//  broken implementations: one that echoes the token, one that leaks the provider's response, one that accepts
//  expired tokens.
//

#include <snicholls/core/interfaces/token_verifier_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

using namespace snicholls;

namespace {

enum class defect { echoes, leaks, ignores_expiry };

template <defect D>
struct broken_verifier {
    test::issued_tokens tokens;
    mutable bool armed = false;
    capability_result<token_claims> verify(std::string_view t) const
    {
        if (D == defect::leaks && armed) {
            armed = false;
            return err(capability_error{capability_errc::unavailable, *provider_code::make(capability_redaction_marker), {}});
        }
        auto r = tokens.verifier.verify(t);
        if (D == defect::echoes && r.is_err() && provider_code::is_code(t))
            return err(capability_error{r.unwrap_err().code, *provider_code::make(t), {}});
        if (D == defect::ignores_expiry && r.is_err() && r.unwrap_err().code == capability_errc::stale) {
            token_claims c;
            c.subject = test::probe_party;
            return ok(c);
        }
        return r;
    }
};

template <defect D>
struct broken_probe {
    broken_verifier<D> v;
    auto& verifier() { return v; }
    std::string valid_token() { return v.tokens.valid; }
    std::string expired_token() { return v.tokens.expired; }
    static uuid subject() { return test::probe_party; }
    void provider_fails(capability_errc c, std::string_view r)
    {
        v.armed = true;
        v.tokens.verifier.fail_next(c, r);
    }
};

TEST(TokenVerifierConformance, TheReferencePasses)
{
    const auto r = run_token_verifier_conformance<test::token_verifier_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
    EXPECT_EQ(r.checks, 6u);
}

TEST(TokenVerifierConformance, EchoingTheTokenIsCaught)
{
    const auto r = run_token_verifier_conformance<broken_probe<defect::echoes>>();
    EXPECT_TRUE(r.failed("a refused token is never echoed")) << test::describe(r);
}

TEST(TokenVerifierConformance, ALeakIsCaught)
{
    const auto r = run_token_verifier_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(TokenVerifierConformance, AcceptingAnExpiredTokenIsCaught)
{
    const auto r = run_token_verifier_conformance<broken_probe<defect::ignores_expiry>>();
    EXPECT_TRUE(r.failed("an expired token is stale")) << test::describe(r);
}

} // namespace
