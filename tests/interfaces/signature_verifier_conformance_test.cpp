//
//  signature_verifier_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/signature_verifier_conformance.hpp>: the kit passes the reference and catches
//  broken implementations: one that leaks the provider's response, one that accepts a prefix of the tag, one that
//  answers ok for an algorithm it cannot check.
//

#include <snicholls/core/interfaces/signature_verifier_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>

using namespace snicholls;

namespace {

enum class defect { leaks, accepts_a_prefix, accepts_anything_unknown };

template <defect D>
struct broken_verifier {
    memory_signature_verifier inner;
    mutable bool armed = false;
    capability_result<void> verify(const verify_request& r) const
    {
        if (D == defect::leaks && armed) {
            armed = false;
            return err(capability_error{capability_errc::unavailable, *provider_code::make(capability_redaction_marker), {}});
        }
        if (D == defect::accepts_anything_unknown && r.algorithm != signature_algorithm::hs256)
            return ok();
        if (D == defect::accepts_a_prefix && r.signature.size() < 32)
            return ok();   // compares only the bytes it was given
        return inner.verify(r);
    }
};

template <defect D>
struct broken_probe {
    test::hs256_keys keys;
    broken_verifier<D> v{memory_signature_verifier{keys.store}};
    auto& verifier() { return v; }
    signed_sample sign(std::span<const std::byte> m) { return keys.sign(m); }
    static signature_algorithm unsupported() { return signature_algorithm::es256; }
    void provider_fails(capability_errc c, std::string_view r)
    {
        v.armed = true;
        v.inner.fail_next(c, r);
    }
};

TEST(SignatureVerifierConformance, TheReferencePasses)
{
    const auto r = run_signature_verifier_conformance<test::signature_verifier_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
    EXPECT_EQ(r.checks, 6u);
}

TEST(SignatureVerifierConformance, ALeakIsCaught)
{
    const auto r = run_signature_verifier_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(SignatureVerifierConformance, AcceptingATruncatedTagIsCaught)
{
    const auto r = run_signature_verifier_conformance<broken_probe<defect::accepts_a_prefix>>();
    EXPECT_TRUE(r.failed("a truncated signature never verifies")) << test::describe(r);
}

TEST(SignatureVerifierConformance, AcceptingAnUnknownAlgorithmIsCaught)
{
    const auto r = run_signature_verifier_conformance<broken_probe<defect::accepts_anything_unknown>>();
    EXPECT_TRUE(r.failed("an algorithm it cannot check is unsupported")) << test::describe(r);
}

} // namespace
