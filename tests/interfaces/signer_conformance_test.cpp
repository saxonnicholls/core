//
//  signer_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/signer_conformance.hpp>: the kit passes the reference and catches broken
//  implementations: one that leaks the KMS's response and one that signs with retired keys.
//

#include <snicholls/core/interfaces/signer_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

// The redaction rule's opposite: an implementation that copies the provider's response into its error.
capability_error leak(capability_errc code)
{
    return capability_error{code, *provider_code::make(capability_redaction_marker), {}};
}

struct broken_signer {
    memory_signer inner = memory_signer::deterministic();
    bool leaks = false;
    bool signs_retired = false;
    bool armed = false;
    capability_result<signature> sign(const sign_request& r)
    {
        if (armed && leaks) {
            armed = false;
            return err(leak(capability_errc::unavailable));
        }
        auto s = inner.sign(r);
        if (signs_retired && s.is_err() && s.unwrap_err().code == capability_errc::conflict)
            return ok(signature{signature_algorithm::hs256, *provider_id::make(r.key_id), 1, {std::byte{1}}});
        return s;
    }
    [[nodiscard]] capability_result<public_key> public_key_of(const key_ref& k) const { return inner.public_key_of(k); }
};

template <bool Leaks, bool SignsRetired>
struct broken_probe {
    broken_signer s{memory_signer::deterministic(), Leaks, SignsRetired};
    auto& signer() { return s; }
    std::string active_key() { return std::string(s.inner.create({signature_algorithm::hs256, "jwt"})->key_id.view()); }
    std::string retired_key()
    {
        std::string id = active_key();
        static_cast<void>(s.inner.retire(id));
        return id;
    }
    bool verify(const signature& sig, std::span<const std::byte> p) { return s.inner.verify(sig, p); }
    static bool deterministic() { return true; }
    void provider_fails(capability_errc c, std::string_view r)
    {
        s.armed = true;
        s.inner.fail_next(c, r);
    }
};

TEST(SignerConformance, TheReferencePasses)
{
    const auto r = run_signer_conformance<test::signer_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(SignerConformance, ALeakIsCaught)
{
    const auto r = run_signer_conformance<broken_probe<true, false>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(SignerConformance, SigningWithARetiredKeyIsCaught)
{
    const auto r = run_signer_conformance<broken_probe<false, true>>();
    EXPECT_TRUE(r.failed("a retired key no longer signs")) << test::describe(r);
}

} // namespace
