//
//  encryptor_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/encryptor_conformance.hpp>: the kit passes the test double and catches broken
//  encryptors: one with a fixed nonce, one that ignores the associated data, one that returns unauthenticated data, one
//  that leaks the plaintext into its error.
//

#include <snicholls/core/interfaces/encryptor_conformance.hpp>

#include "contract_probes.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

enum class defect { fixed_nonce, ignores_aad, unauthenticated, leaks };

template <defect D>
struct broken_encryptor {
    insecure_test_encryptor inner = test::make_test_encryptor();
    capability_result<std::vector<std::byte>> encrypt(std::string_view k, std::span<const std::byte> p, std::span<const std::byte> a)
    {
        if (D == defect::fixed_nonce) {
            insecure_test_encryptor fresh = test::make_test_encryptor();
            return fresh.encrypt(k, p, D == defect::ignores_aad ? std::span<const std::byte>{} : a);
        }
        return inner.encrypt(k, p, D == defect::ignores_aad ? std::span<const std::byte>{} : a);
    }
    capability_result<std::vector<std::byte>> decrypt(std::span<const std::byte> s, std::span<const std::byte> a) const
    {
        auto r = inner.decrypt(s, D == defect::ignores_aad ? std::span<const std::byte>{} : a);
        if (D == defect::unauthenticated && r.is_err() && r.unwrap_err().code == capability_errc::signature_invalid)
            return ok(std::vector<std::byte>(s.begin(), s.end()));
        if (D == defect::leaks && r.is_err())
            return err(capability_error{r.unwrap_err().code, *provider_code::make(capability_redaction_marker), {}});
        return r;
    }
    std::string_view algorithm() const { return "broken"; }
};

template <defect D>
struct broken_probe {
    broken_encryptor<D> e;
    auto& encryptor() { return e; }
    static std::string_view key() { return "k1"; }
    static std::string_view other_key() { return "k2"; }
    void forget(std::string_view id) { e.inner.remove_key(id); }
};

TEST(EncryptorConformance, TheTestDoublePasses)
{
    const auto r = run_encryptor_conformance<test::encryptor_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(EncryptorConformance, AFixedNonceIsCaught)
{
    const auto r = run_encryptor_conformance<broken_probe<defect::fixed_nonce>>();
    EXPECT_TRUE(r.failed("every seal is fresh")) << test::describe(r);
}

TEST(EncryptorConformance, IgnoringTheAssociatedDataIsCaught)
{
    const auto r = run_encryptor_conformance<broken_probe<defect::ignores_aad>>();
    EXPECT_TRUE(r.failed("other associated data does not open")) << test::describe(r);
}

TEST(EncryptorConformance, UnauthenticatedDataIsCaught)
{
    const auto r = run_encryptor_conformance<broken_probe<defect::unauthenticated>>();
    EXPECT_TRUE(r.failed("a changed envelope does not open")) << test::describe(r);
}

TEST(EncryptorConformance, ALeakIsCaught)
{
    const auto r = run_encryptor_conformance<broken_probe<defect::leaks>>();
    EXPECT_TRUE(r.failed("errors carry no secret")) << test::describe(r);
}

} // namespace
