//
//  key_store_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/key_store_conformance.hpp>: the kit passes the reference and catches broken
//  implementations: one that leaks the KMS's response and one that reuses destroyed ids.
//

#include <snicholls/core/interfaces/key_store_conformance.hpp>

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

struct broken_store {
    memory_key_store inner = memory_key_store::deterministic();
    bool leaks = false;
    bool reuses_ids = false;
    bool armed = false;
    capability_result<key_info> create(const key_spec& spec)
    {
        if (armed && leaks) {
            armed = false;
            return err(leak(capability_errc::unavailable));
        }
        auto made = inner.create(spec);
        if (reuses_ids && made.is_ok())
            made->key_id = "key-1";   // the reference's first id, every time
        return made;
    }
    capability_result<key_info> rotate(std::string_view id) { return inner.rotate(id); }
    capability_result<void> retire(std::string_view id) { return inner.retire(id); }
    capability_result<void> destroy(std::string_view id) { return inner.destroy(id); }
    [[nodiscard]] capability_result<key_info> info(std::string_view id) const { return inner.info(id); }
    [[nodiscard]] capability_result<std::vector<key_info>> list() const { return inner.list(); }
};

template <bool Leaks, bool Reuses>
struct broken_probe {
    broken_store s{memory_key_store::deterministic(), Leaks, Reuses};
    auto& store() { return s; }
    static key_spec supported() { return {signature_algorithm::hs256, "jwt"}; }
    static std::optional<key_spec> unsupported() { return std::nullopt; }
    void provider_fails(capability_errc c, std::string_view r)
    {
        s.armed = true;
        s.inner.fail_next(c, r);
    }
};

TEST(KeyStoreConformance, TheReferencePasses)
{
    const auto r = run_key_store_conformance<test::key_store_probe<test::form::concept_model>>();
    EXPECT_TRUE(r.ok()) << test::describe(r);
}

TEST(KeyStoreConformance, ALeakIsCaught)
{
    const auto r = run_key_store_conformance<broken_probe<true, false>>();
    EXPECT_TRUE(r.failed("a provider failure never leaks its response")) << test::describe(r);
    EXPECT_EQ(r.failures.size(), 1u) << test::describe(r);
}

TEST(KeyStoreConformance, AReusedIdIsCaught)
{
    const auto r = run_key_store_conformance<broken_probe<false, true>>();
    EXPECT_TRUE(r.failed("destroy removes every version and the id is never reused")) << test::describe(r);
}

} // namespace
