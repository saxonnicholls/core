//
//  key_store_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/key_store.hpp>: the conformance kit passes on the reference in every form (the
//  concept, the virtual interface over a borrowing adapter, the handle), and the adapters forward every operation.
//

#include <snicholls/core/interfaces/key_store.hpp>
#include <snicholls/core/interfaces/key_store_conformance.hpp>

#include "capability_probes.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

static_assert(key_store<key_store_interface>);
static_assert(key_store<key_store_ref>);
static_assert(std::is_abstract_v<key_store_interface>);

TEST(KeyStoreInterface, TheKitPassesInEveryForm)
{
    const auto direct = run_key_store_conformance<test::key_store_probe<test::form::concept_model>>();
    const auto virtual_form = run_key_store_conformance<test::key_store_probe<test::form::interface>>();
    const auto handle = run_key_store_conformance<test::key_store_probe<test::form::ref>>();
    EXPECT_TRUE(direct.ok()) << test::describe(direct);
    EXPECT_TRUE(virtual_form.ok()) << test::describe(virtual_form);
    EXPECT_TRUE(handle.ok()) << test::describe(handle);
    EXPECT_EQ(direct.checks, 8u);
    EXPECT_EQ(virtual_form.checks, 8u);
    EXPECT_EQ(handle.checks, 8u);
}

struct counting_store {
    mutable int calls = 0;
    capability_result<key_info> create(const key_spec&) { ++calls; return fail(capability_errc::unsupported); }
    capability_result<key_info> rotate(std::string_view) { ++calls; return fail(capability_errc::not_found); }
    capability_result<void> retire(std::string_view) { ++calls; return fail(capability_errc::not_found); }
    capability_result<void> destroy(std::string_view) { ++calls; return fail(capability_errc::not_found); }
    [[nodiscard]] capability_result<key_info> info(std::string_view) const { ++calls; return fail(capability_errc::not_found); }
    [[nodiscard]] capability_result<std::vector<key_info>> list() const { ++calls; return ok(std::vector<key_info>{}); }
};
static_assert(key_store<key_store_adapter<counting_store>>);

TEST(KeyStoreInterface, TheAdaptersForwardEveryOperation)
{
    counting_store impl;
    key_store_adapter<counting_store&> borrowed{impl};
    key_store_ref k{borrowed};
    EXPECT_EQ(k.create({}).unwrap_err().code, capability_errc::unsupported);
    EXPECT_EQ(k.rotate("x").unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(k.retire("x").unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(k.destroy("x").unwrap_err().code, capability_errc::not_found);
    EXPECT_EQ(k.info("x").unwrap_err().code, capability_errc::not_found);
    EXPECT_TRUE(k.list()->empty());
    EXPECT_EQ(impl.calls, 6);
    key_store_adapter owned{memory_key_store::deterministic()};
    EXPECT_TRUE(owned.create({signature_algorithm::hs256, "x"}).is_ok());
    EXPECT_EQ(owned.get().list()->size(), 1u);
}

} // namespace
