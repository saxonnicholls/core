//
//  key_store_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/key_store.hpp>: the models and the non-models, the value types.
//

#include <snicholls/core/concepts/key_store.hpp>
#include <snicholls/core/interfaces/key_store_reference.hpp>

#include <gtest/gtest.h>

#include <string_view>
#include <vector>

using namespace snicholls;

namespace {

static_assert(key_store<memory_key_store>);
static_assert(key_store<key_store_interface>);
static_assert(key_store<key_store_adapter<memory_key_store>>);
static_assert(key_store<key_store_adapter<memory_key_store&>>);
static_assert(key_store<key_store_ref>);

struct no_list {
    capability_result<key_info> create(const key_spec&);
    capability_result<key_info> rotate(std::string_view);
    capability_result<void> retire(std::string_view);
    capability_result<void> destroy(std::string_view);
    capability_result<key_info> info(std::string_view) const;
};
struct exports_nothing_but_destroy_returns_bool {
    capability_result<key_info> create(const key_spec&);
    capability_result<key_info> rotate(std::string_view);
    capability_result<void> retire(std::string_view);
    bool destroy(std::string_view);
    capability_result<key_info> info(std::string_view) const;
    capability_result<std::vector<key_info>> list() const;
};
static_assert(!key_store<no_list>);
static_assert(!key_store<exports_nothing_but_destroy_returns_bool>);
static_assert(!key_store<int>);

static_assert(key_spec{signature_algorithm::hs256, "a"} == key_spec{signature_algorithm::hs256, "a"});
static_assert(key_info{} == key_info{});

TEST(KeyStoreConcept, KeyInfoComparesByEveryMember)
{
    key_info a{"key-1", 1, signature_algorithm::hs256, key_state::active, "jwt", utc_s::from_unix_seconds(1)};
    key_info b = a;
    EXPECT_EQ(a, b);
    b.state = key_state::retired;
    EXPECT_NE(a, b);
}

} // namespace
