//
//  kv_cas_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/kv_cas.hpp>: which types model kv_cas and which do not, and kv_compare_and_swap
//  over text keys on the reference store.
//

#include <snicholls/core/concepts/kv_cas.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_cas.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <string_view>

using namespace snicholls;

namespace {

static_assert(kv_cas<reference_key_value_store>);
static_assert(kv_cas<kv_cas_interface>);
static_assert(kv_cas<kv_cas_adapter<reference_key_value_store>>);
static_assert(kv_cas<kv_cas_adapter<reference_key_value_store&>>);

struct returns_bool {
    bool compare_and_swap(kv_bytes, std::optional<kv_bytes>, std::optional<kv_bytes>);
};
struct no_erase {
    result<bool, kv_errc> compare_and_swap(kv_bytes, kv_bytes, kv_bytes);
};
static_assert(!kv_cas<returns_bool>);
static_assert(!kv_cas<no_erase>);
static_assert(!kv_cas<int>);

kv_bytes b(std::string_view s) { return {reinterpret_cast<const std::byte*>(s.data()), s.size()}; }

TEST(KvCasConcept, SwapsOnlyFromTheExpectedValue)
{
    reference_key_value_store s;
    const std::uint64_t before = *s.sequence();
    EXPECT_TRUE(*kv_compare_and_swap(s, std::string_view{"lease"}, std::nullopt, b("a")));
    EXPECT_FALSE(*kv_compare_and_swap(s, std::string_view{"lease"}, std::nullopt, b("b")));
    EXPECT_EQ(*s.sequence(), before + 1);   // a refused swap does not advance the sequence
    EXPECT_TRUE(*kv_compare_and_swap(s, std::string_view{"lease"}, b("a"), b("b")));
    EXPECT_TRUE(*kv_compare_and_swap(s, std::string_view{"lease"}, b("b"), std::nullopt));
    EXPECT_FALSE(*s.contains(b("lease")));
    EXPECT_FALSE(*kv_compare_and_swap(s, std::string_view{"lease"}, b("b"), std::nullopt));
}

TEST(KvCasConcept, AnUnavailableStoreIsAnError)
{
    reference_key_value_store s;
    s.set_unavailable(true);
    EXPECT_EQ(s.compare_and_swap(b("k"), std::nullopt, b("v")).unwrap_err(), kv_errc::unavailable);
}

} // namespace
