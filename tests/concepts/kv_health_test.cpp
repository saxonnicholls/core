//
//  kv_health_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/kv_health.hpp>: which types model kv_health (the reference, the interface,
//  the adapter) and which do not, at compile time; kv_diagnostics' defaults.
//

#include <snicholls/core/concepts/kv_health.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_health.hpp>

#include <gtest/gtest.h>

#include <type_traits>

using namespace snicholls;

namespace {

struct healthy {
    result<void, kv_errc> ping() const;
    result<kv_diagnostics, kv_errc> diagnostics() const;
};
struct ping_returns_bool {
    bool ping() const;
    result<kv_diagnostics, kv_errc> diagnostics() const;
};
struct diagnostics_as_text {
    result<void, kv_errc> ping() const;
    result<const char*, kv_errc> diagnostics() const;
};
struct ping_not_const {
    result<void, kv_errc> ping();
    result<kv_diagnostics, kv_errc> diagnostics() const;
};
struct no_diagnostics {
    result<void, kv_errc> ping() const;
};

static_assert(kv_health<healthy>);
static_assert(kv_health<reference_key_value_store>);
static_assert(kv_health<kv_health_interface>, "the virtual interface models the concept");
static_assert(kv_health<kv_health_adapter<reference_key_value_store>>);
static_assert(!kv_health<ping_returns_bool>, "a failure is a kv_errc, never a bool");
static_assert(!kv_health<diagnostics_as_text>, "diagnostics are plain data, never text or JSON");
static_assert(!kv_health<ping_not_const>, "a probe never changes the store");
static_assert(!kv_health<no_diagnostics>);
static_assert(!kv_health<reference_kv_snapshot>, "health is a capability of its own");
static_assert(std::is_aggregate_v<kv_diagnostics>);

} // namespace

TEST(KvDiagnostics, DefaultsToAnEmptyVolatileStore)
{
    const kv_diagnostics d;
    EXPECT_TRUE(d.backend.empty());
    EXPECT_EQ(d.sequence, 0u);
    EXPECT_FALSE(d.keys.has_value());
    EXPECT_FALSE(d.bytes.has_value());
    EXPECT_FALSE(d.persistent);
    EXPECT_FALSE(d.read_only);
}
