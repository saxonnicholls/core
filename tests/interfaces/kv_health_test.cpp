//
//  kv_health_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/kv_health.hpp>: the health kit through the concept and through the
//  virtual interface, the adapter's forwarding and exception translation, and asking a store for its health with
//  dynamic_cast.
//

#include <snicholls/core/interfaces/key_value_store_conformance.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_health.hpp>

#include "kv_virtual_backend.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>

using namespace snicholls;

TEST(KvHealthConformance, ReferencePassesThroughTheConcept)
{
    const auto report = run_kv_health_conformance([] { return reference_key_value_store{}; });
    EXPECT_TRUE(report.ok());
    EXPECT_EQ(report.checks, 2u);
}

TEST(KvHealthConformance, ReferencePassesThroughTheVirtualInterface)
{
    const auto report = run_kv_health_conformance([] { return std::make_unique<kv_test::kv_virtual_backend>(); });
    EXPECT_TRUE(report.ok());
}

TEST(KvHealthAdapter, ForwardsToTheModel)
{
    reference_key_value_store store;
    static_cast<void>(kv_put(store, "k", "v"));
    const kv_health_adapter<reference_key_value_store&> adapter{store};
    const kv_health_interface& health = adapter;
    EXPECT_TRUE(health.ping().is_ok());
    EXPECT_EQ(health.diagnostics()->keys, std::optional<std::uint64_t>(1));
    store.set_unavailable(true);
    EXPECT_EQ(health.ping().unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(&adapter.get(), &store);
    const kv_health_adapter owning{reference_key_value_store{}};
    EXPECT_EQ(owning.diagnostics()->sequence, 0u);
}

TEST(KvHealthAdapter, TranslatesExceptionsOfTheModel)
{
    struct throwing {
        result<void, kv_errc> ping() const { throw std::runtime_error("driver"); }
        result<kv_diagnostics, kv_errc> diagnostics() const { throw std::bad_alloc(); }
    };
    const kv_health_adapter<throwing> adapter{throwing{}};
    EXPECT_EQ(adapter.ping().unwrap_err(), kv_errc::backend);
    EXPECT_EQ(adapter.diagnostics().unwrap_err(), kv_errc::out_of_memory);
}

TEST(KvHealthInterface, IsFoundFromTheStoreByDynamicCast)
{
    kv_test::kv_virtual_backend backend;
    const key_value_store_interface& store = backend;
    const auto* health = dynamic_cast<const kv_health_interface*>(&store);
    ASSERT_NE(health, nullptr);
    EXPECT_TRUE(health->ping().is_ok());
    const key_value_store_adapter plain{reference_key_value_store{}};
    EXPECT_EQ(dynamic_cast<const kv_health_interface*>(static_cast<const key_value_store_interface*>(&plain)), nullptr)
        << "a store that does not offer health says so";
}
