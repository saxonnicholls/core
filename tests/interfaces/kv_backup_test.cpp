//
//  kv_backup_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/kv_backup.hpp>: the backup kit through the concept and through the
//  virtual interface, the adapter's forwarding and exception translation.
//

#include <snicholls/core/interfaces/key_value_store_conformance.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_backup.hpp>

#include "kv_virtual_backend.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <utility>

using namespace snicholls;

TEST(KvBackupConformance, ReferencePassesThroughTheConcept)
{
    const auto report = run_kv_backup_conformance([] { return reference_key_value_store{}; }, "checkpoint-1");
    EXPECT_TRUE(report.ok());
    EXPECT_EQ(report.checks, 2u);
}

TEST(KvBackupConformance, ReferencePassesThroughTheVirtualInterface)
{
    const auto report = run_kv_backup_conformance([] { return std::make_unique<kv_test::kv_virtual_backend>(); }, "checkpoint-1");
    EXPECT_TRUE(report.ok());
}

TEST(KvBackupAdapter, ForwardsToTheModel)
{
    reference_key_value_store store;
    static_cast<void>(kv_put(store, "k", "v"));
    kv_backup_adapter<reference_key_value_store&> adapter{store};
    kv_backup_interface& backup = adapter;
    EXPECT_EQ(backup.checkpoint("nightly")->sequence, 1u);
    EXPECT_TRUE(store.open_checkpoint("nightly").has_value());
    EXPECT_EQ(backup.verify()->keys_checked, 1u);
    EXPECT_EQ(&adapter.get(), &store);
    EXPECT_EQ(&std::as_const(adapter).get(), &store);
    kv_backup_adapter owning{reference_key_value_store{}};
    EXPECT_TRUE(owning.checkpoint("a").is_ok());
}

TEST(KvBackupAdapter, TranslatesExceptionsOfTheModel)
{
    struct throwing {
        result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view) { throw std::runtime_error("disk"); }
        result<kv_verify_report, kv_errc> verify() const { throw std::bad_alloc(); }
    };
    kv_backup_adapter<throwing> adapter{throwing{}};
    EXPECT_EQ(adapter.checkpoint("x").unwrap_err(), kv_errc::backend);
    EXPECT_EQ(adapter.verify().unwrap_err(), kv_errc::out_of_memory);
}
