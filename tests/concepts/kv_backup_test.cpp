//
//  kv_backup_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/kv_backup.hpp>: which types model kv_backup (the reference, the interface,
//  the adapter) and which do not, at compile time; the reports' defaults.
//

#include <snicholls/core/concepts/kv_backup.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_backup.hpp>

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <type_traits>

using namespace snicholls;

namespace {

struct backs_up {
    result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view);
    result<kv_verify_report, kv_errc> verify() const;
};
struct checkpoint_returns_bool {
    bool checkpoint(std::string_view);
    result<kv_verify_report, kv_errc> verify() const;
};
struct verify_not_const {
    result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view);
    result<kv_verify_report, kv_errc> verify();
};
struct checkpoint_takes_an_int {
    result<kv_checkpoint_info, kv_errc> checkpoint(int);
    result<kv_verify_report, kv_errc> verify() const;
};
struct no_verify {
    result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view);
};

static_assert(kv_backup<backs_up>);
static_assert(kv_backup<reference_key_value_store>);
static_assert(kv_backup<kv_backup_interface>, "the virtual interface models the concept");
static_assert(kv_backup<kv_backup_adapter<reference_key_value_store>>);
static_assert(!kv_backup<checkpoint_returns_bool>, "a failure is a kv_errc, never a bool");
static_assert(!kv_backup<verify_not_const>, "a check changes nothing");
static_assert(!kv_backup<checkpoint_takes_an_int>, "the destination is text");
static_assert(!kv_backup<no_verify>);
static_assert(!kv_backup<const reference_key_value_store>, "a checkpoint needs a non-const store");

static_assert(kv_verify_report{}.ok() && !kv_verify_report{3, 1}.ok());
static_assert(std::is_aggregate_v<kv_checkpoint_info> && std::is_aggregate_v<kv_verify_report>);

} // namespace

TEST(KvBackupReports, DefaultToNothingCapturedAndNothingFound)
{
    const kv_checkpoint_info c;
    EXPECT_EQ(c.sequence, 0u);
    EXPECT_FALSE(c.bytes.has_value());
    const kv_verify_report v;
    EXPECT_EQ(v.keys_checked, 0u);
    EXPECT_TRUE(v.ok());
}
