//
//  kv_backup_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/kv_backup.hpp>: a guard that checkpoints a store through
//  kv_backup_interface before a risky migration, fed by kv_backup_adapter, with a model's exception turned into
//  an error. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_backup.hpp>

#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

//! [guard]
// Before a migration: the data must verify, and a checkpoint must exist to go back to.
bool checkpoint_before_migration(kv_backup_interface& backup, std::string_view name)
{
    const auto check = backup.verify();
    if (check.is_err() || !check->ok())
        return false;
    const auto taken = backup.checkpoint(name);
    std::printf("  %.*s: %s\n", static_cast<int>(name.size()), name.data(),
                taken.is_ok() ? "checkpoint taken" : "no checkpoint, migration refused");
    return taken.is_ok();
}
//! [guard]

struct full_disk {   // a backend whose checkpoint throws
    result<kv_checkpoint_info, kv_errc> checkpoint(std::string_view) { throw std::runtime_error("ENOSPC"); }
    result<kv_verify_report, kv_errc> verify() const { return ok(kv_verify_report{}); }
};

} // namespace

int main()
{
    std::printf("snicholls-core kv_backup_interface demo\n");
    bool ok = true;
    reference_key_value_store store;
    static_cast<void>(kv_put(store, "schema", "v1"));

    kv_backup_adapter<reference_key_value_store&> backup{store};
    ok &= checkpoint_before_migration(backup, "pre-v2");
    static_cast<void>(kv_put(store, "schema", "v2"));   // the migration
    ok &= !checkpoint_before_migration(backup, "pre-v2");   // the name is taken
    ok &= *kv_get(*backup.get().open_checkpoint("pre-v2"), "schema") == std::optional<std::string>("v1");

    kv_backup_adapter disk{full_disk{}};
    ok &= !checkpoint_before_migration(disk, "pre-v3");
    ok &= disk.checkpoint("x").unwrap_err() == kv_errc::backend && disk.verify().is_ok();

    std::printf(ok ? "kv_backup_interface demo: ok\n" : "kv_backup_interface demo: FAILED\n");
    return ok ? 0 : 1;
}
