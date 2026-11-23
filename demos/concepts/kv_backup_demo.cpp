//
//  kv_backup_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/kv_backup.hpp>: a nightly job written against the concept - verify, then
//  checkpoint, recording the sequence beside the digest - run on the reference store, which keeps its
//  checkpoints in memory. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/key_value_store.hpp>
#include <snicholls/core/concepts/kv_backup.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <cstdio>
#include <optional>
#include <string>
#include <string_view>

using namespace snicholls;

//! [backup]
// Verify first, then checkpoint; record (sequence, digest) so a later run can prove it reads the same state.
template <typename S>
    requires kv_backup<S> && key_value_store<S>
bool nightly(S& store, std::string_view destination)
{
    const auto check = store.verify();
    if (check.is_err() || !check->ok())
        return false;
    const auto taken = store.checkpoint(destination);
    if (taken.is_err())
        return false;
    const auto digest = kv_digest_v1(store);
    std::printf("  %.*s: %llu keys verified, checkpoint at sequence %llu (%llu bytes), digest %s\n",
                static_cast<int>(destination.size()), destination.data(),
                static_cast<unsigned long long>(check->keys_checked), static_cast<unsigned long long>(taken->sequence),
                static_cast<unsigned long long>(taken->bytes.value_or(0)),
                digest.is_ok() ? std::string(digest->hex().view()).c_str() : "-");
    return true;
}
//! [backup]

int main()
{
    std::printf("snicholls-core kv_backup demo\n");
    bool ok = true;
    reference_key_value_store store;
    static_cast<void>(kv_put(store, "position/AAPL", "100"));
    ok &= nightly(store, "2026-11-23");
    static_cast<void>(kv_put(store, "position/AAPL", "150"));

    ok &= !nightly(store, "2026-11-23");   // the destination exists: refused, never overwritten
    const auto again = store.checkpoint("2026-11-23");
    ok &= again.is_err() && again.unwrap_err_unchecked() == kv_errc::invalid_argument;
    ok &= store.checkpoint("").is_err();

    const auto restored = store.open_checkpoint("2026-11-23");   // the reference's own way back
    ok &= restored && *kv_get(*restored, "position/AAPL") == std::optional<std::string>("100");

    std::printf(ok ? "kv_backup demo: ok\n" : "kv_backup demo: FAILED\n");
    return ok ? 0 : 1;
}
