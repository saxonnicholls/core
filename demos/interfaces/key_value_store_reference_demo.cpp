//
//  key_value_store_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/key_value_store_reference.hpp>: the reference store as a test fake - the
//  code under test retries a write that fails once, survives a failure in the middle of a batch, and reports an
//  unreachable store - then its snapshots, copies, checkpoints, diagnostics and verification. Exits non-zero if
//  a result is not the one documented.
//

#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <cstdio>
#include <optional>
#include <string>

using namespace snicholls;

namespace {
// The code under test: saves an order, retrying once on a storage error.
template <key_value_store S>
result<void, kv_errc> save_order(S& store, std::string_view id, std::string_view body)
{
    auto r = kv_put(store, "order/" + std::string(id), body);
    if (r.is_err())
        r = kv_put(store, "order/" + std::string(id), body);
    return r;
}
} // namespace

int main()
{
    std::printf("snicholls-core reference_key_value_store demo\n");
    bool ok = true;

    //! [fake]
    reference_key_value_store fake;
    fake.inject_failure(kv_errc::io);   // the next write fails, once
    ok &= save_order(fake, "1", "2 x AAPL").is_ok() && *kv_count(fake) == 1;   // the retry succeeded

    fake.inject_failure(kv_errc::io, 1);   // a batch fails at its second write ...
    kv_write_batch batch;
    batch.put("order/2", "1 x MSFT").put("order/3", "5 x NVDA");
    ok &= fake.apply(batch).is_err() && *kv_count(fake) == 1;   // ... and changes nothing

    fake.set_unavailable(true);   // as a lost connection would
    ok &= save_order(fake, "4", "x").unwrap_err() == kv_errc::unavailable;
    fake.set_unavailable(false);
    //! [fake]
    std::printf("  after the injected failures: %zu order(s), sequence %llu\n", *kv_count(fake),
                static_cast<unsigned long long>(*fake.sequence()));
    ok &= *fake.sequence() == 1;

    // Snapshots and copies share the data until one side writes.
    const auto snap = *fake.snapshot();
    reference_key_value_store copy = fake;
    static_cast<void>(kv_put(copy, "order/1", "changed"));
    ok &= *kv_get(fake, "order/1") == std::optional<std::string>("2 x AAPL");
    ok &= *kv_get(snap, "order/1") == std::optional<std::string>("2 x AAPL") && *snap.sequence() == 1;
    ok &= *snap.contains(kv_bytes_of("order/1")) && *kv_count(snap) == 1;

    // Checkpoints in memory, reopened as stores; health and verification.
    ok &= fake.checkpoint("eod")->sequence == 1;
    ok &= fake.open_checkpoint("eod").has_value() && !fake.open_checkpoint("missing").has_value();
    ok &= fake.ping().is_ok() && fake.diagnostics()->backend == "reference" && fake.verify()->ok();
    ok &= *kv_insert(fake, "order/9", "x") && *kv_erase(fake, "order/9") && *fake.erase_range(kv_range::all()) == 1;

    std::printf(ok ? "reference_key_value_store demo: ok\n" : "reference_key_value_store demo: FAILED\n");
    return ok ? 0 : 1;
}
