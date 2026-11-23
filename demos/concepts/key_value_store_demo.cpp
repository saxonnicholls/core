//
//  key_value_store_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/key_value_store.hpp>: a ledger written against the concept - appending
//  entries once with insert, an atomic batch, prefix and chunked scans, a snapshot, the sequence and the digest -
//  run on the reference store. Exits non-zero if any result is not the one documented.
//

#include <snicholls/core/concepts/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

//! [contract]
// Code written once against the contract: it runs on any backend, and inlines on the in-memory one.
template <key_value_store S>
result<bool, kv_errc> append(S& ledger, std::string_view book, std::size_t n, std::string_view entry)
{
    const std::string key = "e/" + std::string(book) + "/" + std::to_string(n);
    return kv_insert(ledger, key, entry);   // false: someone else already wrote entry n
}
//! [contract]

std::string text(kv_bytes b) { return {reinterpret_cast<const char*>(b.data()), b.size()}; }

} // namespace

int main()
{
    std::printf("snicholls-core key_value_store demo\n");
    bool ok = true;
    reference_key_value_store ledger;

    ok &= *append(ledger, "cash", 0, "open 1000");
    ok &= *append(ledger, "cash", 1, "+250");
    const bool stale = *append(ledger, "cash", 1, "+999");   // a second writer with a stale n
    std::printf("  stale append of entry 1: %s\n", stale ? "written (wrong)" : "refused, entry exists");
    ok &= !stale;

    // A batch is all or nothing: its insert finds entry 1 present, so the index update is not applied either.
    kv_write_batch batch;
    batch.put("idx/cash", "2").insert("e/cash/1", "+999");
    const auto refused = ledger.apply(batch);
    ok &= refused.is_err() && refused.unwrap_err_unchecked() == kv_errc::conflict;
    ok &= !*kv_contains(ledger, "idx/cash");
    std::printf("  batch with a conflicting insert: %s\n", refused.is_err() ? "conflict, nothing applied" : "applied");

    // A batch of views needs no allocation.
    const std::array<kv_write, 2> views{kv_write::insert(kv_bytes_of("e/cash/2"), kv_bytes_of("-75")),
                                        kv_write::put(kv_bytes_of("idx/cash"), kv_bytes_of("3"))};
    ok &= ledger.apply(views).is_ok();

    //! [scan]
    // Exactly the keys of one book, in unsigned byte order; then the same keys two at a time.
    std::printf("  book cash:\n");
    const auto n = kv_scan_prefix(ledger, "e/cash/", [](kv_bytes k, kv_bytes v) {
        std::printf("    %s = %s\n", text(k).c_str(), text(v).c_str());
    });
    const kv_range book = kv_range::prefixed(kv_bytes_of("e/cash/"));
    std::string last;
    std::size_t chunks = 0, seen = 0;
    for (kv_range rest = book;; rest = book.after(kv_bytes_of(last))) {
        std::size_t in_chunk = 0;
        static_cast<void>(ledger.scan(rest, [&](kv_bytes k, kv_bytes) {
            last = text(k);
            return ++in_chunk < 2;
        }));
        if (in_chunk == 0)
            break;
        ++chunks;
        seen += in_chunk;
    }
    //! [scan]
    std::printf("  %zu entries, scanned again in %zu chunks\n", *n, chunks);
    ok &= *n == 3 && seen == 3 && chunks == 2;

    // A snapshot keeps its moment; the sequence and the digest name it.
    auto snap = *ledger.snapshot();
    static_cast<void>(kv_put(ledger, "idx/cash", "4"));
    const auto then = *kv_digest_v1(snap, book);
    const auto now = *kv_digest_v1(ledger, book);
    std::printf("  snapshot at sequence %llu, store at %llu; book digest %s (unchanged: %s)\n",
                static_cast<unsigned long long>(*snap.sequence()), static_cast<unsigned long long>(*ledger.sequence()),
                std::string(then.hex().view()).c_str(), then == now ? "yes" : "no");
    ok &= *kv_get(snap, "idx/cash") == std::optional<std::string>("3") && then == now && *snap.sequence() < *ledger.sequence();

    // The other reads and writes, through the conveniences.
    std::string buffer;
    ok &= *kv_get(ledger, "e/cash/0", buffer) && buffer == "open 1000";
    std::size_t length = 0;
    ok &= *kv_read(ledger, "e/cash/1", [&](kv_bytes v) { length = v.size(); }) && length == 4;
    ok &= kv_put(ledger, "tmp", "x").is_ok() && *kv_erase(ledger, "tmp") && !*kv_erase(ledger, "tmp");
    ok &= *ledger.erase_range(kv_range::between(kv_bytes_of("idx/"), kv_bytes_of("idx0"))) == 1;
    ok &= *kv_count(ledger) == 3;

    std::printf(ok ? "key_value_store demo: ok\n" : "key_value_store demo: FAILED\n");
    return ok ? 0 : 1;
}
