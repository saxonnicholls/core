//
//  remote_files_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/remote_files.hpp>: a collector written once against the concept, which lists a
//  drop, fetches what is new into its own store and verifies the digest; the name rules and the digester. Exits
//  non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/remote_files.hpp>
#include <snicholls/core/interfaces/remote_files_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

// Generic code: any drop, chosen at compile time. Fetches every file not yet in `store`, verifies it, archives it.
template <remote_files R>
capability_result<std::size_t> collect(R& drop, std::map<std::string, std::vector<std::byte>>& store)
{
    SNICHOLLS_TRY_ASSIGN(const auto files, drop.list("outgoing/"));
    std::size_t fetched = 0;
    for (const remote_file& f : files) {
        const std::string name(f.name.view());
        if (store.contains(name))
            continue;
        std::vector<std::byte> bytes;
        file_digester mine;   // the collector's own digest of what it stored
        const auto into_store = [&](std::span<const std::byte> chunk) -> capability_result<void> {
            bytes.insert(bytes.end(), chunk.begin(), chunk.end());
            mine.update(chunk);
            return ok();
        };
        SNICHOLLS_TRY_ASSIGN(const file_digest sent, drop.fetch(name, into_store));
        if (mine.finish() != sent || sent.size != f.size)
            return fail(capability_errc::internal, {}, "the file changed in flight");
        store.emplace(name, std::move(bytes));
        if (drop.supports(remote_files_op::move)) {
            SNICHOLLS_TRY_VOID(drop.move(name, "archive/" + name.substr(name.find('/') + 1)));
        }
        ++fetched;
    }
    return ok(fetched);
}

std::vector<std::byte> bytes_of(std::string_view s)
{
    std::vector<std::byte> out;
    for (const char c : s)
        out.push_back(static_cast<std::byte>(c));
    return out;
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/remote_files.hpp demo\n");
    const utc_s at = utc_s::from_unix_seconds(1'800'000'000);
    //! [concept]
    memory_remote_files drop{true, 16};   // in memory; SFTP and HTTPS implementations live above core
    static_cast<void>(drop.put("outgoing/APST_20261125.csv", bytes_of("date,amount\n2026-11-25,1200.00\n"), at));
    static_cast<void>(drop.put("outgoing/APST_20261126.csv", bytes_of("date,amount\n2026-11-26,2500.00\n"), at));
    std::map<std::string, std::vector<std::byte>> store;
    const auto collected = collect(drop, store);   // fetches both, verifies, moves them to archive/
    //! [concept]
    check(collected.is_ok() && *collected == 2, "two files collected");
    check(store.size() == 2 && store.begin()->second == bytes_of("date,amount\n2026-11-25,1200.00\n"), "stored byte for byte");
    check(drop.list("outgoing/")->empty() && drop.list("archive/")->size() == 2, "and archived in the drop");
    check(collect(drop, store).is_ok() && store.size() == 2, "a second run finds nothing new");
    check(is_remote_name("outgoing/APST_20261126.csv") && !is_remote_name("../etc/passwd"), "is_remote_name");
    check(is_remote_prefix("outgoing/") && !is_remote_prefix("/outgoing"), "is_remote_prefix");
    check(validate_remote_name("a//b").unwrap_err().code == capability_errc::invalid_argument, "validate_remote_name");
    check(validate_remote_prefix("").is_ok(), "validate_remote_prefix");
    check(file_digester::of({}).size == 0, "file_digester::of");
    return failures == 0 ? 0 : 1;
}
