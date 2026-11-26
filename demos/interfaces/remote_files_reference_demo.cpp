//
//  remote_files_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/remote_files_reference.hpp>: a drop in memory, fetched in small chunks, moved, and
//  made to fail once. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/remote_files_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <span>
#include <string_view>
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
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/remote_files_reference.hpp demo\n");
    const std::string_view text = "date,amount\n2026-11-26,2500.00\n";
    const auto bytes = std::as_bytes(std::span{text.data(), text.size()});
    //! [reference]
    memory_remote_files drop{true, 8};   // move supported, 8-byte chunks
    const auto put = drop.put("outgoing/APST_20261126.csv", bytes, utc_s::from_unix_seconds(1'800'000'000));
    std::size_t chunks = 0;
    const auto digest = drop.fetch("outgoing/APST_20261126.csv", [&](std::span<const std::byte>) -> capability_result<void> {
        ++chunks;
        return ok();
    });
    drop.fail_next(capability_errc::unavailable, "503 Service Unavailable");   // the next call fails once
    //! [reference]
    check(put.is_ok() && drop.size() == 1, "put a file");
    check(digest.is_ok() && *digest == file_digester::of(bytes) && chunks == 4, "fetched in four chunks with its digest");
    const auto down = drop.list("");
    check(down.is_err() && down.unwrap_err().provider_code == std::string_view{"503"}, "the injected failure keeps the code only");
    check(drop.list("")->size() == 1, "and the next call succeeds");
    check(drop.supports(remote_files_op::move) && drop.move("outgoing/APST_20261126.csv", "archive/APST_20261126.csv").is_ok(),
          "move");
    check(drop.contents("archive/APST_20261126.csv") != nullptr && drop.contents("outgoing/APST_20261126.csv") == nullptr,
          "contents");
    check(drop.put("archive/APST_20261126.csv", bytes, {}).unwrap_err().code == capability_errc::conflict, "a taken name");
    return failures == 0 ? 0 : 1;
}
