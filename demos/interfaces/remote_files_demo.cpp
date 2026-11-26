//
//  remote_files_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/remote_files.hpp>: a drop chosen at run time behind the interface, and the
//  handle. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/remote_files.hpp>
#include <snicholls/core/interfaces/remote_files_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <memory>
#include <span>

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
    std::printf("snicholls-core interfaces/remote_files.hpp demo\n");
    const std::byte csv[]{std::byte{'1'}, std::byte{','}, std::byte{'2'}};
    memory_remote_files read_only{false};
    static_cast<void>(read_only.put("in/a.csv", csv, utc_s::from_unix_seconds(1'800'000'000)));
    //! [virtual]
    std::unique_ptr<remote_files_interface> chosen = std::make_unique<remote_files_adapter<memory_remote_files>>(std::move(read_only));
    remote_files_ref handle{*chosen};   // copyable, models the concept
    std::size_t received = 0;
    const auto digest = handle.fetch("in/a.csv", [&](std::span<const std::byte> c) -> capability_result<void> {
        received += c.size();
        return ok();
    });
    //! [virtual]
    check(digest.is_ok() && digest->size == 3 && received == 3, "fetched through the interface");
    check(handle.list("in/")->size() == 1, "listed through the handle");
    check(!chosen->supports(remote_files_op::move) && handle.supports(remote_files_op::fetch), "supports answers through both");
    check(handle.move("in/a.csv", "archive/a.csv").unwrap_err().code == capability_errc::unsupported, "a read-only drop does not move");
    auto& adapter = static_cast<remote_files_adapter<memory_remote_files>&>(*chosen);
    check(adapter.get().size() == 1, "the adapter reaches the model");
    return failures == 0 ? 0 : 1;
}
