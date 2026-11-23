//
//  key_value_store_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/key_value_store_conformance.hpp>: the kit run on the reference store
//  through both forms, the failure checks, the health and backup kits, and the kit catching a store whose insert
//  overwrites. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store_conformance.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <memory>

using namespace snicholls;

namespace {

void print(const char* what, const kv_conformance_report& report)
{
    std::printf("  %-34s %zu checks, %zu failure(s)\n", what, report.checks, report.failures.size());
    for (const auto& f : report.failures)
        std::printf("    %.*s: %.*s\n", static_cast<int>(f.check.size()), f.check.data(), static_cast<int>(f.expectation.size()),
                    f.expectation.data());
}

// A store with one defect: insert overwrites (a ledger would write an entry twice).
struct overwriting_insert : reference_key_value_store {
    result<bool, kv_errc> insert(kv_bytes k, kv_bytes v)
    {
        static_cast<void>(put(k, v));
        return ok(true);
    }
};

} // namespace

int main()
{
    std::printf("snicholls-core key_value_store conformance demo\n");
    bool ok = true;

    //! [kit]
    // Any implementation, either form: make() returns a fresh, empty store (by value, or a handle to one).
    const auto by_concept = run_kv_conformance([] { return reference_key_value_store{}; });
    const auto by_interface = run_kv_conformance([]() -> std::unique_ptr<key_value_store_interface> {
        return std::make_unique<key_value_store_adapter<reference_key_value_store>>(reference_key_value_store{});
    });
    //! [kit]
    print("reference, through the concept:", by_concept);
    print("reference, through the interface:", by_interface);
    ok &= by_concept.ok() && by_interface.ok() && by_concept.checks == 25;

    // The checks that need the owner's way to make the store fail.
    reference_key_value_store store;
    const auto failures = kv_check_failure_changes_nothing(store, [](reference_key_value_store& s, std::size_t n) {
        s.inject_failure(kv_errc::io, n);
    });
    const auto unreachable = kv_check_no_exception_escapes(store, [](reference_key_value_store& s) { s.set_unavailable(true); });
    print("injected failures:", failures);
    print("unreachable store:", unreachable);
    ok &= failures.ok() && unreachable.ok();

    const auto health = run_kv_health_conformance([] { return reference_key_value_store{}; });
    const auto backup = run_kv_backup_conformance([] { return reference_key_value_store{}; }, "kit-checkpoint");
    print("health:", health);
    print("backup:", backup);
    ok &= health.ok() && backup.ok();

    // What the kit is for: a defect is reported by name.
    const auto broken = run_kv_conformance([] { return overwriting_insert{}; }, kv_conformance_options{.binary_safe = true});
    print("a store whose insert overwrites:", broken);
    ok &= broken.failed("insert_only_when_absent") && broken.failed("ledger_entry_is_written_once");

    std::printf(ok ? "key_value_store conformance demo: ok\n" : "key_value_store conformance demo: FAILED\n");
    return ok ? 0 : 1;
}
