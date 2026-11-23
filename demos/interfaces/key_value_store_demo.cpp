//
//  key_value_store_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/key_value_store.hpp>: a composition root that picks the store at run time
//  and hands application code a key_value_store_interface&; the same generic function on both forms; a snapshot
//  through the interface; and the adapter turning a model's exception into an error. Exits non-zero if a result
//  is not the one documented.
//

#include <snicholls/core/interfaces/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {

//! [composition_root]
// The composition root chooses the backend (here from a name, in production from configuration) ...
std::unique_ptr<key_value_store_interface> open_store(std::string_view backend)
{
    if (backend == "memory")
        return std::make_unique<key_value_store_adapter<reference_key_value_store>>(reference_key_value_store{});
    return nullptr;   // above core: "rocksdb", "postgres", ...
}

// ... and application code sees only the interface, which models the concept, so the conveniences work on it.
bool record_login(key_value_store_interface& store, std::string_view user)
{
    return kv_put(store, "login/" + std::string(user), "2026-11-23T09:00:00Z").is_ok();
}
//! [composition_root]

// Generic code takes either form (and a snapshot of either: it only reads).
template <kv_readable S>
std::size_t logins(const S& store)
{
    return kv_count(store, kv_range::prefixed(kv_bytes_of("login/"))).unwrap_or(0);
}

struct flaky_driver {   // a model whose driver throws: the adapter keeps the exception from crossing
    reference_key_value_store inner;
    template <typename F>
    result<bool, kv_errc> read(kv_bytes, F&&) const { throw std::runtime_error("socket closed"); }
    result<bool, kv_errc> contains(kv_bytes k) const { return inner.contains(k); }
    template <typename F>
    result<std::size_t, kv_errc> scan(const kv_range& r, F&& fn) const { return inner.scan(r, fn); }
    result<std::uint64_t, kv_errc> sequence() const { return inner.sequence(); }
    result<void, kv_errc> put(kv_bytes k, kv_bytes v) { return inner.put(k, v); }
    result<bool, kv_errc> insert(kv_bytes k, kv_bytes v) { return inner.insert(k, v); }
    result<bool, kv_errc> erase(kv_bytes k) { return inner.erase(k); }
    result<std::size_t, kv_errc> erase_range(const kv_range& r) { return inner.erase_range(r); }
    result<void, kv_errc> apply(std::span<const kv_write> b) { return inner.apply(b); }
    result<reference_kv_snapshot, kv_errc> snapshot() const { return inner.snapshot(); }
};

} // namespace

int main()
{
    std::printf("snicholls-core key_value_store_interface demo\n");
    bool ok = true;

    const auto store = open_store("memory");
    ok &= store != nullptr && open_store("unknown") == nullptr;
    ok &= record_login(*store, "ada") && record_login(*store, "grace");
    std::printf("  logins through the interface: %zu\n", logins(*store));
    ok &= logins(*store) == 2;

    reference_key_value_store direct;   // the static form, the same function
    static_cast<void>(kv_put(direct, "login/ada", "x"));
    ok &= logins(direct) == 1;

    // The adapter refers to a store as well as owning one.
    key_value_store_adapter<reference_key_value_store&> view{direct};
    ok &= kv_insert(view, "login/linus", "y").unwrap_or(false) && logins(direct) == 2 && &view.get() == &direct;

    // A snapshot through the interface is a kv_reader_interface: read-only by type.
    auto snap = store->snapshot();
    ok &= snap.is_ok();
    static_cast<void>(kv_erase(*store, "login/ada"));
    const kv_reader_interface& before = **snap;
    std::printf("  logins now %zu, in the snapshot %zu\n", logins(*store), logins(before));
    ok &= logins(*store) == 1 && logins(before) == 2;

    // Every other operation, through the interface.
    ok &= *store->insert(kv_bytes_of("k"), kv_bytes_of("v")) && *store->contains(kv_bytes_of("k"));
    ok &= *store->erase(kv_bytes_of("k")) && store->apply({}).is_ok();
    ok &= *store->erase_range(kv_range::all()) == 1 && *store->sequence() > 0;
    bool seen = false;
    ok &= !*store->read(kv_bytes_of("k"), [&](kv_bytes) { seen = true; }) && !seen;
    ok &= *store->scan(kv_range::all(), [](kv_bytes, kv_bytes) { return true; }) == 0;

    // A model that throws: the caller gets an error, never the exception.
    key_value_store_adapter flaky{flaky_driver{}};
    const auto r = kv_get(flaky, "login/ada");
    std::printf("  a throwing driver answers: %s\n", r.is_err() && r.unwrap_err_unchecked() == kv_errc::backend ? "kv_errc::backend" : "?");
    ok &= r.is_err() && r.unwrap_err_unchecked() == kv_errc::backend;

    std::printf(ok ? "key_value_store_interface demo: ok\n" : "key_value_store_interface demo: FAILED\n");
    return ok ? 0 : 1;
}
