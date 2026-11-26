//
//  kv_cas_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/kv_cas.hpp>: the kit passes on the reference directly and through the adapter,
//  and catches a store whose swap ignores the expected value.
//

#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_cas.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <type_traits>

using namespace snicholls;

namespace {

static_assert(std::is_abstract_v<kv_cas_interface>);

// Reads through the store, swaps through the virtual interface.
struct virtual_cas {
    reference_key_value_store store;
    kv_cas_adapter<reference_key_value_store&> adapter{store};
    result<bool, kv_errc> compare_and_swap(kv_bytes k, std::optional<kv_bytes> e, std::optional<kv_bytes> d)
    {
        return static_cast<kv_cas_interface&>(adapter).compare_and_swap(k, e, d);
    }
    result<bool, kv_errc> read(kv_bytes k, kv_read_visitor f) const { return store.read(k, f); }
    result<bool, kv_errc> contains(kv_bytes k) const { return store.contains(k); }
    result<std::size_t, kv_errc> scan(const kv_range& r, kv_scan_visitor f) const { return store.scan(r, f); }
    result<std::uint64_t, kv_errc> sequence() const { return store.sequence(); }
};

// Swaps whatever the value is: a put that only checks presence.
struct blind_cas {
    reference_key_value_store store;
    result<bool, kv_errc> compare_and_swap(kv_bytes k, std::optional<kv_bytes> e, std::optional<kv_bytes> d)
    {
        const bool present = *store.contains(k);
        if (present != e.has_value())
            return ok(false);
        if (d)
            static_cast<void>(store.put(k, *d));
        else
            static_cast<void>(store.erase(k));
        return ok(true);
    }
    result<bool, kv_errc> read(kv_bytes k, kv_read_visitor f) const { return store.read(k, f); }
    result<bool, kv_errc> contains(kv_bytes k) const { return store.contains(k); }
    result<std::size_t, kv_errc> scan(const kv_range& r, kv_scan_visitor f) const { return store.scan(r, f); }
    result<std::uint64_t, kv_errc> sequence() const { return store.sequence(); }
};

TEST(KvCasInterface, TheKitPassesDirectlyAndThroughTheAdapter)
{
    reference_key_value_store direct;
    EXPECT_EQ(check_kv_cas(direct), std::nullopt);
    virtual_cas through;
    EXPECT_EQ(check_kv_cas(through), std::nullopt);
    kv_cas_adapter owned{reference_key_value_store{}};
    EXPECT_TRUE(*owned.compare_and_swap(kv_bytes{}, std::nullopt, kv_bytes{}));
    EXPECT_TRUE(*owned.get().contains(kv_bytes{}));
}

TEST(KvCasInterface, TheKitCatchesABlindSwap)
{
    blind_cas s;
    EXPECT_EQ(check_kv_cas(s), std::optional<std::string_view>{"kv_cas: a wrong expected value refuses and changes nothing"});
}

} // namespace
