//
//  key_value_store_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/key_value_store.hpp>: the conformance kit run against the reference store
//  through the concept and through the virtual interface (owning and referring adapters, and a backend deriving
//  from the interface), the failure checks through both forms, and the adapters' promise that no exception of
//  the adapted store crosses them while the caller's visitor's exceptions do.
//

#include <snicholls/core/interfaces/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store_conformance.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include "kv_virtual_backend.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace snicholls;

namespace {

std::string describe(const kv_conformance_report& report)
{
    std::string out = std::to_string(report.checks) + " checks";
    for (const auto& f : report.failures)
        out += "\n  " + std::string(f.check) + ": " + std::string(f.expectation);
    return out;
}

std::unique_ptr<key_value_store_interface> make_virtual()
{
    return std::make_unique<key_value_store_adapter<reference_key_value_store>>(reference_key_value_store{});
}

// Generic code, written once against the concept.
template <key_value_store S>
std::size_t append_twice(S& store)
{
    static_cast<void>(kv_insert(store, "e/1", "a"));
    static_cast<void>(kv_insert(store, "e/1", "b"));
    return *kv_count(store);
}

} // namespace

TEST(KvConformance, ReferencePassesThroughTheConcept)
{
    const auto report = run_kv_conformance([] { return reference_key_value_store{}; });
    EXPECT_TRUE(report.ok()) << describe(report);
    EXPECT_EQ(report.checks, 25u);
}

TEST(KvConformance, ReferencePassesThroughTheVirtualInterface)
{
    const auto report = run_kv_conformance(make_virtual);
    EXPECT_TRUE(report.ok()) << describe(report);
    EXPECT_EQ(report.checks, 25u);
}

TEST(KvConformance, ReferencePassesThroughAReferringAdapter)
{
    reference_key_value_store store;
    const auto report = run_kv_conformance([&] {
        static_cast<void>(store.erase_range(kv_range::all()));
        return std::make_unique<key_value_store_adapter<reference_key_value_store&>>(store);
    });
    EXPECT_TRUE(report.ok()) << describe(report);
}

TEST(KvConformance, ABackendDerivingFromTheInterfacePasses)
{
    const auto report = run_kv_conformance([] { return std::make_unique<kv_test::kv_virtual_backend>(); });
    EXPECT_TRUE(report.ok()) << describe(report);
}

TEST(KvConformance, AStoreAdaptedTwiceStillPasses)
{
    // The virtual interface is itself a model, so it adapts again: its snapshots pass straight through.
    std::vector<std::unique_ptr<key_value_store_interface>> inners;
    const auto report = run_kv_conformance([&] {
        inners.push_back(make_virtual());
        return std::make_unique<key_value_store_adapter<key_value_store_interface&>>(*inners.back());
    });
    EXPECT_TRUE(report.ok()) << describe(report);
}

TEST(KvConformance, FailuresChangeNothingThroughBothForms)
{
    reference_key_value_store direct;
    const auto a = kv_check_failure_changes_nothing(direct, [](reference_key_value_store& s, std::size_t n) {
        s.inject_failure(kv_errc::io, n);
    });
    EXPECT_TRUE(a.ok()) << describe(a);

    key_value_store_adapter<reference_key_value_store> adapted{reference_key_value_store{}};
    key_value_store_interface& through = adapted;
    const auto b = kv_check_failure_changes_nothing(through, [&](key_value_store_interface&, std::size_t n) {
        adapted.get().inject_failure(kv_errc::io, n);
    });
    EXPECT_TRUE(b.ok()) << describe(b);
}

TEST(KvConformance, NoExceptionEscapesAnUnavailableStoreInEitherForm)
{
    reference_key_value_store direct;
    const auto a = kv_check_no_exception_escapes(direct, [](reference_key_value_store& s) { s.set_unavailable(true); });
    EXPECT_TRUE(a.ok()) << describe(a);

    key_value_store_adapter<reference_key_value_store> adapted{reference_key_value_store{}};
    key_value_store_interface& through = adapted;
    const auto b = kv_check_no_exception_escapes(through, [&](key_value_store_interface&) { adapted.get().set_unavailable(true); });
    EXPECT_TRUE(b.ok()) << describe(b);
}

TEST(KvInterface, GenericCodeTakesEitherForm)
{
    reference_key_value_store direct;
    EXPECT_EQ(append_twice(direct), 1u);
    auto adapted = make_virtual();
    EXPECT_EQ(append_twice(*adapted), 1u);
    EXPECT_EQ(*kv_get(*adapted, "e/1"), std::optional<std::string>("a"));
}

TEST(KvInterface, AnAdaptedSnapshotIsAReaderInterface)
{
    auto store = make_virtual();
    static_cast<void>(kv_put(*store, "a", "1"));
    auto snap = store->snapshot();
    ASSERT_TRUE(snap.is_ok());
    static_cast<void>(kv_put(*store, "a", "2"));
    const kv_reader_interface& reader = **snap;
    EXPECT_EQ(*kv_get(reader, "a"), std::optional<std::string>("1"));
    EXPECT_EQ(*reader.sequence(), 1u);
}

TEST(KvAdapter, ReferringAdapterSharesTheModel)
{
    reference_key_value_store store;
    key_value_store_adapter<reference_key_value_store&> adapter{store};
    static_cast<void>(kv_put(adapter, "k", "v"));
    EXPECT_TRUE(*kv_contains(store, "k"));
    EXPECT_EQ(&adapter.get(), &store);
    const auto& constant = adapter;
    EXPECT_EQ(&constant.get(), &store);
}

TEST(KvAdapter, DeductionGuideOwnsTheModel)
{
    key_value_store_adapter adapter{reference_key_value_store{}};
    static_assert(std::is_same_v<decltype(adapter), key_value_store_adapter<reference_key_value_store>>);
    EXPECT_TRUE(adapter.put(kv_bytes_of("k"), kv_bytes_of("v")).is_ok());
}

namespace {

// A model whose every operation throws what it is told to: the adapter must translate it.
struct throwing_store {
    enum class mode { nothing, runtime_error, bad_alloc } throws = mode::nothing;
    reference_key_value_store inner;

    void maybe_throw() const
    {
        if (throws == mode::runtime_error)
            throw std::runtime_error("a third-party driver failed");
        if (throws == mode::bad_alloc)
            throw std::bad_alloc();
    }
    template <typename F>
    result<bool, kv_errc> read(kv_bytes k, F&& fn) const { maybe_throw(); return inner.read(k, fn); }
    result<bool, kv_errc> contains(kv_bytes k) const { maybe_throw(); return inner.contains(k); }
    template <typename F>
    result<std::size_t, kv_errc> scan(const kv_range& r, F&& fn) const { maybe_throw(); return inner.scan(r, fn); }
    result<std::uint64_t, kv_errc> sequence() const { maybe_throw(); return inner.sequence(); }
    result<void, kv_errc> put(kv_bytes k, kv_bytes v) { maybe_throw(); return inner.put(k, v); }
    result<bool, kv_errc> insert(kv_bytes k, kv_bytes v) { maybe_throw(); return inner.insert(k, v); }
    result<bool, kv_errc> erase(kv_bytes k) { maybe_throw(); return inner.erase(k); }
    result<std::size_t, kv_errc> erase_range(const kv_range& r) { maybe_throw(); return inner.erase_range(r); }
    result<void, kv_errc> apply(std::span<const kv_write> b) { maybe_throw(); return inner.apply(b); }
    result<reference_kv_snapshot, kv_errc> snapshot() const { maybe_throw(); return inner.snapshot(); }
};
static_assert(key_value_store<throwing_store>);

} // namespace

TEST(KvAdapter, TranslatesEveryExceptionOfTheModel)
{
    key_value_store_adapter<throwing_store> adapter{throwing_store{}};
    key_value_store_interface& s = adapter;
    for (const auto& [mode, expected] : {std::pair{throwing_store::mode::runtime_error, kv_errc::backend},
                                        std::pair{throwing_store::mode::bad_alloc, kv_errc::out_of_memory}}) {
        adapter.get().throws = mode;
        const auto k = kv_bytes_of("k");
        EXPECT_EQ(s.put(k, k).unwrap_err(), expected);
        EXPECT_EQ(s.insert(k, k).unwrap_err(), expected);
        EXPECT_EQ(s.erase(k).unwrap_err(), expected);
        EXPECT_EQ(s.erase_range(kv_range::all()).unwrap_err(), expected);
        EXPECT_EQ(s.apply({}).unwrap_err(), expected);
        EXPECT_EQ(s.read(k, [](kv_bytes) {}).unwrap_err(), expected);
        EXPECT_EQ(s.contains(k).unwrap_err(), expected);
        EXPECT_EQ(s.scan(kv_range::all(), [](kv_bytes, kv_bytes) { return true; }).unwrap_err(), expected);
        EXPECT_EQ(s.sequence().unwrap_err(), expected);
        EXPECT_EQ(s.snapshot().unwrap_err(), expected);
    }
    adapter.get().throws = throwing_store::mode::runtime_error;
    EXPECT_TRUE(kv_check_no_exception_escapes(s, [](key_value_store_interface&) {}).ok());
}

TEST(KvAdapter, LetsTheVisitorsOwnExceptionsThroughUnchanged)
{
    key_value_store_adapter<throwing_store> adapter{throwing_store{}};
    key_value_store_interface& s = adapter;
    static_cast<void>(kv_put(s, "k", "v"));
    EXPECT_THROW(static_cast<void>(s.read(kv_bytes_of("k"), [](kv_bytes) { throw std::logic_error("mine"); })), std::logic_error);
    EXPECT_THROW(static_cast<void>(s.scan(kv_range::all(), [](kv_bytes, kv_bytes) -> bool { throw std::bad_alloc(); })),
                 std::bad_alloc)
        << "even a bad_alloc, when the visitor threw it";
    auto snap = s.snapshot();
    ASSERT_TRUE(snap.is_ok());
    EXPECT_THROW(static_cast<void>((*snap)->read(kv_bytes_of("k"), [](kv_bytes) { throw std::logic_error("mine"); })),
                 std::logic_error);
    EXPECT_TRUE(*kv_contains(s, "k")) << "the store is usable afterwards";
}

TEST(KvReaderAdapter, TranslatesExceptionsOfTheView)
{
    struct throwing_view {
        result<bool, kv_errc> read(kv_bytes, kv_read_visitor) const { throw std::runtime_error("x"); }
        result<bool, kv_errc> contains(kv_bytes) const { throw std::runtime_error("x"); }
        result<std::size_t, kv_errc> scan(const kv_range&, kv_scan_visitor) const { throw std::bad_alloc(); }
        result<std::uint64_t, kv_errc> sequence() const { throw std::runtime_error("x"); }
    };
    const kv_reader_adapter<throwing_view> reader{throwing_view{}};
    EXPECT_EQ(reader.read(kv_bytes{}, [](kv_bytes) {}).unwrap_err(), kv_errc::backend);
    EXPECT_EQ(reader.contains(kv_bytes{}).unwrap_err(), kv_errc::backend);
    EXPECT_EQ(reader.scan(kv_range::all(), [](kv_bytes, kv_bytes) { return true; }).unwrap_err(), kv_errc::out_of_memory);
    EXPECT_EQ(reader.sequence().unwrap_err(), kv_errc::backend);
}
