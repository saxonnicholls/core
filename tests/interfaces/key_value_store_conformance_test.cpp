//
//  key_value_store_conformance_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/key_value_store_conformance.hpp>: the kit is only worth what it catches,
//  so each test hands it a store broken in one way and checks that exactly the right checks fail.
//

#include <snicholls/core/interfaces/key_value_store_conformance.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

using namespace snicholls;

namespace {

enum class flaw { none, insert_overwrites, sequence_frozen, batch_not_atomic, missing_is_an_error, scan_ignores_stop,
                  snapshot_is_live, put_throws, scan_includes_upper };

// The reference store, broken in one way.
template <flaw F>
struct broken_store {
    reference_key_value_store inner;

    template <typename V>
    result<bool, kv_errc> read(kv_bytes k, V&& fn) const
    {
        if constexpr (F == flaw::missing_is_an_error)
            if (!*inner.contains(k))
                return err(kv_errc::io);
        return inner.read(k, fn);
    }
    result<bool, kv_errc> contains(kv_bytes k) const { return inner.contains(k); }
    template <typename V>
    result<std::size_t, kv_errc> scan(const kv_range& r, V&& fn) const
    {
        if constexpr (F == flaw::scan_ignores_stop)
            return inner.scan(r, [&](kv_bytes k, kv_bytes v) { static_cast<void>(fn(k, v)); return true; });
        if constexpr (F == flaw::scan_includes_upper)
            if (r.bounded())
                return inner.scan(kv_range::from(r.lower()), [&](kv_bytes k, kv_bytes v) { return kv_less(r.upper(), k) ? false : fn(k, v); });
        return inner.scan(r, fn);
    }
    result<std::uint64_t, kv_errc> sequence() const
    {
        if constexpr (F == flaw::sequence_frozen)
            return ok(std::uint64_t{0});
        return inner.sequence();
    }
    result<void, kv_errc> put(kv_bytes k, kv_bytes v)
    {
        if constexpr (F == flaw::put_throws)
            throw std::runtime_error("leaks a driver's exception");
        return inner.put(k, v);
    }
    result<bool, kv_errc> insert(kv_bytes k, kv_bytes v)
    {
        if constexpr (F == flaw::insert_overwrites) {
            static_cast<void>(inner.put(k, v));
            return ok(true);
        }
        return inner.insert(k, v);
    }
    result<bool, kv_errc> erase(kv_bytes k) { return inner.erase(k); }
    result<std::size_t, kv_errc> erase_range(const kv_range& r) { return inner.erase_range(r); }
    result<void, kv_errc> apply(std::span<const kv_write> batch)
    {
        if constexpr (F != flaw::batch_not_atomic)
            return inner.apply(batch);
        for (const kv_write& w : batch) {   // one write at a time: a conflict leaves the earlier writes applied
            const auto r = inner.apply(std::span<const kv_write>(&w, 1));
            if (r.is_err())
                return r;
        }
        return ok();
    }
    result<reference_kv_snapshot, kv_errc> snapshot() const
    {
        if constexpr (F == flaw::snapshot_is_live)
            return err(kv_errc::backend);
        return inner.snapshot();
    }
};

template <flaw F>
kv_conformance_report run()
{
    return run_kv_conformance([] { return broken_store<F>{}; });
}

} // namespace

TEST(KvConformanceKit, PassesAStoreWithNoFlaw)
{
    const auto report = run<flaw::none>();
    EXPECT_TRUE(report.ok());
    EXPECT_EQ(report.checks, 25u);
}

TEST(KvConformanceKit, CatchesAnInsertThatOverwrites)
{
    const auto report = run<flaw::insert_overwrites>();
    EXPECT_TRUE(report.failed("insert_only_when_absent"));
    EXPECT_TRUE(report.failed("ledger_entry_is_written_once")) << "the stale writer must be told the entry exists";
    EXPECT_FALSE(report.failed("put_overwrites"));
}

TEST(KvConformanceKit, CatchesABatchThatIsNotAllOrNothing)
{
    const auto report = run<flaw::batch_not_atomic>();
    EXPECT_TRUE(report.failed("batch_insert_conflict_changes_nothing"));
    EXPECT_TRUE(report.failed("ledger_entry_is_written_once"));
    EXPECT_FALSE(report.failed("apply_runs_writes_in_order"));
}

TEST(KvConformanceKit, CatchesAFrozenSequence)
{
    const auto report = run<flaw::sequence_frozen>();
    EXPECT_TRUE(report.failed("sequence_rules"));
    EXPECT_TRUE(report.failed("snapshot_is_isolated"));
}

TEST(KvConformanceKit, CatchesAMissingKeyReportedAsAnError)
{
    const auto report = run<flaw::missing_is_an_error>();
    EXPECT_TRUE(report.failed("missing_key_is_not_an_error"));
    EXPECT_TRUE(report.failed("starts_empty"));
}

TEST(KvConformanceKit, CatchesAScanThatIgnoresItsVisitor)
{
    const auto report = run<flaw::scan_ignores_stop>();
    EXPECT_TRUE(report.failed("scan_stops_when_the_visitor_says_so"));
    EXPECT_FALSE(report.failed("ranges_are_half_open")) << "a scan that never stops early still visits the right keys";
}

TEST(KvConformanceKit, CatchesAnInclusiveUpperBound)
{
    const auto report = run<flaw::scan_includes_upper>();
    EXPECT_TRUE(report.failed("ranges_are_half_open"));
    EXPECT_FALSE(report.failed("scan_prefix_is_a_true_prefix"));
}

TEST(KvConformanceKit, CatchesAMissingSnapshot)
{
    const auto report = run<flaw::snapshot_is_live>();
    EXPECT_TRUE(report.failed("snapshot_is_isolated"));
    EXPECT_FALSE(report.failed("sequence_rules"));
}

TEST(KvConformanceKit, RecordsAnEscapingExceptionAsAFailureAndCarriesOn)
{
    const auto report = run<flaw::put_throws>();
    EXPECT_TRUE(report.failed("put_then_read_round_trips"));
    EXPECT_TRUE(report.failed("no_exception_crosses_the_interface"));
    EXPECT_EQ(report.checks, 25u) << "every check still ran";
    bool named = false;
    for (const auto& f : report.failures)
        named = named || f.expectation == "no exception crosses the interface";
    EXPECT_TRUE(named);
}

TEST(KvConformanceKit, BinarySafeOptionSkipsTheByteChecks)
{
    const auto report = run_kv_conformance([] { return reference_key_value_store{}; }, kv_conformance_options{.binary_safe = false});
    EXPECT_TRUE(report.ok());
    EXPECT_EQ(report.checks, 23u);
}

TEST(KvConformanceKit, FailureCheckCatchesAStoreThatIgnoresItsFault)
{
    reference_key_value_store s;
    const auto report = kv_check_failure_changes_nothing(s, [](reference_key_value_store&, std::size_t) {});   // never fails
    EXPECT_FALSE(report.ok());
    EXPECT_TRUE(report.failed("failure_changes_nothing"));
    const auto escapes = kv_check_no_exception_escapes(s, [](reference_key_value_store&) {});   // a working store
    EXPECT_TRUE(escapes.failed("no_exception_escapes")) << "a working store answers successes, not errors";
}

TEST(KvConformanceKit, GoldenVectorIsTheDigestOfTheKitsPairs)
{
    reference_key_value_store s;
    for (const auto& [k, v] : {std::pair{"", "e"}, std::pair{"a", ""}, std::pair{"ab", "x"}, std::pair{"b", "y"}})
        static_cast<void>(kv_put(s, k, v));
    EXPECT_EQ(kv_digest_v1(s)->hex().view(), kv_digest_v1_kit_vector);
}
