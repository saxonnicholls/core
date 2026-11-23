//
//  key_value_store_reference_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/interfaces/key_value_store_reference.hpp>: what the reference store promises beyond
//  the kit - its fake's controls (injected failures, unavailability), copy-on-write sharing between the store, its
//  copies, snapshots and checkpoints, its sequence, diagnostics and verification.
//

#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

using namespace snicholls;

namespace {
std::optional<std::string> value(const reference_key_value_store& s, std::string_view key) { return *kv_get(s, key); }
} // namespace

TEST(KvReference, StartsEmptyAtSequenceZero)
{
    const reference_key_value_store s;
    EXPECT_EQ(*s.sequence(), 0u);
    EXPECT_EQ(*kv_count(s), 0u);
    EXPECT_TRUE(s.ping().is_ok());
}

TEST(KvReference, SequenceRisesByOnePerChangingCall)
{
    reference_key_value_store s;
    static_cast<void>(kv_put(s, "a", "1"));
    static_cast<void>(kv_put(s, "a", "2"));   // an overwrite is a change
    EXPECT_EQ(*s.sequence(), 2u);
    static_cast<void>(kv_insert(s, "a", "3"));
    static_cast<void>(kv_erase(s, "missing"));
    static_cast<void>(s.erase_range(kv_range::prefixed(kv_bytes_of("zz"))));
    static_cast<void>(s.apply({}));
    EXPECT_EQ(*s.sequence(), 2u) << "calls that changed nothing";
    kv_write_batch batch;
    batch.put("b", "1").put("c", "1");
    static_cast<void>(s.apply(batch));
    EXPECT_EQ(*s.sequence(), 3u) << "a batch is one change";
    EXPECT_EQ(*s.erase_range(kv_range::all()), 3u);
    EXPECT_EQ(*s.sequence(), 4u);
}

TEST(KvReference, InjectedFailureFailsTheNthWriteOnceAndChangesNothing)
{
    reference_key_value_store s;
    s.inject_failure(kv_errc::io, 1);
    EXPECT_TRUE(kv_put(s, "a", "1").is_ok()) << "one write succeeds first";
    EXPECT_EQ(kv_put(s, "b", "1").unwrap_err(), kv_errc::io);
    EXPECT_FALSE(*kv_contains(s, "b"));
    EXPECT_TRUE(kv_put(s, "b", "1").is_ok()) << "one-shot";
    for (auto fail : {+[](reference_key_value_store& x) { return x.insert(kv_bytes_of("c"), kv_bytes_of("1")).is_err(); },
                      +[](reference_key_value_store& x) { return x.erase(kv_bytes_of("a")).is_err(); },
                      +[](reference_key_value_store& x) { return x.erase_range(kv_range::all()).is_err(); },
                      +[](reference_key_value_store& x) { return x.checkpoint("cp").is_err(); }}) {
        s.inject_failure(kv_errc::corrupt);
        EXPECT_TRUE(fail(s));
    }
    EXPECT_EQ(*kv_count(s), 2u);
    EXPECT_EQ(*s.sequence(), 2u);
    EXPECT_FALSE(s.open_checkpoint("cp").has_value());
}

TEST(KvReference, InjectedFailureMidBatchRollsTheWholeBatchBack)
{
    reference_key_value_store s;
    static_cast<void>(kv_put(s, "keep", "1"));
    s.inject_failure(kv_errc::limit, 2);
    kv_write_batch batch;
    batch.put("a", "1").erase("keep").put("b", "1").put("c", "1");
    EXPECT_EQ(s.apply(batch).unwrap_err(), kv_errc::limit);
    EXPECT_EQ(*kv_count(s), 1u);
    EXPECT_EQ(value(s, "keep"), "1");
    EXPECT_TRUE(s.apply(batch).is_ok()) << "spent by the failure";
}

TEST(KvReference, UnavailableRefusesEveryOperationUntilRestored)
{
    reference_key_value_store s;
    static_cast<void>(kv_put(s, "a", "1"));
    s.set_unavailable(true);
    EXPECT_EQ(s.ping().unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(s.diagnostics().unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(s.verify().unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(s.checkpoint("cp").unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(s.snapshot().unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(kv_get(s, "a").unwrap_err(), kv_errc::unavailable);
    s.set_unavailable(false);
    EXPECT_EQ(value(s, "a"), "1") << "nothing was lost";
}

TEST(KvReference, CopiesAreIndependentValues)
{
    reference_key_value_store a;
    static_cast<void>(kv_put(a, "k", "1"));
    reference_key_value_store b = a;   // shares until one writes
    static_cast<void>(kv_put(b, "k", "2"));
    EXPECT_EQ(value(a, "k"), "1");
    EXPECT_EQ(value(b, "k"), "2");
    reference_key_value_store c = std::move(b);
    EXPECT_EQ(value(c, "k"), "2");
}

TEST(KvReference, SnapshotOutlivesTheStore)
{
    std::optional<reference_kv_snapshot> snap;
    {
        reference_key_value_store s;
        static_cast<void>(kv_put(s, "k", "v"));
        snap.emplace(*s.snapshot());
    }
    EXPECT_EQ(*kv_get(*snap, "k"), std::optional<std::string>("v"));
    EXPECT_TRUE(*snap->contains(kv_bytes_of("k")));
    EXPECT_EQ(*snap->sequence(), 1u);
}

TEST(KvReference, CheckpointsReopenAsWritableStores)
{
    reference_key_value_store s;
    static_cast<void>(kv_put(s, "k", "1"));
    const auto info = s.checkpoint("before");
    ASSERT_TRUE(info.is_ok());
    EXPECT_EQ(info->sequence, 1u);
    EXPECT_EQ(info->bytes, std::optional<std::uint64_t>(2));
    static_cast<void>(kv_put(s, "k", "2"));
    auto opened = s.open_checkpoint("before");
    ASSERT_TRUE(opened.has_value());
    EXPECT_EQ(value(*opened, "k"), "1");
    EXPECT_EQ(*opened->sequence(), 1u);
    static_cast<void>(kv_put(*opened, "k", "3"));
    EXPECT_EQ(value(*s.open_checkpoint("before"), "k"), "1") << "writing to an opened checkpoint leaves it intact";
    EXPECT_EQ(value(s, "k"), "2");
    EXPECT_FALSE(s.open_checkpoint("never").has_value());
    EXPECT_EQ(s.checkpoint("before").unwrap_err(), kv_errc::invalid_argument);
    EXPECT_EQ(s.checkpoint("").unwrap_err(), kv_errc::invalid_argument);
}

TEST(KvReference, DiagnosticsAndVerifyAreExact)
{
    reference_key_value_store s;
    static_cast<void>(kv_put(s, "ab", "cde"));
    static_cast<void>(kv_put(s, "", "f"));
    const auto d = *s.diagnostics();
    EXPECT_EQ(d.backend, "reference");
    EXPECT_EQ(d.sequence, 2u);
    EXPECT_EQ(d.keys, std::optional<std::uint64_t>(2));
    EXPECT_EQ(d.bytes, std::optional<std::uint64_t>(6));
    EXPECT_FALSE(d.persistent);
    EXPECT_FALSE(d.read_only);
    const auto v = *s.verify();
    EXPECT_EQ(v.keys_checked, 2u);
    EXPECT_TRUE(v.ok());
}
