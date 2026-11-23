//
//  key_value_store_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/concepts/key_value_store.hpp>: at compile time, which types model the contract (the
//  reference, its snapshot, the virtual interface and both adapters) and which do not (an operation missing, a
//  wrong signature); kv_range, kv_write and the order as constant expressions; then the batch builder, the
//  conveniences and the digest's golden vectors.
//

#include <snicholls/core/concepts/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace snicholls;

namespace {

// --- models ----------------------------------------------------------------------------------------------------

static_assert(key_value_store<reference_key_value_store>);
static_assert(kv_readable<reference_kv_snapshot>);
static_assert(!kv_writable<reference_kv_snapshot>, "a snapshot is read-only by type");
static_assert(key_value_store<key_value_store_interface>, "the virtual interface models the concept");
static_assert(kv_readable<kv_reader_interface>);
static_assert(!kv_writable<kv_reader_interface>);
static_assert(key_value_store<key_value_store_adapter<reference_key_value_store>>, "the owning adapter models it");
static_assert(key_value_store<key_value_store_adapter<reference_key_value_store&>>, "the referring adapter models it");
static_assert(kv_readable<kv_reader_adapter<reference_kv_snapshot>>);
static_assert(kv_readable<kv_reader_adapter<std::unique_ptr<kv_reader_interface>>>, "a handle to a reader adapts too");
static_assert(std::is_base_of_v<key_value_store_interface, key_value_store_adapter<reference_key_value_store>>);

// --- near misses: each differs from a model in one operation -----------------------------------------------------

#define SNICHOLLS_KV_TEST_READS                                                                                    \
    result<bool, kv_errc> read(kv_bytes, kv_read_visitor) const;                                                    \
    result<bool, kv_errc> contains(kv_bytes) const;                                                                 \
    result<std::size_t, kv_errc> scan(const kv_range&, kv_scan_visitor) const;                                      \
    result<std::uint64_t, kv_errc> sequence() const;

#define SNICHOLLS_KV_TEST_WRITES                                                                                   \
    result<void, kv_errc> put(kv_bytes, kv_bytes);                                                                  \
    result<bool, kv_errc> insert(kv_bytes, kv_bytes);                                                               \
    result<bool, kv_errc> erase(kv_bytes);                                                                          \
    result<std::size_t, kv_errc> erase_range(const kv_range&);                                                      \
    result<void, kv_errc> apply(std::span<const kv_write>);

struct reader_only {
    SNICHOLLS_KV_TEST_READS
};
struct complete {
    SNICHOLLS_KV_TEST_READS
    SNICHOLLS_KV_TEST_WRITES
    result<reader_only, kv_errc> snapshot() const;
};
struct no_snapshot {
    SNICHOLLS_KV_TEST_READS
    SNICHOLLS_KV_TEST_WRITES
};
struct snapshot_not_readable {
    SNICHOLLS_KV_TEST_READS
    SNICHOLLS_KV_TEST_WRITES
    result<int, kv_errc> snapshot() const;
};
struct snapshot_by_handle {
    SNICHOLLS_KV_TEST_READS
    SNICHOLLS_KV_TEST_WRITES
    result<std::unique_ptr<reader_only>, kv_errc> snapshot() const;
};
struct no_scan {
    result<bool, kv_errc> read(kv_bytes, kv_read_visitor) const;
    result<bool, kv_errc> contains(kv_bytes) const;
    result<std::uint64_t, kv_errc> sequence() const;
};
struct no_insert {
    SNICHOLLS_KV_TEST_READS
    result<void, kv_errc> put(kv_bytes, kv_bytes);
    result<bool, kv_errc> erase(kv_bytes);
    result<std::size_t, kv_errc> erase_range(const kv_range&);
    result<void, kv_errc> apply(std::span<const kv_write>);
};
struct put_returns_bool {
    SNICHOLLS_KV_TEST_READS
    bool put(kv_bytes, kv_bytes);
    result<bool, kv_errc> insert(kv_bytes, kv_bytes);
    result<bool, kv_errc> erase(kv_bytes);
    result<std::size_t, kv_errc> erase_range(const kv_range&);
    result<void, kv_errc> apply(std::span<const kv_write>);
};
struct read_not_const {
    result<bool, kv_errc> read(kv_bytes, kv_read_visitor);
    result<bool, kv_errc> contains(kv_bytes) const;
    result<std::size_t, kv_errc> scan(const kv_range&, kv_scan_visitor) const;
    result<std::uint64_t, kv_errc> sequence() const;
};
struct scan_with_wrong_visitor {
    result<bool, kv_errc> read(kv_bytes, kv_read_visitor) const;
    result<bool, kv_errc> contains(kv_bytes) const;
    result<std::size_t, kv_errc> scan(const kv_range&, function_ref<void(kv_bytes)>) const;
    result<std::uint64_t, kv_errc> sequence() const;
};
struct string_keys {
    result<bool, kv_errc> read(const std::string&, kv_read_visitor) const;
    result<bool, kv_errc> contains(kv_bytes) const;
    result<std::size_t, kv_errc> scan(const kv_range&, kv_scan_visitor) const;
    result<std::uint64_t, kv_errc> sequence() const;
};
struct errors_of_another_type {
    result<bool, int> read(kv_bytes, kv_read_visitor) const;
    result<bool, kv_errc> contains(kv_bytes) const;
    result<std::size_t, kv_errc> scan(const kv_range&, kv_scan_visitor) const;
    result<std::uint64_t, kv_errc> sequence() const;
};
struct batch_of_strings {
    SNICHOLLS_KV_TEST_READS
    result<void, kv_errc> put(kv_bytes, kv_bytes);
    result<bool, kv_errc> insert(kv_bytes, kv_bytes);
    result<bool, kv_errc> erase(kv_bytes);
    result<std::size_t, kv_errc> erase_range(const kv_range&);
    result<void, kv_errc> apply(const std::vector<std::string>&);
};

#undef SNICHOLLS_KV_TEST_READS
#undef SNICHOLLS_KV_TEST_WRITES

static_assert(kv_readable<reader_only> && !kv_writable<reader_only>);
static_assert(key_value_store<complete>);
static_assert(kv_writable<no_snapshot> && !key_value_store<no_snapshot>, "snapshot() is part of the contract");
static_assert(!key_value_store<snapshot_not_readable>, "a snapshot must be readable");
static_assert(key_value_store<snapshot_by_handle>, "a snapshot may be a handle to a reader");
static_assert(!kv_readable<no_scan>, "scan() is required");
static_assert(!kv_writable<no_insert>, "insert() is required");
static_assert(!kv_writable<put_returns_bool>, "a failure is a kv_errc, never a bool");
static_assert(!kv_readable<read_not_const>, "reads work on a const store");
static_assert(!kv_readable<scan_with_wrong_visitor>, "a scan visitor answers whether to continue");
static_assert(!kv_readable<string_keys>, "keys are byte spans");
static_assert(!kv_readable<errors_of_another_type>, "errors are kv_errc");
static_assert(!kv_writable<batch_of_strings>, "a batch is a span of kv_write");
static_assert(!kv_readable<int> && !key_value_store<std::string>);

// --- compile-time vocabulary -------------------------------------------------------------------------------------

constexpr std::array<std::byte, 1> b_a{std::byte{'a'}};
constexpr std::array<std::byte, 2> b_ab{std::byte{'a'}, std::byte{'b'}};
constexpr std::array<std::byte, 1> b_b{std::byte{'b'}};
constexpr std::array<std::byte, 1> b_ff{std::byte{0xFF}};

static_assert(kv_less(b_a, b_ab) && kv_less(b_ab, b_b) && !kv_less(b_b, b_a));
static_assert(kv_less(b_b, b_ff), "0xFF sorts after every ASCII byte: unsigned order");
static_assert(kv_less(kv_bytes{}, b_a) && !kv_less(kv_bytes{}, kv_bytes{}));
static_assert(kv_equal(b_a, b_a) && !kv_equal(b_a, b_ab));
static_assert(kv_starts_with(b_ab, b_a) && !kv_starts_with(b_a, b_ab) && kv_starts_with(b_a, kv_bytes{}));

static_assert(kv_range::all().contains(kv_bytes{}) && kv_range::all().contains(b_ff) && !kv_range::all().empty());
static_assert(kv_range::from(b_ab).contains(b_ab) && kv_range::from(b_ab).contains(b_b) && !kv_range::from(b_ab).contains(b_a));
static_assert(kv_range::between(b_a, b_b).contains(b_a) && kv_range::between(b_a, b_b).contains(b_ab));
static_assert(!kv_range::between(b_a, b_b).contains(b_b), "the upper bound is exclusive");
static_assert(kv_range::between(b_b, b_a).empty() && kv_range::between(b_a, b_a).empty() && !kv_range::between(b_a, b_b).empty());
static_assert(kv_range::prefixed(b_a).contains(b_a) && kv_range::prefixed(b_a).contains(b_ab) && !kv_range::prefixed(b_a).contains(b_b));
static_assert(kv_range::prefixed(b_a).beyond(b_b) && !kv_range::prefixed(b_a).beyond(b_ab));
static_assert(!kv_range::prefixed(b_a).after(b_a).contains(b_a) && kv_range::prefixed(b_a).after(b_a).contains(b_ab), "after() is exclusive");
static_assert(kv_range::prefixed(b_a).after(b_a).lower_open() && kv_range::between(b_a, b_b).bounded());
static_assert(noexcept(kv_range::prefixed(b_a).contains(b_a)) && noexcept(kv_less(b_a, b_b)));

static_assert(kv_write::put(b_a, b_b).op == kv_op::put && kv_write::insert(b_a, b_b).op == kv_op::insert);
static_assert(kv_write::erase(b_a).op == kv_op::erase && kv_write::erase(b_a).value.empty());
static_assert(std::is_trivially_copyable_v<kv_range> && std::is_trivially_copyable_v<kv_write>);
static_assert(kv_digest_v1_tag.size() == 18 && kv_digest_v1_tag.back() == '\0');

std::string text(kv_bytes b) { return {reinterpret_cast<const char*>(b.data()), b.size()}; }

} // namespace

TEST(KvErrc, EveryErrorIsNonZeroAndDistinct)
{
    const std::array all{kv_errc::unavailable, kv_errc::io, kv_errc::corrupt, kv_errc::read_only, kv_errc::conflict,
                         kv_errc::limit, kv_errc::invalid_argument, kv_errc::out_of_memory, kv_errc::backend};
    for (std::size_t i = 0; i < all.size(); ++i) {
        EXPECT_NE(static_cast<int>(all[i]), 0);
        for (std::size_t j = i + 1; j < all.size(); ++j)
            EXPECT_NE(all[i], all[j]);
    }
}

TEST(KvBytes, AnyOneByteSequenceIsViewedWithoutCopy)
{
    const std::string s = "key";
    EXPECT_EQ(kv_bytes_of(s).data(), reinterpret_cast<const std::byte*>(s.data()));
    EXPECT_EQ(kv_bytes_of("lit").size(), 3u);   // a literal's length is N - 1
    EXPECT_EQ(kv_bytes_of(std::u8string_view(u8"é")).size(), 2u);
    const std::vector<std::byte> v{std::byte{1}, std::byte{2}};
    EXPECT_EQ(kv_bytes_of(v).size(), 2u);
    EXPECT_TRUE(kv_bytes_of(static_cast<const char*>(nullptr)).empty());
}

TEST(KvWriteBatch, CopiesItsBytesAndKeepsTheirAddressesAsItGrows)
{
    kv_write_batch batch;
    EXPECT_TRUE(batch.empty());
    std::string key = "k0";
    batch.put(key, "v0");
    key = "changed";   // the batch holds a copy
    const kv_bytes first = batch.writes()[0].key;
    for (int i = 1; i < 100; ++i)
        batch.put("k" + std::to_string(i), "v");
    batch.insert("i", "1").erase("e");
    ASSERT_EQ(batch.size(), 102u);
    EXPECT_EQ(batch.writes()[0].key.data(), first.data()) << "storage never moves";
    EXPECT_EQ(text(batch.writes()[0].key), "k0");
    EXPECT_EQ(text(batch.writes()[0].value), "v0");
    EXPECT_EQ(batch.writes()[100].op, kv_op::insert);
    EXPECT_EQ(batch.writes()[101].op, kv_op::erase);
    EXPECT_TRUE(batch.writes()[101].value.empty());
    const std::span<const kv_write> as_span = batch;
    EXPECT_EQ(as_span.size(), 102u);
    batch.clear();
    EXPECT_TRUE(batch.empty());
}

TEST(KvConveniences, AcceptStringsAndBytesAndReturnResults)
{
    reference_key_value_store s;
    ASSERT_TRUE(kv_put(s, std::string("a"), "1").is_ok());
    ASSERT_TRUE(kv_put(s, std::vector<std::byte>{std::byte{'b'}}, std::string_view("2")).is_ok());
    EXPECT_EQ(*kv_get(s, "a"), std::optional<std::string>("1"));
    EXPECT_EQ(*kv_get(s, "missing"), std::nullopt);
    std::vector<std::byte> bytes;
    EXPECT_TRUE(*kv_get(s, "b", bytes));
    EXPECT_EQ(bytes, std::vector<std::byte>{std::byte{'2'}});
    EXPECT_TRUE(*kv_contains(s, "a"));
    EXPECT_FALSE(*kv_insert(s, "a", "x"));
    EXPECT_TRUE(*kv_insert(s, "c", "3"));
    EXPECT_TRUE(*kv_erase(s, "c"));
    EXPECT_EQ(*kv_count(s), 2u);
    std::string seen;
    EXPECT_EQ(*kv_scan(s, kv_range::all(), [&](kv_bytes k, kv_bytes) { seen += text(k); }), 2u);   // void visitor
    EXPECT_EQ(seen, "ab");
    EXPECT_EQ(*kv_scan_prefix(s, "a", [](kv_bytes, kv_bytes) { return true; }), 1u);
    std::size_t length = 0;
    EXPECT_TRUE(*kv_read(s, "a", [&](kv_bytes v) { length = v.size(); }));
    EXPECT_EQ(length, 1u);
}

TEST(KvConveniences, PassStorageErrorsThrough)
{
    reference_key_value_store s;
    s.set_unavailable(true);
    EXPECT_EQ(kv_get(s, "a").unwrap_err(), kv_errc::unavailable);
    std::string out;
    EXPECT_EQ(kv_get(s, "a", out).unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(kv_count(s).unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(kv_digest_v1(s).unwrap_err(), kv_errc::unavailable);
    EXPECT_EQ(kv_put(s, "a", "1").unwrap_err(), kv_errc::unavailable);
}

namespace {
std::string digest_of(std::initializer_list<std::pair<std::string_view, std::string_view>> pairs,
                      const kv_range& range = kv_range::all())
{
    reference_key_value_store s;
    for (const auto& [k, v] : pairs)
        static_cast<void>(kv_put(s, k, v));
    return std::string(kv_digest_v1(s, range)->hex().view());
}
} // namespace

// Golden vectors computed independently (Python's hashlib over the stream the definition gives).
TEST(KvDigestV1, MatchesTheGoldenVectors)
{
    EXPECT_EQ(digest_of({}), "49cc8b474da690ee60d4ed608adda992b1849d0ad932af2522fab22b51926bf7");
    EXPECT_EQ(digest_of({{"k", "v"}}), "46bcb6a12f89e26a7a1ae7e7c4521e92f215b4f4e758b9a71ee442364c9a1084");
    EXPECT_EQ(digest_of({{"", ""}}), "4f16f1ce0b78fd2f921d150633940b441c3ad3c35730ff200ecc51ff71a3fa57");
    EXPECT_EQ(digest_of({{"aaa", "3"}, {"a", "1"}, {"aa", "2"}}),
              "bc612522a5a114924c29a3027669712af0a0d85625495db6e2da3f649d3934dd");
    EXPECT_EQ(digest_of({{std::string_view("\xff", 1), std::string_view("\0", 1)}, {std::string_view("\0\xff", 2), "\x80"}}),
              "970600518e628d242ec3feda0e0af94a5d52bb98c783c04c577ac619821fd038");
}

TEST(KvDigestV1, CoversOnlyItsRangeAndSeesEveryByte)
{
    const std::string whole = digest_of({{"k", "v"}, {"z", "outside"}}, kv_range::prefixed(kv_bytes_of("k")));
    EXPECT_EQ(whole, digest_of({{"k", "v"}})) << "keys outside the range do not count";
    EXPECT_NE(digest_of({{"ab", "c"}}), digest_of({{"a", "bc"}})) << "lengths are hashed, so splits differ";
    EXPECT_NE(digest_of({}), digest_of({{"", ""}})) << "the pair count is hashed";
}
