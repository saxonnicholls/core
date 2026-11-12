//
//  index_pool_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/index_pool.hpp>.
//

// snicholls::containers::index_pool: layout constants, format and attach, LIFO reuse of indices, full, rebase after
// the region grows or moves (indices keep their meaning), and every refusal of a region.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <set>
#include <span>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/containers/index_pool.hpp>

namespace {

using snicholls::containers::index_pool;
using snicholls::containers::index_pool_error;

struct order
{
    std::int64_t id;
    std::int32_t quantity;
    std::int32_t side;
};

struct alignas(64) wide
{
    std::byte b[64];
};

using pool32 = index_pool<order>;
using pool64 = index_pool<order, std::uint64_t>;

static_assert(std::is_trivially_copyable_v<pool32>, "the pool is a view, copied freely");
static_assert(pool32::slot_size == 16);
static_assert(pool32::header_size == 64);
static_assert(pool32::bytes_for(10) == 64 + 160);
static_assert(index_pool<std::uint8_t>::slot_size == 4, "a slot holds at least an index");
static_assert(index_pool<std::uint8_t, std::uint64_t>::slot_size == 8);
static_assert(index_pool<wide>::region_alignment == 64 && index_pool<wide>::header_size == 64);
static_assert(pool32::npos == 0xffffffffu);

// A region of n slots on 64-byte alignment, as a mapping would be.
struct region
{
    std::vector<std::byte> bytes;
    std::size_t offset = 0;
    explicit region(std::size_t n_bytes) : bytes(n_bytes + 64)
    {
        auto addr = reinterpret_cast<std::uintptr_t>(bytes.data());
        offset = (64 - addr % 64) % 64;
    }
    std::span<std::byte> span() { return {bytes.data() + offset, bytes.size() - 64}; }
};

TEST(IndexPoolFormat, AnEmptyPoolHasTheCapacityTheRegionHolds)
{
    region r(pool32::bytes_for(10) + 7);   // the 7 spare bytes are not a slot
    auto p = pool32::format(r.span());
    ASSERT_TRUE(p.is_ok());
    EXPECT_TRUE(p->attached());
    EXPECT_EQ(p->capacity(), 10u);
    EXPECT_EQ(p->live(), 0u);
    EXPECT_EQ(p->high_water(), 0u);
    EXPECT_EQ(p->data(), r.span().data());
    EXPECT_FALSE(pool32{}.attached());
}

TEST(IndexPoolCreate, HandsOutIndicesInOrderAndHoldsTheValues)
{
    region r(pool32::bytes_for(8));
    auto pool = *pool32::format(r.span());
    for (std::uint32_t i = 0; i < 8; ++i) {
        auto ix = pool.create(order{i, static_cast<std::int32_t>(10 * i), 1});
        ASSERT_TRUE(ix.is_ok());
        EXPECT_EQ(*ix, i);
    }
    for (std::uint32_t i = 0; i < 8; ++i) EXPECT_EQ(pool[i].quantity, static_cast<std::int32_t>(10 * i));
    EXPECT_EQ(pool.live(), 8u);
    EXPECT_EQ(pool.high_water(), 8u);
    const pool32& view = pool;
    EXPECT_EQ(view[3].id, 3);
}

TEST(IndexPoolFull, ReportsFullAndChangesNothing)
{
    region r(pool32::bytes_for(2));
    auto pool = *pool32::format(r.span());
    (void)pool.create(order{1, 1, 1});
    (void)pool.create(order{2, 2, 2});
    EXPECT_EQ(pool.create(order{3, 3, 3}).err(), index_pool_error::full);
    EXPECT_EQ(pool.allocate().err(), index_pool_error::full);
    EXPECT_EQ(pool.live(), 2u);
    EXPECT_EQ(pool[1].id, 2);
}

TEST(IndexPoolReuse, DestroyedIndicesComeBackLastInFirstOut)
{
    region r(pool32::bytes_for(8));
    auto pool = *pool32::format(r.span());
    for (int i = 0; i < 5; ++i) (void)pool.create(order{i, 0, 0});
    pool.destroy(1);
    pool.destroy(3);
    EXPECT_EQ(pool.live(), 3u);
    EXPECT_EQ(*pool.allocate(), 3u);
    EXPECT_EQ(*pool.create(order{99, 0, 0}), 1u);
    EXPECT_EQ(pool[1].id, 99);
    EXPECT_EQ(*pool.allocate(), 5u);   // free list empty: the high-water mark moves
    EXPECT_EQ(pool.high_water(), 6u);
}

TEST(IndexPoolPersistence, AttachSeesTheSamePoolIncludingItsFreeList)
{
    region r(pool32::bytes_for(16));
    {
        auto pool = *pool32::format(r.span());
        for (int i = 0; i < 6; ++i) (void)pool.create(order{i, i, i});
        pool.destroy(2);
        pool.destroy(4);
    }
    auto again = pool32::attach(r.span());
    ASSERT_TRUE(again.is_ok());
    EXPECT_EQ(again->live(), 4u);
    EXPECT_EQ(again->high_water(), 6u);
    EXPECT_EQ(again->capacity(), 16u);
    EXPECT_EQ((*again)[5].id, 5);
    EXPECT_EQ(*again->allocate(), 4u);   // the free list survived
    EXPECT_EQ(*again->allocate(), 2u);
}

TEST(IndexPoolRebase, IndicesKeepTheirMeaningWhenTheRegionGrowsAndMoves)
{
    region small(pool32::bytes_for(4));
    auto pool = *pool32::format(small.span());
    for (int i = 0; i < 4; ++i) (void)pool.create(order{100 + i, i, 0});
    pool.destroy(0);
    (void)pool.allocate();   // takes 0 back
    EXPECT_EQ(pool.create(order{0, 0, 0}).err(), index_pool_error::full);

    // Grow by copying into a larger region at another address, as a remap would.
    region big(pool32::bytes_for(64));
    std::memcpy(big.span().data(), small.span().data(), small.span().size());
    ASSERT_TRUE(pool.rebase(big.span()).is_ok());
    EXPECT_EQ(pool.data(), big.span().data());
    EXPECT_EQ(pool.capacity(), 64u);
    EXPECT_EQ(pool[3].id, 103);
    auto next = pool.create(order{104, 4, 0});
    ASSERT_TRUE(next.is_ok());
    EXPECT_EQ(*next, 4u);
    EXPECT_EQ(pool.live(), 5u);
}

TEST(IndexPoolRebase, RefusesARegionTooSmallForTheUsedSlotsAndStaysPut)
{
    region r(pool32::bytes_for(8));
    auto pool = *pool32::format(r.span());
    for (int i = 0; i < 6; ++i) (void)pool.create(order{i, 0, 0});
    EXPECT_EQ(pool.rebase(r.span().first(pool32::bytes_for(5))).err(), index_pool_error::too_small);
    EXPECT_EQ(pool.capacity(), 8u);
    ASSERT_TRUE(pool.rebase(r.span().first(pool32::bytes_for(6))).is_ok());
    EXPECT_EQ(pool.capacity(), 6u);
    EXPECT_EQ(pool.allocate().err(), index_pool_error::full);
}

TEST(IndexPoolRefusal, FormatRefusesAMisalignedOrShortRegion)
{
    region r(pool32::bytes_for(4));
    EXPECT_EQ(pool32::format(r.span().subspan(4)).err(), index_pool_error::misaligned);
    EXPECT_EQ(pool32::format(r.span().first(pool32::header_size - 1)).err(), index_pool_error::too_small);
    EXPECT_TRUE(pool32::format(r.span().first(pool32::header_size)).is_ok());   // a pool of no slots
    EXPECT_EQ(pool32::attach(r.span())->capacity(), 4u);
}

TEST(IndexPoolRefusal, AttachRefusesWhatIsNotItsPool)
{
    region r(pool32::bytes_for(4));
    std::fill(r.bytes.begin(), r.bytes.end(), std::byte{0});
    EXPECT_EQ(pool32::attach(r.span()).err(), index_pool_error::not_formatted);
    EXPECT_EQ(pool32::attach(r.span().subspan(4)).err(), index_pool_error::misaligned);
    EXPECT_EQ(pool32::attach(r.span().first(10)).err(), index_pool_error::too_small);

    (void)pool32::format(r.span());
    EXPECT_EQ(pool64::attach(r.span()).err(), index_pool_error::layout_mismatch);   // another index width
    EXPECT_EQ((index_pool<std::int64_t>::attach(r.span()).err()), index_pool_error::layout_mismatch);   // slot size
}

TEST(IndexPoolRefusal, AttachRefusesCountersThatContradictEachOther)
{
    region r(pool32::bytes_for(4));
    auto pool = *pool32::format(r.span());
    for (int i = 0; i < 3; ++i) (void)pool.create(order{i, 0, 0});
    auto poke = [&](std::size_t word, std::uint64_t v) { std::memcpy(r.span().data() + 8 * word, &v, 8); };
    auto peek = [&](std::size_t word) {
        std::uint64_t v;
        std::memcpy(&v, r.span().data() + 8 * word, 8);
        return v;
    };
    const std::vector<std::byte> good(r.span().begin(), r.span().end());
    auto restore = [&] { std::copy(good.begin(), good.end(), r.span().begin()); };

    poke(5, 4);   // live 4 > high water 3
    EXPECT_EQ(pool32::attach(r.span()).err(), index_pool_error::corrupt);
    restore();
    poke(4, 3);   // a free head at the high-water mark
    EXPECT_EQ(pool32::attach(r.span()).err(), index_pool_error::corrupt);
    restore();
    poke(5, 2);   // one slot neither live nor free
    EXPECT_EQ(pool32::attach(r.span()).err(), index_pool_error::corrupt);
    restore();
    poke(3, 9);   // high water past the region
    poke(5, 9);
    EXPECT_EQ(pool32::attach(r.span()).err(), index_pool_error::too_small);
    restore();
    EXPECT_EQ(peek(3), 3u);
    EXPECT_TRUE(pool32::attach(r.span()).is_ok());
}

TEST(IndexPoolWide, SixtyFourBitIndicesAndOverAlignedSlots)
{
    region r(index_pool<wide, std::uint64_t>::bytes_for(3));
    auto pool = *index_pool<wide, std::uint64_t>::format(r.span());
    wide w{};
    w.b[63] = std::byte{7};
    std::set<std::uint64_t> seen;
    for (int i = 0; i < 3; ++i) seen.insert(*pool.create(w));
    EXPECT_EQ(seen.size(), 3u);
    for (std::uint64_t i : seen) {
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&pool[i]) % 64, 0u);
        EXPECT_EQ(pool[i].b[63], std::byte{7});
    }
    static_assert(std::is_same_v<decltype(pool)::index_type, std::uint64_t>);
}

} // namespace
