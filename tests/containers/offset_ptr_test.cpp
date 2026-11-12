//
//  offset_ptr_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/offset_ptr.hpp>.
//

// snicholls::containers::offset_ptr: null, assignment and copy (recomputed for the new address), comparison, and a
// linked list that still walks after its whole buffer is copied to another address.

#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/containers/offset_ptr.hpp>

namespace {

using snicholls::containers::offset_ptr;

static_assert(sizeof(offset_ptr<int>) == sizeof(std::ptrdiff_t));
static_assert(std::is_nothrow_copy_constructible_v<offset_ptr<int>>);
static_assert(!std::is_trivially_copyable_v<offset_ptr<int>>, "a copy recomputes the offset");
static_assert(std::is_trivially_destructible_v<offset_ptr<int>>);

TEST(OffsetPtrNull, DefaultAndNullptrAreNull)
{
    offset_ptr<int> a;
    offset_ptr<int> b = nullptr;
    int* raw = nullptr;
    offset_ptr<int> c = raw;
    EXPECT_FALSE(a);
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_TRUE(b == nullptr);
    EXPECT_TRUE(c == nullptr);
    EXPECT_EQ(a.offset(), 1);
    EXPECT_TRUE(a == b);
}

TEST(OffsetPtrAssign, PointsWhereItWasToldAndStoresTheDistance)
{
    struct pair
    {
        offset_ptr<int> p;
        int value = 5;
    } s;
    s.p = &s.value;
    EXPECT_TRUE(s.p);
    EXPECT_EQ(*s.p, 5);
    EXPECT_EQ(s.p.offset(), static_cast<std::ptrdiff_t>(offsetof(pair, value)));
    *s.p = 6;
    EXPECT_EQ(s.value, 6);
    s.p = nullptr;
    EXPECT_FALSE(s.p);
}

TEST(OffsetPtrCopy, ACopyAtAnotherAddressPointsAtTheSameTarget)
{
    int target = 1;
    offset_ptr<int> a = &target;
    offset_ptr<int> b(a);
    offset_ptr<int> c;
    c = a;
    EXPECT_EQ(b.get(), &target);
    EXPECT_EQ(c.get(), &target);
    EXPECT_NE(b.offset(), a.offset());   // different addresses, different distances
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a == &target);
    auto& self = a;
    a = self;
    EXPECT_EQ(a.get(), &target);
}

TEST(OffsetPtrSelf, MayPointAtItself)
{
    struct node
    {
        offset_ptr<node> next;
    } n;
    n.next = &n;
    EXPECT_EQ(n.next.offset(), 0);
    EXPECT_EQ(n.next.get(), &n);
}

TEST(OffsetPtrArrow, ReachesMembers)
{
    struct thing
    {
        int x = 9;
    } t;
    const offset_ptr<thing> p = &t;
    EXPECT_EQ(p->x, 9);
}

struct node
{
    int value;
    offset_ptr<node> next;
};

TEST(OffsetPtrRegion, ALinkedListSurvivesAByteCopyOfItsWholeRegion)
{
    constexpr int n = 50;
    alignas(node) std::byte a[sizeof(node) * n];
    alignas(node) std::byte b[sizeof(node) * n];
    node* nodes = reinterpret_cast<node*>(a);
    // Link them out of order: 0 -> 49 -> 1 -> 48 -> ...
    std::vector<int> order;
    for (int lo = 0, hi = n - 1; lo <= hi; ++lo, --hi) {
        order.push_back(lo);
        if (lo != hi) order.push_back(hi);
    }
    for (int i = 0; i < n; ++i) ::new (&nodes[i]) node{i * 10, nullptr};
    for (std::size_t k = 0; k + 1 < order.size(); ++k) nodes[order[k]].next = &nodes[order[k + 1]];

    std::memcpy(b, a, sizeof a);   // the region moves as a whole
    std::memset(a, 0xff, sizeof a);   // and the old copy is gone
    const node* walk = std::launder(reinterpret_cast<node*>(b)) + order[0];
    std::vector<int> seen;
    for (; walk; walk = walk->next.get()) seen.push_back(walk->value / 10);
    EXPECT_EQ(seen, order);
}

} // namespace
