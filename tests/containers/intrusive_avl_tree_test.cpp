//
//  intrusive_avl_tree_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/containers/intrusive_avl_tree.hpp>.
//

// snicholls::containers::intrusive_avl_tree: model-checked against std::set (both orderings), validated after every
// operation; iteration, bounds, multiple hooks, clear_and_dispose, moves, and adversarial (monotonic) inserts.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/containers/intrusive_avl_tree.hpp>

namespace {

using snicholls::containers::avl_hook;
using snicholls::containers::intrusive_avl_tree;
using snicholls::containers::member_avl_traits;

struct node
{
    std::int64_t key = 0;
    avl_hook<node> by_key;     // tree 1
    avl_hook<node> by_key_2;   // tree 2 (a second, independent membership)
};

using ascending = intrusive_avl_tree<node, member_avl_traits<node, &node::by_key, &node::key, std::less<>>>;
using descending = intrusive_avl_tree<node, member_avl_traits<node, &node::by_key, &node::key, std::greater<>>>;
using second_hook = intrusive_avl_tree<node, member_avl_traits<node, &node::by_key_2, &node::key, std::less<>>>;

static_assert(snicholls::containers::avl_traits<member_avl_traits<node, &node::by_key, &node::key>, node>);
static_assert(std::bidirectional_iterator<ascending::iterator>);
static_assert(std::bidirectional_iterator<ascending::const_iterator>);
static_assert(noexcept(std::declval<ascending&>().erase(std::declval<node&>())));

template <class Tree>
std::vector<std::int64_t> keys(const Tree& t)
{
    std::vector<std::int64_t> out;
    for (const node& n : t) out.push_back(n.key);
    return out;
}

TEST(IntrusiveAvlTreeBasics, EmptyTreeIsValid)
{
    ascending t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.front(), nullptr);
    EXPECT_EQ(t.back(), nullptr);
    EXPECT_EQ(t.height(), 0);
    EXPECT_EQ(t.find(1), nullptr);
    EXPECT_EQ(t.begin(), t.end());
    EXPECT_TRUE(t.validate());
}

TEST(IntrusiveAvlTreeBasics, InsertFindEraseKeepOrderAndCachedEnds)
{
    std::vector<node> nodes(5);
    const std::int64_t ks[] = {30, 10, 50, 20, 40};
    ascending t;
    for (int i = 0; i < 5; ++i) {
        nodes[i].key = ks[i];
        EXPECT_TRUE(t.insert(nodes[i]).second);
        EXPECT_TRUE(t.validate());
    }
    EXPECT_EQ(keys(t), (std::vector<std::int64_t>{10, 20, 30, 40, 50}));
    EXPECT_EQ(t.front()->key, 10);
    EXPECT_EQ(t.back()->key, 50);
    EXPECT_EQ(t.find(40), &nodes[4]);
    EXPECT_EQ(t.lower_bound(25)->key, 30);
    EXPECT_EQ(t.lower_bound(30)->key, 30);
    EXPECT_EQ(t.upper_bound(30)->key, 40);
    EXPECT_EQ(t.upper_bound(50), nullptr);
    EXPECT_EQ(t.lower_bound(51), nullptr);
    t.erase(nodes[1]);   // 10, the front
    EXPECT_FALSE(nodes[1].by_key.is_linked());
    EXPECT_EQ(t.front()->key, 20);
    t.erase(nodes[2]);   // 50, the back
    EXPECT_EQ(t.back()->key, 40);
    EXPECT_TRUE(t.validate());
    EXPECT_EQ(keys(t), (std::vector<std::int64_t>{20, 30, 40}));
}

TEST(IntrusiveAvlTreeBasics, DuplicateKeyReturnsExistingNodeAndLinksNothing)
{
    node a{7, {}, {}}, b{7, {}, {}};
    ascending t;
    ASSERT_TRUE(t.insert(a).second);
    auto [existing, inserted] = t.insert(b);
    EXPECT_FALSE(inserted);
    EXPECT_EQ(existing, &a);
    EXPECT_FALSE(b.by_key.is_linked());
    EXPECT_EQ(t.size(), 1u);
    EXPECT_TRUE(t.validate());
}

TEST(IntrusiveAvlTreeOrdering, GreaterPutsTheLargestKeyFirst)
{
    std::vector<node> nodes(4);
    descending t;
    for (int i = 0; i < 4; ++i) {
        nodes[i].key = i * 10;
        t.insert(nodes[i]);
    }
    EXPECT_EQ(keys(t), (std::vector<std::int64_t>{30, 20, 10, 0}));
    EXPECT_EQ(t.front()->key, 30);
    EXPECT_EQ(t.lower_bound(25)->key, 20);   // first not ordered before 25 under greater
    EXPECT_TRUE(t.validate());
}

TEST(IntrusiveAvlTreeIteration, IteratorsWalkBothWaysIncludingFromEnd)
{
    std::vector<node> nodes(6);
    ascending t;
    for (int i = 0; i < 6; ++i) {
        nodes[i].key = i;
        t.insert(nodes[i]);
    }
    auto it = t.end();
    --it;
    EXPECT_EQ(it->key, 5);
    std::vector<std::int64_t> back;
    for (auto r = t.end(); r != t.begin();) back.push_back((--r)->key);
    EXPECT_EQ(back, (std::vector<std::int64_t>{5, 4, 3, 2, 1, 0}));
    const ascending& ct = t;
    ascending::const_iterator ci = t.begin();   // mutable -> const conversion
    EXPECT_EQ(ci, ct.begin());
    EXPECT_EQ((ci++)->key, 0);
    EXPECT_EQ(ci->key, 1);
    EXPECT_EQ(ascending::prev(&nodes[0]), nullptr);
    EXPECT_EQ(ascending::next(&nodes[5]), nullptr);
}

TEST(IntrusiveAvlTreeHooks, ANodeCanSitInTwoTreesThroughTwoHooks)
{
    std::vector<node> nodes(10);
    ascending a;
    second_hook b;
    for (int i = 0; i < 10; ++i) {
        nodes[i].key = 9 - i;
        a.insert(nodes[i]);
        if (i % 2 == 0) b.insert(nodes[i]);
    }
    a.erase(nodes[0]);   // leaves tree b untouched
    EXPECT_TRUE(a.validate());
    EXPECT_TRUE(b.validate());
    EXPECT_EQ(a.size(), 9u);
    EXPECT_EQ(b.size(), 5u);
    EXPECT_EQ(b.back()->key, 9);
}

TEST(IntrusiveAvlTreeBalance, MonotonicInsertsStayLogarithmic)
{
    // The unbalanced reference tree degenerates to a list here.
    constexpr int n = 100000;
    std::vector<node> nodes(n);
    ascending t;
    for (int i = 0; i < n; ++i) {
        nodes[i].key = i;
        t.insert(nodes[i]);
    }
    EXPECT_TRUE(t.validate());
    EXPECT_LE(t.height(), 25);   // 1.44 log2(100002) ~ 23.9
    for (int i = 0; i < n; i += 2) t.erase(nodes[i]);
    EXPECT_TRUE(t.validate());
    EXPECT_LE(t.height(), 24);
}

TEST(IntrusiveAvlTreeClear, ClearAndDisposeUnlinksEveryNodeOnce)
{
    std::vector<node> nodes(100);
    ascending t;
    for (int i = 0; i < 100; ++i) {
        nodes[i].key = (i * 37) % 100;
        t.insert(nodes[i]);
    }
    int disposed = 0;
    t.clear_and_dispose([&](node& n) {
        EXPECT_FALSE(n.by_key.is_linked());
        ++disposed;
    });
    EXPECT_EQ(disposed, 100);
    EXPECT_TRUE(t.empty());
    EXPECT_TRUE(t.validate());
    EXPECT_TRUE(t.insert(nodes[3]).second);   // reusable afterwards
    t.clear();
    EXPECT_FALSE(nodes[3].by_key.is_linked());
}

TEST(IntrusiveAvlTreeMove, MoveTransfersNodesAndEmptiesTheSource)
{
    std::vector<node> nodes(3);
    ascending a;
    for (int i = 0; i < 3; ++i) {
        nodes[i].key = i;
        a.insert(nodes[i]);
    }
    ascending b(std::move(a));
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(b.size(), 3u);
    ascending c;
    c = std::move(b);
    EXPECT_EQ(keys(c), (std::vector<std::int64_t>{0, 1, 2}));
    EXPECT_TRUE(c.validate());
}

template <class Tree, class Compare>
void model_check(unsigned seed)
{
    std::mt19937_64 rng(seed);
    constexpr int universe = 400;
    std::vector<node> nodes(universe);
    for (int i = 0; i < universe; ++i) nodes[i].key = i;
    std::set<std::int64_t, Compare> model;
    Tree t;
    for (int step = 0; step < 20000; ++step) {
        const int k = static_cast<int>(rng() % universe);
        if (rng() % 3 != 0) {
            const bool inserted = t.insert(nodes[k]).second;
            EXPECT_EQ(inserted, model.insert(k).second);
        } else if (nodes[k].by_key.is_linked()) {
            t.erase(nodes[k]);
            model.erase(k);
        }
        ASSERT_TRUE(t.validate()) << "seed " << seed << " step " << step;
        ASSERT_EQ(t.size(), model.size());
        if (!model.empty()) {
            ASSERT_EQ(t.front()->key, *model.begin());
            ASSERT_EQ(t.back()->key, *model.rbegin());
        }
        if (step % 500 == 0) {
            std::vector<std::int64_t> expect(model.begin(), model.end());
            ASSERT_EQ(keys(t), expect);
            const std::int64_t probe = static_cast<std::int64_t>(rng() % universe);
            auto lb = model.lower_bound(probe);
            node* got = t.lower_bound(probe);
            ASSERT_EQ(got ? got->key : -1, lb == model.end() ? -1 : *lb);
            auto ub = model.upper_bound(probe);
            got = t.upper_bound(probe);
            ASSERT_EQ(got ? got->key : -1, ub == model.end() ? -1 : *ub);
        }
    }
}

TEST(IntrusiveAvlTreeModel, AscendingMatchesStdSetUnderRandomInsertErase)
{
    for (unsigned seed : {1u, 2u, 3u, 42u}) model_check<ascending, std::less<>>(seed);
}

TEST(IntrusiveAvlTreeModel, DescendingMatchesStdSetUnderRandomInsertErase)
{
    for (unsigned seed : {5u, 6u}) model_check<descending, std::greater<>>(seed);
}

} // namespace
