//
//  intrusive_avl_tree_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/containers/intrusive_avl_tree.hpp>.
//

// Demo: snicholls::containers::intrusive_avl_tree - one template, linkage and ordering as policies. The same tree type
// serves an ascending and a descending index over the same nodes (the bid/ask and buy-stop/sell-stop trees of a
// limit order book), with no allocation by the tree. Exits non-zero if a result is not as printed.

#include <cstdio>
#include <functional>
#include <vector>

#include <snicholls/core/containers/intrusive_avl_tree.hpp>

namespace {

int failures = 0;
void expect(bool ok, const char* what)
{
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) ++failures;
}

//! [tree]
struct level
{
    long price = 0;
    snicholls::containers::avl_hook<level> by_price_up;     // linkage for the ascending tree
    snicholls::containers::avl_hook<level> by_price_down;   // a second, independent linkage
};

using ascending = snicholls::containers::intrusive_avl_tree<
    level, snicholls::containers::member_avl_traits<level, &level::by_price_up, &level::price, std::less<>>>;
using descending = snicholls::containers::intrusive_avl_tree<
    level, snicholls::containers::member_avl_traits<level, &level::by_price_down, &level::price, std::greater<>>>;
//! [tree]

} // namespace

int main()
{
    std::printf("intrusive_avl_tree: nodes owned by the caller, linked into two trees at once\n");
    std::vector<level> levels(8);
    ascending asks;
    descending bids;
    const long prices[] = {105, 101, 108, 103, 102, 107, 104, 106};
    for (int i = 0; i < 8; ++i) {
        levels[i].price = prices[i];
        asks.insert(levels[i]);
        bids.insert(levels[i]);
    }
    std::printf("  ascending: ");
    for (const level& l : asks) std::printf("%ld ", l.price);
    std::printf("\n  descending:");
    for (const level& l : bids) std::printf(" %ld", l.price);
    std::printf("\n");
    expect(asks.front()->price == 101 && bids.front()->price == 108, "front() is the best of each ordering, O(1)");
    expect(asks.find(104) == &levels[6], "find(104) returns the caller's node");
    expect(asks.lower_bound(104)->price == 104 && asks.upper_bound(104)->price == 105, "lower_bound / upper_bound");
    expect(!asks.insert(levels[0]).second, "inserting an existing key links nothing");

    asks.erase(levels[1]);   // 101 leaves the ascending tree only
    expect(asks.front()->price == 102 && bids.size() == 8, "erase from one tree leaves the other intact");
    expect(ascending::next(asks.front())->price == 103 && ascending::prev(asks.front()) == nullptr, "next / prev");
    auto last = asks.end();
    --last;
    expect(last->price == 108, "--end() is back()");
    expect(asks.validate() && bids.validate(), "validate(): order, links, heights, balance");
    std::printf("  height %d for %zu nodes\n", asks.height(), asks.size());

    int disposed = 0;
    bids.clear_and_dispose([&](level&) { ++disposed; });
    asks.clear();
    expect(disposed == 8 && bids.empty() && asks.empty(), "clear_and_dispose visits every node once");
    return failures == 0 ? 0 : 1;
}
