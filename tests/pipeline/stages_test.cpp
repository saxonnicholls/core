//
//  stages_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/pipeline/stages.hpp>.
//

// components/emitter_observer/stages.hpp: compile-time observer chains.

#include <functional>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <snicholls/core/pipeline/stages.hpp>

namespace {

using namespace snicholls::pipeline;

struct even
{
    bool operator()(int v) const { return v % 2 == 0; }
};
struct twice
{
    int operator()(int v) const { return 2 * v; }
};
struct collect
{
    std::vector<int>* out;
    void operator()(int v) const { out->push_back(v); }
};

TEST(Stages, FilterTransformTapCompose)
{
    std::vector<int> out;
    int tapped = 0;
    auto chain = filter(even{}) | transform(twice{}) | tap([&](int) { ++tapped; }) | collect{&out};
    for (int i = 0; i < 6; ++i) chain(i);
    EXPECT_EQ(out, (std::vector<int>{0, 4, 8}));
    EXPECT_EQ(tapped, 3);
}

TEST(Stages, ChainsAreFlatStructsWithoutIndirection)
{
    // Empty function objects and an empty sink: the whole chain is empty, i.e. costs no state at all.
    struct sink
    {
        void operator()(int) const {}
    };
    auto chain = filter(even{}) | transform(twice{}) | sink{};
    static_assert(std::is_empty_v<even>);
    EXPECT_LE(sizeof(chain), 3u);
    static_assert(std::is_trivially_copyable_v<decltype(chain)>);
    static_assert(!std::is_polymorphic_v<decltype(chain)>);
}

TEST(Stages, BatchEmitsFullBatchesAndFlushesTheRest)
{
    std::vector<std::size_t> sizes;
    auto chain = batch<int>(3) | [&](const std::vector<int>& b) { sizes.push_back(b.size()); };
    for (int i = 0; i < 7; ++i) chain(i);
    EXPECT_EQ(sizes, (std::vector<std::size_t>{3, 3}));
    chain.flush();
    chain.flush();   // nothing left: no call
    EXPECT_EQ(sizes, (std::vector<std::size_t>{3, 3, 1}));
    auto one = batch<int>(0) | [&](const std::vector<int>& b) { sizes.push_back(b.size() + 100); };
    one(1);   // size 0 means 1
    EXPECT_EQ(sizes.back(), 101u);
}

TEST(Stages, SinkByReference)
{
    struct counter
    {
        int n = 0;
        void operator()(int) { ++n; }
    } c;
    auto chain = filter(even{}) | std::ref(c);
    chain(2);
    chain(3);
    EXPECT_EQ(c.n, 1);
}

} // namespace
