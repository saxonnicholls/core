//
//  kv_cache_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/kv_cache.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr ml::transformer_config two_heads{.d_model = 4, .n_heads = 2, .n_kv_heads = 2, .head_dim = 2, .d_ff = 4};
constexpr ml::transformer_config windowed{.d_model = 4, .n_heads = 2, .n_kv_heads = 2, .head_dim = 2, .d_ff = 4, .window = 3};

static_assert(ml::kv_cache_view<two_heads>::required_size(3, 5) == 3 * 2 * 5 * 2);

// Position p's keys: head h, channel i = 100 p + 10 h + i; values the negatives
std::array<float, 4> keys_of(std::size_t p)
{
    return {100.0f * p, 100.0f * p + 1, 100.0f * p + 10, 100.0f * p + 11};
}
std::array<float, 4> values_of(std::size_t p)
{
    auto k = keys_of(p);
    for (float& x : k)
        x = -x;
    return k;
}

std::vector<float> positions_in(std::span<const float> run)
{
    std::vector<float> p;
    for (std::size_t j = 0; j < run.size(); j += 2)
        p.push_back(run[j]);
    return p;
}

TEST(MlKvCache, AFlatCacheHoldsEveryPositionInOneRun)
{
    std::vector<float> k(ml::kv_cache_view<two_heads>::required_size(2, 4)), v(k.size());
    const ml::kv_cache_view<two_heads> cache(k, v, 2, 4);
    for (std::size_t p = 0; p < 4; ++p)
        cache.append(1, p, keys_of(p), values_of(p));
    const auto runs = cache.visible(1, 1, ml::attention_window(3));
    EXPECT_EQ(positions_in(runs.keys[0]), (std::vector<float>{10, 110, 210, 310}));
    EXPECT_EQ(positions_in(runs.values[0]), (std::vector<float>{-10, -110, -210, -310}));
    EXPECT_TRUE(runs.keys[1].empty());
    // Layer 0 was never written
    EXPECT_EQ(positions_in(cache.visible(0, 0, ml::attention_window(3)).keys[0]), (std::vector<float>{0, 0, 0, 0}));
}

TEST(MlKvCache, ARingWrapsIntoTwoRunsInPositionOrder)
{
    std::vector<float> k(ml::kv_cache_view<windowed>::required_size(1, 3)), v(k.size());
    const ml::kv_cache_view<windowed> cache(k, v, 1, 3);
    for (std::size_t p = 0; p < 5; ++p)
        cache.append(0, p, keys_of(p), values_of(p));
    const auto runs = cache.visible(0, 0, ml::attention_window(4, 3));   // positions 2, 3, 4 in slots 2, 0, 1
    EXPECT_EQ(positions_in(runs.keys[0]), (std::vector<float>{200}));
    EXPECT_EQ(positions_in(runs.keys[1]), (std::vector<float>{300, 400}));
    EXPECT_EQ(positions_in(runs.values[1]), (std::vector<float>{-300, -400}));
}

TEST(MlKvCache, AnEmptyRangeHasNoRuns)
{
    std::vector<float> k(ml::kv_cache_view<two_heads>::required_size(1, 2)), v(k.size());
    const ml::kv_cache_view<two_heads> cache(k, v, 1, 2);
    const auto runs = cache.visible(0, 0, ml::key_range{1, 1});
    EXPECT_TRUE(runs.keys[0].empty() && runs.keys[1].empty());
    EXPECT_EQ(cache.layers(), 1u);
    EXPECT_EQ(cache.capacity(), 2u);
    EXPECT_EQ(ml::kv_cache_view<two_heads>{}.capacity(), 0u);
}

TEST(MlKvCache, AnOwningCacheIsIndependentOfItsCopies)
{
    static ml::kv_cache<two_heads, float, 1, 2> a;
    a.view().append(0, 0, keys_of(1), values_of(1));
    static auto b = a;
    b.view().append(0, 0, keys_of(2), values_of(2));
    EXPECT_EQ(a.view().visible(0, 0, {0, 1}).keys[0][0], 100.0f);
    EXPECT_EQ(b.view().visible(0, 0, {0, 1}).keys[0][0], 200.0f);
}

constexpr bool compile_time()
{
    ml::kv_cache<two_heads, double, 1, 2> c{};
    const std::array<double, 4> k{1, 2, 3, 4}, v{5, 6, 7, 8};
    c.view().append(0, 1, k, v);
    const auto r = c.view().visible(0, 1, {1, 2});
    return r.keys[0][0] == 3 && r.values[0][1] == 8;
}
static_assert(compile_time());

} // namespace
