//
//  masking_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/masking.hpp>: block sizes within their ranges and the papers' bounds, rounding halves
//  to even; I-JEPA's masks deterministic from (seed, step, sample), sorted, target blocks rectangles of the step's
//  size, the same size across a batch, context and targets disjoint, the context above min_keep, the relaxation when
//  the targets leave too little; V-JEPA's tubes spanning every frame group, their union the targets and the rest the
//  context; random masks of the stated count; the collation by truncation (equal counts) and by padding (lengths);
//  compile-time use and the noexcept claims.
//

#include <snicholls/core/ml/masking.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

static_assert(ml::detail::masking::round_even(2.5) == 2 && ml::detail::masking::round_even(3.5) == 4
              && ml::detail::masking::round_even(2.4999) == 2 && ml::detail::masking::round_even(2.51) == 3);
static_assert(noexcept(ml::sample_block_extent({}, {}, 0.5, ml::block_rule::ijepa)));

// The extremes of I-JEPA's ranges on a 14 x 14 grid: u = 0 gives scale 0.15 (keep 29) at aspect 0.75, u -> 1 scale 0.2
// (keep 39) at aspect 1.5
constexpr ml::token_grid g14{1, 14, 14};
constexpr ml::block_sampler targets{0.15, 0.2, 0.75, 1.5, 1.0, 1.0, 4};
static_assert(ml::sample_block_extent(g14, targets, 0.0, ml::block_rule::ijepa).height == 5);   // round(sqrt(29 * 0.75)) = round(4.66)
static_assert(ml::sample_block_extent(g14, targets, 0.0, ml::block_rule::ijepa).width == 6);    // round(sqrt(29 / 0.75)) = round(6.22)
// The context block: scale 1 is the whole grid, lowered below it under I-JEPA's rule and kept under V-JEPA's
static_assert(ml::sample_block_extent(g14, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1}, 0.5, ml::block_rule::ijepa).height == 13);
static_assert(ml::sample_block_extent(g14, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1}, 0.5, ml::block_rule::vjepa).height == 14);
// Temporal scale 0.5 of 8 frame groups is 4
static_assert(ml::sample_block_extent({8, 14, 14}, {0.15, 0.15, 1.0, 1.0, 0.5, 0.5, 1}, 0.3, ml::block_rule::vjepa).frames == 4);

constexpr bool at_compile_time()
{
    ml::ijepa_mask_config c{};
    std::array<std::uint32_t, 196> context{};
    std::array<std::uint32_t, 4 * 196> t{};
    std::array<std::size_t, 4> counts{};
    const auto n = ml::ijepa_masks(c, 1, 2, 3, context, t, counts);
    return n.context > c.min_keep && n.targets == counts[0] + counts[1] + counts[2] + counts[3];
}
static_assert(at_compile_time());

struct ijepa_sample {
    std::vector<std::uint32_t> context = std::vector<std::uint32_t>(196), targets = std::vector<std::uint32_t>(4 * 196);
    std::array<std::size_t, 4> counts{};
    ml::mask_counts n{};
    ijepa_sample(const ml::ijepa_mask_config& c, std::uint64_t seed, std::uint64_t step, std::uint64_t sample)
    {
        n = ml::ijepa_masks(c, seed, step, sample, context, targets, counts);
        context.resize(n.context);
    }
    [[nodiscard]] std::vector<std::uint32_t> block(std::size_t b) const
    {
        return {targets.begin() + std::ptrdiff_t(b * 196), targets.begin() + std::ptrdiff_t(b * 196 + counts[b])};
    }
};

bool sorted_unique(const std::vector<std::uint32_t>& v)
{
    return std::adjacent_find(v.begin(), v.end(), [](auto a, auto b) { return a >= b; }) == v.end();
}

TEST(MlIjepaMasks, AreDeterministicFromSeedStepAndSample)
{
    const ml::ijepa_mask_config c{};
    const ijepa_sample a(c, 7, 100, 3), b(c, 7, 100, 3), other(c, 7, 100, 4), later(c, 7, 101, 3);
    EXPECT_EQ(a.context, b.context);
    EXPECT_EQ(a.targets, b.targets);
    EXPECT_NE(a.context, other.context);
    EXPECT_NE(a.targets, later.targets);
}

TEST(MlIjepaMasks, TargetsAreBlocksOfTheStepsSizeAndTheContextAvoidsThem)
{
    const ml::ijepa_mask_config c{};
    for (std::uint64_t step = 0; step < 20; ++step) {
        std::size_t first_count = 0;
        for (std::uint64_t sample = 0; sample < 8; ++sample) {
            const ijepa_sample s(c, 42, step, sample);
            EXPECT_TRUE(sorted_unique(s.context));
            EXPECT_GT(s.n.context, c.min_keep);
            std::set<std::uint32_t> hidden;
            for (std::size_t b = 0; b < 4; ++b) {
                const auto blk = s.block(b);
                ASSERT_FALSE(blk.empty());
                EXPECT_TRUE(sorted_unique(blk));
                // A rectangle: its rows and columns span exactly its count
                const std::uint32_t y0 = blk.front() / 14, x0 = blk.front() % 14, y1 = blk.back() / 14, x1 = blk.back() % 14;
                const std::size_t h = y1 - y0 + 1, w = x1 - x0 + 1;
                EXPECT_EQ(h * w, blk.size());
                EXPECT_LT(h, 14u);
                EXPECT_LT(w, 14u);
                // Its scale within [0.15, 0.2] of the grid and its aspect within [0.75, 1.5], up to the rounding of h and w
                EXPECT_GE(double(h * w), 0.15 * 196 * 0.75);
                EXPECT_LE(double(h * w), 0.2 * 196 * 1.25);
                EXPECT_GE(double(h) / double(w), 0.75 * 0.7);
                EXPECT_LE(double(h) / double(w), 1.5 * 1.3);
                if (sample == 0 && b == 0)
                    first_count = blk.size();
                EXPECT_EQ(blk.size(), first_count);   // one size for the whole step's batch
                hidden.insert(blk.begin(), blk.end());
            }
            for (const auto t : s.context)
                EXPECT_EQ(hidden.count(t), 0u) << t;
        }
    }
}

TEST(MlIjepaMasks, TooLittleContextDropsTargetsFromItsConstraint)
{
    // Four targets of half a 6 x 6 grid leave a 5 x 5 context fewer than 11 tokens nearly always: after every 20
    // failures one target stops constraining it, and a context above min_keep is found
    ml::ijepa_mask_config c{};
    c.grid = {1, 6, 6};
    c.targets = {0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 4};
    c.context = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1};
    std::vector<std::uint32_t> context(36), t(4 * 36);
    std::array<std::size_t, 4> counts{};
    for (std::uint64_t sample = 0; sample < 10; ++sample) {
        const auto n = ml::ijepa_masks(c, 3, 0, sample, context, t, counts);
        EXPECT_GT(n.context, c.min_keep);
        EXPECT_EQ(counts[0], 16u);   // keep 18, round(sqrt 18) = 4
    }
}

TEST(MlVjepaMasks, TubesSpanEveryFrameAndTheContextIsTheRest)
{
    for (const auto& c : {ml::vjepa_short_range, ml::vjepa_long_range}) {
        const std::size_t n = c.grid.tokens(), plane = c.grid.height * c.grid.width;
        std::vector<std::uint32_t> context(n), targets(n);
        for (std::uint64_t sample = 0; sample < 6; ++sample) {
            const auto counts = ml::vjepa_masks(c, 9, 5, sample, context, targets);
            ASSERT_EQ(counts.context + counts.targets, n);
            EXPECT_GT(counts.context, 0u);
            std::vector<std::uint32_t> ctx(context.begin(), context.begin() + std::ptrdiff_t(counts.context)),
                tgt(targets.begin(), targets.begin() + std::ptrdiff_t(counts.targets));
            EXPECT_TRUE(sorted_unique(ctx));
            EXPECT_TRUE(sorted_unique(tgt));
            std::vector<std::uint32_t> all;
            std::merge(ctx.begin(), ctx.end(), tgt.begin(), tgt.end(), std::back_inserter(all));
            for (std::size_t i = 0; i < n; ++i)
                ASSERT_EQ(all[i], i);   // disjoint, and together every token
            // Temporal scale 1: a hidden position is hidden in every frame group
            std::set<std::uint32_t> hidden(tgt.begin(), tgt.end());
            for (const auto t : tgt)
                for (std::size_t f = 0; f < c.grid.frames; ++f)
                    EXPECT_EQ(hidden.count(std::uint32_t(f * plane + t % plane)), 1u);
            // Deterministic
            std::vector<std::uint32_t> again(n), again_t(n);
            const auto twice = ml::vjepa_masks(c, 9, 5, sample, again, again_t);
            EXPECT_EQ(twice.context, counts.context);
            EXPECT_TRUE(std::equal(ctx.begin(), ctx.end(), again.begin()));
        }
    }
}

TEST(MlRandomMasks, HideTheStatedFractionUniformly)
{
    constexpr std::size_t n = 196;
    std::vector<std::uint32_t> context(n), targets(n);
    const auto c = ml::random_masks(n, 0.75, 1, 2, 3, context, targets);
    EXPECT_EQ(c.context, std::size_t(196 * 0.25));
    EXPECT_EQ(c.targets, n - c.context);
    std::vector<std::uint32_t> ctx(context.begin(), context.begin() + 49), tgt(targets.begin(), targets.begin() + 147), all;
    EXPECT_TRUE(sorted_unique(ctx));
    EXPECT_TRUE(sorted_unique(tgt));
    std::merge(ctx.begin(), ctx.end(), tgt.begin(), tgt.end(), std::back_inserter(all));
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_EQ(all[i], i);
    // Every token is hidden about three times in four over many samples
    std::vector<int> times(n);
    for (std::uint64_t s = 0; s < 400; ++s) {
        const auto k = ml::random_masks(n, 0.75, 1, 2, s, context, targets);
        for (std::size_t i = 0; i < k.targets; ++i)
            ++times[targets[i]];
    }
    for (const int t : times) {
        EXPECT_GT(t, 240);
        EXPECT_LT(t, 360);
    }
    EXPECT_EQ(ml::random_masks(10, 0.0, 1, 2, 3, context, targets).targets, 0u);
    EXPECT_EQ(ml::random_masks(10, 1.0, 1, 2, 3, context, targets).context, 0u);
}

TEST(MlCollateIndices, TruncatesToTheCommonCountOrPadsWithLengths)
{
    const std::vector<std::uint32_t> lists{1, 4, 6, 9, 0, 2, 3, 0, 0, 0, 5, 7, 8, 11, 12};
    const std::array<std::size_t, 3> counts{4, 2, 5};
    EXPECT_EQ(ml::common_count(counts), 2u);
    EXPECT_EQ(ml::common_count({}), 0u);
    std::vector<std::uint32_t> cut(6);
    ml::collate_indices(lists, 5, counts, 2, cut, {});
    EXPECT_EQ(cut, (std::vector<std::uint32_t>{1, 4, 2, 3, 5, 7}));
    std::vector<std::uint32_t> padded(15);
    std::array<std::size_t, 3> lengths{};
    ml::collate_indices(lists, 5, counts, 5, padded, lengths);
    EXPECT_EQ(padded, (std::vector<std::uint32_t>{1, 4, 6, 9, 9, 2, 3, 3, 3, 3, 5, 7, 8, 11, 12}));
    EXPECT_EQ(lengths, (std::array<std::size_t, 3>{4, 2, 5}));
}

} // namespace
