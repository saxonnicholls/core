//
//  kv_cells_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/kv_cells.hpp>: the sequence operations against hand-worked transitions of llama.cpp's
//  llama_kv_cells (llama.cpp 39039aa), the K-shift against keys rotated afresh at their new positions (plain RoPE,
//  interleaved, YaRN, after seq_div), attention over cells against attend and attend_alibi to the bit, masking by
//  sequence and window, and compile-time use.
//

#include <snicholls/core/ml/kv_cells.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;

TEST(MlKvCells, SequenceOperationsAreLlamaCpps)
{
    ml::kv_cells<8> cells;
    EXPECT_EQ(cells.used(), 0u);
    EXPECT_EQ(cells.find_free(), 0u);
    // A prompt of 4 tokens in sequence 0, shared with sequence 1 (llama_kv_cache::seq_cp)
    for (std::size_t i = 0; i < 4; ++i)
        cells.place(i, ml::kv_pos(i), 0);
    cells.seq_cp(0, 1, -1, -1);
    EXPECT_TRUE(cells.seq_has(2, 1));
    // Each sequence continues on its own: positions 4 and 5 of sequence 0, 4 of sequence 1
    cells.place(4, 4, 0);
    cells.place(5, 5, 0);
    cells.place(6, 4, 1);
    EXPECT_EQ(cells.used(), 7u);
    EXPECT_EQ(cells.seq_pos_max(0), 5);
    EXPECT_EQ(cells.seq_pos_max(1), 4);
    EXPECT_EQ(cells.find_free(), 7u);
    // Remove sequence 0 from position 2 on: its own cells are freed, the shared ones keep sequence 1
    EXPECT_EQ(cells.seq_rm(0, 2, -1), 4u);
    EXPECT_FALSE(cells.is_empty(2));
    EXPECT_FALSE(cells.seq_has(2, 0));
    EXPECT_TRUE(cells.is_empty(4) && cells.is_empty(5));
    EXPECT_EQ(cells.pos(4), -1);
    EXPECT_EQ(cells.seq_pos_max(0), 1);
    // Keep sequence 1 only: sequence 0's cells (0 and 1 are shared, so stay) - seq_keep frees what 1 does not have
    cells.place(4, 2, 0);
    EXPECT_EQ(cells.seq_keep(1), 4u);
    EXPECT_TRUE(cells.seq_has(0, 1) && !cells.seq_has(0, 0));
    EXPECT_EQ(cells.used(), 5u);
    // Remove everything in [3, 5) whatever the sequence
    EXPECT_EQ(cells.seq_rm(-1, 3, 5), 3u);
    EXPECT_EQ(cells.used(), 3u);
    EXPECT_EQ(cells.find_free(5), 5u);
    EXPECT_EQ(cells.find_free(7), 7u);
}

TEST(MlKvCells, ShiftsAddUpAndCellsMovedBelowZeroAreFreed)
{
    ml::kv_cells<6> cells;
    for (std::size_t i = 0; i < 6; ++i)
        cells.place(i, ml::kv_pos(10 + i), 0);
    // Context shift: drop positions 10 and 11, move the rest back by 2
    cells.seq_rm(0, 10, 12);
    cells.seq_add(0, 12, -1, -2);
    EXPECT_TRUE(cells.has_shift());
    EXPECT_EQ(cells.pos(2), 10);
    EXPECT_EQ(cells.shift(2), -2);
    cells.seq_add(0, -1, -1, -11);   // 10 - 11 < 0: freed; 13 - 11 = 2 stays
    EXPECT_TRUE(cells.is_empty(2));
    EXPECT_EQ(cells.pos(5), 2);
    EXPECT_EQ(cells.shift(5), -13);
    // seq_div: 2 / 2 = 1, the shift records the change 1 - 2
    cells.seq_div(0, -1, -1, 2);
    EXPECT_EQ(cells.pos(5), 1);
    EXPECT_EQ(cells.shift(5), -14);
    cells.reset_shift();
    EXPECT_FALSE(cells.has_shift());
    EXPECT_EQ(cells.shift(5), 0);
    EXPECT_EQ(cells.seq_pos_min(0), 0);   // 11 - 11 = 0, halved
}

constexpr ml::transformer_config cfg{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 8, .d_ff = 8, .rope = ml::rope_kind::half};
constexpr ml::transformer_config cfg_j{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 8, .d_ff = 8,
                                       .rope = ml::rope_kind::interleaved, .rope_base = 500000};
constexpr ml::transformer_config cfg_yarn{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 8, .d_ff = 8, .rope = ml::rope_kind::half,
                                          .rope_scaling = ml::rope_scaling_kind::yarn, .rope_factor = 4, .rope_original_context = 64};

std::array<double, 8> raw_key(std::size_t token, std::size_t layer)
{
    std::array<double, 8> k{};
    for (std::size_t i = 0; i < 8; ++i)
        k[i] = std::sin(0.37 * double(i) + 1.3 * double(token) + 0.7 * double(layer));
    return k;
}

template <ml::transformer_config C>
std::array<double, 8> rotated(std::size_t token, std::size_t layer, std::size_t pos)
{
    auto k = raw_key(token, layer);
    std::array<double, 4> c{}, s{};
    ml::rope_angles(pos, c, s, ml::rope_base_v<C, double>, ml::rope_scaling_v<C, double>);
    ml::rope(k, c, s, C.layout());
    return k;
}

template <ml::transformer_config C>
void shift_moves_keys()
{
    constexpr std::size_t layers = 2, capacity = 8;
    std::vector<double> keys(ml::kv_cache_view<C, double>::required_size(layers, capacity)), values(keys.size());
    const ml::kv_cache_view<C, double> view(keys, values, layers, capacity);
    ml::kv_cells<capacity> cells;
    for (std::size_t t = 0; t < capacity; ++t) {
        for (std::size_t l = 0; l < layers; ++l) {
            const auto k = rotated<C>(t, l, 100 + t);
            view.append(l, t, k, k);
        }
        cells.place(t, ml::kv_pos(100 + t), 0);
    }
    // Drop the first three tokens and move the rest back by three; then halve every position (self-extend)
    cells.seq_rm(0, 100, 103);
    cells.seq_add(0, 103, -1, -3);
    ml::kv_shift_keys<C>(std::span<double>(keys), layers, cells);
    EXPECT_FALSE(cells.has_shift());
    for (std::size_t t = 3; t < capacity; ++t)
        for (std::size_t l = 0; l < layers; ++l) {
            const auto want = rotated<C>(t, l, 97 + t);
            const auto got = view.visible(l, 0, ml::key_range{t, t + 1}).keys[0];
            for (std::size_t i = 0; i < 8; ++i)
                EXPECT_NEAR(got[i], want[i], 1e-12) << t << " " << l << " " << i;
        }
    cells.seq_div(0, -1, -1, 2);
    ml::kv_shift_keys<C>(std::span<double>(keys), layers, cells);
    for (std::size_t t = 3; t < capacity; ++t) {
        EXPECT_EQ(cells.pos(t), ml::kv_pos((97 + t) / 2));
        const auto want = rotated<C>(t, 1, (97 + t) / 2);   // moved to its new position, whatever the sign convention
        const auto got = view.visible(1, 0, ml::key_range{t, t + 1}).keys[0];
        for (std::size_t i = 0; i < 8; ++i)
            EXPECT_NEAR(got[i], want[i], 1e-12) << t << " " << i;
    }
}

TEST(MlKvCells, TheKShiftMovesEachKeyToItsNewPosition)
{
    shift_moves_keys<cfg>();
    shift_moves_keys<cfg_j>();
    shift_moves_keys<cfg_yarn>();
}

TEST(MlKvCells, AttendingCellsInOrderIsAttendAndAttendAlibiToTheBit)
{
    constexpr std::size_t capacity = 6;
    std::vector<float> keys(ml::kv_cache_view<cfg, float>::required_size(1, capacity)), values(keys.size());
    const ml::kv_cache_view<cfg, float> view(keys, values, 1, capacity);
    ml::kv_cells<capacity> cells;
    std::vector<float> flat_k, flat_v;
    for (std::size_t t = 0; t < capacity; ++t) {
        std::array<float, 8> k{}, v{};
        for (std::size_t i = 0; i < 8; ++i) {
            k[i] = std::sin(float(i * 3 + t));
            v[i] = std::cos(float(i + 5 * t));
        }
        view.append(0, t, k, v);
        cells.place(t, ml::kv_pos(t), 0);
        flat_k.insert(flat_k.end(), k.begin(), k.end());
        flat_v.insert(flat_v.end(), v.begin(), v.end());
    }
    std::array<float, 8> q{0.3f, -0.2f, 0.9f, 0.1f, -0.5f, 0.4f, 0.0f, 0.7f}, a{}, b{};
    const float scale = 1.0f / std::sqrt(8.0f);
    EXPECT_EQ(ml::attend_cells(view, cells, 0, 0, q, a, 0, 5), ml::attend(q, flat_k, flat_v, b, scale));
    EXPECT_EQ(a, b);
    EXPECT_EQ(ml::attend_cells(view, cells, 0, 0, q, a, 0, 5, 0.25f), ml::attend_alibi(q, flat_k, flat_v, b, scale, 0.25f, 5, 0));
    EXPECT_EQ(a, b);
    // Causal: a query at position 2 sees cells 0 to 2
    ml::attend_cells(view, cells, 0, 0, q, a, 0, 2);
    ml::attend(q, std::span(flat_k).first(24), std::span(flat_v).first(24), b, scale);
    EXPECT_EQ(a, b);
    // Another sequence's cell is not seen: move cell 1 to sequence 1
    cells.seq_cp(0, 1, 1, 2);
    cells.seq_rm(0, 1, 2);
    ml::attend_cells(view, cells, 0, 0, q, a, 0, 2);
    std::vector<float> k02(flat_k.begin(), flat_k.begin() + 8), v02(flat_v.begin(), flat_v.begin() + 8);
    k02.insert(k02.end(), flat_k.begin() + 16, flat_k.begin() + 24);
    v02.insert(v02.end(), flat_v.begin() + 16, flat_v.begin() + 24);
    ml::attend(q, k02, v02, b, scale);
    EXPECT_EQ(a, b);
    // Nothing visible: zeros and -infinity
    EXPECT_EQ(ml::attend_cells(view, cells, 0, 0, q, a, 5, 3), -std::numeric_limits<float>::infinity());
    EXPECT_EQ(a, (std::array<float, 8>{}));
}

TEST(MlKvCells, TheWindowIsLlamaCppsStandardSlidingWindow)
{
    ml::kv_cells<8> cells;
    for (std::size_t i = 0; i < 8; ++i)
        cells.place(i, ml::kv_pos(i), 0);
    // is_masked_swa (standard): masked when query_pos - pos >= window
    EXPECT_TRUE(cells.visible(4, 0, 7, 4));
    EXPECT_FALSE(cells.visible(3, 0, 7, 4));
    EXPECT_FALSE(cells.visible(7, 0, 6, 0));   // after the query
    EXPECT_TRUE(cells.visible(0, 0, 7, 0));
}

constexpr bool compile_time()
{
    ml::kv_cells<4> c;
    c.place(0, 0, 0);
    c.place(1, 1, 0);
    c.place(2, 2, 0);
    c.seq_cp(0, 3, 0, 2);
    c.seq_rm(0, 1, -1);
    c.seq_add(3, -1, -1, 5);
    return c.pos(0) == 5 && c.pos(1) == 6 && c.is_empty(2) && c.seq_has(0, 0) && !c.seq_has(1, 0) && c.seq_pos_max(3) == 6
           && c.find_free() == 2 && c.shift(1) == 5;
}
static_assert(compile_time());
static_assert(std::is_trivially_copyable_v<ml::kv_cells<16>>);

} // namespace
