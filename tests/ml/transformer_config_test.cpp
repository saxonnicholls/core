//
//  transformer_config_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/transformer_config.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <type_traits>

namespace {

namespace ml = snicholls::ml;

static_assert(std::is_trivially_copyable_v<ml::transformer_config>);

static_assert(ml::gpt2_small.q_size() == 768 && ml::gpt2_small.kv_size() == 768 && ml::gpt2_small.group() == 1);
static_assert(ml::llama3_8b.q_size() == 4096 && ml::llama3_8b.kv_size() == 1024 && ml::llama3_8b.group() == 4);
static_assert(ml::nanochat_d20.layout() == ml::rope_layout::half);

constexpr ml::transformer_config base{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 16};
static_assert(ml::valid(base));

constexpr ml::transformer_config with(ml::transformer_config c, auto change)
{
    change(c);
    return c;
}

static_assert(!ml::valid(with(base, [](auto& c) { c.d_model = 0; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.d_ff = 0; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.n_kv_heads = 0; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.n_heads = 3; c.n_kv_heads = 2; })));   // 3 heads cannot share 2
static_assert(!ml::valid(with(base, [](auto& c) { c.rope = ml::rope_kind::half; c.head_dim = 3; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.rope = ml::rope_kind::half; c.rope_base = 1; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.norm_eps = 0; })));
static_assert(!ml::valid(with(base, [](auto& c) { c.attention_softcap = -1; })));
static_assert(ml::valid(with(base, [](auto& c) { c.head_dim = 3; })));   // an odd head is fine without RoPE

// The configuration is a template argument
template <ml::transformer_config C>
constexpr std::size_t heads_v = C.n_heads;
static_assert(heads_v<ml::llama3_8b> == 32);

// Clang reads a double member of a template argument wrongly at run time; the readers evaluate it at compile time.
// This runs the reads where the bug showed (a function the optimiser does not fold), so a regression is a failure.
template <ml::transformer_config C>
[[gnu::noinline]] double read_at_run_time(int which)
{
    return which == 0 ? ml::norm_eps_v<C, double> : which == 1 ? ml::rope_base_v<C, double> : ml::attention_softcap_v<C, double>;
}

TEST(MlTransformerConfig, TheFloatingPointMembersReadCorrectlyAtRunTime)
{
    constexpr ml::transformer_config c{.d_model = 8, .n_heads = 1, .n_kv_heads = 1, .head_dim = 8, .d_ff = 9, .norm_eps = 1e-5,
                                       .rope = ml::rope_kind::half, .rope_base = 500, .window = 7, .attention_softcap = 5};
    EXPECT_EQ(read_at_run_time<c>(0), 1e-5);
    EXPECT_EQ(read_at_run_time<c>(1), 500.0);
    EXPECT_EQ(read_at_run_time<c>(2), 5.0);
    EXPECT_EQ((ml::norm_eps_v<ml::nanochat_d20, float>), std::numeric_limits<float>::epsilon());
}

TEST(MlTransformerConfig, PresetsHaveTheirModelsShapes)
{
    EXPECT_EQ(ml::gpt2_small.norm, ml::norm_kind::layer);
    EXPECT_TRUE(ml::gpt2_small.bias);
    EXPECT_EQ(ml::llama3_8b.mlp, ml::mlp_kind::swiglu);
    EXPECT_EQ(ml::nanochat_d20.qk_norm, ml::qk_norm_kind::after_rope);
    EXPECT_EQ(ml::nanochat_d20.mlp, ml::mlp_kind::relu_squared);
    EXPECT_EQ(with(base, [](auto& c) { c.rope = ml::rope_kind::interleaved; }).layout(), ml::rope_layout::interleaved);
}

} // namespace
