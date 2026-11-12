//
//  mlp_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/mlp.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using W = ml::dense_weights<float>;

template <ml::mlp_kind K>
constexpr ml::transformer_config shape{.d_model = 4, .n_heads = 1, .n_kv_heads = 1, .head_dim = 4, .d_ff = 8, .mlp = K};

std::vector<float> pattern(std::size_t n, float scale, std::size_t salt = 0)
{
    std::vector<float> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = scale * (float((i * 37 + salt * 11) % 19) / 19.0f - 0.5f);
    return v;
}

// down(act(up x + b1) [· (gate x + bg)]) + b2, in double, from the definitions
template <ml::mlp_kind K>
std::vector<double> reference(const std::vector<float>& up, const std::vector<float>& gate, const std::vector<float>& down,
                              const std::vector<float>& b1, const std::vector<float>& b2, const std::vector<float>& x)
{
    std::vector<double> h(8), y(4);
    for (std::size_t i = 0; i < 8; ++i) {
        double u = b1[i], g = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            u += double(up[i * 4 + j]) * x[j];
            g += double(gate.empty() ? 0.0f : gate[i * 4 + j]) * x[j];
        }
        switch (K) {
        case ml::mlp_kind::gelu: h[i] = 0.5 * u * (1 + std::erf(u / std::sqrt(2.0))); break;
        case ml::mlp_kind::gelu_tanh: h[i] = 0.5 * u * (1 + std::tanh(std::sqrt(2 / std::numbers::pi) * (u + 0.044715 * u * u * u))); break;
        case ml::mlp_kind::relu_squared: h[i] = u > 0 ? u * u : 0; break;
        case ml::mlp_kind::swiglu: h[i] = g / (1 + std::exp(-g)) * u; break;
        }
    }
    for (std::size_t i = 0; i < 4; ++i) {
        y[i] = b2[i];
        for (std::size_t j = 0; j < 8; ++j)
            y[i] += double(down[i * 8 + j]) * h[j];
    }
    return y;
}

template <ml::mlp_kind K>
void matches_its_definition()
{
    const auto up = pattern(32, 1.0f, 1), gate = pattern(32, 1.0f, 2), down = pattern(32, 1.0f, 3), b1 = pattern(8, 0.2f, 4),
               b2 = pattern(4, 0.2f, 5), x = pattern(4, 2.0f, 6);
    const bool swiglu = K == ml::mlp_kind::swiglu;
    const ml::mlp_weights<W> w{W{up, 8, 4}, swiglu ? W{gate, 8, 4} : W{}, W{down, 4, 8}, b1, {}, b2};
    std::array<float, 4> y{};
    ml::mlp_forward<shape<K>>(w, x, y);
    const auto want = reference<K>(up, swiglu ? gate : std::vector<float>{}, down, b1, b2, x);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(y[i], want[i], 1e-6) << int(K) << " " << i;
}

TEST(MlMlp, EachKindMatchesItsDefinition)
{
    matches_its_definition<ml::mlp_kind::gelu>();
    matches_its_definition<ml::mlp_kind::gelu_tanh>();
    matches_its_definition<ml::mlp_kind::relu_squared>();
    matches_its_definition<ml::mlp_kind::swiglu>();
}

TEST(MlMlp, TheActivationsAreTheActivationsHeaders)
{
    EXPECT_EQ(ml::mlp_activation<ml::mlp_kind::gelu>(0.7f), ml::gelu(0.7f));
    EXPECT_EQ(ml::mlp_activation<ml::mlp_kind::gelu_tanh>(0.7f), ml::gelu_tanh(0.7f));
    EXPECT_EQ(ml::mlp_activation<ml::mlp_kind::relu_squared>(-0.7f), 0.0f);
    EXPECT_EQ(ml::mlp_activation<ml::mlp_kind::swiglu>(0.7f), ml::silu(0.7f));
}

constexpr bool compile_time()
{
    constexpr ml::transformer_config c{.d_model = 1, .n_heads = 1, .n_kv_heads = 1, .head_dim = 1, .d_ff = 2, .mlp = ml::mlp_kind::relu_squared};
    const std::array<double, 2> up{2, -1}, down{1, 1};
    const std::array<double, 1> x{3};
    std::array<double, 1> y{};
    ml::mlp_forward<c>(ml::mlp_weights<ml::dense_weights<double>, double>{{up, 2, 1}, {}, {down, 1, 2}}, x, y);
    return y[0] == 36;   // relu(6)² + relu(-3)²
}
static_assert(compile_time());

} // namespace

namespace {

TEST(MlMlpMoe, RoutesToTheKLargestRouterLogitsWithASoftmaxOverThem)
{
    const std::array<float, 5> router{0.5f, 2.0f, -1.0f, 2.0f, 1.0f};
    const auto r = ml::moe_gate<2>(router);
    ASSERT_EQ(r.count, 2u);
    EXPECT_EQ(r.expert, (std::array<std::size_t, 2>{1, 3}));   // a tie: the lower-numbered expert first
    EXPECT_FLOAT_EQ(r.weight[0], 0.5f);
    EXPECT_FLOAT_EQ(r.weight[1], 0.5f);
    const auto three = ml::moe_gate<3>(router);
    const float e = std::exp(-1.0f), sum = 2.0f + e;
    EXPECT_EQ(three.expert[2], 4u);
    EXPECT_NEAR(three.weight[2], e / sum, 1e-7f);
    EXPECT_NEAR(three.weight[0] + three.weight[1] + three.weight[2], 1.0f, 1e-6f);
    const auto all = ml::moe_gate<8>(std::array<float, 2>{0.0f, 0.0f});   // more slots than experts
    EXPECT_EQ(all.count, 2u);
}

static_assert(ml::moe_gate<1>(std::array<double, 3>{1, 3, 2}).expert[0] == 1 && ml::moe_gate<1>(std::array<double, 3>{1, 3, 2}).weight[0] == 1);

} // namespace
