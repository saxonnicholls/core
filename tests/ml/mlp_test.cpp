//
//  mlp_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/mlp.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
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

// The backward pass, against central differences in double: L = g . mlp(x)
using WD = ml::dense_weights<double>;

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.41 * double(i) + b);
    return v;
}

template <ml::mlp_kind K>
void backward_is_central_differences()
{
    constexpr bool swiglu = K == ml::mlp_kind::swiglu;
    auto up = wave(32, 0.9, 0.1), gate = wave(32, 0.8, 1.2), down = wave(32, 0.7, 2.3), bu = wave(8, 0.3, 0.4),
         bg = wave(8, 0.2, 1.1), bd = wave(4, 0.1, 0.7), x = wave(4, 1.5, 0.9);
    const auto g = wave(4, 1.0, 1.7);
    auto weights = [&] {
        return ml::mlp_weights<WD, double>{WD{up, 8, 4}, swiglu ? WD{gate, 8, 4} : WD{}, WD{down, 4, 8}, bu,
                                           swiglu ? std::span<const double>(bg) : std::span<const double>{}, bd};
    };
    auto loss = [&] {
        std::array<double, 4> y{};
        ml::mlp_forward<shape<K>>(weights(), x, y);
        double l = 0;
        for (std::size_t i = 0; i < 4; ++i)
            l += g[i] * y[i];
        return l;
    };
    std::vector<double> su(8), sg(swiglu ? 8 : 0), dx(4), dup(32), dgate(swiglu ? 32 : 0), ddown(32), dbu(8),
        dbg(swiglu ? 8 : 0), dbd(4);
    std::array<double, 4> y{};
    ml::mlp_forward<shape<K>>(weights(), x, y, ml::mlp_activations<double>{su, sg});
    ml::mlp_backward<shape<K>>(weights(), x, ml::mlp_activations<double>{su, sg}, g, dx,
                               ml::mlp_gradients<double>{dup, dgate, ddown, dbu, dbg, dbd});
    auto check = [&](std::vector<double>& p, const std::vector<double>& grad, const char* what) {
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double keep = p[i], e = 1e-6;
            p[i] = keep + e;
            const double hi = loss();
            p[i] = keep - e;
            const double lo = loss();
            p[i] = keep;
            EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 1e-7) << int(K) << " " << what << " " << i;
        }
    };
    check(x, dx, "x");
    check(up, dup, "up");
    check(down, ddown, "down");
    check(bu, dbu, "up_bias");
    check(bd, dbd, "down_bias");
    if (swiglu) {
        check(gate, dgate, "gate");
        check(bg, dbg, "gate_bias");
    }
}

TEST(MlMlp, BackwardOfEachKindMatchesCentralDifferences)
{
    backward_is_central_differences<ml::mlp_kind::gelu>();
    backward_is_central_differences<ml::mlp_kind::gelu_tanh>();
    backward_is_central_differences<ml::mlp_kind::relu_squared>();
    backward_is_central_differences<ml::mlp_kind::swiglu>();
}

template <ml::mlp_kind K>
void saving_gives_the_same_bits()
{
    const auto up = pattern(32, 1.0f, 1), gate = pattern(32, 1.0f, 2), down = pattern(32, 1.0f, 3), b1 = pattern(8, 0.2f, 4),
               b2 = pattern(4, 0.2f, 5), x = pattern(4, 2.0f, 6);
    const bool swiglu = K == ml::mlp_kind::swiglu;
    const ml::mlp_weights<W> w{W{up, 8, 4}, swiglu ? W{gate, 8, 4} : W{}, W{down, 4, 8}, b1, {}, b2};
    std::array<float, 4> plain{}, saving{};
    std::vector<float> su(8), sg(swiglu ? 8 : 0);
    ml::mlp_forward<shape<K>>(w, x, plain);
    ml::mlp_forward<shape<K>>(w, x, saving, ml::mlp_activations<float>{su, sg});
    EXPECT_EQ(plain, saving);
    // The saved up row is the up projection itself
    std::array<float, 8> h{};
    ml::linear(W{up, 8, 4}, x, h, std::span<const float>(b1));
    EXPECT_TRUE(std::equal(h.begin(), h.end(), su.begin()));
}

TEST(MlMlp, SavingThePreActivationsChangesNoOutput)
{
    saving_gives_the_same_bits<ml::mlp_kind::gelu>();
    saving_gives_the_same_bits<ml::mlp_kind::relu_squared>();
    saving_gives_the_same_bits<ml::mlp_kind::swiglu>();
}

// The batched pair over a sequence of three tokens, through the reference backend, against mlp_forward and
// mlp_backward token by token: the same outputs and gradients within rounding
template <ml::mlp_kind K>
void batched_is_per_token()
{
    constexpr bool swiglu = K == ml::mlp_kind::swiglu;
    constexpr std::size_t n = 3, m = n * 8;
    const auto up = wave(32, 0.9, 0.1), gate = wave(32, 0.8, 1.2), down = wave(32, 0.7, 2.3), bu = wave(8, 0.3, 0.4),
               bg = wave(8, 0.2, 1.1), bd = wave(4, 0.1, 0.7), xs = wave(n * 4, 1.5, 0.9), gs = wave(n * 4, 1.0, 1.7);
    const ml::mlp_weights<WD, double> w{WD{up, 8, 4}, swiglu ? WD{gate, 8, 4} : WD{}, WD{down, 4, 8}, bu,
                                        swiglu ? std::span<const double>(bg) : std::span<const double>{}, bd};
    struct grads_t {
        std::vector<double> dx = std::vector<double>(n * 4), up = std::vector<double>(32), gate = std::vector<double>(swiglu ? 32 : 0),
                            down = std::vector<double>(32), bu = std::vector<double>(8), bg = std::vector<double>(swiglu ? 8 : 0),
                            bd = std::vector<double>(4), scratch = std::vector<double>(2 * m);
        ml::mlp_gradients<double> view() { return {up, gate, down, bu, bg, bd, scratch}; }
    } one, many;
    std::vector<double> ys1(n * 4), su(8), sg(swiglu ? 8 : 0);
    for (std::size_t t = 0; t < n; ++t) {
        const auto x = std::span<const double>(xs).subspan(t * 4, 4);
        ml::mlp_forward<shape<K>>(w, x, std::span<double>(ys1).subspan(t * 4, 4), ml::mlp_activations<double>{su, sg});
        ml::mlp_backward<shape<K>>(w, x, ml::mlp_activations<double>{su, sg}, std::span<const double>(gs).subspan(t * 4, 4),
                                   std::span<double>(one.dx).subspan(t * 4, 4), one.view());
    }
    std::vector<double> ys(n * 4), up_s(m), gate_s(swiglu ? m : 0), hidden(m);
    const ml::mlp_activations<double> save{up_s, gate_s, hidden};
    ml::mlp_forward_sequence<shape<K>>(snicholls::linalg::reference, w, xs, ys, save);
    ml::mlp_backward_sequence<shape<K>>(snicholls::linalg::reference, w, xs, save, gs, many.dx, many.view());
    auto near = [](const std::vector<double>& x, const std::vector<double>& y, const char* what) {
        ASSERT_EQ(x.size(), y.size()) << what;
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(x[i], y[i], 1e-12 * std::max(1.0, std::abs(x[i]))) << int(K) << " " << what << " " << i;
    };
    near(ys1, ys, "y");
    near(one.dx, many.dx, "dx");
    near(one.up, many.up, "up");
    near(one.gate, many.gate, "gate");
    near(one.down, many.down, "down");
    near(one.bu, many.bu, "up_bias");
    near(one.bg, many.bg, "gate_bias");
    near(one.bd, many.bd, "down_bias");
}

TEST(MlMlp, BatchedThroughABackendIsThePerTokenPathWithinRounding)
{
    batched_is_per_token<ml::mlp_kind::gelu>();
    batched_is_per_token<ml::mlp_kind::gelu_tanh>();
    batched_is_per_token<ml::mlp_kind::relu_squared>();
    batched_is_per_token<ml::mlp_kind::swiglu>();
}

TEST(MlMlp, ActivationDerivativesAreTheActivations)
{
    for (double v : {-2.0, -0.3, 0.0, 0.4, 1.7}) {
        EXPECT_EQ(ml::mlp_activation_derivative<ml::mlp_kind::gelu>(v), ml::gelu_derivative(v));
        EXPECT_EQ(ml::mlp_activation_derivative<ml::mlp_kind::gelu_tanh>(v), ml::gelu_tanh_derivative(v));
        EXPECT_EQ(ml::mlp_activation_derivative<ml::mlp_kind::relu_squared>(v), ml::relu_squared_derivative(v));
        EXPECT_EQ(ml::mlp_activation_derivative<ml::mlp_kind::swiglu>(v), ml::silu_derivative(v));
    }
}

template <typename Wt>
concept mlp_trainable = requires(const ml::mlp_weights<Wt, float>& w, std::span<const float> x,
                                 const ml::mlp_activations<float>& a, const ml::mlp_gradients<float>& g, std::span<float> dx) {
    ml::mlp_backward<shape<ml::mlp_kind::gelu>>(w, x, a, x, dx, g);
};
static_assert(mlp_trainable<W> && mlp_trainable<ml::dense_weights<ml::bfloat16>>);
static_assert(!mlp_trainable<ml::quantized_weights<ml::block_q8_0>>, "quantised weights are inference only");

TEST(MlMlp, GateBackwardMatchesCentralDifferencesAndLeavesUnchosenExpertsAtZero)
{
    // L = c . weights(logits): the choice of experts is fixed near the point, so differences are valid
    std::vector<double> logits{0.3, 1.9, -0.4, 1.2, 0.8};
    const std::array<double, 3> c{1.0, -2.0, 0.5};
    auto loss = [&] {
        const auto r = ml::moe_gate<3>(logits);
        double l = 0;
        for (std::size_t i = 0; i < r.count; ++i)
            l += c[i] * r.weight[i];
        return l;
    };
    const auto route = ml::moe_gate<3>(logits);
    std::vector<double> d(5, 99.0);
    ml::moe_gate_backward(route, c, d);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const double keep = logits[i], e = 1e-6;
        logits[i] = keep + e;
        const double hi = loss();
        logits[i] = keep - e;
        const double lo = loss();
        logits[i] = keep;
        EXPECT_NEAR(d[i], (hi - lo) / (2 * e), 1e-8) << i;
    }
    EXPECT_EQ(d[0], 0.0);   // expert 0 and 2 were not chosen
    EXPECT_EQ(d[2], 0.0);
    // A route of one expert has weight 1 whatever its logit: no gradient
    const auto one = ml::moe_gate<1>(logits);
    std::vector<double> d1(5);
    ml::moe_gate_backward(one, std::array<double, 1>{3.0}, d1);
    for (double v : d1)
        EXPECT_EQ(v, 0.0);
}

} // namespace
