//
//  jepa_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/jepa.hpp>: the predictor's configuration, validity and parameter count against a count
//  by hand; jepa_predictor_input_rows against a loop by hand to the bit and its backward by central differences; the
//  normalised targets; each loss against its definition, with lengths, and its gradients in the prediction and the
//  target by central differences; the EMA to the bit against PyTorch's two operations, m = 0 copying and m = 1
//  holding, the segments; the momentum schedules; and a whole I-JEPA run on random smooth images through the batched
//  passes on the reference backend, its loss falling over a few hundred steps while the targets' spread stays above a
//  floor (no collapse).
//

#include <snicholls/core/ml/jepa.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/collapse.hpp>
#include <snicholls/core/ml/gather.hpp>
#include <snicholls/core/ml/linear.hpp>
#include <snicholls/core/ml/masking.hpp>
#include <snicholls/core/ml/optim.hpp>
#include <snicholls/core/ml/patch.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace stats = snicholls::stats;
const auto& backend = snicholls::linalg::reference;

static_assert(ml::valid(ml::ijepa_vit_h14_predictor));
static_assert(!ml::valid(ml::jepa_predictor_config{768, 4, ml::gpt2_small}));   // a causal block is no predictor
static_assert(std::is_trivially_copyable_v<ml::jepa_predictor_config>);
// GPT-2 small's block: 2 layer norms (2 · 768 each), Q K V O with biases, the MLP with biases
static_assert(ml::block_parameter_count(ml::gpt2_small) == 4 * 768 + (4 * 768 * 768 + 4 * 768) + (2 * 3072 * 768 + 3072 + 768));
static_assert(ml::jepa_predictor_parameters({16, 2, {.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 16, .norm = ml::norm_kind::rms,
                                                     .mlp = ml::mlp_kind::swiglu, .mask = ml::attention_mask::bidirectional}})
              == (8 * 16) + 8 + 2 * (2 * 8 + 4 * 8 * 8 + 3 * 16 * 8) + 8 + (16 * 8));
static_assert(ml::ema_momentum(0, 100) == 0.996 && ml::ema_momentum(100, 100) == 1.0 && ml::ema_momentum(500, 100) == 1.0);
static_assert(ml::ema_momentum(50, 100, 0.99, 1.0) == 0.99 + 50 * 0.01 / 100);
static_assert(noexcept(ml::ema_rows(backend, std::span<float>{}, std::span<const float>{}, 0.5)));

constexpr bool ema_at_compile_time()
{
    std::array<double, 2> target{1.0, 2.0};
    const std::array<double, 2> online{3.0, 5.0};
    ml::ema_rows(backend, std::span<double>(target), std::span<const double>(online), 0.5);
    return target[0] == 2.0 && target[1] == 3.5;
}
static_assert(ema_at_compile_time());

std::vector<double> wave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.53 * double(i) + b);
    return v;
}

TEST(MlJepaPredictorInput, IsContextPlusPositionsThenMaskTokensAndItsBackward)
{
    constexpr std::size_t w = 3, seqs = 2, kc = 3, kt = 2, tokens = 6;
    auto context = wave(seqs * kc * w, 1.0, 0.1), mask = wave(w, 0.5, 2.0);
    const auto pos = wave(tokens * w, 0.3, 1.0);
    const std::vector<std::uint32_t> ci{0, 2, 5, 1, 3, 4}, ti{1, 3, 0, 5};
    std::vector<double> out(seqs * (kc + kt) * w);
    ml::jepa_predictor_input_rows(backend, std::span<const double>(context), std::span<const double>(mask), std::span<const double>(pos), ci, ti,
                                  std::span<double>(out), w, seqs);
    for (std::size_t s = 0; s < seqs; ++s)
        for (std::size_t c = 0; c < w; ++c) {
            for (std::size_t j = 0; j < kc; ++j)
                EXPECT_EQ(out[(s * 5 + j) * w + c], context[(s * kc + j) * w + c] + pos[ci[s * kc + j] * w + c]);
            for (std::size_t j = 0; j < kt; ++j)
                EXPECT_EQ(out[(s * 5 + kc + j) * w + c], mask[c] + pos[ti[s * kt + j] * w + c]);
        }
    // No positions: the rows as they are
    std::vector<double> bare(out.size());
    ml::jepa_predictor_input_rows(backend, std::span<const double>(context), std::span<const double>(mask), std::span<const double>{}, ci, ti,
                                  std::span<double>(bare), w, seqs);
    EXPECT_EQ(bare[5 * w + kc * w], mask[0]);
    // The backward: a linear map, so <out, g> differentiated is the backward of g
    const auto g = wave(out.size(), 0.7, 0.4);
    std::vector<double> dcontext(context.size()), dmask(w);
    ml::jepa_predictor_input_rows_backward(backend, std::span<const double>(g), kc, kt, std::span<double>(dcontext), std::span<double>(dmask), w, seqs);
    auto loss = [&] {
        std::vector<double> o(out.size());
        ml::jepa_predictor_input_rows(backend, std::span<const double>(context), std::span<const double>(mask), std::span<const double>(pos), ci, ti,
                                      std::span<double>(o), w, seqs);
        double l = 0;
        for (std::size_t i = 0; i < o.size(); ++i)
            l += o[i] * g[i];
        return l;
    };
    for (auto* p : {&context, &mask}) {
        const auto& grad = p == &context ? dcontext : dmask;
        for (std::size_t i = 0; i < p->size(); ++i) {
            const double keep = (*p)[i], e = 1e-6;
            (*p)[i] = keep + e;
            const double hi = loss();
            (*p)[i] = keep - e;
            const double lo = loss();
            (*p)[i] = keep;
            EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 1e-8);
        }
    }
}

TEST(MlJepaTargets, AreRowsNormalisedWithoutParameters)
{
    const auto x = wave(4 * 6, 2.0, 0.3);
    std::vector<double> out(x.size());
    ml::jepa_target_norm_rows(backend, std::span<const double>(x), std::span<double>(out), 6);
    for (std::size_t r = 0; r < 4; ++r) {
        double mean = 0, var = 0;
        for (std::size_t c = 0; c < 6; ++c)
            mean += out[r * 6 + c] / 6;
        for (std::size_t c = 0; c < 6; ++c)
            var += (out[r * 6 + c] - mean) * (out[r * 6 + c] - mean) / 6;
        EXPECT_NEAR(mean, 0.0, 1e-12);
        EXPECT_NEAR(var, 1.0, 1e-4);   // var / (var + 1e-5)
    }
}

// Each loss written out from its definition, over the valid rows
double by_definition(ml::jepa_loss kind, const std::vector<double>& p, const std::vector<double>& t, std::size_t w, std::span<const std::size_t> lengths)
{
    const std::size_t rows = p.size() / w, seqs = lengths.empty() ? 1 : lengths.size(), per = rows / seqs;
    double total = 0;
    std::size_t count = 0;
    for (std::size_t s = 0; s < seqs; ++s)
        for (std::size_t r = s * per; r < s * per + (lengths.empty() ? per : lengths[s]); ++r) {
            ++count;
            if (kind == ml::jepa_loss::cosine) {
                double pt = 0, pp = 0, tt = 0;
                for (std::size_t c = 0; c < w; ++c) {
                    pt += p[r * w + c] * t[r * w + c];
                    pp += p[r * w + c] * p[r * w + c];
                    tt += t[r * w + c] * t[r * w + c];
                }
                total += 1 - pt / (std::sqrt(pp) * std::sqrt(tt));
                continue;
            }
            for (std::size_t c = 0; c < w; ++c) {
                const double d = p[r * w + c] - t[r * w + c], a = std::abs(d);
                total += kind == ml::jepa_loss::l2 ? d * d : kind == ml::jepa_loss::l1 ? a : (a < 1 ? 0.5 * d * d : a - 0.5);
            }
        }
    return kind == ml::jepa_loss::cosine ? total / count : total / (count * w);
}

TEST(MlJepaLoss, EachKindIsItsDefinitionWithItsGradients)
{
    constexpr std::size_t w = 4, rows = 6;
    const std::array<std::size_t, 2> lengths{3, 2};
    for (const auto kind : {ml::jepa_loss::l2, ml::jepa_loss::l1, ml::jepa_loss::smooth_l1, ml::jepa_loss::cosine})
        for (const bool padded : {false, true}) {
            auto p = wave(rows * w, 1.5, 0.2), t = wave(rows * w, 1.2, 1.9);
            const auto len = padded ? std::span<const std::size_t>(lengths) : std::span<const std::size_t>{};
            std::vector<double> dp(p.size()), dt(t.size());
            const double l = ml::jepa_loss_rows(backend, kind, std::span<const double>(p), std::span<const double>(t), w, std::span<double>(dp),
                                                std::span<double>(dt), len, 0.75);
            EXPECT_NEAR(l, 0.75 * by_definition(kind, p, t, w, len), 1e-14);
            for (auto* x : {&p, &t}) {
                const auto& grad = x == &p ? dp : dt;
                for (std::size_t i = 0; i < x->size(); ++i) {
                    const double keep = (*x)[i], e = 1e-7;
                    (*x)[i] = keep + e;
                    const double hi = 0.75 * by_definition(kind, p, t, w, len);
                    (*x)[i] = keep - e;
                    const double lo = 0.75 * by_definition(kind, p, t, w, len);
                    (*x)[i] = keep;
                    EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 1e-7) << int(kind) << " " << padded << " " << i;
                }
            }
        }
    // No valid rows: 0 and no gradient
    std::vector<double> p(8, 1.0), t(8, 0.0), dp(8);
    const std::array<std::size_t, 2> none{0, 0};
    EXPECT_EQ(ml::jepa_loss_rows(backend, ml::jepa_loss::l2, std::span<const double>(p), std::span<const double>(t), 2, std::span<double>(dp), {}, none),
              0.0);
    EXPECT_EQ(dp, std::vector<double>(8, 0.0));
}

TEST(MlEma, IsPyTorchsTwoOperationsToTheBitCopiesAtZeroAndHoldsAtOne)
{
    stats::wyrand g(5);
    std::vector<float> target(1000), online(1000);
    for (std::size_t i = 0; i < target.size(); ++i) {
        target[i] = float(stats::uniform_real(g) - 0.5);
        online[i] = float(stats::uniform_real(g) - 0.5);
    }
    for (const double m : {0.996, 0.9, 0.5}) {
        auto got = target, want = target;
        ml::ema_rows(backend, std::span<float>(got), std::span<const float>(online), m);
        const float mf = float(m), r = float(1.0 - m);
        for (std::size_t i = 0; i < want.size(); ++i) {
            const float kept = want[i] * mf, added = r * online[i];
            want[i] = kept + added;
        }
        for (std::size_t i = 0; i < want.size(); ++i)
            EXPECT_EQ(std::bit_cast<std::uint32_t>(got[i]), std::bit_cast<std::uint32_t>(want[i]));
    }
    auto copy = target, held = target;
    ml::ema_rows(backend, std::span<float>(copy), std::span<const float>(online), 0.0);
    ml::ema_rows(backend, std::span<float>(held), std::span<const float>(online), 1.0);
    EXPECT_EQ(copy, online);
    EXPECT_EQ(held, target);
    // The segments are the per-tensor calls, in order
    auto a = target, b = target;
    const std::array<ml::ema_segment<float>, 2> segs{{{std::span<float>(a).first(300), std::span<const float>(online).first(300)},
                                                      {std::span<float>(a).subspan(300), std::span<const float>(online).subspan(300)}}};
    ml::ema_segments(backend, std::span<const ml::ema_segment<float>>(segs), 0.99);
    ml::ema_rows(backend, std::span<float>(b), std::span<const float>(online), 0.99);
    EXPECT_EQ(a, b);
}

TEST(MlEma, TheMomentumRisesFromM0ToM1)
{
    double prev = 0;
    for (std::uint64_t s = 0; s <= 100; ++s) {
        const double lin = ml::ema_momentum(s, 100), cos = ml::ema_momentum(s, 100, 0.996, 1.0, ml::momentum_schedule::cosine);
        EXPECT_GE(lin, prev);
        EXPECT_GE(cos, 0.996 - 1e-15);
        EXPECT_LE(cos, 1.0);
        prev = lin;
    }
    EXPECT_NEAR(ml::ema_momentum(0, 100, 0.996, 1.0, ml::momentum_schedule::cosine), 0.996, 1e-15);
    EXPECT_NEAR(ml::ema_momentum(50, 100, 0.996, 1.0, ml::momentum_schedule::cosine), 0.998, 1e-15);
    EXPECT_EQ(ml::ema_momentum(100, 100, 0.996, 1.0, ml::momentum_schedule::cosine), 1.0);
}

// ---- A whole I-JEPA run ---------------------------------------------------------------------------------------------

// 8 x 8 one-channel images in patches of 2: 16 tokens of 4 values
constexpr ml::patch_grid grid{.height = 8, .width = 8, .channels = 1, .patch_height = 2, .patch_width = 2};
constexpr std::size_t tokens = grid.tokens(), patch = grid.token_size(), batch = 8, de = 16, dp = 8;
constexpr ml::transformer_config enc{.d_model = de, .n_heads = 2, .n_kv_heads = 2, .head_dim = 8, .d_ff = 32, .norm = ml::norm_kind::layer,
                                     .norm_eps = 1e-6, .mlp = ml::mlp_kind::gelu, .bias = true, .mask = ml::attention_mask::bidirectional};
constexpr ml::jepa_predictor_config pred{de, 1,
                                         {.d_model = dp, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 16, .norm = ml::norm_kind::layer,
                                          .norm_eps = 1e-6, .mlp = ml::mlp_kind::gelu, .bias = true, .mask = ml::attention_mask::bidirectional}};
static_assert(ml::valid(enc) && ml::valid(pred));

// Spans cut one after another from a flat vector: the parameters and their gradients share one layout
struct slab {
    std::span<float> all;
    std::size_t at = 0;
    std::span<float> take(std::size_t n)
    {
        const auto s = all.subspan(at, n);
        at += n;
        return s;
    }
};

using dw = ml::dense_weights<float>;

template <ml::transformer_config C>
struct block_params {
    std::span<float> n1w, n1b, wq, wk, wv, wo, bq, bk, bv, bo, n2w, n2b, up, down, bup, bdown;
    explicit block_params(slab& s)
        : n1w(s.take(C.d_model)), n1b(s.take(C.d_model)), wq(s.take(C.q_size() * C.d_model)), wk(s.take(C.kv_size() * C.d_model)),
          wv(s.take(C.kv_size() * C.d_model)), wo(s.take(C.d_model * C.q_size())), bq(s.take(C.q_size())), bk(s.take(C.kv_size())),
          bv(s.take(C.kv_size())), bo(s.take(C.d_model)), n2w(s.take(C.d_model)), n2b(s.take(C.d_model)), up(s.take(C.d_ff * C.d_model)),
          down(s.take(C.d_model * C.d_ff)), bup(s.take(C.d_ff)), bdown(s.take(C.d_model))
    {
    }
    [[nodiscard]] ml::block_weights<dw, float> weights() const
    {
        return {n1w, n1b,
                {dw{wq, C.q_size(), C.d_model}, dw{wk, C.kv_size(), C.d_model}, dw{wv, C.kv_size(), C.d_model}, dw{wo, C.d_model, C.q_size()}, bq,
                 bk, bv, bo, {}, {}},
                n2w, n2b, {dw{up, C.d_ff, C.d_model}, {}, dw{down, C.d_model, C.d_ff}, bup, {}, bdown}};
    }
};

// What a block over `rows` saves and works in
template <ml::transformer_config C>
struct block_buffers {
    std::size_t rows;
    std::vector<float> x0, h1, x1, h2, q, k, v, y, lse, up, hidden, dq, dk, dv, dy, scratch, mscratch;
    explicit block_buffers(std::size_t r)
        : rows(r), x0(r * C.d_model), h1(x0.size()), x1(x0.size()), h2(x0.size()), q(r * C.q_size()), k(r * C.kv_size()), v(k.size()), y(q.size()),
          lse(r * C.n_heads), up(r * C.d_ff), hidden(up.size()), dq(q.size()), dk(k.size()), dv(k.size()), dy(q.size()), scratch(x0.size()),
          mscratch(up.size())
    {
    }
    ml::block_activations<float> act() { return {x0, h1, x1, h2, {q, k, v, {}, {}, y, lse}, up, {}, hidden}; }
    ml::block_gradients<float> grads(const block_params<C>& g)
    {
        return {g.n1w, g.n1b, g.n2w, g.n2b, {dq, dk, dv, g.wq, g.wk, g.wv, g.wo, g.bq, g.bk, g.bv, g.bo, {}, {}, dy},
                {g.up, {}, g.down, g.bup, {}, g.bdown, mscratch}, scratch};
    }
};

// The encoder: a patch embedding, the positions, one block. Its parameters in one flat vector
struct encoder_params {
    std::span<float> embed, embed_bias;
    block_params<enc> block;
    explicit encoder_params(slab s) : embed(s.take(de * patch)), embed_bias(s.take(de)), block(s) {}
};
constexpr std::size_t encoder_size = de * patch + de + ml::block_parameter_count(enc);

struct predictor_params {
    std::span<float> in, in_bias, mask, out, out_bias;
    block_params<pred.block> block;
    explicit predictor_params(slab s)
        : in(s.take(dp * de)), in_bias(s.take(dp)), mask(s.take(dp)), out(s.take(de * dp)), out_bias(s.take(de)), block(s)
    {
    }
};
constexpr std::size_t predictor_size = dp * de + dp + dp + de * dp + de + ml::block_parameter_count(pred.block);

void initialise(std::vector<float>& p, stats::wyrand& g, const block_params<enc>* eb, const block_params<pred.block>* pb)
{
    for (float& x : p)
        x = float(0.3 * (2 * stats::uniform_real(g) - 1));
    auto ones = [](std::span<float> s) { std::fill(s.begin(), s.end(), 1.0f); };
    auto zeros = [](std::span<float> s) { std::fill(s.begin(), s.end(), 0.0f); };
    auto norms = [&](const auto& b) {
        ones(b.n1w), zeros(b.n1b), ones(b.n2w), zeros(b.n2b), zeros(b.bq), zeros(b.bk), zeros(b.bv), zeros(b.bo), zeros(b.bup), zeros(b.bdown);
    };
    if (eb != nullptr)
        norms(*eb);
    if (pb != nullptr)
        norms(*pb);
}

// A smooth random image: a sum of three plane waves
void image(stats::wyrand& g, std::span<float> out)
{
    std::array<double, 12> c{};
    for (double& x : c)
        x = stats::uniform_real(g);
    for (std::size_t y = 0; y < 8; ++y)
        for (std::size_t x = 0; x < 8; ++x) {
            double v = 0;
            for (std::size_t k = 0; k < 3; ++k)
                v += (0.5 + c[4 * k]) * std::sin(0.8 * c[4 * k + 1] * double(y) + 0.8 * c[4 * k + 2] * double(x) + 6.28 * c[4 * k + 3]);
            out[y * 8 + x] = float(v);
        }
}

// Adds each row's position: rows (s, j) += pos[index(s, j)], through the backend's gather and add
void add_positions(std::span<const float> pos, std::span<const std::uint32_t> index, std::span<float> rows, std::size_t width)
{
    std::vector<float> p(rows.size());
    ml::gather_rows(backend, pos, index, std::span<float>(p), width, 1);
    ml::add_rows(backend, std::span<const float>(p), rows);
}

TEST(MlJepa, AnIjepaRunOnRandomImagesLearnsWithoutCollapse)
{
    stats::wyrand g(20261119);
    std::vector<float> online(encoder_size), target(encoder_size), predictor(predictor_size);
    std::vector<float> d_online(encoder_size), d_predictor(predictor_size), m1(encoder_size), v1(encoder_size), m2(predictor_size), v2(predictor_size);
    const encoder_params E(slab{online}), dE(slab{d_online}), T(slab{target});
    const predictor_params P(slab{predictor}), dP(slab{d_predictor});
    initialise(online, g, &E.block, nullptr);
    initialise(predictor, g, nullptr, &P.block);
    target = online;
    std::vector<float> pos_e(tokens * de), pos_p(tokens * dp);
    ml::sincos_position_2d<float>(de, 4, 4, pos_e);
    ml::sincos_position_2d<float>(dp, 4, 4, pos_p);
    std::vector<std::uint32_t> every(batch * tokens);
    for (std::size_t i = 0; i < every.size(); ++i)
        every[i] = std::uint32_t(i % tokens);

    ml::ijepa_mask_config masks{};
    masks.grid = {1, 4, 4};
    masks.targets = {0.25, 0.3, 0.75, 1.5, 1.0, 1.0, 1};   // 2 x 2 blocks on a 4 x 4 grid
    masks.context = {0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1};
    masks.min_keep = 2;
    const ml::adamw_config<float> adam{.learning_rate = 2e-3f};
    constexpr std::uint64_t steps = 300;
    std::vector<double> losses;
    double first_spread = 0, last_spread = 0, min_spread = 1e9;

    for (std::uint64_t step = 0; step < steps; ++step) {
        // The batch and its masks
        std::vector<float> volumes(batch * grid.values()), patches(volumes.size());
        for (std::size_t s = 0; s < batch; ++s)
            image(g, std::span<float>(volumes).subspan(s * grid.values(), grid.values()));
        ml::patchify_rows(backend, grid, std::span<const float>(volumes), std::span<float>(patches), batch);
        std::vector<std::uint32_t> ctx_lists(batch * tokens), tgt_lists(batch * tokens);
        std::vector<std::size_t> ctx_counts(batch), tgt_counts(batch);
        for (std::size_t s = 0; s < batch; ++s) {
            std::array<std::size_t, 1> tc{};
            const auto n = ml::ijepa_masks(masks, 77, step, s, std::span<std::uint32_t>(ctx_lists).subspan(s * tokens, tokens),
                                           std::span<std::uint32_t>(tgt_lists).subspan(s * tokens, tokens), tc);
            ctx_counts[s] = n.context;
            tgt_counts[s] = tc[0];
        }
        const std::size_t kc = ml::common_count(ctx_counts), kt = ml::common_count(tgt_counts);
        ASSERT_EQ(kt, *std::max_element(tgt_counts.begin(), tgt_counts.end()));   // one block size per step
        std::vector<std::uint32_t> ci(batch * kc), ti(batch * kt);
        ml::collate_indices(ctx_lists, tokens, ctx_counts, kc, ci, {});
        ml::collate_indices(tgt_lists, tokens, tgt_counts, kt, ti, {});

        // The targets: the target encoder over every token, normalised, at the targets; no gradient
        std::vector<float> te(batch * tokens * de);
        ml::linear_sequence(backend, dw{T.embed, de, patch}, std::span<const float>(patches), std::span<float>(te), T.embed_bias);
        add_positions(pos_e, every, te, de);
        block_buffers<enc> tb(batch * tokens);
        ml::block_forward_sequence<enc>(backend, T.block.weights(), std::span<float>(te), ml::no_rope, tb.act(), batch);
        ml::jepa_target_norm_rows(backend, std::span<const float>(te), std::span<float>(te), de);
        std::vector<float> targets(batch * kt * de);
        ml::gather_rows(backend, std::span<const float>(te), ti, std::span<float>(targets), de, batch);

        // The context encoder over the context tokens only
        std::vector<float> cp(batch * kc * patch), ce(batch * kc * de);
        ml::gather_rows(backend, std::span<const float>(patches), ci, std::span<float>(cp), patch, batch);
        ml::linear_sequence(backend, dw{E.embed, de, patch}, std::span<const float>(cp), std::span<float>(ce), E.embed_bias);
        add_positions(pos_e, ci, ce, de);
        block_buffers<enc> cb(batch * kc);
        ml::block_forward_sequence<enc>(backend, E.block.weights(), std::span<float>(ce), ml::no_rope, cb.act(), batch);

        // The predictor: in, context then mask tokens, one block, the targets' rows, out
        const std::size_t kn = kc + kt;
        std::vector<float> pin(batch * kc * dp), prow(batch * kn * dp), pt(batch * kt * dp), out(batch * kt * de);
        ml::linear_sequence(backend, dw{P.in, dp, de}, std::span<const float>(ce), std::span<float>(pin), P.in_bias);
        ml::jepa_predictor_input_rows(backend, std::span<const float>(pin), std::span<const float>(P.mask), std::span<const float>(pos_p), ci, ti,
                                      std::span<float>(prow), dp, batch);
        block_buffers<pred.block> pb(batch * kn);
        ml::block_forward_sequence<pred.block>(backend, P.block.weights(), std::span<float>(prow), ml::no_rope, pb.act(), batch);
        ml::slice_rows(backend, std::span<const float>(prow), kc, kt, std::span<float>(pt), dp, batch);
        ml::linear_sequence(backend, dw{P.out, de, dp}, std::span<const float>(pt), std::span<float>(out), P.out_bias);

        // The loss and the backward pass
        std::fill(d_online.begin(), d_online.end(), 0.0f);
        std::fill(d_predictor.begin(), d_predictor.end(), 0.0f);
        std::vector<float> dout(out.size());
        losses.push_back(ml::jepa_loss_rows(backend, ml::jepa_loss::l2, std::span<const float>(out), std::span<const float>(targets), de,
                                            std::span<float>(dout)));
        std::vector<float> dpt(pt.size()), dprow(prow.size()), dpin(pin.size()), dce(ce.size());
        ml::linear_sequence_backward(backend, dw{P.out, de, dp}, std::span<const float>(pt), std::span<const float>(dout), std::span<float>(dpt),
                                     dP.out, dP.out_bias);
        ml::slice_rows_backward(backend, std::span<const float>(dpt), kc, kt, std::span<float>(dprow), dp, batch);
        ml::block_backward<pred.block>(backend, P.block.weights(), ml::no_rope, pb.act(), std::span<float>(dprow), pb.grads(dP.block), batch);
        ml::jepa_predictor_input_rows_backward(backend, std::span<const float>(dprow), kc, kt, std::span<float>(dpin), dP.mask, dp, batch);
        ml::linear_sequence_backward(backend, dw{P.in, dp, de}, std::span<const float>(ce), std::span<const float>(dpin), std::span<float>(dce),
                                     dP.in, dP.in_bias);
        ml::block_backward<enc>(backend, E.block.weights(), ml::no_rope, cb.act(), std::span<float>(dce), cb.grads(dE.block), batch);
        ml::linear_sequence_backward(backend, dw{E.embed, de, patch}, std::span<const float>(cp), std::span<const float>(dce), std::span<float>{},
                                     dE.embed, dE.embed_bias);

        // The step, then the target encoder follows
        ml::adamw_rows(backend, std::span<float>(online), std::span<const float>(d_online), std::span<float>(m1), std::span<float>(v1), step + 1, adam);
        ml::adamw_rows(backend, std::span<float>(predictor), std::span<const float>(d_predictor), std::span<float>(m2), std::span<float>(v2), step + 1,
                       adam);
        const std::array<ml::ema_segment<float>, 1> segs{{{std::span<float>(target), std::span<const float>(online)}}};
        ml::ema_segments(backend, std::span<const ml::ema_segment<float>>(segs), ml::ema_momentum(step, steps));

        // The spread of the normalised targets across the batch's tokens: collapse would take it to 0
        std::vector<float> mean(de), m(de), sd(de);
        ml::embedding_moments<float> acc{mean, m};
        acc.add_rows(te);
        acc.standard_deviation(sd);
        const double spread = *std::min_element(sd.begin(), sd.end());
        if (step == 0)
            first_spread = spread;
        last_spread = spread;
        min_spread = std::min(min_spread, spread);
    }
    auto mean_of = [&](std::size_t from, std::size_t to) {
        double s = 0;
        for (std::size_t i = from; i < to; ++i)
            s += losses[i];
        return s / double(to - from);
    };
    const double early = mean_of(0, 20), late = mean_of(steps - 20, steps);
    EXPECT_LT(late, 0.5 * early) << early << " -> " << late;   // 1.0 to 0.15 on the reference backend
    EXPECT_GT(min_spread, 0.1) << first_spread << " -> " << last_spread;   // 0.24 to 0.23, at least 0.17, on the reference backend
}

} // namespace
