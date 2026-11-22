//
//  transformer_block_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/transformer_block.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


// A device library's backend, in its own namespace: it overloads two of the batched passes' operations (found by
// argument-dependent lookup, more specialised than core's), here counting calls and doing the same work
namespace device_like {
struct backend : snicholls::linalg::reference_backend {
    inline static int fills = 0, norms = 0;
};

template <typename T>
constexpr void fill_rows(const backend&, std::span<T> x, std::type_identity_t<T> value)
{
    ++backend::fills;
    for (T& e : x)
        e = value;
}

template <snicholls::ml::transformer_config C, typename T>
constexpr void block_norm_rows(const backend& b, std::type_identity_t<std::span<const T>> x, std::type_identity_t<std::span<const T>> weight,
                               std::type_identity_t<std::span<const T>> bias, std::span<T> out)
{
    ++backend::norms;
    snicholls::ml::block_norm_rows<C>(static_cast<const snicholls::linalg::reference_backend&>(b), x, weight, bias, out);
}
} // namespace device_like

namespace {

namespace ml = snicholls::ml;
using W = ml::dense_weights<float>;

// The fixture: per section, the arrays the generator filled (regenerated here from the same splitmix64 stream, in the
// file's order) and the arrays it computed (float bits, or doubles in hex)
struct fixture {
    std::map<std::string, std::map<std::string, std::vector<float>>> filled;
    std::map<std::string, std::map<std::string, std::vector<float>>> floats;
    std::map<std::string, std::map<std::string, std::vector<double>>> doubles;
};

const fixture& data()
{
    static const fixture f = [] {
        fixture out;
        std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/transformer_fixture.txt");
        EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/transformer_fixture.txt";
        std::uint64_t state = 0x5eed2026;
        std::string line, section;
        while (std::getline(in, line)) {
            std::istringstream s(line);
            std::string word, name;
            s >> word;
            if (word.empty() || word[0] == '#')
                continue;
            if (word == "section") {
                s >> section;
                continue;
            }
            std::size_t n = 0;
            if (word == "fill") {
                std::string scale_text;
                s >> name >> n >> scale_text;
                const double scale = std::strtod(scale_text.c_str(), nullptr);
                std::vector<float> v(n);
                for (float& x : v) {
                    std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
                    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
                    z ^= z >> 31;
                    x = float((2.0 * (double(z >> 40) * 0x1p-24) - 1.0) * scale);
                }
                out.filled[section][name] = std::move(v);
                continue;
            }
            s >> n;
            std::vector<std::string> items(n);
            for (auto& item : items)
                s >> item;
            if (section == "gpt2") {
                std::vector<float> v(n);
                for (std::size_t i = 0; i < n; ++i)
                    v[i] = std::bit_cast<float>(static_cast<std::uint32_t>(std::stoul(items[i], nullptr, 16)));
                out.floats[section][word] = std::move(v);
            } else {
                std::vector<double> v(n);
                for (std::size_t i = 0; i < n; ++i)
                    v[i] = std::strtod(items[i].c_str(), nullptr);
                out.doubles[section][word] = std::move(v);
            }
        }
        return out;
    }();
    return f;
}

const std::vector<float>& filled(const std::string& section, const std::string& name)
{
    return data().filled.at(section).at(name);
}

std::span<const float> maybe(const std::string& section, const std::string& name)
{
    const auto& s = data().filled.at(section);
    const auto it = s.find(name);
    return it == s.end() ? std::span<const float>{} : std::span<const float>(it->second);
}

// The largest error relative to max(1, |reference|)
template <typename R>
double worst(const std::vector<float>& got, const std::vector<R>& want)
{
    EXPECT_EQ(got.size(), want.size());
    double e = 0;
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i)
        e = std::max(e, std::abs(double(got[i]) - double(want[i])) / std::max(1.0, std::abs(double(want[i]))));
    return e;
}

// GPT-2's block at llm.c's test size: C = 32, two heads of 16, learned positions outside the block
constexpr ml::transformer_config gpt2_tiny{.d_model = 32, .n_heads = 2, .n_kv_heads = 2, .head_dim = 16, .d_ff = 128,
                                           .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .mlp = ml::mlp_kind::gelu_tanh, .bias = true};

// The textbook blocks of the generator, field for field
constexpr ml::transformer_config llama_tiny{.d_model = 32, .n_heads = 4, .n_kv_heads = 2, .head_dim = 8, .d_ff = 64,
                                            .norm = ml::norm_kind::rms, .norm_eps = 1e-5, .mlp = ml::mlp_kind::swiglu,
                                            .rope = ml::rope_kind::half, .rope_base = 10000};
constexpr ml::transformer_config nanochat_tiny{.d_model = 32, .n_heads = 4, .n_kv_heads = 4, .head_dim = 8, .d_ff = 64,
                                               .norm = ml::norm_kind::rms, .norm_eps = 1.1920928955078125e-07,
                                               .qk_norm = ml::qk_norm_kind::after_rope, .mlp = ml::mlp_kind::relu_squared,
                                               .rope = ml::rope_kind::half, .rope_base = 10000, .window = 3, .attention_softcap = 5};
constexpr ml::transformer_config mqa_tiny{.d_model = 32, .n_heads = 4, .n_kv_heads = 1, .head_dim = 8, .d_ff = 64,
                                          .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .qk_norm = ml::qk_norm_kind::before_rope,
                                          .mlp = ml::mlp_kind::gelu, .rope = ml::rope_kind::interleaved, .rope_base = 500, .bias = true};

ml::block_weights<W> gpt2_weights()
{
    constexpr std::size_t c = 32, f = 128;
    const std::string g = "gpt2";
    const std::span<const float> qkvw(filled(g, "qkvw")), qkvb(filled(g, "qkvb"));
    ml::block_weights<W> w{};
    w.norm1_weight = filled(g, "ln1w");
    w.norm1_bias = filled(g, "ln1b");
    w.attention = {W{qkvw.subspan(0, c * c), c, c}, W{qkvw.subspan(c * c, c * c), c, c}, W{qkvw.subspan(2 * c * c, c * c), c, c},
                   W{filled(g, "attprojw"), c, c}, qkvb.subspan(0, c), qkvb.subspan(c, c), qkvb.subspan(2 * c, c), filled(g, "attprojb")};
    w.norm2_weight = filled(g, "ln2w");
    w.norm2_bias = filled(g, "ln2b");
    w.mlp = {W{filled(g, "fcw"), f, c}, W{}, W{filled(g, "fcprojw"), c, f}, filled(g, "fcb"), {}, filled(g, "fcprojb")};
    return w;
}

template <ml::transformer_config C>
ml::block_weights<W> textbook_weights(const std::string& s)
{
    ml::block_weights<W> w{};
    w.norm1_weight = filled(s, "norm1_weight");
    w.norm1_bias = maybe(s, "norm1_bias");
    w.attention = {W{filled(s, "wq"), C.q_size(), C.d_model}, W{filled(s, "wk"), C.kv_size(), C.d_model},
                   W{filled(s, "wv"), C.kv_size(), C.d_model}, W{filled(s, "wo"), C.d_model, C.q_size()},
                   maybe(s, "bq"), maybe(s, "bk"), maybe(s, "bv"), maybe(s, "bo"), maybe(s, "q_norm"), maybe(s, "k_norm")};
    w.norm2_weight = filled(s, "norm2_weight");
    w.norm2_bias = maybe(s, "norm2_bias");
    const auto gate = maybe(s, "gate");
    w.mlp = {W{filled(s, "up"), C.d_ff, C.d_model}, W{gate, gate.empty() ? 0 : C.d_ff, gate.empty() ? 0 : C.d_model},
             W{filled(s, "down"), C.d_model, C.d_ff}, maybe(s, "up_bias"), {}, maybe(s, "down_bias")};
    return w;
}

// A run of tokens through one block from position 0, with a cache of `capacity` slots
template <ml::transformer_config C, typename R>
std::vector<float> run(const ml::block_weights<W>& w, std::vector<float> x, std::size_t capacity, const R& rope_src)
{
    std::vector<float> k(ml::kv_cache_view<C>::required_size(1, capacity)), v(k.size());
    const ml::kv_cache_view<C> cache(k, v, 1, capacity);
    ml::block_prefill<C>(w, x, 0, cache, 0, rope_src);
    return x;
}

template <ml::transformer_config C>
std::vector<float> textbook(const std::string& s, std::size_t capacity)
{
    static const ml::rope_table<float, 8, C.head_dim> table(ml::rope_base_v<C, float>);
    return run<C>(textbook_weights<C>(s), filled(s, "x"), capacity, table);
}

TEST(MlTransformerBlock, Gpt2BlockIsLlmcsWithinTolerance)
{
    const auto out = run<gpt2_tiny>(gpt2_weights(), filled("gpt2", "x"), 4, ml::no_rope);
    // llm.c computes attention in three passes with expf and core in one with its own exp, so the bits may differ;
    // measured, they agree within one float epsilon (1.2e-7). The bound is eight.
    EXPECT_LT(worst(out, data().floats.at("gpt2").at("out")), 1e-6);
}

TEST(MlTransformerBlock, TextbookBlocksMatchTheDoublePrecisionReference)
{
    const double llama = worst(textbook<llama_tiny>("llama", 6), data().doubles.at("llama").at("out"));
    const double nanochat = worst(textbook<nanochat_tiny>("nanochat", 3), data().doubles.at("nanochat").at("out"));
    const double mqa = worst(textbook<mqa_tiny>("mqa", 6), data().doubles.at("mqa").at("out"));
    // float against double: measured 1.3e-7 to 1.8e-7, a float epsilon or so; the bound is eight
    EXPECT_LT(llama, 1e-6);
    EXPECT_LT(nanochat, 1e-6);
    EXPECT_LT(mqa, 1e-6);
}

// Prefill is block_forward token by token, so a prompt prefilled and the same prompt decoded step by step agree
TEST(MlTransformerBlock, PrefillEqualsDecodingStepByStep)
{
    const auto w = textbook_weights<llama_tiny>("llama");
    static const ml::rope_table<float, 8, 8> table(ml::rope_base_v<llama_tiny, float>);
    const auto prefilled = run<llama_tiny>(w, filled("llama", "x"), 6, table);
    std::vector<float> stepped = filled("llama", "x");
    std::vector<float> k(ml::kv_cache_view<llama_tiny>::required_size(1, 6)), v(k.size());
    const ml::kv_cache_view<llama_tiny> cache(k, v, 1, 6);
    for (std::size_t t = 0; t < 6; ++t)
        ml::block_forward<llama_tiny>(w, std::span<float>(stepped).subspan(t * 32, 32), t, cache, 0, table);
    EXPECT_EQ(prefilled, stepped);
}

// The llama block with every matrix in Q8_0: close to the float result, within the quantisation error
TEST(MlTransformerBlock, Q8WeightsStayNearTheFloatResult)
{
    using Q = ml::quantized_weights<ml::block_q8_0>;
    const std::string s = "llama";
    std::map<std::string, std::vector<ml::block_q8_0>> blocks;
    auto q8 = [&](const std::string& name, std::size_t rows, std::size_t cols) {
        auto& b = blocks[name];
        b.resize(rows * cols / 32);
        ml::quantize(filled(s, name), std::span<ml::block_q8_0>(b));
        return Q{b, rows, cols};
    };
    constexpr auto c = llama_tiny;
    ml::block_weights<Q> w{};
    w.norm1_weight = filled(s, "norm1_weight");
    w.norm2_weight = filled(s, "norm2_weight");
    w.attention = {q8("wq", c.q_size(), c.d_model), q8("wk", c.kv_size(), c.d_model), q8("wv", c.kv_size(), c.d_model), q8("wo", c.d_model, c.q_size())};
    w.mlp = {q8("up", c.d_ff, c.d_model), q8("gate", c.d_ff, c.d_model), q8("down", c.d_model, c.d_ff)};
    static const ml::rope_table<float, 8, 8> table(ml::rope_base_v<llama_tiny, float>);
    std::vector<float> x = filled(s, "x");
    std::vector<float> k(ml::kv_cache_view<c>::required_size(1, 6)), v(k.size());
    ml::block_prefill<c>(w, x, 0, ml::kv_cache_view<c>(k, v, 1, 6), 0, table);
    EXPECT_LT(worst(x, data().doubles.at("llama").at("out")), 0.02);
}

constexpr bool compile_time()
{
    constexpr ml::transformer_config c{.d_model = 2, .n_heads = 1, .n_kv_heads = 1, .head_dim = 2, .d_ff = 2, .norm = ml::norm_kind::rms,
                                       .mlp = ml::mlp_kind::relu_squared};
    std::array<double, 4> eye{1, 0, 0, 1}, zero{};
    using D = ml::dense_weights<double>;
    ml::block_weights<D, double> w{};
    w.attention = {D{eye, 2, 2}, D{eye, 2, 2}, D{eye, 2, 2}, D{zero, 2, 2}};   // the attention adds nothing
    w.mlp = {D{zero, 2, 2}, {}, D{zero, 2, 2}};                               // nor does the MLP
    std::array<double, 4> k{}, v{};
    std::array<double, 2> x{3, 4};
    ml::block_forward<c>(w, x, 0, ml::kv_cache_view<c, double>(k, v, 1, 2), 0, ml::no_rope);
    return x[0] == 3 && x[1] == 4 && k[0] != 0;   // the residual stream is unchanged, the cache is written
}
static_assert(compile_time());

// Training: the block over a sequence and its backward pass, in double
using WD = ml::dense_weights<double>;

constexpr ml::transformer_config b_gpt2{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 12,
                                        .norm = ml::norm_kind::layer, .mlp = ml::mlp_kind::gelu_tanh};
constexpr ml::transformer_config b_llama{.d_model = 8, .n_heads = 2, .n_kv_heads = 1, .head_dim = 4, .d_ff = 12,
                                         .mlp = ml::mlp_kind::swiglu, .rope = ml::rope_kind::half};
constexpr ml::transformer_config b_nanochat{.d_model = 8, .n_heads = 2, .n_kv_heads = 2, .head_dim = 4, .d_ff = 12,
                                            .qk_norm = ml::qk_norm_kind::after_rope, .mlp = ml::mlp_kind::relu_squared,
                                            .rope = ml::rope_kind::half, .window = 3};

std::vector<double> bwave(std::size_t n, double a, double b)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = a * std::sin(0.47 * double(i) + b);
    return v;
}

static const ml::rope_table<double, 16, 4> brope{};

template <ml::transformer_config C>
const auto& brope_for()
{
    if constexpr (C.rope == ml::rope_kind::none)
        return ml::no_rope;
    else
        return brope;
}

// Every parameter of a block, owned; each vector is one tensor
template <ml::transformer_config C>
struct train_block {
    static constexpr std::size_t n = 5, d = C.d_model, f = C.d_ff;
    static constexpr bool layer = C.norm == ml::norm_kind::layer, swiglu = C.mlp == ml::mlp_kind::swiglu,
                          qk = C.qk_norm != ml::qk_norm_kind::none;
    std::vector<std::vector<double>> p;
    enum { n1w, n1b, wq, wk, wv, wo, bq, bk, bv, bo, nq, nk, n2w, n2b, up, gate, down, bup, bgate, bdown, count };
    train_block() : p(count)
    {
        auto set = [&](int i, std::size_t size, double a) { p[i] = bwave(size, a, 0.37 * i); };
        set(n1w, d, 0.2); set(n2w, d, 0.2);
        for (double& e : p[n1w]) e += 1;
        for (double& e : p[n2w]) e += 1;
        if (layer) { set(n1b, d, 0.1); set(n2b, d, 0.1); }
        set(wq, C.q_size() * d, 0.5); set(wk, C.kv_size() * d, 0.5); set(wv, C.kv_size() * d, 0.6); set(wo, d * C.q_size(), 0.5);
        set(bq, C.q_size(), 0.1); set(bk, C.kv_size(), 0.1); set(bv, C.kv_size(), 0.1); set(bo, d, 0.1);
        if (qk) { set(nq, C.head_dim, 0.1); set(nk, C.head_dim, 0.1); for (double& e : p[nq]) e += 1; for (double& e : p[nk]) e += 1; }
        set(up, f * d, 0.5); set(down, d * f, 0.5); set(bup, f, 0.1); set(bdown, d, 0.1);
        if (swiglu) { set(gate, f * d, 0.5); set(bgate, f, 0.1); }
    }
    ml::block_weights<WD, double> weights() const
    {
        return {p[n1w], p[n1b],
                {WD{p[wq], C.q_size(), d}, WD{p[wk], C.kv_size(), d}, WD{p[wv], C.kv_size(), d}, WD{p[wo], d, C.q_size()},
                 p[bq], p[bk], p[bv], p[bo], p[nq], p[nk]},
                p[n2w], p[n2b],
                {WD{p[up], f, d}, swiglu ? WD{p[gate], f, d} : WD{}, WD{p[down], d, f}, p[bup], p[bgate], p[bdown]}};
    }
};

template <ml::transformer_config C>
struct train_block_buffers {
    static constexpr std::size_t n = train_block<C>::n, d = C.d_model;
    static constexpr bool qk = C.qk_norm != ml::qk_norm_kind::none, swiglu = C.mlp == ml::mlp_kind::swiglu;
    std::vector<double> x0 = std::vector<double>(n * d), h1 = x0, x1 = x0, h2 = x0, q = std::vector<double>(n * C.q_size()),
                        k = std::vector<double>(n * C.kv_size()), v = k, qr = std::vector<double>(qk ? q.size() : 0),
                        kr = std::vector<double>(qk ? k.size() : 0), y = q, lse = std::vector<double>(n * C.n_heads),
                        up = std::vector<double>(n * C.d_ff), gate = std::vector<double>(swiglu ? up.size() : 0);
    ml::block_activations<double> act() { return {x0, h1, x1, h2, {q, k, v, qr, kr, y, lse}, up, gate}; }
};

template <ml::transformer_config C>
void block_sequence_is_prefill()
{
    const train_block<C> b;
    constexpr std::size_t n = train_block<C>::n;
    auto xs = bwave(n * C.d_model, 1.1, 0.2), xp = xs;
    train_block_buffers<C> buf;
    ml::block_forward_sequence<C>(b.weights(), xs, brope_for<C>(), buf.act());
    std::vector<double> keys(ml::kv_cache_view<C, double>::required_size(1, n)), values(keys.size());
    ml::block_prefill<C>(b.weights(), xp, 0, ml::kv_cache_view<C, double>(keys, values, 1, n), 0, brope_for<C>());
    EXPECT_EQ(xs, xp);
}

TEST(MlTransformerBlockTraining, TheSequenceForwardIsPrefillToTheBit)
{
    block_sequence_is_prefill<b_gpt2>();
    block_sequence_is_prefill<b_llama>();
    block_sequence_is_prefill<b_nanochat>();
}

template <ml::transformer_config C>
void block_backward_is_central_differences()
{
    train_block<C> b;
    constexpr std::size_t n = train_block<C>::n;
    auto x = bwave(n * C.d_model, 1.1, 0.2);
    const auto gout = bwave(n * C.d_model, 0.9, 1.4);
    auto loss = [&] {
        auto xs = x;
        train_block_buffers<C> buf;
        ml::block_forward_sequence<C>(b.weights(), xs, brope_for<C>(), buf.act());
        double l = 0;
        for (std::size_t i = 0; i < xs.size(); ++i)
            l += gout[i] * xs[i];
        return l;
    };
    auto xs = x;
    train_block_buffers<C> buf;
    ml::block_forward_sequence<C>(b.weights(), xs, brope_for<C>(), buf.act());
    std::vector<std::vector<double>> gp(b.p.size());
    for (std::size_t i = 0; i < b.p.size(); ++i)
        gp[i].assign(b.p[i].size(), 0.0);
    using B = train_block<C>;
    std::vector<double> dq(buf.q.size()), dk(buf.k.size()), dv(buf.v.size()), scratch(x.size());
    const ml::block_gradients<double> grads{
        gp[B::n1w], gp[B::n1b], gp[B::n2w], gp[B::n2b],
        {dq, dk, dv, gp[B::wq], gp[B::wk], gp[B::wv], gp[B::wo], gp[B::bq], gp[B::bk], gp[B::bv], gp[B::bo], gp[B::nq], gp[B::nk]},
        {gp[B::up], gp[B::gate], gp[B::down], gp[B::bup], gp[B::bgate], gp[B::bdown]},
        scratch};
    auto dx = gout;
    ml::block_backward<C>(b.weights(), brope_for<C>(), buf.act(), dx, grads);
    auto check = [&](std::vector<double>& p, const std::vector<double>& grad, std::size_t which) {
        for (std::size_t i = 0; i < p.size(); ++i) {
            const double keep = p[i], e = 1e-6;
            p[i] = keep + e;
            const double hi = loss();
            p[i] = keep - e;
            const double lo = loss();
            p[i] = keep;
            EXPECT_NEAR(grad[i], (hi - lo) / (2 * e), 3e-7) << "tensor " << which << " at " << i;
        }
    };
    check(x, dx, 999);
    for (std::size_t i = 0; i < b.p.size(); ++i)
        check(b.p[i], gp[i], i);
}

TEST(MlTransformerBlockTraining, BackwardMatchesCentralDifferencesForAGpt2Block)
{
    block_backward_is_central_differences<b_gpt2>();
}

// llm.c's gpt2_backward for one layer, from the fixture: the same block, the same output gradient, every gradient
TEST(MlTransformerBlockTraining, Gpt2BackwardIsLlmcsWithinTolerance)
{
    constexpr std::size_t n = 4, c = 32, f = 128;
    using v = std::vector<float>;
    const auto w = gpt2_weights();
    v xs = filled("gpt2", "x"), x0(n * c), h1(n * c), x1(n * c), h2(n * c), q(n * c), k(n * c), val(n * c), y(n * c), lse(n * 2),
      up(n * f);
    const ml::block_activations<float> act{x0, h1, x1, h2, {q, k, val, {}, {}, y, lse}, up, {}};
    ml::block_forward_sequence<gpt2_tiny>(w, xs, ml::no_rope, act);
    v dln1w(c), dln1b(c), dln2w(c), dln2b(c), dq(n * c), dk(n * c), dv(n * c), dqkvw(3 * c * c), dqkvb(3 * c), dattprojw(c * c),
      dattprojb(c), dfcw(f * c), dfcb(f), dfcprojw(c * f), dfcprojb(c), scratch(n * c);
    const std::span<float> gw(dqkvw), gb(dqkvb);
    const ml::block_gradients<float> grads{
        dln1w, dln1b, dln2w, dln2b,
        {dq, dk, dv, gw.subspan(0, c * c), gw.subspan(c * c, c * c), gw.subspan(2 * c * c, c * c), dattprojw, gb.subspan(0, c),
         gb.subspan(c, c), gb.subspan(2 * c, c), dattprojb, {}, {}},
        {dfcw, {}, dfcprojw, dfcb, {}, dfcprojb},
        scratch};
    v dx = filled("gpt2", "dout");
    ml::block_backward<gpt2_tiny>(w, ml::no_rope, act, dx, grads);
    const auto& want = data().floats.at("gpt2");
    double e = 0;
    for (const auto& [name, got] : std::initializer_list<std::pair<const char*, const v*>>{
             {"dx", &dx}, {"dln1w", &dln1w}, {"dln1b", &dln1b}, {"dqkvw", &dqkvw}, {"dqkvb", &dqkvb}, {"dattprojw", &dattprojw},
             {"dattprojb", &dattprojb}, {"dln2w", &dln2w}, {"dln2b", &dln2b}, {"dfcw", &dfcw}, {"dfcb", &dfcb},
             {"dfcprojw", &dfcprojw}, {"dfcprojb", &dfcprojb}}) {
        const double here = worst(*got, want.at(name));
        EXPECT_LT(here, 1e-6) << name;
        e = std::max(e, here);
    }
    // As the forward: llm.c's attention is three passes with expf, core's one with its own exp. Measured, the worst
    // gradient (dqkvw, dqkvb) is within 2.4e-7, two float epsilons; the bound is eight
    EXPECT_LT(e, 1e-6);
}

// The batched path (every projection a gemm through the reference backend) against the per-token path, which the
// tests above check against central differences: the same values within rounding, forward and every gradient
template <ml::transformer_config C>
void batched_is_per_token()
{
    using B = train_block<C>;
    const B b;
    constexpr std::size_t n = B::n, d = C.d_model;
    struct run_t {
        train_block_buffers<C> buf;
        std::vector<double> hidden = std::vector<double>(n * C.d_ff), xs, dx;
        std::vector<std::vector<double>> gp;
        std::vector<double> dq = std::vector<double>(n * C.q_size()), dk = std::vector<double>(n * C.kv_size()), dv = dk,
                            dy = dq, scratch = std::vector<double>(n * d), mlp_scratch = std::vector<double>(2 * n * C.d_ff);
        ml::block_gradients<double> grads()
        {
            return {gp[B::n1w], gp[B::n1b], gp[B::n2w], gp[B::n2b],
                    {dq, dk, dv, gp[B::wq], gp[B::wk], gp[B::wv], gp[B::wo], gp[B::bq], gp[B::bk], gp[B::bv], gp[B::bo], gp[B::nq], gp[B::nk], dy},
                    {gp[B::up], gp[B::gate], gp[B::down], gp[B::bup], gp[B::bgate], gp[B::bdown], mlp_scratch},
                    scratch};
        }
        ml::block_activations<double> act()
        {
            auto a = buf.act();
            a.mlp_hidden = hidden;
            return a;
        }
    };
    run_t one, many;
    for (run_t* r : {&one, &many}) {
        r->xs = bwave(n * d, 1.1, 0.2);
        r->dx = bwave(n * d, 0.9, 1.4);
        r->gp.resize(b.p.size());
        for (std::size_t i = 0; i < b.p.size(); ++i)
            r->gp[i].assign(b.p[i].size(), 0.0);
    }
    ml::block_forward_sequence<C>(b.weights(), one.xs, brope_for<C>(), one.buf.act());
    ml::block_backward<C>(b.weights(), brope_for<C>(), one.buf.act(), one.dx, one.grads());
    ml::block_forward_sequence<C>(snicholls::linalg::reference, b.weights(), many.xs, brope_for<C>(), many.act());
    ml::block_backward<C>(snicholls::linalg::reference, b.weights(), brope_for<C>(), many.act(), many.dx, many.grads());
    auto near = [](const std::vector<double>& x, const std::vector<double>& y, const char* what) {
        ASSERT_EQ(x.size(), y.size()) << what;
        for (std::size_t i = 0; i < x.size(); ++i)
            EXPECT_NEAR(x[i], y[i], 1e-12 * std::max(1.0, std::abs(x[i]))) << what << " at " << i;
    };
    near(one.xs, many.xs, "output");
    near(one.dx, many.dx, "input gradient");
    near(one.buf.lse, many.buf.lse, "log-sum-exps");
    for (std::size_t i = 0; i < b.p.size(); ++i)
        near(one.gp[i], many.gp[i], "a weight gradient");
}

// The seam of TRAINING.md T9: a backend's own overloads of the operations are the ones the batched passes call
TEST(MlTransformerBlockTraining, ABackendsOwnOperationsAreCalledAndGiveTheSameBits)
{
    const train_block<b_gpt2> b;
    constexpr std::size_t n = train_block<b_gpt2>::n, d = b_gpt2.d_model;
    auto run = [&](const auto& backend, std::vector<double>& xs, std::vector<double>& dx) {
        train_block_buffers<b_gpt2> buf;
        std::vector<double> hidden(n * b_gpt2.d_ff), scratch(n * d), mlp_scratch(2 * n * b_gpt2.d_ff), dq(n * d), dk(n * d), dv(n * d), dy(n * d);
        std::vector<std::vector<double>> gp(b.p.size());
        for (std::size_t i = 0; i < b.p.size(); ++i)
            gp[i].assign(b.p[i].size(), 0.0);
        using B = train_block<b_gpt2>;
        auto act = buf.act();
        act.mlp_hidden = hidden;
        const ml::block_gradients<double> grads{
            gp[B::n1w], gp[B::n1b], gp[B::n2w], gp[B::n2b],
            {dq, dk, dv, gp[B::wq], gp[B::wk], gp[B::wv], gp[B::wo], gp[B::bq], gp[B::bk], gp[B::bv], gp[B::bo], gp[B::nq], gp[B::nk], dy},
            {gp[B::up], gp[B::gate], gp[B::down], gp[B::bup], gp[B::bgate], gp[B::bdown], mlp_scratch},
            scratch};
        ml::block_forward_sequence<b_gpt2>(backend, b.weights(), xs, brope_for<b_gpt2>(), act);
        ml::block_backward<b_gpt2>(backend, b.weights(), brope_for<b_gpt2>(), act, dx, grads);
    };
    auto xs1 = bwave(n * d, 1.1, 0.2), dx1 = bwave(n * d, 0.9, 1.4), xs2 = xs1, dx2 = dx1;
    run(snicholls::linalg::reference, xs1, dx1);
    device_like::backend::fills = device_like::backend::norms = 0;
    run(device_like::backend{}, xs2, dx2);
    EXPECT_EQ(device_like::backend::norms, 2);    // the block's two norms
    EXPECT_GE(device_like::backend::fills, 5);    // the block's, the MLP's and the attention's fills
    EXPECT_EQ(xs1, xs2);                          // the same work: the same bits
    EXPECT_EQ(dx1, dx2);
}

TEST(MlTransformerBlockTraining, BatchedThroughABackendIsThePerTokenPathWithinRounding)
{
    batched_is_per_token<b_gpt2>();
    batched_is_per_token<b_llama>();
    batched_is_per_token<b_nanochat>();
}

// A batch of sequences in one call against one call per sequence through the same backend, the gradients added into
// the same tensors sequence by sequence: the outputs, every activation, the input's gradient and every gradient to the
// bit, except the weight gradients of the products made over all the rows at once (Wq, Wo, the MLP's), within rounding
// The sizes of batch_is_each_sequence and (below) its buffers, classes of their own: MSVC miscompiled them as a local
// class (C1001, or heap corruption at run time)
template <ml::transformer_config C>
struct batch_sizes {
    static constexpr std::size_t n = train_block<C>::n, d = C.d_model, seqs = 3, rows = seqs * n, f = C.d_ff, qn = C.q_size(),
                                 kn = C.kv_size();
    static constexpr bool qk = C.qk_norm != ml::qk_norm_kind::none, swiglu = C.mlp == ml::mlp_kind::swiglu;
};

template <ml::transformer_config C>
struct batch_run {
    using S = batch_sizes<C>;
    using B = train_block<C>;
    std::vector<double> x0 = std::vector<double>(S::rows * S::d), h1 = x0, x1 = x0, h2 = x0, q = std::vector<double>(S::rows * S::qn),
                        k = std::vector<double>(S::rows * S::kn), v = k, qr = std::vector<double>(S::qk ? q.size() : 0),
                        kr = std::vector<double>(S::qk ? k.size() : 0), y = q, lse = std::vector<double>(S::rows * C.n_heads),
                        up = std::vector<double>(S::rows * S::f), gate = std::vector<double>(S::swiglu ? up.size() : 0), hidden = up,
                        dq = q, dk = k, dv = k, dy = q, scratch = x0, mlp_scratch = std::vector<double>(2 * S::rows * S::f),
                        xs = bwave(S::rows * S::d, 1.1, 0.2), dx = bwave(S::rows * S::d, 0.9, 1.4);
    std::vector<std::vector<double>> gp;
    // Sequence s of `count` (all of them for s = 0, count = seqs): every view a slice
    ml::block_activations<double> act(std::size_t s, std::size_t count)
    {
        auto cut = [&](std::vector<double>& v) { return v.empty() ? std::span<double>{} : std::span<double>(v).subspan(s * v.size() / S::seqs, count * v.size() / S::seqs); };
        return {cut(x0), cut(h1), cut(x1), cut(h2), {cut(q), cut(k), cut(v), cut(qr), cut(kr), cut(y), cut(lse)}, cut(up), cut(gate), cut(hidden)};
    }
    ml::block_gradients<double> grads(std::size_t s, std::size_t count)
    {
        auto cut = [&](std::vector<double>& v) { return std::span<double>(v).subspan(s * v.size() / S::seqs, count * v.size() / S::seqs); };
        return {gp[B::n1w], gp[B::n1b], gp[B::n2w], gp[B::n2b],
                {cut(dq), cut(dk), cut(dv), gp[B::wq], gp[B::wk], gp[B::wv], gp[B::wo], gp[B::bq], gp[B::bk], gp[B::bv], gp[B::bo], gp[B::nq],
                 gp[B::nk], cut(dy)},
                {gp[B::up], gp[B::gate], gp[B::down], gp[B::bup], gp[B::bgate], gp[B::bdown], cut(mlp_scratch)},
                cut(scratch)};
    }
};

template <ml::transformer_config C>
void batch_is_each_sequence()
{
    using B = train_block<C>;
    using S = batch_sizes<C>;
    const B b;
    constexpr std::size_t n = S::n, d = S::d, seqs = S::seqs;
    using run_t = batch_run<C>;
    run_t each, all;
    for (run_t* r : {&each, &all}) {
        r->gp.resize(b.p.size());
        for (std::size_t i = 0; i < b.p.size(); ++i)
            r->gp[i].assign(b.p[i].size(), 0.0);
    }
    const auto& backend = snicholls::linalg::reference;
    for (std::size_t s = 0; s < seqs; ++s)
        ml::block_forward_sequence<C>(backend, b.weights(), std::span<double>(each.xs).subspan(s * n * d, n * d), brope_for<C>(), each.act(s, 1));
    for (std::size_t s = 0; s < seqs; ++s)
        ml::block_backward<C>(backend, b.weights(), brope_for<C>(), each.act(s, 1), std::span<double>(each.dx).subspan(s * n * d, n * d),
                              each.grads(s, 1));
    ml::block_forward_sequence<C>(backend, b.weights(), all.xs, brope_for<C>(), all.act(0, seqs), seqs);
    ml::block_backward<C>(backend, b.weights(), brope_for<C>(), all.act(0, seqs), all.dx, all.grads(0, seqs), seqs);
    EXPECT_EQ(each.xs, all.xs);
    EXPECT_EQ(each.dx, all.dx);
    for (const auto m : {&run_t::x0, &run_t::h1, &run_t::x1, &run_t::h2, &run_t::q, &run_t::k, &run_t::v, &run_t::qr, &run_t::kr, &run_t::y,
                         &run_t::lse, &run_t::up, &run_t::gate, &run_t::hidden, &run_t::dq, &run_t::dk, &run_t::dv})
        EXPECT_EQ(each.*m, all.*m);
    for (std::size_t i = 0; i < b.p.size(); ++i) {
        const bool one_product = i == B::wq || i == B::wo || i == B::up || i == B::gate || i == B::down;
        if (!one_product) {
            EXPECT_EQ(each.gp[i], all.gp[i]) << "tensor " << i;
            continue;
        }
        for (std::size_t j = 0; j < each.gp[i].size(); ++j)
            EXPECT_NEAR(each.gp[i][j], all.gp[i][j], 1e-13 * std::max(1.0, std::abs(each.gp[i][j]))) << "tensor " << i << " at " << j;
    }
    // One sequence is the default
    run_t one;
    one.gp = all.gp;
    auto xs1 = std::vector<double>(all.xs.begin(), all.xs.begin() + std::ptrdiff_t(n * d)), xs2 = xs1;
    ml::block_forward_sequence<C>(backend, b.weights(), xs1, brope_for<C>(), one.act(0, 1));
    ml::block_forward_sequence<C>(backend, b.weights(), xs2, brope_for<C>(), one.act(0, 1), 1);
    EXPECT_EQ(xs1, xs2);
}

TEST(MlTransformerBlockTraining, ABatchOfSequencesIsEachSequencesCall)
{
    batch_is_each_sequence<b_gpt2>();
    batch_is_each_sequence<b_llama>();
    batch_is_each_sequence<b_nanochat>();
}

TEST(MlTransformerBlockTraining, BackwardMatchesCentralDifferencesForALlamaBlock)
{
    block_backward_is_central_differences<b_llama>();
}

TEST(MlTransformerBlockTraining, BackwardMatchesCentralDifferencesForANanochatBlock)
{
    block_backward_is_central_differences<b_nanochat>();
}

} // namespace
