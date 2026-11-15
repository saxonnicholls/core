//
//  transformer_block_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  One decode step of a GPT-2-small layer (768 wide, 12 heads of 64, MLP of 3072, layer norm, tanh GELU, biases) at
//  position 256 over a filled KV cache: ml::block_forward against the same layer written as direct loops over raw
//  pointers (the same projections, the same one-pass attention fold, the same order of every multiply and add), with
//  float and with Q8_0 weights. A fairness check refuses to benchmark unless the outputs and the appended keys and
//  values agree to the bit.
//
//  Training: a GPT-2-shaped layer (256 wide, 4 heads of 64, MLP of 1024, layer norm, tanh GELU, biases) over a
//  sequence of 64 tokens, ml::block_forward_sequence and ml::block_backward (which run attention_backward,
//  mlp_backward and linear_backward) against the same passes as direct loops: llm.c's matmul_backward, flash
//  attention's backward from the log-sum-exp, the layer norms through ml::layer_norm_backward. The fairness check
//  compares every saved activation and every gradient to the bit.
//

#include <snicholls/core/ml/transformer_block.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr ml::transformer_config c = ml::gpt2_small;
constexpr std::size_t d = c.d_model, f = c.d_ff, hd = c.head_dim, heads = c.n_heads, positions = 257, pos = 256;

std::vector<float> stream(std::size_t n, float scale, std::uint64_t seed)
{
    std::vector<float> v(n);
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ULL + 1;
    for (float& e : v) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        e = scale * static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 - 0.5);
    }
    return v;
}

struct layer_t {
    std::vector<float> ln1w = stream(d, 2, 1), ln1b = stream(d, 0.2f, 2), qkvw = stream(3 * d * d, 0.1f, 3), qkvb = stream(3 * d, 0.1f, 4),
                       ow = stream(d * d, 0.1f, 5), ob = stream(d, 0.1f, 6), ln2w = stream(d, 2, 7), ln2b = stream(d, 0.2f, 8),
                       upw = stream(f * d, 0.1f, 9), upb = stream(f, 0.1f, 10), downw = stream(d * f, 0.05f, 11), downb = stream(d, 0.1f, 12);
    std::vector<ml::block_q8_0> qkvq, oq, upq, downq;
    layer_t() : qkvq(3 * d * d / 32), oq(d * d / 32), upq(f * d / 32), downq(d * f / 32)
    {
        ml::quantize(qkvw, std::span<ml::block_q8_0>(qkvq));
        ml::quantize(ow, std::span<ml::block_q8_0>(oq));
        ml::quantize(upw, std::span<ml::block_q8_0>(upq));
        ml::quantize(downw, std::span<ml::block_q8_0>(downq));
    }

    template <typename W>
    ml::block_weights<W> weights(W q, W k, W v, W o, W up, W down) const
    {
        ml::block_weights<W> w{};
        w.norm1_weight = ln1w;
        w.norm1_bias = ln1b;
        w.attention = {q, k, v, o, std::span<const float>(qkvb).subspan(0, d), std::span<const float>(qkvb).subspan(d, d),
                       std::span<const float>(qkvb).subspan(2 * d, d), ob};
        w.norm2_weight = ln2w;
        w.norm2_bias = ln2b;
        w.mlp = {up, W{}, down, upb, {}, downb};
        return w;
    }
    ml::block_weights<ml::dense_weights<float>> dense() const
    {
        using W = ml::dense_weights<float>;
        const std::span<const float> qkv(qkvw);
        return weights(W{qkv.subspan(0, d * d), d, d}, W{qkv.subspan(d * d, d * d), d, d}, W{qkv.subspan(2 * d * d, d * d), d, d}, W{ow, d, d},
                       W{upw, f, d}, W{downw, d, f});
    }
    ml::block_weights<ml::quantized_weights<ml::block_q8_0>> q8() const
    {
        using W = ml::quantized_weights<ml::block_q8_0>;
        const std::span<const ml::block_q8_0> qkv(qkvq);
        const std::size_t m = d * d / 32;
        return weights(W{qkv.subspan(0, m), d, d}, W{qkv.subspan(m, m), d, d}, W{qkv.subspan(2 * m, m), d, d}, W{oq, d, d}, W{upq, f, d},
                       W{downq, d, f});
    }
};

const layer_t& layer()
{
    static const layer_t l;
    return l;
}

struct state_t {
    std::vector<float> x = stream(d, 2, 20), keys = stream(positions * d, 1, 21), values = stream(positions * d, 1, 22);
    ml::kv_cache_view<c> cache() { return {keys, values, 1, positions}; }
};

// ---- the same layer by hand ----

void linear_by_hand(const float* w, const float* b, const float* x, float* y, std::size_t rows, std::size_t cols)
{
    for (std::size_t i = 0; i < rows; ++i) {
        float acc = b[i];
        for (std::size_t j = 0; j < cols; ++j) {
            const float p = w[i * cols + j] * x[j];
            acc += p;
        }
        y[i] = acc;
    }
}

void linear_by_hand(const ml::block_q8_0* w, const float* b, const ml::block_q8_0* xq, float* y, std::size_t rows, std::size_t cols)
{
    for (std::size_t i = 0; i < rows; ++i)
        y[i] = b[i] + ml::dot(std::span<const ml::block_q8_0>(w + i * cols / 32, cols / 32), std::span<const ml::block_q8_0>(xq, cols / 32));
}

// The input as each storage reads it: as it is, or quantised to Q8_0 once for every product that shares it
const float* prep(const float*, const float* x, std::size_t, ml::block_q8_0*)
{
    return x;
}
const ml::block_q8_0* prep(const ml::block_q8_0*, const float* x, std::size_t cols, ml::block_q8_0* buffer)
{
    ml::quantize(std::span<const float>(x, cols), std::span<ml::block_q8_0>(buffer, cols / 32));
    return buffer;
}

template <typename Wt>
void block_by_hand(const layer_t& l, const Wt* qkv, const Wt* o, const Wt* up, const Wt* down, std::size_t block, float* x, float* keys,
                   float* values)
{
    float h[d], q[d], k[d], v[d], y[d], a[d], hidden[f];
    ml::block_q8_0 buffer[f / 32];
    ml::layer_norm(std::span<const float>(x, d), l.ln1w, l.ln1b, std::span<float>(h, d), 1e-5f);
    const auto in = prep(qkv, h, d, buffer);
    linear_by_hand(qkv, l.qkvb.data(), in, q, d, d);
    linear_by_hand(qkv + block, l.qkvb.data() + d, in, k, d, d);
    linear_by_hand(qkv + 2 * block, l.qkvb.data() + 2 * d, in, v, d, d);
    for (std::size_t h2 = 0; h2 < heads; ++h2)   // the cache: [kv_head][position][head_dim]
        for (std::size_t i = 0; i < hd; ++i) {
            keys[(h2 * positions + pos) * hd + i] = k[h2 * hd + i];
            values[(h2 * positions + pos) * hd + i] = v[h2 * hd + i];
        }
    const float scale = 1.0f / snicholls::math::sqrt(float(hd));
    for (std::size_t h2 = 0; h2 < heads; ++h2) {
        float max = -std::numeric_limits<float>::infinity(), sum = 0, acc[hd] = {};
        for (std::size_t j = 0; j <= pos; ++j) {
            const float* kj = keys + (h2 * positions + j) * hd;
            const float* vj = values + (h2 * positions + j) * hd;
            float s = 0;
            for (std::size_t i = 0; i < hd; ++i)
                s += q[h2 * hd + i] * kj[i];
            s = scale * s;
            if (s > max) {
                const float r = snicholls::math::exp(max - s);
                for (std::size_t i = 0; i < hd; ++i)
                    acc[i] = acc[i] * r + vj[i];
                sum = sum * r + 1.0f;
                max = s;
            } else {
                const float w = snicholls::math::exp(s - max);
                for (std::size_t i = 0; i < hd; ++i)
                    acc[i] += w * vj[i];
                sum += w;
            }
        }
        for (std::size_t i = 0; i < hd; ++i)
            y[h2 * hd + i] = acc[i] / sum;
    }
    linear_by_hand(o, l.ob.data(), prep(o, y, d, buffer), a, d, d);
    for (std::size_t i = 0; i < d; ++i)
        x[i] += a[i];
    ml::layer_norm(std::span<const float>(x, d), l.ln2w, l.ln2b, std::span<float>(h, d), 1e-5f);
    linear_by_hand(up, l.upb.data(), prep(up, h, d, buffer), hidden, f, d);
    for (float& e : hidden)
        e = ml::gelu_tanh(e);
    linear_by_hand(down, l.downb.data(), prep(down, hidden, f, buffer), a, d, f);
    for (std::size_t i = 0; i < d; ++i)
        x[i] += a[i];
}

void by_hand_dense(state_t& s)
{
    const layer_t& l = layer();
    block_by_hand(l, l.qkvw.data(), l.ow.data(), l.upw.data(), l.downw.data(), d * d, s.x.data(), s.keys.data(), s.values.data());
}

void by_hand_q8(state_t& s)
{
    const layer_t& l = layer();
    block_by_hand(l, l.qkvq.data(), l.oq.data(), l.upq.data(), l.downq.data(), d * d / 32, s.x.data(), s.keys.data(), s.values.data());
}

void core_dense(state_t& s)
{
    static const auto w = layer().dense();
    ml::block_forward<c>(w, s.x, pos, s.cache(), 0, ml::no_rope);
}

void core_q8(state_t& s)
{
    static const auto w = layer().q8();
    ml::block_forward<c>(w, s.x, pos, s.cache(), 0, ml::no_rope);
}

template <void (*Step)(state_t&)>
void BM_Decode(benchmark::State& state)
{
    state_t s;
    const std::vector<float> x0 = s.x;
    for (auto _ : state) {
        s.x = x0;   // the same token every iteration, so every iteration does the same work
        Step(s);
        benchmark::DoNotOptimize(s.x.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations());   // tokens
}
BENCHMARK(BM_Decode<core_dense>)->Name("BM_DecodeGpt2Layer_Float_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Decode<by_hand_dense>)->Name("BM_DecodeGpt2Layer_Float_HandWritten")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Decode<core_q8>)->Name("BM_DecodeGpt2Layer_Q8_0_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Decode<by_hand_q8>)->Name("BM_DecodeGpt2Layer_Q8_0_HandWritten")->Unit(benchmark::kMicrosecond);

// ---- training: a GPT-2-shaped layer over a sequence, forward with its activations saved, then backward ----

constexpr ml::transformer_config tc{.d_model = 256, .n_heads = 4, .n_kv_heads = 4, .head_dim = 64, .d_ff = 1024,
                                    .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .mlp = ml::mlp_kind::gelu_tanh, .bias = true};
constexpr std::size_t td = tc.d_model, tf = tc.d_ff, th = tc.n_heads, thd = tc.head_dim, tn = 64;
constexpr float teps = 1e-5f;

struct tparams_t {
    std::vector<float> ln1w = stream(td, 2, 30), ln1b = stream(td, 0.2f, 31), wq = stream(td * td, 0.2f, 32), wk = stream(td * td, 0.2f, 33),
                       wv = stream(td * td, 0.2f, 34), bq = stream(td, 0.1f, 35), bk = stream(td, 0.1f, 36), bv = stream(td, 0.1f, 37),
                       wo = stream(td * td, 0.1f, 38), bo = stream(td, 0.1f, 39), ln2w = stream(td, 2, 40), ln2b = stream(td, 0.2f, 41),
                       wup = stream(tf * td, 0.1f, 42), bup = stream(tf, 0.1f, 43), wdown = stream(td * tf, 0.05f, 44), bdown = stream(td, 0.1f, 45);
    ml::block_weights<ml::dense_weights<float>> weights() const
    {
        using W = ml::dense_weights<float>;
        ml::block_weights<W> w{};
        w.norm1_weight = ln1w;
        w.norm1_bias = ln1b;
        w.attention = {W{wq, td, td}, W{wk, td, td}, W{wv, td, td}, W{wo, td, td}, bq, bk, bv, bo};
        w.norm2_weight = ln2w;
        w.norm2_bias = ln2b;
        w.mlp = {W{wup, tf, td}, W{}, W{wdown, td, tf}, bup, {}, bdown};
        return w;
    }
};

const tparams_t& tparams()
{
    static const tparams_t p;
    return p;
}

// One sequence's residual rows, what the forward pass saves, the output gradient, and every gradient
struct tstate_t {
    using v = std::vector<float>;
    v xs = stream(tn * td, 2, 50), dout = stream(tn * td, 1, 51), dxs = dout;
    v x0 = v(tn * td), h1 = v(tn * td), x1 = v(tn * td), h2 = v(tn * td), q = v(tn * td), k = v(tn * td), vv = v(tn * td),
      y = v(tn * td), lse = v(tn * th), up = v(tn * tf);
    v gln1w = v(td), gln1b = v(td), gln2w = v(td), gln2b = v(td), gq = v(tn * td), gk = v(tn * td), gv = v(tn * td),
      gwq = v(td * td), gwk = v(td * td), gwv = v(td * td), gwo = v(td * td), gbq = v(td), gbk = v(td), gbv = v(td), gbo = v(td),
      gup = v(tf * td), gupb = v(tf), gdown = v(td * tf), gdownb = v(td), scratch = v(tn * td);
    ml::block_activations<float> act() { return {x0, h1, x1, h2, {q, k, vv, {}, {}, y, lse}, up, {}}; }
    ml::block_gradients<float> grads()
    {
        return {gln1w, gln1b, gln2w, gln2b, {gq, gk, gv, gwq, gwk, gwv, gwo, gbq, gbk, gbv, gbo, {}, {}},
                {gup, {}, gdown, gupb, {}, gdownb}, scratch};
    }
};

// llm.c's matmul_backward for one row: dx += Wᵀ dy, dW += dy xᵀ, db += dy, in linear_backward's order
void linear_backward_by_hand(const float* w, const float* x, const float* dy, float* dx, float* dw, float* db, std::size_t rows,
                             std::size_t cols)
{
    for (std::size_t i = 0; i < rows; ++i) {
        const float g = dy[i];
        for (std::size_t j = 0; j < cols; ++j) {
            const float p = w[i * cols + j] * g;
            dx[j] += p;
        }
        for (std::size_t j = 0; j < cols; ++j) {
            const float p = g * x[j];
            dw[i * cols + j] += p;
        }
        db[i] += g;
    }
}

float tdot(const float* a, const float* b)
{
    float s = 0;
    for (std::size_t i = 0; i < thd; ++i)
        s += a[i] * b[i];
    return s;
}

std::span<const float> row_of(const std::vector<float>& v, std::size_t t, std::size_t width = td)
{
    return std::span<const float>(v).subspan(t * width, width);
}

void forward_by_hand(tstate_t& s)
{
    const tparams_t& w = tparams();
    float* x = s.xs.data();
    std::copy(s.xs.begin(), s.xs.end(), s.x0.begin());
    for (std::size_t t = 0; t < tn; ++t)
        ml::layer_norm(row_of(s.xs, t), w.ln1w, w.ln1b, std::span<float>(s.h1).subspan(t * td, td), teps);
    const float scale = 1.0f / snicholls::math::sqrt(float(thd));
    for (std::size_t t = 0; t < tn; ++t) {
        const float* in = s.h1.data() + t * td;
        float k[td], v[td];
        linear_by_hand(w.wq.data(), w.bq.data(), in, s.q.data() + t * td, td, td);
        linear_by_hand(w.wk.data(), w.bk.data(), in, k, td, td);
        linear_by_hand(w.wv.data(), w.bv.data(), in, v, td, td);
        for (std::size_t h = 0; h < th; ++h)   // saved as [head][position][head_dim]
            for (std::size_t i = 0; i < thd; ++i) {
                s.k[(h * tn + t) * thd + i] = k[h * thd + i];
                s.vv[(h * tn + t) * thd + i] = v[h * thd + i];
            }
        for (std::size_t h = 0; h < th; ++h) {
            const float* q = s.q.data() + t * td + h * thd;
            float* out = s.y.data() + t * td + h * thd;
            float max = -std::numeric_limits<float>::infinity(), sum = 0, acc[thd] = {};
            for (std::size_t j = 0; j <= t; ++j) {
                const float* kj = s.k.data() + (h * tn + j) * thd;
                const float* vj = s.vv.data() + (h * tn + j) * thd;
                const float sc = scale * tdot(q, kj);
                if (sc > max) {
                    const float r = snicholls::math::exp(max - sc);
                    for (std::size_t i = 0; i < thd; ++i)
                        acc[i] = acc[i] * r + vj[i];
                    sum = sum * r + 1.0f;
                    max = sc;
                } else {
                    const float e = snicholls::math::exp(sc - max);
                    for (std::size_t i = 0; i < thd; ++i)
                        acc[i] += e * vj[i];
                    sum += e;
                }
            }
            for (std::size_t i = 0; i < thd; ++i)
                out[i] = acc[i] / sum;
            s.lse[t * th + h] = max + snicholls::math::log(sum);
        }
        linear_by_hand(w.wo.data(), w.bo.data(), s.y.data() + t * td, x + t * td, td, td);
    }
    for (std::size_t i = 0; i < tn * td; ++i)
        x[i] = s.x0[i] + x[i];
    std::copy(s.xs.begin(), s.xs.end(), s.x1.begin());
    for (std::size_t t = 0; t < tn; ++t) {
        float hidden[tf], a[td];
        ml::layer_norm(row_of(s.xs, t), w.ln2w, w.ln2b, std::span<float>(s.h2).subspan(t * td, td), teps);
        linear_by_hand(w.wup.data(), w.bup.data(), s.h2.data() + t * td, s.up.data() + t * tf, tf, td);
        for (std::size_t i = 0; i < tf; ++i)
            hidden[i] = ml::gelu_tanh(s.up[t * tf + i]);
        linear_by_hand(w.wdown.data(), w.bdown.data(), hidden, a, td, tf);
        for (std::size_t i = 0; i < td; ++i)
            x[t * td + i] += a[i];
    }
}

// The backward of a layer norm row: recompute its statistics, then layer_norm_backward
void norm_backward_by_hand(std::span<const float> dh, std::span<const float> x, const std::vector<float>& weight,
                           const std::vector<float>& bias, float* dn, std::vector<float>& dweight, std::vector<float>& dbias)
{
    float scratch[td];
    const auto stats = ml::layer_norm(x, weight, bias, std::span<float>(scratch, td), teps);
    ml::layer_norm_backward(dh, x, weight, stats, std::span<float>(dn, td), std::span<float>(dweight), std::span<float>(dbias));
}

void backward_by_hand(tstate_t& s)
{
    const tparams_t& w = tparams();
    float* g = s.dxs.data();
    for (std::size_t t = 0; t < tn; ++t) {   // the MLP and the second norm, plus the residual
        float hidden[tf], dhid[tf] = {}, dh[td] = {}, dn[td];
        const float* up = s.up.data() + t * tf;
        for (std::size_t i = 0; i < tf; ++i)
            hidden[i] = ml::gelu_tanh(up[i]);
        linear_backward_by_hand(w.wdown.data(), hidden, g + t * td, dhid, s.gdown.data(), s.gdownb.data(), td, tf);
        for (std::size_t i = 0; i < tf; ++i)
            dhid[i] = dhid[i] * ml::gelu_tanh_derivative(up[i]);
        linear_backward_by_hand(w.wup.data(), s.h2.data() + t * td, dhid, dh, s.gup.data(), s.gupb.data(), tf, td);
        norm_backward_by_hand(std::span<const float>(dh, td), row_of(s.x1, t), w.ln2w, w.ln2b, dn, s.gln2w, s.gln2b);
        for (std::size_t i = 0; i < td; ++i)
            g[t * td + i] += dn[i];
    }
    std::fill(s.scratch.begin(), s.scratch.end(), 0.0f);
    std::fill(s.gk.begin(), s.gk.end(), 0.0f);
    std::fill(s.gv.begin(), s.gv.end(), 0.0f);
    const float scale = 1.0f / snicholls::math::sqrt(float(thd));
    for (std::size_t t = 0; t < tn; ++t) {   // Wo, then flash attention's backward per head
        float dy[td] = {};
        const float* yt = s.y.data() + t * td;
        linear_backward_by_hand(w.wo.data(), yt, g + t * td, dy, s.gwo.data(), s.gbo.data(), td, td);
        for (std::size_t h = 0; h < th; ++h) {
            const float* q = s.q.data() + t * td + h * thd;
            const float* gout = dy + h * thd;
            float* gq = s.gq.data() + t * td + h * thd;
            for (std::size_t i = 0; i < thd; ++i)
                gq[i] = 0.0f;
            const float lse = s.lse[t * th + h], big_d = tdot(gout, yt + h * thd);
            for (std::size_t j = 0; j <= t; ++j) {
                const std::size_t at = (h * tn + j) * thd;
                const float* kj = s.k.data() + at;
                const float p = snicholls::math::exp(scale * tdot(q, kj) - lse);
                const float ds = p * (tdot(gout, s.vv.data() + at) - big_d);
                for (std::size_t i = 0; i < thd; ++i) {
                    s.gv[at + i] += p * gout[i];
                    gq[i] += scale * ds * kj[i];
                    s.gk[at + i] += scale * ds * q[i];
                }
            }
        }
    }
    for (std::size_t t = 0; t < tn; ++t) {   // Wq, Wk, Wv into the first norm's output
        float dk[td], dv[td];
        for (std::size_t h = 0; h < th; ++h)
            for (std::size_t i = 0; i < thd; ++i) {
                dk[h * thd + i] = s.gk[(h * tn + t) * thd + i];
                dv[h * thd + i] = s.gv[(h * tn + t) * thd + i];
            }
        const float* xt = s.h1.data() + t * td;
        float* dxt = s.scratch.data() + t * td;
        linear_backward_by_hand(w.wq.data(), xt, s.gq.data() + t * td, dxt, s.gwq.data(), s.gbq.data(), td, td);
        linear_backward_by_hand(w.wk.data(), xt, dk, dxt, s.gwk.data(), s.gbk.data(), td, td);
        linear_backward_by_hand(w.wv.data(), xt, dv, dxt, s.gwv.data(), s.gbv.data(), td, td);
    }
    for (std::size_t t = 0; t < tn; ++t) {   // the first norm, plus the residual
        float dn[td];
        norm_backward_by_hand(row_of(s.scratch, t), row_of(s.x0, t), w.ln1w, w.ln1b, dn, s.gln1w, s.gln1b);
        for (std::size_t i = 0; i < td; ++i)
            g[t * td + i] += dn[i];
    }
}

void forward_core(tstate_t& s)
{
    static const auto w = tparams().weights();
    ml::block_forward_sequence<tc>(w, s.xs, ml::no_rope, s.act());
}

void backward_core(tstate_t& s)
{
    static const auto w = tparams().weights();
    ml::block_backward<tc>(w, ml::no_rope, s.act(), s.dxs, s.grads());
}

template <void (*Forward)(tstate_t&)>
void BM_TrainForward(benchmark::State& state)
{
    tstate_t s;
    const std::vector<float> x0 = s.xs;
    for (auto _ : state) {
        s.xs = x0;
        Forward(s);
        benchmark::DoNotOptimize(s.xs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(tn));   // tokens
}

template <void (*Forward)(tstate_t&), void (*Backward)(tstate_t&)>
void BM_TrainBackward(benchmark::State& state)
{
    tstate_t s;
    Forward(s);
    for (auto _ : state) {
        s.dxs = s.dout;   // the same output gradient every iteration; the weights' gradients keep accumulating
        Backward(s);
        benchmark::DoNotOptimize(s.dxs.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * std::int64_t(tn));
}
BENCHMARK(BM_TrainForward<forward_core>)->Name("BM_TrainGpt2ShapedLayerForward_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TrainForward<forward_by_hand>)->Name("BM_TrainGpt2ShapedLayerForward_HandWritten")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TrainBackward<forward_core, backward_core>)->Name("BM_TrainGpt2ShapedLayerBackward_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TrainBackward<forward_by_hand, backward_by_hand>)
    ->Name("BM_TrainGpt2ShapedLayerBackward_HandWritten")
    ->Unit(benchmark::kMicrosecond);

bool same(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

template <void (*A)(state_t&), void (*B)(state_t&)>
bool agree()
{
    state_t a, b;
    A(a);
    B(b);
    return same(a.x, b.x) && same(a.keys, b.keys) && same(a.values, b.values);
}

bool agree_training()
{
    tstate_t a, b;
    forward_core(a);
    forward_by_hand(b);
    using m = std::vector<float> tstate_t::*;
    for (const m f : {&tstate_t::xs, &tstate_t::x0, &tstate_t::h1, &tstate_t::x1, &tstate_t::h2, &tstate_t::q, &tstate_t::k,
                      &tstate_t::vv, &tstate_t::y, &tstate_t::lse, &tstate_t::up})
        if (!same(a.*f, b.*f))
            return false;
    backward_core(a);
    backward_by_hand(b);
    for (const m f : {&tstate_t::dxs, &tstate_t::gln1w, &tstate_t::gln1b, &tstate_t::gln2w, &tstate_t::gln2b, &tstate_t::gq,
                      &tstate_t::gk, &tstate_t::gv, &tstate_t::gwq, &tstate_t::gwk, &tstate_t::gwv, &tstate_t::gwo, &tstate_t::gbq,
                      &tstate_t::gbk, &tstate_t::gbv, &tstate_t::gbo, &tstate_t::gup, &tstate_t::gupb, &tstate_t::gdown,
                      &tstate_t::gdownb, &tstate_t::scratch})
        if (!same(a.*f, b.*f))
            return false;
    return true;
}

bool fair()
{
    return agree<core_dense, by_hand_dense>() && agree<core_q8, by_hand_q8>() && agree_training();
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "transformer_block benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
