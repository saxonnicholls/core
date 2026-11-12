//
//  generate_transformer_fixture.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Writes tests/fixtures/ml/transformer_fixture.txt: what two independent references compute for the transformer
//  blocks of tests/ml/transformer_block_test.cpp. Core is never used here.
//
//  1. "gpt2": one GPT-2 block as llm.c computes it (layernorm_forward, matmul_forward, attention_forward,
//     gelu_forward, residual_forward from llm.c's train_gpt2.c), forward and backward, in float. C = 32, NH = 2
//     (head size 16, so the score scale 0.25 is exact), T = 4 tokens.
//  2. "llama", "nanochat", "mqa": blocks with the features llm.c lacks (GQA and MQA, RMS norm, QK norm before and
//     after RoPE, both RoPE layouts, SwiGLU, ReLU², the exact GELU, a sliding window, a soft cap), computed here in
//     double by the textbook two-pass formulas.
//
//  Inputs and weights are not stored: both sides draw them from one splitmix64 stream, in the order of the "fill"
//  lines, as uniform floats in [-scale, scale) (u = (z >> 40) * 2^-24, value = float((2u - 1) * scale)).
//
//  Build (llm.c's train_gpt2.c compiled without its main; an empty omp.h stands in for OpenMP):
//
//    git clone https://github.com/karpathy/llm.c          # commit f1e2ace
//    mkdir -p stub && : > stub/omp.h
//    cc -O2 -ffp-contract=off -DTESTING -Istub -Illm.c -c llm.c/train_gpt2.c -o llmc.o
//    c++ -std=c++20 -O2 -ffp-contract=off scripts/generate_transformer_fixture.cpp llmc.o -lm -o gen
//    ./gen > tests/fixtures/ml/transformer_fixture.txt

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias, int B, int T, int C);
void layernorm_backward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C);
void matmul_forward(float* out, const float* inp, const float* weight, const float* bias, int B, int T, int C, int OC);
void matmul_backward(float* dinp, float* dweight, float* dbias, const float* dout, const float* inp, const float* weight, int B, int T,
                     int C, int OC);
void attention_forward(float* out, float* preatt, float* att, float* inp, int B, int T, int C, int NH);
void attention_backward(float* dinp, float* dpreatt, float* datt, float* dout, float* inp, float* att, int B, int T, int C, int NH);
void gelu_forward(float* out, float* inp, int N);
void gelu_backward(float* dinp, float* inp, float* dout, int N);
void residual_forward(float* out, float* inp1, float* inp2, int N);
void residual_backward(float* dinp1, float* dinp2, float* dout, int N);
}

namespace {

std::uint64_t state = 0x5eed2026;

float draw(float scale)
{
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    const double u = double(z >> 40) * 0x1p-24;
    return float((2.0 * u - 1.0) * scale);
}

std::vector<float> fill(const char* name, std::size_t n, float scale)
{
    std::printf("fill %s %zu %a\n", name, n, double(scale));
    std::vector<float> v(n);
    for (float& x : v)
        x = draw(scale);
    return v;
}

void put(const char* name, const float* p, std::size_t n)
{
    std::printf("%s %zu", name, n);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t b;
        std::memcpy(&b, &p[i], 4);
        std::printf(" %08x", b);
    }
    std::printf("\n");
}

void put(const char* name, const std::vector<double>& v)
{
    std::printf("%s %zu", name, v.size());
    for (const double x : v)
        std::printf(" %a", x);
    std::printf("\n");
}

// 1. GPT-2 as llm.c

void gpt2()
{
    constexpr int B = 1, T = 4, C = 32, NH = 2, F = 4 * C;
    std::printf("section gpt2\n");
    auto x = fill("x", T * C, 1.0f);
    auto ln1w = fill("ln1w", C, 1.0f), ln1b = fill("ln1b", C, 0.1f);
    auto qkvw = fill("qkvw", 3 * C * C, 0.2f), qkvb = fill("qkvb", 3 * C, 0.05f);
    auto attprojw = fill("attprojw", C * C, 0.2f), attprojb = fill("attprojb", C, 0.05f);
    auto ln2w = fill("ln2w", C, 1.0f), ln2b = fill("ln2b", C, 0.1f);
    auto fcw = fill("fcw", F * C, 0.2f), fcb = fill("fcb", F, 0.05f);
    auto fcprojw = fill("fcprojw", C * F, 0.1f), fcprojb = fill("fcprojb", C, 0.05f);
    auto dout = fill("dout", T * C, 1.0f);

    std::vector<float> ln1(T * C), ln1_mean(T), ln1_rstd(T), qkv(T * 3 * C), atty(T * C), preatt(NH * T * T), att(NH * T * T),
        attproj(T * C), residual2(T * C), ln2(T * C), ln2_mean(T), ln2_rstd(T), fch(T * F), fch_gelu(T * F), fcproj(T * C),
        residual3(T * C);
    layernorm_forward(ln1.data(), ln1_mean.data(), ln1_rstd.data(), x.data(), ln1w.data(), ln1b.data(), B, T, C);
    matmul_forward(qkv.data(), ln1.data(), qkvw.data(), qkvb.data(), B, T, C, 3 * C);
    attention_forward(atty.data(), preatt.data(), att.data(), qkv.data(), B, T, C, NH);
    matmul_forward(attproj.data(), atty.data(), attprojw.data(), attprojb.data(), B, T, C, C);
    residual_forward(residual2.data(), x.data(), attproj.data(), T * C);
    layernorm_forward(ln2.data(), ln2_mean.data(), ln2_rstd.data(), residual2.data(), ln2w.data(), ln2b.data(), B, T, C);
    matmul_forward(fch.data(), ln2.data(), fcw.data(), fcb.data(), B, T, C, F);
    gelu_forward(fch_gelu.data(), fch.data(), T * F);
    matmul_forward(fcproj.data(), fch_gelu.data(), fcprojw.data(), fcprojb.data(), B, T, F, C);
    residual_forward(residual3.data(), residual2.data(), fcproj.data(), T * C);
    put("ln1", ln1.data(), ln1.size());
    put("qkv", qkv.data(), qkv.size());
    put("atty", atty.data(), atty.size());
    put("residual2", residual2.data(), residual2.size());
    put("fch_gelu", fch_gelu.data(), fch_gelu.size());
    put("out", residual3.data(), residual3.size());

    // The backward pass, as llm.c's gpt2_backward does one layer: every gradient starts at zero and accumulates
    std::vector<float> dx(T * C), dln1w(C), dln1b(C), dqkvw(3 * C * C), dqkvb(3 * C), dattprojw(C * C), dattprojb(C), dln2w(C), dln2b(C),
        dfcw(F * C), dfcb(F), dfcprojw(C * F), dfcprojb(C), dresidual2(T * C), dfcproj(T * C), dfch_gelu(T * F), dfch(T * F), dln2(T * C),
        dattproj(T * C), datty(T * C), dpreatt(NH * T * T), datt(NH * T * T), dqkv(T * 3 * C), dln1(T * C);
    residual_backward(dresidual2.data(), dfcproj.data(), dout.data(), T * C);
    matmul_backward(dfch_gelu.data(), dfcprojw.data(), dfcprojb.data(), dfcproj.data(), fch_gelu.data(), fcprojw.data(), B, T, F, C);
    gelu_backward(dfch.data(), fch.data(), dfch_gelu.data(), T * F);
    matmul_backward(dln2.data(), dfcw.data(), dfcb.data(), dfch.data(), ln2.data(), fcw.data(), B, T, C, F);
    layernorm_backward(dresidual2.data(), dln2w.data(), dln2b.data(), dln2.data(), residual2.data(), ln2w.data(), ln2_mean.data(),
                       ln2_rstd.data(), B, T, C);
    residual_backward(dx.data(), dattproj.data(), dresidual2.data(), T * C);
    matmul_backward(datty.data(), dattprojw.data(), dattprojb.data(), dattproj.data(), atty.data(), attprojw.data(), B, T, C, C);
    attention_backward(dqkv.data(), dpreatt.data(), datt.data(), datty.data(), qkv.data(), att.data(), B, T, C, NH);
    matmul_backward(dln1.data(), dqkvw.data(), dqkvb.data(), dqkv.data(), ln1.data(), qkvw.data(), B, T, C, 3 * C);
    layernorm_backward(dx.data(), dln1w.data(), dln1b.data(), dln1.data(), x.data(), ln1w.data(), ln1_mean.data(), ln1_rstd.data(), B, T, C);
    put("dx", dx.data(), dx.size());
    put("dln1w", dln1w.data(), dln1w.size());
    put("dln1b", dln1b.data(), dln1b.size());
    put("dqkvw", dqkvw.data(), dqkvw.size());
    put("dqkvb", dqkvb.data(), dqkvb.size());
    put("dattprojw", dattprojw.data(), dattprojw.size());
    put("dattprojb", dattprojb.data(), dattprojb.size());
    put("dln2w", dln2w.data(), dln2w.size());
    put("dln2b", dln2b.data(), dln2b.size());
    put("dfcw", dfcw.data(), dfcw.size());
    put("dfcb", dfcb.data(), dfcb.size());
    put("dfcprojw", dfcprojw.data(), dfcprojw.size());
    put("dfcprojb", dfcprojb.data(), dfcprojb.size());
}

// 2. The textbook block in double

enum class norm { layer, rms };
enum class mlp { gelu, relu2, swiglu };
enum class rope { none, half, interleaved };
enum class qkn { none, before, after };

struct config
{
    const char* name;
    int d, heads, kv, hd, ff;
    norm nrm;
    double eps;
    qkn qk;
    bool qk_weights;
    mlp m;
    rope r;
    double base;
    int window;
    double cap;
    bool bias;
    int tokens;
};

using vec = std::vector<double>;

vec widen(const std::vector<float>& v)
{
    return vec(v.begin(), v.end());
}

// y = W x + b, W rows x cols row-major
vec matvec(const vec& w, const vec& b, const vec& x, int rows, int cols)
{
    vec y(rows);
    for (int i = 0; i < rows; ++i) {
        double s = b.empty() ? 0.0 : b[i];
        for (int j = 0; j < cols; ++j)
            s += w[std::size_t(i) * cols + j] * x[j];
        y[i] = s;
    }
    return y;
}

vec normalise(const vec& x, const vec& w, const vec& b, norm kind, double eps)
{
    const int n = int(x.size());
    vec y(n);
    if (kind == norm::layer) {
        double m = 0, v = 0;
        for (double e : x)
            m += e;
        m /= n;
        for (double e : x)
            v += (e - m) * (e - m);
        v /= n;
        const double s = 1.0 / std::sqrt(v + eps);
        for (int i = 0; i < n; ++i)
            y[i] = (x[i] - m) * s * (w.empty() ? 1.0 : w[i]) + (b.empty() ? 0.0 : b[i]);
    } else {
        double ms = 0;
        for (double e : x)
            ms += e * e;
        const double s = 1.0 / std::sqrt(ms / n + eps);
        for (int i = 0; i < n; ++i)
            y[i] = x[i] * s * (w.empty() ? 1.0 : w[i]);
    }
    return y;
}

void rotate(double* h, int hd, int pos, rope r, double base)
{
    if (r == rope::none)
        return;
    for (int i = 0; i < hd / 2; ++i) {
        const double a = pos / std::pow(base, 2.0 * i / hd), c = std::cos(a), s = std::sin(a);
        double& x1 = r == rope::half ? h[i] : h[2 * i];
        double& x2 = r == rope::half ? h[i + hd / 2] : h[2 * i + 1];
        const double u = x1, v = x2;
        x1 = u * c - v * s;
        x2 = u * s + v * c;
    }
}

void head_norm(double* h, int hd, const vec& w, double eps)
{
    vec x(h, h + hd);
    const vec y = normalise(x, w, {}, norm::rms, eps);
    std::copy(y.begin(), y.end(), h);
}

void textbook(const config& c)
{
    std::printf("section %s\n", c.name);
    const int d = c.d, q_size = c.heads * c.hd, kv_size = c.kv * c.hd, n = c.tokens;
    auto x = widen(fill("x", std::size_t(n) * d, 1.0f));
    auto n1w = widen(fill("norm1_weight", d, 1.0f)), n1b = c.bias ? widen(fill("norm1_bias", d, 0.1f)) : vec{};
    auto wq = widen(fill("wq", std::size_t(q_size) * d, 0.2f)), wk = widen(fill("wk", std::size_t(kv_size) * d, 0.2f)),
         wv = widen(fill("wv", std::size_t(kv_size) * d, 0.2f)), wo = widen(fill("wo", std::size_t(d) * q_size, 0.2f));
    vec bq, bk, bv, bo;
    if (c.bias) {
        bq = widen(fill("bq", q_size, 0.05f));
        bk = widen(fill("bk", kv_size, 0.05f));
        bv = widen(fill("bv", kv_size, 0.05f));
        bo = widen(fill("bo", d, 0.05f));
    }
    vec qn, kn;
    if (c.qk_weights) {
        qn = widen(fill("q_norm", c.hd, 1.0f));
        kn = widen(fill("k_norm", c.hd, 1.0f));
    }
    auto n2w = widen(fill("norm2_weight", d, 1.0f)), n2b = c.bias ? widen(fill("norm2_bias", d, 0.1f)) : vec{};
    auto up = widen(fill("up", std::size_t(c.ff) * d, 0.2f));
    vec gate = c.m == mlp::swiglu ? widen(fill("gate", std::size_t(c.ff) * d, 0.2f)) : vec{};
    auto down = widen(fill("down", std::size_t(d) * c.ff, 0.1f));
    vec bup, bdown;
    if (c.bias) {
        bup = widen(fill("up_bias", c.ff, 0.05f));
        bdown = widen(fill("down_bias", d, 0.05f));
    }

    std::vector<vec> keys, values;   // per position, kv_size
    vec out;
    for (int t = 0; t < n; ++t) {
        vec s(x.begin() + std::size_t(t) * d, x.begin() + std::size_t(t + 1) * d);
        const vec h = normalise(s, n1w, n1b, c.nrm, c.eps);
        vec q = matvec(wq, bq, h, q_size, d), k = matvec(wk, bk, h, kv_size, d), v = matvec(wv, bv, h, kv_size, d);
        for (int hh = 0; hh < c.heads; ++hh) {
            if (c.qk == qkn::before)
                head_norm(&q[hh * c.hd], c.hd, qn, c.eps);
            rotate(&q[hh * c.hd], c.hd, t, c.r, c.base);
            if (c.qk == qkn::after)
                head_norm(&q[hh * c.hd], c.hd, qn, c.eps);
        }
        for (int hh = 0; hh < c.kv; ++hh) {
            if (c.qk == qkn::before)
                head_norm(&k[hh * c.hd], c.hd, kn, c.eps);
            rotate(&k[hh * c.hd], c.hd, t, c.r, c.base);
            if (c.qk == qkn::after)
                head_norm(&k[hh * c.hd], c.hd, kn, c.eps);
        }
        keys.push_back(k);
        values.push_back(v);
        const int first = c.window == 0 || c.window > t ? 0 : t + 1 - c.window;
        vec y(q_size, 0.0);
        for (int hh = 0; hh < c.heads; ++hh) {
            const int g = hh / (c.heads / c.kv);
            vec score;
            double mx = -INFINITY;
            for (int j = first; j <= t; ++j) {
                double sc = 0;
                for (int i = 0; i < c.hd; ++i)
                    sc += q[hh * c.hd + i] * keys[j][g * c.hd + i];
                sc /= std::sqrt(double(c.hd));
                if (c.cap > 0)
                    sc = c.cap * std::tanh(sc / c.cap);
                score.push_back(sc);
                mx = std::max(mx, sc);
            }
            double sum = 0;
            for (double& e : score)
                sum += (e = std::exp(e - mx));
            for (int j = first; j <= t; ++j)
                for (int i = 0; i < c.hd; ++i)
                    y[hh * c.hd + i] += score[j - first] / sum * values[j][g * c.hd + i];
        }
        const vec a = matvec(wo, bo, y, d, q_size);
        for (int i = 0; i < d; ++i)
            s[i] += a[i];
        const vec h2 = normalise(s, n2w, n2b, c.nrm, c.eps);
        vec u = matvec(up, bup, h2, c.ff, d);
        if (c.m == mlp::swiglu) {
            const vec gg = matvec(gate, {}, h2, c.ff, d);
            for (int i = 0; i < c.ff; ++i)
                u[i] = gg[i] / (1.0 + std::exp(-gg[i])) * u[i];
        } else {
            for (double& e : u)
                e = c.m == mlp::relu2 ? (e > 0 ? e * e : 0.0) : 0.5 * e * (1.0 + std::erf(e / std::sqrt(2.0)));
        }
        const vec m2 = matvec(down, bdown, u, d, c.ff);
        for (int i = 0; i < d; ++i)
            out.push_back(s[i] + m2[i]);
    }
    put("out", out);
}

} // namespace

int main()
{
    std::printf("# tests/fixtures/ml/transformer_fixture.txt: written by scripts/generate_transformer_fixture.cpp; never edit by hand\n");
    gpt2();
    textbook({"llama", 32, 4, 2, 8, 64, norm::rms, 1e-5, qkn::none, false, mlp::swiglu, rope::half, 10000, 0, 0, false, 6});
    textbook({"nanochat", 32, 4, 4, 8, 64, norm::rms, 1.1920928955078125e-07, qkn::after, false, mlp::relu2, rope::half, 10000, 3, 5.0,
              false, 6});
    textbook({"mqa", 32, 4, 1, 8, 64, norm::layer, 1e-5, qkn::before, true, mlp::gelu, rope::interleaved, 500, 0, 0, true, 6});
}
