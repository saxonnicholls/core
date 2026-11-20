//
//  rwkv_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::wkv6_forward and wkv7_forward (RWKV-6's and RWKV-7's WKV through the recurrence driver) against the
//  same recurrences written out by hand, over 256 tokens of 8 heads of 64: the same exp, the same products in the same
//  order (ggml's). main() runs a fairness check first and refuses to benchmark if a pair disagrees in a single bit
//  (CLAUDE.md "Zero overhead").
//
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/rwkv.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;

constexpr ml::transformer_config C = [] {
    auto c = ml::tiny_gated_deltanet;
    c.mixer = ml::mixer_kind::rwkv7;
    c.mixer_heads = 8;
    c.mixer_head_dim = 64;
    c.mixer_state = 64;
    c.mixer_groups = 8;
    c.mixer_chunk = 64;
    c.hybrid_period = c.hybrid_attention_offset = 0;
    c.conv_width = 0;
    return c;
}();
constexpr std::size_t n = 256, H = 8, N = 64, S = H * N * N;

std::vector<float> make(std::size_t count, double lo, double hi, std::uint64_t seed)
{
    std::vector<float> out(count);
    std::uint64_t x = seed;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(lo + (hi - lo) * static_cast<double>(x >> 11) * 0x1.0p-53);
    }
    return out;
}

bool bits_equal(const std::vector<float>& a, const std::vector<float>& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

template <void (*F)(std::vector<float>&, std::vector<float>&)>
void run(benchmark::State& state)
{
    std::vector<float> y(n * H * N), s(S);
    for (auto _ : state) {
        F(y, s);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

bool same_pair(void (*a)(std::vector<float>&, std::vector<float>&), void (*b)(std::vector<float>&, std::vector<float>&))
{
    std::vector<float> ya(n * H * N), yb(n * H * N), sa(S), sb(S);
    a(ya, sa);
    b(yb, sb);
    return bits_equal(ya, yb) && bits_equal(sa, sb);
}

const std::vector<float> rs = make(n * H * N, -1, 1, 1), ks = make(n * H * N, -1, 1, 2), vs = make(n * H * N, -1, 1, 3),
                         lws = make(n * H * N, -0.6, -0.001, 4), us = make(H * N, -1, 1, 5), as = make(n * H * N, -0.1, 0.1, 6),
                         bs = make(n * H * N, -0.1, 0.1, 7);

void core6(std::vector<float>& o, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    ml::wkv6_forward<C>(snicholls::linalg::reference, ml::wkv6_inputs<float>{rs, ks, vs, lws, us}, std::span<float>(o), 1, std::span<float>(s));
}

void by_hand6(std::vector<float>& o, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    float out[N];
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < H; ++h) {
            const std::size_t base = (t * H + h) * N;
            float* st = s.data() + h * N * N;
            for (std::size_t p = 0; p < N; ++p)
                out[p] = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                const float k = ks[base + i], r = rs[base + i], u = us[h * N + i], w = std::exp(lws[base + i]);
                for (std::size_t p = 0; p < N; ++p) {
                    const float kv = vs[base + p] * k, prev = st[i * N + p];
                    out[p] += (kv * u + prev) * r;
                    st[i * N + p] = prev * w + kv;
                }
            }
            for (std::size_t p = 0; p < N; ++p)
                o[base + p] = out[p];
        }
}

void core7(std::vector<float>& o, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    ml::wkv7_forward<C>(snicholls::linalg::reference, ml::wkv7_inputs<float>{rs, lws, ks, vs, as, bs}, std::span<float>(o), 1, std::span<float>(s));
}

void by_hand7(std::vector<float>& o, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    float sa[N], out[N];
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < H; ++h) {
            const std::size_t base = (t * H + h) * N;
            float* st = s.data() + h * N * N;
            for (std::size_t p = 0; p < N; ++p)
                sa[p] = out[p] = 0.0f;
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t p = 0; p < N; ++p)
                    sa[p] += as[base + i] * st[i * N + p];
            for (std::size_t i = 0; i < N; ++i) {
                const float k = ks[base + i], w = std::exp(lws[base + i]), b = bs[base + i], r = rs[base + i];
                for (std::size_t p = 0; p < N; ++p) {
                    st[i * N + p] = st[i * N + p] * w + vs[base + p] * k + sa[p] * b;
                    out[p] += st[i * N + p] * r;
                }
            }
            for (std::size_t p = 0; p < N; ++p)
                o[base + p] = out[p];
        }
}

BENCHMARK(run<core6>)->Name("BM_Wkv6_Core");
BENCHMARK(run<by_hand6>)->Name("BM_Wkv6_HandWritten");
BENCHMARK(run<core7>)->Name("BM_Wkv7_Core");
BENCHMARK(run<by_hand7>)->Name("BM_Wkv7_HandWritten");

bool same() { return same_pair(core6, by_hand6) && same_pair(core7, by_hand7); }
} // namespace

int main(int argc, char** argv)
{
    if (!same()) {
        std::fprintf(stderr, "rwkv benchmark: fairness check FAILED - the pair disagrees; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
