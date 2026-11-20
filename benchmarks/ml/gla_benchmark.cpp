//
//  gla_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::gla_forward (gated linear attention through the recurrence driver) against the same recurrence
//  written out by hand, over 256 tokens of 8 heads of 64 by 64: the same exp, the same products in the same order.
//  The chunked form (chunks of 64) is measured beside them. main() runs a fairness check first and refuses to
//  benchmark if the pair disagrees in a single bit (CLAUDE.md "Zero overhead").
//
#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/gla.hpp>

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
    c.mixer = ml::mixer_kind::gla;
    c.mixer_heads = 8;
    c.mixer_head_dim = 64;
    c.mixer_state = 64;
    c.mixer_groups = 8;
    c.mixer_chunk = 64;
    c.hybrid_period = c.hybrid_attention_offset = 0;
    c.conv_width = 0;
    return c;
}();
constexpr std::size_t n = 256, H = 8, P = 64, N = 64, G = 8, S = H * N * P;

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
    std::vector<float> y(n * H * P), s(S);
    for (auto _ : state) {
        F(y, s);
        benchmark::DoNotOptimize(y.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

bool same_pair(void (*a)(std::vector<float>&, std::vector<float>&), void (*b)(std::vector<float>&, std::vector<float>&))
{
    std::vector<float> ya(n * H * P), yb(n * H * P), sa(S), sb(S);
    a(ya, sa);
    b(yb, sb);
    return bits_equal(ya, yb) && bits_equal(sa, sb);
}

const std::vector<float> qs = make(n * G * N, -1, 1, 1), ks = make(n * G * N, -1, 1, 2), vs = make(n * H * P, -1, 1, 3),
                         gks = make(n * H * N, -0.5, -0.001, 4);

void core(std::vector<float>& o, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    ml::gla_forward<C>(snicholls::linalg::reference, ml::gla_inputs<float>{qs, ks, vs, gks}, std::span<float>(o), 1, std::span<float>(s));
}

void chunked(std::vector<float>& o, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    std::vector<float> work(ml::gla_chunk_work(C));
    ml::gla_forward_chunked<C>(snicholls::linalg::reference, ml::gla_inputs<float>{qs, ks, vs, gks}, std::span<float>(o), 1, std::span<float>(s), {},
                               std::span<float>(work));
}

void by_hand(std::vector<float>& o, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    constexpr float scale = ml::mixer_scale_v<C, float>;
    float out[P];
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < H; ++h) {
            const float* q = qs.data() + (t * G + h) * N;
            const float* k = ks.data() + (t * G + h) * N;
            const float* v = vs.data() + (t * H + h) * P;
            float* st = s.data() + h * N * P;
            for (std::size_t p = 0; p < P; ++p)
                out[p] = 0.0f;
            for (std::size_t i = 0; i < N; ++i) {
                const float qn = q[i] * scale, gate = std::exp(gks[(t * H + h) * N + i]);
                for (std::size_t p = 0; p < P; ++p) {
                    st[i * P + p] = st[i * P + p] * gate + v[p] * k[i];
                    out[p] += st[i * P + p] * qn;
                }
            }
            for (std::size_t p = 0; p < P; ++p)
                o[(t * H + h) * P + p] = out[p];
        }
}

BENCHMARK(run<core>)->Name("BM_GlaRecurrent_Core");
BENCHMARK(run<by_hand>)->Name("BM_GlaRecurrent_HandWritten");
BENCHMARK(run<chunked>)->Name("BM_GlaChunked_Core");

bool same() { return same_pair(core, by_hand); }
} // namespace

int main(int argc, char** argv)
{
    if (!same()) {
        std::fprintf(stderr, "gla benchmark: fairness check FAILED - the pair disagrees; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
