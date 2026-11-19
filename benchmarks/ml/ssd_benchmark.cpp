//
//  ssd_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::ssd_forward (Mamba-2's selective scan through the recurrence driver) against the same scan written
//  out by hand, over 256 tokens of 8 heads of 64 channels sharing one group of 64 state channels: the same exp, the
//  same products in the same order. The chunked form (chunks of 64) is measured beside them. main() runs a fairness
//  check first and refuses to benchmark if the pair disagrees in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/ssd.hpp>

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
    auto c = ml::tiny_mamba2;
    c.mixer_heads = 8;
    c.mixer_head_dim = 64;
    c.mixer_state = 64;
    c.mixer_groups = 1;
    c.mixer_chunk = 64;
    return c;
}();
constexpr std::size_t n = 256, H = 8, P = 64, N = 64, S = H * N * P;

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

const std::vector<float> xs = make(n * H * P, -1, 1, 0x243f6a8885a308d3ULL), dts = make(n * H, 0.01, 0.2, 0x13198a2e03707344ULL),
                         as = make(H, -2, -0.1, 0xa4093822299f31d0ULL), bs = make(n * N, -1, 1, 0x082efa98ec4e6c89ULL),
                         cs = make(n * N, -1, 1, 0x452821e638d01377ULL);

void core(std::vector<float>& y, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    ml::ssd_forward<C>(snicholls::linalg::reference, ml::ssd_inputs<float>{xs, dts, as, bs, cs}, std::span<float>(y), 1, std::span<float>(s));
}

void chunked(std::vector<float>& y, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    ml::ssd_forward_chunked<C>(snicholls::linalg::reference, ml::ssd_inputs<float>{xs, dts, as, bs, cs}, std::span<float>(y), 1, std::span<float>(s));
}

void by_hand(std::vector<float>& y, std::vector<float>& s)
{
    std::fill(s.begin(), s.end(), 0.0f);
    float xdt[P], out[P];
    for (std::size_t t = 0; t < n; ++t)
        for (std::size_t h = 0; h < H; ++h) {
            const float dt = dts[t * H + h], alpha = std::exp(dt * as[h]);
            const float* x = xs.data() + (t * H + h) * P;
            const float* b = bs.data() + t * N;
            const float* c = cs.data() + t * N;
            float* st = s.data() + h * N * P;
            for (std::size_t p = 0; p < P; ++p) {
                xdt[p] = x[p] * dt;
                out[p] = 0.0f;
            }
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t p = 0; p < P; ++p) {
                    st[i * P + p] = st[i * P + p] * alpha + b[i] * xdt[p];
                    out[p] += st[i * P + p] * c[i];
                }
            for (std::size_t p = 0; p < P; ++p)
                y[(t * H + h) * P + p] = out[p];
        }
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

BENCHMARK(run<core>)->Name("BM_SsdScan_Core");
BENCHMARK(run<by_hand>)->Name("BM_SsdScan_HandWritten");
BENCHMARK(run<chunked>)->Name("BM_SsdChunked_Core");

bool same()
{
    std::vector<float> a(n * H * P), b(n * H * P), sa(S), sb(S);
    core(a, sa);
    by_hand(b, sb);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    for (std::size_t i = 0; i < sa.size(); ++i)
        if (std::bit_cast<std::uint32_t>(sa[i]) != std::bit_cast<std::uint32_t>(sb[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!same()) {
        std::fprintf(stderr, "ssd benchmark: fairness check FAILED - the pair disagrees; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
