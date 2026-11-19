//
//  logprobs_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::token_logprobs_rows (with entropies) and token_logprobs_rows_backward (log-probability and entropy
//  gradients) against the same loops written by hand, over 16 rows of 50,304 float logits: the maximum, the shifted
//  sum, then each element in the same order. main() runs a fairness check first and refuses to benchmark if any
//  element differs in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/logprobs.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr std::size_t rows = 16, vocab = 50304;

std::vector<float> make_logits()
{
    std::vector<float> out(rows * vocab);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 20.0 - 10.0);
    }
    return out;
}

const std::vector<float> logits = make_logits();
const std::vector<std::uint32_t> targets = [] {
    std::vector<std::uint32_t> t(rows);
    for (std::size_t i = 0; i < rows; ++i)
        t[i] = static_cast<std::uint32_t>((i * 7919) % vocab);
    return t;
}();
const std::vector<float> dlogps(rows, 0.5f), dents(rows, 0.01f);

struct stats_t { float m, log_s, inv; };

stats_t stats_by_hand(const float* x)
{
    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < vocab; ++i) {
        if (x[i] != x[i]) { m = x[i]; break; }
        if (x[i] > m) m = x[i];
    }
    float s = 0.0f;
    for (std::size_t i = 0; i < vocab; ++i)
        s += std::exp(x[i] - m);
    return {m, std::log(s), 1.0f / s};
}

float entropy_by_hand(const float* x, const stats_t& st)
{
    float h = 0.0f;
    for (std::size_t i = 0; i < vocab; ++i) {
        const float lp = (x[i] - st.m) - st.log_s;
        h -= std::exp(x[i] - st.m) * st.inv * lp;
    }
    return h;
}

void forward_by_hand(const float* in, float* logps, float* ents)
{
    for (std::size_t t = 0; t < rows; ++t) {
        const float* x = in + t * vocab;
        const stats_t st = stats_by_hand(x);
        logps[t] = (x[targets[t]] - st.m) - st.log_s;
        ents[t] = entropy_by_hand(x, st);
    }
}

void backward_by_hand(float* g)
{
    for (std::size_t t = 0; t < rows; ++t) {
        float* x = g + t * vocab;
        const stats_t st = stats_by_hand(x);
        const float dh = dents[t], h = entropy_by_hand(x, st), dl = dlogps[t];
        for (std::size_t i = 0; i < vocab; ++i) {
            const float lp = (x[i] - st.m) - st.log_s;
            const float p = std::exp(x[i] - st.m) * st.inv;
            const float gi = dl * ((i == targets[t] ? 1.0f : 0.0f) - p);
            x[i] = gi - dh * p * (lp + h);
        }
    }
}

void BM_TokenLogprobs_Core(benchmark::State& state)
{
    std::vector<float> lp(rows), en(rows);
    for (auto _ : state) {
        ml::token_logprobs_rows<float>(snicholls::linalg::reference, logits, targets, lp, en);
        benchmark::DoNotOptimize(lp.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows * vocab));
}
BENCHMARK(BM_TokenLogprobs_Core);

void BM_TokenLogprobs_HandWritten(benchmark::State& state)
{
    std::vector<float> lp(rows), en(rows);
    for (auto _ : state) {
        forward_by_hand(logits.data(), lp.data(), en.data());
        benchmark::DoNotOptimize(lp.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows * vocab));
}
BENCHMARK(BM_TokenLogprobs_HandWritten);

void BM_TokenLogprobsBackward_Core(benchmark::State& state)
{
    std::vector<float> g(logits.size());
    for (auto _ : state) {
        state.PauseTiming();
        g = logits;
        state.ResumeTiming();
        ml::token_logprobs_rows_backward<float>(snicholls::linalg::reference, g, targets, dlogps, dents);
        benchmark::DoNotOptimize(g.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows * vocab));
}
BENCHMARK(BM_TokenLogprobsBackward_Core);

void BM_TokenLogprobsBackward_HandWritten(benchmark::State& state)
{
    std::vector<float> g(logits.size());
    for (auto _ : state) {
        state.PauseTiming();
        g = logits;
        state.ResumeTiming();
        backward_by_hand(g.data());
        benchmark::DoNotOptimize(g.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows * vocab));
}
BENCHMARK(BM_TokenLogprobsBackward_HandWritten);

bool same(const std::vector<float>& a, const std::vector<float>& b)
{
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return a.size() == b.size();
}

bool fair()
{
    std::vector<float> lp1(rows), en1(rows), lp2(rows), en2(rows), g1 = logits, g2 = logits;
    ml::token_logprobs_rows<float>(snicholls::linalg::reference, logits, targets, lp1, en1);
    forward_by_hand(logits.data(), lp2.data(), en2.data());
    ml::token_logprobs_rows_backward<float>(snicholls::linalg::reference, g1, targets, dlogps, dents);
    backward_by_hand(g2.data());
    return same(lp1, lp2) && same(en1, en2) && same(g1, g2);
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "logprobs benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
