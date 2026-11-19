//
//  distill_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::distill_rows with the forward KL at temperature 2 against the same loops written by hand, over 16
//  rows of 50,304 float logits for the student and the teacher: each row's maximum and tempered sum for both, then the
//  KL and the gradient in one pass in the same order. main() runs a fairness check first and refuses to benchmark if
//  any element differs in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/distill.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

namespace ml = snicholls::ml;
constexpr std::size_t rows = 16, vocab = 50304;
constexpr float temperature = 2.0f, scale = 4.0f / rows;

std::vector<float> make(std::uint64_t seed)
{
    std::vector<float> out(rows * vocab);
    std::uint64_t x = 0x243f6a8885a308d3ULL ^ seed;   // a fixed xorshift stream: identical data for both
    for (auto& v : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53 * 20.0 - 10.0);
    }
    return out;
}

const std::vector<float> student = make(1), teacher = make(2);

struct tempered { float m, inv, log_s; };

tempered tempered_by_hand(const float* x, float inv)
{
    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < vocab; ++i)
        m = x[i] > m ? x[i] : m;
    float s = 0.0f;
    for (std::size_t i = 0; i < vocab; ++i)
        s += std::exp((x[i] - m) * inv);
    return {m, inv, std::log(s)};
}

void by_hand(float* losses, float* grads)
{
    const float inv = 1.0f / temperature;
    for (std::size_t r = 0; r < rows; ++r) {
        const float* s = student.data() + r * vocab;
        const float* t = teacher.data() + r * vocab;
        float* g = grads + r * vocab;
        const tempered ps = tempered_by_hand(s, inv), pt = tempered_by_hand(t, inv);
        float kl = 0.0f;
        for (std::size_t i = 0; i < vocab; ++i) {
            const float lp = (t[i] - pt.m) * pt.inv - pt.log_s, lq = (s[i] - ps.m) * ps.inv - ps.log_s;
            const float p = std::exp(lp), q = std::exp(lq);
            if (p != 0.0f)
                kl += p * (lp - lq);
            g[i] = scale * inv * (q - p);
        }
        losses[r] = kl;
    }
}

void core(float* losses, float* grads)
{
    ml::distill_rows<float>(snicholls::linalg::reference, ml::forward_kl{.temperature = temperature}, student, teacher, {}, {losses, rows},
                            {grads, rows * vocab}, scale);
}

void BM_DistillForwardKl_Core(benchmark::State& state)
{
    std::vector<float> l(rows), g(rows * vocab);
    for (auto _ : state) {
        core(l.data(), g.data());
        benchmark::DoNotOptimize(g.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows * vocab));
}
BENCHMARK(BM_DistillForwardKl_Core);

void BM_DistillForwardKl_HandWritten(benchmark::State& state)
{
    std::vector<float> l(rows), g(rows * vocab);
    for (auto _ : state) {
        by_hand(l.data(), g.data());
        benchmark::DoNotOptimize(g.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows * vocab));
}
BENCHMARK(BM_DistillForwardKl_HandWritten);

bool fair()
{
    std::vector<float> l1(rows), g1(rows * vocab), l2(rows), g2(rows * vocab);
    core(l1.data(), g1.data());
    by_hand(l2.data(), g2.data());
    for (std::size_t i = 0; i < g1.size(); ++i)
        if (std::bit_cast<std::uint32_t>(g1[i]) != std::bit_cast<std::uint32_t>(g2[i]))
            return false;
    for (std::size_t i = 0; i < rows; ++i)
        if (std::bit_cast<std::uint32_t>(l1[i]) != std::bit_cast<std::uint32_t>(l2[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "distill benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
