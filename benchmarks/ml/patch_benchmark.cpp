//
//  patch_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::ml::patchify_rows against the loop written by hand over every value: a batch of 16 RGB images of 224 x
//  224 in patches of 16 (ViT-B/16's input, 196 tokens of 768 values each), and a batch of 4 videos of 16 frames in
//  tubelets of 2 x 16 x 16 (V-JEPA's). main() runs a fairness check first and refuses to benchmark if any element
//  differs in a single bit (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/ml/patch.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
const auto& backend = snicholls::linalg::reference;

constexpr ml::patch_grid image{.height = 224, .width = 224, .channels = 3, .patch_height = 16, .patch_width = 16};
constexpr ml::patch_grid video{.frames = 16, .height = 224, .width = 224, .channels = 3, .tubelet = 2, .patch_height = 16, .patch_width = 16};

std::vector<float> filled(std::size_t n)
{
    std::vector<float> v(n);
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;   // a fixed xorshift stream: identical data for both
    for (auto& e : v) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        e = static_cast<float>(static_cast<double>(x >> 11) * 0x1.0p-53);
    }
    return v;
}

// Every value of every token, (dt, dy, dx, c) innermost
void by_hand(const ml::patch_grid& g, const float* v, float* out, std::size_t samples)
{
    std::size_t o = 0;
    for (std::size_t s = 0; s < samples; ++s)
        for (std::size_t gt = 0; gt < g.grid_frames(); ++gt)
            for (std::size_t gy = 0; gy < g.grid_height(); ++gy)
                for (std::size_t gx = 0; gx < g.grid_width(); ++gx)
                    for (std::size_t dt = 0; dt < g.tubelet; ++dt)
                        for (std::size_t dy = 0; dy < g.patch_height; ++dy) {
                            const float* row = v + s * g.values()
                                               + (((gt * g.tubelet + dt) * g.height + gy * g.patch_height + dy) * g.width + gx * g.patch_width) * g.channels;
                            for (std::size_t i = 0; i < g.patch_width * g.channels; ++i)
                                out[o++] = row[i];
                        }
}

template <const ml::patch_grid& G, std::size_t Samples>
void BM_Patchify_Core(benchmark::State& state)
{
    const auto v = filled(Samples * G.values());
    std::vector<float> out(v.size());
    for (auto _ : state) {
        ml::patchify_rows(backend, G, std::span<const float>(v), std::span<float>(out), Samples);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * std::int64_t(v.size() * sizeof(float)));
}

template <const ml::patch_grid& G, std::size_t Samples>
void BM_Patchify_HandWritten(benchmark::State& state)
{
    const auto v = filled(Samples * G.values());
    std::vector<float> out(v.size());
    for (auto _ : state) {
        by_hand(G, v.data(), out.data(), Samples);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(state.iterations() * std::int64_t(v.size() * sizeof(float)));
}

BENCHMARK(BM_Patchify_Core<image, 16>)->Name("BM_PatchifyImages_Core");
BENCHMARK(BM_Patchify_HandWritten<image, 16>)->Name("BM_PatchifyImages_HandWritten");
BENCHMARK(BM_Patchify_Core<video, 4>)->Name("BM_PatchifyVideos_Core");
BENCHMARK(BM_Patchify_HandWritten<video, 4>)->Name("BM_PatchifyVideos_HandWritten");

bool fair(const ml::patch_grid& g, std::size_t samples)
{
    const auto v = filled(samples * g.values());
    std::vector<float> a(v.size()), b(v.size());
    ml::patchify_rows(backend, g, std::span<const float>(v), std::span<float>(a), samples);
    by_hand(g, v.data(), b.data(), samples);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(b[i]))
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair(image, 2) || !fair(video, 1)) {
        std::fprintf(stderr, "patch benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
