//
//  ssd_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/ssd.hpp>: the scan against a transcription of llama.cpp's GGML_OP_SSM_SCAN (commit
//  39039aa, Mamba-2's scalar-A path); the chunked form against the scan within the stated rounding (float and double,
//  a last chunk shorter than the others, two sequences, initial states); decoding token by token to the bit; the
//  backward pass against central differences in double from either forward pass's checkpoints; sizes and a scan at
//  compile time.
//

#include <snicholls/core/ml/ssd.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include "../support/mixer_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using snicholls::core_testing::central_difference;
using snicholls::core_testing::max_relative_difference;
using snicholls::core_testing::uniform;
using snicholls::core_testing::weighted;
constexpr auto& ref = snicholls::linalg::reference;

// 4 heads of 3 channels in 2 groups of 2 state channels, chunks of 3
constexpr ml::transformer_config C = [] {
    auto c = ml::tiny_mamba2;
    c.mixer_heads = 4;
    c.mixer_head_dim = 3;
    c.mixer_state = 2;
    c.mixer_groups = 2;
    c.mixer_chunk = 3;
    return c;
}();
constexpr std::size_t H = 4, P = 3, N = 2, G = 2, S = H * N * P;
static_assert(ml::ssd_state_size(C) == S && ml::ssd_state_size(ml::tiny_mamba2) == 2 * 4 * 4);

template <typename T>
struct problem
{
    std::size_t rows;
    std::vector<T> x, dt, a, b, c;
    problem(std::size_t r, std::uint64_t seed)
        : rows(r), x(uniform<T>(r * H * P, seed)), dt(uniform<T>(r * H, seed + 1, 0.05, 0.8)), a(uniform<T>(H, seed + 2, -1.5, -0.2)),
          b(uniform<T>(r * G * N, seed + 3)), c(uniform<T>(r * G * N, seed + 4))
    {
    }
    ml::ssd_inputs<T> in() const { return {x, dt, a, b, c}; }
};

constexpr double compile_time_scan()
{
    constexpr ml::transformer_config c1 = [] {
        auto c = ml::tiny_mamba2;
        c.mixer_heads = 1;
        c.mixer_head_dim = 1;
        c.mixer_state = 1;
        c.mixer_chunk = 2;
        return c;
    }();
    const std::array<double, 2> x{1, 2}, dt{1, 1}, b{1, 1}, cc{1, 1};
    const std::array<double, 1> a{0};   // no decay: a running sum
    std::array<double, 2> y{};
    std::array<double, 1> s{};
    ml::ssd_forward<c1>(ref, ml::ssd_inputs<double>{x, dt, a, b, cc}, std::span<double>(y), 1, std::span<double>(s));
    return y[1] + s[0];
}
static_assert(compile_time_scan() == 6.0);

// ggml_compute_forward_ssm_scan_f32, the scalar-A branch, scalar tail: s {d_state, dim, n_head}, x {dim, n_head},
// B and C {d_state, n_group}; the softplus of dt is left out because ssd's dt is already positive
std::vector<float> ggml_ssm_scan(std::vector<float>& s, const problem<float>& pr)
{
    std::vector<float> y(pr.rows * H * P);
    for (std::size_t i2 = 0; i2 < pr.rows; ++i2)
        for (std::size_t h = 0; h < H; ++h) {
            const float dt_soft_plus = pr.dt[i2 * H + h];
            const float dA = std::exp(dt_soft_plus * pr.a[h]);
            const std::size_t g = h / (H / G);
            for (std::size_t i1 = 0; i1 < P; ++i1) {
                const std::size_t ii = i1 + h * P;
                const float x_dt = pr.x[i2 * H * P + ii] * dt_soft_plus;
                float sumf = 0.0f;
                for (std::size_t i0 = 0; i0 < N; ++i0) {
                    const std::size_t i = i0 + ii * N, ig = i0 + g * N;
                    const float state = (s[i] * dA) + (pr.b[i2 * G * N + ig] * x_dt);
                    sumf += state * pr.c[i2 * G * N + ig];
                    s[i] = state;
                }
                y[i2 * H * P + ii] = sumf;
            }
        }
    return y;
}

TEST(MlSsd, TheScanIsLlamaCppsSsmScan)
{
    const problem<float> pr(7, 1);
    std::vector<float> y(pr.x.size()), s(S, 0.0f), gs(S, 0.0f);
    ml::ssd_forward<C>(ref, pr.in(), std::span<float>(y), 1, std::span<float>(s));
    const auto expected = ggml_ssm_scan(gs, pr);
    EXPECT_LT(max_relative_difference(y, expected), 1e-6);
    for (std::size_t h = 0; h < H; ++h)   // ggml keeps the state [p][n], ssd [n][p]
        for (std::size_t n = 0; n < N; ++n)
            for (std::size_t p = 0; p < P; ++p)
                EXPECT_NEAR(s[(h * N + n) * P + p], gs[(h * P + p) * N + n], 1e-6f);
}

template <typename T>
void chunked_is_the_scan(double bound)
{
    constexpr std::size_t n = 8, seqs = 2;   // chunks of 3, 3, 2
    const problem<T> pr(n * seqs, 10);
    const auto init = uniform<T>(seqs * S, 20);
    const std::size_t m = ml::recurrence_checkpoints(n, C.mixer_chunk);
    std::vector<T> y1(pr.x.size()), y2(pr.x.size()), s1 = init, s2 = init, c1(seqs * m * S), c2(seqs * m * S);
    ml::ssd_forward<C>(ref, pr.in(), std::span<T>(y1), seqs, std::span<T>(s1), std::span<T>(c1));
    ml::ssd_forward_chunked<C>(ref, pr.in(), std::span<T>(y2), seqs, std::span<T>(s2), std::span<T>(c2));
    EXPECT_LT(max_relative_difference(y1, y2), bound);
    EXPECT_LT(max_relative_difference(s1, s2), bound);
    EXPECT_LT(max_relative_difference(c1, c2), bound);
}

TEST(MlSsd, TheChunkedFormIsTheScanWithinRounding)
{
    chunked_is_the_scan<double>(1e-13);
    chunked_is_the_scan<float>(2e-6);
}

TEST(MlSsd, DecodingIsTheScanToTheBit)
{
    constexpr std::size_t n = 6;
    const problem<float> pr(n, 30);
    std::vector<float> y(pr.x.size()), s(S, 0.0f), step(S, 0.0f), yt(H * P);
    ml::ssd_forward<C>(ref, pr.in(), std::span<float>(y), 1, std::span<float>(s));
    for (std::size_t t = 0; t < n; ++t) {
        const ml::ssd_inputs<float> row{std::span<const float>(pr.x).subspan(t * H * P, H * P), std::span<const float>(pr.dt).subspan(t * H, H),
                                        pr.a, std::span<const float>(pr.b).subspan(t * G * N, G * N),
                                        std::span<const float>(pr.c).subspan(t * G * N, G * N)};
        ml::ssd_step<C>(row, std::span<float>(yt), std::span<float>(step));
        for (std::size_t i = 0; i < H * P; ++i)
            EXPECT_EQ(yt[i], y[t * H * P + i]) << t;
    }
    EXPECT_EQ(step, s);
}

void backward_matches_central_differences(bool chunked)
{
    constexpr std::size_t n = 5, seqs = 2;
    problem<double> pr(n * seqs, 40);
    auto init = uniform<double>(seqs * S, 41);
    const auto wy = uniform<double>(pr.x.size(), 42), ws = uniform<double>(seqs * S, 43);
    const auto run = [&](std::vector<double>& y, std::vector<double>& s, std::vector<double>& cps) {
        if (chunked)
            ml::ssd_forward_chunked<C>(ref, pr.in(), std::span<double>(y), seqs, std::span<double>(s), std::span<double>(cps));
        else
            ml::ssd_forward<C>(ref, pr.in(), std::span<double>(y), seqs, std::span<double>(s), std::span<double>(cps));
    };
    const std::size_t m = ml::recurrence_checkpoints(n, C.mixer_chunk);
    const auto loss = [&] {
        std::vector<double> y(pr.x.size()), s = init, cps;
        run(y, s, cps);
        return weighted(y, wy) + weighted(s, ws);
    };
    std::vector<double> y(pr.x.size()), s = init, cps(seqs * m * S);
    run(y, s, cps);
    std::vector<double> dx(pr.x.size()), ddt(pr.dt.size()), da(H), db(pr.b.size()), dc(pr.c.size()), ds = ws,
        work(ml::ssd_backward_work<C, double>());
    ml::ssd_backward<C>(ref, pr.in(), std::span<const double>(wy), ml::ssd_gradients<double>{dx, ddt, da, db, dc}, seqs,
                        std::span<const double>(cps), std::span<double>(ds), std::span<double>(work));
    const auto check = [&](std::vector<double>& v, const std::vector<double>& g, const char* what) {
        for (std::size_t i = 0; i < v.size(); ++i)
            EXPECT_NEAR(g[i], central_difference(v, i, loss), 1e-7) << what << " " << i;
    };
    check(pr.x, dx, "x");
    check(pr.dt, ddt, "dt");
    check(pr.a, da, "A");
    check(pr.b, db, "B");
    check(pr.c, dc, "C");
    check(init, ds, "initial state");
}

TEST(MlSsd, TheBackwardPassMatchesCentralDifferences)
{
    backward_matches_central_differences(false);
}

TEST(MlSsd, TheBackwardPassFromTheChunkedFormsCheckpointsMatchesCentralDifferences)
{
    backward_matches_central_differences(true);
}

} // namespace
