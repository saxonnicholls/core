//
//  randomized_svd_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::linalg::truncated_svd (the leading k = 10 triplets, 2 power iterations, oversampling 10) against the
//  exact svd of the same matrix, both through the reference backend, for 500 x 100 and 2000 x 200 matrices with a
//  decaying spectrum (sigma_i = 0.8^i): what the randomised method saves when only the leading factors are wanted. The
//  two are different algorithms, so the fairness check is on what they share: the leading k singular values agree to
//  1e-9 relative, or main() refuses to benchmark.
//

#include <snicholls/core/linalg/randomized_svd.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace la = snicholls::linalg;
using right = la::layout_right;
constexpr std::size_t k = 10;

std::vector<double> orthonormal(std::size_t m, std::size_t r, std::uint64_t seed)
{
    std::vector<double> g(m * r), q(m * r), tau(r);
    la::gaussian_sketch<>{seed}(std::span<double>(g), m, r);
    (void)la::qr_factor(la::reference, la::matrix_view<double, right>(g.data(), {m, r}), std::span<double>(tau));
    (void)la::qr_q(la::reference, la::matrix_view<const double, right>(g.data(), {m, r}), std::span<const double>(tau),
                   la::matrix_view<double, right>(q.data(), {m, r}));
    return q;
}

struct problem {
    std::size_t m, n;
    std::vector<double> a;
    problem(std::size_t m_, std::size_t n_) : m(m_), n(n_), a(m_ * n_, 0.0)
    {
        const auto u = orthonormal(m, n, 1), v = orthonormal(n, n, 2);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t p = 0; p < n; ++p) {
                const double us = u[i * n + p] * std::pow(0.8, double(p));
                for (std::size_t j = 0; j < n; ++j)
                    a[i * n + j] += us * v[j * n + p];
            }
    }
};

std::vector<double> truncated(const problem& p)
{
    std::vector<double> s(k), u(p.m * k), v(p.n * k), work(la::truncated_svd_work_size(p.m, p.n, k));
    (void)la::truncated_svd(la::reference, la::matrix_view<const double, right>(p.a.data(), {p.m, p.n}), std::span<double>(s),
                            la::matrix_view<double, right>(u.data(), {p.m, k}), la::matrix_view<double, right>(v.data(), {p.n, k}),
                            std::span<double>(work));
    return s;
}

std::vector<double> exact(const problem& p)
{
    std::vector<double> u = p.a, s(p.n), v(p.n * p.n);
    (void)la::svd(la::reference, la::matrix_view<double, right>(u.data(), {p.m, p.n}), std::span<double>(s),
                  la::matrix_view<double, right>(v.data(), {p.n, p.n}));
    return s;
}

const problem& shape(std::int64_t m)
{
    static const problem small(500, 100), large(2000, 200);
    return m == 500 ? small : large;
}

template <std::vector<double> (*Run)(const problem&)>
void BM_Svd(benchmark::State& state)
{
    const problem& p = shape(state.range(0));
    for (auto _ : state) {
        auto s = Run(p);
        benchmark::DoNotOptimize(s.data());
    }
}
BENCHMARK(BM_Svd<truncated>)->Name("BM_TruncatedSvd_K10")->Arg(500)->Arg(2000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Svd<exact>)->Name("BM_ExactSvd")->Arg(500)->Arg(2000)->Unit(benchmark::kMillisecond);

bool fair()
{
    for (const std::int64_t m : {500, 2000}) {
        const auto t = truncated(shape(m)), e = exact(shape(m));
        for (std::size_t i = 0; i < k; ++i)
            if (std::abs(t[i] - e[i]) > 1e-9 * e[0])
                return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "randomized_svd benchmark: fairness check FAILED - the leading singular values disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
