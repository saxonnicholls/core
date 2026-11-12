//
//  batched_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/linalg/batched.hpp>: a thousand small kriging-style systems (a covariance
//  block bordered by a row of ones, symmetric and indefinite) solved in one call by LDLT, one system with a
//  repeated sample point reported singular and solved again by the caller's fallback; the same batch on
//  the caller's threads with the same bits; and the LU and Cholesky batches. Exits non-zero if a result is
//  not the one documented.
//

#include <snicholls/core/linalg/reference.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

namespace la = snicholls::linalg;

namespace {

constexpr std::size_t blocks = 1000, k = 6, m = k + 1;   // k samples per block, plus the Lagrange row

// Block b's system: C_ij = exp(-|x_i - x_j| / 3) for its samples, bordered by ones; right-hand side c_i0, 1.
void build(std::size_t b, std::span<double> a, std::span<double> rhs)
{
    double x[k];
    for (std::size_t i = 0; i < k; ++i)
        x[i] = static_cast<double>(i) + 0.37 * std::sin(static_cast<double>(b * k + i));
    if (b == 417)
        x[3] = x[2];   // a repeated sample point: two equal rows
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j)
            a[i * m + j] = i < k && j < k ? std::exp(-std::abs(x[i] - x[j]) / 3.0) : (i == k && j == k ? 0.0 : 1.0);
        rhs[i] = i < k ? std::exp(-std::abs(x[i] - 2.5) / 3.0) : 1.0;
    }
}

bool same_bits(const std::vector<double>& x, const std::vector<double>& y)
{
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0;
}

} // namespace

int main()
{
    std::printf("snicholls-core linalg batched demo\n");
    std::vector<double> a(blocks * m * m), rhs(blocks * m);
    for (std::size_t b = 0; b < blocks; ++b)
        build(b, std::span<double>(a).subspan(b * m * m, m * m), std::span<double>(rhs).subspan(b * m, m));
    const std::vector<double> a0 = a, rhs0 = rhs;

    //! [batched]
    // One call factorises and solves every block; each block's outcome is its own status.
    std::vector<la::status> status(blocks);
    const la::batch_view<double, la::layout_right> systems(a.data(), {blocks, m, m});
    const la::md_span<double, 2, la::layout_right> weights(rhs.data(), {blocks, m});
    const auto failures = la::ldlt_solve_batched(la::reference, systems, weights, std::span<la::status>(status));

    // The caller falls back for the blocks that failed, and only for those: here, drop the repeated sample.
    for (std::size_t b = 0; b < blocks; ++b)
        if (status[b] != la::status::ok)
            std::printf("  block %zu: %s, falls back\n", b, status[b] == la::status::singular ? "singular" : "failed");
    //! [batched]
    bool ok = failures && *failures == 1 && status[417] == la::status::singular && status[0] == la::status::ok;
    double sum = 0;   // ordinary kriging weights sum to one
    for (std::size_t i = 0; i < k; ++i)
        sum += rhs[i];
    std::printf("  %zu blocks of order %zu: %zu failed; block 0's weights sum to %.15f\n", blocks, m, failures ? *failures : 0, sum);
    ok = ok && std::abs(sum - 1.0) < 1e-12;

    //! [executor]
    // The caller's parallelism: any callable that runs f(begin, end) over [0, n). Core starts no threads.
    auto on_four_threads = [](std::size_t n, auto&& f) {
        std::vector<std::thread> pool;
        for (std::size_t t = 0; t < 4; ++t)
            pool.emplace_back([&f, n, t] { f(n * t / 4, n * (t + 1) / 4); });
        for (auto& thread : pool)
            thread.join();
    };
    std::vector<double> a2 = a0, rhs2 = rhs0;
    std::vector<la::status> status2(blocks);
    const auto failures2 = la::ldlt_solve_batched(la::reference, la::batch_view<double, la::layout_right>(a2.data(), {blocks, m, m}),
                                                  la::md_span<double, 2, la::layout_right>(rhs2.data(), {blocks, m}),
                                                  std::span<la::status>(status2), on_four_threads);
    //! [executor]
    std::printf("  on four threads: %zu failed, solutions %s\n", failures2 ? *failures2 : 0, same_bits(rhs, rhs2) ? "bit-identical" : "DIFFERENT");
    ok = ok && failures2 && *failures2 == 1 && same_bits(rhs, rhs2) && status == status2;

    // LU (a nonsymmetric solver, pivots one row of m per system) and Cholesky (which refuses the indefinite systems).
    std::vector<double> a3 = a0, rhs3 = rhs0, a4 = a0, rhs4 = rhs0;
    std::vector<la::pivot_index> pivots(blocks * m);
    const auto failures3 = la::lu_solve_batched(la::reference, la::batch_view<double, la::layout_right>(a3.data(), {blocks, m, m}), std::span(pivots),
                                                la::md_span<double, 2, la::layout_right>(rhs3.data(), {blocks, m}), std::span<la::status>(status2));
    const auto failures4 = la::cholesky_solve_batched(la::reference, la::batch_view<double, la::layout_right>(a4.data(), {blocks, m, m}),
                                                      la::md_span<double, 2, la::layout_right>(rhs4.data(), {blocks, m}), std::span<la::status>(status2));
    double largest = 0;
    for (std::size_t i = 0; i < rhs.size(); ++i)
        if (status[i / m] == la::status::ok)
            largest = std::max(largest, std::abs(rhs[i] - rhs3[i]));
    std::printf("  LU: %zu failed, largest difference from LDLT %.1e; Cholesky: %zu of %zu not positive definite\n", failures3 ? *failures3 : 0,
                largest, failures4 ? *failures4 : 0, blocks);
    ok = ok && failures3 && *failures3 == 1 && largest < 1e-12 && failures4 && *failures4 == blocks &&
         la::count_failures(status2) == blocks && status2[0] == la::status::not_positive_definite;

    // for_each_system is the loop a backend's batched member is built from.
    std::size_t visited = 0;
    la::inline_executor here;
    la::for_each_system(here, 10, [&](std::size_t) { ++visited; });
    ok = ok && visited == 10;

    std::printf(ok ? "linalg batched demo: ok\n" : "linalg batched demo: FAILED\n");
    return ok ? 0 : 1;
}
