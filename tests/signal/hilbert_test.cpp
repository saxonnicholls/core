//
//  hilbert_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/hilbert.hpp>: scipy.signal.hilbert for even and odd lengths, a cosine's transform
//  is the sine, the envelope of an amplitude-modulated tone, the backward pass against central differences, and
//  compile-time use.
//

#include <snicholls/core/signal/hilbert.hpp>

#include "../support/signal_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;
using cd = s::complex<double>;

// The Hilbert transform of cos is sin, at compile time (16 samples, two whole periods).
constexpr bool cosine_at_compile_time()
{
    std::array<double, 16> x{}, h{};
    for (std::size_t i = 0; i < 16; ++i)
        x[i] = m::cos(2 * m::pi * 2.0 * static_cast<double>(i) / 16.0);
    std::array<cd, 32> work{};
    s::hilbert_transform(s::static_fft_plan_v<double, 16>.plan(), std::span<const double>(x), std::span<double>(h), std::span<cd>(work));
    for (std::size_t i = 0; i < 16; ++i)
        if (m::abs(h[i] - m::sin(2 * m::pi * 2.0 * static_cast<double>(i) / 16.0)) > 1e-14)
            return false;
    return true;
}
static_assert(cosine_at_compile_time());

struct planned
{
    std::vector<cd> storage, work;
    s::fft_plan<double> plan;
    explicit planned(std::size_t n) : storage(s::fft_plan<double>::storage_size(n)), work(n + s::fft_plan<double>::work_size(n)), plan(n, std::span<cd>(storage)) {}
};

TEST(Hilbert, AnalyticSignalIsScipysForEvenAndOddLengths)
{
    for (const std::size_t n : {std::size_t{10}, std::size_t{11}}) {
        const auto& x = snicholls::test::signal_record("hilbert_input_" + std::to_string(n));
        const auto ref = snicholls::test::signal_complex("hilbert_" + std::to_string(n));
        planned p(n);
        std::vector<cd> z(n);
        s::analytic_signal(p.plan, std::span<const double>(x), std::span<cd>(z), std::span<cd>(p.work));
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(z[i].re, ref[i].re, 1e-14);
            EXPECT_NEAR(z[i].im, ref[i].im, 1e-14);
        }
    }
}

TEST(Hilbert, EnvelopeOfAModulatedToneIsItsAmplitude)
{
    const std::size_t n = 1000;
    planned p(n);
    std::vector<double> x(n), env(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        x[i] = (1.0 + 0.5 * std::cos(2 * m::pi * 3 * t)) * std::cos(2 * m::pi * 100 * t);
    }
    s::envelope(p.plan, std::span<const double>(x), std::span<double>(env), std::span<cd>(p.work));
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_NEAR(env[i], 1.0 + 0.5 * std::cos(2 * m::pi * 3 * static_cast<double>(i) / static_cast<double>(n)), 1e-12);
}

TEST(HilbertBackward, MatchesCentralDifferences)
{
    for (const std::size_t n : {std::size_t{8}, std::size_t{9}}) {
        planned p(n);
        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i)
            x[i] = std::sin(1.7 * static_cast<double>(i)) + 0.2;
        const auto loss = [&](const std::vector<double>& v, std::vector<cd>* grad) {
            std::vector<cd> z(n), w(p.work.size());
            s::analytic_signal(p.plan, std::span<const double>(v), std::span<cd>(z), std::span<cd>(w));
            double l = 0;
            for (std::size_t k = 0; k < n; ++k) {
                const double a = 0.3 + 0.05 * static_cast<double>(k);
                l += a * s::norm(z[k]) + 0.1 * z[k].im;
                if (grad)
                    (*grad)[k] = z[k] * (2 * a) + cd{0, 0.1};
            }
            return l;
        };
        std::vector<cd> g(n);
        loss(x, &g);
        std::vector<double> gx(n);
        s::analytic_signal_backward(p.plan, std::span<cd>(g), std::span<double>(gx), std::span<cd>(p.work));
        const double h = 1e-6;
        for (std::size_t j = 0; j < n; ++j) {
            auto up = x, down = x;
            up[j] += h;
            down[j] -= h;
            EXPECT_NEAR(gx[j], (loss(up, nullptr) - loss(down, nullptr)) / (2 * h), 1e-7) << n << " " << j;
        }
    }
}

} // namespace
