//
//  convolve_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/convolve.hpp>: convolution and correlation in every mode against scipy.signal, the
//  FFT forms against the direct one (including a prime transform length), overlap-add and overlap-save against the
//  direct convolution for several block sizes, empty inputs, and compile-time use.
//

#include <snicholls/core/signal/convolve.hpp>

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

static_assert(s::convolve_size(9, 4, s::convolve_mode::full) == 12 && s::convolve_size(9, 4, s::convolve_mode::same) == 9);
static_assert(s::convolve_size(9, 4, s::convolve_mode::valid) == 6 && s::convolve_size(4, 9, s::convolve_mode::valid) == 6);
static_assert(s::convolve_size(0, 4, s::convolve_mode::full) == 0);

// Direct and FFT convolution in constant evaluation.
constexpr bool convolution_at_compile_time()
{
    constexpr std::array<double, 4> a{1, 2, 3, 4};
    constexpr std::array<double, 3> b{1, 0, -1};
    std::array<double, 6> direct{}, fast{};
    s::convolve<double>(a, b, direct);
    const auto plan = s::static_fft_plan_v<double, 8>.plan();
    std::array<cd, 16> work{};
    s::fft_convolve(plan, std::span<const double>(a), std::span<const double>(b), std::span<double>(fast), std::span<cd>(work));
    constexpr std::array<double, 6> expected{1, 2, 2, 2, -3, -4};
    for (std::size_t i = 0; i < 6; ++i)
        if (direct[i] != expected[i] || m::abs(fast[i] - expected[i]) > 1e-14)
            return false;
    std::array<double, 4> corr{};
    s::correlate<double>(a, b, corr, s::convolve_mode::same);   // c_k = a_(k+j) b_j around the centre
    return corr[0] == -2 && corr[1] == -2 && corr[3] == 3;
}
static_assert(convolution_at_compile_time());

std::vector<double> random(std::size_t n, unsigned seed)
{
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = std::sin(0.77 * static_cast<double>(i * seed) + seed);
    return v;
}

struct planned
{
    std::vector<cd> storage, work;
    s::fft_plan<double> plan;
    explicit planned(std::size_t n) : storage(s::fft_plan<double>::storage_size(n)), work(2 * n + s::fft_plan<double>::work_size(n)), plan(n, std::span<cd>(storage)) {}
};

TEST(Convolve, EveryModeIsScipysDirectAndThroughTheFft)
{
    const auto& a = snicholls::test::signal_record("conv_a");
    const auto& b = snicholls::test::signal_record("conv_b");
    planned p(12), prime(13);
    for (const auto& [name, mode] : std::array<std::pair<const char*, s::convolve_mode>, 3>{
             {{"full", s::convolve_mode::full}, {"same", s::convolve_mode::same}, {"valid", s::convolve_mode::valid}}}) {
        const auto& conv = snicholls::test::signal_record(std::string("convolve_") + name);
        const auto& corr = snicholls::test::signal_record(std::string("correlate_") + name);
        const auto& conv_ba = snicholls::test::signal_record(std::string("convolve_ba_") + name);
        std::vector<double> d(conv.size()), f(conv.size()), g(conv.size()), c(corr.size()), cf(corr.size()), ba(conv_ba.size());
        s::convolve<double>(a, b, d, mode);
        s::fft_convolve(p.plan, std::span<const double>(a), std::span<const double>(b), std::span<double>(f), std::span<cd>(p.work), mode);
        s::fft_convolve(prime.plan, std::span<const double>(a), std::span<const double>(b), std::span<double>(g), std::span<cd>(prime.work), mode);
        s::correlate<double>(a, b, c, mode);
        s::fft_correlate(p.plan, std::span<const double>(a), std::span<const double>(b), std::span<double>(cf), std::span<cd>(p.work), mode);
        s::convolve<double>(b, a, ba, mode);
        for (std::size_t i = 0; i < conv.size(); ++i) {
            EXPECT_NEAR(d[i], conv[i], 1e-14) << name;
            EXPECT_NEAR(f[i], conv[i], 1e-14) << name;
            EXPECT_NEAR(g[i], conv[i], 1e-14) << name;
        }
        for (std::size_t i = 0; i < corr.size(); ++i) {
            EXPECT_NEAR(c[i], corr[i], 1e-14) << name;
            EXPECT_NEAR(cf[i], corr[i], 1e-14) << name;
        }
        for (std::size_t i = 0; i < conv_ba.size(); ++i)
            EXPECT_NEAR(ba[i], conv_ba[i], 1e-14) << name;   // same takes the first argument's length; valid commutes
    }
}

TEST(Convolve, FftFormIsTheDirectOneForManyLengths)
{
    for (const std::size_t na : {std::size_t{1}, std::size_t{17}, std::size_t{100}})
        for (const std::size_t nb : {std::size_t{1}, std::size_t{5}, std::size_t{64}}) {
            const auto a = random(na, 3), b = random(nb, 5);
            planned p(s::next_power_of_two(na + nb - 1));
            std::vector<double> d(na + nb - 1), f(na + nb - 1);
            s::convolve<double>(a, b, d);
            s::fft_convolve(p.plan, std::span<const double>(a), std::span<const double>(b), std::span<double>(f), std::span<cd>(p.work));
            for (std::size_t i = 0; i < d.size(); ++i)
                EXPECT_NEAR(f[i], d[i], 1e-13) << na << " " << nb;
        }
    std::vector<double> none, out;
    planned p(4);
    s::convolve<double>(none, random(3, 1), out);
    s::fft_convolve(p.plan, std::span<const double>(none), std::span<const double>(random(3, 1)), std::span<double>(out), std::span<cd>(p.work));
    EXPECT_TRUE(out.empty());
}

TEST(Convolve, OverlapAddAndOverlapSaveAreTheDirectConvolution)
{
    const auto x = random(1000, 7);
    for (const std::size_t taps : {std::size_t{1}, std::size_t{31}, std::size_t{64}})
        for (const std::size_t n : {std::size_t{64}, std::size_t{128}, std::size_t{100}}) {
            if (n < taps)
                continue;
            const auto h = random(taps, 11);
            planned p(n);
            std::vector<double> full(x.size() + taps - 1), oa(full.size()), os(x.size());
            s::convolve<double>(x, h, full);
            s::overlap_add(p.plan, std::span<const double>(h), std::span<const double>(x), std::span<double>(oa), std::span<cd>(p.work));
            s::overlap_save(p.plan, std::span<const double>(h), std::span<const double>(x), std::span<double>(os), std::span<cd>(p.work));
            for (std::size_t i = 0; i < full.size(); ++i)
                ASSERT_NEAR(oa[i], full[i], 1e-12) << taps << " " << n << " " << i;
            for (std::size_t i = 0; i < x.size(); ++i)
                ASSERT_NEAR(os[i], full[i], 1e-12) << taps << " " << n << " " << i;   // the causal output is the full one's start
        }
}

} // namespace
