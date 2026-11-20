//
//  dct_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/dct.hpp>: types II, III and IV of the DCT and DST against scipy.fft and against
//  their definitions in long double for every length to 40, unnormalised and orthonormal; the inverses; a truncated
//  DCT-II; the DCT-II's backward pass against central differences; compile-time use.
//

#include <snicholls/core/signal/dct.hpp>

#include "../support/signal_fixture.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;
using cd = s::complex<double>;

// A DCT-II and its inverse in constant evaluation.
constexpr bool dct_at_compile_time()
{
    const std::array<double, 6> x{1, 2, -1, 0.5, 3, -2};
    const auto plan = s::static_fft_plan_v<double, 6>.plan();
    std::array<cd, 12> work{};
    std::array<double, 6> y{}, back{};
    s::dct2(plan, std::span<const double>(x), std::span<double>(y), std::span<cd>(work), s::dct_norm::ortho);
    s::dct3(plan, std::span<const double>(y), std::span<double>(back), std::span<cd>(work), s::dct_norm::ortho);
    for (std::size_t i = 0; i < 6; ++i)
        if (m::abs(back[i] - x[i]) > 1e-14)
            return false;
    return m::near(y[0], 3.5 / m::sqrt(6.0), 1e-14);   // the orthonormal DC term: sum / sqrt(n)
}
static_assert(dct_at_compile_time());

struct planned
{
    std::vector<cd> storage, work;
    s::fft_plan<double> plan;
    explicit planned(std::size_t n) : storage(s::fft_plan<double>::storage_size(n)), work(n + s::fft_plan<double>::work_size(n)), plan(n, std::span<cd>(storage)) {}
};

enum class kind { dct2, dct3, dct4, dst2, dst3, dst4 };

void run(kind k, const s::fft_plan<double>& plan, const std::vector<double>& x, std::vector<double>& y, std::vector<cd>& work, s::dct_norm norm)
{
    const std::span<const double> in(x);
    const std::span<double> out(y);
    const std::span<cd> w(work);
    switch (k) {
    case kind::dct2: s::dct2(plan, in, out, w, norm); break;
    case kind::dct3: s::dct3(plan, in, out, w, norm); break;
    case kind::dct4: s::dct4(plan, in, out, w, norm); break;
    case kind::dst2: s::dst2(plan, in, out, w, norm); break;
    case kind::dst3: s::dst3(plan, in, out, w, norm); break;
    case kind::dst4: s::dst4(plan, in, out, w, norm); break;
    }
}

// The definitions, in long double (scipy.fft's, norm=None), then scipy's orthonormal factors.
long double definition(kind k, const std::vector<double>& x, std::size_t out, bool ortho)
{
    const long double pi = 3.14159265358979323846264338327950288L;
    const std::size_t n = x.size();
    const long double nn = static_cast<long double>(n), kk = static_cast<long double>(out);
    long double y = 0;
    for (std::size_t j = 0; j < n; ++j) {
        const long double jj = static_cast<long double>(j), v = x[j];
        switch (k) {
        case kind::dct2: y += 2 * v * std::cos(pi * kk * (2 * jj + 1) / (2 * nn)); break;
        case kind::dct3: y += (j == 0 ? v : 2 * v * std::cos(pi * jj * (2 * kk + 1) / (2 * nn))); break;
        case kind::dct4: y += 2 * v * std::cos(pi * (2 * jj + 1) * (2 * kk + 1) / (4 * nn)); break;
        case kind::dst2: y += 2 * v * std::sin(pi * (kk + 1) * (2 * jj + 1) / (2 * nn)); break;
        case kind::dst3: y += (j == n - 1 ? (out % 2 ? -v : v) : 2 * v * std::sin(pi * (jj + 1) * (2 * kk + 1) / (2 * nn))); break;
        case kind::dst4: y += 2 * v * std::sin(pi * (2 * jj + 1) * (2 * kk + 1) / (4 * nn)); break;
        }
    }
    if (!ortho)
        return y;
    switch (k) {
    case kind::dct2: return y * (out == 0 ? std::sqrt(1 / (4 * nn)) : std::sqrt(1 / (2 * nn)));
    case kind::dst2: return y * (out == n - 1 ? std::sqrt(1 / (4 * nn)) : std::sqrt(1 / (2 * nn)));
    case kind::dct4:
    case kind::dst4: return y * std::sqrt(1 / (2 * nn));
    case kind::dct3:
    case kind::dst3: {
        // scipy scales the input, x_0 (dct3) or x_(n-1) (dst3) by 1 / sqrt(n) and the rest by 1 / sqrt(2n), then sums
        auto scaled = x;
        const std::size_t special = k == kind::dct3 ? 0 : n - 1;
        for (std::size_t j = 0; j < n; ++j)
            scaled[j] = static_cast<double>(x[j] * (j == special ? 1 / std::sqrt(nn) : 1 / std::sqrt(2 * nn)));
        return definition(k, scaled, out, false);
    }
    }
    return y;
}

TEST(Dct, EveryTypeIsScipysUnnormalisedAndOrthonormal)
{
    const std::array<std::pair<const char*, kind>, 6> kinds{{{"dct2", kind::dct2}, {"dct3", kind::dct3}, {"dct4", kind::dct4},
                                                             {"dst2", kind::dst2}, {"dst3", kind::dst3}, {"dst4", kind::dst4}}};
    for (const std::size_t n : {std::size_t{7}, std::size_t{8}}) {
        const auto& x = snicholls::test::signal_record("dct_input_" + std::to_string(n));
        planned p(n), p2(2 * n);
        for (const auto& [name, k] : kinds)
            for (const auto norm : {s::dct_norm::none, s::dct_norm::ortho}) {
                const std::string key = std::string(name) + "_" + std::to_string(n) + "_" + (norm == s::dct_norm::none ? "none" : "ortho");
                const auto& ref = snicholls::test::signal_record(key);
                std::vector<double> y(n);
                const bool four = k == kind::dct4 || k == kind::dst4;
                run(k, four ? p2.plan : p.plan, x, y, four ? p2.work : p.work, norm);
                for (std::size_t i = 0; i < n; ++i)
                    EXPECT_NEAR(y[i], ref[i], 1e-13) << key << " " << i;
            }
    }
}

TEST(Dct, EveryLengthToFortyIsTheDefinitionInLongDouble)
{
    if (std::numeric_limits<long double>::digits <= std::numeric_limits<double>::digits)
        GTEST_SKIP() << "long double is double here (Apple ARM64, MSVC): no wider reference to check the last bits against";

    for (std::size_t n = 1; n <= 40; ++n) {
        planned p(n), p2(2 * n);
        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i)
            x[i] = std::sin(0.7 * static_cast<double>(i * i) + 0.3);
        for (const kind k : {kind::dct2, kind::dct3, kind::dct4, kind::dst2, kind::dst3, kind::dst4})
            for (const bool ortho : {false, true}) {
                std::vector<double> y(n);
                const bool four = k == kind::dct4 || k == kind::dst4;
                run(k, four ? p2.plan : p.plan, x, y, four ? p2.work : p.work, ortho ? s::dct_norm::ortho : s::dct_norm::none);
                for (std::size_t i = 0; i < n; ++i)
                    EXPECT_NEAR(y[i], static_cast<double>(definition(k, x, i, ortho)), 2e-15 * static_cast<double>(n)) << n << " " << i;
            }
    }
}

TEST(Dct, InversesAndTruncation)
{
    for (const std::size_t n : {std::size_t{5}, std::size_t{16}, std::size_t{29}}) {
        planned p(n), p2(2 * n);
        std::vector<double> x(n), y(n), back(n);
        for (std::size_t i = 0; i < n; ++i)
            x[i] = std::cos(1.3 * static_cast<double>(i)) - 0.2;
        // unnormalised: dct3(dct2(x)) = 2n x; orthonormal: exact inverses; type IV is its own inverse
        run(kind::dct2, p.plan, x, y, p.work, s::dct_norm::none);
        run(kind::dct3, p.plan, y, back, p.work, s::dct_norm::none);
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(back[i], 2.0 * static_cast<double>(n) * x[i], 1e-12);
        run(kind::dst2, p.plan, x, y, p.work, s::dct_norm::ortho);
        run(kind::dst3, p.plan, y, back, p.work, s::dct_norm::ortho);
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(back[i], x[i], 1e-14);
        run(kind::dct4, p2.plan, x, y, p2.work, s::dct_norm::ortho);
        run(kind::dct4, p2.plan, y, back, p2.work, s::dct_norm::ortho);
        for (std::size_t i = 0; i < n; ++i)
            EXPECT_NEAR(back[i], x[i], 1e-14);
        // the first three coefficients alone are the full transform's first three
        std::vector<double> full(n), first(3);
        s::dct2(p.plan, std::span<const double>(x), std::span<double>(full), std::span<cd>(p.work), s::dct_norm::ortho);
        s::dct2(p.plan, std::span<const double>(x), std::span<double>(first), std::span<cd>(p.work), s::dct_norm::ortho);
        for (std::size_t i = 0; i < 3; ++i)
            EXPECT_EQ(first[i], full[i]);
    }
}

TEST(DctBackward, MatchesCentralDifferencesTruncatedOrNot)
{
    const std::size_t n = 9;
    planned p(n);
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 0.1 * static_cast<double>(i) - std::sin(static_cast<double>(i));
    for (const std::size_t kept : {n, std::size_t{4}})
        for (const auto norm : {s::dct_norm::none, s::dct_norm::ortho}) {
            const auto loss = [&](const std::vector<double>& v, std::vector<double>* grad) {
                std::vector<double> y(kept);
                std::vector<cd> w(p.work.size());
                s::dct2(p.plan, std::span<const double>(v), std::span<double>(y), std::span<cd>(w), norm);
                double l = 0;
                for (std::size_t k = 0; k < kept; ++k) {
                    l += (0.3 + 0.1 * static_cast<double>(k)) * y[k] * y[k];
                    if (grad)
                        (*grad)[k] = 2 * (0.3 + 0.1 * static_cast<double>(k)) * y[k];
                }
                return l;
            };
            std::vector<double> gy(kept), gx(n);
            loss(x, &gy);
            s::dct2_backward(p.plan, std::span<const double>(gy), std::span<double>(gx), std::span<cd>(p.work), norm);
            const double h = 1e-6;
            for (std::size_t j = 0; j < n; ++j) {
                auto up = x, down = x;
                up[j] += h;
                down[j] -= h;
                EXPECT_NEAR(gx[j], (loss(up, nullptr) - loss(down, nullptr)) / (2 * h), 2e-6 * (1 + std::abs(gx[j]))) << kept << " " << j;
            }
        }
}

} // namespace
