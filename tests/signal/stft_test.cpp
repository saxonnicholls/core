//
//  stft_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/stft.hpp>: torch.stft's frame counts and padding, every frame against the direct
//  transform of the padded, windowed frame, one- and two-sided, normalised or not; istft round trips (Whisper's
//  window and hop, centred and not); spectrograms; the backward passes against central differences; compile-time use.
//

#include <snicholls/core/signal/stft.hpp>
#include <snicholls/core/signal/window.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

namespace s = snicholls::signal;
namespace m = snicholls::math;
using cd = s::complex<double>;

constexpr s::stft_options whisper{400, 160, true, s::pad_mode::reflect, false};
static_assert(s::stft_frames(480000, whisper) == 3001 && s::stft_frames(1600, whisper) == 11 && s::stft_frames(0, whisper) == 1);
static_assert(s::stft_frames(399, {400, 160, false, s::pad_mode::reflect, false}) == 0);
static_assert(s::istft_length(11, whisper) == 1600 && s::istft_length(0, whisper) == 0);
static_assert(s::detail::stft::source_index(0, 3, 10, s::pad_mode::reflect) == 3 && s::detail::stft::source_index(12, 3, 10, s::pad_mode::reflect) == 9);
static_assert(s::detail::stft::source_index(13, 3, 10, s::pad_mode::reflect) == 8 && s::detail::stft::source_index(0, 3, 10, s::pad_mode::constant) == -1);

// A short-time transform and its inverse in constant evaluation: n_fft 8, hop 2, a periodic Hann window.
constexpr bool round_trip_at_compile_time()
{
    constexpr std::size_t n = 20;
    std::array<double, n> x{};
    for (std::size_t i = 0; i < n; ++i)
        x[i] = static_cast<double>((i * 7) % 5) - 2.0;
    constexpr s::stft_options o{8, 2, true, s::pad_mode::reflect, false};
    const auto w = s::make_window<double, 8>(s::hann{});
    const auto plan = s::static_rfft_plan_v<double, 8>.plan();
    std::array<cd, s::stft_frames(n, o) * 5> spec{};
    std::array<cd, s::static_rfft_plan<double, 8>::work_size> work{};
    s::stft(plan, std::span<const double>(x), std::span<const double>(w), std::span<cd>(spec), std::span<cd>(work), o);
    std::array<double, n> back{};
    s::istft(plan, std::span<const cd>(spec), std::span<const double>(w), std::span<double>(back), std::span<cd>(work), o);
    for (std::size_t i = 0; i < n; ++i)
        if (m::abs(back[i] - x[i]) > 1e-13)
            return false;
    return m::near(spec[0].im, 0.0, 0.0);
}
static_assert(round_trip_at_compile_time());

std::vector<double> signal(std::size_t n, std::uint64_t seed)
{
    std::vector<double> x(n);
    for (auto& v : x) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        v = static_cast<double>(seed >> 11) * 0x1.0p-53 - 0.5;
    }
    return x;
}

struct real_planned
{
    std::vector<cd> storage, work;
    s::rfft_plan<double> plan;
    explicit real_planned(std::size_t n)
        : storage(s::rfft_plan<double>::storage_size(n)), work(s::rfft_plan<double>::work_size(n) + n), plan(n, std::span<cd>(storage))
    {
    }
};

struct complex_planned
{
    std::vector<cd> storage, work;
    s::fft_plan<double> plan;
    explicit complex_planned(std::size_t n)
        : storage(s::fft_plan<double>::storage_size(n)), work(s::fft_plan<double>::work_size(n) + n), plan(n, std::span<cd>(storage))
    {
    }
};

// The padded signal written out, as torch.nn.functional.pad would.
std::vector<double> padded(const std::vector<double>& x, std::size_t pad, s::pad_mode mode)
{
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(x.size());
    std::vector<double> out;
    for (std::ptrdiff_t i = -static_cast<std::ptrdiff_t>(pad); i < n + static_cast<std::ptrdiff_t>(pad); ++i) {
        if (i >= 0 && i < n)
            out.push_back(x[static_cast<std::size_t>(i)]);
        else if (mode == s::pad_mode::constant)
            out.push_back(0.0);
        else
            out.push_back(x[static_cast<std::size_t>(i < 0 ? -i : 2 * (n - 1) - i)]);
    }
    return out;
}

TEST(Stft, EveryFrameIsTheDirectTransformOfItsPaddedWindowedSamples)
{
    const auto x = signal(103, 1);
    for (const bool center : {true, false})
        for (const auto mode : {s::pad_mode::reflect, s::pad_mode::constant})
            for (const bool normalized : {false, true}) {
                const s::stft_options o{16, 5, center, mode, normalized};
                std::vector<double> w(16);
                s::fill_window(std::span<double>(w), s::hamming{});
                real_planned rp(16);
                complex_planned cp(16);
                const std::size_t frames = s::stft_frames(x.size(), o);
                ASSERT_EQ(frames, 1 + (x.size() + (center ? 16 : 0) - 16) / 5);
                std::vector<cd> one(frames * 9), two(frames * 16);
                s::stft(rp.plan, std::span<const double>(x), std::span<const double>(w), std::span<cd>(one), std::span<cd>(rp.work), o);
                s::stft(cp.plan, std::span<const double>(x), std::span<const double>(w), std::span<cd>(two), std::span<cd>(cp.work), o);
                const auto p = padded(x, center ? 8 : 0, mode);
                for (std::size_t t = 0; t < frames; ++t) {
                    std::vector<cd> frame(16), ref(16);
                    for (std::size_t j = 0; j < 16; ++j)
                        frame[j] = {w[j] * p[t * 5 + j], 0};
                    s::dft<double>(frame, ref, false, normalized ? s::fft_norm::ortho : s::fft_norm::backward);
                    for (std::size_t k = 0; k < 16; ++k) {
                        EXPECT_LE(s::abs(two[t * 16 + k] - ref[k]), 1e-14);
                        if (k < 9) {
                            EXPECT_LE(s::abs(one[t * 9 + k] - ref[k]), 1e-14);
                        }
                    }
                }
            }
}

TEST(Stft, InverseReconstructsWithWhispersWindowAndHop)
{
    const auto x = signal(1600, 2);
    std::vector<double> w(400);
    s::fill_window(std::span<double>(w), s::hann{});
    real_planned p(400);
    for (const bool normalized : {false, true}) {
        s::stft_options o = whisper;
        o.normalized = normalized;
        std::vector<cd> spec(s::stft_frames(x.size(), o) * 201);
        s::stft(p.plan, std::span<const double>(x), std::span<const double>(w), std::span<cd>(spec), std::span<cd>(p.work), o);
        std::vector<double> back(s::istft_length(spec.size() / 201, o));
        ASSERT_EQ(back.size(), x.size());
        s::istft(p.plan, std::span<const cd>(spec), std::span<const double>(w), std::span<double>(back), std::span<cd>(p.work), o);
        for (std::size_t i = 0; i < x.size(); ++i)
            ASSERT_NEAR(back[i], x[i], 1e-13) << i;
        // Past the last frame's end the output is zero, as torch.istft pads a longer length.
        std::vector<double> longer(x.size() + 300);
        s::istft(p.plan, std::span<const cd>(spec), std::span<const double>(w), std::span<double>(longer), std::span<cd>(p.work), o);
        EXPECT_NEAR(longer[100], x[100], 1e-13);
        EXPECT_EQ(longer[x.size() + 299], 0.0);
    }
}

TEST(Stft, InverseReconstructsUncentredTwoSidedAndInFloat)
{
    const auto x = signal(300, 3);
    std::vector<double> w(32);
    s::fill_window(std::span<double>(w), s::hamming{});   // no zero, so the ends are covered without centring
    const s::stft_options o{32, 8, false, s::pad_mode::reflect, false};
    complex_planned cp(32);
    const std::size_t frames = s::stft_frames(x.size(), o);
    std::vector<cd> spec(frames * 32);
    s::stft(cp.plan, std::span<const double>(x), std::span<const double>(w), std::span<cd>(spec), std::span<cd>(cp.work), o);
    std::vector<double> back(s::istft_length(frames, o));
    s::istft(cp.plan, std::span<const cd>(spec), std::span<const double>(w), std::span<double>(back), std::span<cd>(cp.work), o);
    for (std::size_t i = 0; i < back.size(); ++i)
        EXPECT_NEAR(back[i], x[i], 1e-13);

    std::vector<float> xf(x.begin(), x.end()), wf(400), backf(xf.size());
    s::fill_window(std::span<float>(wf), s::hann{});
    std::vector<s::complex<float>> st(s::rfft_plan<float>::storage_size(400)), work(s::rfft_plan<float>::work_size(400));
    const s::rfft_plan<float> pf(400, st);
    std::vector<s::complex<float>> spec_f(s::stft_frames(xf.size(), whisper) * 201);
    s::stft(pf, std::span<const float>(xf), std::span<const float>(wf), std::span<s::complex<float>>(spec_f), std::span<s::complex<float>>(work), whisper);
    s::istft(pf, std::span<const s::complex<float>>(spec_f), std::span<const float>(wf), std::span<float>(backf), std::span<s::complex<float>>(work), whisper);
    for (std::size_t i = 0; i < xf.size(); ++i)
        EXPECT_NEAR(backf[i], xf[i], 2e-6f);
}

TEST(Stft, SpectrogramIsTheMagnitudeOrPowerOfTheTransform)
{
    const auto x = signal(500, 4);
    std::vector<double> w(64);
    s::fill_window(std::span<double>(w), s::blackman{});
    const s::stft_options o{64, 16, true, s::pad_mode::reflect, true};
    real_planned p(64);
    const std::size_t frames = s::stft_frames(x.size(), o), bins = 33;
    std::vector<cd> spec(frames * bins);
    s::stft(p.plan, std::span<const double>(x), std::span<const double>(w), std::span<cd>(spec), std::span<cd>(p.work), o);
    std::vector<double> pw(frames * bins), mg(frames * bins), pw2(frames * bins), mg2(frames * bins);
    std::vector<cd> work(s::spectrogram_work_size(p.plan));
    s::spectrogram(p.plan, std::span<const double>(x), std::span<const double>(w), std::span<double>(pw), std::span<cd>(work), o);
    s::spectrogram(p.plan, std::span<const double>(x), std::span<const double>(w), std::span<double>(mg), std::span<cd>(work), o,
                   s::spectrum_kind::magnitude);
    s::power<double>(spec, pw2);
    s::magnitude<double>(spec, mg2);
    for (std::size_t i = 0; i < spec.size(); ++i) {
        EXPECT_NEAR(pw[i], pw2[i], 1e-14 * (1 + pw2[i]));
        EXPECT_NEAR(mg[i], mg2[i], 1e-14 * (1 + mg2[i]));
    }
}

double spec_loss(const std::vector<cd>& y, std::vector<cd>* grad)
{
    double l = 0;
    for (std::size_t k = 0; k < y.size(); ++k) {
        const double a = 0.2 + 0.01 * static_cast<double>(k % 13);
        const cd c{0.1 * static_cast<double>(k % 3) - 0.1, 0.03 * static_cast<double>(k % 7)};
        l += a * s::norm(y[k]) + (c * y[k]).re;
        if (grad)
            (*grad)[k] = y[k] * (2 * a) + s::conj(c);
    }
    return l;
}

TEST(StftBackward, StftAndIstftMatchCentralDifferences)
{
    const auto x = signal(37, 5);
    std::vector<double> w(8);
    s::fill_window(std::span<double>(w), s::hann{});
    real_planned p(8);
    for (const auto mode : {s::pad_mode::reflect, s::pad_mode::constant})
        for (const bool normalized : {false, true}) {
            const s::stft_options o{8, 3, true, mode, normalized};
            const std::size_t frames = s::stft_frames(x.size(), o);
            const auto forward = [&](const std::vector<double>& v) {
                std::vector<cd> out(frames * 5);
                s::stft(p.plan, std::span<const double>(v), std::span<const double>(w), std::span<cd>(out), std::span<cd>(p.work), o);
                return out;
            };
            std::vector<cd> g(frames * 5);
            spec_loss(forward(x), &g);
            std::vector<double> gx(x.size());
            s::stft_backward(p.plan, std::span<const cd>(g), std::span<const double>(w), std::span<double>(gx), std::span<cd>(p.work), o);
            const double h = 1e-6;
            for (std::size_t j = 0; j < x.size(); ++j) {
                auto up = x, down = x;
                up[j] += h;
                down[j] -= h;
                EXPECT_NEAR(gx[j], (spec_loss(forward(up), nullptr) - spec_loss(forward(down), nullptr)) / (2 * h), 1e-7) << j;
            }
            // istft: X -> y with L = sum_i c_i y_i^2
            const auto spec = forward(x);
            const auto inverse = [&](const std::vector<cd>& v) {
                std::vector<double> out(x.size());
                s::istft(p.plan, std::span<const cd>(v), std::span<const double>(w), std::span<double>(out), std::span<cd>(p.work), o);
                return out;
            };
            const auto loss = [](const std::vector<double>& y, std::vector<double>* grad) {
                double l = 0;
                for (std::size_t i = 0; i < y.size(); ++i) {
                    const double c = 0.5 + 0.1 * static_cast<double>(i % 4);
                    l += c * y[i] * y[i];
                    if (grad)
                        (*grad)[i] = 2 * c * y[i];
                }
                return l;
            };
            std::vector<double> gy(x.size());
            loss(inverse(spec), &gy);
            std::vector<cd> gs(spec.size());
            s::istft_backward(p.plan, std::span<const double>(gy), std::span<const double>(w), std::span<cd>(gs), std::span<cd>(p.work), o);
            for (std::size_t k = 0; k < spec.size(); ++k)
                for (int part = 0; part < 2; ++part) {
                    auto up = spec, down = spec;
                    (part == 0 ? up[k].re : up[k].im) += h;
                    (part == 0 ? down[k].re : down[k].im) -= h;
                    EXPECT_NEAR(part == 0 ? gs[k].re : gs[k].im, (loss(inverse(up), nullptr) - loss(inverse(down), nullptr)) / (2 * h), 1e-7) << k;
                }
        }
}

TEST(StftBackward, PowerAndMagnitudeMatchCentralDifferences)
{
    const std::vector<cd> z{{0.3, -0.4}, {1.5, 0.2}, {-0.7, 0.9}, {0, 0}};
    const std::vector<double> gp{0.5, -1.0, 2.0, 1.0};
    std::vector<cd> dz(4), dm(4);
    s::power_backward<double>(z, gp, dz);
    s::magnitude_backward<double>(z, gp, dm);
    const double h = 1e-6;
    for (std::size_t i = 0; i < 3; ++i)
        for (int part = 0; part < 2; ++part) {
            auto up = z[i], down = z[i];
            (part == 0 ? up.re : up.im) += h;
            (part == 0 ? down.re : down.im) -= h;
            EXPECT_NEAR(part == 0 ? dz[i].re : dz[i].im, gp[i] * (s::norm(up) - s::norm(down)) / (2 * h), 1e-8);
            EXPECT_NEAR(part == 0 ? dm[i].re : dm[i].im, gp[i] * (s::abs(up) - s::abs(down)) / (2 * h), 1e-8);
        }
    EXPECT_EQ(dm[3], (cd{0, 0}));   // the magnitude's gradient at 0 is 0, as PyTorch's
}

} // namespace
