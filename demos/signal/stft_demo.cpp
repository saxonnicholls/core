//
//  stft_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/stft.hpp>: a chirp's short-time transform with torch.stft's defaults (centred,
//  reflect padding), the peak bin of each frame rising with the frequency; its exact inverse; a two-sided transform; a
//  power spectrogram without the complex spectrum; and the backward passes. Exits non-zero if a result is not what it
//  documents.
//

#include <snicholls/core/signal/stft.hpp>
#include <snicholls/core/signal/window.hpp>

#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal stft demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    using cd = s::complex<double>;
    bool ok = true;

    //! [stft]
    // One second of a chirp from 500 Hz to 3 kHz at 8 kHz; frames of 256 samples every 64.
    const std::size_t fs = 8000, n_fft = 256;
    std::vector<double> x(fs);
    for (std::size_t i = 0; i < fs; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(fs);
        x[i] = m::sin(2 * m::pi * (500.0 * t + 1250.0 * t * t));
    }
    const s::stft_options o{n_fft, 64, true, s::pad_mode::reflect, false};
    constexpr auto window = s::make_window<double, 256>(s::hann{});
    std::vector<cd> storage(s::rfft_plan<double>::storage_size(n_fft)), work(s::rfft_plan<double>::work_size(n_fft));
    const s::rfft_plan<double> plan(n_fft, storage);
    const std::size_t frames = s::stft_frames(x.size(), o), bins = plan.bins();   // 126 frames of 129 bins
    std::vector<cd> spec(frames * bins);
    s::stft(plan, std::span<const double>(x), std::span<const double>(window), std::span<cd>(spec), std::span<cd>(work), o);
    std::vector<double> back(s::istft_length(frames, o));                         // the signal's length again
    s::istft(plan, std::span<const cd>(spec), std::span<const double>(window), std::span<double>(back), std::span<cd>(work), o);
    //! [stft]

    const auto peak = [&](std::size_t t) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < bins; ++k)
            best = s::norm(spec[t * bins + k]) > s::norm(spec[t * bins + best]) ? k : best;
        return static_cast<double>(best) * static_cast<double>(fs) / static_cast<double>(n_fft);
    };
    double worst = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        worst = m::abs(back[i] - x[i]) > worst ? m::abs(back[i] - x[i]) : worst;
    std::printf("  %zu frames x %zu bins; peak at frame 10: %.0f Hz, frame 62: %.0f Hz, frame 115: %.0f Hz\n", frames, bins, peak(10), peak(62), peak(115));
    std::printf("  istft(stft(x)): %zu samples, max error %.3g\n", back.size(), worst);
    ok &= frames == 126 && back.size() == x.size() && worst < 1e-12;
    ok &= m::abs(peak(10) - 700) < 40 && m::abs(peak(62) - 1750) < 40 && m::abs(peak(115) - 2800) < 40;

    // Two-sided, from a complex plan: bins n_fft - k mirror bins k.
    std::vector<cd> st2(s::fft_plan<double>::storage_size(n_fft)), w2(n_fft + s::fft_plan<double>::work_size(n_fft)), two(frames * n_fft);
    const s::fft_plan<double> cplan(n_fft, st2);
    s::stft(cplan, std::span<const double>(x), std::span<const double>(window), std::span<cd>(two), std::span<cd>(w2), o);
    std::vector<double> back2(x.size());
    s::istft(cplan, std::span<const cd>(two), std::span<const double>(window), std::span<double>(back2), std::span<cd>(w2), o);
    std::printf("  two-sided: bin 40 %.6g, bin 216 conj %.6g; round trip x[1000] %.12g\n", s::abs(two[10 * n_fft + 40]), s::abs(two[10 * n_fft + 216]), back2[1000]);
    ok &= m::near(s::abs(two[10 * n_fft + 40] - s::conj(two[10 * n_fft + 216])), 0.0, 1e-12) && m::near(back2[1000], x[1000], 1e-12);

    // Power and magnitude spectrograms frame by frame (no complex spectrum stored), normalised.
    s::stft_options on = o;
    on.normalized = true;
    std::vector<double> power(frames * bins), mag(frames * bins), from_spec(frames * bins);
    std::vector<cd> sw(s::spectrogram_work_size(plan));
    s::spectrogram(plan, std::span<const double>(x), std::span<const double>(window), std::span<double>(power), std::span<cd>(sw), on);
    s::spectrogram(plan, std::span<const double>(x), std::span<const double>(window), std::span<double>(mag), std::span<cd>(sw), on,
                   s::spectrum_kind::magnitude);
    s::power<double>(spec, from_spec);
    std::printf("  normalised power at frame 62's peak: %.6g = |X|^2 / n_fft %.6g; magnitude %.6g\n", power[62 * bins + 56], from_spec[62 * bins + 56] / 256,
                mag[62 * bins + 56]);
    ok &= m::near(power[62 * bins + 56], from_spec[62 * bins + 56] / 256, 1e-9) && m::near(mag[62 * bins + 56] * mag[62 * bins + 56], power[62 * bins + 56], 1e-9);
    s::magnitude<double>(spec, from_spec);

    // Backward passes: a loss on the power spectrogram back to the samples.
    std::vector<double> grad_power(frames * bins, 0.0), grad_x(x.size());
    grad_power[62 * bins + 56] = 1.0;   // L = |X[62][56]|^2
    std::vector<cd> grad_spec(frames * bins);
    s::power_backward<double>(spec, grad_power, grad_spec);
    s::stft_backward(plan, std::span<const cd>(grad_spec), std::span<const double>(window), std::span<double>(grad_x), std::span<cd>(work), o);
    std::vector<double> gm(frames * bins, 0.0);
    std::vector<cd> gs(frames * bins), gspec(frames * bins);
    s::magnitude_backward<double>(spec, gm, gs);
    std::vector<double> gy(x.size(), 0.0);
    s::istft_backward(plan, std::span<const double>(gy), std::span<const double>(window), std::span<cd>(gspec), std::span<cd>(work), o);
    double outside = 0;
    for (std::size_t i = 0; i < 3500; ++i)
        outside += m::abs(grad_x[i]);
    std::printf("  dL/dx is zero outside frame 62's samples: %g; inside, dL/dx[3968] = %.6g\n", outside, grad_x[3968]);
    ok &= outside == 0.0 && m::abs(grad_x[3968]) > 0 && s::abs(gs[0]) == 0.0 && s::abs(gspec[100]) == 0.0;

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
