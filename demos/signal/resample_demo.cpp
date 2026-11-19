//
//  resample_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/resample.hpp>: 44.1 kHz to 48 kHz through a kernel the compiler builds, and 48 kHz
//  to 16 kHz (Whisper's rate) through a kernel built at run time. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/signal/resample.hpp>

#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal resample demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    bool ok = true;

    //! [resample]
    // CD to DAT: 160 / 147, a kernel of 3,201 taps written by the compiler.
    static constexpr s::polyphase_resampler<float, 160, 147> cd_to_dat{};
    std::vector<float> cd(44100), dat(cd_to_dat.output_size(cd.size()));   // one second each
    for (std::size_t i = 0; i < cd.size(); ++i)
        cd[i] = static_cast<float>(m::sin(2 * m::pi * 440.0 * static_cast<double>(i) / 44100.0));
    cd_to_dat.process(cd, dat);
    // 48 kHz to 16 kHz for a speech model: the factor 1 / 3 known only at run time.
    const std::size_t up = 16000, down = 48000;
    std::vector<double> h(s::resample_kernel_size(up, down)), speech(4800), out(s::resample_size(speech.size(), up, down));
    s::resample_kernel(up, down, std::span<double>(h));
    for (std::size_t i = 0; i < speech.size(); ++i)
        speech[i] = m::sin(2 * m::pi * 300.0 * static_cast<double>(i) / 48000.0) + 0.5 * m::sin(2 * m::pi * 12000.0 * static_cast<double>(i) / 48000.0);
    s::resample_poly<double>(speech, up, down, h, out);   // the 12 kHz component is above the new Nyquist: removed
    //! [resample]

    double worst_dat = 0, worst_speech = 0;
    for (std::size_t i = 500; i + 500 < dat.size(); ++i)
        worst_dat = m::abs(static_cast<double>(dat[i]) - m::sin(2 * m::pi * 440.0 * static_cast<double>(i) / 48000.0)) > worst_dat
                        ? m::abs(static_cast<double>(dat[i]) - m::sin(2 * m::pi * 440.0 * static_cast<double>(i) / 48000.0))
                        : worst_dat;
    for (std::size_t i = 100; i + 100 < out.size(); ++i)
        worst_speech = m::abs(out[i] - m::sin(2 * m::pi * 300.0 * static_cast<double>(i) / 16000.0)) > worst_speech
                           ? m::abs(out[i] - m::sin(2 * m::pi * 300.0 * static_cast<double>(i) / 16000.0))
                           : worst_speech;
    const auto r = s::resample_ratio::of(up, down);
    std::printf("  44.1 -> 48 kHz: %zu -> %zu samples, %zu taps, within %.2e of the tone at 48 kHz\n", cd.size(), dat.size(), cd_to_dat.taps, worst_dat);
    std::printf("  48 -> 16 kHz (ratio %zu/%zu, %zu taps): %zu samples, the 12 kHz alias removed, within %.2e of the 300 Hz tone\n", r.up, r.down,
                h.size(), out.size(), worst_speech);
    ok &= dat.size() == 48000 && cd_to_dat.taps == 3201 && worst_dat < 2e-3 && cd_to_dat.kernel()[1600] > 0.9f;
    ok &= r.up == 1 && r.down == 3 && h.size() == 61 && out.size() == 1600 && worst_speech < 5e-3;

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
