//
//  mel_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/signal/mel.hpp>: the mel scales; a filterbank built by the compiler; a mel spectrogram in
//  decibels and its MFCCs; OpenAI Whisper's log-mel front end on a tone, and its gradient with respect to the audio.
//  Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/signal/mel.hpp>

#include <cstddef>
#include <cstdio>
#include <vector>

int main()
{
    std::printf("snicholls-core signal mel demo\n");
    namespace s = snicholls::signal;
    namespace m = snicholls::math;
    using cd = s::complex<double>;
    bool ok = true;

    //! [mel]
    // The scales: 1 kHz is 15 Slaney mels and about 1000 HTK mels.
    static_assert(m::near(s::hz_to_mel(1000.0), 15.0, 1e-13) && m::near(s::hz_to_mel(1000.0, s::mel_scale::htk), 1000.0, 0.02));
    // librosa.filters.mel(sr=16000, n_fft=512, n_mels=40), computed by the compiler.
    constexpr auto filters = s::make_mel_filterbank<double, 40, 512>(16000.0);
    // A power spectrogram (here 2 frames of 257 bins of white noise, power 1) to mel bands, decibels and MFCCs.
    const std::vector<double> power(2 * 257, 1.0);
    std::vector<double> mel(2 * 40), db(2 * 40), coefficients(2 * 13);
    s::mel_spectrogram<double>(filters, 40, power, mel);
    s::log_compress<double>(mel, db, s::decibels<double>());   // librosa's power_to_db, top_db 80
    std::vector<cd> st(s::fft_plan<double>::storage_size(40));
    const s::fft_plan<double> dct_plan(40, st);
    std::vector<cd> work(s::dct_work_size(dct_plan));
    s::mfcc(dct_plan, std::span<const double>(db), std::span<double>(coefficients), 13, std::span<cd>(work));
    //! [mel]

    std::printf("  hz_to_mel(4000) = %.6f slaney, %.6f htk; mel_to_hz(30) = %.3f Hz\n", s::hz_to_mel(4000.0), s::hz_to_mel(4000.0, s::mel_scale::htk),
                s::mel_to_hz(30.0));
    std::printf("  white noise through Slaney-normalised filters: band 0 %.6g, band 39 %.6g (equal area per hertz)\n", mel[0], mel[39]);
    std::printf("  MFCC 0 = %.6f (the mean log energy, times sqrt(40)), MFCC 1 = %.6f\n", coefficients[0], coefficients[1]);
    ok &= m::near(s::mel_to_hz(s::hz_to_mel(4000.0)), 4000.0, 1e-9) && m::near(s::mel_to_hz(15.0), 1000.0, 1e-12);
    ok &= mel[0] > 0 && coefficients[0] < 0 && coefficients[13] == coefficients[0];
    double sum = 0;
    for (std::size_t i = 0; i < 40; ++i)
        sum += db[i];
    ok &= m::near(coefficients[0], sum / m::sqrt(40.0), 1e-12);

    // Backward through the decibels and the filters: dL/dP for L = the sum of the decibels.
    const std::vector<double> ones(2 * 40, 1.0);
    std::vector<double> grad_mel(2 * 40), grad_power(2 * 257);
    s::log_compress_backward<double>(mel, ones, grad_mel, s::decibels<double>());   // 10 / (mel ln 10)
    s::mel_backward<double>(filters, 40, grad_mel, grad_power);
    std::printf("  d sum(dB) / d mel[0] = %.6g = 10 / (mel[0] ln 10); d / d P[0][10] = %.6g\n", grad_mel[0], grad_power[10]);
    ok &= m::near(grad_mel[0], 10 / (mel[0] * m::log(10.0)), 1e-9) && grad_power[10] > 0;

    //! [whisper]
    // Whisper's front end on a tenth of a second of a 1 kHz tone: 80 mels x 10 frames, mel-major as Whisper's tensor.
    std::vector<float> audio(1600);
    for (std::size_t i = 0; i < audio.size(); ++i)
        audio[i] = static_cast<float>(0.5 * m::sin(2 * m::pi * 1000.0 * static_cast<double>(i) / 16000.0));
    static constexpr auto window = s::whisper_window<float>();            // torch.hann_window(400)
    static constexpr auto mel80 = s::whisper_mel_filters<float, 80>();    // whisper's mel_filters.npz, by the compiler
    std::vector<s::complex<float>> rst(s::rfft_plan<float>::storage_size(s::whisper_n_fft));
    const s::rfft_plan<float> plan(s::whisper_n_fft, rst);
    std::vector<s::complex<float>> wwork(s::whisper_work_size(plan));
    const std::size_t frames = s::whisper_frames(audio.size());           // 10
    std::vector<float> log_mel(80 * frames);
    s::whisper_log_mel(plan, std::span<const float>(window), std::span<const float>(mel80), 80, std::span<const float>(audio), 0,
                       std::span<float>(log_mel), std::span<s::complex<float>>(wwork));
    //! [whisper]

    std::size_t top_band = 0;
    for (std::size_t b = 0; b < 80; ++b)
        top_band = log_mel[b * frames + 5] > log_mel[top_band * frames + 5] ? b : top_band;
    const double centre = s::mel_to_hz(s::hz_to_mel(0.0) + (s::hz_to_mel(8000.0) - s::hz_to_mel(0.0)) * static_cast<double>(top_band + 1) / 81.0);
    std::printf("  whisper_log_mel: %zu frames; frame 5's loudest band %zu (centre %.0f Hz), value %.4f\n", frames, top_band, centre,
                static_cast<double>(log_mel[top_band * frames + 5]));
    ok &= frames == 10 && m::abs(centre - 1000.0) < 60.0 && log_mel[top_band * frames + 5] > 0.5f;

    // The gradient of a loss on the log-mel spectrogram with respect to the audio: training through the front end.
    std::vector<float> grad_out(80 * frames, 0.0f), grad_audio(audio.size()), scratch(80 * frames);
    grad_out[top_band * frames + 5] = 1.0f;
    s::whisper_log_mel_backward(plan, std::span<const float>(window), std::span<const float>(mel80), 80, std::span<const float>(audio), 0,
                                std::span<const float>(grad_out), std::span<float>(grad_audio), std::span<float>(scratch),
                                std::span<s::complex<float>>(wwork));
    std::printf("  whisper_log_mel_backward: dL/d audio[805] = %.6g, zero before frame 5's samples: %g\n", static_cast<double>(grad_audio[805]),
                static_cast<double>(grad_audio[100]));
    ok &= m::abs(grad_audio[805]) > 1e-4f && grad_audio[100] == 0.0f;   // (at 800 the tone crosses zero, and so does dL/d audio)

    std::printf(ok ? "  ok\n" : "  FAILED\n");
    return ok ? 0 : 1;
}
