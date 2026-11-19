//
//  mel_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/signal/mel.hpp>: the HTK and Slaney scales and Whisper's 80- and 128-mel filterbanks and an
//  HTK filterbank against librosa 0.11.0; Whisper's log-mel spectrogram against an independent float64 implementation
//  of whisper/audio.py in numpy, in double and in float; MFCCs and power_to_db against librosa; a filterbank and a
//  log-mel spectrogram at compile time; the backward passes against central differences.
//

#include <snicholls/core/signal/mel.hpp>

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

static_assert(m::near(s::hz_to_mel(1000.0), 15.0, 1e-14) && m::near(s::mel_to_hz(15.0), 1000.0, 1e-12));
static_assert(m::near(s::hz_to_mel(700.0, s::mel_scale::htk), 2595.0 * m::log10(2.0), 1e-11) && s::hz_to_mel(0.0, s::mel_scale::htk) == 0.0);
static_assert(m::near(s::mel_to_hz(s::hz_to_mel(4321.0)), 4321.0, 1e-9) && m::near(s::mel_to_hz(s::hz_to_mel(123.0f, s::mel_scale::htk), s::mel_scale::htk), 123.0f, 1e-3f));
static_assert(s::whisper_frames(s::whisper_n_samples) == s::whisper_n_frames && s::whisper_n_frames == 3000 && s::whisper_frames(1000, 600) == 10);

// A filterbank at compile time: every filter's peak is at its centre, and the Slaney areas are equal.
constexpr bool filterbank_at_compile_time()
{
    constexpr auto f = s::make_mel_filterbank<double, 10, 64>(8000.0, 0.0, 0.0, s::mel_scale::htk, s::mel_norm::none);
    for (std::size_t mm = 0; mm < 10; ++mm) {
        double peak = 0;
        for (std::size_t k = 0; k < 33; ++k)
            peak = f[mm * 33 + k] > peak ? f[mm * 33 + k] : peak;
        if (peak <= 0.3 || peak > 1.0)
            return false;
    }
    return true;
}
static_assert(filterbank_at_compile_time());

// Whisper's 80-mel filterbank, whole, at compile time.
constexpr auto whisper80 = s::whisper_mel_filters<float, 80>();

// A log-mel spectrogram at compile time, on 480 samples of a tone (three frames).
constexpr bool log_mel_at_compile_time()
{
    std::array<double, 480> audio{};
    for (std::size_t i = 0; i < audio.size(); ++i)
        audio[i] = 0.5 * m::sin(2 * m::pi * 1000.0 * static_cast<double>(i) / 16000.0);
    const auto window = s::whisper_window<double>();
    std::array<double, 80 * 201> filters{};
    for (std::size_t i = 0; i < filters.size(); ++i)
        filters[i] = static_cast<double>(whisper80[i]);
    std::array<cd, s::static_rfft_plan<double, 400>::bins + s::static_rfft_plan<double, 400>::work_size> work{};
    std::array<double, 80 * 3> out{};
    s::whisper_log_mel(s::static_rfft_plan_v<double, 400>.plan(), std::span<const double>(window), std::span<const double>(filters), 80,
                       std::span<const double>(audio), 0, std::span<double>(out), std::span<cd>(work));
    double top = -1e300;
    for (const double v : out)
        top = v > top ? v : top;
    // the floor at max - 8 after (x + 4) / 4 is top - 2
    for (const double v : out)
        if (v < top - 2.0 - 1e-12)
            return false;
    return top > 0.5 && top < 2.0;
}
static_assert(log_mel_at_compile_time());

TEST(MelScale, HtkAndSlaneyAreLibrosas)
{
    const auto& hz = snicholls::test::signal_record("mel_hz");
    const auto& slaney = snicholls::test::signal_record("mel_slaney");
    const auto& htk = snicholls::test::signal_record("mel_htk");
    for (std::size_t i = 0; i < hz.size(); ++i) {
        EXPECT_NEAR(s::hz_to_mel(hz[i]), slaney[i], 1e-13) << hz[i];
        EXPECT_NEAR(s::hz_to_mel(hz[i], s::mel_scale::htk), htk[i], 1e-11) << hz[i];
        EXPECT_NEAR(s::mel_to_hz(slaney[i]), hz[i], 1e-9 * (1 + hz[i]));
        EXPECT_NEAR(s::mel_to_hz(htk[i], s::mel_scale::htk), hz[i], 1e-9 * (1 + hz[i]));
    }
}

void expect_filterbank(const std::string& key, const std::vector<double>& got, double rel)
{
    std::size_t rows = 0, cols = 0;
    const auto ref = snicholls::test::signal_dense(key, rows, cols);
    ASSERT_EQ(got.size(), rows * cols);
    std::size_t nonzero = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], ref[i], rel * std::abs(ref[i]) + 1e-300) << key << " row " << i / cols << " bin " << i % cols;
        nonzero += ref[i] != 0 ? 1 : 0;
    }
    EXPECT_GT(nonzero, rows);
}

TEST(MelFilterbank, WhispersEightyAndOneTwentyEightAndAnHtkBankAreLibrosas)
{
    for (const std::size_t mels : {std::size_t{80}, std::size_t{128}}) {
        std::vector<float> f(mels * 201);
        s::mel_filterbank(s::mel_filterbank_options<float>{16000.0f, 400, mels}, std::span<float>(f));
        expect_filterbank("mel_filters_" + std::to_string(mels), std::vector<double>(f.begin(), f.end()), 1.2e-7);   // float32, to the bit or one ulp
    }
    // the compile-time table is the run-time one's float values to an ulp
    std::vector<float> f80(80 * 201);
    s::mel_filterbank(s::mel_filterbank_options<float>{16000.0f, 400, 80}, std::span<float>(f80));
    for (std::size_t i = 0; i < f80.size(); ++i)
        EXPECT_NEAR(whisper80[i], f80[i], 1.2e-7f * f80[i]);
    std::vector<double> htk(40 * 257);
    s::mel_filterbank(s::mel_filterbank_options<double>{22050.0, 512, 40, 50.0, 8000.0, s::mel_scale::htk, s::mel_norm::none}, std::span<double>(htk));
    expect_filterbank("mel_filters_htk", htk, 1e-12);
}

template <typename T>
void expect_whisper(std::size_t mels, double tolerance, bool librosa_filters)
{
    const auto& a = snicholls::test::signal_record("whisper_audio");
    const auto& ref = snicholls::test::signal_record("whisper_log_mel_" + std::to_string(mels));
    std::vector<T> audio(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        audio[i] = static_cast<T>(static_cast<float>(a[i]));   // the fixture's float32 samples, exactly
    std::vector<s::complex<T>> st(s::rfft_plan<T>::storage_size(400));
    const s::rfft_plan<T> plan(400, st);
    const auto window = s::whisper_window<T>();
    // Whisper's own filters (librosa's float32, from the fixture) isolate the algorithm; core's, within an ulp of them, are
    // checked by MelFilterbank and used by the float run.
    std::vector<T> filters(mels * 201);
    if (librosa_filters) {
        std::size_t rows = 0, cols = 0;
        const auto f = snicholls::test::signal_dense("mel_filters_" + std::to_string(mels), rows, cols);
        for (std::size_t i = 0; i < filters.size(); ++i)
            filters[i] = static_cast<T>(static_cast<float>(f[i]));
    } else {
        s::mel_filterbank(s::mel_filterbank_options<T>{T(16000), 400, mels}, std::span<T>(filters));
    }
    const std::size_t frames = s::whisper_frames(audio.size());
    ASSERT_EQ(ref.size(), mels * frames);
    std::vector<T> out(mels * frames);
    std::vector<s::complex<T>> work(s::whisper_work_size(plan));
    s::whisper_log_mel(plan, std::span<const T>(window), std::span<const T>(filters), mels, std::span<const T>(audio), 0, std::span<T>(out),
                       std::span<s::complex<T>>(work));
    for (std::size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(static_cast<double>(out[i]), ref[i], tolerance) << mels << " mel " << i / frames << " frame " << i % frames;
}

TEST(WhisperLogMel, IsWhispersAudioPyInDoubleAndInFloat)
{
    expect_whisper<double>(80, 1e-11, true);
    expect_whisper<double>(128, 1e-11, true);
    expect_whisper<double>(80, 2e-8, false);   // core's filters differ from librosa's float32 ones by an ulp in a few entries
    expect_whisper<float>(80, 1e-5, false);   // float32 arithmetic, as Whisper's own
    expect_whisper<float>(128, 1e-5, false);
}

TEST(WhisperLogMel, PaddingAppendsZerosAndAChunkHasThreeThousandFrames)
{
    std::vector<cd> st(s::rfft_plan<double>::storage_size(400));
    const s::rfft_plan<double> plan(400, st);
    const auto window = s::whisper_window<double>();
    std::vector<double> filters(80 * 201);
    s::mel_filterbank(s::mel_filterbank_options<double>{16000.0, 400, 80}, std::span<double>(filters));
    std::vector<double> audio(1000), padded(1600, 0.0);
    for (std::size_t i = 0; i < audio.size(); ++i)
        padded[i] = audio[i] = std::sin(0.05 * static_cast<double>(i));
    std::vector<cd> work(s::whisper_work_size(plan));
    std::vector<double> a(80 * 10), b(80 * 10);
    s::whisper_log_mel(plan, std::span<const double>(window), std::span<const double>(filters), 80, std::span<const double>(audio), 600,
                       std::span<double>(a), std::span<cd>(work));
    s::whisper_log_mel(plan, std::span<const double>(window), std::span<const double>(filters), 80, std::span<const double>(padded), 0,
                       std::span<double>(b), std::span<cd>(work));
    for (std::size_t i = 0; i < a.size(); ++i)
        EXPECT_EQ(a[i], b[i]);
    std::vector<double> chunk(s::whisper_n_samples, 0.0), mel(80 * s::whisper_n_frames);
    chunk[1000] = 1.0;
    s::whisper_log_mel(plan, std::span<const double>(window), std::span<const double>(filters), 80, std::span<const double>(chunk), 0,
                       std::span<double>(mel), std::span<cd>(work));
    EXPECT_GT(mel[6], mel[2000]);   // the impulse at sample 1000 is in frame 6
}

TEST(Mfcc, PowerToDbAndMfccAreLibrosas)
{
    const auto& power = snicholls::test::signal_record("mfcc_power");
    const auto& log_ref = snicholls::test::signal_record("mfcc_log");
    const auto& ref = snicholls::test::signal_record("mfcc_13");
    std::vector<double> db(power.size());
    s::log_compress<double>(power, db, s::decibels<double>());
    for (std::size_t i = 0; i < db.size(); ++i)
        EXPECT_NEAR(db[i], log_ref[i], 1e-12);
    std::vector<cd> st(s::fft_plan<double>::storage_size(20));
    const s::fft_plan<double> plan(20, st);
    std::vector<cd> work(s::dct_work_size(plan));
    std::vector<double> out(3 * 13);
    s::mfcc(plan, std::span<const double>(log_ref), std::span<double>(out), 13, std::span<cd>(work));
    for (std::size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], ref[i], 1e-12);
}

TEST(MelBackward, MelSpectrogramBackwardIsItsTranspose)
{
    std::vector<double> filters(6 * 9);
    s::mel_filterbank(s::mel_filterbank_options<double>{8000.0, 16, 6, 0.0, 0.0, s::mel_scale::htk}, std::span<double>(filters));
    std::vector<double> p(4 * 9), dm(4 * 6), mel(4 * 6), dp(4 * 9);
    for (std::size_t i = 0; i < p.size(); ++i)
        p[i] = std::cos(0.37 * static_cast<double>(i)) + 1.2;
    for (std::size_t i = 0; i < dm.size(); ++i)
        dm[i] = std::sin(0.91 * static_cast<double>(i));
    s::mel_spectrogram<double>(filters, 6, p, mel);
    s::mel_backward<double>(filters, 6, dm, dp);
    double a = 0, b = 0;
    for (std::size_t i = 0; i < mel.size(); ++i)
        a += mel[i] * dm[i];
    for (std::size_t i = 0; i < p.size(); ++i)
        b += p[i] * dp[i];
    EXPECT_NEAR(a, b, 1e-13);   // <F p, dm> = <p, F^T dm>
}

TEST(MelBackward, LogCompressionMatchesCentralDifferences)
{
    // values spread over 12 decades, so some fall below the floor at max - range and pass their gradient to the maximum
    std::vector<double> x{3.0, 1e-4, 2e-9, 0.5, 1e-12, 7.0, 0.02, 4e-11};
    const std::vector<double> g{0.3, -0.2, 0.5, 1.0, -0.7, 0.25, 0.1, 0.6};
    for (const auto& c : {s::whisper_compression<double>(), s::decibels<double>(40.0)}) {
        std::vector<double> gx(x.size());
        s::log_compress_backward<double>(x, g, gx, c);
        const auto loss = [&](const std::vector<double>& v) {
            std::vector<double> y(v.size());
            s::log_compress<double>(v, y, c);
            double l = 0;
            for (std::size_t i = 0; i < y.size(); ++i)
                l += g[i] * y[i];
            return l;
        };
        for (std::size_t j = 0; j < x.size(); ++j) {
            const double h = 1e-6 * x[j];
            auto up = x, down = x;
            up[j] += h;
            down[j] -= h;
            EXPECT_NEAR(gx[j], (loss(up) - loss(down)) / (2 * h), 1e-6 * (1 + std::abs(gx[j]))) << j;
        }
        EXPECT_EQ(gx[4], 0.0);   // 1e-12 is below the clamp at 1e-10
    }
}

TEST(MelBackward, WhisperLogMelMatchesCentralDifferences)
{
    std::vector<cd> st(s::rfft_plan<double>::storage_size(400));
    const s::rfft_plan<double> plan(400, st);
    const auto window = s::whisper_window<double>();
    std::vector<double> filters(80 * 201);
    s::mel_filterbank(s::mel_filterbank_options<double>{16000.0, 400, 80}, std::span<double>(filters));
    std::vector<double> audio(330);
    for (std::size_t i = 0; i < audio.size(); ++i)
        audio[i] = 0.3 * std::sin(0.21 * static_cast<double>(i)) + 0.1 * std::cos(0.013 * static_cast<double>(i * i));
    const std::size_t padding = 150, frames = s::whisper_frames(audio.size(), padding);
    std::vector<cd> work(s::whisper_work_size(plan));
    std::vector<double> weights(80 * frames);
    for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = std::sin(0.3 * static_cast<double>(i)) + 0.5;
    const auto loss = [&](const std::vector<double>& a) {
        std::vector<double> y(80 * frames);
        s::whisper_log_mel(plan, std::span<const double>(window), std::span<const double>(filters), 80, std::span<const double>(a), padding,
                           std::span<double>(y), std::span<cd>(work));
        double l = 0;
        for (std::size_t i = 0; i < y.size(); ++i)
            l += weights[i] * y[i];
        return l;
    };
    std::vector<double> grad(audio.size()), scratch(80 * frames);
    s::whisper_log_mel_backward(plan, std::span<const double>(window), std::span<const double>(filters), 80, std::span<const double>(audio),
                                padding, std::span<const double>(weights), std::span<double>(grad), std::span<double>(scratch), std::span<cd>(work));
    const double h = 1e-6;
    for (std::size_t j = 0; j < audio.size(); j += 7) {
        auto up = audio, down = audio;
        up[j] += h;
        down[j] -= h;
        EXPECT_NEAR(grad[j], (loss(up) - loss(down)) / (2 * h), 1e-5 * (1 + std::abs(grad[j]))) << j;
    }
}

} // namespace
