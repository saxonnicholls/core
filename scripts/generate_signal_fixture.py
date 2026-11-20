#!/usr/bin/env python3
#
#  generate_signal_fixture.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Writes tests/fixtures/signal/signal_fixture.txt: reference values for the tests of include/snicholls/core/signal/,
#  computed by numpy, scipy and librosa (independent implementations, never core's own output). Each line is a name
#  followed by numbers (%.17g; float32 values as %.9g, which round-trips them).
#
#  It needs numpy, scipy and librosa, which CI does not install, so unlike the time fixtures there is no CTest that
#  regenerates it; the versions it was written with are recorded in the file's header. To regenerate:
#
#      python3 -m venv venv && venv/bin/pip install numpy scipy librosa
#      venv/bin/python scripts/generate_signal_fixture.py
#
#  Whisper's log-mel spectrogram is recomputed here in float64 from the steps of whisper/audio.py
#  (log_mel_spectrogram): torch.hann_window(400) (periodic), torch.stft(n_fft=400, hop=160, center=True,
#  pad_mode='reflect'), the last frame dropped, |X|^2, librosa.filters.mel(sr=16000, n_fft=400, n_mels) in float32
#  (whisper's mel_filters.npz was saved from exactly that call), log10 of the clamp at 1e-10, the floor at max - 8,
#  and (x + 4) / 4.

import os
import sys

import numpy as np
import scipy
import scipy.fft
import scipy.signal
import librosa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "tests", "fixtures", "signal", "signal_fixture.txt")

lines = []


def put(name, values, single=False, fmt=None):
    fmt = fmt or ("%.9g" if single else "%.17g")
    flat = np.asarray(values).ravel()
    lines.append(name + " " + " ".join(fmt % float(v) for v in flat))


def put_complex(name, values):
    flat = np.asarray(values).ravel()
    put(name, np.column_stack([flat.real, flat.imag]))


def put_sparse(name, matrix, single=False):
    m = np.asarray(matrix)
    rows, cols = np.nonzero(m)
    fmt = "%.9g" if single else "%.17g"
    lines.append(name + " %d %d " % m.shape + " ".join("%d %d %s" % (r, c, fmt % float(m[r, c])) for r, c in zip(rows, cols)))


rng = np.random.default_rng(20261120)

# The FFT's conventions: numpy's norm modes on one sequence of 12 and a real one of 10.
x12 = rng.standard_normal(12) + 1j * rng.standard_normal(12)
put_complex("fft12_input", x12)
for norm in ("backward", "forward", "ortho"):
    put_complex("fft12_" + norm, np.fft.fft(x12, norm=norm))
    put_complex("ifft12_" + norm, np.fft.ifft(x12, norm=norm))
r10 = rng.standard_normal(10)
put("rfft10_input", r10)
put_complex("rfft10", np.fft.rfft(r10))
r11 = rng.standard_normal(11)
put("rfft11_input", r11)
put_complex("rfft11", np.fft.rfft(r11))
h6 = rng.standard_normal(6) + 1j * rng.standard_normal(6)
put_complex("irfft_input", h6)
put("irfft10", np.fft.irfft(h6, 10))
put("irfft11", np.fft.irfft(h6, 11))
a34 = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
put_complex("fft2_input", a34)
put_complex("fft2", np.fft.fft2(a34))

# Windows: scipy.signal.get_window, periodic (fftbins=True) and symmetric, 8 and 9 points.
windows = {
    "rectangular": "boxcar",
    "hann": "hann",
    "hamming": "hamming",
    "blackman": "blackman",
    "blackman_harris": "blackmanharris",
    "kaiser": ("kaiser", 8.6),
    "gaussian": ("gaussian", 1.7),
    "tukey": ("tukey", 0.4),
}
for name, spec in windows.items():
    for n in (8, 9, 1):
        for periodic in (True, False):
            put("window_%s_%d_%s" % (name, n, "periodic" if periodic else "symmetric"),
                scipy.signal.get_window(spec, n, fftbins=periodic))

# DCT and DST, types II to IV, scipy.fft's definitions, 7 and 8 points.
for n in (7, 8):
    v = rng.standard_normal(n)
    put("dct_input_%d" % n, v)
    for t in (2, 3, 4):
        for norm in (None, "ortho"):
            suffix = "%d_%d_%s" % (t, n, norm or "none")
            put("dct" + suffix, scipy.fft.dct(v, type=t, norm=norm))
            put("dst" + suffix, scipy.fft.dst(v, type=t, norm=norm))

# Convolution and correlation in every mode.
ca = rng.standard_normal(9)
cb = rng.standard_normal(4)
put("conv_a", ca)
put("conv_b", cb)
for mode in ("full", "same", "valid"):
    put("convolve_" + mode, scipy.signal.convolve(ca, cb, mode=mode, method="direct"))
    put("correlate_" + mode, scipy.signal.correlate(ca, cb, mode=mode, method="direct"))
    put("convolve_ba_" + mode, scipy.signal.convolve(cb, ca, mode=mode, method="direct"))

# FIR design: scipy.signal.firwin.
put("firwin_lowpass", scipy.signal.firwin(31, 0.3))
put("firwin_highpass", scipy.signal.firwin(31, 0.3, pass_zero=False))
put("firwin_bandpass", scipy.signal.firwin(31, [0.2, 0.5], pass_zero=False))
put("firwin_bandstop", scipy.signal.firwin(31, [0.2, 0.5]))
put("firwin_kaiser", scipy.signal.firwin(20, 0.25, window=("kaiser", 8.0)))

# Butterworth: the magnitude and phase of scipy.signal.butter(..., output='sos') at sample frequencies, and sosfilt.
w = np.linspace(0.01, np.pi - 0.01, 16)
put("butter_w", w)
designs = {
    "lowpass4": (4, 0.2, "lowpass"),
    "lowpass5": (5, 0.35, "lowpass"),
    "highpass3": (3, 0.4, "highpass"),
    "bandpass3": (3, [0.2, 0.45], "bandpass"),
    "bandstop2": (2, [0.1, 0.3], "bandstop"),
}
for name, (order, wn, btype) in designs.items():
    sos = scipy.signal.butter(order, wn, btype=btype, output="sos")
    _, h = scipy.signal.sosfreqz(sos, worN=w)
    put_complex("butter_" + name, h)
sig = rng.standard_normal(64)
put("sosfilt_input", sig)
put("sosfilt_lowpass4", scipy.signal.sosfilt(scipy.signal.butter(4, 0.2, output="sos"), sig))

# Polyphase resampling.
rx = rng.standard_normal(50)
put("resample_input", rx)
for up, down in ((3, 2), (2, 3), (1, 4), (5, 1), (160, 147)):
    put("resample_%d_%d" % (up, down), scipy.signal.resample_poly(rx, up, down))

# The analytic signal.
for n in (10, 11):
    hx = rng.standard_normal(n)
    put("hilbert_input_%d" % n, hx)
    put_complex("hilbert_%d" % n, scipy.signal.hilbert(hx))

# Mel scales and filterbanks: librosa.
f = np.array([0.0, 100.0, 440.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0, 11025.0])
put("mel_hz", f)
put("mel_slaney", librosa.hz_to_mel(f, htk=False))
put("mel_htk", librosa.hz_to_mel(f, htk=True))
put_sparse("mel_filters_80", librosa.filters.mel(sr=16000, n_fft=400, n_mels=80), single=True)
put_sparse("mel_filters_128", librosa.filters.mel(sr=16000, n_fft=400, n_mels=128), single=True)
put_sparse("mel_filters_htk", librosa.filters.mel(sr=22050, n_fft=512, n_mels=40, fmin=50.0, fmax=8000.0, htk=True,
                                                  norm=None, dtype=np.float64))

# Whisper's log-mel spectrogram of 0.06 s of a test signal, 80 and 128 mels, in float64 from float32 audio.
t = np.arange(960) / 16000.0
audio = (0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.25 * np.sin(2 * np.pi * 3000.0 * t + 0.3)
         + 0.05 * rng.standard_normal(960)).astype(np.float32)
put("whisper_audio", audio, single=True)
for n_mels in (80, 128):
    a = audio.astype(np.float64)
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)
    padded = np.pad(a, 200, mode="reflect")
    frames = 1 + (len(padded) - 400) // 160
    spec = np.stack([np.fft.rfft(window * padded[i * 160:i * 160 + 400]) for i in range(frames)], axis=1)
    mag = np.abs(spec[:, :-1]) ** 2
    filters = librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels).astype(np.float64)
    mel = filters @ mag
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    put("whisper_log_mel_%d" % n_mels, (log + 4.0) / 4.0, fmt="%.12g")

# MFCCs: librosa.feature.mfcc of a log-power mel spectrogram (20 mels, 3 frames), 13 coefficients.
S = np.abs(rng.standard_normal((20, 3))) ** 2 + 1e-3
logS = librosa.power_to_db(S)
put("mfcc_power", S.T)                              # frame-major
put("mfcc_log", logS.T)
put("mfcc_13", librosa.feature.mfcc(S=logS, n_mfcc=13).T)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as out:
    out.write("# signal fixtures from scripts/generate_signal_fixture.py: numpy %s, scipy %s, librosa %s, Python %d.%d\n"
              % (np.__version__, scipy.__version__, librosa.__version__, sys.version_info[0], sys.version_info[1]))
    for line in lines:
        out.write(line + "\n")
print("wrote %s (%d records)" % (OUT, len(lines)))
