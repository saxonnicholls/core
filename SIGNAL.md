# snicholls::signal - transforms, spectrograms and filters

Status: new in 0.17.0 (unreleased), experimental. Headers: `include/snicholls/core/signal/` (standard library and core
only), namespace `snicholls::signal`.

## Why

Audio and time-series models start with a front end (a short-time Fourier transform, mel filters, a log), and the
front end has to be the one the model was trained with, to the last detail of padding and normalisation. Filters,
resampling and spectral features belong in the same place, because they are the same arithmetic. Everything here
works in constant evaluation (a table, a plan, a whole transform can be a `static_assert`) and at run time over
memory the caller owns, with no allocation, no recursion and loops bounded by the length: the forms a GPU kernel or
an FPGA pipeline takes.

## Decisions

| Id | Decision |
|---|---|
| S1 | **Core's own `complex<T>`, not `std::complex`.** Two `T`s, real first, trivially copyable and structural; every operation `constexpr` in `float`, `double`, `long double` or a `math::math_number` of the caller's. `std::complex` is `constexpr` in C++20 only in part, recovers from infinities with calls that are not `constexpr` everywhere, and cannot take a caller's number type. The layout is `std::complex`'s and cuFFT's `cufftComplex`'s, so data is exchanged by copying. |
| S2 | **Stockham autosort for every smooth length, Bluestein for the rest, in-place radix 2 for a power of two known at compile time.** Stockham (radix 4 and 2 written out, a generic butterfly for the odd primes to 31) needs no bit reversal and reads and writes in natural order, which is what a GPU wants; Bluestein's chirp z-transform makes every length `O(n log n)`. `static_fft_plan<T, N>` of a power of two also holds the bit-reversal permutation and transforms in place with no second buffer, the FPGA form. |
| S3 | **Plans over caller memory; static plans by the compiler; work buffers from the caller.** `fft_plan<T>(n, storage)` fills `storage_size(n)` elements and refers to them; `static_fft_plan<T, N>` fills `std::array`s in a `consteval` constructor, so a fixed-size plan costs nothing at run time. A transform takes a work buffer of `work_size()` elements. The real transforms' building blocks take an accessor and a sink (`forward_from`, `inverse_to`), so a frame of a padded, windowed signal is transformed without a copy. |
| S4 | **numpy's and PyTorch's conventions.** `fft_norm::backward` (default), `forward`, `ortho`, and `none`; `rfft` gives `n / 2 + 1` bins and `irfft` ignores the imaginary parts of bin 0 and Nyquist; `stft`/`istft` take `torch.stft`'s options (centre padding by reflection or zeros, hop, normalised). The DCTs and DSTs are `scipy.fft`'s, the filters `scipy.signal`'s (`firwin`, `butter`, `sosfilt`, `resample_poly`, `hilbert`), the mel filters `librosa.filters.mel`'s. |
| S5 | **Gradients are adjoints, beside each transform, in PyTorch's convention.** For a real loss of a complex output the gradient is `dL/d re + i dL/d im`; the backward pass of a linear transform is its adjoint (the inverse direction with the forward's scale; for the half spectrum, the bins between 0 and Nyquist halved or doubled). `stft_backward` folds reflected samples back onto their sources; `whisper_log_mel_backward` recomputes each frame's spectrum rather than storing it. Every backward pass is checked against central differences. |
| S6 | **Accuracy is a stated bound, checked against long double.** `max_k |X_k - X*_k| <= 2 eps ceil(log2 L) ||x||_2` (`L` = `n`, or Bluestein's `m`), checked for every length from 1 to 300, the powers of two to 2^16 and primes to 10007 bin by bin, in `float` and `double`; measured, the worst case is a third of the bound. Twiddles are reduced to the first octant in integers, so quarter turns are exact and `twiddle(n - k)` is `conj(twiddle(k))` to the bit. |
| S7 | **Frame-major spectrograms; Whisper's own layout for Whisper.** `stft` writes `frames x bins` (each frame contiguous, the transpose of PyTorch's `(bins, frames)`); `whisper_log_mel` writes `n_mels x frames`, Whisper's tensor. |
| S8 | **Whisper's front end from whisper/audio.py.** Read for this work (`log_mel_spectrogram`): `padding` zeros appended, `torch.hann_window(400)` (periodic), `torch.stft(n_fft=400, hop=160, return_complex=True)` with PyTorch's defaults (`center=True`, `pad_mode='reflect'`), the last frame dropped, `abs() ** 2`, the filters of `mel_filters.npz` (saved, its docstring says, from `librosa.filters.mel(sr=16000, n_fft=400, n_mels=80 or 128)`), `clamp(min=1e-10).log10()`, `maximum(x, x.max() - 8)`, `(x + 4) / 4`. Checked against an independent float64 implementation of those steps in numpy with librosa's float32 filters, to 1e-11; core's own filters agree with librosa's float32 ones to an ulp. PyTorch's `center` and `pad_mode` defaults are recalled from its documentation, not read from Whisper's code. |
| S9 | **Butterworth by closed-form pole placement, one section per conjugate pair.** The prototype's poles `e^(i pi (2k + N + 1) / 2N)`, prewarped edges, the low-pass to band-pass and band-stop maps solved per pole, the bilinear transform per section. The cascade's response equals `scipy.signal.butter(..., output='sos')`'s to 1e-12; the split of the gain between sections differs (each low- or high-pass section has unit gain at DC or Nyquist), so `sosfilt` outputs agree with scipy's to rounding, not to the bit. |
| S10 | **Compile time within the compilers' default limits.** Clang's default (`-fconstexpr-steps`, about a million) admits, and the tests `static_assert`: a 1,024-point Parseval check (plan and transform), Bluestein at 17, Whisper's 80-mel filterbank, a one-frame Whisper log-mel spectrogram (libstdc++'s checked span indexing costs several times libc++'s in steps), a 63-tap FIR design, a sixth-order Butterworth design, the 3,201-tap 44.1-to-48 kHz resampling kernel (half computed, half mirrored, so it fits), a short STFT and its inverse. Larger tables raise the limit or build at run time with the same functions. |
| S11 | **What a device library adds.** Core's transforms are the reference and the host path. A GPU library (snicholls-gpu) adds batched transforms over many rows at once: either its own kernels (a Stockham pass per launch, or one shared-memory kernel per short row, reading core's tables, which are plain arrays built at compile time) or cuFFT / rocFFT / VkFFT behind a backend whose results are checked against core's within S6's bound. The STFT and mel front end map to one batched real transform per frame row, an elementwise power, and a matrix product with the filterbank (the existing `linalg` backend); the backward passes map the same way. Nothing in core needs to change for either. |

## Verification

| Header | Checked against |
|---|---|
| `fft.hpp` | The direct transform in `long double` (every length 1 to 300; powers of two to 2^16 and primes to 10007 bin by bin); numpy 2.0.2's `fft`/`ifft` in every `norm`, `rfft`, `irfft`, `fft2`; central differences for every backward pass. |
| `window.hpp` | `scipy.signal.get_window` 1.13.1, periodic and symmetric, eight windows. |
| `dct.hpp` | `scipy.fft.dct` / `dst`, types II to IV, `norm=None` and `"ortho"`; the definitions in `long double` for every length to 40. |
| `stft.hpp` | The direct transform of each padded, windowed frame; `torch.stft`'s frame count; `istft` round trips; central differences. |
| `mel.hpp` | librosa 0.11.0 (`hz_to_mel`, `filters.mel` for Whisper's 80 and 128 mels and an HTK bank, `power_to_db`, `feature.mfcc`); Whisper's steps as S8. |
| `convolve.hpp` | `scipy.signal.convolve` / `correlate` in every mode; the direct convolution for the FFT and block forms. |
| `fir.hpp` | `scipy.signal.firwin` (four bands, Hamming and Kaiser). |
| `iir.hpp` | The Audio EQ Cookbook's defining gains; `scipy.signal.butter` + `sosfreqz` and `sosfilt`; the closed-form Butterworth magnitude for orders 1 to 8. |
| `resample.hpp` | `scipy.signal.resample_poly` for five factors including 160/147. |
| `hilbert.hpp` | `scipy.signal.hilbert`, even and odd lengths; a modulated tone's envelope. |

The fixtures are `tests/fixtures/signal/signal_fixture.txt`, written by `scripts/generate_signal_fixture.py`, which
records the versions it ran with. It needs numpy, scipy and librosa, which CI does not install, so it is regenerated
by hand, never by a CTest.

## Deferred

- Radix-3 and radix-5 butterflies written out (lengths such as 1000 run the generic butterfly, about 3.5 times the
  cost per point of a power of two), split-radix, and SIMD.
- A real transform of half the length inside the DCTs, and two real blocks per complex transform in overlap-add and
  overlap-save.
- Complex-input STFT, the two-sided backward passes, a window shorter than `n_fft` (`win_length`), streaming STFT and
  resampling state.
- Chebyshev and elliptic IIR designs, Parks-McClellan FIR design, `filtfilt`, zero-phase and initial-condition
  (`sosfilt_zi`) helpers.
- DCT and DST of type I; a multi-dimensional real transform (`rfftn`).
