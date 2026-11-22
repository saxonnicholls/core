# Changelog

All notable changes to snicholls-core. The format follows Keep a Changelog; versions follow Semantic Versioning.
A version is released when its tag is on a commit where every blocking CI leg is green.

## 0.18.1 - 2026-11-22

The MSVC leg builds again. Nothing changes on the other compilers.

### Fixed

- The hashers (`crc32`, `crc32c`, `crc64_xz`, `sha1`, every SHA-2 and SHA-3, Keccak, RIPEMD-160, BLAKE2 and BLAKE3) have
  the documented size on MSVC. Their two empty bases, the `update` and one-shot `hash` mixins, now form one chain:
  MSVC gives no size only to the first of several empty bases side by side. The interface is unchanged.
- `ml/delta_rule.hpp`, `ml/gla.hpp`, `ml/ssd.hpp`: no warning C4189 under `/W4 /WX` for the shape constants that only
  an `assert` reads (MSVC ignores `[[maybe_unused]]` there).
- The test, demo and benchmarks that MSVC failed on the same grounds: `attention_layer_test`, `quant_k_demo` (float
  literals) and the `bpe_merges`, `token_trie` and `unigram` benchmarks (a local that shadowed another).

## 0.18.0 - 2026-11-21

### Added

- `stats/uniform_int.hpp`: `uniform_int<T, D>`, a drop-in for `std::uniform_int_distribution<T>` whose algorithm is
  a template parameter instead of whatever the standard library ships. `draw::core` (the default) is
  `uniform_index`; `draw::libstdcxx` reproduces libstdc++'s distribution (GCC 11 and later, 64-bit targets) and
  `draw::libcxx` libc++'s, to the bit, on every platform. A program can now agree with another program's seeded
  choices (llama.cpp's HellaSwag selection, a benchmark subset picked on Linux) wherever it is built. Checked draw
  for draw against the real libraries: libc++ under Apple Clang, libstdc++ under GCC 13, GCC 15 and Clang 18; 27
  committed sequences of 10,000 draws over `mt19937`, `mt19937_64`, `minstd_rand`, `ranlux24_base` and `wyrand`
  are checked on every platform. `constexpr` over a `constexpr` generator. MSVC's algorithm is not reproduced.

## 0.17.0 - 2026-11-20

Inference, quantisation, tokenizers, signal processing, linear-time sequence mixers and hashes, each a `constexpr`
building block over spans:

- serving and encoder operations with llama.cpp's semantics, checked against llama.cpp and ggml at commit 39039aa
  (INFERENCE.md I12);
- every ggml weight format byte for byte, with the importance-matrix quantisation and ggml's dot products, codebooks
  and tables built at compile time; BitNet b1.58's quantisation, kernels and quantisation-aware training; the Hadamard
  rotation as a quantisation pre-step (INFERENCE.md I13 to I15, TRAINING.md T14);
- tokenizers from text to token ids and back, token for token with llama.cpp on its own vocabulary test vectors
  (INFERENCE.md I16);
- signal processing at compile time and at run time: transforms of any length, windows, the short-time Fourier
  transform, mel front ends (OpenAI Whisper's among them), cosine transforms, convolution, FIR and IIR filters,
  resampling and the analytic signal, each with its backward pass (SIGNAL.md);
- linear-time sequence mixers (Mamba-2, gated linear attention, RWKV-6 and 7, the gated delta rule) and hybrid stacks
  (TRAINING.md T15);
- every hash that can be `constexpr` (SHA-1, SHA-2, SHA-3 and SHAKE, Keccak-256, RIPEMD-160, BLAKE2, BLAKE3, CRCs),
  HMAC, HKDF, PBKDF2 and scrypt, checked against their standards' vectors and an independent implementation.

Every new header is experimental.

### Added

- **Activations and gated units:** ELU, leaky ReLU, hard sigmoid, hard swish, quick GELU, xIELU, step and sign with
  their derivatives (`ml/activations.hpp`); `ml/glu.hpp`: ReGLU, GEGLU (tanh, erf, quick), SwiGLU, gpt-oss's clamped
  SwiGLU and SwiGLU with a clamp, per element, over rows and in ggml's split form, with backward. `mlp_kind` gains
  `gelu_quick` and the gated kinds (`gated(k)`, `mlp_glu`, `mlp_glu_backward`), run by the MLP, the mixture of experts
  and the blocks; SwiGLU keeps its bits. `transformer_config::glu_alpha` and `glu_limit`, one group whose defaults change
  no preset.
- **Norms:** `group_norm` (ggml's channel split, `group_channels`) and `l2_norm`, forward and backward (`ml/norm.hpp`).
- **Attention:** `ml/alibi.hpp` (slopes as ggml computes them, a compile-time table, `attend_alibi` and its backward);
  `ml/rope_modes.hpp` (ggml's NORM / NEOX names for core's layouts, M-RoPE and interleaved M-RoPE sections, vision RoPE,
  LongRoPE factors and attention factor, `rope_shift`); `ml/kv_cells.hpp` (positions and sequences per slot,
  `seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`, the K-shift `kv_shift_keys`, `attend_cells` with a window and
  ALiBi); `ml/kv_cache_quant.hpp` (Q8_0, Q4_0 and MXFP4 key and value rows, `attend_quantized`, the quantised K-shift).
- **Vision and audio encoders:** `ml/conv.hpp` (im2col / col2im and convolution in one to three dimensions with stride,
  padding, dilation and groups, transposed convolution, forward and backward); `ml/spatial.hpp` (pooling with
  backward, nearest, bilinear and bicubic upscaling, pad, reflect pad, roll, SAM's windows and relative-position bias);
  `ml/tensor_ops.hpp` (timestep embedding, arange, cumsum, argsort, top-k mask, triangular fill and solve).
- **Sampling:** `ml/samplers.hpp`: llama.cpp's candidate arrays and its samplers (greedy, dist, temperature, top-k,
  top-p, min-p, penalties, locally typical, dynamic temperature, XTC, top-n-sigma, mirostat v1 and v2, DRY, logit
  bias), every draw from `stats::wyrand`; `sampler_chain<S...>` at compile time and a run-time chain over a variant.
- **Constrained decoding:** `ml/grammar.hpp` (GBNF parsed at run time or compiled by `compile_grammar<"...">()` into
  fixed tables, the pushdown matcher, the token mask and bit mask with partial UTF-8 across tokens);
  `ml/json_schema.hpp` (a practical subset of JSON schema to GBNF, rule for rule as llama.cpp's converter).
- **Embeddings, rerankers, steering:** `ml/pooling.hpp` (mean, CLS, last, the RANK head); `ml/control_vector.hpp`.
- **Speculative decoding:** `ml/speculative.hpp` (n-gram prompt lookup, greedy verification, the rejection rule that
  preserves the target distribution, token trees with their mask, drafts from MTP heads).
- **ggml's type ids and block traits** (`ml/quantize.hpp`): `ggml_type` (the ids a GGUF file stores),
  `ggml_type_traits(id)` and `ggml_row_bytes` (name, block size and bytes of every type, at compile time), the
  `quantized_block` concept and `block_traits<B>` (elements, bytes, name, bits per weight, the activation block).
  Every block has `type`, `elements` and `dot_block`; `quantize_row(x, importance, blocks)` is ggml's
  `quantize_<type>` for one row (an empty importance row is the reference, a non-empty one the importance-matrix
  search), `dequantize_row` and `dot` its counterparts; `quantize(x, importance, blocks)` takes any rows.
- **The 32-element formats and the small ones** (`ml/quantize.hpp`): Q4_1, Q5_0, Q5_1 and Q8_1, the
  importance-weighted Q4_0 to Q5_1, NVFP4 (ggml's 64-element block of four UE4M3-scaled groups of 16), Q1_0, Q2_0,
  F8_E4M3 rows, and Q8_K (the super-block formats' activations), with their dot products; ggml's
  `make_qx_quants`, `make_qkx2_quants` / `make_qkx3_quants` and `make_qp_quants` scale searches, shared by the
  headers below.
- **K-quants** (`ml/quant_k.hpp`): Q2_K, Q3_K, Q4_K, Q5_K and Q6_K, reference and importance-weighted
  quantisation, dequantisation, and the dot products with Q8_K in ggml's eight-lane order.
- **The IQ family** (`ml/quant_iq.hpp`, `ml/quant_iq_grids.hpp`): IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S, IQ4_NL and IQ4_XS: ggml's importance-weighted quantisers, dequantisation and the dot products with Q8_K
  (IQ4_NL with Q8_0); the lattice grids, sign table and non-linear values built at compile time from ggml's compact
  index lists and hash-checked against ggml-common.h; the neighbour search computed on demand in ggml's order.
- **Ternary formats** (`ml/quant_ternary.hpp`): TQ1_0 and TQ2_0 with their dot products with Q8_K.
- **BitNet b1.58** (`ml/bitnet.hpp`): absmean ternary weights, per-token absmax activations (and a b-bit form),
  fake-quantised forwards, the straight-through estimator (identity and clipped) with its explicit backward,
  `ternary_dot`, `bitlinear` and `bitlinear_normed` (BitNet's SubLN placement). `ml/bitnet_kernels.hpp`:
  bitnet.cpp's I2_S (x86 and ARM blocks) and the TL1 / TL2 lookup-table layouts byte for byte (BitNet 0b341e5),
  with their tables built at compile time and reference GEMVs equal to the ternary product.
- **Hadamard rotation** (`ml/hadamard.hpp`): the fast Walsh-Hadamard transform, its orthonormal and block-diagonal
  forms bit for bit as ggml's CPU `GGML_HINT_SRC0_IS_HADAMARD` product, the randomised rotation of QuaRot and
  SpinQuant (`hadamard_signs`, `random_hadamard`, `random_hadamard_inverse`), folding into weights (`rotate_rows`),
  and the Sylvester matrix (`hadamard_entry`, `hadamard_matrix`, llama.cpp's KV-cache rotation bit for bit).
- **Fixtures and generators**: `scripts/generate_quant_formats_fixture.cpp`, `generate_quant_iq_fixture.cpp`,
  `generate_quant_ternary_fixture.cpp`, `generate_bitnet_i2s_fixture.cpp`, `generate_bitnet_lut_fixture.py`,
  `generate_hadamard_fixture.cpp` and `scripts/ggml_ref_stubs.c` (what ggml's quantisation files need from the rest
  of ggml), with the shared input rows `tests/ml/quant_rows.hpp`; outputs are hashed, so the fixtures stay small.
- **Unicode for tokenizers** (`ml/unicode.hpp`): strict UTF-8 `decode_utf8` / `encode_utf8`, `is_valid_utf8`,
  `replace_invalid_utf8`; `category` (every general category of Unicode 16.0) with `is_letter`, `is_mark`,
  `is_number`, `is_punctuation`, `is_symbol`, `is_separator`, `is_other` and GPT-4o's `is_uppercase_class` /
  `is_lowercase_class`; `is_whitespace` (White_Space); `to_lower`; `decompose` (full canonical decomposition, Hangul
  included) and `combining_class`. The tables (`detail/unicode_tables.hpp`) are generated by
  `scripts/generate_unicode_tables.py` from the committed extract `scripts/data/ucd-16.0.0.txt` of UnicodeData.txt,
  cross-checked against llama.cpp's tables; the CTest `generated.unicode_tables_up_to_date` fails if they drift.
- **Pre-tokenisers** (`ml/pretokenize.hpp`): GPT-2's, Llama 3's, Qwen 2's and 3.5's, GPT-4o's (o200k) and DeepSeek
  V3's split patterns, and the sequences of DeepSeek LLM, DeepSeek Coder, Falcon, StarCoder and Gemma 4, as
  hand-written `constexpr` matchers over categories with no regex engine; `split_isolated`, `split_sequence`,
  `pretokenize(kind, text, emit)`, `pretokenizer_for(gguf name)`, and `split_added_tokens` (leftmost-longest).
  Checked against Python's `regex` module running the original expressions (`scripts/generate_pretokenize_fixture.py`).
- **The vocabulary as a trie** (`ml/token_trie.hpp`): the ids sorted by piece, built by one sort
  (`build_token_index`, with a filter; `static_token_trie<N>` at compile time) and walked a byte at a time
  (`cursor`, `advance`, `at`, `find`, `longest_prefix`); `token_kind`, numbered as GGUF stores it.
- **Merge-list BPE** (`ml/bpe_merges.hpp`): `merge_encode`, merging by any priority over pairs (a merge list's rank,
  or SentencePiece's score) with byte fallback, in O(n log n); `merge_table`, `build_merge_index`, `split_merge`;
  GPT-2's `byte_to_unicode` / `unicode_to_byte` computed by the compiler, `byte_level_encode` / `byte_level_decode`.
- **SentencePiece BPE** (`ml/sentencepiece.hpp`): `sentencepiece_escape` (the dummy prefix),
  `sentencepiece_bpe_encode` (merges by score), `byte_fallback` / `make_byte_fallback` (`<0xNN>`), and
  `sentencepiece_decode`, shared with Unigram.
- **Unigram** (`ml/unigram.hpp`): `precompiled_charsmap`, SentencePiece's normalisation rules read in place from the
  blob GGUF and `tokenizer.json` carry; `unigram_normalize` (charsmap, user-defined pieces, the whitespace rules);
  `unigram_encode`, Viterbi over the vocabulary's log-probabilities with one unknown per run.
- **WordPiece** (`ml/wordpiece.hpp`): BERT's basic tokenisation (Hugging Face's rules: nonspacing marks stripped,
  `\p{C}` dropped, punctuation and CJK split off, U+0130's full lowercase) and greedy longest match, in Hugging
  Face's ("##") or GGUF's ("▁") spelling; `wordpiece_decode`.
- **RWKV** (`ml/rwkv_tokenizer.hpp`): `rwkv_encode` (greedy longest match) and `rwkv_unescape`.
- **The pipeline** (`ml/tokenizer.hpp`): `tokenize` and `detokenize` over `tokenizer_vocab`, the non-owning view a
  GGUF or `tokenizer.json` loader fills (pieces, scores, kinds, tries, merges, pre-tokenizer, flags, charsmap,
  byte fallback), for SentencePiece BPE, BPE (byte-level and Gemma 4's), Unigram, WordPiece and RWKV, with
  `tokenizer_scratch_need` sizing the caller's scratch.
- **Fixtures from llama.cpp**: `scripts/generate_tokenizer_fixtures.cpp` links llama.cpp and writes, for sixteen of
  its test vocabularies (Llama, Phi-3, GPT-2, Llama 3, Qwen 2, Qwen 3.5, DeepSeek LLM and Coder, Falcon,
  StarCoder, Refact, Command R, MPT, Gemma 4, BGE, nomic-embed), llama.cpp's tokens for its test strings and edge
  cases and the part of each vocabulary they reach (`tests/fixtures/ml/tokenizers/`). Core agrees on every one of
  llama.cpp's test strings; the differences recorded are invalid UTF-8 (SentencePiece's lead-byte cut, an encoded
  surrogate), Phi-3's "</s>" (which llama.cpp re-marks as a control token) and BERT's Hangul (Hugging Face's jamo).
- Benchmarks against the hand-written equivalents: `unicode`, `token_trie`, `pretokenize`, `bpe_merges`, `unigram`.
- **`signal/`, a new area (`snicholls::signal`, SIGNAL.md):**
  - `signal/complex.hpp`: `complex<T>`, trivially copyable and structural, every operation `constexpr`; `twiddle<T>(k, n)`
    reduced to the first octant in integers, so quarter turns are exact.
  - `signal/fft.hpp`: complex transforms of every length (Stockham radix 4 / 2 and odd primes to 31, Bluestein for the
    rest), forward and inverse with numpy's `norm` (`backward`, `forward`, `ortho`) and `none`; `fft_plan` over caller
    memory and `static_fft_plan<T, N>` built by a `consteval` constructor (with the bit-reversal table and an in-place
    radix-2 transform for a power of two); `fft(std::array)` / `ifft` on a plan the compiler builds; `rfft` / `irfft`
    with `rfft_plan` and `static_rfft_plan`; `fft_rows`, `fft2`, `fftn`; the direct `dft` as the reference; the
    backward passes `fft_backward`, `ifft_backward`, `rfft_backward`, `irfft_backward`, `fftn_backward`.
  - `signal/window.hpp`: rectangular, Hann, Hamming, Blackman, Blackman-Harris, Kaiser, Gaussian and Tukey windows,
    periodic or symmetric, as values, filled spans and `make_window` tables; `is_cola`, `is_nola`.
  - `signal/stft.hpp`: `stft` / `istft` with `torch.stft`'s options (centre padding by reflection or zeros, hop,
    normalised; one- and two-sided), `stft_frames`, `istft_length`, `spectrogram` (magnitude or power), `power`,
    `magnitude`, and the backward passes `stft_backward`, `istft_backward`, `power_backward`, `magnitude_backward`.
  - `signal/mel.hpp`: `hz_to_mel` / `mel_to_hz` (HTK, Slaney), `mel_filterbank` / `make_mel_filterbank`
    (`librosa.filters.mel`, Slaney area normalisation), `mel_spectrogram` and `mel_backward`, `log_compress` with
    `decibels` (librosa's `power_to_db`) and `whisper_compression`, and `log_compress_backward`; Whisper's front end,
    `whisper_log_mel` (80 or 128 mels, 3,000 frames per 30-second chunk) and `whisper_log_mel_backward`, with
    `whisper_window` and `whisper_mel_filters` as compile-time tables; `mfcc`.
  - `signal/dct.hpp`: `dct2`, `dct3`, `dct4`, `dst2`, `dst3`, `dst4` through the FFT, scipy's definitions,
    `dct_norm::ortho`; `dct2_backward`.
  - `signal/convolve.hpp`: `convolve` / `correlate` (direct) and `fft_convolve` / `fft_correlate` in scipy's `full`,
    `same` and `valid` modes; `overlap_add`, `overlap_save`.
  - `signal/fir.hpp`: `fir_design` / `make_fir` (scipy's `firwin`: low-, high-, band-pass, band-stop, any window),
    `fir_response`, `fir_filter`, `sinc`.
  - `signal/iir.hpp`: `biquad`, `biquad_state`, `rbj_biquad` (the Audio EQ Cookbook's eight filters), `butterworth` /
    `make_butterworth` as second-order sections (low-, high-, band-pass, band-stop), `sos_response`, `sosfilt`.
  - `signal/resample.hpp`: `resample_poly` (scipy's, Kaiser 5 windowed sinc), `resample_kernel`,
    `polyphase_resampler<T, Up, Down>` with its kernel built by the compiler.
  - `signal/hilbert.hpp`: `analytic_signal` (scipy's `hilbert`), `hilbert_transform`, `envelope`,
    `analytic_signal_backward`.
- `math::log10` and `math::bessel_i0` (`math/constexpr_math.hpp`).
- Reference fixtures from numpy 2.0.2, scipy 1.13.1 and librosa 0.11.0 (`tests/fixtures/signal/`,
  `scripts/generate_signal_fixture.py`); benchmarks `signal/fft_benchmark` and `signal/convolve_benchmark`.
- **Linear-time sequence mixers** (TRAINING.md T15), each with a forward pass over a batch of sequences, its backward
  pass for training, and a single step for decoding with a fixed-size state, every op on core's backend (T9) and
  checked against llama.cpp's CPU reference at commit 39039aa:
  - `ml/recurrence.hpp`: `recurrence`, a step and its backward pass, and the driver that gives every mixer its sequence
    forward with checkpoints at chunk starts (`recurrence_forward`), backpropagation through time recomputing each
    chunk from its checkpoint (`recurrence_backward`, memory `n / chunk + chunk` states), and decoding;
    `recurrence_checkpoints`, `recurrence_backward_work`.
  - `ml/short_conv.hpp`: Mamba's short causal depthwise convolution (`GGML_OP_SSM_CONV`, to the bit), its state carried
    between pieces of a sequence and into decoding, its backward pass (taps, bias, inputs, initial state).
  - `ml/ssd.hpp`: Mamba-2's selective state space, the scan (`GGML_OP_SSM_SCAN`'s scalar-A path) and the chunked SSD
    form (quadratic within a chunk, the state passed between chunks), equal within rounding; the backward pass from
    either form's checkpoints; `ssd_step` for decoding.
  - `ml/gla.hpp`: gated linear attention with per-channel log-gates (`GGML_OP_GATED_LINEAR_ATTN`, to the bit), recurrent
    and chunked, its backward pass and step.
  - `ml/rwkv.hpp`: token shift (forward, backward, step); RWKV-6's WKV with its bonus and RWKV-7's generalised delta rule
    (`GGML_OP_RWKV_WKV6` and `_WKV7`, to the bit), their backward passes and steps.
  - `ml/delta_rule.hpp`: the gated delta rule (`GGML_OP_GATED_DELTA_NET`, a gate per head as Qwen3-Next or per key
    channel as Kimi), recurrent and chunked through the WY / UT representation with the backend's unit lower-triangular
    `triangular_solve`, its backward pass and step; `l2_normalise_rows` and its backward pass.
  - `ml/mixer_layer.hpp`: Mamba-2's layer (projection, convolution, `softplus` steps, SSD, the `D` skip, the SiLU gate,
    the grouped RMS norm) and the gated delta rule's (Qwen3-Next's: convolution, L2-normalised queries and keys, `β`
    and the gate, the gated per-head norm), batched with their backward passes and one token at a time; their buffers'
    sizes and layouts (`mixer_parameter_count`, `mixer_activation_values`, `mixer_gradient_work`, `mixer_state_values`,
    `mixer_weights_in`, `mixer_activations_in`, `mixer_gradients_in`); `recurrent_cache`, `hybrid_cache`,
    `hybrid_attention_index`, `hybrid_mixer_index`.
  - `ml/composed_block.hpp`: a mixer as the composed block's attention part, so hybrid stacks (Qwen3-Next's three
    gated-delta-rule layers then attention, Jamba's one attention layer in eight) train through `block_forward_sequence`
    and `block_backward` and decode through `block_forward` with a hybrid cache.
  - `ml/transformer_config.hpp`: the mixer fields as one group with neutral defaults (`mixer`, `mixer_heads`,
    `mixer_head_dim`, `mixer_state`, `mixer_groups`, `conv_width`, `mixer_chunk`, `hybrid_period`,
    `hybrid_attention_offset`), `mixer_kind`, `valid_mixer`, `mixer_state_size`, `layer_is_attention`, `mixer_scale_v`,
    and the presets `tiny_mamba2` and `tiny_gated_deltanet`.
  - Benchmarks of every recurrence against loops by hand (`benchmarks/ml/ssd_benchmark.cpp`, `gla_benchmark.cpp`,
    `rwkv_benchmark.cpp`, `delta_rule_benchmark.cpp`, `short_conv_benchmark.cpp`), each pair checked to the bit.
- `hash/digest.hpp`: `digest<N>`, the value every hash returns (N bytes, structural, trivially copyable, `==` and
  `<=>`, `hex()` without allocating, `from_hex`, `first<M>()`, printable); `hash_input`, any one-byte sequence under
  core's string rules or a digest; `hash_function`, the interface `hmac` and the KDFs are written against; and
  `md_hash<Policy>`, the Merkle-Damgard engine behind the next three.
- `hash/sha1.hpp`: `sha1`, public (it was `detail/sha1.hpp`, behind `uuid`), documented as broken for collisions.
- `hash/sha2.hpp`: `sha224`, `sha256`, `sha384`, `sha512`, `sha512_224`, `sha512_256` (FIPS 180-4).
- `hash/sha3.hpp`: `sha3_224` ... `sha3_512`, `shake128` / `shake256` (squeezed in any pieces), Ethereum's
  `keccak256` (padding `0x01`), all on one `keccak_f1600`, which is public.
- `hash/ripemd160.hpp`: `ripemd160`, for Bitcoin's `hash160`.
- `hash/blake2.hpp`: `blake2b<N>` and `blake2s<N>` (RFC 7693), keyed or not, any output length.
- `hash/blake3.hpp`: `blake3`: hash, keyed hash, key derivation (`derive_key`, `key_derivation`) and extendable
  output from any offset, the tree mode on one thread.
- `hash/crc.hpp`: `crc32` (zlib's), `crc32c` (Castagnoli), `crc64_xz`, slicing-by-8 tables built by the compiler,
  `resume(crc)` for zlib's running CRC.
- `hash/hmac.hpp`: `hmac<H>` over any hash function (RFC 2104), with `verify` in constant time.
- `hash/kdf.hpp`: `hkdf_extract`, `hkdf_expand`, `hkdf` (RFC 5869) and `pbkdf2_hmac` (RFC 8018), into the caller's
  span or as a `digest<L>`, failing with `result<void, kdf_error>`.
- `hash/scrypt.hpp`: `scrypt` (RFC 7914) in working memory the caller provides (`scrypt_work_words`), wiped before
  it returns.
- `hash/secure.hpp`: `secure_wipe`, which the optimiser may not remove, and `constant_time_equal`.
- Tests against every published vector at hand: NIST's byte-oriented SHA-1, SHA-2, SHA-3 short-message and Monte
  Carlo vectors and SHAKE Monte Carlo vectors (`tests/fixtures/hash/nist/`, unmodified), RFC 4231, 2202, 5869,
  6070, 7693 and 7914, BLAKE3's official `test_vectors.json`; Python's hashlib, hmac and zlib, PyCryptodome, the
  `blake3` package and crcmod at every block boundary and on random lengths to 10,000
  (`scripts/generate_hash_fixtures.py`, checked by the CTest `generated.hash_fixtures_up_to_date`); every split of
  an input; digests computed at compile time.
- Benchmarks of every algorithm against a hand-written loop, with OpenSSL's EVP beside them when it is installed
  (informational; `SNICHOLLS_CORE_BENCH_OPENSSL`).

### Changed

- `quant_block` (what `quantized_weights` in `ml/linear.hpp` takes) is now any 32-element `quantized_block` paired
  with Q8_0 activations, so Q5_0 and IQ4_NL weights work there too. The per-block `detail` functions became the
  public `quantize_row` / `dequantize_row`.
- `ml/quantize.hpp` gives ggml's x86 result where ggml converts a float out of an integer's range (the inverse scale
  of a block of subnormals; an MXFP4 amax below 2^-125), so those blocks are byte-identical too.
- `secret<std::string>` wipes its string with `secure_wipe`; `make_uuid_v5` hashes with the public `sha1`.

### Removed

- `detail/sha1.hpp` (internal): its one user, `uuid`, uses `hash/sha1.hpp`.

### Notes

- llama.cpp's `seq_div` stores its K-shift with the opposite sign of `seq_add`'s; core's moves keys to their new
  positions (`ml/kv_cells.hpp`, Known issues; recorded in `erratum/llama.cpp.md`).

## 0.16.0 - 2026-11-19

Training primitives for mixture-of-experts models, DeepSeek-V3's attention and block, joint embedding predictive
architectures and post-training, each a reference implementation on core's backend (TRAINING.md T9) that a device
library can overload and is checked against; bfloat16 pseudo-gradients for DiLoCo; and GCC 15 in the local gate.
Every new header is experimental.

### Added

- **Mixture of experts for training** (TRAINING.md T10), every step a backend call over rows (T9) whose reductions run
  in a stated order, so a device library's overloads can match core's reference and the combine is deterministic:
  - `ml/moe.hpp`: `moe_route_rows` and its backward (top-K over the router logits with a selection-only bias; scoring
    by `moe_scoring::softmax_top_k` as Mixtral, `softmax` as Switch and OLMoE, or `sigmoid` normalised and scaled as
    DeepSeek-V3; ties to the lower expert; `moe_gate` row by row to the bit); `moe_count_rows`, `moe_offsets` and
    `moe_permutation` (the unique stable order by expert, row and slot, and its inverse); `moe_gather_rows`,
    `moe_combine_rows` and their backward passes; `grouped_weights`, `grouped_linear` and `grouped_linear_backward`
    (each expert's segment times its own matrix); `moe_experts_forward` / `moe_experts_backward` (the experts' GELU,
    ReLU² or SwiGLU MLPs over the permuted rows, the dense MLP's activation steps); shared experts as one dense MLP
    (`moe_shared_config`); DeepSeek-V3's bias rule `moe_update_bias` (the sign decided in integers), the auxiliary
    balance loss `moe_balance_loss` (batch-wise or sequence-wise) and ST-MoE's router z-loss `moe_z_loss`, with
    their gradients and a denominator for a worker holding part of a batch; `moe_load<E>`, load statistics that merge
    across workers; and the layer, `moe_forward` / `moe_backward`, with `moe_weights`, `moe_activations`,
    `moe_gradients` sized by `moe_parameter_count`, `moe_activation_values`, `moe_activation_indices`,
    `moe_gradient_work` and laid out by `moe_weights_in`, `moe_activations_in`, `moe_gradients_in`.
  - `ml/moe_block.hpp`: the batched `block_forward_sequence` and `block_backward` for a block whose MLP is a mixture of
    experts (`moe_block_weights`, `moe_block_activations`, `moe_block_gradients`), returning the auxiliary losses;
    with one expert it is the dense block to the bit.
  - `ml/expert_parallel.hpp`, expert parallelism's arithmetic: contiguous placement (`experts_per_rank`,
    `first_expert`, `expert_rank`), `moe_send_offsets`, `moe_receive_offsets`, `moe_receive_layout`,
    `moe_permute_rows` / `moe_unpermute_rows`, and `moe_all_to_all`, a reference all-to-all over in-process ranks
    that fixes the order of every received row. One, two and four ranks give the single-process layer's outputs,
    input gradients and expert gradients to the bit.
  - `transformer_config`'s expert fields (`experts`, `top_k`, `shared_experts`, `expert_d_ff`, `scoring`,
    `route_scale`, `bias_update_rate`, `balance_loss_weight`, `z_loss_weight`, read through `route_scale_v` and the
    other readers), `moe()`, `valid()` extended; every existing preset is unchanged. New presets: `tiny_moe` and
    `olmoe_1b_7b`.
  - `ml::to_bfloat16_rows` and `ml::to_float_rows` (`ml/minifloat.hpp`): rows converted on the backend, for a
    payload sent in half the bytes.

- **Joint embedding predictive architectures** (TRAINING.md T11): every primitive to train I-JEPA, V-JEPA / V-JEPA 2 and
  LLM-JEPA on core's reference backend, each a backend call a device library can overload.
  - Attention masks: `ml::attention_mask` (`causal`, `bidirectional`, `block_causal`) and `ml::attention_keys`
    (`ml/attention.hpp`); `transformer_config::mask` and `mask_block`, after the existing fields with defaults that
    change no preset, and the preset `ml::vit_base`; `attention_activations::lengths`, a key-padding mask per
    sequence; a rope source with `sequence(s)` gives each sequence of a batch its own positions; `self_attention_head`
    and its backward take a mask and a block.
  - `ml/patch.hpp`: `patch_grid`, `patchify_rows` / `unpatchify_rows` for images, video tubelets and spectrograms;
    2-D and 3-D sine-cosine position embeddings (functions and `constexpr` tables); `axial_rope_table`;
    `gathered_rope`.
  - `ml/masking.hpp`: I-JEPA's multi-block masks, V-JEPA's tubes (`vjepa_short_range`, `vjepa_long_range`), random
    masks, deterministic from (seed, step, sample) through `stats::wyrand` streams; `common_count` and
    `collate_indices` (truncation, or padding with lengths).
  - `ml/gather.hpp`: `gather_rows`, `scatter_add_rows` (fixed order), `scatter_rows`, `concat_rows`, `slice_rows`
    and its backward, over a batch of sequences.
  - `ml/jepa.hpp`: `jepa_predictor_config` with `block_parameter_count` and `jepa_predictor_parameters`, the preset
    `ijepa_vit_h14_predictor`; `jepa_predictor_input_rows` and its backward; `jepa_target_norm_rows`;
    `jepa_loss_rows` (L2, L1, smooth L1, cosine) with gradients in both arguments and padding; `ema_rows`,
    `ema_segments` and `ema_momentum` (linear, cosine).
  - `ml/collapse.hpp`: `embedding_moments` (mergeable mean, co-moment, standard deviation, covariance, effective
    rank); VICReg's variance and covariance terms with gradients; LeJEPA's SIGReg (`sigreg_quadrature`,
    `sigreg_directions`, `sigreg_moments_rows`, `sigreg_value`, `sigreg_backward_rows`, `sigreg_rows`), its sums
    merging across workers.
  - `ml/probe.hpp`: `mean_pool_rows` and V-JEPA's attentive probe, forward and backward.

- **Latent attention, multi-token prediction and long context**, DeepSeek-V3's attention side as references (TRAINING.md
  T12):
  - `ml/latent_attention.hpp`: multi-head latent attention (DeepSeek-V2 and V3), queries and keys/values from RMS-normed
    low-rank latents with decoupled RoPE, in DeepSeek-V3's weight layout. `ml::latent_attention_forward_sequence` and
    `ml::latent_attention_backward` over a batch of sequences through a `linalg` backend, their non-product steps
    backend calls; `ml::latent_kv_cache_view` / `ml::latent_kv_cache` holding only `[c^KV; k^R]`;
    `ml::latent_attention_forward`, the absorbed decode, equal to the training form within rounding;
    `ml::kv_cache_elements_per_token` and `ml::kv_cache_elements` as constant expressions; `ml::latent_attention_scale`.
  - `ml/latent_block.hpp`: `ml::latent_block_weights` and overloads of `block_forward`, `block_forward_sequence` and
    `block_backward` for a block whose attention is MLA.
  - `ml/multi_token_prediction.hpp`: DeepSeek-V3's MTP modules: `ml::mtp_combine_rows` and its backward pass,
    `ml::mtp_forward_sequence`, `ml::mtp_backward`, `ml::mtp_loss_sequence` through the shared head, `ml::mtp_input_ids`.
  - `ml/attention_sink.hpp`: `ml::attend_with_sink` and its backward pass (gpt-oss's learnable sink logit, keys and
    values of different widths), `ml::attention_sink_weight`.
  - `ml/sparse_attention.hpp`: `ml::attend_selected` and its backward pass; DeepSeek Sparse Attention's lightning
    indexer, `ml::indexer_scores` and its backward pass, `ml::select_top_k`, `ml::indexer_kl_loss`.
  - `ml/rope.hpp`: long-context rescaling, `ml::rope_scaling` (position interpolation, NTK-aware, YaRN as its paper and
    as its code and DeepSeek-V3 compute it), `ml::rope_scaled_frequency`, `ml::yarn_ramp`, `ml::yarn_correction_dim`,
    `ml::yarn_attention_factor`, and `rope_angles` / `rope_table` overloads taking a rescaling.
  - `ml::transformer_config`: one group of fields for MLA, RoPE rescaling, sinks, sparse selection and MTP, every default
    leaving a model unchanged (`ml::valid_long_context`); presets `ml::deepseek_v3` and `ml::tiny_mla`; readers
    `rope_factor_v`, `mtp_loss_weight_v`, `rope_scaling_v`, `yarn_attention_factor_v`.
  - `ml::lm_head_loss_sequence` takes a `weight` and a `denominator` (defaults: the mean, to the bit), for a loss that
    is a part of a larger one.
  - `ml/composed_block.hpp`: `ml::composed_block_weights`, `composed_block_activations` and `composed_block_gradients`,
    one pre-norm block whose attention (standard or latent) and MLP (dense or a mixture of experts) are chosen by their
    types, with batched `block_forward_sequence` (returning `moe_losses`, the sequences the balance loss's groups),
    `block_backward` and decoding `block_forward`. DeepSeek-V3's block as `ml::latent_moe_block_weights`,
    `_activations` and `_gradients`, sized by `latent_moe_block_parameter_count`, `_activation_values`,
    `_activation_indices` and `_gradient_work`, laid out by `latent_moe_block_weights_in`, `_activations_in` and
    `_gradients_in`. With one expert of weight 1 it is the latent block to the bit; `deepseek_v3` with V3's 61 blocks
    and vocabulary gives the paper's 671B parameters and 37B active (`static_assert`s).
  - `ml/latent_attention.hpp`: `ml::latent_attention_parameter_count`, `latent_attention_activation_values`,
    `latent_attention_gradient_work` and the layouts `latent_attention_weights_in`, `_activations_in`,
    `_gradients_in`.
  - DeepSeek-V3's node-limited routing: `transformer_config::expert_groups` and `group_top_k` (0 and 0 by default, so
    no other preset changes; `deepseek_v3` now 8 groups, 4 kept), honoured by `ml::moe_route_rows` and everything
    above it; one group, or every group kept, is the plain choice to the bit.
  - `ml::mlp_forward` overload for `moe_weights`: the mixture of experts for one token, for decoding.
  - `ml::mtp_forward_sequence` returns its block's auxiliary losses when the block is a mixture of experts
    (`moe_block_weights`, `latent_moe_block_weights`); `ml::mtp_backward` includes their gradients.

- **Post-training** (TRAINING.md T13): every objective a function of per-token log-probabilities, each batch a
  backend call over rows (T9), masks and rollouts plain data, losses a parameter struct and an overload set.
  - `ml/logprobs.hpp`: `token_logprob` (`-cross_entropy` to the bit), `token_logprob_with_gradient`,
    `token_entropy`, `token_logprobs_rows` and its fused backward (log-probability and entropy gradients, the logits
    their own buffer), `sequence_logprobs` / `sequence_logprobs_backward` (masked sum or mean over the response).
  - `ml/preference.hpp`: `preference_loss` for `dpo`, `ipo`, `simpo` and `orpo`, each with its gradient and implicit
    rewards; `preference_loss_rows`; `preference_metrics` (loss, rewards, margin, accuracy), mergeable.
  - `ml/reward_model.hpp`: the Bradley-Terry loss (`bradley_terry{margin}`), a scalar value head and its backward,
    `last_response_positions` (the index list `gather_rows` reads).
  - `ml/rl.hpp`: GRPO's `group_advantages` (or Dr. GRPO's centring), `expand_rows`, the k3 KL estimator, the clipped
    objective with DAPO's asymmetric clip and an entropy bonus, `policy_loss_rows` with the aggregation stated
    (`sequence_mean` GRPO, `token_mean` DAPO, `constant` Dr. GRPO), `policy_loss_metrics`; PPO's `gae_rows`,
    `value_loss_rows`, `masked_whiten`; `rollout_view`.
  - `ml/distill.hpp`: forward and reverse KL at a temperature with the student's gradient (`distill_loss`,
    `distill_rows`), and the top-k teacher form (`teacher_topk`, `distill_loss_topk`, `distill_topk_rows`).
  - `ml/nf4.hpp`: QLoRA's NormalFloat4 with double-quantised scales (E4M3 in groups of 256 after the mean),
    `nf4_quantize` / `nf4_dequantize`, and products dequantising on the fly equal to the dense product of the
    dequantised matrix to the bit.
  - `ml/lora.hpp`: `lora_weights` over a dense, bfloat16 or NF4 base; `lora_linear`, `lora_sequence` and `lora_heads`
    with their backward passes (gradients to A and B only); `lora_initialise` (B zero, A Kaiming-uniform or Gaussian
    from `stats::wyrand`); `lora_merge` / `lora_unmerge`; `lora_parameters`, the optimiser's view of the adapter.

- **bfloat16 pseudo-gradients** in `ml/diloco.hpp` (TRAINING.md T5), half the bytes on the wire, specified to the bit so that a transport
  and a training library agree:
  - `ml::compress_bf16(delta, residual, out)`: `delta + residual` rounded to bfloat16 (to nearest, ties to even), with
    error feedback: the residual keeps exactly what the rounding dropped and is added to the next round's delta.
  - `ml::ordered_sum_bf16(parts, out)`: the workers' bfloat16 parts decoded and summed in rank order in float, the
    reducing rank's arithmetic.
  - `ml::ordered_mean_bf16(parts, out)`: what a bfloat16 all-reduce and the mean leave on every worker, the float sum
    rounded once to bfloat16 and divided by the worker count.
- The documentation of `ml/diloco.hpp` states what a worker that leaves and rejoins needs (nothing new in the outer
  step: the mean divides by that round's parts).

- **GCC 15 in the Ubuntu gate**: `scripts/test_ubuntu.sh` builds a second image, `docker/ubuntu-26.04` (GCC 15 from
  the Ubuntu archive), and runs GCC 15 Release with benchmark smoke runs and GCC 15 ASan+UBSan beside the GCC 13 and
  Clang 18 configurations: eight in all. GCC 15 is what CI's x86-64 GCC leg uses, and it warns where GCC 13 does not.
  `--quick` runs GCC 13 Release and GCC 15 Release; a failed configuration shows its first diagnostics above the
  log's tail.

### Changed

- Documentation: `ml::moe_update_bias` may also be called after every micro-batch with that micro-batch's counts, on
  one worker or within a DiLoCo round (whose end averages the biases), which undoes the early collapse of routing as
  many times faster as there are micro-batches a step. Measured: with 16 micro-batches a step, a 64-expert model had
  every expert within 0.5 to 2 times the mean load by step 148, where updating once a step had not settled in 300
  steps, at no cost in loss.
- `attention_forward_sequence` (the per-token path) projects every token before any attends, so a bidirectional or
  block-causal layer reads later keys; a causal layer's bits are unchanged.

### Fixed

- The Windows leg builds again: a newer MSVC on the runner reported constants that only a local struct's member
  initialisers read in the attention layer and transformer block tests as unused (C4189); they are marked
  `[[maybe_unused]]`. What the tests check is unchanged.

## 0.15.0 - 2026-11-19

### Added

- `ml/diloco.hpp`, the arithmetic of data-parallel training across machines (TRAINING.md T5):
  - `ml::ordered_sum(parts, out)`: the workers' parts summed left to right in rank order, in the value's own
    arithmetic, so a run does not depend on which part arrives first or on the cluster's topology; the reference any
    all-reduce reproduces to the bit. `ml::ordered_mean` divides it by the worker count.
  - `ml::pseudo_gradient(global, local, out)` and `ml::diloco_outer_step(global, mean_delta, momentum, config)`:
    DiLoCo's outer optimiser (Douillard et al., 2023), SGD with Nesterov momentum on the mean of the workers'
    weight changes after their local steps; `ml::diloco_config` defaults to the paper's outer learning rate 0.7 and
    momentum 0.9. The transport belongs to ts-moveables, the device kernels to snicholls-gpu.

## 0.14.0 - 2026-11-18

### Added

- `ml::gpt2_medium` and `ml::gpt2_large` (`ml/transformer_config.hpp`): the shapes of GPT-2 medium (1024 wide, 16
  heads) and large (1280 wide, 20 heads), beside `gpt2_small`.

- `ml::adam_bias_correction(beta, step)` (`ml/optim.hpp`): Adam's `1 - β^t` as `-expm1(t · log1p(-(1 - β)))`, the
  factor `adamw_step` and `adamw_rows` divide by, public so that a device library's AdamW kernel computes the same
  two factors on its host side.

### Fixed

- **Accuracy of the ML formulas** (found by certifying them against symbolic derivatives; no formula was
  wrong, each lost digits to rounding):
  - `silu_derivative`, `gelu_tanh_derivative` and `sigmoid_derivative` (`ml/activations.hpp`) take `1 - sigmoid(z)`
    as `sigmoid(-z)`, from the same `e^-|z|`, instead of subtracting a sigmoid near 1 from 1. The worst error of
    SiLU' and GELU-tanh' over `|x| <= 16` falls from 7.2 and 16.5 units in the last place to under 2 in float (11.5 and
    43.7 in double before); `sigmoid_derivative(20.0f)` is no longer 0.
  - `log_softmax`, `cross_entropy` and `cross_entropy_with_gradient` (`ml/softmax.hpp`) take each difference from
    `logsumexp` as `(x_i - max) - log(sum)`, so a result of order 1 does not carry the rounding of a logit of order
    1e4: with logits near 1e4, the loss, the log-probabilities and the gradient are within 4 units in the last place,
    where they lost about 90 in float and 600 in double. The fused gradient's softmax is now `softmax`'s to the bit.
    `logsumexp` itself is unchanged.
  - `adamw_step` and `adamw_rows` (`ml/optim.hpp`) compute the bias correction `1 - β^t` as
    `-expm1(t · log1p(-(1 - β)))`: `1 - pow(β, t)` cancelled for `β` near 1 in the first steps (at `β2 = 0.9999`,
    t = 3, 830 units in the last place in float, and the update off by 415; t = 1 was exact and still is to rounding).
  - The documented range of Muon's five Newton-Schulz steps (`ml/optim.hpp`, `ml/muon.hpp`) is about
    `[0.682, 1.2024]` from a singular value of at least `1.55e-3`, not `[0.7, 1.2]`; below that a value `x` reaches
    only about `a⁵ x`. Documentation only.

  Results change in the last bits; the hand-written equivalents in the benchmarks compute the same new forms.
- The `lm_head` demo converts its targets to `std::uint32_t` explicitly, which MSVC's C4267 required: the Windows leg
  builds again (it had failed since 0.10.0).

## 0.13.0 - 2026-11-17

### Added

- **A batch of sequences in one call** (`ml/transformer_block.hpp`, `ml/attention_layer.hpp`, `ml/linear.hpp`): the
  batched `block_forward_sequence` and `block_backward`, `attention_forward_sequence` and `attention_backward`, the four
  attention steps (`attention_position_rows`, `attention_attend_rows` and their backward passes) and `linear_heads` /
  `linear_heads_backward` take a trailing `sequences` (default 1). The rows are that many sequences of `n` tokens end
  to end: every row-wise step and every product runs once over all of them, and attention stays within each sequence,
  positions restarting at 0. The activations are the sequences' own side by side, sequence-major (the keys and values
  `sequences` blocks of `n_kv_heads · n · head_dim`), so one sequence's are a slice. On the reference backend a batch
  gives the per-sequence calls' results to the bit (outputs, activations, the input's gradient, the biases', norms'
  and `Wk`'s and `Wv`'s gradients); the gradients of `Wq`, `Wo` and the MLP's matrices, each one product over every
  row, agree within rounding. One sequence is the existing call, unchanged to the bit.

### Changed

- **Muon on the backend** (`ml/muon.hpp`, TRAINING.md T9): every step of `muon_step` and `muon_orthogonalise` that
  is not a product is now a call on the backend (`copy_rows`, `scale_rows`, `add_rows`, `sum_squares`), and a tall
  matrix is iterated with `A = Xᵀ X` instead of being transposed, so a device library's overloads run a whole Muon
  step over device memory (snicholls-gpu's `cuda_training_backend`), as `adamw_rows` does. The API and the working
  space are unchanged, and through the reference backend the results are 0.12.0's to the bit (tested against its loops,
  float and double, wide, tall and square, with and without Nesterov).

## 0.12.0 - 2026-11-17

### Added

- **A padded head** (`ml/lm_head.hpp`, `ml/softmax.hpp`): `lm_head_loss_sequence(..., classes)` and
  `cross_entropy_rows(..., classes)` take logits rows wider than the classes, the padding not a class and its gradient
  0, as llm.c lays out GPT-2's 50,257 tokens in 50,304 rows so that the head's product suits the tensor cores. With
  every row a class, the results are the unpadded ones to the bit; a padded head gives the unpadded head's loss and
  gradients to the bit, and nothing to the padding's rows.

## 0.11.0 - 2026-11-17

### Added

- **`linear_heads` and `linear_heads_backward`** (`ml/linear.hpp`): `linear_sequence` with the outputs laid out head
  by head, as attention keeps its keys and values. On the host each head is its own `linear_sequence`, so results are
  unchanged to the bit; a device library overloads them for its backend to make one product for every head (with
  snicholls-gpu's CUDA backend, one product and a permutation instead of a product per head).

### Changed

- The batched attention (`attention_forward_sequence`, `attention_backward` with a backend) projects its keys and
  values through `linear_heads` instead of a loop of per-head products; the same results on every existing backend.

## 0.10.0 - 2026-11-16

### Added

- **The model's other steps on the backend** (TRAINING.md T9): `embed_rows` and `embed_rows_backward` (32-bit ids,
  the backward in token order), `cross_entropy_rows` (the fused classifier over a sequence), `lm_head_loss_sequence`
  (the whole sequence's logits as one product, the same loss as `lm_head_loss` to the bit through the reference),
  `adamw_rows`, and the reductions `sum_values`, `sum_squares` and `scale_rows` (gradient clipping). With them a whole
  model's training step can be written against a backend.
- **The batched passes run on an operations backend** (TRAINING.md T9): every step of `block_forward_sequence`,
  `block_backward`, the batched attention and MLP passes and `linear_sequence` that is not a matrix product is a call
  made with the backend: `copy_rows`, `fill_rows`, `add_rows`, `add_bias_rows`, `sum_rows`, `mlp_activate_rows` and
  its backward, `attention_position_rows`, `attention_attend_rows` and their backward passes, `block_norm_rows`,
  `block_norm_rows_backward_add`. Each has an overload for any backend that runs on the host, as before (the same
  bits); a device library overloads them for its own backend type, found by argument-dependent lookup, so a training
  step can stay on the device. A test and the block demo override two of them from another namespace. Proven by snicholls-gpu's
  CUDA backend, whose overloads run a whole training step of GPT-2-, Llama-, nanochat- and multi-query blocks on the
  GPU against this reference.

## 0.9.0 - 2026-11-16

### Added

- **`linalg/randomized_svd.hpp`: the truncated SVD.** `truncated_svd(backend, a, s, u, v, work, options, sketch)`
  gives the leading `k` singular triplets by randomised range finding (Halko, Martinsson and Tropp): a random sketch,
  power iterations re-orthonormalised by QR, and the exact SVD of the small projection, on any backend with the BLAS,
  factorisations and decompositions; `truncated_svd_work_size` sizes the caller's space. The random test matrix is a
  seam: `sketch` is any callable that fills it, `gaussian_sketch<Normal>` (over any Gaussian sampler) by default and
  `rademacher_sketch` beside it. Checked against matrices built from known singular values (an exact low rank to
  rounding, a decaying spectrum to the best rank-k error), the exact SVD when the sketch is full, every sketch and
  sampler and one of the test's own, layouts, float, errors, and at compile time; 19 to 64 times faster than the exact
  SVD for the leading 10 of 500 x 100 and 2000 x 200.
- **`stats/normal.hpp`: Gaussian samplers behind one concept.** `normal_sampler<S, G>`, with `normal_inverse_cdf`
  (the normal quantile of an exact open uniform: the same draws on every platform, never infinite) and
  `normal_box_muller` (four times faster); a ziggurat or a device's generator drops in as a third.
- **Batched training through a `linalg` backend** (`ml/linear.hpp`, `ml/mlp.hpp`, `ml/attention_layer.hpp`,
  `ml/transformer_block.hpp`): `linear_sequence` and `linear_sequence_backward` are a sequence's projection and its
  backward as matrix products through any `linalg::blas_backend`; `mlp_forward_sequence` / `mlp_backward_sequence`
  and backend-first overloads of `attention_forward_sequence`, `attention_backward`, `block_forward_sequence` and
  `block_backward` run every projection of a layer that way, so a BLAS or a GPU backend in another library trains the
  layer with the same call. QK norm, RoPE and the heads are the per-token path's own code, now in shared helpers (the
  per-token path's bits unchanged, checked by the bit-for-bit tests and the benchmark's fairness check). New working
  buffers: `mlp_activations::hidden`, `mlp_gradients::scratch`, `attention_gradients::y`,
  `block_activations::mlp_hidden`. Checked against the per-token path in double for every configuration (GQA, QK
  norm before and after RoPE, the soft cap, windows, SwiGLU, ReLU², GELU); `linear_sequence` is level with the same
  products by hand through `linalg::reference`, bit for bit (`benchmarks/ml/linear_benchmark.cpp`).

- **The error-sink conformance suite is exported** (`snicholls::core_conformance`): `#include
  <snicholls/core/conformance/error_sink.hpp>` and `SNICHOLLS_ERROR_SINK_CONFORMANCE(Prefix, probes...)` register a
  sink from another library (super-log's) in the same typed suite as core's `text_error_sink`; it was
  `tests/support/error_sink_contract.hpp`, reachable only from core's own tests. The suite moved word for word, into
  `namespace snicholls::conformance` beside linalg's.
- **`ml/self_attention.hpp`: one causal self-attention head on its own.** `self_attention_head<DModel, DHead>` takes
  `Wq`, `Wk`, `Wv` (any `weight_matrix`: dense float, double, bfloat16 or quantised) and a sequence of tokens and
  writes each position's `softmax(q · K / sqrt(DHead)) · V`, causal or within a sliding window, with optional RoPE.
  It has no configuration and no cache. The queries, keys, values and log-sum-exps go to `self_attention_activations`,
  buffers the caller owns, for a backward pass to read. Checked against the naive masked softmax in double; a
  prefix gives the whole sequence's bits; level with the same head written by hand, bit for bit.
- **`self_attention_head_backward`**: the head's backward pass from what the forward saved (`attend_backward` from
  the log-sum-exps, `rope_inverse`, then the projections), into `self_attention_gradients`. Checked against central
  differences in double, causal, windowed and with RoPE; level with the same pass by hand, bit for bit.
- **`ml/muon.hpp`: the Muon optimiser.** `muon_orthogonalise` runs the Newton-Schulz quintic on a matrix (on its
  transpose when tall) through any `linalg::blas_backend`, and `muon_step` adds momentum (Nesterov by default) and
  nanochat's `sqrt(max(1, rows / cols))` scaling; `muon_work_size` sizes the caller's working space. Checked
  against the iteration as plain loops, the singular values in Muon's band by `linalg::svd`, and at compile time.
- **`ml/lm_head.hpp`: the loss of a sequence through the head.** `lm_head_loss(head, hs, targets, logits)` is the
  mean next-token cross-entropy; with `dhs` and `dhead` it also adds the gradients, one row of logits at a time
  (never `n x V`), turned into its gradient in place. A tied head is the embedding table viewed as
  `dense_weights`. Checked against central differences in double, tied and untied.
- **The transformer block trains** (`ml/transformer_block.hpp`): `block_forward_sequence` runs a block over a
  sequence in place and saves `block_activations` (each token's result `block_prefill`'s, to the bit);
  `block_backward` turns the gradient in its output into the gradient in its input and adds every weight's into
  `block_gradients`. Checked against central differences in double for GPT-2-, Llama- and nanochat-shaped blocks
  (layer and RMS norm, GELU, SwiGLU and ReLU², GQA, QK norm, RoPE, a window), and against llm.c's `gpt2_backward`
  from the committed fixture (every gradient within two float epsilons); the demo trains one block by gradient
  descent. Benchmarked on a GPT-2-shaped layer over 64 tokens against the same passes by hand, after a bit-for-bit
  check on every activation and gradient: within 5% forward and 3% backward.
- **A tiny GPT-2 trained end to end** (`demos/ml/muon_demo.cpp`): embeddings, two blocks, the final norm and a tied
  head, Muon on the blocks' matrices and AdamW on the rest; the loss falls from a uniform guess to 0.01 in 30 steps,
  with the same bits on a second run.
- **The attention layer trains** (`ml/attention_layer.hpp`): `attention_forward_sequence` saves
  `attention_activations` over a sequence (each token's output `attention_forward`'s, to the bit), and
  `attention_backward` gives every gradient (`attention_gradients`: the projections, biases, QK-norm weights and the
  inputs) through grouped-query heads, QK norm before or after RoPE, a sliding window and the soft cap;
  `attention_head_backward` is one configured head's. Checked against central differences in double in five
  configurations.
- **`mlp_backward`** (`ml/mlp.hpp`) for GELU, tanh GELU, ReLU² and SwiGLU, from the pre-activations a new
  `mlp_forward(w, x, y, save)` writes into `mlp_activations` (the output is the same to the bit);
  `mlp_activation_derivative<K>`; **`moe_gate_backward`**, the router's gradient through the top-K softmax. Checked
  against central differences in double.
- **`log_softmax_backward`** (`ml/softmax.hpp`), checked against central differences and against `softmax_backward`.
- **`linear_backward`** (`ml/linear.hpp`): `dx += Wᵀ dy`, `dW += dy xᵀ`, `db += dy` for dense float, double and
  bfloat16 weights, as llm.c's `matmul_backward`. Quantised weights have no backward: training them is a compile
  error.

### Changed

- `attention_backward` now leaves `attention_gradients::k` holding the gradients in the projected keys (QK norm and
  RoPE undone), as its documentation always said; it held them before the undoing.
- `rope_source` and `no_rope` moved from `ml/attention_layer.hpp` to `ml/rope.hpp`, so a head can take RoPE without
  the layer's header. Same names, same namespace; `attention_layer.hpp` still provides them.

## 0.8.0 - 2026-11-12

The transformer layer, linear algebra, mapped storage, and the family's rules.

### Added

- **The transformer layer for inference** (`ml/`, plan in `INFERENCE.md`; training is planned in `TRAINING.md`):
  `transformer_config` (the architecture as one compile-time value: MHA, GQA and MQA, layer or RMS norm, QK norm
  before or after RoPE, both RoPE layouts, GELU, ReLU² or SwiGLU, a sliding window, an attention soft cap, biases;
  presets `gpt2_small`, `llama3_8b`, `nanochat_d20`), `linear` (dense float, double or bfloat16, or ggml's quantised
  blocks, the input prepared once), `mlp` (and `moe_gate<K>`, a mixture-of-experts router), `kv_cache` (a flat or
  ring cache, and a paged cache with content-addressed pages, `kv_page_hash`, so requests sharing a prefix share its
  pages), `attention_layer` and `transformer_block` (prefill is decode token by token). Checked against llm.c's
  GPT-2 block and a double-precision textbook block for the rest, within about one float epsilon; a GPT-2-small
  decode step runs level with the same layer by hand, float and Q8_0.
- `ml/sampling.hpp`: `sample_min_p`, `min_p_threshold`, and `apply_penalties` (repetition, presence, frequency).
- **Linear algebra** (`linalg/`, `snicholls::linalg`): backend concepts and free functions over `md_span` views
  (`dot`, `axpy`, `gemv`, `gemm`, `syrk`, `triangular_solve`, LU, Cholesky, LDLT, QR, least squares, the symmetric
  eigenproblem, SVD, log-determinants), errors as `result<V, linalg::error>`; batched LU, Cholesky and LDLT solves
  with a status per system on the caller's executor, bit-for-bit the same for every thread count with the reference
  backend; a `constexpr` reference backend; a conformance suite for any backend, exported as the test-only target
  `snicholls::core_conformance`.
- `md_span<T, Rank, Layout>` takes a layout as `std::mdspan` does: `layout_right` (the default), `layout_left`,
  `layout_stride`, with `submdspan` and the `std::mdspan` spellings.
- **Mapped storage, the platform-free half** (the mapping is ts-moveables'): `object_pool` takes a
  `std::pmr::memory_resource`; `index_pool` (slots by index over a region that may move); `mapped_header` (a
  checked 64-byte file header); `offset_ptr`.

### Changed

- `CLAUDE.md`: "The family", twelve rules every snicholls library follows (each function once, the best version
  wins, placement by need, interfaces in core, shared vocabulary extended from each type's namespace, one owner per
  concept, the bare minimum of headers, proposals for approval, and the rest).
- 17 unused includes removed from 16 headers.

### Fixed

- A run-time read of a `double` member of a class-type template argument is miscompiled by Clang (Apple clang 17,
  Clang 18): `transformer_config`'s floating-point members are read through `norm_eps_v`, `rope_base_v` and
  `attention_softcap_v`, which evaluate them at compile time.

## 0.7.0 - 2026-11-12

More machine-learning primitives, experimental, from llm.c and nanochat.

### Added

- `ml/rope.hpp`: rotary position embedding: `rope_frequency`, `rope_angles`, `rope_table` (built at compile time),
  `rope` and `rope_inverse` (its backward pass), in the half and interleaved layouts.
- `ml/attention.hpp`: `attend` for one query in one pass (flash attention's recurrence) and `attend_backward`;
  `attention_accumulator<T, D>`, a partial that merges across workers; `attention_window` for causal and sliding
  windows.
- `ml/embedding.hpp`: `embedding_row`, `embed`, `embed_with_position` (GPT-2) and `embedding_backward`.
- `ml/sampling.hpp`: `argmax`, `sample`, `sample_logits` at a temperature, `top_k<K>`, `sample_top_k<K>` and
  `sample_top_p`, each from one uniform draw the caller supplies.
- `ml/optim.hpp`: `adamw_step` (llm.c's update), `warmup_cosine` and nanochat's `warmup_warmdown`, `squared_norm`
  and `clip_coefficient`, Muon's `newton_schulz_polynomial`, `transformer_flops_per_token` and
  `model_flops_utilisation`.
- `ml/quantize.hpp`: ggml's Q8_0 and Q4_0 blocks and the MXFP4 block, byte for byte, with quantisation,
  dequantisation and dot products against Q8_0 activations equal to ggml's scalar reference on a committed
  fixture; symmetric int8 rows.
- `containers/byte_map.hpp`: `byte_map_view` (over the caller's spans) and `byte_map<Slots, ArenaBytes, Values>`
  (owning, with a `consteval` constructor): byte strings to 32-bit values in one table of 16-byte slots, with
  `probe` then `find` so a batch of lookups overlaps its cache misses, and an optional reverse table (`key_of`).
  An `ml::bpe_vocabulary` as it stands. Contributed from ts-moveables' tokenizer.

### Changed

- A benchmark's smoke CTest times out after 300 seconds, so a benchmark that hangs fails rather than holding a CI
  job until the runner kills it.

### Fixed

- `math::round` judged the half on `x + 0.5`, which rounds the largest float below one half (0.49999997f) up to 1;
  it now gives 0, as `std::round` does.
- `time/rfc3339.hpp` includes `<charconv>` for `std::from_chars_result`, which it had reached through other
  headers; LLVM 23's libc++ no longer provides it that way.

## 0.6.1 - 2026-11-11

### Added

- `ml/bpe.hpp`: `bpe_encode_long_piece`, the same tokens as `bpe_encode_piece` in O(n log n) from a heap of
  candidate pairs, for the long pieces a split pattern can keep whole (a run of spaces, a minified line, a base64
  blob); `bpe_long_scratch_size(n)` (5n + 1) and `bpe_long_piece_threshold` (1024 bytes, where it is at least as
  fast on every shape measured). Contributed from ts-moveables' parallel tokenizer.

## 0.6.0 - 2026-11-11

Machine-learning primitives, experimental.

### Added

- `ml/` (`snicholls::ml`), experimental: the primitives of language models, `constexpr` and alloc-free.
  - `activations.hpp`: sigmoid, softplus, log-sigmoid, logit, ReLU, ReLU², SiLU, GELU (erf) and GPT-2's tanh GELU,
    the logit soft cap, each with its derivative, in forms that neither overflow nor cancel.
  - `softmax.hpp`: softmax, log-softmax, logsumexp, the softmax backward pass, cross-entropy and its fused gradient
    (in place), bits per byte, and `online_softmax`, a running softmax whose partials merge across workers.
  - `norm.hpp`: LayerNorm and RMSNorm, forward and backward, with or without learned parameters.
  - `minifloat.hpp`: `minifloat<E, M, Kind>` with `bfloat16`, `float16`, FP8 E4M3 and E5M2, MX FP6 and FP4: exact
    decoding, round to nearest even, stochastic rounding, overflow policies, compile-time decode tables.
  - `bpe.hpp`: byte-level BPE merging that gives tiktoken's tokens, with the caller's vocabulary lookup. On 12.3 MB
    of Python source, one thread, it ran at 9.2 million tokens/s with cl100k against tiktoken's 2.3 million
    (README).
  - `row.hpp`: the `row` and `mutable_row` concepts the functions take: any contiguous range of numbers.
- `math`: `expm1` and `log1p`.

## 0.5.1 - 2026-11-10

The fixes CI found once it could run on every platform.

### Fixed

- `math`: compile-time `sin` and `cos` of a 113-bit `long double` (Linux ARM64) reduce the argument with pi/2 in four
  parts; the three parts `double` and the 80-bit `long double` use carry only about 100 bits. `double` is unchanged.
- `parse_number<T>` for a floating-point `T` builds where the standard library has no floating-point `std::from_chars`
  (Xcode 16's libc++): the same syntax, checked first, then the C library's conversion.
- `type_list` permutations compile on MSVC.
- The README and CLAUDE.md show `snicholls_core_require(0.5.0)`; status lines say when a header became public.

### Changed

- The `type_list` and `string_traits` demos name types with an `overloaded` set over `std::type_identity`, not an
  `if constexpr` chain.
- CI records every benchmark's throughput in each job's summary and keeps the raw results
  (`scripts/benchmarks_md.py --output - --json DIR`); the README shows CI's status.

## 0.5.0 - 2026-11-10

The first public release. Versions before it were developed privately; this entry describes what core holds.

### Vocabulary types

- `overloaded` (an overload set for `std::visit`) and `first_match` (an ordered set: the first callable that accepts
  the arguments runs), with function pointers as elements; `function_ref`; `forward_like`.
- `fixed_string` (a string as a template parameter), `string_traits` and `string_access` (every string and byte
  sequence classified, zero-copy access).
- `result<V, E>` (Rust's `Result`, zero overhead, `SNICHOLLS_TRY`) and `traced<E>` (an error with its location,
  context, fields and stack, for logs). Design: `RESULT_PLAN.md`.
- `strong<T, Tag>`, `secret`, checked arithmetic (`checked.hpp`, `parse_number`), `uint256` with token units,
  `decimal`, `uuid` (v4, v5, v7 composed from given bits), `backoff_policy`.
- Time: `utc<D>` with RFC 3339 text, Unix conversions, `std::format`, the civil calendar, business days and solar
  times. Portable to the tick and the character on every supported platform, checked against fixtures generated
  independently from Python's `datetime` (`tests/fixtures/time/`).

### Compile-time machinery

- `constexpr_for`, `constexpr_nest`, `unrolled_for`; `type_list` with filter, fold, products, `Thread` and
  permutations; `thread_apply`; `zip`; `invoke_with`; `constexpr_map`.
- Enum reflection (`meta/enums.hpp`): names, values, index and cast, from the compiler alone.
- Reflection over plain aggregates (`member_count`, `tie`, `to_tuple`, `from_tuple`) up to 256 members.
- Compile-time attributes (`meta/attributes.hpp`): any number of attributes on any type or enumerator, declared
  once beside the type and read by every printer.

### Containers

- `inplace_vector`, `static_map` and `static_set` (sorted at compile time), `perfect_map` (a perfect hash built by
  the compiler), `md_array` and `md_span`, `bloom_filter`, `dense_id_map`, `flat_id_map`, `hierarchical_bitset`,
  `intrusive_avl_tree`, `object_pool`.

### Numerics

- `math/constexpr_math.hpp`: exp, log, the trigonometric and hyperbolic functions, erf, erfc, lgamma, the
  incomplete gamma, and the normal and chi-square distributions, `constexpr`, for `float`, `double`, `long double`
  or a type of your own through a `math_limits` specialisation.
- Randomness and resampling: `wyrand` and unbiased bounded draws, their output pinned by sequences computed
  independently from the published algorithms; the bootstrap by indices, counts, Poisson weights,
  balanced, stationary and moving blocks; BCa intervals; Sobol and Halton sequences.
- Online statistics that merge across workers (`stats/accumulators.hpp`), Ledoit-Wolf covariance, one-pass pivot
  tables with typed keys, histograms and per-cell resampling.
- Martingale tools (the Doob decomposition, variance ratio and Ljung-Box tests) and the Snell envelope.
- Cooperative games: exact and sampled Shapley values, Owen values, Harsanyi dividends, game properties, and P&L
  attribution by greek and factor.
- Combinatorics: permutations, set partitions, subsets.

### Printing

- `acout`: thread-safe printing, one message one write, of any value, container, tuple, optional, variant, result
  or type list, with colour that never leaks between threads.
- Display attributes (`display.hpp`): colour, colour by sign with a dead zone, colour by thresholds, label, number
  format checked by the compiler, units, scale, minor units and tooltip, used by `acout` and pivot tables and
  exported as JSON for other front ends.

### Build

- CMake target `snicholls::core`; one version of core per build, enforced at configure time
  (`snicholls_core_require`, `snicholls_core_assert_single_version`).
- Every public header has tests, a demo that exits non-zero on a wrong result, Doxygen documentation built with
  warnings as errors, and, where it has a run-time cost, a benchmark against the hand-written equivalent
  (`BENCHMARKS.md`).
