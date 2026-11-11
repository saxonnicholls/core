# Changelog

All notable changes to snicholls-core. The format follows Keep a Changelog; versions follow Semantic Versioning.
A version is released when its tag is on a commit where every blocking CI leg is green.

## 0.7.0 - unreleased

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

### Fixed

- `math::round` judged the half on `x + 0.5`, which rounds the largest float below one half (0.49999997f) up to 1;
  it now gives 0, as `std::round` does.

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
