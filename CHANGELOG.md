# Changelog

All notable changes to snicholls-core. The format follows Keep a Changelog; versions follow Semantic Versioning.
A version is released when its tag is on a commit where every blocking CI leg is green.

## Unreleased

### Added

- bfloat16 pseudo-gradients in `ml/diloco.hpp`, half the bytes on the wire, specified to the bit so that a transport
  and a training library agree:
  - `ml::compress_bf16(delta, residual, out)`: `delta + residual` rounded to bfloat16 (to nearest, ties to even), with
    error feedback: the residual keeps exactly what the rounding dropped and is added to the next round's delta.
  - `ml::ordered_sum_bf16(parts, out)`: the workers' bfloat16 parts decoded and summed in rank order in float, the
    reducing rank's arithmetic.
  - `ml::ordered_mean_bf16(parts, out)`: what a bfloat16 all-reduce and the mean leave on every worker, the float sum
    rounded once to bfloat16 and divided by the worker count.
- The documentation of `ml/diloco.hpp` states what a worker that leaves and rejoins needs (nothing new in the outer
  step: the mean divides by that round's parts).

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
