# snicholls::ml - training the transformer layer

Status: planned, for 0.9.0, after `INFERENCE.md`. Headers: `include/snicholls/core/ml/` (standard library and core
only).

## Why

Most of the arithmetic exists: the backward passes of `attend`, `layer_norm` / `rms_norm`, the activations, the
softmax and cross-entropy, `embedding`; `adamw_step`, the schedules, clipping by the global norm, stochastic rounding
to `bfloat16`. What training adds is the backward pass of the layer and the block, and the forward has to keep what
that reads. The inference plan is shaped for this (`INFERENCE.md` I2, I3, I8), so training adds functions rather
than changing them.

## Decisions

| Id | Decision |
|---|---|
| T1 | **The forward saves activations by a compile-time policy.** `activations<Config, T, save::all>` holds what the backward reads: each norm's input, mean and reciprocal standard deviation; Q, K, V after RoPE; the attention output and its log-sum-exp per query and head; the MLP's pre-activation. `save::none` (inference) stores nothing and costs nothing. One forward, not two. |
| T2 | **Attention saves the log-sum-exp, never the T x T probabilities.** `attend_backward` rebuilds each probability from q, k and the lse, so memory is O(T x heads), as flash attention's backward. |
| T3 | **Gradients have the shape of the weights.** `block_grads<Config>` is `block_weights<Config, W>` over mutable float storage, so the two cannot drift apart. The backward accumulates (`+=`), so micro-batches add without a copy. |
| T4 | **The weight storage decides what is trainable.** Float and `bfloat16` weights are; quantised blocks are inference only, and `linear_backward` (`dx = W^T dy`, `dW += dy x^T`) accepts only trainable storage, so training a Q4_0 model is a compile error. Master weights are float, rounded to `bfloat16` stochastically (`ml/minifloat.hpp`). |
| T5 | **Gradients merge.** Each data-parallel worker's gradients sum in a fixed order (reproducible); `squared_norm` and `clip_coefficient` already merge across shards. The all-reduce across machines is ts-moveables'. |
| T6 | **A sequence's loss is the fused LM-head cross-entropy** over its positions, from the per-row `cross_entropy` with its fused gradient. A tied LM head reuses `embedding_backward`. |
| T7 | **References: llm.c's backward, and finite differences.** The generator of `INFERENCE.md` I10 also writes llm.c's `matmul_backward`, `attention_backward`, `layernorm_backward`, `gelu_backward` for the same tiny GPT-2. The features llm.c lacks (GQA, QK norm, ReLU², SwiGLU, RoPE, soft cap) are checked by central differences in double. |
| T8 | **Muon's matrix iteration waits for the linear algebra.** `newton_schulz_polynomial` exists; the iteration `X <- a X + (b A + c A^2) X` with `A = X X^T` needs a matrix product, which comes with the linear algebra backend. |

## Headers

| Header | Adds |
|---|---|
| `linear.hpp` | `linear_backward` for trainable weights |
| `mlp.hpp` | `mlp_backward` |
| `attention_layer.hpp` | `attention_backward` per head and per layer, from the saved lse |
| `transformer_block.hpp` | `activations<Config, T, Save>`, `block_grads<Config>`, `block_backward` |
| `softmax.hpp` | the sequence cross-entropy with its fused gradient over the LM head |
| `optim.hpp` | the Muon iteration, once a matrix product is available |

## Order of work

1. `save::all` in the forward, with a test that it changes no output.
2. `linear_backward`, `mlp_backward`, then `attention_backward`, then `block_backward`.
3. Tests against the llm.c backward fixture; central differences for the rest.
4. A demo that trains the tiny GPT-2 for a few steps and shows the loss falling, reproducibly.
5. Benchmarks against the same loops by hand; documentation; the gate; release 0.9.0.
