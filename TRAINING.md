# snicholls::ml - training the transformer layer

Status: done in 0.9.0, except where a decision says otherwise (T5, and bfloat16 training under T4); what comes next
is under "After 0.9.0". Headers: `include/snicholls/core/ml/` (standard library and core
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
| T1 | **Training has its own sequence forward, which saves into the caller's buffers; inference is unchanged.** Planned as one forward with a compile-time policy (`activations<Config, T, save::all>`, `save::none` for inference). Built instead: `attention_forward_sequence` and `block_forward_sequence` write `attention_activations` / `block_activations`, spans the caller owns (Q and K after QK norm and RoPE, V, the raw projections when there is a QK norm, the heads' outputs and log-sum-exps, each norm's input and output, the MLP's pre-activations), and `mlp_forward` takes an optional `mlp_activations`. Why: inference runs one token at a time against a KV cache, training a whole sequence with its keys in the activations, so the two loops differ anyway; a policy parameter would have touched every inference signature for nothing. The norms' statistics are not saved but recomputed in the backward (`d_model` operations a row). The contract that replaces "one forward": each saving forward gives the inference path's result to the bit, tested (`TheSequenceForwardIsPrefillToTheBit`, `TheSequenceForwardIsAttentionForwardTokenByTokenToTheBit`, `SavingThePreActivationsChangesNoOutput`). |
| T2 | **Attention saves the log-sum-exp, never the T x T probabilities.** `attend_backward` rebuilds each probability from q, k and the lse, so memory is O(T x heads), as flash attention's backward. |
| T3 | **Gradients are views the caller owns, one per weight, and accumulate.** Planned as `block_weights` over mutable storage. Built instead: `block_gradients`, `attention_gradients`, `mlp_gradients`, `self_attention_gradients`, one span per tensor with the weight's size (asserted), plus the backward's working buffers (`scratch`, the Q, K, V gradients). Why: a weight is often a view of storage that cannot hold a gradient (a quantised or bfloat16 matrix, a tied embedding), and the gradients need buffers the weights do not have. Weight gradients are added into (`+=`), so micro-batches accumulate without a copy; the Q, K, V gradients are written. |
| T4 | **The weight storage decides what is trainable.** Float and `bfloat16` weights are; quantised blocks are inference only, and `linear_backward` (`dx = W^T dy`, `dW += dy x^T`) accepts only trainable storage, so training a Q4_0 model is a compile error. Master weights are float, rounded to `bfloat16` stochastically (`ml/minifloat.hpp`). Status: the backward passes accept `bfloat16` storage; no test or demo trains that way yet (after 0.9.0). |
| T5 | **Gradients merge.** Each data-parallel worker's gradients sum in a fixed order (reproducible); `squared_norm` and `clip_coefficient` already merge across shards. The all-reduce across machines is ts-moveables'. Status: after 0.9.0; the fixed-order sum has no helper or test yet. |
| T6 | **A sequence's loss is the fused LM-head cross-entropy** over its positions, from the per-row `cross_entropy` with its fused gradient. A tied LM head reuses `embedding_backward`. |
| T7 | **References: llm.c's backward, and finite differences.** The generator of `INFERENCE.md` I10 also writes llm.c's `matmul_backward`, `attention_backward`, `layernorm_backward`, `gelu_backward` for the same tiny GPT-2. The features llm.c lacks (GQA, QK norm, ReLU², SwiGLU, RoPE, soft cap) are checked by central differences in double. |
| T8 | **Muon runs through the linear algebra backend.** `muon_orthogonalise` and `muon_step` (`ml/muon.hpp`) run the iteration `X <- a X + (b A + c A^2) X`, `A = X X^T`, on any `linalg::blas_backend`, on the transpose of a tall matrix. Done. |

## Headers

| Header | Adds |
|---|---|
| `linear.hpp` | `linear_backward` for trainable weights (done; with `self_attention_head_backward`, one head on its own) |
| `mlp.hpp` | `mlp_backward` (done; with `moe_gate_backward`) |
| `attention_layer.hpp` | `attention_backward` per head and per layer, from the saved lse (done) |
| `transformer_block.hpp` | `block_activations`, `block_gradients`, `block_backward` (done; benchmarked against the loops by hand) |
| `softmax.hpp` | the sequence cross-entropy with its fused gradient over the LM head (done: `ml/lm_head.hpp`) |
| `optim.hpp` | the Muon iteration, once a matrix product is available (done: `ml/muon.hpp`, over a `linalg` backend) |

## Order of work

1. The saving forwards, with tests that they give the inference result to the bit (done; see T1).
2. `linear_backward`, `mlp_backward`, then `attention_backward`, then `block_backward` (done).
3. Tests against the llm.c backward fixture (done: `Gpt2BackwardIsLlmcsWithinTolerance`, every gradient within two float epsilons); central differences for the rest (done).
4. A demo that trains the tiny GPT-2 for a few steps and shows the loss falling, reproducibly (done: `demos/ml/muon_demo.cpp`, `train_gpt2`).
5. Benchmarks against the same loops by hand (done: `transformer_block_benchmark.cpp`, within 5% forward and 3%
   backward); documentation; the gate; release 0.9.0.

## After 0.9.0

1. **Batched products.** Training multiplies one token at a time, matrix by vector; a sequence path through
   `linalg::gemm` (any backend: the reference, a BLAS, a device) for the projections, forward and backward.
2. **bfloat16 training**, tested end to end (T4).
3. **The fixed-order gradient sum** across workers (T5).
4. nanochat's logit soft cap in `lm_head_loss`; Muon in `bfloat16`.
5. An MoE block in `block_forward_sequence` / `block_backward`; post-norm and sandwich norms.
