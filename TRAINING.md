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
| T5 | **Gradients merge.** Each data-parallel worker's gradients sum in a fixed order (reproducible); `squared_norm` and `clip_coefficient` already merge across shards. The all-reduce across machines is ts-moveables'. Status: the fixed-order sum exists (0.15.0): `ordered_sum` (`ml/diloco.hpp`) adds the workers' parts left to right in rank order in the value's own arithmetic, tested to the bit against a loop by hand, and is the reference an all-reduce reproduces; `ordered_mean` divides it by the worker count. |
| T6 | **A sequence's loss is the fused LM-head cross-entropy** over its positions, from the per-row `cross_entropy` with its fused gradient. A tied LM head reuses `embedding_backward`. |
| T7 | **References: llm.c's backward, and finite differences.** The generator of `INFERENCE.md` I10 also writes llm.c's `matmul_backward`, `attention_backward`, `layernorm_backward`, `gelu_backward` for the same tiny GPT-2. The features llm.c lacks (GQA, QK norm, ReLU², SwiGLU, RoPE, soft cap) are checked by central differences in double. |
| T8 | **Muon runs through the linear algebra backend.** `muon_orthogonalise` and `muon_step` (`ml/muon.hpp`) run the iteration `X <- a X + (b A + c A^2) X`, `A = X X^T`, on any `linalg::blas_backend`, on the transpose of a tall matrix. Done. |
| T9 | **Training runs on an operations backend; core's is the reference.** The batched passes are written against a backend of training operations (the products of `linear_sequence`; the norms and their backward passes; attention over a sequence and its backward; activations; residual adds, fills and copies), each over views whose data lives where the backend says. Core's backend is today's functions on the host, so the per-token and batched passes stay the single source of truth and every device result is checked against it; a device library (snicholls-gpu) implements the same operations as kernels over device memory, and a whole training step stays on the device. Built (0.10.0): the operations are free functions with a host overload for any backend, the passes call them with their backend, and a device library's overloads for its own type are chosen by argument-dependent lookup; the bits unchanged (the existing tests and the benchmarks' fairness checks). A batch of sequences is one call (Unreleased): the passes and the attention steps take a number of sequences laid end to end, the products made over every row of the batch, attention within each sequence, the activations each sequence's own side by side; on the reference backend the per-sequence calls' bits, except the weight gradients of the products over all the rows, within rounding. |

## Headers

| Header | Adds |
|---|---|
| `linear.hpp` | `linear_backward` for trainable weights (done; with `self_attention_head_backward`, one head on its own) |
| `mlp.hpp` | `mlp_backward` (done; with `moe_gate_backward`) |
| `attention_layer.hpp` | `attention_backward` per head and per layer, from the saved lse (done) |
| `transformer_block.hpp` | `block_activations`, `block_gradients`, `block_backward` (done; benchmarked against the loops by hand) |
| `softmax.hpp` | the sequence cross-entropy with its fused gradient over the LM head (done: `ml/lm_head.hpp`) |
| `optim.hpp` | the Muon iteration, once a matrix product is available (done: `ml/muon.hpp`, every step on the backend, so it runs on a device as `adamw_rows` does) |

## Order of work

1. The saving forwards, with tests that they give the inference result to the bit (done; see T1).
2. `linear_backward`, `mlp_backward`, then `attention_backward`, then `block_backward` (done).
3. Tests against the llm.c backward fixture (done: `Gpt2BackwardIsLlmcsWithinTolerance`, every gradient within two float epsilons); central differences for the rest (done).
4. A demo that trains the tiny GPT-2 for a few steps and shows the loss falling, reproducibly (done: `demos/ml/muon_demo.cpp`, `train_gpt2`).
5. Benchmarks against the same loops by hand (done: `transformer_block_benchmark.cpp`, within 5% forward and 3%
   backward); documentation; the gate; release 0.9.0.

## After 0.9.0

1. **Batched products.** Training multiplies one token at a time, matrix by vector; a sequence path through
   `linalg::gemm` (any backend: the reference, a BLAS, a device) for the projections, forward and backward. Done (0.9.0): `linear_sequence` and the backend-first overloads of the layer, MLP and block passes.
2. **bfloat16 training**, tested end to end (T4).
3. **The fixed-order gradient sum** across workers (T5). Done (0.15.0): `ml::ordered_sum`, `ml::ordered_mean`.
4. nanochat's logit soft cap in `lm_head_loss`; Muon in `bfloat16`.
5. An MoE block in `block_forward_sequence` / `block_backward`; post-norm and sandwich norms.

## Towards large-scale training

The objective beyond this file: training transformer language models from scratch at scale, on GPUs across several
machines. Core's part is the arithmetic and the interfaces, each with its reference on the CPU; devices, files,
networks and the model type live in the libraries that pin core (a GPU backend:
[snicholls-gpu](https://github.com/saxonnicholls/gpu), whose `ROADMAP.md` is the device side). In the order the work
needs them:

1. **A training-operations backend** (decision T9 below): every step of the batched passes, not only the products,
   becomes a call on a backend, so a device runs the whole step without the data leaving it. Measured on an RTX PRO
   4000 Blackwell, one GPT-2-small layer's training step over 128 tokens through the GPU's matrix products takes 56
   ms: 38 ms in its 84 products (about 4 ms of arithmetic, the rest moving operands to and from the device) and 18 ms
   in the steps still run on the host (norms, the attention loop, residuals, copies). Device memory for the views
   alone would leave the second half on the host.
2. **A reference for every training kernel a device runs**: the passes here already are; added as a device needs them
   (fused forms, recomputation of a block in its backward pass for memory).
3. **bfloat16 training** tested end to end (T4): bfloat16 products, float master weights and optimiser state,
   stochastic rounding.
4. **The fixed-order gradient sum** (T5): partial gradients from any number of workers summed in an order that does not
   depend on timing, so results do not depend on scheduling. Done (0.15.0): `ordered_sum` (`ml/diloco.hpp`).
5. **Training that communicates rarely**: the outer optimiser of DiLoCo (Douillard et al., 2023), in which workers take
   hundreds of local steps between exchanges of their weights' change, for clusters whose network is too slow for an
   exchange every step; and the compression of what is exchanged (bfloat16, 8-bit with error feedback). The arithmetic
   only: the transport is a library's above core. The outer optimiser is done (0.15.0): `pseudo_gradient` and
   `diloco_outer_step` (`ml/diloco.hpp`), with four workers on a least-squares problem reaching one worker's loss;
   the compression is not.
6. **Sharding arithmetic** for models too large for one device: the partition of optimiser state and weights (ZeRO),
   and the split of a model's blocks across stages.

Each is checked as everything here is: against a reference implementation, central differences, or to the bit.

