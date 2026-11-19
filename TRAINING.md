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
| T10 | **Mixture of experts: every step a backend call over rows, each reduction in a stated order.** Designs: Switch (Fedus et al., 2021) and OLMoE (Muennighoff et al., 2024), Mixtral (Jiang et al., 2024), DeepSeekMoE (Dai et al., 2024) and DeepSeek-V3 (2024), ST-MoE (Zoph et al., 2022). Configured by `transformer_config`'s expert fields: `experts` (E; 0 is a dense MLP and every other expert field is 0, so no preset changes), `top_k` (K), `shared_experts`, `expert_d_ff`, `scoring` (`moe_scoring::softmax_top_k` Mixtral, `softmax` Switch / OLMoE, `sigmoid` DeepSeek-V3), `route_scale`, `bias_update_rate` (gamma), `balance_loss_weight` (alpha), `z_loss_weight` (beta); the `double`s read through `route_scale_v` and friends. Presets `tiny_moe` and `olmoe_1b_7b`. Index data is `std::uint32_t`. The ops (`ml/moe.hpp`; `C` the configuration, `b` the backend, `d = d_model`, `n = rows · K`): **routing** `moe_route_rows<C>(b, logits[rows·E], bias[E or empty], experts[rows·K], weights[rows·K])`: affinities `s` = `z`, `softmax(z)` or `sigmoid(z)`; key `s + bias` for the choice only; `top_k<K>` of the keys (largest first, ties to the lower expert); weights `softmax` over the chosen logits (max subtracted, sum in slot order) times scale, the chosen probabilities times scale, or the chosen sigmoids over their slot-order sum times scale; at scale 1 with no bias `softmax_top_k` is `moe_gate` to the bit. `moe_route_rows_backward<C>(b, logits, experts, dweights, dlogits)` writes `dlogits` (the choice has no gradient). **Permutation** `moe_count_rows<C>(b, experts, counts[E])`, `moe_offsets<C>(b, counts, offsets[E+1])` (exclusive prefix sum), `moe_permutation<C>(b, experts, offsets, order[n], inverse[n])`: the unique stable order of the pairs `j = row·K + slot` by `(expert, j)`, `order[p] = j`, `inverse[j] = p`. **Gather / combine** `moe_gather_rows<C>(b, xs, order, xp[n·d])` (`xp[p] = xs[order[p]/K]`), `moe_gather_rows_backward<C>(b, dxp, inverse, dxs)` (`dxs_r += dxp[inverse[r·K+i]]`, `i = 0..K-1` in order), `moe_combine_rows<C>(b, yp, weights, inverse, ys)` (`y = w_0·yp_0`, then `y = y + w_i·yp_i` in slot order: written, deterministic, no atomics), `moe_combine_rows_backward<C>(b, dys, yp, weights, inverse, dyp, dweights)` (`dyp = w·dy`; `dw` = dot of `yp` and `dy` from 0 in element order). **Products** `grouped_weights<T>{data, experts, rows, cols}`; `grouped_linear(b, w, offsets, xs, ys, bias)` and `grouped_linear_backward(b, w, offsets, xs, dys, dxs, dw, db)`: per expert in order, one `linear_sequence` (`_backward`) on its segment, empty segments skipped; `moe_experts_forward<C>(b, mlp_weights<grouped_weights<T>,T>, offsets, xp, yp, mlp_activations)` and `moe_experts_backward<C>(b, w, offsets, xp, saved, dyp, dxp, mlp_gradients)`: `mlp_forward_sequence` / `mlp_backward_sequence` with grouped products and the dense `mlp_activate_rows` steps. **Shared experts**: one dense MLP of `shared_experts · expert_d_ff` (`moe_shared_config(C)`), the dense passes. **Balancing** `moe_update_bias<C>(b, counts, bias)`: `bias_e += gamma · sign(total - E·count_e)`, decided in integers; `moe_balance_loss<C>(b, logits, experts, dlogits, groups = 1, denominator = 0) -> T`: per group of `T` rows `f_e = c_e/(T·K)`, `P_e = (Σ_t p_te)/T` (softmax, or for `sigmoid` the sigmoids normalised), `L = (alpha·E)·Σ_e f_e P_e`, the loss `Σ L / D`, the gradient `(alpha·E)·f_e/(T·D)` in `p` taken to the logits and added; `moe_z_loss<C>(b, logits, dlogits, denominator = 0) -> T`: `beta·Σ lse²/D`, gradient `(2·beta·lse/D)·softmax` added. A worker holding part of a batch passes the batch's denominator, so its gradients are the single process's to the bit. `moe_load<E>`: counts that `add` and `merge`, `max_over_mean`, `coefficient_of_variation`. **The layer** `moe_weights<T>{router, selection_bias, experts, shared}`, `moe_activations<T>`, `moe_gradients<T>`, sized by `moe_parameter_count(C)`, `moe_activation_values(C, rows)`, `moe_activation_indices(C, rows)`, `moe_gradient_work(C, rows)` and laid out by `moe_weights_in` / `moe_activations_in` / `moe_gradients_in`; `moe_forward<C>(b, w, xs, ys, act, sequences = 1) -> moe_losses<T>`: router `linear_sequence`, route, count, offsets, permutation, gather, experts, combine into `ys`, then `ys = shared_y + ys`, then the weighted losses; `moe_backward<C>(b, w, xs, act, dys, dxs, grads, sequences = 1)`: combine backward, experts backward into a zeroed `dxp`, `dxs +=` the gather's backward, route backward, `+=` the losses' gradients, the router's backward (`dxs += dlogits W`), the shared MLP's backward. The block with an MoE MLP is `ml/moe_block.hpp`: `moe_block_weights`, `moe_block_activations`, `moe_block_gradients` and batched `block_forward_sequence` / `block_backward` overloads. **Expert parallelism** (`ml/expert_parallel.hpp`, the arithmetic; the transport is a library's above core): rank `r` of `P` holds experts `[r·E/P, (r+1)·E/P)` (`expert_rank`, `first_expert`, `experts_per_rank`); each rank routes and permutes its own rows, the ranks exchange their counts, `moe_send_offsets<C>(b, offsets, P, send_offsets[P+1])` cuts its permuted rows by destination, `moe_all_to_all(b, send[P], send_offsets[P], width, receive[P])` delivers to each destination the sources' segments in source order, `moe_receive_layout<C>(b, counts[P][E], r, expert_offsets[E/P+1], order)` gives the order `(expert, source, row, slot)` the rank's grouped products read (`moe_permute_rows` / `moe_unpermute_rows`), and `moe_receive_offsets<C>(b, counts, r, receive_offsets[P+1])` sends the outputs back by the same all-to-all. The rows an expert multiplies are the single process's in the same order, so the outputs, the input gradients and the experts' weight gradients are the single-process layer's to the bit; the router's and shared experts' gradients are each rank's part, summed by `ordered_sum` (T5). The payload may travel as `bfloat16` (`to_bfloat16_rows` / `to_float_rows`, `ml/minifloat.hpp`), the combine in float. A device library overloads each op for its backend and is checked against these. |

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
5. An MoE block in `block_forward_sequence` / `block_backward` (T10, 0.16.0: `ml/moe.hpp`, `ml/moe_block.hpp`, `ml/expert_parallel.hpp`); post-norm and sandwich norms.

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

