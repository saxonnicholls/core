# snicholls::ml - the transformer layer for inference

Status: in progress, for 0.8.0. Headers: `include/snicholls/core/ml/` (standard library and core only). Training
builds on the same headers and is planned in `TRAINING.md`; the decisions here are made so it does not have to
change them.

## Why

`ml/` has the arithmetic of a transformer for one row or one query at a time: `attend` and
`attention_accumulator`, `rope`, `layer_norm` / `rms_norm`, the activations, `embedding`, `softmax`, `sampling`, the
quantised blocks. Nothing ties them into heads, a layer and a cache, so every model that uses them wires them by
hand. This plan adds that wiring once, generic over the architecture, with the architecture a compile-time value.

## Decisions

| Id | Decision |
|---|---|
| I1 | **The architecture is a compile-time value.** `transformer_config` is a structural aggregate passed as a template parameter: `d_model`, `n_heads`, `n_kv_heads`, `head_dim`, `d_ff`, `norm` (layer / RMS), `qk_norm`, `mlp` (GELU, tanh GELU, ReLU², SwiGLU), `rope` (none / half / interleaved) and `rope_base`, `window` (0 = causal over everything), `attention_softcap`, `bias`, `norm_eps`. `consteval validate(c)` rejects an inconsistent one (`n_heads % n_kv_heads`, odd `head_dim` with RoPE). Every size is then a constant: scratch is fixed-size, loops are bounded, nothing is allocated, and a configuration is synthesisable. MHA, GQA and MQA are the same code with `n_kv_heads` = `n_heads`, a divisor, or 1. |
| I2 | **Inference first; training later on the same headers** (`TRAINING.md`). Forward takes a compile-time `save` policy (`save::none` now; `save::all` for training), so adding the backward pass does not fork the forward. |
| I3 | **The block over T tokens is the primary forward; decode is T = 1.** `block_forward` processes a run of consecutive positions. Its keys and values go into a `kv_cache` and attention reads them back from it, so prefill (T > 1) and decode (T = 1) are one code path, and training (no persistent cache) is the same path over a cache the size of the sequence. |
| I4 | **Projections are `ml/linear.hpp`, over any weight storage.** A `weight_matrix` concept (`rows`, `cols`, `row_dot(i, x)`) with two adapters: `dense_weights<T>` (float, double, `bfloat16` rows) and `quantized_weights<Block>` (Q8_0, Q4_0, MXFP4; the input is quantised to Q8_0 once per call, as ggml does). The linear algebra backend in development may replace the matrix-vector loop later; the concept is the seam. |
| I5 | **The KV cache is laid out `[layer][kv_head][position][head_dim]`,** so the keys of one head are contiguous, which is what `attend` reads. Two forms, as `byte_map`: `kv_cache_view<Config>` over the caller's spans, and `kv_cache<Config, Layers, Capacity>` owning fixed-size arrays. A sliding window is a ring buffer: a wrapped run is two contiguous runs whose `attention_accumulator` partials `merge`. |
| I6 | **Attention never materialises the T x T scores.** Each query attends with the one-pass recurrence (`attend`), keeping one log-sum-exp per query and head. The GQA mapping is query head `h` reads KV head `h / (n_heads / n_kv_heads)`. |
| I7 | **The heads are the unit of parallelism.** A per-head entry point (`attention_head`) lets the caller run heads on threads or devices; a split over keys merges partials. Core runs them serially; ts-moveables or a device runs them in parallel. |
| I8 | **Weights are non-owning views.** `attention_weights<Config, W>`, `mlp_weights<Config, W>`, `block_weights<Config, W>`: spans and weight-matrix adapters the caller fills from wherever the weights live. Gradients (training) are the same templates over mutable float storage. |
| I9 | **Core stops at the block.** A whole model (the stack of blocks, the embedding, the LM head), weight files (GGUF, safetensors), reading tokenizer files and the generate loop belong in a machine-learning library above core, which pins core and ts-moveables. (Revised in 0.17.0 by I12: the tokenizers' algorithms are core.) |
| I10 | **References, never core's own output.** A generator links llm.c's forward functions for a tiny random GPT-2 and writes a committed fixture; a second, written in Python with float32 arithmetic, covers what llm.c lacks (GQA, QK norm, ReLU², SwiGLU, RoPE, soft cap). Both are drift-checked like the time fixtures. Where the order of summation differs from the reference (llm.c's attention over a whole sequence), the tolerance is stated in ulps and justified in the test. |
| I11 | **No fused multiply-add assumed.** Each multiply and add is its own statement, and the tests build with `-ffp-contract=off` (see `ml/quantize.hpp`), so results match across x86 and ARM. ARM64 runs of the fixture tests are checked before a push. |
| I12 | **Tokenizers are core; tokenizer files are not.** Every algorithm between text and token ids is a standard-library-only, allocation-free building block with a compile-time form: the Unicode tables (`ml/unicode.hpp`, generated from UnicodeData.txt 16.0 and cross-checked against llama.cpp's), the split patterns of each model family as hand-written matchers over categories with no regex engine (`ml/pretokenize.hpp`), the vocabulary as a sorted-array trie (`ml/token_trie.hpp`), merge-list BPE with GPT-2's byte alphabet (`ml/bpe_merges.hpp`), SentencePiece BPE (`ml/sentencepiece.hpp`), Unigram with its precompiled charsmap read in place (`ml/unigram.hpp`), WordPiece (`ml/wordpiece.hpp`), RWKV (`ml/rwkv.hpp`), and the whole pipeline over a non-owning `tokenizer_vocab` (`ml/tokenizer.hpp`). The reference is llama.cpp's tokenizer on its own vocabulary test vectors (whose expected tokens are Hugging Face's), token for token; where core departs from llama.cpp it follows Hugging Face (BERT's Hangul, GPT-4o's case classes) or strict UTF-8 (an encoded surrogate), and the fixtures record the difference. Parsing GGUF metadata and `tokenizer.json`, and choosing the options from them, is the loaders' (a library above core); they fill `tokenizer_vocab` with views of their own memory. |

## Headers

| Header | Contents |
|---|---|
| `transformer_config.hpp` | `transformer_config`, `norm_kind`, `mlp_kind`, `rope_kind`, `validate`, derived sizes (`q_size`, `kv_size`, `group`, `scale`), presets `gpt2_small`, a nanochat-style and a Llama-style config |
| `linear.hpp` | `weight_matrix`, `dense_weights<T>`, `quantized_weights<Block>`, `linear(w, x, y)`, `linear(w, bias, x, y)` |
| `mlp.hpp` | `mlp_weights<Config, W>`, `mlp_forward`: `down(act(up x))`, or SwiGLU `down(silu(gate x) * up x)` |
| `kv_cache.hpp` | `kv_cache_view<Config>`, `kv_cache<Config, Layers, Capacity>`, `append`, `keys` / `values` as one or two runs, `required_floats` |
| `attention_layer.hpp` | `attention_weights<Config, W>`, `attention_head`, `attention_forward`: project, QK norm, RoPE, append, GQA, `attend`, output projection |
| `transformer_block.hpp` | `block_weights<Config, W>`, `block_scratch<Config>`, `block_forward`: `x += attn(norm1(x)); x += mlp(norm2(x))` |
| `unicode.hpp` | `decode_utf8`, `encode_utf8`, `category` (Unicode 16.0) and its groups, `is_whitespace`, `to_lower`, `decompose`, `combining_class`, `replace_invalid_utf8` |
| `pretokenize.hpp` | the split patterns (GPT-2, Llama 3, Qwen 2 and 3.5, GPT-4o, DeepSeek V3, LLM and Coder, Falcon, StarCoder, Gemma 4), `split_isolated`, `split_sequence`, `pretokenize`, `pretokenizer_for`, `split_added_tokens` |
| `token_trie.hpp` | `token_trie`, `build_token_index`, `static_token_trie<N>`, `token_kind` |
| `bpe_merges.hpp` | `merge_encode`, `merge_table`, `build_merge_index`, `split_merge`, `byte_to_unicode`, `byte_level_encode` / `_decode` |
| `sentencepiece.hpp` | `sentencepiece_escape`, `sentencepiece_bpe_encode`, `byte_fallback`, `sentencepiece_decode` |
| `unigram.hpp` | `precompiled_charsmap`, `unigram_normalize`, `unigram_encode` |
| `wordpiece.hpp` | `wordpiece_encode`, `wordpiece_decode` |
| `rwkv.hpp` | `rwkv_encode`, `rwkv_unescape` |
| `tokenizer.hpp` | `tokenizer_vocab` (the loaders' seam), `tokenize`, `detokenize`, `tokenizer_scratch_need` |

Also: `attend` with a logit soft cap; logit processors in `sampling.hpp` (repetition, presence and frequency
penalties, `min_p`); `top_k_gating<K>` for a mixture-of-experts router.

## Order of work

1. `transformer_config`, `linear` with both adapters, `mlp`.
2. `kv_cache`, flat then ring; the soft cap in `attend`.
3. `attention_layer`, `transformer_block` over T tokens; decode as T = 1.
4. The llm.c generator (forward and backward fixtures, so `TRAINING.md` has its references) and the Python
   generator; the tests against them.
5. Logit processors, the MoE router.
6. Benchmarks (the decode step of a GPT-2-small layer against the same loops by hand; float, Q8_0, Q4_0 weights),
   demos, documentation, CLAUDE.md idioms; the local gate, Ubuntu, an ARM64 run; release 0.8.0.
