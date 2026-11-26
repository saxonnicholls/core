# snicholls-core

[![CI](https://github.com/saxonnicholls/core/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saxonnicholls/core/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-doxygen-blue)](https://saxonnicholls.github.io/core/)

**Documentation:** <https://saxonnicholls.github.io/core/>: the newest release at the root, every version under
`/vX.Y.Z/`, and the latest `main` under `/main/`.

A header-only C++20 library that depends on nothing but the standard library, and puts as much as it can into the
compiler. It holds what the snicholls libraries share: vocabulary types, reflection, exact money and time. It also
holds a large body of numerics in the same style:
- state machines;
- functional programming in the manner of Haskell, OCaml and Rust;
- signal processing, target tracking and optimisation;
- statistics and game-theoretic attribution;
- option pricing;
- the primitives of transformer models.

Almost everything is `constexpr`, allocation-free and checked against a reference implementation. It has
more than 400 headers, each with its tests, a demo that checks itself and its documentation, and five showcase programs that
combine many areas into one program.

## Why

Several libraries, among them ts-moveables and base-encode-decode, had grown their own copies of the same
small things: a lambda overload set for `std::visit`, a string usable as a template parameter, a compile-time
`for`, a way to count the members of a plain struct. Each copy was slightly different, and each library imported
the others' through allowlists and `#ifndef` guards to make sure it got the right one. That is the kind of
duplication that survives, because every copy is correct, until one gets a fix the others do not.

So the shared parts live here, once. The libraries that need them pin a release of core and use the names from
`namespace snicholls`. Core depends on nothing, does no I/O, starts no threads and knows nothing about objects,
sockets or codecs; those are the business of the libraries above it.

## What is here

For how the pieces fit together, start with [the patterns page](docs/PATTERNS.md): each pattern's problem, its
types and compiled examples. For whole programs, see [`showcase/`](showcase/).

### Foundations

- **Vocabulary types:** `overloaded` and `first_match`, `fixed_string`, `function_ref`, `result<V, E>` (Rust's `?`
  as `SNICHOLLS_TRY`) and `traced<E>`, `strong<T, Tag>`, `uuid` (v4, v5, v7), `decimal`, `uint256`,
  `inplace_string<N>`, checked arithmetic, `secret<T>`, E.164 phone numbers, ISO 3166 countries.
- **Compile-time loops, types and algorithms:** `constexpr_for` and `constexpr_nest`, `type_list` (filter, products,
  permutations), `thread_apply`, `zip`, `invoke_with`, and `fill`, `copy`, `iota`, `equal` and `reverse` usable at
  compile time from C++17.
- **Reflection:** member counts, `tie` / `to_tuple` / `from_tuple` over plain aggregates (up to 256 members), enum
  names without a table, type names, and a classification of every string and byte-sequence type with zero-copy
  access.
- **Presentation:** colour, labels, number formats and tooltips declared once as compile-time attributes of a type
  (P&L green up, red down), read by `acout`, thread-safe printing of any value, container or tuple, and exported as
  JSON for imgui and React.
- **Containers:** `inplace_vector`, `static_map` and `perfect_map` built by the compiler, `md_array` and `md_span`
  (`submdspan`), a k-d tree with bounded, tie-ordered queries, an approximate-nearest-neighbour forest, a Bloom
  filter, a hierarchical bitset, an intrusive AVL tree, id maps and object pools, `byte_map` for a BPE vocabulary,
  and the platform-free half of mapped storage (`index_pool`, `mapped_header`, `offset_ptr`).
- **Hashing, at compile time:** SHA-1, SHA-2, SHA-3 and Keccak-256, BLAKE2 and BLAKE3, RIPEMD-160, CRC-32/32C/64,
  FNV-1a and wyhash; HMAC, HKDF, PBKDF2 and scrypt; constant-time compare and a wipe the optimiser may not remove.

### Money, time and markets

- **Exact money:** `money` (256-bit, any asset), ISO 4217 currencies and the major chains' native assets and tokens
  as types, exact rates and conversion with a named rounding, exact ratios and year fractions.
- **Time:** `utc<D>` with RFC 3339 text, identical on every platform (checked against fixtures generated from an
  independent implementation), ISO 8601 week dates, civil calendar arithmetic, business days, and the sun's position.
- **Market conventions:** day counts (ACT/360, ACT/365F, 30/360, 30E/360, ACT/ACT ISDA) with exact year fractions,
  business-day rules over the caller's calendar, dated cash flows and the cash ladder, time- and money-weighted
  returns, and the check digits of ISIN, IBAN, LEI, CUSIP, SEDOL and Luhn.
- **Option pricing by the compiler:** Black-Scholes-Merton and its greeks in closed form, binomial trees for American
  exercise on the Snell envelope, P&L explained by greek and factor.

### State machines and functional programming

- **State machines** (`fsm/`) in sml's notation, `*state<A> + event<E> [guard] / action = state<B>`, checked by the
  compiler. They are saved and loaded through any key-value store, drawn as Mermaid, DOT or PlantUML, and come with
  the reference lifecycles of payments, trades, fund applications and corporate actions. The demos also run them
  functionally: as a fold over events, as event sourcing, as Mealy machines, as a railway of `result`s and as a
  reachability check at compile time.
- **Functional programming** (`functional/`), all `constexpr`:
  - Haskell's type classes as concepts (Semigroup, Monoid, Functor, Applicative, Monad, Foldable), with instances for
    `optional` (Maybe), `result` (Either), tuples, `array`, `vector`, strings and `inplace_vector`;
  - the Reader, Writer and State monads;
  - folds, scans and unfolds, and recursion schemes (`cata`, `ana`, `hylo`, `para`) with a trampoline;
  - lazy values and infinite streams;
  - currying, placeholders and pattern matching with guards;
  - Data.List over fixed-size arrays, and type-level `map` / `filter` / `fold`;
  - OCaml's Option, Result and List modules, and Rust's `Option` methods, iterators and slices;
  - Wolfram-style rewriting by rules (`/.` and `//.`), and functions lifted pointwise into an algebra.

### Mathematics and numerics

- **Maths:** `constexpr` elementary and special functions (gamma, erf, Lambert W, Bessel, the normal quantile) in any
  floating-point type, root finding (Brent, safeguarded Newton), and complex numbers usable at compile time.
- **Linear algebra:** an interface resolved at compile time (BLAS operations, LU, Cholesky, LDLT, QR, least squares,
  the symmetric eigenproblem, SVD, batched solves with a status per system) over `md_span` views, a
  standard-library reference backend, and a conformance suite for backends in other libraries. On top of it:
  - fixed-size matrices for state estimation;
  - the conjugate-gradient and MINRES solvers;
  - the randomised truncated SVD.

  A GPU backend (CUDA and OpenCL) is [snicholls-gpu](https://github.com/saxonnicholls/gpu).
- **Sparse matrices and optimisation:** CSC matrices, AMD and reverse Cuthill-McKee orderings, a sparse LDLᵀ of
  quasi-definite systems; L-BFGS, Armijo and strong-Wolfe line searches, proximal operators, projections onto the
  cones of conic optimisation, and Ruiz equilibration.
- **Signal processing:** FFTs of any length at compile time and at run time, the DCT and DST, convolution, FIR design
  by windowed sinc, IIR biquads and Butterworth sections, Goertzel detectors, the Hilbert transform, polyphase
  resampling, the STFT and its inverse, and mel spectrograms. Each matches scipy, librosa or torch.
- **Statistics:** random generators that reproduce on every standard library, the bootstrap (by counts, balanced,
  stationary for returns), Sobol and Halton sequences, streaming accumulators that merge across workers, covariance
  with Ledoit-Wolf shrinkage, RANSAC, particle resampling, and one-pass pivot tables whose keys and measures are
  types.
- **Stochastic processes:** martingale tools (the Doob decomposition, quadratic variation, Markov chains) and the Snell
  envelope of optimal stopping.
- **Game theory and attribution:** Shapley values exactly or by sampling, Owen values for groups, Harsanyi dividends,
  the properties of a game, and "why did the value move" over any model.
- **Combinatorics:** permutations, partitions, subsets and tuples at compile time, and the linear assignment problem.

### Tracking, geometry and distances

- **Target tracking** (`tracking/`):
  - Kalman, extended, unscented and cubature filters, with the information form;
  - motion models (constant velocity, Singer, constant turn) and sensor models (bearing-range and others);
  - probabilistic and joint probabilistic data association;
  - covariance intersection for fusion with unknown correlation;
  - Gaussian-mixture reduction, and hidden-Markov filters for a target's class.
- **Geometry:** WGS 84 geodetic, ECEF and local ENU coordinates, polar and spherical frames, rigid and similarity
  alignment (Umeyama), boxes with IoU, GIoU and non-max suppression, and pinhole cameras.
- **Metrics:** Minkowski, cosine and angular, Hamming and Jaccard, haversine, Mahalanobis, Hellinger and KL, the
  edit distance, and the set distances OSPA, GOSPA and Hausdorff. Each one plugs into the k-d tree and the
  nearest-neighbour forest.

### Capability contracts

- **Interfaces** (`concepts/`, `interfaces/`): 34 capabilities, each a concept and its virtual twin, with a reference
  implementation and a conformance kit that other libraries' implementations must pass. They include payments, key
  stores and signers, token verification, key-value stores, HTTP and SQL clients, random sources, uuid generators,
  time zones, task pools and loop clocks.

### Machine learning (experimental)

- **The primitives:**
  - activations with their derivatives, softmax and cross-entropy, and a running softmax that merges across workers;
  - LayerNorm and RMSNorm, RoPE, and one-pass attention with mergeable partials;
  - embeddings and sampling;
  - bfloat16, float16, FP8 and the MX formats, and ggml's quantised blocks (Q8_0, Q4_0, MXFP4, the K-quants, the
    IQ formats);
  - byte-level BPE that gives tiktoken's tokens, SentencePiece, WordPiece and unigram tokenizers.
- **The layers:** transformers configured at compile time for GPT-2, Llama and nanochat shapes, with a KV cache
  (flat, ring or paged) for decoding. There are also mixture-of-experts, latent attention, state-space and
  linear-attention mixers, LoRA and speculative decoding.
- **Training:** the backward pass of every primitive, checked against llm.c and central differences, AdamW and Muon,
  and DiLoCo.

All of `ml/` is `constexpr` and allocation-free.

`CHANGELOG.md` lists every header by release.

### Tokenizer throughput

`ml/bpe.hpp` against tiktoken and rustbpe. The test text is 12.3 MB of Python source, 2.8 million pieces. Every
token matched tiktoken's on both vocabularies.

| Vocabulary | snicholls-core | tiktoken 0.14 | rustbpe 0.1 |
|---|---|---|---|
| cl100k_base (GPT-4) | **9.2 M tokens/s** (37 MB/s) | 2.3 M tokens/s | - |
| 32,768 tokens, trained by rustbpe | **11.0 M tokens/s** (47 MB/s) | 2.7 M tokens/s | 0.9 M tokens/s |

How it was run:

- **Machine and build:** one thread on a Xeon W-3245, Release `-O3 -march=native`, Apple clang 17, 2026-11-11.
- **Core:** splits the text with PCRE2 JIT, using each vocabulary's own pattern, then merges with `bpe_encode_piece`
  and an open-addressing vocabulary. The merge alone runs at 19 M tokens/s with cl100k and 26 M tokens/s with the
  32K vocabulary; splitting takes the other half of the time.
- **tiktoken and rustbpe:** called from Python, one document per call. Their times include building the Python list
  of tokens.

## Using it

As a git submodule, pinned to a release tag:

```cmake
add_subdirectory(third_party/core EXCLUDE_FROM_ALL)
snicholls_core_require(0.28.0)
target_link_libraries(my_target PRIVATE snicholls::core)

# last line of the top-level CMakeLists.txt
snicholls_core_assert_single_version()
```

Installed, with `cmake --install`:

```cmake
find_package(snicholls_core 0.28 REQUIRED)
target_link_libraries(my_target PRIVATE snicholls::core)
```

In code:

```cpp
#include <snicholls/core/version.hpp>

SNICHOLLS_CORE_REQUIRE_VERSION(0, 28, 0);   // fails to compile against an older core
static_assert(snicholls::core::version() >= SNICHOLLS_CORE_VERSION_CHECK(0, 28, 0));
```

### Two copies in one build

A library that pins core and also pins another library that pins core can reach it twice in one build. Core handles
that itself: the first copy added defines the target and later copies step aside; every consumer states the minimum
it needs with `snicholls_core_require` and configure fails if the winning copy is older; and
`snicholls_core_assert_single_version` fails configure if copies at two different versions were reached. The rules
are tested, not just documented: `ctest -L boundary`.

## Building and testing

```sh
git clone --recurse-submodules <url>
make test            # configure, build, every test, every demo
make test-werror     # as CI does it: warnings are errors
make asan            # Address + UB sanitizers
make docs            # Doxygen, warnings are errors; output in build-docs/docs/html
make showcase        # the showcases, with their output
```

Every header has a demo (`demos/`, mirroring `include/`). The showcases (`showcase/`) are whole programs, each
weaving many areas together: a trading day (exact money, the trade lifecycle, a P&L pivot, Shapley and Owen
attribution of a compile-time Black-Scholes revaluation), a shop's payments on the capability contracts' reference
implementations (uuid v7, result railways, retries, webhooks, reconciliation, a cash ladder), a harbour crossing
(Goertzel tone detection, a k-d tree, a state machine, merged accumulators), an option priced four ways (closed form
and Snell envelope at compile time, Monte Carlo with a bootstrap, Sobol quasi-Monte Carlo) and a report the compiler
writes in the functional style (reflection, curry, fold, match, Maybe, Either and State). Each checks itself and is
a CTest (`ctest -L showcase`).

CI runs the same on Linux (x86-64 and ARM64, GCC and Clang), macOS (ARM64 and x86-64) and Windows (MSVC), plus a
Debug build with asserts live and the documentation build. Every leg blocks except Windows, which is reported but
not gated on while no consumer builds there.

The standards the code is held to - what may be admitted, the mirror rule for tests and demos, the documentation
standard - are in `CLAUDE.md`.

## Licence

MIT. Copyright 2010-2026 Saxon Nicholls.
