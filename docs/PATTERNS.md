# Patterns {#patterns}

How the pieces of snicholls-core fit together. Each pattern names the problem, the types that solve it, a compiled
example (every snippet below is cut from a demo that runs as a CTest, so none of it can rot) and when not to use it.
The per-header pages hold the detail; this page is the map. The programs in `showcase/` put many of these patterns
to work at once.

[TOC]

## Principles behind every pattern

- **Decide at compile time.** Tables, maps, attribution weights and stopping regions are built by the compiler
  (`consteval` constructors, `constexpr` functions); the hot path is a lookup or a few arithmetic operations.
- **Standard library only, header-only, C++20.** Nothing here depends on a library above core.
- **Device-ready primitives.** Pure `constexpr` functions, spans of plain data in, trivially copyable state, partial
  results that `merge`, randomness by stream or by index: callable from a GPU kernel, synthesizable for an FPGA in
  the fixed-size forms.
- **Zero overhead, proved.** Anything with a run-time footprint carries `static_assert`s on its layout and a
  like-for-like benchmark against the hand-written equivalent.

---

## Errors

### Expected failure is a value: `result<V, E>`

**Problem.** A parse, a lookup or a call to a provider can fail in the ordinary course of things; `bool` plus an
out-parameter loses the reason, and an exception hides the path.

**Pieces.** `snicholls::result<V, E>`, `ok(v)` / `err(e)`, `SNICHOLLS_TRY` (Rust's `?`), and the portable
`SNICHOLLS_TRY_ASSIGN` / `SNICHOLLS_TRY_VOID`. A result of trivial types is trivially copyable and travels in
registers.

\snippet result_demo.cpp try

**Not for** the unexpected (a broken invariant): that stays an exception or an assertion.

### Errors for logs: `traced<E>`

**Problem.** An error that reaches a log needs where it happened and what was going on, without paying for that on
the success path.

**Pieces.** `snicholls::traced<E>`: the error plus a source location and lazily allocated context, fields and stack;
`error_sink` and `log_err(sink)` hand it to the logger the program chose.

\snippet traced_demo.cpp chain

**Not for** the wire: send `error()` or the report fields, never the trace. Never put a secret in context text.

---

## Capabilities: a contract, its twin, a reference and a kit

**Problem.** A library needs "something that can sign", "a key-value store", "a source of random bytes" without
depending on the library that implements it, and every implementation must behave the same.

**Pieces.** For each capability core holds four things:

1. a **concept** (`random_source_like`), for static dispatch with no cost;
2. a **virtual twin** (`random_source_interface`) with adapters both ways (`random_source_adapter<Impl>`,
   `any_random_source`), for when the implementation is chosen at run time;
3. a **reference implementation** (`test_random_source`), deterministic and `constexpr` where it can be, for tests;
4. a **conformance kit** (`check_random_source`), which every real implementation, in whichever library, must pass.

Failures are `capability_errc`, and nothing secret is ever written into an error.

\snippet concepts/random_source_demo.cpp concept

\snippet interfaces/random_source_demo.cpp interface

\snippet interfaces/random_source_reference_demo.cpp reference

\snippet interfaces/random_source_conformance_demo.cpp kit

The same shape serves the payment, on-ramp, signer, verifier, key store, key-value store, SQL, HTTP, message,
remote-file, task-pool, clock, time-zone and UUID capabilities under `interfaces/`.

**Not for** a helper with one obvious implementation: write the function.

---

## State machines

### A machine is a table

**Problem.** Lifecycles (an order, a payment, a security, a session) are where bugs hide when written as flags and
`if`s.

**Pieces.** `fsm/state_machine.hpp`, in the style of Boost.SML: a transition table of `state + event [guard] / action
= next`, checked by the compiler, with composite and orthogonal regions, history, deferral, logging and visitors.

\snippet fsm/state_machine_hello_world_demo.cpp table

\snippet fsm/state_machine_hello_world_demo.cpp use

### Save, load and draw it

A machine's state saves and restores, so a lifecycle survives a restart; the same table draws itself.

\snippet fsm/persist_demo.cpp save_load

\snippet fsm/diagram_demo.cpp mermaid

### Lifecycles that read as the domain

\snippet fsm/lifecycle_demo.cpp table

### Machines, functionally

A machine is a function from state and event to state, so everything in `functional/` applies to it.

- **A run is a fold**, and its trace a scan:

\snippet fsm/state_machine_fold_demo.cpp fold

- **Event sourcing is a law**: a snapshot plus the remaining events equals the full replay, proved by `static_assert`:

\snippet fsm/state_machine_event_sourcing_demo.cpp law

- **A Mealy machine is the State monad**: the same machine as a table and as `fn::state`, identical on every input:

\snippet fsm/state_machine_mealy_demo.cpp proof

- **Guards from combinators, transitions on the railway:**

\snippet fsm/state_machine_railway_demo.cpp guards

\snippet fsm/state_machine_railway_demo.cpp railway

- **Reachability is a fixed point** of the transition relation:

\snippet fsm/state_machine_reachability_demo.cpp fixed_points

**Not for** a two-state toggle: a `bool` is fine.

---

## Presentation is an attribute of the type

**Problem.** Colour, labels, number formats and tooltips written at each call site drift apart, and P&L coloured by
hand is eventually coloured wrong.

**Pieces.** A type or enumeration declares a `constexpr` function `attributes` in its own namespace returning
`attrs(...)` (`meta/attributes.hpp`): `colour`, `sign_colours`, `label`, `number_format` (checked by the compiler),
`tooltip`. `acout`, `print_to`, containers and `stats::pivot` read them, and `write_enum_json` exports them for a user
interface, so each is written once.

\snippet meta/attributes_demo.cpp attributes

\snippet meta/attributes_demo.cpp attributes_print

\snippet stats/pivot_demo.cpp pivot_colour

**Rule.** Green and red mean up and down, never an identity.

---

## Printing

### One message, one write

**Problem.** Lines from several threads interleave mid-line.

**Pieces.** `acout{} << ...`, `acerr{}`, `print(...)` / `println(...)`: each message is built whole and written under
one lock. Containers, tuples, optionals, variants and results print as they are, recursively.

\snippet acout_demo.cpp print

\snippet acout_demo.cpp containers

### A type that prints itself

One member, `void print(snicholls::acout&) const`, and nothing else: no per-type `operator<<`.

\snippet acout_demo.cpp printable

\snippet acout_demo.cpp printable_use

---

## Dispatch over types and values

### Overload sets, never `if constexpr` chains

**Problem.** A chain of `if constexpr` over a type is hard to read and impossible to extend from outside.

**Pieces.** `overloaded{...}` for cases that do not overlap; `first_match{...}` with concept-constrained lambdas
(`meta/concepts.hpp`) when they do: the first that accepts the argument runs, in the order written.

\snippet overloaded_demo.cpp first_match

\snippet overloaded_demo.cpp visit

### Values with `constexpr_for`, types with `type_list`

If what you loop over is a value, `constexpr_for`; if it is a type, `type_list`.

\snippet constexpr_for_demo.cpp tuple

\snippet meta/type_list_demo.cpp demo

\snippet thread_apply_demo.cpp thread_apply

### Arguments that exist only at run time

A command line or a JSON object calls a typed function through `invoke_with`: the source supplies `size()` and
`get<T>(i)`.

\snippet invoke_with_demo.cpp commands

---

## Reflection over plain aggregates

**Problem.** Serialising, hashing, comparing or printing a struct field by field by hand.

**Pieces.** `member_count_v<T>`, `tie` / `to_tuple` / `from_tuple` (`reflect/struct_tuple.hpp`), `type_name`, scoped
enumerations by name (`meta/enums.hpp`).

\snippet reflect/struct_tuple_demo.cpp to_tuple

\snippet meta/enums_demo.cpp enums

**Not for** classes with invariants, bases or reference members: give those an explicit interface.

---

## Tables built by the compiler

**Problem.** Lookups that are known when the program is written should cost nothing to build and little to query.

**Pieces.** `constexpr_map` (an enumeration's names, both ways), `containers::static_map` and `perfect_map` (built in
`consteval` constructors), `fnv1a_64` for a hash that can be a `switch` label.

\snippet constexpr_map_demo.cpp table

\snippet containers/perfect_map_demo.cpp perfect_map

\snippet hash/fnv_demo.cpp switch

---

## Exact numbers, time and identity

### Money is exact

`money` (asset plus signed minor units), `decimal` (exact, one canonical text), `uint256` for on-chain amounts,
`checked` arithmetic that reports overflow. Never `double` for an amount.

\snippet money/money_demo.cpp money

\snippet decimal_demo.cpp decimal

\snippet uint256_demo.cpp amounts

\snippet checked_demo.cpp checked

### Time is explicit

`utc<D>`: one `sys_time`, RFC 3339 text, the same ticks and fields on every platform. "Now" is read only through
`utc<D>::now()`; default construction is the epoch.

\snippet time/utc_demo.cpp demo

### Identity is a type

`uuid` (v4 / v7 from bits the caller supplies, v5 from a name, a checked `_uuid` literal) and `strong<T, Tag>` so an
order id never passes for a client id.

\snippet uuid_demo.cpp demo

\snippet strong_demo.cpp strong

### Secrets do not leak

\snippet secret_demo.cpp masking

---

## Dated money: one cash flow

`finance::cash_flow` is one dated, exact, signed flow (positive in, negative out) shared by the liquidity ladder and
the return measures.

\snippet finance/cash_flow_demo.cpp cash_flow

\snippet finance/cash_ladder_demo.cpp ladder

\snippet finance/returns_demo.cpp twr

---

## Statistics that merge

**Problem.** Recomputing a statistic from history on every tick, or a `std::map<std::map<...>>` loop to group and
aggregate.

**Pieces.** Accumulators (`stats::variance`, `covariance`, `ewma`, ...) that update in O(1) and `merge` across
workers (a parallel reduction); `stats::pivot<Row, row keys, column keys, measures...>` groups and aggregates in one
pass.

\snippet stats/accumulators_demo.cpp accumulators

\snippet stats/pivot_demo.cpp pivot

### Randomness that reproduces

`stats::wyrand` by stream, `uniform_index` (never `std::uniform_int_distribution`, whose output differs between
standard libraries), Sobol points for quasi-Monte Carlo, the bootstrap for error bars.

\snippet stats/random_demo.cpp random

\snippet stats/low_discrepancy_demo.cpp sobol

\snippet stats/bootstrap_demo.cpp bootstrap

---

## Attribution: who moved the number

**Problem.** One-at-a-time and sequential attribution depend on the order you ask in.

**Pieces.** A game is a callable from `game::coalition` to a value; `game::shapley<N>` exactly to about 20 players,
`game::owen` with groups beyond, the samplers when evaluations are expensive, `finance::explain` for P&L by greek and
factor.

\snippet game/shapley_demo.cpp shapley

\snippet game/owen_demo.cpp owen

\snippet finance/attribution_demo.cpp greeks

---

## Functional programming, at compile time

`functional/` (namespace `snicholls::fn`) brings the vocabulary of Haskell, OCaml and Rust to C++20, every piece
`constexpr`.

### Functions that return functions

\snippet functional/curry_demo.cpp curry

\snippet functional/curry_demo.cpp partial

\snippet functional/combinators_demo.cpp compose

\snippet functional/bind_demo.cpp bind

### Folds, unfolds and laziness

\snippet functional/fold_demo.cpp fold

\snippet functional/fold_demo.cpp unfold

\snippet functional/lazy_demo.cpp stream

\snippet functional/recursion_demo.cpp schemes

### Type classes and effects

\snippet functional/typeclasses_demo.cpp monoid

\snippet functional/monad_demo.cpp sequence

\snippet functional/state_demo.cpp state

\snippet functional/match_demo.cpp variant

### OCaml's pipes and functors

\snippet functional/ocaml_demo.cpp pipeline

\snippet functional/ocaml_demo.cpp functor

### Parsers from combinators

\snippet functional/typeclasses_parser_demo.cpp parser

### Classics, written in C++

The showcases are the programs a functional programmer reaches for first, each checked by `static_assert` as well as
at run time.

\snippet functional/lazy_showcase_demo.cpp primes

\snippet functional/lazy_showcase_demo.cpp fibonacci

\snippet functional/list_showcase_demo.cpp queens

\snippet functional/instances_list_monad_demo.cpp knight

\snippet functional/fold_showcase_demo.cpp horner

\snippet functional/recursion_showcase_demo.cpp cata

\snippet functional/combinators_showcase_demo.cpp point_free

\snippet functional/curry_showcase_demo.cpp sections

### An interpreter: algebraic data types and pattern matching

\snippet functional/match_evaluator_demo.cpp adt

\snippet functional/match_evaluator_demo.cpp run

### Effects: railway, state, reader and writer

\snippet functional/monad_railway_demo.cpp railway

\snippet functional/state_rpn_demo.cpp rpn

\snippet functional/typeclasses_map_reduce_demo.cpp words

\snippet functional/ocaml_pipes_demo.cpp sum_of_squares

### Rust's vocabulary

`std::optional` with Rust's Option methods, orderings that chain, slices in windows and chunks, iterator searches with
Rust's tie rules. `result` already carries Result's methods as members.

\snippet functional/rust_demo.cpp option

\snippet functional/rust_demo.cpp cmp

\snippet functional/rust_iter_demo.cpp slices

\snippet functional/rust_showcase_demo.cpp pipeline

### Rewriting and point-wise functions

Rules applied once (`/.`) or to a fixed point (`//.`) over any tree whose children the caller can map; functions
combined point by point with ordinary operators.

\snippet functional/rewrite_simplifier_demo.cpp simplify

\snippet functional/pointwise_demo.cpp payoff

**Not for** a hot inner loop the compiler cannot see through: measure, then write the loop.

---

## Signals at compile time

Coefficients, windows and whole transforms evaluate in the compiler when their input is known, and the same code runs
at run time.

\snippet signal/goertzel_demo.cpp compile_time

\snippet signal/goertzel_demo.cpp dtmf

\snippet signal/fft_demo.cpp compile_time

---

## Fixed sizes for the device

Sizes known at compile time, bounded loops, no allocation: the forms an FPGA can synthesize and a GPU kernel can call.

\snippet containers/inplace_vector_demo.cpp inplace_vector

\snippet containers/md_array_demo.cpp md_array

\snippet containers/kd_tree_demo.cpp kd_tree

---

## Containers keyed by identity

**Problem.** Pointers into a growing vector dangle, and a `std::unordered_map` from id to object is slow and wasteful.

**Pieces.** `index_pool` hands out dense indices and recycles them; `dense_id_map` / `flat_id_map` hold objects by id
contiguously; `object_pool` reuses storage; `bloom_filter` sits in front of an authoritative set when nearly every
query misses.

\snippet containers/index_pool_demo.cpp pool

\snippet containers/dense_id_map_demo.cpp dense

\snippet containers/bloom_filter_demo.cpp watch

---

## Hashing: the right hash for the job

`fnv1a_64` when a tiny hash must agree across machines or be a `switch` label; `wyhash` for speed over longer keys;
`hash_combine` only for `std::hash` values that never leave the process; SHA-2, SHA-3, BLAKE2/3, HMAC and the KDFs
when it must be cryptographic. All `constexpr`.

\snippet hash/wyhash_demo.cpp hash

\snippet hash/hash_combine_demo.cpp struct

\snippet hash/sha2_demo.cpp sha256

---

## Linear algebra: one interface, a backend chosen by type

**Problem.** The same numerics must run on a reference CPU path, a tuned library and an accelerator, without a copy of
each algorithm per target.

**Pieces.** `linalg/` operates on views of plain data, with the backend a template parameter (or chosen at run time
where it must be); fixed-size forms for the device; batched execution for many small problems; Krylov solvers and
randomized SVD built on the same operations.

\snippet linalg/backend_demo.cpp backend

\snippet linalg/backend_demo.cpp run_time_choice

\snippet linalg/fixed_demo.cpp fixed

\snippet linalg/batched_demo.cpp batched

---

## Machine learning as small, composable kernels

`ml/` is a transformer, its training and its inference, built from pure functions over rows of plain data: each
layer has a forward and a backward, the backend is a template parameter, and a configuration composes the blocks.

\snippet ml/transformer_config_demo.cpp transformer_config

\snippet ml/transformer_block_demo.cpp transformer_block

\snippet ml/transformer_block_demo.cpp own_backend

\snippet ml/quantize_demo.cpp formats

\snippet ml/tokenizer_demo.cpp tokenizer

\snippet ml/grammar_demo.cpp grammar_mask

\snippet ml/muon_demo.cpp train_gpt2

---

## Optimisation, sparse systems and distances

\snippet optim/lbfgs_demo.cpp lbfgs

\snippet optim/prox_demo.cpp prox

\snippet sparse/ldl_demo.cpp ldl

\snippet metric/concepts_demo.cpp concepts

---

## Estimation: tracking a moving thing

Gaussian state, transition and measurement models, Kalman and unscented filters, association and fusion, each a
`constexpr` step over trivially copyable state.

\snippet tracking/kalman_demo.cpp kalman

\snippet tracking/fusion_demo.cpp fusion

---

## Stochastic control and pricing

The Snell envelope prices the right to stop (an American option) over a fixed-size lattice; martingale checks keep a
model honest.

\snippet stochastic/snell_demo.cpp snell

A recombining lattice (binomial, trinomial) is a fixed stencil, not a matrix: `lattice_envelope<Steps>` keeps one row,
O(Steps) state in a `std::array`, and returns the value and the early-exercise boundary.

\snippet stochastic/snell_demo.cpp lattice

A vanilla option is priced by `finance::black_scholes` and its greeks by `finance::black_scholes_greeks`, in closed form
and in the option's floating type, never a hand-written formula or greeks by finite differences; `finance::binomial`
prices American exercise on a CRR tree solved by that lattice envelope.

\snippet finance/option_pricing_demo.cpp black_scholes

\snippet finance/option_pricing_demo.cpp binomial

\snippet stochastic/martingale_demo.cpp martingale

---

## Orderings, subsets and assignments

Orderings come from `index_permutations` / `for_each_permutation`, never a hand-rolled table.

\snippet combinatorics/permutations_demo.cpp permutations

\snippet combinatorics/assignment_demo.cpp assignment

---

## Pipelines of stages

\snippet pipeline/stages_demo.cpp demo

---

## Retrying

`backoff_policy` with `backoff_delay` / `next_attempt`: the caller owns the clock, the randomness and the sleep.

\snippet backoff_demo.cpp schedule

---

## Showcase: the whole library at work {#patterns_showcase}

The programs in `showcase/` each weave many of the patterns above into one realistic program. Each prints what it
does, proves what it can at compile time and exits non-zero if any result is not what it documents; they run as the
CTests `showcase.<name>` and with `make showcase`.

| Program | What it shows | Patterns at work |
|---|---|---|
| `trading_day.cpp` | Fills to P&L in exact money, folded and checked to the cent at compile time; orders through the trade lifecycle with T+2 settlement; P&L pivoted by desk and instrument; the book's overnight move attributed by Shapley and Owen against one-at-a-time; greeks in closed form by `finance::black_scholes_greeks` | money, functional folds, fsm, time, pivot, attribution, presentation |
| `payments_ledger.cpp` | Checkouts on the railway (`SNICHOLLS_TRY`), v7 identifiers matched to their compile-time twin, a timeout retried with backoff under the same idempotency key, refunds and payouts through the lifecycle, a signed webhook that fails closed when tampered, a statement reconciled to the cent, a cash ladder that finds the overdraft | capabilities, result, identity, retrying, fsm, dated money |
| `harbour_beacons.cpp` | A compile-time Goertzel bank hears a beacon's tone through seeded noise, a k-d tree picks the beacon, a state machine holds and regains the lock, two workers' statistics merge into the single pass | signals, fixed sizes, fsm, statistics that merge, reproducible randomness |
| `option_risk.cpp` | One put priced four ways: `finance::black_scholes` and a CRR lattice solved by its Snell envelope (`finance::binomial`) at compile time, Monte Carlo with a bootstrap interval, Sobol with shifted copies and a tighter error bar | stochastic control, constexpr maths, randomness, statistics |
| `functional_report.cpp` | A report over plain structs computed by the compiler (reflection, type lists, curried folds, partition, State) and the same lambdas driving a run-time pipeline that must agree exactly | reflection, functional, pipelines, result |

