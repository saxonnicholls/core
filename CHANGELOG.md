# Changelog

All notable changes to snicholls-core. The format follows Keep a Changelog; versions follow Semantic Versioning.
A version is released when its tag is on a commit where every blocking CI leg is green.

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
