# snicholls-core

Header-only C++20 library of the vocabulary types and the compile-time reflection shared by the snicholls
libraries, among them ts-moveables and base-encode-decode. It sits at the bottom of that stack and depends on nothing
but the standard library. Namespace `snicholls`, include root `<snicholls/core/...>`, CMake target
`snicholls::core`.

## Authorship and commits

This library is the author's own work, gathered here from his earlier libraries, with designs going back to 2010. Commits are authored by the repository owner and carry
**no AI attribution**: no `Co-Authored-By` trailers, no "Generated with" lines, in commit messages, pull request
descriptions, file headers, README, CHANGELOG or this file. Every source file starts with the banner

    //
    //  <name>.hpp
    //  snicholls-core
    //
    //  Copyright 2010-2026 Saxon Herschel Nicholls
    //

## What belongs here (MUST)

A header is admitted only if all of the following hold.

- **Standard library only.** No other snicholls library, no third-party header, no platform header. `third_party/` holds
  GoogleTest and Google Benchmark for the tests and benchmarks; the library never includes them.
- **Header-only, C++20.** Every header compiles on its own under `-std=c++20 -Wall -Wextra -Wpedantic -Werror` and
  under `/W4 /WX /permissive-`. The `snicholls_core_header_check` target compiles each one standalone, twice.
  One exception downward: `constexpr_for.hpp` stays C++17-clean, because ts-moveables (C++17) uses it;
  `snicholls_core_cpp17_check` compiles it as C++17.
- **Compile-time by nature.** Vocabulary types (`overloaded`, `fixed_string`, `function_ref`, `forward_like`,
  `result<V, E>`), compile-time loops (`constexpr_for`, `pp_for_each`), reflection over plain aggregates
  (`member_count`, `tie` / `to_tuple` / `from_tuple`, `type_list`, `string_traits`), small pure functions
  (`hash_combine`). constexpr-friendly wherever the language allows.
- **No I/O, no threads, no networking, no codecs, no logging.** Those live in the libraries that pin
  core: SHA-256 and the base-N codecs in base-encode-decode; synchronisation, the event loop and sockets in
  ts-moveables; object models in the libraries above. One named exception:
  `acout.hpp`, thread-safe printing, which does nothing but hand one finished message to a stream the caller
  names, under one program-wide mutex held for that write alone (a `FILE*` target uses the C library's own
  per-call lock). It is not a logger; levels, fields and sinks stay in ts-moveables. For its colour it may read
  `NO_COLOR` and `TERM` with `std::getenv`, once; it does not call `isatty`, by the platform-header rule above.
- **Time belongs here, including reading the system clock.** `snicholls::utc<D>`, its text, calendar arithmetic and
  `utc<D>::now()` (a read of `std::chrono::system_clock`) are core. "Now" is only ever read explicitly through
  `now()`; nothing constructs it implicitly (default construction is the epoch). Clock *sources* (injectable,
  replay and master clocks) and time zones stay with the libraries above.
- **Run-time numerics by exception.** `game/` (Shapley, Owen, Harsanyi, the samplers) and `finance/attribution.hpp`
  are admitted as run-time numerics. They follow every other rule: standard library only, `constexpr` wherever the
  language allows, the caller's own random generator.
- **Nothing coupled to a consumer's object model.** The libraries above core pin it, so anything that names their
  objects, fields or composition would be a cycle. Core reflects *plain aggregates*.

- **Compile time first; the hot path minimal.** Whatever can be decided before the program runs is decided by
  the compiler: fixed tables and maps are built in `consteval` constructors (`static_map`, `perfect_map`), stopping
  regions and attribution weights in `constexpr` functions. What runs on the hot path is a lookup or a few arithmetic
  operations. Prefer the shortest correct implementation: code is recompiled often, so readable headers and fast
  compiles beat completeness.
- **Ready for the GPU and the FPGA.** Core is simple, fast primitives that support offload; heavy numerics will run on
  accelerators, with ts-moveables as the host-side fabric that batches work to the device. For FPGA synthesis the
  fixed-size forms (`inplace_vector`, `md_array`, `std::array` tables, the fixed-size Snell envelope) matter most:
  sizes known at compile time, bounded loops, no recursion, no allocation. Core never includes a device header, but
  its compute primitives stay device-ready: pure `constexpr` functions with no allocation in the inner loop (callable from CUDA kernels under `--expt-relaxed-constexpr`), spans
  of plain data in, trivially copyable state, partial results that `merge` (a parallel reduction), and randomness by
  stream or by index (`wyrand::stream`, `sobol::point`).

Who pins whom, and this never reverses: core pins nothing; base-encode-decode and ts-moveables pin core; the
libraries above them pin core and those. Nothing in core depends on a library that depends on core.

## The family

Core is the root of a family of libraries that each make sense on their own and together, each with complete
integrity and consistency. The same eleven rules hold in every one of them.

1. **One function, one place.** Each function appears once in the whole family. A copy in a demo, a test, a `utils/`
   folder or a local helper is a defect. Before writing a helper, search the family; if it exists, call it; if not,
   add it to its owner (rule 3) with its test and demo.
2. **The very best version wins.** When a function is found more than once, every copy is compared (correctness,
   edge cases, tests, API, measured speed); the best becomes the owner's, takes in what the others did better, and is
   brought to this file's standard; then every caller switches to it and the copies are deleted, in the same change.
   Which library is lower, or which copy came first, decides nothing.
3. **Place code by what it needs**, the first row that applies:

   | If the code needs | It goes in |
   |---|---|
   | the standard library only, and no I/O | core |
   | hashing or encoding bytes | base-encode-decode |
   | the operating system: threads, synchronisation, files, sockets, clock sources, `getenv`, process lifecycle | ts-moveables |
   | a third-party library | an adapter package (core-adapters for JSON and Python) |
   | an object model, a domain, a product | the libraries and applications above |

   Code that fits two rows is two things: split it.
4. **Interfaces in core, implementations where they belong.** Core holds the contract of any capability, even one
   whose every implementation needs the operating system or a third-party library, in
   `interfaces/<capability>.hpp`, in standard C++ types only (`std::string_view`, `std::span<const std::byte>`,
   `std::optional`, `result` with the interface's own error enumeration): the concept, for static dispatch; a pure
   virtual interface of the same operations, for run-time choice; an adapter that lets any type satisfying the
   concept be used through the virtual interface (which satisfies the concept too). Beside it, in their own headers
   (rule 11): a standard-library reference implementation (the fake tests use) and a conformance kit (plain functions
   that drive any implementation through the contract and say which check failed, so no test framework is
   included). Implementations live where rule 3
   puts them and can come later; callers depend on the contract, never on a concrete library. An interface is
   experimental until two real implementations pass its kit. Already this shape: the linear algebra backend,
   `error_sink`, `ml::attention_cache`, `ml::weight_matrix`. Concepts that classify types (`integer`, `map_like`) stay
   in `meta/`.
5. **Shared vocabulary in core; every type extends it from its own namespace.** Knowledge every library should be able
   to use, and that is standard-library compile-time data, belongs to core even when it describes an outside system:
   type support (C++ types to the names of SQL, protobuf, Rust or Go types), presentation (`attributes()`),
   numeric limits (`math::math_limits`), printing. Core carries the entries for standard types and its own; every
   other type declares its own entries next to its definition, found by ADL or a specialisation. Core never names a
   type from a library above it.
6. **Pins point down; one core version** (see "Versioning and the diamond").
7. **One owner per concept.** Core owns time as values, `result`, printing from threads, the fast hashes, enum and
   aggregate reflection, type support, the interfaces, and the build mode; ts-moveables owns the run-time
   environment, reading files, process-wide initialisation, configuration files, clock sources and log levels;
   core-adapters owns the JSON forms of core's types. A second copy of a concept is deleted, not kept as a variant.
8. **One contract everywhere**: this file's documentation standard, mirrored tests, demos and benchmarks, version
   checks, naming, platforms and commit rules, each enforced by the library's own CI.
9. **Public libraries name nothing private**: no private repository, product, person, credential or local path.
10. **No move without an order.** Code moves between libraries only when the owner of the repositories orders it, in
   dependency order (the owner of the function first, then each caller), each step with its tests green.
11. **The bare minimum of headers at any point.** Every translation unit pulls in only what it needs. Each header
   includes exactly what it uses and nothing more (the standalone-header check proves nothing is missing;
   `clang-include-cleaner` proves nothing is extra); the lightest standard header that serves (`<concepts>` over
   `<algorithm>`, `<iosfwd>` over `<iostream>`, `<span>` over `<vector>` in interfaces); heavy parts behind their
   own header; library code never includes an umbrella header such as `core.hpp`, which is for applications; the
   compile time of the heaviest header is recorded, and a regression is treated like a slower benchmark.

## Layout

- `include/snicholls/core/<name>.hpp`, and `include/snicholls/core/<area>/<name>.hpp` once an area has more than
  a few headers. `core.hpp` is the umbrella. `detail/` holds internals (`namespace snicholls::detail`): not public
  API, still tested, not demonstrated.
- Areas: `interfaces/` for the contracts of capabilities ("The family", rule 4); `meta/` for type traits, concepts and type-level algorithms (`concepts`, `callable_traits`, `type_list`,
  `type_name`, `compare`, `assignable`, `values`, `attributes`); `reflect/` for reflection over aggregates and strings
  (`member_count`, `struct_tuple`, `string_traits`, `string_access`); `stats/` (`random`, `bootstrap`, `low_discrepancy`, `accumulators`, `pivot`, `covariance`; `snicholls::stats`);
  `stochastic/` (`martingale`, `snell`; `snicholls::stochastic`); `math/` (`constexpr_math`; `snicholls::math`); `ml/` (`row`, `activations`, `softmax`, `norm`,
  `minifloat`, `bpe`, `rope`, `attention`, `embedding`, `sampling`, `optim`, `quantize`; `snicholls::ml`); `combinatorics/` (`permutations`,
  `partition`); `hash/` (`fnv`, `wyhash`, `hash_combine`); `containers/`; `game/` (`snicholls::game`); `finance/`
  (`snicholls::finance`). A new trait or concept goes in `meta/`, never in a library above core.
- Keep the tree neat: a header joins an area when one fits, and moves with `scripts/move_header.py <old> <new>`,
  which takes its test, demo and benchmark along and rewrites every include, guard and registration. CMake lists
  and `core.hpp` are sorted: top-level files first, then each area, alphabetically.
- Public names live directly in `namespace snicholls`. The library's own metadata (version) in `snicholls::core`.
- Header guards are named, `snicholls_core_<path>_hpp`, not `#pragma once`: a consumer can test the guard to prove
  an include resolved to core rather than to another root with the same file name. Every header includes what it uses.
- Macros carry the `SNICHOLLS_` prefix; `SNICHOLLS_CORE_` is reserved for the library's own metadata.
- `tests/`, `demos/` and `benchmarks/` mirror `include/snicholls/core/` (below). `scripts/` holds the CI checks.
  `third_party/` holds the test-only submodules.

## Versioning and the diamond (MUST)

Semantic versioning. A release is a tag `vX.Y.Z` on a commit where every CI leg is green. Consumers pin the tag,
never a branch tip. The version is written in three places - `include/snicholls/core/version.hpp`,
`project(... VERSION)` in `CMakeLists.txt`, and the tag - and `scripts/check_version.py` fails CI if they disagree.

A library can reach core through two of its dependencies, so one build can reach core twice, and two copies at
different commits link silently. Three rules, implemented in `CMakeLists.txt` and
`cmake/SnichollsCoreRequire.cmake` and tested by `tests/cmake/diamond_test.cmake` (CTest `cmake.diamond`):

1. The target is defined only `if(NOT TARGET snicholls::core)`. The first copy added wins: the top-level
   project's, whose pin a person chose. A skipped copy at a different version is a CMake warning that names both.
2. Every consumer calls `snicholls_core_require(<minimum>)` after adding core, and configure fails if the winning
   copy is older. The compile-time counterpart, for a consumer's header, is
   `SNICHOLLS_CORE_REQUIRE_VERSION(major, minor, patch);`.
3. Every copy reached records itself in the global property `SNICHOLLS_CORE_VERSIONS_SEEN`. The top-level project
   calls `snicholls_core_assert_single_version()` at the end of its `CMakeLists.txt`, and configure fails if more
   than one version was reached.

Consuming core as a submodule:

    add_subdirectory(third_party/core EXCLUDE_FROM_ALL)
    snicholls_core_require(0.5.0)
    target_link_libraries(my_target PRIVATE snicholls::core)
    # last line of the top-level CMakeLists.txt:
    snicholls_core_assert_single_version()

Installed: `find_package(snicholls_core 0.5 REQUIRED)` provides the same target and the same functions.

**One core version everywhere.** Every project that uses core pins the same tag; there is never a diamond with two
versions, inside one build or across separately built projects:

1. **Pin a tag, as a submodule at `third_party/core`.** Never a branch tip, a copy, or `FetchContent` of a branch.
2. **The top-level project's pin is the version.** A library in the middle keeps its own pin only to build and test
   itself; when it is consumed, core is already added by the top level and its copy is skipped (rule 1). It states
   what it needs with `snicholls_core_require(<minimum>)`, never by forcing its pin.
3. **Every top-level project ends with `snicholls_core_assert_single_version()`**, so reaching a second version is a
   configure error, not a warning.
4. **A core release moves through its consumers in dependency order**, each bump a commit of its own,
   `third_party: bump core vX.Y.Z -> vA.B.C`, with that project's tests green. Raise `snicholls_core_require` only
   when the project uses something new.

## Tests (MUST)

Every public entity has tests; a change without tests is not done. Framework: GoogleTest + GoogleMock
(`third_party/googletest`, `v1.18.0`).

- **Layout mirrors `include/snicholls/core/`**: `<sub>/<name>.hpp` has `tests/<sub>/<name>_test.cpp`, listed in
  `tests/CMakeLists.txt`. `scripts/check_mirror.py` fails CI on a missing or unlisted test, and on a test file
  that has no header.
- Success path, edge cases (empty, nested, boundary values), error paths; `constexpr` and `noexcept` claims as
  `static_assert`; exception-safety guarantees. A compile-time facility is tested at compile time.
- Test names `TEST(<Unit><Aspect>, <BehaviourInPlainWords>)`, e.g. `CoreVersion.StringIsMajorDotMinorDotPatch`.
- CTest labels: `unit`, `demo`, `boundary` (the CMake-level tests), `docs`, `benchmark`.

## Demos (MUST)

Every public header has a demo that exercises every public function and branch. Tests prove behaviour; demos show
how to use it.

- **Layout mirrors `include/snicholls/core/`**: `<sub>/<name>.hpp` has `demos/<sub>/<name>_demo.cpp`, registered
  with `snicholls_core_add_demo(...)` in `demos/CMakeLists.txt`. `scripts/check_mirror.py` enforces it.
- Each demo prints what it does and **exits non-zero if the result is not what it documents**, so demos run as
  CTests (label `demo`).
- Regions marked `//! [tag]` are pulled into the documentation with `\snippet <name>_demo.cpp tag`, so every
  documented example is compiled code. A header's `\snippet` points at *core's* demo, never at a demo in the
  repository the header came from.

## Zero overhead (MUST)

Anything with a runtime footprint - `result<V, E>`, `hash_combine`, the aggregate `to_tuple` path - carries a
layout contract as `static_assert`s (`sizeof`, `alignof`, triviality where claimed) and a like-for-like Google
Benchmark group (`third_party/benchmark`, `v1.9.5`, `-DSNICHOLLS_CORE_BUILD_BENCHMARKS=ON`) against the
hand-written equivalent: identical data, identical arithmetic in the same order, and a fairness check that refuses
to benchmark if any pair disagrees. Only Release numbers mean anything. A short smoke run of every benchmark is a
CTest (label `benchmark`).

## Documentation (MUST) - Doxygen

Every public entity is documented in the source with Doxygen. Warnings are errors for everything from the first
header (`docs/Doxyfile.in`, `WARN_AS_ERROR = FAIL_ON_WARNINGS`). There is no list of unconverted modules to shrink,
because nothing is admitted unconverted. Build: `make docs`, or `-DSNICHOLLS_CORE_BUILD_DOCS=ON` and the `docs`
target; output `build-docs/docs/html/index.html`. CI builds it on every push and publishes it to GitHub Pages
(<https://saxonnicholls.github.io/core/>): `main` to `/main/`, each release tag to `/vX.Y.Z/`, the root to the newest
release. The `gh-pages` branch is generated; never edit it.

Each header, class or major function is documented in this order. It is the snicholls libraries' common standard, so a header
moving between them needs no rewrite:

1. `\brief` - one line.
2. **Overview** - what it is, the problem it solves, when to use it and when not.
3. **Relationships** - what it works with (`\see`), its group (`\ingroup`).
4. **Implementations / variants** - for an interface, every implementation and its trade-offs.
5. **Guarantees** - thread safety, exception safety (`noexcept` / strong / unchanged on failure), complexity,
   ownership and lifetime, invariants (`\pre` / `\post`).
6. **Errors** - what is thrown or returned, and when (`\throws`).
7. **Examples** - `\snippet` from a compiled demo, never hand-copied code that can rot.
8. **Performance** - benchmark numbers with platform, build type and date, or "not benchmarked"; never omitted
   silently.
9. **Tests and demos** - the test file and the demo.
10. **Dependencies** - standard headers only, by rule; say so if a header is heavier than it looks.
11. **Known issues and gotchas** - compiler-specific behaviour (MSVC's preprocessor, GCC's brace-init probing).
12. **Status** - experimental / stable / deprecated.

Members and functions: `\brief`, `\param`, `\tparam`, `\return`, `\throws` (or `noexcept`), complexity where not
obvious. Public declarations use `///`; plain `//` is for implementation notes. Files start with the copyright
banner, then `/// \file`, `\brief` and `\ingroup`. `snicholls::detail` is excluded from the documentation.

## Build and CI

    make test            # configure, build, every CTest
    make test-werror     # as CI does it: warnings are errors
    make asan            # Address + UB sanitizers
    make docs            # Doxygen, warnings are errors
    make check           # scripts/check_version.py and scripts/check_mirror.py
    make ubuntu          # the full suite on Ubuntu 24.04 in Docker: GCC and Clang, Release and Debug, sanitizers
    make demos           # build and run every demo with its output shown (scripts/run_demos.sh --all, or <name>...)
    make benchmarks      # every benchmark in Release; rewrites BENCHMARKS.md, which is never edited by hand

CI records throughput on every leg: `scripts/benchmarks_md.py --output -` writes each platform's tables into its job
summary, and the raw Google Benchmark JSON is kept as an artifact. Those numbers come from shared runners: a record,
never a gate. BENCHMARKS.md and the headers' **Performance** sections quote quiet-machine runs.

**The local gate is macOS plus Ubuntu-in-Docker**, matching CI's two Unix families: run `make test-werror` on the Mac
and `make ubuntu` before pushing. Ubuntu is a supported target and must be testable without waiting for CI, from any
machine with Docker. `scripts/test_ubuntu.sh` builds core's own image (`docker/ubuntu/Dockerfile`: GCC 13, Clang 18
with compiler-rt so Clang's sanitizers link, CMake, Ninja, the pinned Doxygen) and runs six configurations, each a
fresh build with warnings as errors: GCC Release with docs and benchmark smoke runs, GCC Debug, Clang Release, Clang
Debug, and ASan+UBSan under both compilers. `--quick` runs GCC Release only. macOS 13 and older Xcode are not supported targets.

CI (`.github/workflows/ci.yml`) runs on Linux x86-64 and ARM64 with GCC and Clang, macOS ARM64 and x86-64 with
Apple Clang, and Windows with MSVC: the static checks, a Release build with warnings as errors, a Debug build at
`-O0` with asserts live, ASan+UBSan on the Clang legs, and the documentation. **Every leg blocks except Windows.**
The MSVC leg runs with `continue-on-error`: no consumer builds on Windows for now, so an MSVC failure is reported,
not gated on. Keep headers MSVC-friendly where it costs nothing,
and fix an MSVC failure when it is cheap; the rule returns if a consumer ships on Windows. A change is done only when
every other leg is green.

ThreadSanitizer is not in core's CI because core has no threads, by rule. (For libraries with threads: on a macOS 26
host Apple's TSan runtime traps before `main` in every instrumented binary, so TSan evidence comes from Linux.)

## Moving a header in from another library

- **Copy from `git ls-files`, never `cp -r`:** a directory copy carries untracked files. Before a push, run
  `git ls-files | xargs du -ch | tail -1` and look at anything over 100 KB.
- **A moved header changes only what the move requires:** include paths, header guards, `SNICHOLLS_` macro names,
  `\snippet` targets and namespace names in `static_assert` messages. Behaviour, contracts and documented rules move
  word for word. Tests and demos it did not have are written as part of the move.
- **Making a moved header build on GCC (and, while it is cheap, MSVC) is part of its move**, because those legs
  block. A header that cannot be made to build there without changing its behaviour is not admitted.

## Idioms: which tool for which job

- `constexpr_for<Start, End, Inc>(f)` iterates **values** at compile time: an index range, the elements of a
  tuple-like by index. `constexpr_nest` for 2-, 3- and 4-deep index spaces; `unrolled_for` when the body runs at
  run time but the loop is unrolled.
- `type_list<Ts...>` operates on **types**: filter, transform, `index_of`, `contains`, fold, with no values involved.
  Products: `outer_product_t` (every combination), `dot_product_t` (corresponding elements), `thread_t` (Mathematica's
  `Thread`), `permutations_t`. To run an overload set over a type list, `for_each_tag<L>(overloaded{...})`.
- Values: `thread_apply(f, tuples..., scalars...)` for `Thread` over tuples (per-element overload resolution);
  `zip(ranges...)` for run-time ranges walked together; `index_permutations<N, R>()` / `for_each_permutation` for
  orderings, never a hand-rolled `next_permutation` table.
- The line: if the thing you are looping over is a value, `constexpr_for`; if it is a type, `type_list`. Do not
  build a type-level algorithm out of `constexpr_for` and `index_sequence` tricks when `type_list` already has it.
- `overloaded{...}` for a lambda overload set in `std::visit`. `fixed_string` for a string as a non-type template
  parameter. `function_ref` for a non-owning callable parameter. `result<V, E>` where failure is a value rather
  than an exception.
- **Never an `if constexpr` / `else if constexpr` chain over a type** in user-facing code. `first_match{...}` with
  concept-constrained lambdas is that chain: the first callable that accepts the argument runs, in the order
  written, and `meta/concepts.hpp` (`integer`, `number`, `character`, `enumeration`, `streamable`, `map_like`,
  `set_like`, `specialisation_of`) supplies the conditions. `overloaded` is for cases that do not overlap.
- `invoke_with(set, source)` / `invoke_any(set, {args...})` when the arguments exist only at run time; a source
  for another carrier (JSON, a command line) is `size()` plus `get<T>(i)`. `callable_traits<F>` to read a
  callable's signature; never a hand-rolled `function_traits`.
- `constexpr_map` for a table known when the program is written (an enumeration's names), looked up both ways.
- `fnv1a_64` where a tiny, specified-by-the-byte hash must agree across machines or be a `switch` label;
  `wyhash` for speed over longer keys and avalanche, also constexpr and machine-independent; `hash_combine` only
  for `std::hash` values that never leave the process. `containers::bloom_filter` in front of an authoritative
  set when nearly every query is a miss. `containers::byte_map` (owning, `consteval`-buildable) or
  `byte_map_view` (over the caller's spans) for byte strings to 32-bit values built once and read often, such as a
  BPE vocabulary: it is an `ml::bpe_vocabulary` as it stands, and `probe` a batch then `find` it once the table
  is past L2. Never `std::unordered_map<std::string, ...>` on a tokenizer's hot path.
- `uint256` for EVM amounts: never `std::uint64_t` (18 decimals overflow it at 18 tokens) and never `double`.
  Checked arithmetic and parsing return `std::optional`, the one place core prefers it to `result`, so the type
  stays light.
- `backoff_policy` with `backoff_ceiling` / `backoff_delay` / `next_attempt` for any retry loop; the caller owns
  the clock, the randomness and the sleep.
- `acout{} << ...`, `acerr{}`, `print(...)` / `println(...)` for anything printed from more than one thread, or
  from one thread that must not be interleaved by another: one message, one write. Never a bare `std::cout <<`
  chain in threaded code, and never a mutex around printing.
- A type that prints itself has one member and nothing else (`snicholls::printable`): preferably
  `void print(snicholls::acout&) const`, inside which `out.print("{}", ...)` format strings and `out << ...` both
  work and the whole object is written in one call; the classic `void print(std::ostream&) const` still
  qualifies. `acout` finds either, `print(value)` / `println(value)` are the one-liners, and a plain stream gets
  them with `using snicholls::print_operators::operator<<;`. Do not write a per-type `operator<<` beside a `print`.
- Colour in printing is `colours::red` in an `acout` (ends with the message at the latest), `styled(colour, value)`
  for one value, `println(colour, "...")` for a line. Never write escape codes into text by hand: they would reach
  files and string streams. `ansi_start` / `hex` (`colour.hpp`) are for renderers that decide for themselves. A
  program that may be redirected checks `isatty` itself and calls `set_colour_mode(colour_mode::never)`.
- **Presentation is a compile-time attribute of the type, never of the call site.** A type or an enumeration
  declares a `constexpr` function `attributes` in its own namespace returning `attrs(...)` (`meta/attributes.hpp`):
  `colour`, `sign_colours` (P&L green up, red down, grey at zero), `label`, `number_format` (checked by the
  compiler), `tooltip`, or the caller's own types (`display.hpp`). An enumeration uses
  `per_enumerator(e, [] { return std::array{attrs(...), ...}; })`, one set per enumerator or a compile error. acout,
  `print_to`, containers and `stats::pivot` read them; `write_enum_json` exports them for imgui and React from a
  program the build runs, so a colour or a tooltip is written once. Never colour by hand at the call site
  (`p >= 0 ? green : red`, a switch from enumerator to colour, an `ImVec4` table, a CSS `.pos` / `.neg`); declare it
  on the type. `by_sign(x)` is for a bare number with no type of its own. Green and red mean up and down: never give
  an identity (a currency, a greek) either.
- Compile-time maths: `snicholls::math` (`exp`, `log`, `normal_quantile`, ...) in the argument's type: `float` for
  a table a GPU kernel reads, `double`, `long double`. A type of your own gets them by specialising
  `math::math_limits<T>`, as `std::numeric_limits` is; never a `double`-only copy of a function.
- Machine learning: `snicholls::ml` holds the pure, fixed-size primitives (activations with their derivatives,
  softmax and losses, norms, `minifloat` formats, BPE merging), all `constexpr` and alloc-free, taking any `ml::row`.
  Models, training loops, file formats (GGUF, safetensors, tokenizer files), the split pattern's regex, tokenizer
  training and threading belong to the libraries above core. A low-precision value is a `minifloat` (`bfloat16`,
  `float16`, `float8_e4m3`, ...), never a hand-rolled bit trick; stochastic rounding takes the caller's random bits.
  `bpe_encode_piece` gives tiktoken's tokens (`tests/fixtures/ml/bpe_fixture.txt`, from
  `scripts/generate_bpe_fixture.py`, which needs tiktoken and rustbpe and so is not rerun in CI);
  `bpe_encode_long_piece` from `bpe_long_piece_threshold` bytes. Attention for one query is `attend` (one pass, no
  row of scores) or `attention_accumulator` partials merged across workers, never a stored score matrix; RoPE is
  `rope` / `rope_inverse` with a `rope_table` the compiler can build. Sampling takes one uniform draw from the caller
  (`stats::uniform_real`), so a seed reproduces everywhere. Quantised weights are `block_q8_0`, `block_q4_0` and
  `block_mxfp4`, byte for byte ggml's, with `dot` equal to ggml's scalar reference
  (`tests/fixtures/ml/quant_fixture.txt`, from `scripts/generate_quant_fixture.cpp` built against ggml, not rerun in
  CI); a kernel that must agree with them across machines writes each multiplication and addition as its own
  statement, so no compiler fuses an FMA. Matrix work (matmul, Muon's matrix iteration) belongs with the matrices,
  above core.
- Randomness: `stats::wyrand` and `stats::uniform_index`, never `std::uniform_int_distribution` (its output differs
  between standard libraries, so a seed would not reproduce). Resampling: `stats::bootstrap_moments` for anything
  built from means and variances (no resample is gathered), `stats::resample_table` when the same `n` is bootstrapped
  for many series, the stationary bootstrap for returns, `stats::bootstrap` for anything else. Quasi-Monte Carlo:
  `stats::sobol<D>`, shifted copies for an error bar. `detail/sobol_directions.hpp` is generated; never edit it.
- Group-by and aggregate: `stats::pivot<Row, type_list<row keys...>, type_list<column keys...>, measure<value,
  accumulator>...>`, one pass, never a `std::map<std::map<...>>` loop. Live statistics: `stats::variance`, `covariance`,
  `ewma` and friends, merged across workers, never recomputed from history. Uncertainty per cell:
  `stats::bootstrapped<Acc, B>` as the measure.
- Attribution: a game is a callable from `game::coalition` to a value (a backtest, a pricer, a model over a
  dataset). `game::shapley<N>` exactly up to about 20 players; `game::owen` with groups for more; the samplers
  (`shapley_sampled`, `shapley_sampled_lifts`) with a `memoized` game when evaluations are expensive;
  `attribute_change` for "why did the value move"; `finance::explain` for P&L by greek and factor in O(K^2). Never
  one-at-a-time or sequential attribution as the answer: they are there to compare against.
- Containers, pairs, tuples, `std::optional`, `std::variant`, `result` and `type_list` print as they are through
  `acout`, `print(value)` / `println(value)` and `print_to(os, value)`, recursively, with the elements' own
  `print` or `operator<<` inside (`snicholls::printable_any<T>` is "it prints"). Never write a loop to print a
  container, and never a per-container `operator<<`; a plain stream gets the shapes with the same
  `print_operators` opt-in. A string inside a container is quoted, a scoped enumeration prints by name
  (`meta/enums.hpp`; never write a table of enum names by hand), and a `static_assert` names the three ways to make an unprintable type print.

## Struct / tuple conversion rules


- Everything is `constexpr`. `tie` is always `noexcept`; every other conversion is `noexcept` **exactly when**
  the copies/moves/assignments it performs are (a `std::string` copy may throw `bad_alloc`, so copying one is
  not `noexcept`; moving it is). Never declare a conversion unconditionally `noexcept` when it can allocate -
  that turns `bad_alloc` into `std::terminate`.
- Input is strict: sizes must match exactly and arithmetic conversions must not narrow (brace-init rules, so
  `int -> double` and `int -> bool` are rejected). Errors are compile-time `static_assert`s with a `snicholls::`
  message.
- Plain structs: members of any type incl. move-only, const, `std::optional`/`any`/`variant`, nested aggregates,
  `std::array`. Not supported: reference members (clear error), C arrays (miscounted - use `std::array`),
  bit-fields, base classes, unions. Up to `SNICHOLLS_MAX_STRUCT_MEMBERS` (256) members.
- `detail/struct_binders.hpp` is **generated** by `scripts/generate_struct_binders.py`; never edit it by hand. The
  CTest `generated.struct_binders_up_to_date` fails if it drifts from the generator.

## Strings

`reflect/string_traits.hpp` classifies every string / byte-sequence type; `reflect/string_access.hpp` gives
zero-copy access. Accept strings generically with these, never with a hand-picked list of types or `std::string`
copies.

- **Properties are separate** (each `trait<S>` + `trait_v<S>`, any cv/ref): `code_unit_t`, `code_unit_category_v`
  (text: char, wchar_t, char8_t, char16_t, char32_t; bytes: `std::byte`, `unsigned char` = `std::uint8_t`),
  `is_owning_sequence_v`, `has_mutable_code_units_v`, `is_contiguous_sequence_v` (always true),
  `null_termination_v` (none / by_contract / guaranteed), `length_kind_v` (compile_time / constant_time /
  linear_time), `static_length_v` (`std::dynamic_extent` if not in the type), `char_traits_t`. `string_traits<S>`
  bundles them. Concepts: `code_unit_sequence`, `string_like` (text), `byte_sequence`, `owning_string`,
  `string_view_like`, `null_terminated_string`, `fixed_length_string`, `compile_time_string` (basic_fixed_string
  and wrappers - the only template-argument strings).
- **Carriers**: `C*`, `C[N]` (and references), `std::basic_string` (any traits/allocator incl. pmr),
  `std::basic_string_view`, `std::span` (both extents), `std::vector`, `std::array`, `snicholls::basic_fixed_string`.
  Pointers, basic_string, string_view and fixed_string carry text only. Not sequences: a single code unit,
  `signed char`/`std::int8_t` sequences (numbers), byte pointers (no length), volatile, unbounded arrays,
  `std::deque`/`std::list` (not contiguous), `std::nullptr_t`.
- **Rules**: a character array is a string literal: length N-1 including embedded NULs (never strlen), last
  element must be NUL (compile error in constant evaluation, assert at run time) - use `std::array`/`std::span`
  for fixed-width unterminated buffers. A byte array has length N. A reference never owns. Owning containers
  propagate top-level const to their code units; views do not.
- **Access** (all `noexcept`, no allocation, O(1) except `C*` = one strlen): `to_string_view(s)` (keeps the
  string's traits), `length_of(s)`, `c_str_of(s)` (null_terminated_string only), `as_bytes_view(s)` ->
  `std::span<const std::byte>` for bytes and one-byte text (char, char8_t) only - wider text bytes are
  endian-dependent. All constexpr except `as_bytes_view` over non-`std::byte` units. A null `C*` is the empty
  string everywhere (never UB); `nullptr` itself is not a string.
- Never name `std::char_traits` of a byte or `signed char` unit (`std::basic_string<std::byte>`): libc++ deprecated
  and then removed those specialisations.

## Result and errors

`result.hpp` and `traced.hpp` (design and decisions: `RESULT_PLAN.md`).

- **Expected failures return `snicholls::result<V, E>`** (`ok(v)` / `err(e)`, `SNICHOLLS_TRY` as Rust's `?`, or the
  portable `SNICHOLLS_TRY_ASSIGN(decl, expr);` / `SNICHOLLS_TRY_VOID(expr);`, which are the only forms MSVC has). Never
  `bool` plus an out-parameter. Exceptions stay for the unexpected.
- **Zero overhead:** a union of V and E plus a `bool`, no allocation, no virtual, no RTTI. Each special member is
  trivial exactly when it is for V and E, so a result of trivial types is trivially copyable and is passed and
  returned in registers. The layout and triviality are static_asserts in `tests/result_test.cpp`.
- **Unwrapping an error throws `snicholls::bad_result_access<E>`** (a `std::logic_error`) by default.
  `SNICHOLLS_RESULT_THROW=0` (the default without exceptions) prints and calls `std::terminate`, and makes
  unwrap/expect `noexcept`. Set it for the whole program, never per file: the two policies live in different inline
  namespaces, so mixing them is a link error, not an ODR violation.
- **Logged errors are `snicholls::traced<E>`:** E plus a source location and lazily allocated context, fields and
  stack. `snicholls::error_sink` is the logging hook (`log_err(sink)`); a super-log adapter lives in the consuming library. **Never
  put a secret in context text:** format arguments are printed as given. Fields mask anything with `reveal()`.
  `traced` is for logs, not the wire: send its `error()` or its `error_report_fields`.
- Standard library only, like everything in core.

## Time, identity and exact decimals


- `snicholls::utc<D>` (`time/utc.hpp`): one `sys_time<D>`, ns by default, RFC 3339 text via `to_chars`/`from_chars`
  (`time/rfc3339.hpp`; strict parsing, `parse_options` to relax), Unix numbers floor. Never construct "now"
  implicitly: default construction is the epoch, and the clock is read only through `utc<D>::now()`.
- **utc is portable:** the same instant gives the same ticks, text and calendar fields on every platform. The
  committed fixtures in `tests/fixtures/time/` are generated by `scripts/generate_time_fixtures.py` from Python's
  `datetime` and exact integer arithmetic (an independent implementation, never core's own output; CTest
  `generated.time_fixtures_up_to_date`) and are checked by the same tests on macOS, every `make ubuntu`
  configuration and CI. A divergence is a bug: fix it and add the case to the fixtures. What is and is not
  guaranteed identical is listed under "Portability" in `time/utc.hpp`.
- JSON (and other third-party) forms of these types do not live in core, which has no third-party dependencies:
  they belong to the companion package that pins core.
- `snicholls::uuid`: 16 bytes, network order, trivially copyable, constexpr compare/hash, structural (usable as an
  NTTP). Text is canonical lowercase 8-4-4-4-12 via allocation-free `to_chars` / exception-free `parse_uuid` (braces,
  urn and 32-digit forms only on request); `"..."_uuid` literal checked at compile time. Core composes v4 / v7 from
  bits the caller supplies and v5 from a name (constexpr); generating identifiers needs OS randomness and stays
  outside core. Never `std::random_device` / `mt19937` for identifiers.
- `snicholls::decimal`: exact, normalised, one canonical text; not an arithmetic type. `to_double()` is for display.

## Naming convention: Boost / standard library style

| Entity | Style | Example |
|---|---|---|
| Namespaces | lower_snake | `snicholls`, `snicholls::detail` |
| Classes, structs, templates | lower_snake | `snicholls::fixed_string` |
| Functions, variables | lower_snake | `member_count_v`, `to_tuple` |
| Concepts | lower_snake noun or adjective | `plain_aggregate` |
| Type traits | `trait<T>` plus `trait_v<T>` / `trait_t<T>` | `member_count<T>`, `member_count_v<T>` |
| Template parameters | CamelCase | `T`, `Ts...`, `Start` |
| Macros | `SNICHOLLS_` prefix, UPPER_SNAKE | `SNICHOLLS_PP_FOR_EACH` |
| Files | lower_snake `.hpp` | `include/snicholls/core/type_list.hpp` |

`noexcept` must be accurate: conditional `noexcept(...)` when it depends on the parameters. Operator overloads live
in `namespace snicholls` and are found by ADL, never in the global namespace. No specialisations for `std` types we
do not own beyond what ADL-scoped helpers need.

## Dependencies

| Dependency | Path | Pinned to | Used by |
|---|---|---|---|
| GoogleTest / GoogleMock | `third_party/googletest` | `v1.18.0` | tests only |
| Google Benchmark | `third_party/benchmark` | `v1.9.5` | benchmarks only |

The rules: pin to a release tag, never a branch tip; HTTPS URLs; `shallow = true`; never edit a file
inside a submodule; consume via `add_subdirectory(... EXCLUDE_FROM_ALL)` with the dependency's tests and install
turned off; never mark a submodule's includes SYSTEM; one bump per commit, with the message
`third_party: bump <name> <old-tag> -> <new-tag>`. The library itself has no dependencies and never will.
