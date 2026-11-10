# snicholls::result - Rust's `Result<V, E>`, zero overhead, rethrowable, with traced errors

Status: experimental. Headers: `include/snicholls/core/result.hpp`, `include/snicholls/core/traced.hpp` (standard
library only). The decisions keep the ids they were made under; the gaps are decisions since folded into others.

## Decisions

| Id | Decision |
|---|---|
| D1 | Rust names and semantics: `ok(v)` / `err(e)` tags, `is_ok`, `is_err`, `is_ok_and`, `is_err_and`, `ok()` / `err()` as `std::optional`, `map`, `map_err`, `map_or`, `map_or_else`, `and_then`, `or_else`, `and_` / `or_` (`and` / `or` are keywords), `unwrap`, `expect`, `unwrap_or`, `unwrap_or_else`, `unwrap_or_default`, `unwrap_err`, `expect_err`, `unwrap_unchecked`, `unwrap_err_unchecked`, `inspect`, `inspect_err`, `flatten`, `transpose`, `iter` / `iter_mut` (a `std::span` of 0 or 1). Ref-qualified overloads (`&`, `const&`, `&&`) so values move out. `result<void, E>` is `Result<(), E>`. |
| D2 | No implicit construction from a bare V or E (an earlier `Result` had it; `Result<double, int>(3)` was ambiguous in intent). Always `ok(v)` / `err(e)`, or `std::in_place` / `snicholls::err_in_place`. No default constructor (Rust has none). |
| D3 | Layout: `union { V; E; }` + `bool`, no allocation, no virtual, no RTTI. Each special member is trivial exactly when it is for V and E. Destructor and assignments are P0848 constrained members; copy / move constructors are layered base classes, because Clang treats a class that *declares* any non-trivial copy/move constructor (even one whose constraint fails) as non-trivial for calls and returns it through memory (found by codegen inspection, guarded by a `__is_trivially_relocatable` static_assert). |
| D4 | Niche: opt-in `snicholls::result_niche<V>` (`static constexpr V value`) + an empty, trivial E gives `sizeof(result) == sizeof(V)`. Nothing built in: a raw pointer or integer has no invalid value (`nullptr`, `-1` are valid successes), so `result<T*, empty>` with a null niche would be unsound. Empty E uses `[[no_unique_address]]` (`[[msvc::no_unique_address]]` on MSVC). |
| D5 | Unwrap policy at compile time: `SNICHOLLS_RESULT_THROW=1` (default when exceptions are on) throws `snicholls::bad_result_access<E>`; `=0` (default under `-fno-exceptions`) prints and `std::terminate`s, and unwrap/expect become `noexcept`. It is a whole-program switch. The policies live in different inline namespaces (`result_throws` / `result_terminates`): mixing them is a link error wherever a result appears in a signature; otherwise it is an ODR violation - set the policy for the whole program. On MSVC, `#pragma detect_mismatch("snicholls_result_throw", ...)` makes any mix a link error; GCC and Clang have no equivalent. **Decided:** keep the compile-time switch, default 1 (throw), 0 automatically without exceptions. The per-call choices are in the table below; a per-type policy parameter (`result<V, E, Policy>`) was rejected because it splits the type and spreads into signatures. |
| D6 | `bad_result_access<void>` derives from `std::logic_error` (earlier code threw `std::logic_error`, so old `catch` sites still work). `bad_result_access<E>` carries the error; a non-copyable E throws the base (an exception object must be copyable). |
| D7 | Rethrowable: `rethrow()`, `rethrow_if_err()`, `value_or_throw()` throw the error as its own exception via `snicholls::throw_error`: customisation `snicholls::error_exception<E>::raise`, then `std::exception_ptr` rethrown, a `std::exception` thrown as itself, `std::error_code` as `std::system_error`, `traced<X>` raises X, else `bad_result_access<E>`. `try_invoke(f, args...)` / `result_from_exceptions(f[, map])` turn exceptions into `err(std::exception_ptr)` or a mapped E. |
| D8 | `SNICHOLLS_TRY(expr)` (Rust's `?`) is a GNU statement expression (`__extension__`, so `-Wpedantic` is quiet) on GCC/Clang; `SNICHOLLS_TRY_ASSIGN(decl, expr)` and `SNICHOLLS_TRY_VOID(expr)` are the portable statement forms (MSVC). The error converts through `snicholls::error_conversion<To, From>` (Rust's `From`). |
| D9 | Detailed errors are opt-in: `snicholls::traced<E>` = E + `source_location` + one pointer to lazily allocated details (context frames, fields, stack). `context(fmt, args...)` (anyhow), `with_field`, `with_stack`, `trace()`, `snicholls::fail(fmt, ...)` (anyhow's `bail!`), `snicholls::traced_err(e)`. `SNICHOLLS_TRY` adds a "propagated" frame with the expression text. Equality / hash use E only. Fields mask anything with `reveal()` (`snicholls::secret`). |
| D10 | Std-only and portable for `snicholls-core`: no dependency on a consumer's object model, `std::source_location` behind `__cpp_lib_source_location` (`snicholls::source_location` falls back to an empty location), statement expressions only behind `SNICHOLLS_RESULT_HAS_STATEMENT_EXPRESSIONS`. |
| D11 | Logging: core defines `snicholls::error_sink` (`write(level, error_report)`, `log_error` never throws), `snicholls::error_report`, `format_error_report` (text) and `error_report_fields` (flat, ordered string fields), and `snicholls::text_error_sink`. `r.log_err(sink[, level])` logs and returns the result. A super-log adapter belongs in the consuming library, not in core. |
| D12 | super-log event: `msg` = one-line chain ("outer: inner: error"), `src` = origin `file:line`, `tag` = `result`, `fields` = `where: "result"` (super-log's vocabulary: `uncaught` in the terminate handler, `panic` in the Rust SDK), `error`, `context.N` (innermost first), user fields, `stack`. Keys that look secret are masked too. `superlog::origin` is the process identity, not the source location, so the location goes in `src`. |
| D13 | Wire / remote results: a `result` of plain V and E is itself a plain value (no pointer, trivially copyable when they are), so it can be serialised as (state, V or E). `traced<E>` holds a pointer and is for logging, not for the wire: send its `error()` or its `error_report_fields`. |

## Per-call choices

| Call | On an error |
|---|---|
| `unwrap()`, `expect(msg)` | follow the policy (`SNICHOLLS_RESULT_THROW`) |
| `value_or_throw()` | always throws (the error as its own exception) |
| `unwrap_or_terminate()` | always prints and terminates; `noexcept` |
| `unwrap_unchecked()`, `*r` | no check: `is_ok()` is a precondition |
| `unwrap_or(v)`, `unwrap_or_else(f)`, `unwrap_or_default()` | a fallback |
| `SNICHOLLS_TRY` (and `_ASSIGN`, `_VOID`) | propagate to the caller |

Every `&&` overload that hands out the value or the error (`unwrap`, `expect`, `value_or_throw`, `unwrap_or_terminate`,
`unwrap_unchecked`, `*r`, `unwrap_err`, `expect_err`, `unwrap_err_unchecked`) returns it **by value**, moved out, as
Rust's `unwrap` does: a reference into a temporary result would dangle (`for (int x : make().unwrap())`).

## API at a glance

| Area | API |
|---|---|
| Construct | `snicholls::ok(v)`, `snicholls::ok()`, `snicholls::err(e)`, `result(std::in_place, args...)`, `result(snicholls::err_in_place, args...)`, converting from `result<U, G>` |
| Observe | `is_ok`, `is_err`, `explicit operator bool`, `is_ok_and(f)`, `is_err_and(f)`, `ok()`, `err()`, `iter()`, `iter_mut()`, `*r`, `r->` |
| Extract | `unwrap`, `expect(msg)`, `unwrap_or_terminate`, `unwrap_or(v)`, `unwrap_or_else(f)`, `unwrap_or_default`, `unwrap_err`, `expect_err(msg)`, `unwrap_unchecked`, `unwrap_err_unchecked` |
| Transform | `map`, `map_err`, `map_or`, `map_or_else`, `and_then`, `or_else`, `and_`, `or_`, `inspect`, `inspect_err`, `flatten`, `transpose` |
| Exceptions | `rethrow`, `rethrow_if_err`, `value_or_throw`, `snicholls::throw_error`, `snicholls::try_invoke`, `snicholls::result_from_exceptions`, `snicholls::error_exception<E>` |
| Propagate | `SNICHOLLS_TRY(expr)`, `SNICHOLLS_TRY_ASSIGN(decl, expr)`, `SNICHOLLS_TRY_VOID(expr)`, `snicholls::error_conversion<To, From>` |
| Trace / log | `context`, `with_field`, `with_stack`, `trace`, `log_err`, `snicholls::traced`, `snicholls::traced_err`, `snicholls::fail`, `snicholls::error_sink`, `snicholls::text_error_sink`, `snicholls::make_error_report`, `snicholls::format_error_report`, `snicholls::error_report_fields`, `snicholls::set_stack_trace_hook` |
| Other | `==` (also against `ok_t` / `err_t`), `<=>` (every ok before every err, as Rust's derived `Ord`), `std::hash`, `swap`, `snicholls::error_text`, `to_expected()` / `snicholls::to_result` under `__cpp_lib_expected` |

## Layout guarantees (tests/result_test.cpp)

- `sizeof` / `alignof` equal to `struct { union { V v; E e; }; bool ok; }`; `result<void, E>` equal to
  `struct { E e; bool ok; }`; a niche result equal to `sizeof(V)`.
- Trivially copyable / destructible, and each special member trivial, exactly when V and E are.
- Passed and returned in registers when V and E are trivial (Clang: `__is_trivially_relocatable` static_assert).
  At -O2 on Apple clang 17, `parse()` returning `result<int, E>` and its `SNICHOLLS_TRY` forwarder compile to the same
  register code as the hand-written struct (inspected, not benchmarked).
- `noexcept` on every member is exact (static_asserts), `constexpr` throughout except the error paths of
  unwrap/expect/rethrow and anything touching `traced` (it allocates).

## Performance

Not yet benchmarked in Release (the numbers wait for a quiet machine). `benchmarks/result_benchmark.cpp` compares
`snicholls::result`, a hand-written tagged union, an error-code out-parameter and exceptions at 0/1/10/50/100% error rates,
`traced` on the success path, and one propagation layer (`SNICHOLLS_TRY` vs `if (!r.ok) return r`), after a fairness check.
CTest `benchmark.result_benchmark` runs it in every build.

`traced<E>` allocation, precisely: creating one stores the location and allocates nothing beyond E's own constructor;
the first `context` / `with_field` / `with_stack` allocates one details block, and each frame / field / stack allocates
its strings. A success never constructs a traced, so `result<V, traced<E>>` costs on the success path what a result of
the same size costs (`sizeof(traced<E>)` = E + one pointer for the location (three words on MSVC) + one pointer).

## Interop

- `std::expected` (C++23): `result(const std::expected<V, E>&)`, `to_expected()`, `snicholls::to_result(expected)` under
  `__cpp_lib_expected` (not compiled in today's C++20 builds; untested until the repo moves to C++23).

## Open items

1. Release benchmark numbers on a quiet machine, then fill the Performance section.
2. `std::stacktrace` as the default stack hook under `__cpp_lib_stacktrace` (C++23).
3. super-log: structured (non-string) `fields` or an `error` object in the protocol, so the chain need not be
   flattened to `context.N`.
4. A `std::formatter<snicholls::result<V, E>>`; printing is via `error_text` and acout today.
