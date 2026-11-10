# snicholls-core

[![CI](https://github.com/saxonnicholls/core/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saxonnicholls/core/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-doxygen-blue)](https://saxonnicholls.github.io/core/)

**Documentation:** <https://saxonnicholls.github.io/core/>: the newest release at the root, every version under
`/vX.Y.Z/`, and the latest `main` under `/main/`.

The vocabulary types and the compile-time reflection shared by the snicholls C++ libraries, in one header-only,
standard-library-only, C++20 library at the bottom of the stack.

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

- **Vocabulary types:** `overloaded` and `first_match`, `fixed_string`, `function_ref`, `result<V, E>` and
  `traced<E>`, `strong<T, Tag>`, `uuid`, `decimal`, `uint256`, checked arithmetic, UTC time and RFC 3339.
- **Compile-time loops and types:** `constexpr_for` and `constexpr_nest`, `type_list` (filter, products,
  permutations), `thread_apply`, `zip`, enum reflection, and reflection over plain aggregates.
- **Containers:** `inplace_vector`, `static_map`, `perfect_map`, `md_array`, and allocation-free id maps and pools.
- **Numerics:** `constexpr` maths for any floating-point type, random numbers and the bootstrap, Sobol sequences,
  online accumulators, one-pass pivot tables, martingale tools and the Snell envelope, and Shapley and Owen values.
- **Printing:** `acout`, thread-safe printing of any value, container or tuple, with compile-time display
  attributes (colour, label, format, tooltip) exported to JSON for other front ends.

`CHANGELOG.md` lists every header by release.

## Using it

As a git submodule, pinned to a release tag:

```cmake
add_subdirectory(third_party/core EXCLUDE_FROM_ALL)
snicholls_core_require(0.5.0)
target_link_libraries(my_target PRIVATE snicholls::core)

# last line of the top-level CMakeLists.txt
snicholls_core_assert_single_version()
```

Installed, with `cmake --install`:

```cmake
find_package(snicholls_core 0.5 REQUIRED)
target_link_libraries(my_target PRIVATE snicholls::core)
```

In code:

```cpp
#include <snicholls/core/version.hpp>

SNICHOLLS_CORE_REQUIRE_VERSION(0, 5, 0);   // fails to compile against an older core
static_assert(snicholls::core::version() >= SNICHOLLS_CORE_VERSION_CHECK(0, 5, 0));
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
```

CI runs the same on Linux (x86-64 and ARM64, GCC and Clang), macOS (ARM64 and x86-64) and Windows (MSVC), plus a
Debug build with asserts live and the documentation build. Every leg blocks except Windows, which is reported but
not gated on while no consumer builds there.

The standards the code is held to - what may be admitted, the mirror rule for tests and demos, the documentation
standard - are in `CLAUDE.md`.

## Licence

MIT. Copyright 2010-2026 Saxon Nicholls.
