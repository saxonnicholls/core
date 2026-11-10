//
//  hash_combine.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The definition is unchanged; the guard and documentation are core's.
//

/// \file
/// \brief `snicholls::hash_combine`: mix one hash value into a running seed, in 64 bits on every platform.
/// \ingroup core
///
/// \overview
/// The building block for hashing a composite value from the hashes of its parts. A hash of a struct is
/// `hash_combine` applied to each member's hash in declaration order, starting from seed 0.
///
/// It is the Boost.ContainerHash 1.81+ `hash_combine`, so the same chain gives the same answer as Boost's
/// on a 64-bit platform: `seed = mix(seed + 0x9e3779b9 + h)`, where `mix` is a 64-bit xor-shift / multiply
/// finaliser with full avalanche (constant `0xe9846af9b1a615d`).
///
/// \relations
/// - A `std::hash` written as this chain over an object's fields gives the same value as a hand-written hash of
///   the equivalent struct.
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - `constexpr` and `noexcept`; O(1).
/// - A chain from a fixed seed is order-sensitive, because every step mixes the running seed: permuted
///   values hash differently. A single step is not: it adds `seed` and `h` before mixing, so
///   `hash_combine(1, 2) == hash_combine(2, 1)`, as in Boost.
/// - Computed in 64 bits whatever the width of `std::size_t`, then narrowed to it.
///
/// \errors
/// None.
///
/// \examples
/// A hash for a plain struct, usable as an `std::unordered_set` key:
/// \snippet hash/hash_combine_demo.cpp struct
///
/// \perf
/// Benchmarked like for like against the same combine written by hand (`benchmarks/hash/hash_combine_benchmark.cpp`,
/// with a fairness check); not yet measured in Release.
///
/// \testing
/// - Tests: tests/hash/hash_combine_test.cpp.
/// - Demo: demos/hash/hash_combine_demo.cpp.
/// - Benchmark: benchmarks/hash/hash_combine_benchmark.cpp.
///
/// \deps
/// `<cstddef>`, `<cstdint>`.
///
/// \gotchas
/// - **Not portable across builds** when fed `std::hash` values: `std::hash` of the leaves is
///   implementation-defined (libc++ and libstdc++ differ for strings and doubles), so never persist,
///   transmit or compare such hashes across builds. `hash_combine` itself is deterministic.
///
/// \status
/// Stable.

#ifndef snicholls_core_hash_hash_combine_hpp
#define snicholls_core_hash_hash_combine_hpp

#include <cstddef>
#include <cstdint>

namespace snicholls {

/// \brief Mixes hash value `h` into `seed` (order-sensitive), in 64 bits on every platform.
/// \ingroup core
///
/// Mixes hash value h into seed (order-sensitive). The Boost.ContainerHash 1.81+ hash_combine:
/// seed = mix(seed + 0x9e3779b9 + h), where mix is a 64-bit xor-shift / multiply finaliser with full
/// avalanche (constant 0xe9846af9b1a615d). Computed in 64 bits on every platform.
/// \param seed the running hash (start from 0).
/// \param h the next element's hash.
/// \return the new seed.
constexpr std::size_t hash_combine(std::size_t seed, std::size_t h) noexcept
{
    constexpr std::uint64_t m = 0xe9846af9b1a615dULL;
    std::uint64_t x = static_cast<std::uint64_t>(seed) + 0x9e3779b9U + static_cast<std::uint64_t>(h);
    x ^= x >> 32;
    x *= m;
    x ^= x >> 32;
    x *= m;
    x ^= x >> 28;
    return static_cast<std::size_t>(x);
}

} // namespace snicholls

#endif /* snicholls_core_hash_hash_combine_hpp */
