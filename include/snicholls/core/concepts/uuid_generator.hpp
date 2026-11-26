//
//  uuid_generator.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The contract of crtp's UUID generators (interfaces/uuid/uuid_generator.hpp, make_uuid / _v7 / _v4, behind
//  uuid_interface.hpp), with the randomness and the clock as the caller's sources: core composes the identifiers
//  (uuid.hpp), the operating system's randomness stays above core.
//

/// \file
/// \brief `snicholls::uuid_generator`: anything that makes version 4 and version 7 UUIDs.
/// \ingroup core
///
/// **Overview.** Core composes UUIDs from bits it is given (`make_uuid_v4`, `make_uuid_v7` in `uuid.hpp`); a
/// generator is where those bits come from: a random source and a clock. Production generators draw from the
/// operating system through a `random_source_like` source (`concepts/random_source.hpp`; `getrandom`, `arc4random_buf`,
/// `BCryptGenRandom` in ts-moveables) and read the system clock; tests and replays use `basic_uuid_generator` (`interfaces/uuid_generator_reference.hpp`) over
/// `stats::wyrand` and `manual_utc_clock`, so every run makes the same identifiers. Code that makes identifiers takes a
/// `uuid_generator`, never `std::random_device` or `std::mt19937`.
///
/// **The contract** (`interfaces/uuid_generator_conformance.hpp` checks it):
/// - `v4()` is version 4 with the RFC 9562 variant, from 122 random bits;
/// - `v7()` is version 7 with the RFC 9562 variant, its 48-bit timestamp the clock's Unix millisecond (or, to stay
///   monotonic, a later one: RFC 9562 section 6.2), and **strictly increasing** within one generator, even when many
///   are made in one millisecond and when the clock steps back;
/// - a random source that fails is `capability_errc::unavailable`, and nothing is returned; a clock outside v7's
///   range (before 1970, after the year 10889) is `capability_errc::invalid_argument`.
///
/// \snippet concepts/uuid_generator_demo.cpp concept
///
/// **Relationships.** `uuid.hpp` (the value and the composers), `capability_error.hpp`; the twin
/// `interfaces/uuid_generator.hpp`, the reference `interfaces/uuid_generator_reference.hpp`, the kit
/// `interfaces/uuid_generator_conformance.hpp`; the clock is anything with `now().unix_millis()`. Group: \ref core.
///
/// **Implementations.** `basic_uuid_generator<Random, Clock>` (standard library only, deterministic for a seeded
/// source). Above core: the operating system's generator in crtp (`make_uuid`) and ts-moveables.
///
/// **Guarantees.** A concept: no code. Thread safety is the implementation's (one generator per thread, or a lock).
///
/// **Errors.** `capability_error`, as the contract says.
///
/// **Performance.** Not benchmarked here: the reference costs two or three `wyrand` draws per identifier.
///
/// **Tests and demos.** `tests/concepts/uuid_generator_test.cpp`, `demos/concepts/uuid_generator_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `capability_error.hpp`, `uuid.hpp`.
///
/// **Known issues.** A deterministic generator makes predictable identifiers: never let one reach production, where
/// an identifier may be guessed.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_uuid_generator_hpp
#define snicholls_core_concepts_uuid_generator_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/uuid.hpp>

#include <concepts>

namespace snicholls {

    /// \brief A UUID generator: `g.v4()` and `g.v7()`, each a `capability_result<uuid>`.
    /// \ingroup core
    template <typename G>
    concept uuid_generator = requires(G& g) {
        { g.v4() } -> std::same_as<capability_result<uuid>>;
        { g.v7() } -> std::same_as<capability_result<uuid>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_uuid_generator_hpp
