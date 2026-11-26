//
//  random_source.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::random_source_like`: a source of cryptographically secure random bytes, which fails closed.
/// \ingroup core
///
/// **Overview.** The contract for random bytes that must be unpredictable: keys, nonces, salts, session tokens,
/// UUID v4 and v7's random bits. A source answers `fill(data, size)`, writing exactly `size` bytes, and returns
/// `std::errc{}` (zero: success); or it returns an error (`random_failed(e)`) and promises nothing about `data`. The
/// result is `std::errc`, a literal type, so a source can be `constexpr`: `test_random_source` is, and keys, nonces
/// and UUIDs drawn from it can be built and checked at compile time. It **fails closed**: a source that
/// is unavailable or not yet seeded reports an error, and never falls back to a non-cryptographic generator. Core does
/// no I/O, so it defines the contract and a test double only; the real source is the operating system's generator,
/// implemented in ts-moveables (`getrandom`, `getentropy`, `arc4random_buf`, `BCryptGenRandom`). crypto, which pins
/// only core, and core's own uuid generator take randomness through this contract, so neither depends on ts-moveables.
///
/// This is the concept half of a pair; the virtual twin is `snicholls::random_source_interface`
/// (`interfaces/random_source.hpp`). Use `stats::wyrand` for simulation and Monte Carlo, never this; use this for
/// secrets, never `stats::wyrand` or `std::mt19937`.
///
/// \snippet concepts/random_source_demo.cpp concept
///
/// **Relationships.** `interfaces/random_source.hpp` (the twin and the adapters), `interfaces/random_source_reference.hpp`
/// (`test_random_source`), `interfaces/random_source_conformance.hpp` (the kit). Group: \ref core.
///
/// **Implementations.** ts-moveables' operating-system source; `test_random_source` (deterministic, tests only);
/// anything with the member.
///
/// **Guarantees.** A trait and a concept: no code, no cost. **C++17-clean** (ts-moveables is C++17): the trait
/// `is_random_source_like_v` works in C++17, the concept `random_source_like` where the compiler has concepts.
///
/// **Errors.** None here.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/random_source_test.cpp`, `demos/concepts/random_source_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<system_error>`, `<type_traits>`, `<utility>`.
///
/// **Known issues.** `fill` takes a pointer and a size rather than `std::span`, which is C++20. A platform's own
/// error (a Windows status, an `errno`) is mapped by the implementation to the nearest `std::errc`.
///
/// **Status.** New in 0.25.0.

#ifndef snicholls_core_concepts_random_source_hpp
#define snicholls_core_concepts_random_source_hpp

#include <cstddef>
#include <system_error>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename T, typename = void>
        struct is_random_source_like : std::false_type {};

        template <typename T>
        struct is_random_source_like<
            T, std::void_t<decltype(std::declval<T&>().fill(std::declval<std::byte*>(), std::declval<std::size_t>()))>>
            : std::bool_constant<
                  std::is_same_v<decltype(std::declval<T&>().fill(std::declval<std::byte*>(), std::declval<std::size_t>())),
                                 std::errc> &&
                  noexcept(std::declval<T&>().fill(std::declval<std::byte*>(), std::declval<std::size_t>()))> {};
    } // namespace detail

    /// \brief Whether `T` is a random source: `std::errc fill(std::byte* data, std::size_t size) noexcept`, on a
    /// mutable object. C++17.
    /// \tparam T The type.
    template <typename T>
    inline constexpr bool is_random_source_like_v = detail::is_random_source_like<T>::value;

    /// \brief Whether a `fill` failed: anything but `std::errc{}`.
    /// \param e What `fill` returned.
    /// \return `true` on failure.
    [[nodiscard]] constexpr bool random_failed(std::errc e) noexcept { return e != std::errc{}; }

#if defined(__cpp_concepts) && __cpp_concepts >= 201907L
    /// \brief A source of cryptographically secure random bytes (`is_random_source_like_v`), as a C++20 concept.
    /// \tparam T The type.
    template <typename T>
    concept random_source_like = is_random_source_like_v<T>;
#endif

} // namespace snicholls

#endif // snicholls_core_concepts_random_source_hpp
