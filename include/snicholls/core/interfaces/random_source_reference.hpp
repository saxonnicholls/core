//
//  random_source_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief Test doubles for `random_source_like`: `test_random_source`, deterministic from a seed, and
/// `unavailable_random_source`, which always fails. **Never for secrets.**
/// \ingroup core
///
/// **Overview.** Core does no I/O, so it holds no real source of entropy (that is ts-moveables'). What it holds are
/// the doubles every consumer's tests need:
/// - `test_random_source{seed}` writes a deterministic stream (SplitMix64 in counter mode), so a test that generates a
///   key, a nonce or a UUID is reproducible. It is **not** cryptographic: anyone who knows or guesses the seed knows
///   every byte. Its name says so on purpose.
/// - `unavailable_random_source{code}` always fails with `code` (default `resource_unavailable_try_again`, as an
///   unseeded system source reports), so a consumer can prove its own failure path fails closed.
///
/// \snippet interfaces/random_source_reference_demo.cpp reference
///
/// **Relationships.** `concepts/random_source.hpp`, `interfaces/random_source.hpp`,
/// `interfaces/random_source_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** The two doubles above.
///
/// **Guarantees.** `fill` is `noexcept` and allocation-free; `test_random_source` fills any length, chunk by chunk,
/// and the same seed gives the same stream across platforms and runs (little-endian bytes of each 64-bit word).
/// Trivially copyable. **C++17-clean.**
///
/// **Errors.** `unavailable_random_source` only.
///
/// **Performance.** Not benchmarked: a test double.
///
/// **Tests and demos.** `tests/interfaces/random_source_reference_test.cpp`,
/// `demos/interfaces/random_source_reference_demo.cpp`.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<system_error>`, `concepts/random_source.hpp`.
///
/// **Known issues.** None: it is a test double by design.
///
/// **Status.** New in 0.25.0.

#ifndef snicholls_core_interfaces_random_source_reference_hpp
#define snicholls_core_interfaces_random_source_reference_hpp

#include <snicholls/core/concepts/random_source.hpp>

#include <cstddef>
#include <cstdint>
#include <system_error>

namespace snicholls {

    /// \brief A deterministic source for tests: SplitMix64 in counter mode from a seed. **Not cryptographic.**
    /// \ingroup core
    class test_random_source {
    public:
        /// \brief A source whose stream is fixed by `seed`.
        /// \param seed The seed.
        explicit constexpr test_random_source(std::uint64_t seed) noexcept : seed_(seed) {}

        /// \brief Writes the next `size` bytes of the stream.
        /// \param data Where to write; may be null when `size` is 0.
        /// \param size How many bytes.
        /// \return Always `std::errc{}`. `constexpr`, so a test value can be drawn at compile time.
        constexpr std::errc fill(std::byte* data, std::size_t size) noexcept
        {
            for (std::size_t i = 0; i < size; ++i) {
                if (used_ == 8) {
                    word_ = mix(seed_ + 0x9e3779b97f4a7c15ULL * ++counter_);
                    used_ = 0;
                }
                data[i] = static_cast<std::byte>(word_ >> (8 * used_++));
            }
            return std::errc{};
        }

        /// \brief How many 64-bit blocks have been drawn.
        /// \return The counter.
        [[nodiscard]] constexpr std::uint64_t blocks() const noexcept { return counter_; }

    private:
        static constexpr std::uint64_t mix(std::uint64_t z) noexcept
        {
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
        }

        std::uint64_t seed_;
        std::uint64_t counter_ = 0;
        std::uint64_t word_ = 0;
        unsigned used_ = 8;
    };

    /// \brief A source that always fails, so a consumer can test that its own failure path fails closed.
    /// \ingroup core
    class unavailable_random_source {
    public:
        /// \brief Fails with `resource_unavailable_try_again`, as an unseeded system source does.
        constexpr unavailable_random_source() noexcept = default;
        /// \brief Fails with `code`.
        /// \param code The error every `fill` reports.
        explicit constexpr unavailable_random_source(std::errc code) noexcept : code_(code) {}

        /// \brief Fails, writing nothing.
        /// \return The error.
        constexpr std::errc fill(std::byte*, std::size_t) noexcept
        {
            return code_ != std::errc{} ? code_ : std::errc::resource_unavailable_try_again;
        }

    private:
        std::errc code_{};
    };

    static_assert(is_random_source_like_v<test_random_source>, "test_random_source must model random_source_like");
    static_assert(is_random_source_like_v<unavailable_random_source>, "unavailable_random_source must model it too");

} // namespace snicholls

#endif // snicholls_core_interfaces_random_source_reference_hpp
