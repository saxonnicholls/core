//
//  uuid_generator_reference.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::basic_uuid_generator`: the standard-library reference `uuid_generator`, over the caller's random
/// source and clock.
/// \ingroup core
///
/// **Overview.** Makes version 4 and version 7 UUIDs from a random source `Random` (for real use a
/// `random_source_like` source, `concepts/random_source.hpp`, whose real implementation is ts-moveables' operating-system
/// source; for tests a seeded one, or a callable returning `std::uint64_t` such as `stats::wyrand`, or
/// `std::optional<std::uint64_t>` when it can fail) and a clock `Clock`
/// (anything whose `now()` has `unix_millis()`, such as `manual_utc_clock`). Version 7 is monotonic by RFC
/// 9562 section 6.2, method 1: a 12-bit counter in `rand_a`, started from a random value below 2048 at each new
/// millisecond and incremented within one; when it would overflow, or when the clock steps back, the timestamp moves
/// one millisecond past the last one used instead. `reference_uuid_generator` is the deterministic one tests use.
///
/// \snippet interfaces/uuid_generator_reference_demo.cpp reference
///
/// **Relationships.** Models `uuid_generator` (`concepts/uuid_generator.hpp`); passes
/// `interfaces/uuid_generator_conformance.hpp` directly and through `interfaces/uuid_generator.hpp`. `stats/random.hpp`
/// (`wyrand`), `time/utc.hpp`. Group: \ref core.
///
/// **Implementations / variants.** `basic_uuid_generator<Random, Clock>`; `reference_uuid_generator` =
/// `basic_uuid_generator<stats::wyrand, manual_utc_clock>`, made by `make_reference_uuid_generator(seed,
/// start)`.
///
/// **Guarantees.** Not thread-safe (one per thread). No allocation; `noexcept` when the source and the clock are. A
/// failed draw changes nothing.
///
/// **Errors.** `unavailable` when the random source fails; `invalid_argument` when the clock is before 1970 or past
/// v7's 48-bit range.
///
/// **Performance.** Not benchmarked.
///
/// **Tests and demos.** `tests/interfaces/uuid_generator_reference_test.cpp`,
/// `demos/interfaces/uuid_generator_reference_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<chrono>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<system_error>`, `<type_traits>`, `<utility>`,
/// `interfaces/uuid_generator.hpp`, `stats/random.hpp`, `time/utc.hpp`.
///
/// **Known issues.** Predictable by design when the source is `wyrand`: a test double, never an identifier source in
/// production.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_uuid_generator_reference_hpp
#define snicholls_core_interfaces_uuid_generator_reference_hpp

#include <snicholls/core/interfaces/uuid_generator.hpp>
#include <snicholls/core/stats/random.hpp>
#include <snicholls/core/time/utc.hpp>

#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <system_error>
#include <type_traits>
#include <utility>

namespace snicholls {

    namespace detail {
        template <typename R>
        concept uuid_fill_source = requires(R& r, std::byte* data, std::size_t size) {
            { r.fill(data, size) } -> std::same_as<std::errc>;
        };
    } // namespace detail

    /// \brief A random source a UUID generator draws from: a `random_source_like` source
    /// (`concepts/random_source.hpp`: `std::errc fill(std::byte*, std::size_t) noexcept`, the operating system's in
    /// production, `test_random_source` in tests), or a callable returning `std::uint64_t` (such as `stats::wyrand`,
    /// for tests only) or `std::optional<std::uint64_t>` when it can fail.
    /// \ingroup core
    template <typename R>
    concept uuid_random_source = detail::uuid_fill_source<R> || requires(R& r) {
        requires std::same_as<decltype(r()), std::uint64_t> || std::same_as<decltype(r()), std::optional<std::uint64_t>>;
    };

    /// \brief A clock a UUID generator reads: `c.now().unix_millis()`.
    /// \ingroup core
    template <typename C>
    concept uuid_clock = requires(const C& c) {
        { c.now().unix_millis() } -> std::convertible_to<std::int64_t>;
    };

    /// \brief A wall clock in milliseconds that moves only when told: the fixed clock of a deterministic generator.
    /// \ingroup core
    struct manual_utc_clock {
        utc_ms t{};   ///< the instant it reads

        /// \brief The instant.
        /// \return `t`.
        [[nodiscard]] constexpr utc_ms now() const noexcept { return t; }
        /// \brief Moves forward by `d` (a negative `d` is ignored).
        /// \param d The step.
        constexpr void advance(std::chrono::milliseconds d) noexcept
        {
            if (d.count() > 0)
                t += d;
        }
    };

    namespace detail {
        // A random_source_like source: eight bytes, or nothing when it fails (it fails closed).
        template <typename R>
            requires uuid_fill_source<R>
        std::optional<std::uint64_t> uuid_draw(R& r) noexcept
        {
            std::array<std::byte, 8> b{};
            if (r.fill(b.data(), b.size()) != std::errc{})
                return std::nullopt;
            std::uint64_t v = 0;
            for (const std::byte x : b)
                v = (v << 8) | static_cast<std::uint64_t>(x);
            return v;
        }

        template <typename R>
            requires(!uuid_fill_source<R>) && std::same_as<std::invoke_result_t<R&>, std::uint64_t>
        constexpr std::optional<std::uint64_t> uuid_draw(R& r) noexcept(noexcept(r()))
        {
            return r();
        }

        template <typename R>
            requires(!uuid_fill_source<R>) && std::same_as<std::invoke_result_t<R&>, std::optional<std::uint64_t>>
        constexpr std::optional<std::uint64_t> uuid_draw(R& r) noexcept(noexcept(r()))
        {
            return r();
        }

        inline constexpr std::int64_t uuid_v7_max_ms = (std::int64_t{1} << 48) - 1;
    } // namespace detail

    /// \brief The reference UUID generator over a random source and a clock.
    /// \ingroup core
    /// \tparam Random A `uuid_random_source`.
    /// \tparam Clock  A `uuid_clock`.
    template <uuid_random_source Random, uuid_clock Clock>
    class basic_uuid_generator {
    public:
        /// \brief A generator over `random` and `clock`.
        /// \param random The random source.
        /// \param clock  The clock.
        constexpr basic_uuid_generator(Random random, Clock clock) noexcept(
            std::is_nothrow_move_constructible_v<Random> && std::is_nothrow_move_constructible_v<Clock>)
            : random_(std::move(random)), clock_(std::move(clock))
        {
        }

        /// \brief A version 4 UUID from 122 random bits.
        /// \return It, or `unavailable`.
        constexpr capability_result<uuid> v4()
        {
            const auto hi = detail::uuid_draw(random_);
            if (!hi)
                return fail(capability_errc::unavailable, {}, "the random source failed");
            const auto lo = detail::uuid_draw(random_);
            if (!lo)
                return fail(capability_errc::unavailable, {}, "the random source failed");
            std::array<std::uint8_t, 16> bytes{};
            for (std::size_t i = 0; i < 8; ++i) {
                bytes[i] = static_cast<std::uint8_t>(*hi >> (56 - 8 * i));
                bytes[8 + i] = static_cast<std::uint8_t>(*lo >> (56 - 8 * i));
            }
            return ok(make_uuid_v4(bytes));
        }

        /// \brief A version 7 UUID at the clock's millisecond, strictly after every earlier one from this generator.
        /// \return It, `unavailable` or `invalid_argument`.
        constexpr capability_result<uuid> v7()
        {
            const std::int64_t now = static_cast<std::int64_t>(clock_.now().unix_millis());
            if (now < 0 || now > detail::uuid_v7_max_ms)
                return fail(capability_errc::invalid_argument, {}, "the clock is outside version 7's range");
            const auto r = detail::uuid_draw(random_);
            if (!r)
                return fail(capability_errc::unavailable, {}, "the random source failed");
            std::int64_t ms = now;
            std::uint32_t counter = static_cast<std::uint32_t>(*r >> 53);   // 11 bits: room to count within the ms
            if (started_ && ms <= last_ms_) {
                ms = last_ms_;
                counter = last_counter_ + 1;
                if (counter > 0x0FFFu) {
                    if (ms == detail::uuid_v7_max_ms)
                        return fail(capability_errc::invalid_argument, {}, "the clock is outside version 7's range");
                    ++ms;
                    counter = static_cast<std::uint32_t>(*r >> 53);
                }
            }
            started_ = true;
            last_ms_ = ms;
            last_counter_ = counter;
            return ok(make_uuid_v7(static_cast<std::uint64_t>(ms), static_cast<std::uint16_t>(counter), *r));
        }

        /// \brief The random source.
        /// \return It.
        [[nodiscard]] constexpr Random& random() noexcept { return random_; }
        /// \brief The clock, to move a manual one.
        /// \return It.
        [[nodiscard]] constexpr Clock& clock() noexcept { return clock_; }

    private:
        Random random_;
        Clock clock_;
        bool started_ = false;
        std::int64_t last_ms_ = 0;
        std::uint32_t last_counter_ = 0;
    };

    /// \brief The deterministic generator tests use: `stats::wyrand` and a manual clock in milliseconds.
    /// \ingroup core
    using reference_uuid_generator = basic_uuid_generator<stats::wyrand, manual_utc_clock>;

    /// \brief A `reference_uuid_generator` seeded with `seed`, its clock at `start`.
    /// \ingroup core
    /// \param seed  The `wyrand` seed.
    /// \param start The clock's instant.
    /// \return The generator: the same seed and start make the same identifiers.
    [[nodiscard]] constexpr reference_uuid_generator make_reference_uuid_generator(std::uint64_t seed, utc_ms start) noexcept
    {
        return reference_uuid_generator{stats::wyrand{seed}, manual_utc_clock{start}};
    }

    static_assert(uuid_generator<reference_uuid_generator>, "snicholls::reference_uuid_generator must model snicholls::uuid_generator");

} // namespace snicholls

#endif // snicholls_core_interfaces_uuid_generator_reference_hpp
