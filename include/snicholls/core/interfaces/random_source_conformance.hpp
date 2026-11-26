//
//  random_source_conformance.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::check_random_source`: the conformance kit of the `random_source_like` contract.
/// \ingroup core
///
/// **Overview.** Drives any source, in either form (a `random_source_like` value, a `random_source_interface&`, an
/// `any_random_source`), through the contract and returns the failures as text, so any test framework can report them
/// (an empty list is a pass). The checks: every length is filled exactly, the bytes past it untouched (0, 1, odd
/// lengths, 256 and the lengths across it where `getentropy` must chunk, 1000); a zero-length fill with a null pointer
/// succeeds; two fills differ; a fill is not all zero bytes. It does not test randomness quality: that is the
/// operating system's, and statistical tests belong to the implementation's own suite.
///
/// For a source that should fail (an unavailable one), `check_random_source_fails_closed` checks that every length is
/// refused with an error.
///
/// \snippet interfaces/random_source_conformance_demo.cpp kit
///
/// **Relationships.** `concepts/random_source.hpp`, `interfaces/random_source.hpp`,
/// `interfaces/random_source_reference.hpp`. Group: \ref core.
///
/// **Implementations / variants.** One kit for every form.
///
/// **Guarantees.** No exception escapes the kit except `std::bad_alloc` from building its report. **C++17-clean.**
///
/// **Errors.** Failures are data.
///
/// **Performance.** Not benchmarked: a test kit.
///
/// **Tests and demos.** `tests/interfaces/random_source_conformance_test.cpp`,
/// `demos/interfaces/random_source_conformance_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cstddef>`, `<string>`, `<system_error>`, `<vector>`,
/// `concepts/random_source.hpp`.
///
/// **Known issues.** "Two fills differ" can fail by chance for a true source with probability 2^-256; the kit draws
/// 32 bytes each time.
///
/// **Status.** New in 0.27.0.

#ifndef snicholls_core_interfaces_random_source_conformance_hpp
#define snicholls_core_interfaces_random_source_conformance_hpp

#include <snicholls/core/concepts/random_source.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <system_error>
#include <vector>

namespace snicholls {

    /// \brief Runs every check of the `random_source_like` contract on `source`.
    /// \ingroup core
    /// \tparam Source A `random_source_like` type (an interface or adapter included).
    /// \param source The source under test.
    /// \return The failures, one line each; empty when the source conforms.
    template <typename Source>
    [[nodiscard]] std::vector<std::string> check_random_source(Source& source)
    {
        static_assert(is_random_source_like_v<Source>, "snicholls::check_random_source: not a random source");
        std::vector<std::string> failures;
        constexpr std::byte sentinel{0xA5};
        for (const std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{33}, std::size_t{255}, std::size_t{256},
                                    std::size_t{257}, std::size_t{1000}}) {
            std::vector<std::byte> buffer(n + 16, sentinel);
            const std::errc e = source.fill(buffer.data(), n);
            if (random_failed(e)) {
                failures.push_back("a fill of " + std::to_string(n) + " bytes failed: " + std::make_error_code(e).message());
                continue;
            }
            bool past_untouched = true;
            for (std::size_t i = n; i < buffer.size(); ++i)
                past_untouched = past_untouched && buffer[i] == sentinel;
            if (!past_untouched)
                failures.push_back("a fill of " + std::to_string(n) + " bytes wrote past its end");
            bool all_zero = n >= 7;
            for (std::size_t i = 0; i < n; ++i)
                all_zero = all_zero && buffer[i] == std::byte{0};
            if (all_zero)
                failures.push_back("a fill of " + std::to_string(n) + " bytes is all zero");
        }
        if (const std::errc e = source.fill(nullptr, 0); random_failed(e))
            failures.push_back("a zero-length fill failed: " + std::make_error_code(e).message());
        // Padded like the buffers above: a source that writes past its end (one the kit exists to catch) must not
        // corrupt the kit's own stack.
        std::array<std::byte, 32 + 16> a{}, b{};
        const std::errc ea = source.fill(a.data(), 32);
        const std::errc eb = source.fill(b.data(), 32);
        if (random_failed(ea) || random_failed(eb))
            failures.push_back("two 32-byte fills did not both succeed");
        else if (std::equal(a.begin(), a.begin() + 32, b.begin()))
            failures.push_back("two fills gave the same bytes");
        return failures;
    }

    /// \brief Checks that a source which should fail (unavailable, unseeded) refuses every length with an error.
    /// \ingroup core
    /// \tparam Source A `random_source_like` type.
    /// \param source The source under test.
    /// \return The failures; empty when it fails closed.
    template <typename Source>
    [[nodiscard]] std::vector<std::string> check_random_source_fails_closed(Source& source)
    {
        static_assert(is_random_source_like_v<Source>, "snicholls::check_random_source_fails_closed: not a random source");
        std::vector<std::string> failures;
        for (const std::size_t n : {std::size_t{1}, std::size_t{32}, std::size_t{257}}) {
            std::vector<std::byte> buffer(n + 16);   // padded: a broken source must not write outside the kit's buffer
            if (!random_failed(source.fill(buffer.data(), n)))
                failures.push_back("an unavailable source claimed to fill " + std::to_string(n) + " bytes");
        }
        return failures;
    }

} // namespace snicholls

#endif // snicholls_core_interfaces_random_source_conformance_hpp
