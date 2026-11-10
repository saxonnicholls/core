//
//  utc_format.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `std::format` support for `snicholls::utc`.
/// \ingroup core
///
/// std::format support for snicholls::utc (opt-in: \<format> is a heavy header).
///
/// \code
///   std::format("{}", t)               // RFC 3339 through snicholls::to_chars: "2024-04-30T14:23:05.123456789Z"
///   std::format("{:%F}", t)            // any chrono spec is forwarded to std::formatter<std::chrono::sys_time<D>>
///   std::format("{:%a %d %b %Y}", t)   // "Tue 30 Apr 2024"
/// \endcode
///
/// The empty spec never touches the locale or allocates beyond the output iterator. A utc outside years
/// 0000..9999 (only possible for utc_us and coarser) throws std::format_error with the empty spec.

#ifndef snicholls_core_time_utc_format_hpp
#define snicholls_core_time_utc_format_hpp

#include <algorithm>
#include <chrono>
#include <format>

#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/utc.hpp>

// \brief `std::format` support for `snicholls::utc` (opt-in: `<format>` is a heavy header).
// \ingroup core
//
// \overview
// See the file description above.
//
// \relations
// - Text through `snicholls::to_chars`.
// - Group: \ref core.
//
// \impls
// None: a single implementation.
//
// \contracts
// - Same text as `snicholls::to_chars`.
//
// \errors
// A bad spec is `std::format_error`.
//
// \examples
// \snippet utc_format_demo.cpp demo
//
// \perf
// Not benchmarked.
//
// \testing
// - Tests: tests/time/utc_format_test.cpp.
// - Demo: demos/time/utc_format_demo.cpp.
//
// \deps
// CMake target `snicholls::core`; standard library only .
//
// \gotchas
// None known.
//
// \status
// Stable, pre-1.0.
//
// \tparam D the duration.
/// \cond
template <snicholls::utc_duration D>
struct std::formatter<snicholls::utc<D>, char>
{
    /// \brief Parses text.
    /// \param ctx the context.
    /// \return the result.
    constexpr auto parse(std::format_parse_context& ctx)
    {
        auto it = ctx.begin();
        rfc3339_ = it == ctx.end() || *it == '}';
        return rfc3339_ ? it : chrono_.parse(ctx);
    }

    /// \brief Formats.
    /// \param t the t.
    /// \param ctx the context.
    /// \return the result.
    template <typename FormatContext>
    auto format(const snicholls::utc<D>& t, FormatContext& ctx) const
    {
        if (!rfc3339_) return chrono_.format(t.sys_time(), ctx);
        const snicholls::rfc3339_text text = snicholls::to_text(t);
        if (!text) throw std::format_error("snicholls::utc: year outside 0000..9999 has no RFC 3339 form");
        return std::copy(text.buffer.data(), text.buffer.data() + text.size, ctx.out());
    }

private:
    std::formatter<std::chrono::sys_time<D>, char> chrono_;
    bool rfc3339_ = true;
};
/// \endcond

#endif /* snicholls_core_time_utc_format_hpp */
