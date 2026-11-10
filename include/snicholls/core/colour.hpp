//
//  colour.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::colour`: one display colour, as RGB for user interfaces and an ANSI code for terminals.
/// \ingroup core
///
/// **Overview.** One value carries both forms a display needs: an RGB colour for user
/// interfaces (CSS hex, ImGui) and an ANSI terminal colour. To print in colour, use it with
/// `acout.hpp`, which decides whether the target should see the escape codes at all; the
/// functions here only render text and never ask.
///
/// \code
///   snicholls::hex(snicholls::colours::green).view()        -> "#2da44e"
///   snicholls::ansi_start(snicholls::colours::green).view() -> "\x1b[32m"            (the 16-colour code when one is set)
///   snicholls::ansi_start(snicholls::rgb(0x123456)).view()  -> "\x1b[38;2;18;52;86m" (24-bit otherwise)
/// \endcode
///
/// \snippet colour_demo.cpp palette
///
/// **Relationships.** `acout.hpp` prints with it (`styled`, `colours::red` in a message); `display.hpp`
/// attaches it to types as an attribute. Group: \ref core.
///
/// **Guarantees.** Everything is `constexpr`, `noexcept` and allocation-free, and a colour is a
/// trivially copyable pair. `colour_text` is bounded: appending past its capacity stops.
///
/// **Errors.** None.
///
/// **Performance.** Not benchmarked here; `acout.hpp` benchmarks coloured messages.
///
/// **Tests and demos.** `tests/colour_test.cpp`, `demos/colour_demo.cpp`.
///
/// **Dependencies.** `<array>`, `<cstddef>`, `<cstdint>`, `<string_view>`.
///
/// **Known issues.** Foreground colours only: no background, bold or underline. A terminal
/// without 24-bit support shows `rgb(...)` colours as its nearest guess, or not at all; the
/// named palette uses the 16-colour codes for that reason.
///
/// **Status.** Stable.

#ifndef snicholls_core_colour_hpp
#define snicholls_core_colour_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace snicholls {

    /// \brief The 16 standard ANSI foreground colours (SGR codes); none = use 24-bit RGB.
    /// \ingroup core
    enum class ansi_colour : std::uint8_t {
        none = 0,   ///< none
        black = 30,   ///< black
        red = 31,   ///< red
        green = 32,   ///< green
        yellow = 33,   ///< yellow
        blue = 34,   ///< blue
        magenta = 35,   ///< magenta
        cyan = 36,   ///< cyan
        white = 37,   ///< white
        bright_black = 90,   ///< bright black
        bright_red = 91,   ///< bright red
        bright_green = 92,   ///< bright green
        bright_yellow = 93,   ///< bright yellow
        bright_blue = 94,   ///< bright blue
        bright_magenta = 95,   ///< bright magenta
        bright_cyan = 96,   ///< bright cyan
        bright_white = 97,   ///< bright white
    };

    /// \brief A display colour: RGB for UIs and an optional 16-colour ANSI code for terminals.
    /// \ingroup core
    struct colour
    {
        std::uint32_t rgb = 0;   ///< 0xRRGGBB
        ansi_colour ansi = ansi_colour::none;   ///< terminal colour; none = 24-bit from rgb

        /// \brief Equality.
        /// \return the result.
        friend constexpr bool operator==(const colour&, const colour&) = default;
    };

    /// \brief A 24-bit colour with no 16-colour terminal equivalent.
    ///
    /// Bits above 0xFFFFFF are ignored.
    /// \ingroup core
    /// \param value the value.
    /// \return a 24-bit colour with no 16-colour terminal equivalent.
    constexpr colour rgb(std::uint32_t value) noexcept { return colour{value & 0xFFFFFFu, ansi_colour::none}; }

    // A palette whose RGB values are readable on light and dark backgrounds (GitHub Primer 5), each paired with the ANSI
    // colour a terminal shows instead.
    namespace colours {
    /// \brief The red.
    inline constexpr colour red{0xcf222e, ansi_colour::red};
    /// \brief The green.
    inline constexpr colour green{0x2da44e, ansi_colour::green};
    /// \brief The yellow.
    inline constexpr colour yellow{0xbf8700, ansi_colour::yellow};
    /// \brief The orange.
    inline constexpr colour orange{0xbc4c00, ansi_colour::bright_red};
    /// \brief The blue.
    inline constexpr colour blue{0x0969da, ansi_colour::blue};
    /// \brief The magenta.
    inline constexpr colour magenta{0x8250df, ansi_colour::magenta};
    /// \brief The cyan.
    inline constexpr colour cyan{0x1b7c83, ansi_colour::cyan};
    /// \brief The grey.
    inline constexpr colour grey{0x6e7781, ansi_colour::bright_black};
    } // namespace colours

    /// \brief A short text kept by value (no allocation): what hex() and ansi_start() return.
    /// \ingroup core
    template <std::size_t Capacity>
    struct colour_text
    {
        /// \brief The data.
        std::array<char, Capacity> data{};
        /// \brief The number of elements.
        std::size_t size = 0;

        /// \brief The view.
        /// \return the view.
        constexpr std::string_view view() const noexcept { return std::string_view(data.data(), size); }
        /// \brief The text as a view.
        /// \return the view.
        constexpr operator std::string_view() const noexcept { return view(); }

        /// \brief Appends rows.
        /// \param c the c.
        constexpr void append(char c) noexcept
        {
            if (size < Capacity) data[size++] = c;
        }
        /// \brief Appends rows.
        /// \param s the string.
        constexpr void append(std::string_view s) noexcept
        {
            for (const char c : s) append(c);
        }
        /// \brief The append decimal.
        /// \param value the value.
        constexpr void append_decimal(unsigned value) noexcept
        {
            char digits[10]{};
            std::size_t n = 0;
            do {
                digits[n++] = static_cast<char>('0' + value % 10);
                value /= 10;
            } while (value != 0);
            while (n != 0) append(digits[--n]);
        }
    };

    /// \brief CSS form, lowercase: "\#rrggbb".
    /// \ingroup core
    /// \param c the c.
    /// \return the result.
    constexpr colour_text<7> hex(colour c) noexcept
    {
        constexpr std::string_view digits = "0123456789abcdef";
        colour_text<7> out;
        out.append('#');
        for (int shift = 20; shift >= 0; shift -= 4) out.append(digits[(c.rgb >> shift) & 0xFu]);
        return out;
    }

    /// \brief The SGR sequence that starts this colour: the 16-colour code if set, else 24-bit "38;2;r;g;b".
    /// \ingroup core
    /// \param c the c.
    /// \return the SGR sequence that starts this colour: the 16-colour code if set, else 24-bit "38;2;r;g;b".
    constexpr colour_text<20> ansi_start(colour c) noexcept
    {
        colour_text<20> out;
        out.append("\x1b[");
        if (c.ansi != ansi_colour::none) {
            out.append_decimal(static_cast<unsigned>(c.ansi));
        } else {
            out.append("38;2;");
            out.append_decimal((c.rgb >> 16) & 0xFFu);
            out.append(';');
            out.append_decimal((c.rgb >> 8) & 0xFFu);
            out.append(';');
            out.append_decimal(c.rgb & 0xFFu);
        }
        out.append('m');
        return out;
    }

    /// \brief Resets every SGR attribute.
    /// \ingroup core
    inline constexpr std::string_view ansi_reset = "\x1b[0m";

} // namespace snicholls

#endif /* snicholls_core_colour_hpp */
