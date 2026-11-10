//
//  display_demo.cpp
//  snicholls-core
//
//  Demo for <snicholls/core/display.hpp>: P&L green or red by sign, through a type's attributes and through
//  by_sign for a bare number; a compile-checked number format; a currency theme exported as JSON for imgui and
//  React, tooltips included; a colour as imgui's floats; a neutral band, staleness thresholds, basis points,
//  millions and a currency's minor units. Exits non-zero if any output is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/display.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>

namespace book {

//! [display]
enum class currency : std::uint8_t { usd, eur, jpy };
constexpr auto attributes(currency c)
{
    using namespace snicholls;
    return per_enumerator(c, [] { return std::array{attrs(colours::cyan, label{"USD"}, tooltip{"US dollar, 2 decimals"}, minor_units{2}),
                                        attrs(colours::blue, label{"EUR"}, tooltip{"Euro, 2 decimals"}, minor_units{2}),
                                        attrs(colours::magenta, label{"JPY"}, tooltip{"Japanese yen, no decimals"}, minor_units{0})}; });
}

struct pnl {
    double value;
};
constexpr auto attributes(const pnl&) { return snicholls::attrs(snicholls::sign_colours{}, snicholls::number_format{"{:+.2f}"}); }
//! [display]

//! [display_numbers]
struct desk_pnl {   // grey within +/-0.5: noise is not news
    double value;
};
constexpr auto attributes(const desk_pnl&) { return snicholls::attrs(snicholls::sign_colours{.dead_zone = 0.5}, snicholls::number_format{"{:+.1f}"}); }

struct quote_age {   // green, amber from 5 s, red from 15 s
    double value;
};
constexpr auto attributes(const quote_age&)
{
    using namespace snicholls;
    return attrs(threshold_colours{colours::green, {{5, colours::yellow}, {15, colours::red}}}, number_format{"{:.1f}"}, units{"s"});
}

struct basis_points {
    double value;
};
constexpr auto attributes(const basis_points&) { return snicholls::attrs(snicholls::sign_colours{}, snicholls::number_format{"{:+.2f}"}, snicholls::units{"bps"}); }

struct notional_usd {   // millions, then the currency
    double value;
};
constexpr auto attributes(const notional_usd&) { return snicholls::attrs(snicholls::scale{1e6, "m"}, snicholls::number_format{"{:.2f}"}, snicholls::units{"USD"}); }

constexpr auto decimals(currency c) { return snicholls::attributes_of(c).get<snicholls::minor_units>().digits; }
//! [display_numbers]

} // namespace book

int main()
{
    std::printf("snicholls-core display demo\n");
    using namespace book;
    bool ok = true;

    //! [display_print]
    snicholls::acout{} << "  " << currency::usd << ' ' << pnl{1250.5} << "  " << currency::eur << ' ' << pnl{-310} << '\n';
    const double move = -0.75;   // a bare number: by_sign colours it for one print
    snicholls::acout{} << "  move " << snicholls::by_sign(move) << '\n';
    //! [display_print]

    // The colour each value takes, decided by the compiler where it can be.
    static_assert(snicholls::display_colour(snicholls::attributes_of(pnl{2}), pnl{2}) == snicholls::colours::green);
    static_assert(snicholls::display_colour(snicholls::attributes_of(pnl{-2}), pnl{-2}) == snicholls::colours::red);
    static_assert(snicholls::display_colour(snicholls::attributes_of(pnl{0}), pnl{0}) == snicholls::colours::grey);
    static_assert(snicholls::display_colour(snicholls::attributes_of(currency::jpy), currency::jpy) == snicholls::colours::magenta);
    static_assert(decimals(currency::jpy) == 0 && decimals(currency::usd) == 2);
    const pnl nan{std::numeric_limits<double>::quiet_NaN()};
    ok &= !snicholls::display_colour(snicholls::attributes_of(nan), nan);   // NaN has no sign

    // In a plain stream, no escape codes: the label and the format only.
    std::ostringstream plain;
    snicholls::print_to(plain, std::array{pnl{1.5}, pnl{-2}});
    std::printf("  plain: %s\n", plain.str().c_str());
    ok &= plain.str() == "[+1.50, -2.00]";

    //! [display_json]
    // The theme for imgui and React, written by a program the build runs: one source of colours and tooltips.
    std::ostringstream theme;
    snicholls::write_enum_json<currency>(theme);
    //! [display_json]
    std::printf("  theme: %s\n", theme.str().c_str());
    ok &= theme.str().find(R"("usd": {"colour": "#1b7c83", "label": "USD", "tooltip": "US dollar, 2 decimals", "minor_units": 2})") != std::string::npos;

    //! [display_numbers_print]
    snicholls::acout{} << "  " << desk_pnl{0.3} << ' ' << desk_pnl{-12.4} << "  age " << quote_age{2} << ' ' << quote_age{7} << ' ' << quote_age{20}
                       << "  " << basis_points{-3.25} << "  " << notional_usd{12'500'000} << '\n';
    //! [display_numbers_print]
    std::ostringstream numbers;
    snicholls::print_to(numbers, std::array{notional_usd{2'500'000}});
    snicholls::print_to(numbers, basis_points{1.5});
    ok &= numbers.str() == "[2.50m USD]+1.50 bps";

    constexpr auto floats = snicholls::rgba(snicholls::colours::green);   // ImVec4(floats[0], ...)
    static_assert(floats[3] == 1.0f && floats[0] < floats[1]);

    std::printf(ok ? "display demo: ok\n" : "display demo: FAILED\n");
    return ok ? 0 : 1;
}
