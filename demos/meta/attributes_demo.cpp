//
//  attributes_demo.cpp
//  snicholls-core
//
//  Demo for <snicholls/core/meta/attributes.hpp>: the greeks, each with a colour, a label and a tooltip
//  declared once at compile time; P&L by greek printed through acout in those colours with the P&L green
//  or red; the caller's own attribute (a column width) beside core's. Exits non-zero if an attribute read
//  back is not the one declared.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/meta/attributes.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>

namespace risk {

//! [attributes]
enum class greek : std::uint8_t { delta, gamma, vega, vanna, volga, theta, rho };

struct column_width {   // an attribute of the caller's own: any literal type
    int characters;
};

// One set per greek, in enumerator order; leaving one out is a compile error.
constexpr auto attributes(greek g)
{
    using namespace snicholls;
    return per_enumerator(g, [] { return std::array{
        attrs(colours::blue,    label{"delta"}, tooltip{"dV/dS: P&L per unit move in the underlying"},           column_width{8}),
        attrs(colours::cyan,    label{"gamma"}, tooltip{"d2V/dS2: how delta changes as the underlying moves"},   column_width{8}),
        attrs(colours::magenta, label{"vega"},  tooltip{"dV/dsigma: P&L per volatility point"},                   column_width{8}),
        attrs(colours::orange,  label{"vanna"}, tooltip{"d2V/dS dsigma: delta's change with volatility"},         column_width{8}),
        attrs(colours::yellow,  label{"volga"}, tooltip{"d2V/dsigma2: vega's change with volatility"},            column_width{8}),
        attrs(colours::grey,    label{"theta"}, tooltip{"dV/dt: P&L from one day passing"},                       column_width{8}),
        attrs(rgb(0x953800),     label{"rho"},   tooltip{"dV/dr: P&L per basis point of rates"},                   column_width{8}),
    }; });
}

// Green and red mean up and down, so no greek is green or red.
struct pnl {   // P&L is green when up and red when down, wherever it prints
    double value;
};
constexpr auto attributes(const pnl&) { return snicholls::attrs(snicholls::sign_colours{}, snicholls::number_format{"{:+.0f}"}); }

// Everything is known to the compiler.
static_assert(snicholls::attributes_of(greek::vanna).get<snicholls::tooltip>().text.starts_with("d2V/dS dsigma"));
static_assert(snicholls::attribute_or(greek::rho, snicholls::colours::red) == snicholls::rgb(0x953800));
static_assert(snicholls::attribute_or(3.0, column_width{12}).characters == 12);   // a double has none
//! [attributes]

} // namespace risk

int main()
{
    std::printf("snicholls-core attributes demo\n");
    using namespace risk;
    bool ok = true;

    //! [attributes_print]
    // P&L by greek: each greek in its colour, each P&L green or red, one message.
    const std::array<std::pair<greek, pnl>, 7> by_greek{{{greek::delta, {12'500}}, {greek::gamma, {-3'200}}, {greek::vega, {4'100}},
                                                          {greek::vanna, {-900}}, {greek::volga, {350}}, {greek::theta, {-1'800}},
                                                          {greek::rho, {0}}}};
    snicholls::acout out;
    for (const auto& [g, p] : by_greek)
        out << "  " << g << '\t' << p << '\n';
    //! [attributes_print]
    const std::string text = out.message();
    out.emit();
    ok &= text.find("vanna") != std::string::npos && text.find("-900") != std::string::npos && text.find("+12500") != std::string::npos;

    for (const greek g : snicholls::enum_values_v<greek>)
        ok &= snicholls::attributes_of(g).get<column_width>().characters == 8;
    std::printf("  tooltip for vanna: %.*s\n", static_cast<int>(snicholls::attributes_of(greek::vanna).get<snicholls::tooltip>().text.size()),
                snicholls::attributes_of(greek::vanna).get<snicholls::tooltip>().text.data());
    ok &= snicholls::has_attributes<greek> && !snicholls::has_attributes<double>;

    std::printf(ok ? "attributes demo: ok\n" : "attributes demo: FAILED\n");
    return ok ? 0 : 1;
}
