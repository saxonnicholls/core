//
//  display_test.cpp
//  snicholls-core
//
//  Tests for <snicholls/core/display.hpp>, with the printing it drives in <snicholls/core/acout.hpp>: colour
//  by sign (zero and NaN included), a colour attribute, labels, compile-checked number formats, tooltips left
//  out of the terminal, escape codes only where colour is wanted, the JSON export and imgui's floats; dead
//  zones, threshold colours, units, scale and minor units.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/display.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace display_test {

enum class ccy : std::uint8_t { usd, eur };
constexpr auto attributes(ccy c)
{
    using namespace snicholls;
    return per_enumerator(c, [] { return std::array{attrs(colours::green, label{"USD"}, tooltip{"US \"dollar\""}),
                                        attrs(colours::blue, label{"EUR"}, tooltip{"Euro"})}; });
}

struct pnl {
    double value;
};
constexpr auto attributes(const pnl&) { return snicholls::attrs(snicholls::sign_colours{}, snicholls::number_format{"{:+.1f}"}); }

struct count {   // an integer formatted as a double
    int value;
};
constexpr auto attributes(const count&) { return snicholls::attrs(snicholls::number_format{"{:>5.0f}"}); }

struct both {   // colour and sign colours: the sign wins for a number
    double value;
};
constexpr auto attributes(const both&) { return snicholls::attrs(snicholls::colours::cyan, snicholls::sign_colours{}); }

struct font {   // a caller's attribute, exported by specialising attribute_json
    bool bold;
};
struct titled {
    int x;
};
constexpr auto attributes(const titled&) { return snicholls::attrs(snicholls::label{"T"}, font{true}); }

struct pnl_banded {   // grey within +/-0.5
    double value;
};
constexpr auto attributes(const pnl_banded&)
{
    return snicholls::attrs(snicholls::sign_colours{.dead_zone = 0.5}, snicholls::number_format{"{:+.1f}"});
}

struct age_seconds {   // green, amber from 5 s, red from 15 s; and a sign colour the thresholds beat
    double value;
};
constexpr auto attributes(const age_seconds&)
{
    using namespace snicholls;
    return attrs(threshold_colours{colours::green, {{5, colours::yellow}, {15, colours::red}}}, sign_colours{}, units{"s"},
                 number_format{"{:.1f}"});
}

struct spread_bps {
    double value;
};
constexpr auto attributes(const spread_bps&) { return snicholls::attrs(snicholls::units{"bps"}, snicholls::sign_colours{}); }

struct notional {   // millions with an "m", then the currency
    double value;
};
constexpr auto attributes(const notional&)
{
    return snicholls::attrs(snicholls::scale{1e6, "m"}, snicholls::number_format{"{:+.2f}"}, snicholls::units{"USD"});
}

enum class iso : std::uint8_t { usd, jpy, kwd };
constexpr auto attributes(iso c)
{
    using namespace snicholls;
    return per_enumerator(c, [] { return std::array{attrs(label{"USD"}, minor_units{2}), attrs(label{"JPY"}, minor_units{0}),
                                                    attrs(label{"KWD"}, minor_units{3})}; });
}

} // namespace display_test

template <>
struct snicholls::attribute_json<display_test::font> {
    static constexpr std::string_view name = "font";
    static void write(std::ostream& os, const display_test::font& f) { os << (f.bold ? "\"bold\"" : "\"normal\""); }
};

namespace {

using namespace display_test;
using snicholls::colours::green;
using snicholls::colours::red;

template <typename T>
std::string plain_text(const T& v)
{
    std::ostringstream os;
    snicholls::print_to(os, v);
    return os.str();
}

struct always_colour {
    always_colour() { snicholls::set_colour_mode(snicholls::colour_mode::always); }
    ~always_colour() { snicholls::set_colour_mode(snicholls::colour_mode::automatic); }
};

// Colour by sign, at compile time.
static_assert(snicholls::sign_colours{}.of(1.0) == green && snicholls::sign_colours{}.of(-1e-300) == red);
static_assert(snicholls::sign_colours{}.of(0.0) == snicholls::colours::grey && snicholls::sign_colours{}.of(-0.0) == snicholls::colours::grey);
static_assert(!snicholls::sign_colours{}.of(std::numeric_limits<double>::quiet_NaN()));
static_assert(snicholls::display_colour(snicholls::attributes_of(both{-1}), both{-1}) == red);
static_assert(snicholls::display_colour(snicholls::attributes_of(ccy::eur), ccy::eur) == snicholls::colours::blue);
static_assert(!snicholls::display_colour(snicholls::attributes_of(count{1}), count{1}));
static_assert(snicholls::has_display_attributes<pnl> && snicholls::has_display_attributes<titled> && !snicholls::has_display_attributes<int>);
static_assert(snicholls::number_like<pnl> && snicholls::number_like<int> && !snicholls::number_like<titled>);
static_assert(snicholls::numeric_of(pnl{2.5}) == 2.5 && snicholls::numeric_of(3) == 3);

// Formats the compiler accepts (anything else is a compile error).
static_assert(snicholls::number_format{"{}"}.spec == "{}");
static_assert(snicholls::number_format{"{:*^+#012.4Lf}"}.spec.size() == 14);
static_assert(snicholls::number_format{"{:.3e}"}.spec == "{:.3e}");

constexpr auto floats = snicholls::rgba(snicholls::rgb(0xff8000));
static_assert(floats[0] == 1.0f && floats[1] == 128.0f / 255.0f && floats[2] == 0.0f && floats[3] == 1.0f);

TEST(DisplayPlain, LabelsAndFormatsWithoutEscapeCodes)
{
    EXPECT_EQ(plain_text(ccy::usd), "USD");
    EXPECT_EQ(plain_text(pnl{1.25}), "+1.2");
    EXPECT_EQ(plain_text(pnl{-3}), "-3.0");
    EXPECT_EQ(plain_text(count{42}), "   42");
    EXPECT_EQ(plain_text(titled{1}), "T");
}

TEST(DisplayPlain, InsideContainersEveryElementTakesItsOwnAttributes)
{
    EXPECT_EQ(plain_text(std::vector{pnl{1}, pnl{-1}}), "[+1.0, -1.0]");
    EXPECT_EQ(plain_text(std::map<ccy, pnl>{{ccy::usd, {2}}, {ccy::eur, {-2}}}), "{USD: +2.0, EUR: -2.0}");
}

TEST(DisplayColour, PositiveGreenNegativeRedZeroGrey)
{
    const always_colour on;
    EXPECT_EQ(plain_text(pnl{5}), "\x1b[32m+5.0\x1b[0m");
    EXPECT_EQ(plain_text(pnl{-5}), "\x1b[31m-5.0\x1b[0m");
    EXPECT_EQ(plain_text(pnl{0}), "\x1b[90m+0.0\x1b[0m");
    EXPECT_EQ(plain_text(pnl{std::numeric_limits<double>::quiet_NaN()}), "+nan");   // no sign, no colour
}

TEST(DisplayColour, AnEnumeratorInItsColourAndAMessageReturnsToItsOwn)
{
    const always_colour on;
    snicholls::acout out(std::cout);
    out << snicholls::colours::yellow << "a " << ccy::eur << " b";
    EXPECT_EQ(out.message(), "\x1b[33ma \x1b[34mEUR\x1b[33m b");
    out << snicholls::colour_reset;
    std::ostringstream discard;
    out.copy_into(discard);
}

TEST(DisplayColour, NeverModeWritesNoEscapeCodes)
{
    snicholls::set_colour_mode(snicholls::colour_mode::never);
    EXPECT_EQ(plain_text(pnl{5}), "+5.0");
    snicholls::acout out(std::cout);
    out << snicholls::by_sign(-2.0);
    EXPECT_EQ(out.message(), "-2");
    std::ostringstream discard;
    out.copy_into(discard);
    snicholls::set_colour_mode(snicholls::colour_mode::automatic);
}

TEST(DisplayBySign, ABareNumberInItsSignsColour)
{
    const always_colour on;
    snicholls::acout out(std::cout);
    out << snicholls::by_sign(-2.5) << ' ' << snicholls::by_sign(1) << ' ' << snicholls::by_sign(std::numeric_limits<double>::quiet_NaN());
    EXPECT_EQ(out.message(), "\x1b[31m-2.5\x1b[0m \x1b[32m1\x1b[0m \x1b[90mnan\x1b[0m");
    std::ostringstream discard;
    out.copy_into(discard);
}

TEST(DisplayJson, AnEnumerationAsATheme)
{
    std::ostringstream os;
    snicholls::write_enum_json<ccy>(os);
    EXPECT_EQ(os.str(), R"({"usd": {"colour": "#2da44e", "label": "USD", "tooltip": "US \"dollar\""}, )"
                        R"("eur": {"colour": "#0969da", "label": "EUR", "tooltip": "Euro"}})");
}

TEST(DisplayJson, SignColoursFormatsAndTheCallersOwnAttribute)
{
    std::ostringstream a, b, c;
    snicholls::write_json(a, snicholls::attributes_of(pnl{}));
    EXPECT_EQ(a.str(), R"({"sign_colours": {"positive": "#2da44e", "negative": "#cf222e", "zero": "#6e7781", "dead_zone": 0}, "format": "{:+.1f}"})");
    snicholls::write_json(b, snicholls::attributes_of(titled{}));
    EXPECT_EQ(b.str(), R"({"label": "T", "font": "bold"})");
    snicholls::write_json(c, snicholls::attribute_set<>{});
    EXPECT_EQ(c.str(), "{}");
}

TEST(DisplayJson, AnAttributeWithoutAttributeJsonIsLeftOut)
{
    struct unexported {
        int n;
    };
    std::ostringstream os;
    snicholls::write_json(os, snicholls::attrs(unexported{1}, snicholls::label{"L\n"}));
    EXPECT_EQ(os.str(), R"({"label": "L\n"})");
}

// Dead zones, thresholds, units, scale, minor units.
static_assert(snicholls::sign_colours{.dead_zone = 0.5}.of(0.4) == snicholls::colours::grey);
static_assert(snicholls::sign_colours{.dead_zone = 0.5}.of(-0.5) == snicholls::colours::grey);
static_assert(snicholls::sign_colours{.dead_zone = 0.5}.of(0.51) == green && snicholls::sign_colours{.dead_zone = 0.5}.of(-0.51) == red);
static_assert(!snicholls::sign_colours{.dead_zone = 0.5}.of(std::numeric_limits<double>::quiet_NaN()));
constexpr snicholls::threshold_colours staleness{snicholls::colours::green, {{5, snicholls::colours::yellow}, {15, snicholls::colours::red}}};
static_assert(staleness.of(4.99) == green && staleness.of(5) == snicholls::colours::yellow && staleness.of(14.9) == snicholls::colours::yellow);
static_assert(staleness.of(15) == red && staleness.of(1e9) == red && staleness.of(-1) == green && staleness.count == 2);
static_assert(!staleness.of(std::numeric_limits<double>::quiet_NaN()));
static_assert(snicholls::display_colour(snicholls::attributes_of(age_seconds{20}), age_seconds{20}) == red);   // threshold beats sign
static_assert(snicholls::display_colour(snicholls::attributes_of(age_seconds{1}), age_seconds{1}) == green);
static_assert(snicholls::attributes_of(iso::kwd).get<snicholls::minor_units>().digits == 3);
static_assert(snicholls::has_display_attributes<spread_bps> && snicholls::has_display_attributes<notional>);

TEST(DisplayNumbers, UnitsScaleAndFormatCompose)
{
    EXPECT_EQ(plain_text(spread_bps{12.5}), "12.5 bps");
    EXPECT_EQ(plain_text(notional{1'250'000}), "+1.25m USD");
    EXPECT_EQ(plain_text(notional{-40'000}), "-0.04m USD");
    EXPECT_EQ(plain_text(age_seconds{7.25}), "7.2 s");
    EXPECT_EQ(plain_text(iso::jpy), "JPY");
}

TEST(DisplayColour, ADeadZoneIsGreyAndThresholdsBeatSigns)
{
    const always_colour on;
    EXPECT_EQ(plain_text(pnl_banded{0.3}), "\x1b[90m+0.3\x1b[0m");
    EXPECT_EQ(plain_text(pnl_banded{-0.7}), "\x1b[31m-0.7\x1b[0m");
    EXPECT_EQ(plain_text(age_seconds{6}), "\x1b[33m6.0 s\x1b[0m");
    EXPECT_EQ(plain_text(age_seconds{16}), "\x1b[31m16.0 s\x1b[0m");
}

TEST(DisplayJson, TheNewAttributesAreExported)
{
    std::ostringstream a, b, c, d;
    snicholls::write_json(a, snicholls::attributes_of(pnl_banded{}));
    EXPECT_EQ(a.str(), R"({"sign_colours": {"positive": "#2da44e", "negative": "#cf222e", "zero": "#6e7781", "dead_zone": 0.5}, "format": "{:+.1f}"})");
    snicholls::write_json(b, snicholls::attributes_of(age_seconds{}));
    EXPECT_EQ(b.str(), R"({"threshold_colours": {"below": "#2da44e", "steps": [{"from": 5, "colour": "#bf8700"}, {"from": 15, "colour": "#cf222e"}]}, )"
                       R"("sign_colours": {"positive": "#2da44e", "negative": "#cf222e", "zero": "#6e7781", "dead_zone": 0}, "units": "s", "format": "{:.1f}"})");
    snicholls::write_json(c, snicholls::attributes_of(notional{}));
    EXPECT_EQ(c.str(), R"({"scale": {"factor": 1e+06, "suffix": "m"}, "format": "{:+.2f}", "units": "USD"})");
    snicholls::write_enum_json<iso>(d);
    EXPECT_EQ(d.str(), R"({"usd": {"label": "USD", "minor_units": 2}, "jpy": {"label": "JPY", "minor_units": 0}, "kwd": {"label": "KWD", "minor_units": 3}})");
}

} // namespace
