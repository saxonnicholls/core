//
//  trading_day.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase: one trading day on a small multi-asset book. Fills priced in exact money (their P&L proved by the
//  compiler), each order walked through the trade lifecycle to settlement two business days later, the day's P&L
//  pivoted by desk and instrument, and the overnight move of the book explained by market factor with Shapley and
//  Owen values over a full revaluation (Black-Scholes, computed at compile time) and by the greeks. Everything is
//  coloured by its type's attributes. Exits non-zero if any figure is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/finance/attribution.hpp>
#include <snicholls/core/fsm/securities.hpp>
#include <snicholls/core/functional/fold.hpp>
#include <snicholls/core/game/owen.hpp>
#include <snicholls/core/game/shapley.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/money/asset.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/stats/pivot.hpp>
#include <snicholls/core/time/business_days.hpp>
#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/utc.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

namespace day {

using namespace snicholls;
namespace chr = std::chrono;
namespace math = snicholls::math;

// ---- The vocabulary, and how each part of it looks wherever it prints ---------------------------------------
enum class desk : std::uint8_t { rates, credit, equity };
enum class instrument : std::uint8_t { bond, cds, stock, option };

constexpr auto attributes(desk d)
{
    return per_enumerator(d, [] { return std::array{attrs(colours::blue, label{"Rates"}, tooltip{"Government bonds"}),
                                                    attrs(colours::magenta, label{"Credit"}, tooltip{"Corporate credit"}),
                                                    attrs(colours::cyan, label{"Equity"}, tooltip{"Shares and options"})}; });
}
constexpr auto attributes(instrument i)
{
    return per_enumerator(i, [] { return std::array{attrs(rgb(0x953800), label{"bond"}), attrs(rgb(0x8250df), label{"CDS"}),
                                                    attrs(colours::yellow, label{"stock"}), attrs(colours::grey, label{"option"})}; });
}
struct pnl {   // green up, red down, grey within a cent
    double value;
};
constexpr auto attributes(const pnl&) { return attrs(sign_colours{.dead_zone = 0.005}, number_format{"{:+.2f}"}); }

// ---- The fills: exact amounts, and their P&L known before the program runs ----------------------------------
constexpr asset usd = checked_asset("USD", 2);
constexpr money dollars(std::string_view text) { return *parse_money(usd, text); }

struct fill {
    std::string_view id;
    desk book;
    instrument what;
    std::int64_t quantity;   // signed: a sale is negative
    std::string_view price, close;
};

// (close - price) x quantity, to the cent: no double anywhere.
constexpr money pnl_of(const fill& f)
{
    const money move = *subtract(dollars(f.close), dollars(f.price));
    const money size = *multiply(move, static_cast<std::uint64_t>(f.quantity < 0 ? -f.quantity : f.quantity));
    return f.quantity < 0 ? -size : size;
}

constexpr std::array<fill, 7> blotter{{
    {"T-001", desk::rates, instrument::bond, 2'000, "99.81", "99.91"},
    {"T-002", desk::rates, instrument::bond, -500, "100.12", "99.91"},
    {"T-003", desk::credit, instrument::cds, 10, "1247.50", "1231.00"},
    {"T-004", desk::equity, instrument::stock, 3'000, "187.42", "186.10"},
    {"T-005", desk::equity, instrument::stock, -1'200, "188.05", "186.10"},
    {"T-006", desk::equity, instrument::option, 400, "6.35", "7.10"},
    {"T-007", desk::credit, instrument::cds, -25, "1260.00", "1231.00"},
}};

// The day's P&L, folded over the blotter by the compiler, exactly.
constexpr money day_total = fn::foldl_of([](money a, const fill& f) { return *add(a, pnl_of(f)); }, money::zero(usd), blotter);
// For display only: money has no double, a decimal does
inline double as_double(const money& m) { return to_decimal(m)->to_double(); }

static_assert(pnl_of(blotter[0]) == dollars("200.00"));
static_assert(day_total == dollars("-455.00"));

// The pivot's measure: P&L in money, computed, so a getter; the keys are the fill's members. It shows as P&L does
struct pnl_value {
    double operator()(const fill& f) const { return as_double(pnl_of(f)); }
};
constexpr auto attributes(const pnl_value&) { return attributes(pnl{}); }

// ---- Settlement: T+2 business days, in the calendar of the type system ---------------------------------------
constexpr chr::sys_days trade_date = chr::year{2026} / chr::November / 26;   // a Thursday
constexpr chr::sys_days settle_date = add_business_days(trade_date, 2);
static_assert(settle_date == chr::sys_days{chr::year{2026} / chr::November / 30});   // over the weekend to Monday

// ---- Overnight: the book revalued in full, then explained ------------------------------------------------------
struct market {
    double rate, spread, spot, vol;
};
constexpr market yesterday{0.0410, 0.0125, 186.10, 0.24};
constexpr market today{0.0425, 0.0140, 181.70, 0.29};   // rates and spreads up, equities down, volatility bid

constexpr double call(double s, double k, double vol, double t, double r)
{
    const double sd = vol * math::sqrt(t), d1 = (math::log(s / k) + (r + vol * vol / 2) * t) / sd;
    return s * math::normal_cdf(d1) - k * math::exp(-r * t) * math::normal_cdf(d1 - sd);
}
constexpr double book_value(const market& m)
{
    const double y = m.rate + m.spread, dy = y - (yesterday.rate + yesterday.spread);
    const double bonds = 1'500'000 * (1 - 7.2 * dy + 0.5 * 62.0 * dy * dy);                 // duration and convexity
    const double protection = 750 * 10'000 * 4.1 * (m.spread - yesterday.spread);            // CDS bought: gains as spreads widen
    const double shares = 1'800 * m.spot;
    const double options = 400 * 100 * call(m.spot, 185.0, m.vol, 0.25, m.rate);
    return bonds + protection + shares + options;
}
// A coalition of factors is the set that moved: the rest stay at yesterday's level.
enum class factor : std::uint8_t { rates, credit, equity, volatility };
constexpr auto attributes(factor f)
{
    return per_enumerator(f, [] { return std::array{attrs(colours::blue, label{"rates"}), attrs(colours::magenta, label{"credit"}),
                                                    attrs(colours::cyan, label{"equity"}), attrs(colours::yellow, label{"volatility"})}; });
}
constexpr auto overnight = [](game::coalition s) {
    const market m{game::contains(s, 0) ? today.rate : yesterday.rate, game::contains(s, 1) ? today.spread : yesterday.spread,
                   game::contains(s, 2) ? today.spot : yesterday.spot, game::contains(s, 3) ? today.vol : yesterday.vol};
    return book_value(m) - book_value(yesterday);
};
constexpr auto by_factor = game::shapley<4>(overnight);
constexpr std::array<game::coalition, 2> asset_classes{0b0011, 0b1100};   // fixed income, equity
constexpr auto by_owen = game::owen<4>(overnight, asset_classes);
static_assert(math::abs(by_factor.unattributed()) < 1e-6, "Shapley attributes the whole move");
static_assert(by_factor.values[2] < 0 && by_factor.values[3] > 0, "equities cost, volatility paid");

// The option book by its greeks, by central differences of the same pricer (also at compile time).
constexpr finance::greeks<> option_greeks = [] {
    const auto v = [](double s, double vol, double r) { return 400 * 100 * call(s, 185.0, vol, 0.25, r); };
    const double s = yesterday.spot, vol = yesterday.vol, r = yesterday.rate, h = 0.01, hv = 0.0001;
    return finance::greeks<>{.delta = (v(s + h, vol, r) - v(s - h, vol, r)) / (2 * h),
                             .gamma = (v(s + h, vol, r) - 2 * v(s, vol, r) + v(s - h, vol, r)) / (h * h),
                             .vega = (v(s, vol + hv, r) - v(s, vol - hv, r)) / (2 * hv),
                             .vanna = (v(s + h, vol + hv, r) - v(s + h, vol - hv, r) - v(s - h, vol + hv, r) + v(s - h, vol - hv, r)) / (4 * h * hv),
                             .volga = (v(s, vol + hv, r) - 2 * v(s, vol, r) + v(s, vol - hv, r)) / (hv * hv),
                             .theta = 0.0,
                             .rho = (v(s, vol, r + hv) - v(s, vol, r - hv)) / (2 * hv)};
}();

} // namespace day

int main()
{
    using namespace day;
    using snicholls::acout;
    using snicholls::println;
    namespace fsm = snicholls::fsm;
    namespace stats = snicholls::stats;
    bool ok = true;
    const auto expect = [&](bool good, std::string_view what) {
        acout{} << "  " << (good ? snicholls::styled(snicholls::colours::green, "ok  ") : snicholls::styled(snicholls::colours::red, "FAIL")) << ' ' << what
                << '\n';
        ok &= good;
    };

    println(snicholls::colours::cyan, "A trading day, {} (T+2 settles {})", snicholls::to_string(chr::year_month_day{trade_date}),
            snicholls::to_string(chr::year_month_day{settle_date}));

    // ---- 1. The blotter: each order through its lifecycle -------------------------------------------------------
    println("\n1. Orders, filled at their prices and marked at the close (exact, in cents)");
    snicholls::utc_ms stamp = snicholls::utc_ms{trade_date} + chr::hours{14} + chr::minutes{30};
    std::vector<fsm::trade_state> states;
    for (const fill& f : blotter) {
        fsm::sm<fsm::trade_lifecycle> order;   // placed
        order.process_event(fsm::trade_event::fill{});
        order.process_event(fsm::trade_event::instruct{});
        // T-007's counterparty misses its delivery: the trade fails, every other settles on the value date
        f.id == "T-007" ? order.process_event(fsm::trade_event::fail{}) : order.process_event(fsm::trade_event::settle{});
        states.push_back(order.current());
        acout{} << "  " << f.id << "  " << stamp << "  " << f.book << ' ' << f.what << "  " << f.quantity << " @ " << f.price << " -> "
                << f.close << "  " << order.current() << "  " << pnl{as_double(pnl_of(f))} << '\n';
        stamp += chr::minutes{17};
    }
    acout{} << "  total " << day_total << ", folded by the compiler: " << pnl{as_double(day_total)} << '\n';
    expect(states.back() == fsm::trade_state::failed && states.front() == fsm::trade_state::settled, "six settle on Monday, one fails");

    // ---- 2. The pivot: P&L by desk and instrument, one pass --------------------------------------------------------
    println("\n2. P&L by desk and instrument");
    using grid = stats::pivot<fill, snicholls::meta::type_list<stats::member<&fill::book>>, snicholls::meta::type_list<stats::member<&fill::what>>,
                              stats::measure<pnl_value, stats::sum<double>>>;
    grid book;
    book.add_all(blotter);
    acout{} << book;
    const double equity_stock = *book.value<0>({desk::equity, instrument::stock});
    expect(equity_stock == -1620.00, "equity desk's stock: -3960 on the buy, +2340 on the sale");
    expect(std::abs(std::get<0>(book.total()).value() - as_double(day_total)) < 1e-9, "the grid's total is the exact total");

    // ---- 3. Overnight: why the book moved ------------------------------------------------------------------------
    println("\n3. Overnight the market moved; the book is revalued in full and the move explained");
    const double moved = book_value(today) - book_value(yesterday);
    acout{} << "  move " << pnl{moved} << '\n';
    acout{} << "  Shapley by factor (computed by the compiler):\n";
    for (std::size_t i = 0; i < 4; ++i)
        acout{} << "    " << static_cast<factor>(i) << '\t' << pnl{by_factor.values[i]} << '\n';
    const auto brinson = snicholls::game::one_at_a_time<4>(overnight);
    acout{} << "  one factor at a time leaves " << pnl{brinson.unattributed()} << " unexplained; Shapley leaves "
            << pnl{by_factor.unattributed()} << '\n';
    acout{} << "  Owen by asset class: fixed income " << pnl{by_owen.values[0] + by_owen.values[1]} << ", equity "
            << pnl{by_owen.values[2] + by_owen.values[3]} << '\n';
    expect(std::abs(by_factor.attributed() - by_factor.baseline - moved) < 1e-6, "the factors add up to the move");
    expect(std::abs(brinson.unattributed()) > 1.0, "one at a time does not");
    expect(std::abs(by_owen.unattributed()) < 1e-6, "Owen attributes it all, within asset classes first");

    // The option book alone, by its greeks: the Taylor terms against its own full revaluation
    constexpr auto options_only = [](const market& m) { return 400 * 100 * call(m.spot, 185.0, m.vol, 0.25, m.rate); };
    constexpr double option_move = options_only(today) - options_only(yesterday);
    constexpr auto why = snicholls::finance::explain(
        option_greeks, {.spot = today.spot - yesterday.spot, .vol = today.vol - yesterday.vol, .time = 0, .rate = today.rate - yesterday.rate},
        option_move);
    acout{} << "  options by the greeks: delta " << pnl{why.delta} << " gamma " << pnl{why.gamma} << " vega " << pnl{why.vega} << " vanna "
            << pnl{why.vanna} << " volga " << pnl{why.volga} << " rho " << pnl{why.rho} << "\n    explained " << pnl{why.explained}
            << " of " << pnl{why.actual} << ", unexplained " << pnl{why.unexplained} << '\n';
    static_assert(math::abs(why.unexplained) < 0.10 * math::abs(why.actual), "second order explains a large move to within 10%");
    expect(why.unexplained < 0 && std::abs(why.unexplained) > 0.05 * std::abs(why.actual), "a large move leaves third-order terms: the greeks miss 5-10%");

    acout{} << '\n';
    println(ok ? snicholls::colours::green : snicholls::colours::red, "trading day: {}", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
