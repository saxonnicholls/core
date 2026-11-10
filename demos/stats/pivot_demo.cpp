//
//  pivot_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/pivot.hpp>: P&L and return by currency and index from a blotter
//  of strong-typed tuples, in one pass, two workers' pivots merged, and the grid printed through acout in
//  each currency's and index's colour with P&L green or red. Exits non-zero if a cell is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/stats/pivot.hpp>

#include <array>

#include <cstdint>
#include <cstdio>
#include <sstream>
#include <tuple>
#include <vector>

namespace blotter {
enum class currency : std::uint8_t { usd, eur, gbp };
enum class equity_index : std::uint8_t { spx, sx5e, ukx };
struct pnl {
    double value;
};
struct ret {
    double value;
};
using fill = std::tuple<currency, equity_index, pnl, ret>;
struct pnl_of {   // a getter: any callable type
    double operator()(const fill& f) const { return std::get<pnl>(f).value; }
};

//! [pivot_attributes]
// Declared once, beside the types: every printer (acout, the pivot grid, imgui and React through the JSON) shows a
// currency and an index in the same colour, and P&L green when up and red when down (so no identity is green or red).
constexpr auto attributes(currency c)
{
    using namespace snicholls;
    return per_enumerator(c, [] { return std::array{attrs(colours::cyan, label{"USD"}, tooltip{"US dollar"}),
                                        attrs(colours::blue, label{"EUR"}, tooltip{"Euro"}),
                                        attrs(colours::magenta, label{"GBP"}, tooltip{"Pound sterling"})}; });
}
constexpr auto attributes(equity_index i)
{
    using namespace snicholls;
    return per_enumerator(i, [] { return std::array{attrs(rgb(0x953800), label{"S&P 500"}, tooltip{"Standard & Poor's 500"}),
                                        attrs(colours::yellow, label{"Euro Stoxx 50"}, tooltip{"Euro Stoxx 50"}),
                                        attrs(colours::orange, label{"FTSE 100"}, tooltip{"FTSE 100"})}; });
}
constexpr auto attributes(const pnl_of&) { return snicholls::attrs(snicholls::sign_colours{}, snicholls::number_format{"{:+.0f}"}); }
//! [pivot_attributes]
struct ret_of {
    double operator()(const fill& f) const { return std::get<ret>(f).value; }
};
} // namespace blotter

int main()
{
    std::printf("snicholls-core pivot demo\n");
    using namespace blotter;
    namespace stats = snicholls::stats;
    bool ok = true;

    //! [pivot]
    // Rows by currency, columns by index (read with std::get), P&L summed and return averaged
    // (read by getters), every cell filled in one pass over the fills.
    using by_currency_and_index = stats::pivot<fill, snicholls::meta::type_list<currency>, snicholls::meta::type_list<equity_index>,
                                               stats::measure<pnl_of, stats::sum<double>>, stats::measure<ret_of, stats::mean<double>>>;
    const std::vector<fill> fills{{currency::usd, equity_index::spx, {120}, {0.012}},
                                  {currency::eur, equity_index::sx5e, {-40}, {-0.004}},
                                  {currency::usd, equity_index::spx, {30}, {0.003}},
                                  {currency::gbp, equity_index::ukx, {15}, {0.002}}};
    by_currency_and_index today;
    today.add_all(fills);
    const auto usd_spx = today.value<0>({currency::usd, equity_index::spx});   // 150
    const auto total = std::get<0>(today.total()).value();               // 125

    // In parallel: each worker pivots its share, then the pivots merge.
    by_currency_and_index worker_a, worker_b;
    worker_a.add(fills[0]);
    worker_b.add_all(std::vector<fill>(fills.begin() + 1, fills.end()));
    worker_a.merge(worker_b);
    //! [pivot]

    std::ostringstream grid;
    today.print(grid);
    std::printf("%s", grid.str().c_str());
    ok &= grid.str().find("USD") != std::string::npos && grid.str().find("+150") != std::string::npos && grid.str().find("-40") != std::string::npos;

    //! [pivot_colour]
    snicholls::acout{} << today;   // the same grid, in colour on a terminal
    //! [pivot_colour]
    std::printf("  usd/spx %.0f, total %.0f, merged usd/spx %.0f\n", *usd_spx, total, *worker_a.value<0>({currency::usd, equity_index::spx}));
    ok &= usd_spx == 150.0 && total == 125.0 && worker_a.value<0>({currency::usd, equity_index::spx}) == 150.0;

    std::printf(ok ? "pivot demo: ok\n" : "pivot demo: FAILED\n");
    return ok ? 0 : 1;
}
