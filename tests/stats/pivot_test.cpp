//
//  pivot_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/stats/pivot.hpp>: keys read by type and by getter, enumeration, bool,
//  declared and discovered dimensions, every measure in one pass, margins, merging partial pivots,
//  dropped rows, and the printed grid.
//

#include <snicholls/core/stats/pivot.hpp>

#include <snicholls/core/fixed_string.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace book {
enum class currency : std::uint8_t { usd, eur, jpy };
enum class equity_index : std::uint8_t { spx, sx5e };
struct pnl {
    double value;
};
struct ret {
    double value;
};
struct desk {
    std::string name;
    friend bool operator==(const desk&, const desk&) = default;
    friend std::ostream& operator<<(std::ostream& os, const desk& d) { return os << d.name; }
};
using row = std::tuple<currency, equity_index, pnl, ret, desk, bool>;

// Getters: any default-constructible callable type
struct pnl_of {
    double operator()(const row& r) const { return std::get<pnl>(r).value; }
};
struct ret_of {
    double operator()(const row& r) const { return std::get<ret>(r).value; }
};
struct desk_name {
    std::string operator()(const row& r) const { return std::get<desk>(r).name; }
};
struct winner {
    bool operator()(const row& r) const { return std::get<pnl>(r).value > 0; }
};
// A key with a declared domain: placed by arithmetic like an enumeration
struct region {
    char code;
    friend constexpr bool operator==(region, region) = default;
    friend std::ostream& operator<<(std::ostream& os, region r) { return os << r.code; }
};
struct region_of {
    region operator()(const row& r) const { return std::get<currency>(r) == currency::usd ? region{'A'} : region{'E'}; }
};
} // namespace book

template <>
struct std::hash<book::desk> {
    std::size_t operator()(const book::desk& d) const noexcept { return std::hash<std::string>{}(d.name); }
};

template <>
struct snicholls::stats::pivot_domain<book::region> {
    static constexpr std::array values{book::region{'A'}, book::region{'E'}};
};

namespace {

using namespace book;
namespace stats = snicholls::stats;
using snicholls::meta::type_list;

const std::vector<row> trades{
    {currency::usd, equity_index::spx, {10}, {0.010}, {"delta"}, true},
    {currency::usd, equity_index::spx, {-4}, {-0.002}, {"vol"}, true},
    {currency::usd, equity_index::sx5e, {3}, {0.004}, {"delta"}, false},
    {currency::eur, equity_index::sx5e, {7}, {0.006}, {"vol"}, true},
    {currency::jpy, equity_index::spx, {-1}, {-0.001}, {"delta"}, false},
    {currency::eur, equity_index::sx5e, {2}, {0.002}, {"delta"}, true},
};

using by_currency_and_index =
    stats::pivot<row, type_list<currency>, type_list<equity_index>, stats::measure<pnl_of, stats::sum<double>>,
                 stats::measure<ret_of, stats::mean<double>>, stats::measure<pnl_of, stats::count>>;

TEST(Pivot, EveryMeasureOfEveryCellInOnePass)
{
    by_currency_and_index p;
    p.add_all(trades);
    EXPECT_EQ(p.rows(), 6u);
    EXPECT_EQ((p.value<0>({currency::usd, equity_index::spx})), 6.0);
    EXPECT_DOUBLE_EQ((*p.value<1>({currency::usd, equity_index::spx})), 0.004);
    EXPECT_EQ((p.value<2>({currency::usd, equity_index::spx})), 2u);
    EXPECT_EQ((p.value<0>({currency::eur, equity_index::sx5e})), 9.0);
    EXPECT_FALSE((p.value<0>({currency::jpy, equity_index::sx5e})).has_value());   // no row reached it
    EXPECT_EQ(p.labels<0>(), (std::vector<currency>{currency::usd, currency::eur, currency::jpy}));
}

TEST(Pivot, MarginsAndTheGrandTotal)
{
    const auto p = stats::make_pivot<row, type_list<currency>, type_list<equity_index>, stats::measure<pnl_of, stats::sum<double>>>(trades);
    EXPECT_EQ(std::get<0>(p.row_total({currency::usd})).value(), 9.0);
    EXPECT_EQ(std::get<0>(p.column_total({equity_index::sx5e})).value(), 12.0);
    EXPECT_EQ(std::get<0>(p.total()).value(), 17.0);
}

TEST(Pivot, KeysReadByTypeByGetterDiscoveredBoolAndDeclared)
{
    // desk names discovered at run time, a bool from a getter, a declared domain
    stats::pivot<row, type_list<desk_name, winner>, type_list<region_of>, stats::measure<pnl_of, stats::sum<double>>> p;
    p.add_all(trades);
    EXPECT_EQ((p.value<0>({"delta", true, region{'A'}})), 13.0);
    EXPECT_EQ((p.value<0>({"vol", false, region{'A'}})), -4.0);
    EXPECT_EQ((p.value<0>({"vol", true, region{'E'}})), 7.0);
    EXPECT_EQ(p.labels<0>(), (std::vector<std::string>{"delta", "vol"}));   // in order of first appearance
    // a key type read with std::get
    stats::pivot<row, type_list<desk>, type_list<>, stats::measure<pnl_of, stats::sum<double>>> by_desk;
    by_desk.add_all(trades);
    EXPECT_EQ((by_desk.value<0>({desk{"delta"}})), 14.0);
}

TEST(Pivot, ManyDiscoveredKeysWidenTheTableAndKeepEveryCell)
{
    stats::pivot<row, type_list<desk_name>, type_list<currency>, stats::measure<pnl_of, stats::sum<double>>> p;
    for (int i = 0; i < 100; ++i)
        p.add({currency::usd, equity_index::spx, {static_cast<double>(i)}, {0}, {"desk" + std::to_string(i % 37)}, true});
    double total = 0;
    for (int d = 0; d < 37; ++d)
        total += *p.value<0>({"desk" + std::to_string(d), currency::usd});
    EXPECT_EQ(total, 4950.0);
    EXPECT_EQ(std::get<0>(p.total()).value(), 4950.0);
}

TEST(Pivot, PartialPivotsMergeIntoTheOnePassResult)
{
    by_currency_and_index whole, first, second;
    whole.add_all(trades);
    for (std::size_t i = 0; i < trades.size(); ++i)
        (i % 2 == 0 ? first : second).add(trades[i]);
    first.merge(second);
    for (const auto c : {currency::usd, currency::eur, currency::jpy})
        for (const auto x : {equity_index::spx, equity_index::sx5e}) {
            EXPECT_EQ((first.value<0>({c, x})), whole.value<0>({c, x}));
            EXPECT_EQ((first.value<2>({c, x})), whole.value<2>({c, x}));
            if (const auto m = whole.value<1>({c, x})) {
                EXPECT_NEAR((*first.value<1>({c, x})), *m, 1e-15);
            }
        }
    EXPECT_EQ(first.rows(), 6u);
}

TEST(Pivot, ARowWhoseEnumerationKeyIsNotAnEnumeratorIsDropped)
{
    by_currency_and_index p;
    p.add({static_cast<currency>(9), equity_index::spx, {1}, {0}, {"x"}, true});
    EXPECT_EQ(p.rows(), 1u);
    EXPECT_EQ(p.dropped(), 1u);
}

TEST(Pivot, PrintsMeasureZeroAsAGrid)
{
    const auto p = stats::make_pivot<row, type_list<currency>, type_list<equity_index>, stats::measure<pnl_of, stats::sum<double>>>(trades);
    std::ostringstream os;
    p.print(os);
    EXPECT_EQ(os.str(), "\tspx\tsx5e\nusd\t6\t3\neur\t\t9\njpy\t-1\t\n");
}

TEST(Pivot, AccumulatorsWithParametersStartFromAPrototype)
{
    stats::pivot<row, type_list<currency>, type_list<>, stats::measure<ret_of, stats::histogram<double, 2>>> p(
        stats::histogram<double, 2>{.lo = -0.01, .hi = 0.01});
    p.add_all(trades);
    EXPECT_EQ((p.value<0>({currency::usd}))->at(0), 1u);   // -0.002
    EXPECT_EQ((p.value<0>({currency::usd}))->at(1), 1u);   // 0.004 (0.010 is above)
}

// ---- names, tag structs, collected cells, per-cell bootstraps --------------------------------------

// A row read by compile-time name
struct named_fill {
    book::currency ccy;
    double pnl;
    template <snicholls::basic_fixed_string Name>
    auto get() const
    {
        if constexpr (Name == snicholls::fixed_string{"ccy"})
            return ccy;
        else
            return pnl;
    }
};

// Nested-struct tags
struct ccy_tag {
    static book::currency get(const named_fill& f) { return f.ccy; }
};

TEST(PivotKeys, ByNameAndByTagStruct)
{
    const std::vector<named_fill> fills{{currency::usd, 5}, {currency::eur, 2}, {currency::usd, 1}};
    stats::pivot<named_fill, type_list<stats::by_name<"ccy">>, type_list<>, stats::measure<stats::by_name<"pnl">, stats::sum<double>>> by_name;
    by_name.add_all(fills);
    EXPECT_EQ((by_name.value<0>({currency::usd})), 6.0);
    stats::pivot<named_fill, type_list<ccy_tag>, type_list<>, stats::measure<stats::by_name<"pnl">, stats::count>> by_tag;
    by_tag.add_all(fills);
    EXPECT_EQ((by_tag.value<0>({currency::usd})), 2u);
}

std::vector<row> many_trades()
{
    std::vector<row> out;
    snicholls::stats::wyrand g(17);
    for (int i = 0; i < 4000; ++i) {
        const double x = snicholls::stats::uniform_real(g) - 0.5;
        out.push_back({i % 2 == 0 ? currency::usd : currency::eur, equity_index::spx, {x}, {x}, {"d"}, true});
    }
    return out;
}

TEST(PivotCells, CollectKeepsEachCellsValuesForAnyStatistic)
{
    const auto trades = many_trades();
    stats::pivot<row, type_list<currency>, type_list<>, stats::measure<pnl_of, stats::collect<double>>> p;
    p.add_all(trades);
    const auto usd = p.value<0>({currency::usd});
    ASSERT_TRUE(usd.has_value());
    EXPECT_EQ(usd->size(), 2000u);
    snicholls::stats::wyrand g(1);
    const auto d = stats::bootstrap(*usd, 500, g, [](std::span<const double> v) {
        double s = 0;
        for (const double x : v)
            s += x;
        return s / static_cast<double>(v.size());
    });
    EXPECT_NEAR(d.standard_error(), std::sqrt(1.0 / 12 / 2000), 0.15 * std::sqrt(1.0 / 12 / 2000));
}

TEST(PivotCells, EachCellCarriesItsOwnPoissonBootstrap)
{
    const auto trades = many_trades();
    using boot = stats::bootstrapped<stats::mean<double>, 400>;
    stats::pivot<row, type_list<currency>, type_list<>, stats::measure<pnl_of, boot>> p;
    p.add_all(trades);
    const auto usd = *p.value<0>({currency::usd});
    const auto eur = *p.value<0>({currency::eur});
    const double expected = std::sqrt(1.0 / 12 / 2000);
    EXPECT_NEAR(usd.standard_error(), expected, 0.15 * expected);
    EXPECT_NEAR(eur.standard_error(), expected, 0.15 * expected);
    EXPECT_NE(usd.replicates[0] - usd.estimate, eur.replicates[0] - eur.estimate);   // independent streams per cell
    // Merging replicate by replicate
    stats::pivot<row, type_list<currency>, type_list<>, stats::measure<pnl_of, boot>> a, b;
    for (std::size_t i = 0; i < trades.size(); ++i)
        (i < 1000 ? a : b).add(trades[i]);
    a.merge(b);
    EXPECT_NEAR(a.value<0>({currency::usd})->estimate, usd.estimate, 1e-12);
    EXPECT_NEAR(a.value<0>({currency::usd})->standard_error(), expected, 0.15 * expected);
}

} // namespace
