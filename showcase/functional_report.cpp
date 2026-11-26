//
//  functional_report.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase: C++20 written the way a Haskell programmer would write it. A book of positions is a plain aggregate,
//  reflected (its members counted and typed, read as a tuple); the report on it is built from curried, composed,
//  mapped and folded functions, a pattern match over corporate actions that destructures them, Maybe and Either
//  railways over a price feed, and a State computation that runs a margin call - and every number in it is computed
//  by the compiler and proved by static_assert. At run time the same functions drive a push pipeline over the same
//  book and must give the same answers; the report prints through acout in each sector's colour. Exits non-zero if
//  any result is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/constexpr_map.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/curry.hpp>
#include <snicholls/core/functional/fold.hpp>
#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/functional/match.hpp>
#include <snicholls/core/functional/monad.hpp>
#include <snicholls/core/functional/state.hpp>
#include <snicholls/core/functional/typeclasses.hpp>
#include <snicholls/core/math/constexpr_math.hpp>
#include <snicholls/core/meta/type_list.hpp>
#include <snicholls/core/pipeline/stages.hpp>
#include <snicholls/core/reflect/member_count.hpp>
#include <snicholls/core/reflect/struct_tuple.hpp>
#include <snicholls/core/result.hpp>

#include <array>
#include <functional>
#include <cstdint>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace report {

using namespace snicholls;

// ---- The data: plain aggregates, nothing to register -------------------------------------------------------------
enum class sector : std::uint8_t { tech, energy, banks, health };
constexpr auto attributes(sector s)
{
    return per_enumerator(s, [] { return std::array{attrs(colours::cyan, label{"tech"}), attrs(colours::yellow, label{"energy"}),
                                                    attrs(colours::blue, label{"banks"}), attrs(colours::magenta, label{"health"})}; });
}
struct exposure {   // signed money at risk: green long, red short
    double value;
};
constexpr auto attributes(const exposure&) { return attrs(sign_colours{}, number_format{"{:+.0f}"}); }
struct amount {   // a size with no direction: gross, a limit
    double value;
};
constexpr auto attributes(const amount&) { return attrs(colours::grey, number_format{"{:.0f}"}); }

struct position {
    std::string_view symbol;
    sector where;
    double quantity;   // negative: short
    double price;
    double beta;
};

constexpr std::array<position, 8> book{{{"NVDA", sector::tech, 1'200, 180.0, 1.7},
                                        {"MSFT", sector::tech, 800, 410.0, 1.1},
                                        {"XOM", sector::energy, -1'500, 112.0, 0.9},
                                        {"CVX", sector::energy, 600, 150.0, 0.95},
                                        {"JPM", sector::banks, 900, 240.0, 1.2},
                                        {"WFC", sector::banks, -1'000, 70.0, 1.15},
                                        {"LLY", sector::health, 250, 760.0, 0.6},
                                        {"PFE", sector::health, -4'000, 26.0, 0.7}}};

// ---- Reflection: the compiler reads the struct ---------------------------------------------------------------------
static_assert(member_count_v<position> == 5);
using fields = meta::as_type_list_t<decltype(to_tuple(book[0]))>;
static_assert(std::is_same_v<fields, meta::type_list<std::string_view, sector, double, double, double>>);
static_assert(meta::count_v<fields, double> == 3 && meta::index_of_v<fields, sector> == 1);
static_assert(std::get<0>(to_tuple(book[2])) == "XOM" && from_tuple<position>(to_tuple(book[2])).quantity == -1'500);

// ---- Small functions, composed -------------------------------------------------------------------------------------
constexpr auto notional = [](const position& p) { return p.quantity * p.price; };
constexpr auto beta_of = [](const position& p) { return p.beta; };
constexpr auto times = fn::curry([](double a, double b) { return a * b; });
constexpr auto magnitude = [](double x) { return x < 0 ? -x : x; };
constexpr auto beta_adjusted = [](const position& p) { return notional(p) * beta_of(p); };
constexpr auto gross = fn::pipe(notional, magnitude);                 // |q p|
constexpr auto in_thousands = fn::pipe(notional, times(1e-3));        // a curried multiplication, half applied

static_assert(in_thousands(book[0]) == 216.0 && gross(book[2]) == 168'000.0);

// fmap over the book; scanl for the running net; a fold that takes three summaries in one pass (a tuple of monoids)
constexpr auto notionals = fn::fmap(notional, book);
constexpr auto running_net = fn::scanl([](double acc, double n) { return acc + n; }, 0.0, notionals);
constexpr auto summary = fn::fold_map([](const position& p) {
    return std::tuple{fn::sum_of<double>{gross(p)}, fn::max_of<double>{gross(p)}, fn::sum_of<double>{beta_adjusted(p)}};
}, book);
constexpr double gross_total = std::get<0>(summary).value, largest = std::get<1>(summary).value, beta_net = std::get<2>(summary).value;
static_assert(running_net.back() == fn::foldl_of(std::plus<>{}, 0.0, notionals));
static_assert(gross_total == 1'382'000.0 && largest == 328'000.0);

// Sector limits: a table written once, looked up both ways
constexpr auto limit_of = make_constexpr_map(std::pair{sector::tech, 500'000.0}, std::pair{sector::energy, 300'000.0},
                                             std::pair{sector::banks, 300'000.0}, std::pair{sector::health, 300'000.0});
static_assert(limit_of.keys_unique());
constexpr auto sector_gross = [](sector s) {
    return fn::foldl_of([s](double acc, const position& p) { return acc + (p.where == s ? gross(p) : 0.0); }, 0.0, book);
};
constexpr auto breaches = fn::partition([](const position& p) { return sector_gross(p.where) > limit_of.at(p.where); }, book);
static_assert(breaches.first.size() == 2 && breaches.first[0].where == sector::tech, "tech is over its limit: NVDA and MSFT");

// Longs and shorts, and the largest risk first
constexpr auto by_risk = fn::sort_on([](const position& p) { return -magnitude(beta_adjusted(p)); }, book);
static_assert(by_risk[0].symbol == "NVDA" && by_risk[7].symbol == "PFE");

// ---- Pattern matching: corporate actions, destructured -------------------------------------------------------------
struct split {
    int to, from;
};
struct dividend {
    double per_share;
};
struct delisted {};
using action = std::variant<split, dividend, delisted>;

constexpr auto apply_action = [](position p, const action& a) {
    return fn::match(a,
                     fn::destructure([&](int to, int from) { return position{p.symbol, p.where, p.quantity * to / from, p.price * from / to, p.beta}; }),
                     fn::guard([&](const dividend&) { return p.quantity < 0; },
                               fn::destructure([&](double d) { return position{p.symbol, p.where, p.quantity, p.price - d, p.beta}; })),   // a short pays it
                     [&](const dividend&) { return p; },
                     [&](delisted) { return position{p.symbol, p.where, 0.0, 0.0, p.beta}; });
};
static_assert(apply_action(book[0], split{4, 1}).quantity == 4'800 && notional(apply_action(book[0], split{4, 1})) == notional(book[0]),
              "a split changes the shares, not the value");
static_assert(apply_action(book[1], dividend{0.83}).price == 410.0 && apply_action(book[2], dividend{0.99}).price == 111.01);

// ---- Maybe and Either: a price feed that may be missing or broken --------------------------------------------------
enum class feed_error : std::uint8_t { stale, crossed };
constexpr auto mid = [](double bid, double ask) -> result<double, feed_error> {
    if (ask < bid)
        return err(feed_error::crossed);
    return ok((bid + ask) / 2);
};
constexpr auto revalue = fn::lift([](double quantity, double price) { return quantity * price; });
static_assert(revalue(std::optional{1'200.0}, std::optional{181.0}) == std::optional{217'200.0});
static_assert(!revalue(std::optional{1'200.0}, std::optional<double>{}).has_value(), "no price, no value: Nothing");
constexpr auto marks = fn::traverse([](std::pair<double, double> q) { return mid(q.first, q.second); },
                                    std::tuple{std::pair{179.9, 180.1}, std::pair{409.8, 410.2}});
constexpr auto broken = fn::traverse([](std::pair<double, double> q) { return mid(q.first, q.second); },
                                     std::tuple{std::pair{179.9, 180.1}, std::pair{410.4, 410.2}});
static_assert(marks.is_ok() && std::get<0>(*marks) == 180.0 && broken.unwrap_err_unchecked() == feed_error::crossed);

// ---- State: a margin call, as a computation over the cash balance ---------------------------------------------------
constexpr auto post = [](double margin) {   // pay margin if the cash covers it; report whether it did
    return fn::mbind(fn::get_state, [margin](double cash) { return fn::mthen(fn::put_state(cash - margin), fn::pure<fn::state<>>(cash >= margin)); });
};
constexpr auto margin_day = fn::mbind(post(0.05 * gross_total), [](bool first) {
    return fn::fmap([first](bool second) { return first && second; }, post(0.02 * gross_total));
});
static_assert(fn::eval_state(margin_day, 100'000.0) && math::abs(fn::exec_state(margin_day, 100'000.0) - (100'000.0 - 0.07 * gross_total)) < 1e-9);
static_assert(!fn::eval_state(margin_day, 80'000.0), "80k covers the first call, not the second");

} // namespace report

int main()
{
    using namespace report;
    namespace pl = snicholls::pipeline;
    bool ok = true;
    const auto expect = [&](bool good, std::string_view what) {
        acout{} << "  " << styled(good ? colours::green : colours::red, good ? "ok  " : "FAIL") << ' ' << what << '\n';
        ok &= good;
    };
    println(colours::cyan, "A risk report the compiler wrote: {} positions, {} fields each", book.size(), member_count_v<position>);

    println("\n1. The book, largest risk first (sort_on), each row a reflected tuple");
    for (const position& p : by_risk)
        acout{} << "  " << to_tuple(p) << "  notional " << exposure{notional(p)} << "  beta-adjusted " << exposure{beta_adjusted(p)} << '\n';

    println("\n2. Summaries from one fold");
    acout{} << "  gross " << amount{gross_total} << ", largest " << amount{largest} << ", net " << exposure{running_net.back()}
            << ", beta-adjusted net " << exposure{beta_net} << '\n';
    acout{} << "  running net (scanl) " << running_net << '\n';
    for (const auto s : enum_values_v<sector>)
        acout{} << "  " << s << "  gross " << amount{sector_gross(s)} << " of a " << amount{limit_of.at(s)} << " limit"
                << (sector_gross(s) > limit_of.at(s) ? styled(colours::red, std::string_view{"  over"}) : styled(colours::grey, std::string_view{})) << '\n';

    println("\n3. Corporate actions (match), feeds (Maybe, Either), a margin call (State)");
    acout{} << "  NVDA after a 4-for-1 split: " << to_tuple(apply_action(book[0], split{4, 1})) << '\n';
    acout{} << "  marks " << *marks << "; a crossed quote gives " << broken.unwrap_err_unchecked() << '\n';
    acout{} << "  margin of " << exposure{-0.07 * gross_total} << " from 100k leaves " << exposure{fn::exec_state(margin_day, 100'000.0)} << '\n';

    println("\n4. The same functions at run time, pushed through a pipeline");
    double net_beta = 0, long_gross = 0;
    int longs = 0;
    auto beta_sink = pl::transform(beta_adjusted) | [&](double x) { net_beta += x; };
    auto long_sink = pl::filter([](const position& p) { return p.quantity > 0; }) | pl::tap([&](const position&) { ++longs; }) |
                     pl::transform(gross) | [&](double x) { long_gross += x; };
    for (const position& p : book) {
        beta_sink(p);
        long_sink(p);
    }
    constexpr double long_gross_at_compile_time = fn::foldl_of([](double acc, const position& p) { return acc + gross(p); }, 0.0,
                                                                  fn::partition([](const position& p) { return p.quantity > 0; }, book).first);
    acout{} << "  " << longs << " longs, gross " << amount{long_gross} << "; beta-adjusted net " << exposure{net_beta} << '\n';
    expect(net_beta == beta_net, "the pipeline's beta-adjusted net is the fold's, bit for bit");
    expect(long_gross == long_gross_at_compile_time && longs == 5, "and its long gross is the compile-time partition's");

    acout{} << '\n';
    println(ok ? colours::green : colours::red, "functional report: {}", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
