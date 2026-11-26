//
//  rust_iter_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/rust_iter.hpp>: Rust's slice and iterator methods that C++20 lacks.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/rust_iter.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

struct fill {
    std::string_view venue;
    double price;
    int quantity;
};
} // namespace

int main()
{
    namespace slice = snicholls::fn::rust::slice;
    namespace iter = snicholls::fn::rust::iter;
    std::printf("snicholls-core functional/rust_iter demo\n");

    //! [slices]
    // A three-day moving average and five-day bars, over a std::vector, with no copies.
    const std::vector<double> close{100, 101, 103, 102, 104, 107, 106};
    std::vector<double> average;
    for (std::span<const double> w : slice::windows(close, 3)) average.push_back((w[0] + w[1] + w[2]) / 3);
    std::size_t bars = 0;
    for (std::span<const double> bar : slice::chunks(close, 5)) bars += bar.empty() ? 0 : 1;
    //! [slices]
    check(average.size() == 5 && average[0] == 304.0 / 3 && average[4] == 317.0 / 3, "windows: five 3-day averages of seven closes");
    check(bars == 2 && slice::chunks(close, 5)[1].size() == 2, "chunks: a bar of five and a bar of two");

    const auto pairs = slice::chunks_exact(close, 2);
    check(pairs.size() == 3 && pairs.remainder().size() == 1 && pairs.remainder()[0] == 106, "chunks_exact: three pairs and one left");
    double spread_sum = 0;
    for (std::span<const double, 2> p : slice::chunks_exact_of<2>(close)) spread_sum += p[1] - p[0];
    check(spread_sum == 1 + -1 + 3, "chunks_exact_of<2>: spans of static extent");
    double gain = 0;
    for (std::span<const double, 2> d : slice::windows_of<2>(close)) gain += d[1] > d[0] ? d[1] - d[0] : 0;
    check(gain == 1 + 2 + 2 + 3, "windows_of<2>: the up moves add to 8");

    constexpr std::array<int, 8> ticks{5, 5, 5, 6, 6, 5, 7, 7};
    check(slice::dedup(ticks).size() == 4, "dedup: four distinct runs");
    check(slice::dedup_by_key(ticks, [](int t) { return t >= 6; }).size() == 4, "dedup_by_key: runs above and below 6");

    //! [iter]
    // The best fill by price, the first venue that filled, and every other fill, by Rust's names.
    const std::array<fill, 4> fills{{{"XLON", 100.2, 300}, {"XPAR", 100.1, 0}, {"XAMS", 100.1, 200}, {"XETR", 100.3, 100}}};
    const auto cheapest = iter::min_by_key(fills, [](const fill& f) { return f.price; });   // the first of equal prices
    const auto first_live = iter::find_map(fills, [](const fill& f) {
        return f.quantity > 0 ? std::optional<std::string_view>(f.venue) : std::nullopt;
    });
    int every_other = 0;
    for (const fill& f : iter::step_by(fills, 2)) every_other += f.quantity;
    //! [iter]
    check(cheapest->venue == "XPAR" && first_live == std::optional<std::string_view>("XLON") && every_other == 500,
          "min_by_key, find_map, step_by");

    check(iter::max_by_key(fills, [](const fill& f) { return f.quantity; })->venue == "XLON", "max_by_key: the most filled");
    check(iter::max_by(fills, [](const fill& a, const fill& b) { return a.price <=> b.price; })->venue == "XETR", "max_by: the dearest");
    check(iter::min_by(fills, [](const fill& a, const fill& b) { return a.quantity <=> b.quantity; })->venue == "XPAR", "min_by: the emptiest");
    check(iter::nth(fills, 2)->venue == "XAMS" && !iter::nth(fills, 4).has_value(), "nth: by position, or none");
    check(iter::last(fills)->venue == "XETR" && !iter::last(std::vector<fill>{}).has_value(), "last: the last, or none");

    std::size_t weighted = 0;
    for (auto [i, f] : iter::enumerate(fills)) weighted += i * static_cast<std::size_t>(f.quantity);
    check(weighted == 0 * 300 + 1 * 0 + 2 * 200 + 3 * 100, "enumerate: index and element");

    const auto all_filled = iter::try_for_each(fills, [](const fill& f) -> snicholls::result<void, std::string_view> {
        if (f.quantity == 0) return snicholls::err(f.venue);
        return snicholls::ok();
    });
    check(all_filled.is_err() && all_filled.unwrap_err_unchecked() == "XPAR", "try_for_each: stops at the empty fill");
    const auto every_priced = iter::try_for_each(fills, [](const fill& f) {
        return f.price > 0 ? std::optional<std::monostate>(std::monostate{}) : std::nullopt;
    });
    check(every_priced.has_value(), "try_for_each over optionals: every fill is priced");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
