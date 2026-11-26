//
//  rust_showcase_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/rust.hpp> with <snicholls/core/functional/rust_iter.hpp>: a fills file read
//  the way Rust reads one. Each line is split into fields that may be missing (Option), parsed into a fill or the first
//  reason it is not one (Result, chained with and_then), validated in one pass with try_for_each, searched with
//  find_map, summarised with max_by_key and a moving VWAP over windows, sorted by a multi-key comparator, and quoted
//  with zip_with. Run by the compiler, then checked at run time. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/functional/monad.hpp>
#include <snicholls/core/functional/rust.hpp>
#include <snicholls/core/functional/rust_iter.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <span>
#include <string_view>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

namespace fn = snicholls::fn;
namespace option = snicholls::fn::rust::option;
namespace slice = snicholls::fn::rust::slice;
namespace iter = snicholls::fn::rust::iter;
namespace cmp = snicholls::fn::rust::cmp;

//! [parse]
enum class bad_line { missing_field, not_a_number, zero_quantity };

struct fill {
    std::string_view venue;
    long price;   // in cents
    long qty;
};

// The i-th space-separated field, if there is one: Option<&str>.
constexpr std::optional<std::string_view> field(std::string_view line, std::size_t i)
{
    std::size_t start = 0, k = 0;
    for (std::size_t j = 0; j <= line.size(); ++j)
        if (j == line.size() || line[j] == ' ') {
            if (k++ == i) return line.substr(start, j - start);
            start = j + 1;
        }
    return std::nullopt;
}

constexpr snicholls::result<long, bad_line> number(std::string_view s)
{
    long n = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return snicholls::err(bad_line::not_a_number);
        n = n * 10 + (c - '0');
    }
    return snicholls::ok(n);
}

// let venue = field(0).ok_or(Missing)?; let px = field(1).ok_or(Missing).and_then(number)?; ...
constexpr snicholls::result<fill, bad_line> parse(std::string_view line)
{
    const auto part = [line](std::size_t i) { return option::ok_or(field(line, i), bad_line::missing_field); };
    return part(0).and_then([&](std::string_view venue) {
        return part(1).and_then(number).and_then([&](long px) {
            return part(2).and_then(number).and_then([&](long qty) -> snicholls::result<fill, bad_line> {
                if (qty == 0) return snicholls::err(bad_line::zero_quantity);
                return snicholls::ok(fill{venue, px, qty});
            });
        });
    });
}
//! [parse]

constexpr std::array<std::string_view, 6> file{"XLON 10025 300", "XPAR 10020 100", "XAMS 10030 500",
                                               "XLON 10010 200", "XETR 10040 400", "XPAR 10035 100"};
constexpr std::array<std::string_view, 3> broken{"XLON 10025 300", "XPAR 10x20 100", "XAMS 10030"};

// All lines parsed, or the first bad one: collect::<Result<Vec<_>, _>>() is fn::traverse.
constexpr auto fills = fn::traverse(parse, file).unwrap_unchecked();

constexpr long vwap(std::span<const fill> fs)
{
    long notional = 0, qty = 0;
    for (const fill& f : fs) notional += f.price * f.qty, qty += f.qty;
    return notional / qty;
}
} // namespace

int main()
{
    std::printf("snicholls-core functional/rust showcase demo\n");

    //! [pipeline]
    static_assert(parse("XLON 10025 300").unwrap_unchecked().qty == 300);
    static_assert(parse("XLON 10025").unwrap_err_unchecked() == bad_line::missing_field);
    static_assert(parse("XLON 1x025 300").unwrap_err_unchecked() == bad_line::not_a_number);
    static_assert(parse("XLON 10025 0").unwrap_err_unchecked() == bad_line::zero_quantity);
    static_assert(fn::traverse(parse, broken).unwrap_err_unchecked() == bad_line::not_a_number);   // the first bad line

    // Validate everything in one pass, stopping at the first failure: try_for_each.
    constexpr auto all_parse = iter::try_for_each(file, [](std::string_view l) {
        return parse(l).and_then([](const fill&) -> snicholls::result<void, bad_line> { return snicholls::ok(); });
    });
    static_assert(all_parse.is_ok());

    // The first Paris fill's price: find_map.
    constexpr auto paris =
        iter::find_map(fills, [](const fill& f) { return f.venue == "XPAR" ? std::optional<long>(f.price) : std::nullopt; });
    static_assert(paris == std::optional<long>(10020));

    // The largest fill (the last of equals, as Rust) and a moving VWAP over three fills.
    static_assert(iter::max_by_key(fills, [](const fill& f) { return f.qty; })->venue == "XAMS");
    static_assert(vwap(slice::windows(fills, 3)[0]) == (10025 * 300 + 10020 * 100 + 10030 * 500) / 900);
    //! [pipeline]
    check(paris == std::optional<long>(10020), "find_map: the first Paris fill at 100.20");
    std::size_t windows = 0;
    for (std::span<const fill> w : slice::windows(fills, 3)) windows += vwap(w) > 10000 ? 1 : 0;
    check(windows == 4, "four three-fill windows, each with a VWAP above 100.00");

    //! [quote]
    // Best bid and best ask may be missing; the spread exists only when both do: zip_with.
    constexpr std::optional<long> bid = 10020, ask = 10035, none;
    static_assert(option::zip_with(bid, ask, [](long b, long a) { return a - b; }) == std::optional<long>(15));
    static_assert(option::zip_with(bid, none, [](long b, long a) { return a - b; }) == std::nullopt);
    static_assert(option::map_or(option::filter(ask, [](long a) { return a < 10030; }), -1L, [](long a) { return a; }) == -1);
    //! [quote]
    check(option::xor_(bid, none) == bid && option::unwrap_or_default(none) == 0, "xor_ and unwrap_or_default");

    //! [sort]
    // By venue, then the dearest first, then the largest: one comparator from three keys.
    std::array<fill, 6> book = fills;
    std::ranges::sort(book, cmp::as_less(cmp::lexicographic(cmp::comparing(&fill::venue), cmp::reversed(cmp::comparing(&fill::price)),
                                                            cmp::reversed(cmp::comparing(&fill::qty)))));
    //! [sort]
    check(book[0].venue == "XAMS" && book[1].venue == "XETR" && book[2].price == 10025 && book[3].price == 10010 &&
              book[4].price == 10035 && book[5].price == 10020,
          "sorted by venue, then price descending");
    std::size_t i = 0;
    bool ordered = true;
    for (auto [k, f] : iter::enumerate(book)) ordered = ordered && k == i++ && !f.venue.empty();
    check(ordered, "enumerate walks the sorted book with its index");
    long every_other = 0;
    for (const fill& f : iter::step_by(book, 2)) every_other += f.qty;
    check(every_other == 500 + 300 + 100, "step_by(2): every other fill");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
