//
//  typeclasses_map_reduce_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/typeclasses.hpp>: map-reduce with monoids. Each record is mapped to a
//  summary, summaries are combined with <>, and because <> is associative the work can be split into chunks,
//  reduced separately (by threads, by machines) and merged, with the same answer as one pass. Count, sum, minimum,
//  maximum, first and last in one pass; word counts as an array of counters. All by the compiler. Exits non-zero if a
//  result is not the one documented.
//

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/functional/typeclasses.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

namespace fn = snicholls::fn;

//! [summary]
// One trade price mapped into six monoids at once: a tuple of monoids is a monoid.
using summary = std::tuple<fn::sum_of<int>, fn::sum_of<long>, fn::min_of<long>, fn::max_of<long>, fn::first_of<long>, fn::last_of<long>>;
constexpr summary one(long px) { return {{1}, {px}, {px}, {px}, {px}, {px}}; }

constexpr std::array<long, 8> ticks{101, 99, 104, 100, 98, 107, 103, 102};
constexpr summary all = fn::fold_map(one, ticks);
//! [summary]

//! [chunks]
// Split into chunks, reduce each alone, merge the partial results: associativity makes them equal.
constexpr summary reduce(std::span<const long> part)
{
    summary acc = fn::mempty<summary>();
    for (long px : part) acc = fn::combine(acc, one(px));
    return acc;
}
constexpr std::span<const long> whole{ticks};
constexpr summary merged = fn::sconcat(reduce(whole.subspan(0, 3)), reduce(whole.subspan(3, 2)), reduce(whole.subspan(5)));
//! [chunks]

//! [words]
// Word counts for a fixed vocabulary: each word maps to a one-hot array of counters, and arrays of monoids combine
// element by element.
constexpr std::array<std::string_view, 3> vocabulary{"buy", "sell", "hold"};
using counts = std::array<fn::sum_of<int>, 3>;

constexpr counts one_hot(std::string_view word)
{
    counts c{};
    for (std::size_t i = 0; i < vocabulary.size(); ++i)
        if (vocabulary[i] == word) c[i] = {1};
    return c;
}

// Words split at spaces, by hand: string_view::find is not used, so this runs in any compiler's constant evaluator.
constexpr counts count_words(std::string_view text)
{
    counts acc = fn::mempty<counts>();
    std::size_t start = 0;
    for (std::size_t i = 0; i <= text.size(); ++i)
        if (i == text.size() || text[i] == ' ') {
            if (i > start) acc = fn::combine(acc, one_hot(text.substr(start, i - start)));
            start = i + 1;
        }
    return acc;
}
//! [words]
} // namespace

int main()
{
    std::printf("snicholls-core functional/typeclasses map-reduce demo\n");

    static_assert(std::get<0>(all).value == 8 && std::get<1>(all).value == 814);
    static_assert(std::get<2>(all).value == 98 && std::get<3>(all).value == 107);
    static_assert(std::get<4>(all).value == std::optional<long>(101) && std::get<5>(all).value == std::optional<long>(102));
    check(std::get<1>(all).value / std::get<0>(all).value == 101, "the mean of eight ticks is 101 (whole cents)");

    static_assert(merged == all);
    check(merged == all, "three chunks reduced apart and merged equal one pass");
    check(fn::mconcat(std::array{one(5), one(3)}) == fn::combine(one(5), one(3)), "mconcat is <> over the list");
    check(fn::fold_map(one, std::array<long, 0>{}) == fn::mempty<summary>(), "no ticks: the identity, count 0");

    static_assert(count_words("buy hold buy sell buy") == counts{{{3}, {1}, {1}}});
    check(count_words("") == fn::mempty<counts>(), "no words, no counts");
    check(count_words("  hold   short hold ") == counts{{{0}, {0}, {2}}}, "extra spaces and unknown words are skipped");

    // Booleans as monoids: all and any.
    constexpr auto above_95 = fn::fold_map([](long px) { return fn::all_of{px > 95}; }, ticks);
    constexpr auto any_over_105 = fn::fold_map([](long px) { return fn::any_of{px > 105}; }, ticks);
    static_assert(above_95.value && any_over_105.value);
    check(above_95.value && any_over_105.value, "every tick above 95, some over 105");

    // Products and endomorphisms: compounding and composing.
    check(fn::mconcat(std::array{fn::product_of<int>{2}, fn::product_of<int>{3}, fn::product_of<int>{7}}).value == 42,
          "product_of: 2 * 3 * 7");
    constexpr auto bump = fn::combine(fn::endo{[](long x) { return x + 1; }}, fn::endo{[](long x) { return x * 10; }});
    check(bump(4) == 41, "endo: (+1) . (*10), applied right to left");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
