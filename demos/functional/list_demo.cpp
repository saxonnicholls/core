//
//  list_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/list.hpp>: Data.List on fixed arrays at compile time: a book of fills sorted, grouped, split and summarised. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/list.hpp>

#include <array>
#include <cstdio>
#include <optional>
#include <utility>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/list demo\n");

    //! [list]
    constexpr std::array quantities{500, -200, 300, -100, 700};
    // Split the book into buys and sells, keep their order.
    constexpr auto split = fn::partition([](int q) { return q > 0; }, quantities);
    static_assert(split.first.size() == 3 && split.second.size() == 2);
    // Running position, and the largest fills first.
    static_assert(fn::scanl([](int p, int q) { return p + q; }, 0, quantities) == std::array{0, 500, 300, 600, 500, 1200});
    static_assert(fn::sort_on([](int q) { return -(q < 0 ? -q : q); }, quantities) == std::array{700, 500, 300, -200, -100});
    // Ladders: zip prices with sizes, the first three levels.
    constexpr auto ladder = fn::take<3>(fn::zip(std::array{1.0850, 1.0849, 1.0848, 1.0847}, std::array{5, 10, 20, 50}));
    static_assert(ladder[2] == std::pair{1.0848, 20});
    // A matrix of exposures, transposed from desks-by-currency to currency-by-desks.
    constexpr auto by_ccy = fn::transpose(std::array{std::array{1, 2, 3}, std::array{4, 5, 6}});
    static_assert(by_ccy[2] == std::array{3, 6});
    //! [list]
    check(fn::sum(quantities) == 1200, "the net position");

    //! [search]
    constexpr std::array venues{std::pair{'L', 1.0851}, std::pair{'E', 1.0852}, std::pair{'C', 1.0850}};
    static_assert(fn::lookup('E', venues) == std::optional{1.0852});
    static_assert(fn::find_index([](const auto& v) { return v.second < 1.0851; }, venues) == std::optional<std::size_t>{2});
    static_assert(fn::all([](const auto& v) { return v.second > 1.08; }, venues));
    //! [search]
    check(!fn::lookup('X', venues).has_value(), "no such venue");
    check(fn::group(fn::sort(std::array{3, 1, 3, 2, 1})).size() == 3, "three distinct sizes");
    check(fn::subsequences(std::array{1, 2, 3}).size() == 8 && fn::permutations(std::array{1, 2, 3}).size() == 6,
          "subsets and orderings");
    check(fn::concat(std::array{std::array{1, 2}, std::array{3, 4}}) == std::array{1, 2, 3, 4}, "concat");
    check(fn::intercalate(std::array{0}, std::array{std::array{1}, std::array{2}}) == std::array{1, 0, 2}, "intercalate");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
