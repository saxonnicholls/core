//
//  rust_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/rust.hpp>: Rust's Option methods over std::optional, std::cmp's ordering
//  helpers, and Into. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/rust.hpp>

#include <algorithm>
#include <array>
#include <compare>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

enum class quote_error { stale, missing };

struct quote {
    double bid;
    double ask;
    int age_ms;
};

struct order {
    int priority;
    double price;
    int id;
};
} // namespace

int main()
{
    namespace option = snicholls::fn::rust::option;
    namespace cmp = snicholls::fn::rust::cmp;
    std::printf("snicholls-core functional/rust demo\n");

    //! [option]
    // A quote that may be missing, checked for freshness, turned into a mid price or an error.
    const std::optional<quote> q = quote{99.5, 100.5, 40};
    const auto fresh = option::filter(q, [](const quote& x) { return x.age_ms < 250; });
    const snicholls::result<quote, quote_error> checked = option::ok_or(fresh, quote_error::stale);
    const double mid = option::map_or(fresh, 0.0, [](const quote& x) { return (x.bid + x.ask) / 2; });
    //! [option]
    check(checked.is_ok() && mid == 100.0, "a fresh quote gives its mid, 100");
    check(option::ok_or_else(std::optional<quote>(), [] { return quote_error::missing; }).unwrap_err_unchecked() ==
              quote_error::missing,
          "ok_or_else: no quote is the error the function makes");
    check(option::map_or_else(std::optional<quote>(), [] { return -1.0; }, [](const quote& x) { return x.bid; }) == -1.0,
          "map_or_else: the fallback runs only when empty");

    check(option::is_some_and(q, [](const quote& x) { return x.ask > x.bid; }), "is_some_and: the quote is not crossed");
    check(option::is_none_or(std::optional<quote>(), [](const quote&) { return false; }), "is_none_or: true for none");

    int logged = 0;
    option::inspect(q, [&](const quote& x) { logged = x.age_ms; });
    check(logged == 40, "inspect sees the value and passes the optional on");

    const std::optional<int> a = 2, b = 3, none;
    check(option::and_(a, b) == b && option::and_(none, b) == none, "and_: the second if the first holds");
    check(option::or_(none, b) == b && option::or_(a, b) == a, "or_: the first that holds");
    check(option::xor_(a, none) == a && option::xor_(a, b) == none, "xor_: exactly one");
    check(option::or_else(none, [] { return std::optional<int>(9); }) == std::optional<int>(9), "or_else: computed alternative");
    check(option::unwrap_or_else(none, [] { return 7; }) == 7 && option::unwrap_or_default(none) == 0, "unwrap_or_else, unwrap_or_default");
    check(option::zip(a, b) == std::optional<std::pair<int, int>>({2, 3}), "zip: both, as a pair");
    check(option::zip_with(a, b, std::multiplies<>{}) == std::optional<int>(6), "zip_with: both, combined");
    check(option::unzip(option::zip(a, b)) == std::pair{a, b}, "unzip: a pair of optionals");

    const std::optional<snicholls::result<int, quote_error>> parsed = snicholls::result<int, quote_error>(snicholls::ok(5));
    check(option::transpose(parsed).unwrap_or(none) == std::optional<int>(5), "transpose: optional<result> to result<optional>");

    std::optional<std::string> slot;
    check(option::get_or_insert_with(slot, [] { return std::string("default"); }) == "default", "get_or_insert_with fills an empty slot");
    check(option::get_or_insert(slot, "other") == "default", "get_or_insert keeps what is there");
    check(option::replace(slot, "new") == std::optional<std::string>("default") && slot == "new", "replace returns the old value");
    check(option::take_if(slot, [](std::string& s) { return s.size() > 5; }) == std::nullopt && slot == "new", "take_if leaves it when p fails");
    check(option::take(slot) == std::optional<std::string>("new") && !slot.has_value(), "take empties the slot");
    option::insert(slot, "again") += "!";
    check(slot == "again!", "insert returns a reference to the new value");

    //! [cmp]
    // Highest priority first, then cheapest, then by id: Rust's a.cmp(b).then_with(...), as one comparator.
    std::array<order, 4> book{{{1, 101.0, 7}, {2, 100.5, 3}, {2, 100.0, 9}, {1, 101.0, 2}}};
    const auto best_first = cmp::lexicographic(cmp::reversed(cmp::comparing(&order::priority)), cmp::comparing(&order::price),
                                               cmp::comparing(&order::id));
    std::ranges::sort(book, cmp::as_less(best_first));
    //! [cmp]
    check(book[0].id == 9 && book[1].id == 3 && book[2].id == 2 && book[3].id == 7, "the book sorts best first");

    const auto by_hand = cmp::then_with(1 <=> 1, [] { return cmp::then(2 <=> 2, 3 <=> 4); });
    check(by_hand == std::strong_ordering::less && cmp::reverse(by_hand) == std::strong_ordering::greater,
          "then, then_with and reverse on orderings");

    //! [into]
    // into<T> converts as Rust's Into does: implicitly and never narrowing.
    const double widened = snicholls::fn::rust::into<double>(2.5f);
    const std::string name = snicholls::fn::rust::into<std::string>("EURUSD");
    //! [into]
    check(widened == 2.5 && name == "EURUSD", "into<T>: float to double, a literal to std::string");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
