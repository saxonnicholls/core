//
//  typeclasses_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/typeclasses.hpp>: monoids that summarise a book, and a type of the demo's own
//  joining the Functor class. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/functional/typeclasses.hpp>

#include <array>
#include <cstdio>
#include <optional>
#include <tuple>
#include <type_traits>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

//! [instance]
// A price with the venue it came from: a Functor maps the price and keeps the venue.
template <class T>
struct tagged {
    T value;
    char venue;
};
} // namespace

template <class T>
struct snicholls::fn::functor_instance<tagged<T>> {
    static constexpr bool defined = true;
    template <class F, class M>
    static constexpr auto fmap(F&& f, M&& m)
    {
        return tagged<std::remove_cvref_t<std::invoke_result_t<F, T>>>{std::invoke(f, m.value), m.venue};
    }
};
//! [instance]

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/typeclasses demo\n");

    constexpr auto in_bp = [](double r) { return r * 10'000.0; };
    constexpr auto t = fn::fmap(in_bp, tagged<double>{0.0002, 'L'});
    static_assert(t.venue == 'L');
    check(t.value > 1.99 && t.value < 2.01, "fmap over a type of the demo's own");

    //! [monoid]
    // One pass over the fills, three summaries at once: a tuple of monoids is a monoid.
    constexpr std::array fills{120.0, -40.0, 15.0, 60.0};
    constexpr auto summary = fn::fold_map([](double pnl) {
        return std::tuple{fn::sum_of<double>{pnl}, fn::max_of<double>{pnl}, fn::all_of{pnl > -50.0}};
    }, fills);
    static_assert(std::get<0>(summary).value == 155.0);
    static_assert(std::get<1>(summary).value == 120.0);
    static_assert(std::get<2>(summary).value);
    //! [monoid]
    check(std::get<0>(summary).value == 155.0, "the sum, the best fill and the risk check in one fold");

    // Applicative and Monad over Maybe; Alternative picks the first quote present.
    constexpr std::optional<double> primary{}, backup = 1.0851;
    check(fn::alt(primary, backup) == backup, "alt: the first venue that quoted");
    check(fn::lift_a2([](double a, double b) { return b - a; }, std::optional{1.0}, std::optional{1.5}) == std::optional{0.5},
          "lift_a2 over two present values");
    check(fn::dimap([](int lots) { return lots * 100'000.0; }, [](double n) { return n / 1e6; }, [](double n) { return n * 1.0851; })(2) > 0.217,
          "dimap: adapt a function's input and output");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
