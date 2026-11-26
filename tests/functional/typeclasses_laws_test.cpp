//
//  typeclasses_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The type-class laws, at compile time, over sample values of every instance core defines
//  (<snicholls/core/functional/typeclasses.hpp>, instances.hpp, state.hpp):
//    Functor:  fmap id == id;  fmap (f . g) == fmap f . fmap g
//    Monad:    return a >>= k == k a;  m >>= return == m;  (m >>= k) >>= h == m >>= (\x -> k x >>= h)
//    Monoid:   mempty <> x == x == x <> mempty;  (x <> y) <> z == x <> (y <> z)
//

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/state.hpp>

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace {

namespace fn = snicholls::fn;
using snicholls::err;
using snicholls::ok;
using snicholls::result;

enum class failure { stale, crossed };
using quote = result<int, failure>;

constexpr auto f = [](int x) { return x * 3; };
constexpr auto g = [](int x) { return x + 7; };

// ---- Functor ----------------------------------------------------------------------------------------------------------

template <class M>
constexpr bool functor_laws(const M& m)
{
    return fn::fmap(fn::identity, m) == m && fn::fmap(fn::compose(f, g), m) == fn::fmap(f, fn::fmap(g, m));
}

static_assert(functor_laws(std::optional<int>{2}) && functor_laws(std::optional<int>{}));
static_assert(functor_laws(std::array{1, 2, 3}) && functor_laws(std::array<int, 0>{}));
static_assert(functor_laws(std::tuple{1, 2}) && functor_laws(std::pair{1, 2}));
static_assert(functor_laws(fn::identity_of<int>{4}) && functor_laws(fn::const_of<int>{4}));
static_assert(functor_laws(fn::writer<fn::sum_of<int>, int>{4, {1}}));
// result has no operator== of its own here: compare by side.
constexpr bool same(const quote& a, const quote& b)
{
    return a.is_ok() == b.is_ok() && (a.is_ok() ? a.unwrap_unchecked() == b.unwrap_unchecked()
                                                : a.unwrap_err_unchecked() == b.unwrap_err_unchecked());
}
static_assert(same(fn::fmap(fn::identity, quote(ok(2))), quote(ok(2))));
static_assert(same(fn::fmap(fn::compose(f, g), quote(ok(2))), fn::fmap(f, fn::fmap(g, quote(ok(2))))));
static_assert(same(fn::fmap(fn::compose(f, g), quote(err(failure::stale))), quote(err(failure::stale))));
// Reader and State are functions: compare what they compute.
static_assert(fn::run_reader(fn::fmap(fn::identity, fn::ask), 5) == fn::run_reader(fn::ask, 5));
static_assert(fn::run_reader(fn::fmap(fn::compose(f, g), fn::ask), 5) == fn::run_reader(fn::fmap(f, fn::fmap(g, fn::ask)), 5));
static_assert(fn::run_state(fn::fmap(fn::identity, fn::get_state), 5) == fn::run_state(fn::get_state, 5));
static_assert(fn::run_state(fn::fmap(fn::compose(f, g), fn::get_state), 5) ==
              fn::run_state(fn::fmap(f, fn::fmap(g, fn::get_state)), 5));

// ---- Monad --------------------------------------------------------------------------------------------------------------

constexpr auto k_opt = [](int x) { return x % 2 == 0 ? std::optional<int>{x / 2} : std::nullopt; };
constexpr auto h_opt = [](int x) { return std::optional<int>{x + 1}; };
template <class M, class K, class H, class Eq>
constexpr bool monad_laws(const M& m, int a, K k, H h, Eq eq)
{
    const bool left = eq(fn::mbind(fn::pure<M>(a), k), k(a));
    const bool right = eq(fn::mbind(m, [](int x) { return fn::pure<M>(x); }), m);
    const bool assoc = eq(fn::mbind(fn::mbind(m, k), h), fn::mbind(m, [&](int x) { return fn::mbind(k(x), h); }));
    return left && right && assoc;
}
constexpr auto eq = [](const auto& a, const auto& b) { return a == b; };
static_assert(monad_laws(std::optional<int>{4}, 4, k_opt, h_opt, eq));
static_assert(monad_laws(std::optional<int>{3}, 3, k_opt, h_opt, eq));
static_assert(monad_laws(std::optional<int>{}, 2, k_opt, h_opt, eq));

constexpr auto k_res = [](int x) { return x > 0 ? quote(ok(x - 1)) : quote(err(failure::crossed)); };
constexpr auto h_res = [](int x) { return quote(ok(x * 10)); };
constexpr auto eq_res = [](const quote& a, const quote& b) { return same(a, b); };
static_assert(monad_laws(quote(ok(3)), 3, k_res, h_res, eq_res));
static_assert(monad_laws(quote(ok(0)), 0, k_res, h_res, eq_res));
static_assert(monad_laws(quote(err(failure::stale)), 1, k_res, h_res, eq_res));

constexpr auto k_id = [](int x) { return fn::identity_of<int>{x * 2}; };
constexpr auto h_id = [](int x) { return fn::identity_of<int>{x - 1}; };
static_assert(monad_laws(fn::identity_of<int>{5}, 5, k_id, h_id, eq));

using log_t = fn::sum_of<int>;
constexpr auto k_wr = [](int x) { return fn::writer<log_t, int>{x * 2, {1}}; };
constexpr auto h_wr = [](int x) { return fn::writer<log_t, int>{x + 3, {10}}; };
static_assert(monad_laws(fn::writer<log_t, int>{5, {100}}, 5, k_wr, h_wr, eq));

// The array (list) monad: k doubles each element into two, h into one.
constexpr auto k_arr = [](int x) { return std::array{x, -x}; };
constexpr auto h_arr = [](int x) { return std::array{x * 10}; };
static_assert(fn::mbind(fn::pure<std::array<int, 1>>(3), k_arr) == k_arr(3));                       // left identity
static_assert(fn::mbind(std::array{1, 2}, [](int x) { return fn::pure<std::array<int, 1>>(x); }) == std::array{1, 2});
static_assert(fn::mbind(fn::mbind(std::array{1, 2}, k_arr), h_arr) ==
              fn::mbind(std::array{1, 2}, [](int x) { return fn::mbind(k_arr(x), h_arr); }));     // associativity

// State: the laws as equal runs from a sample state.
constexpr auto k_st = [](int x) { return fn::mthen(fn::modify([x](int s) { return s + x; }), fn::pure<fn::state<>>(x * 2)); };
constexpr auto h_st = [](int x) { return fn::fmap([x](int s) { return s * x; }, fn::get_state); };
static_assert(fn::run_state(fn::mbind(fn::pure<fn::state<>>(3), k_st), 1) == fn::run_state(k_st(3), 1));
static_assert(fn::run_state(fn::mbind(fn::get_state, [](int x) { return fn::pure<fn::state<>>(x); }), 4) ==
              fn::run_state(fn::get_state, 4));
static_assert(fn::run_state(fn::mbind(fn::mbind(fn::get_state, k_st), h_st), 2) ==
              fn::run_state(fn::mbind(fn::get_state, [](int x) { return fn::mbind(k_st(x), h_st); }), 2));

// Reader.
constexpr auto k_rd = [](int x) { return fn::asks([x](int e) { return e * x; }); };
constexpr auto h_rd = [](int x) { return fn::asks([x](int e) { return e + x; }); };
static_assert(fn::run_reader(fn::mbind(fn::pure<fn::reader<>>(3), k_rd), 5) == fn::run_reader(k_rd(3), 5));
static_assert(fn::run_reader(fn::mbind(fn::ask, [](int x) { return fn::pure<fn::reader<>>(x); }), 5) == 5);
static_assert(fn::run_reader(fn::mbind(fn::mbind(fn::ask, k_rd), h_rd), 5) ==
              fn::run_reader(fn::mbind(fn::ask, [](int x) { return fn::mbind(k_rd(x), h_rd); }), 5));

// ---- Monoid -------------------------------------------------------------------------------------------------------------

template <class T>
constexpr bool monoid_laws(const T& x, const T& y, const T& z)
{
    return fn::combine(fn::mempty<T>, x) == x && fn::combine(x, fn::mempty<T>) == x &&
           fn::combine(fn::combine(x, y), z) == fn::combine(x, fn::combine(y, z));
}

static_assert(monoid_laws(fn::sum_of<int>{1}, fn::sum_of<int>{2}, fn::sum_of<int>{3}));
static_assert(monoid_laws(fn::product_of<int>{2}, fn::product_of<int>{3}, fn::product_of<int>{4}));
static_assert(monoid_laws(fn::min_of<int>{2}, fn::min_of<int>{-1}, fn::min_of<int>{7}));
static_assert(monoid_laws(fn::max_of<double>{2}, fn::max_of<double>{-1}, fn::max_of<double>{7}));
static_assert(monoid_laws(fn::first_of<int>{}, fn::first_of<int>{2}, fn::first_of<int>{3}));
static_assert(monoid_laws(fn::last_of<int>{1}, fn::last_of<int>{}, fn::last_of<int>{3}));
static_assert(monoid_laws(fn::all_of{true}, fn::all_of{false}, fn::all_of{true}));
static_assert(monoid_laws(fn::any_of{false}, fn::any_of{true}, fn::any_of{false}));
static_assert(monoid_laws(std::optional<fn::sum_of<int>>{{1}}, std::optional<fn::sum_of<int>>{}, std::optional<fn::sum_of<int>>{{3}}));
static_assert(monoid_laws(std::tuple{fn::sum_of<int>{1}, fn::all_of{true}}, std::tuple{fn::sum_of<int>{2}, fn::all_of{false}},
                          std::tuple{fn::sum_of<int>{3}, fn::all_of{true}}));
static_assert(monoid_laws(std::pair{fn::sum_of<int>{1}, fn::max_of<int>{4}}, std::pair{fn::sum_of<int>{2}, fn::max_of<int>{9}},
                          std::pair{fn::sum_of<int>{3}, fn::max_of<int>{1}}));
static_assert(monoid_laws(std::array{fn::sum_of<int>{1}, fn::sum_of<int>{2}}, std::array{fn::sum_of<int>{3}, fn::sum_of<int>{4}},
                          std::array{fn::sum_of<int>{5}, fn::sum_of<int>{6}}));
static_assert(monoid_laws(std::string("a"), std::string("bc"), std::string("d")));
// The semigroup laws for result (no mempty): associativity.
static_assert(same(fn::combine(fn::combine(quote(err(failure::stale)), quote(ok(1))), quote(ok(2))),
                   fn::combine(quote(err(failure::stale)), fn::combine(quote(ok(1)), quote(ok(2))))));

TEST(FunctionalTypeclassLaws, HoldAtRunTimeForStrings)
{
    const std::string x = "EUR", y = "/", z = "USD";
    EXPECT_EQ(fn::combine(fn::combine(x, y), z), fn::combine(x, fn::combine(y, z)));
    EXPECT_EQ(fn::combine(fn::mempty<std::string>, x), x);
}

} // namespace
