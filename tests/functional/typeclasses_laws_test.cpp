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
//    Applicative: pure id <*> v == v;  pure f <*> pure x == pure (f x);  u <*> pure y == pure ($ y) <*> u;
//                 pure (.) <*> u <*> v <*> w == u <*> (v <*> w)
//    Alternative: empty <|> x == x == x <|> empty;  (a <|> b) <|> c == a <|> (b <|> c)
//    Bifunctor:   bimap id id == id;  bimap (f . g) (h . i) == bimap f h . bimap g i;  first f == bimap f id
//    Profunctor:  dimap id id == id;  dimap (f . g) (h . i) == dimap g h . dimap f i
//    Foldable:    mconcat == foldr (<>) mempty
//

#include <snicholls/core/containers/inplace_vector.hpp>
#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/state.hpp>

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

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
static_assert(functor_laws(snicholls::containers::inplace_vector<int, 4>{1, 2, 3}) &&
              functor_laws(snicholls::containers::inplace_vector<int, 4>{}) && functor_laws(snicholls::containers::inplace_vector<int, 2>{5, 6}));
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
    return fn::combine(fn::mempty<T>(), x) == x && fn::combine(x, fn::mempty<T>()) == x &&
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
    EXPECT_EQ(fn::combine(fn::mempty<std::string>(), x), x);
}

// ---- Applicative ----------------------------------------------------------------------------------------------------

// The four laws for a carrier M of int, with function values u, v, w (carriers of int -> int) and a value y.
constexpr auto compose2 = [](auto a) { return [a](auto b) { return [a, b](int x) { return a(b(x)); }; }; };

template <class M, class U, class V, class W, class Eq>
constexpr bool applicative_laws(const M& v0, const U& u, const V& v, const W& w, int y, Eq equal)
{
    const bool identity = equal(fn::ap(fn::pure<M>(fn::identity), v0), v0);
    const bool homomorphism = equal(fn::ap(fn::pure<M>(f), fn::pure<M>(y)), fn::pure<M>(f(y)));
    const bool interchange = equal(fn::ap(u, fn::pure<M>(y)), fn::ap(fn::pure<M>([y](auto fy) { return fy(y); }), u));
    const bool composition = equal(fn::ap(fn::ap(fn::ap(fn::pure<M>(compose2), u), v), w), fn::ap(u, fn::ap(v, w)));
    return identity && homomorphism && interchange && composition;
}

constexpr std::optional<int (*)(int)> opt_f = +[](int x) { return x * 3; };
constexpr std::optional<int (*)(int)> opt_g = +[](int x) { return x - 2; };
static_assert(applicative_laws(std::optional<int>{5}, opt_f, opt_g, std::optional<int>{4}, 9, eq));
static_assert(applicative_laws(std::optional<int>{}, opt_f, std::optional<int (*)(int)>{}, std::optional<int>{4}, 9, eq));
static_assert(applicative_laws(std::optional<int>{1}, opt_f, opt_g, std::optional<int>{}, 0, eq));

using fquote = result<int (*)(int), failure>;
static_assert(applicative_laws(quote(ok(5)), fquote(ok(+[](int x) { return x * 3; })), fquote(ok(+[](int x) { return x - 2; })),
                               quote(ok(4)), 9, eq_res));
static_assert(applicative_laws(quote(err(failure::stale)), fquote(ok(+[](int x) { return x * 3; })),
                               fquote(err(failure::crossed)), quote(ok(4)), 9, eq_res));

// The array (list) applicative: every function to every value, the laws up to the order of the cartesian product.
constexpr std::array<int (*)(int), 2> arr_u{+[](int x) { return x + 1; }, +[](int x) { return x * 10; }};
constexpr std::array<int (*)(int), 2> arr_v{+[](int x) { return -x; }, +[](int x) { return x * x; }};
static_assert(fn::ap(fn::pure<std::array<int, 1>>(fn::identity), std::array{1, 2, 3}) == std::array{1, 2, 3});
static_assert(fn::ap(fn::pure<std::array<int, 1>>(f), fn::pure<std::array<int, 1>>(4)) == fn::pure<std::array<int, 1>>(f(4)));
static_assert(fn::ap(arr_u, fn::pure<std::array<int, 1>>(3)) ==
              fn::ap(fn::pure<std::array<int, 1>>([](int (*fy)(int)) { return fy(3); }), arr_u));
static_assert(fn::ap(fn::ap(fn::ap(fn::pure<std::array<int, 1>>(compose2), arr_u), arr_v), std::array{2, 5}) ==
              fn::ap(arr_u, fn::ap(arr_v, std::array{2, 5})));

// Identity and Reader are applicatives too.
static_assert(fn::ap(fn::pure<fn::identity_of<>>(fn::identity), fn::identity_of<int>{7}) == fn::identity_of<int>{7});
static_assert(fn::ap(fn::pure<fn::identity_of<>>(f), fn::pure<fn::identity_of<>>(2)) == fn::pure<fn::identity_of<>>(f(2)));
static_assert(fn::run_reader(fn::ap(fn::pure<fn::reader<>>(fn::identity), fn::ask), 6) == 6);
static_assert(fn::run_reader(fn::ap(fn::pure<fn::reader<>>(f), fn::pure<fn::reader<>>(2)), 0) == f(2));
static_assert(fn::run_state(fn::ap(fn::pure<fn::state<>>(fn::identity), fn::get_state), 3) == fn::run_state(fn::get_state, 3));

// lift_a2 is ap after fmap: liftA2 h a b == fmap h a <*> b.
constexpr auto plus_c = [](int a) { return [a](int b) { return a + b; }; };
static_assert(fn::lift_a2(std::plus<>{}, std::optional<int>{2}, std::optional<int>{5}) ==
              fn::ap(fn::fmap(plus_c, std::optional<int>{2}), std::optional<int>{5}));
static_assert(fn::lift_a2(std::plus<>{}, std::array{1, 2}, std::array{10, 20}) == fn::ap(fn::fmap(plus_c, std::array{1, 2}), std::array{10, 20}));

// ---- Kleisli: the monad laws as a category -------------------------------------------------------------------------

// pure >=> f == f == f >=> pure;  (f >=> g) >=> h == f >=> (g >=> h)
constexpr auto pure_opt = [](int x) { return fn::pure<std::optional<int>>(x); };
template <class K>
constexpr bool kleisli_laws_at(int x, K k)
{
    return fn::kleisli(pure_opt, k)(x) == k(x) && fn::kleisli(k, pure_opt)(x) == k(x) &&
           fn::kleisli(fn::kleisli(k, h_opt), k)(x) == fn::kleisli(k, fn::kleisli(h_opt, k))(x);
}
static_assert(kleisli_laws_at(8, k_opt) && kleisli_laws_at(3, k_opt) && kleisli_laws_at(0, k_opt));
static_assert(fn::kleisli_left(h_opt, k_opt)(8) == fn::kleisli(k_opt, h_opt)(8));   // <=< is >=> flipped

// ---- Alternative ----------------------------------------------------------------------------------------------------

template <class M>
constexpr bool alternative_laws(const M& a, const M& b, const M& c)
{
    return fn::alt(fn::mzero<M>, a) == a && fn::alt(a, fn::mzero<M>) == a && fn::alt(fn::alt(a, b), c) == fn::alt(a, fn::alt(b, c));
}
static_assert(alternative_laws(std::optional<int>{1}, std::optional<int>{2}, std::optional<int>{3}));
static_assert(alternative_laws(std::optional<int>{}, std::optional<int>{2}, std::optional<int>{}));
static_assert(alternative_laws(std::optional<int>{}, std::optional<int>{}, std::optional<int>{}));
// Left catch: pure a <|> b == pure a. And for result, associativity of the first success.
static_assert(fn::alt(fn::pure<std::optional<int>>(4), std::optional<int>{9}) == std::optional<int>{4});
static_assert(same(fn::alt(fn::alt(quote(err(failure::stale)), quote(err(failure::crossed))), quote(ok(3))),
                   fn::alt(quote(err(failure::stale)), fn::alt(quote(err(failure::crossed)), quote(ok(3))))));
static_assert(fn::alt(quote(err(failure::stale)), quote(err(failure::crossed))).unwrap_err_unchecked() == failure::crossed);

// ---- Bifunctor and Profunctor ---------------------------------------------------------------------------------------

constexpr auto neg = [](int x) { return -x; };
static_assert(fn::bimap(fn::identity, fn::identity, std::pair{1, 2}) == std::pair{1, 2});
static_assert(fn::bimap(fn::compose(f, g), fn::compose(neg, f), std::pair{1, 2}) ==
              fn::bimap(f, neg, fn::bimap(g, f, std::pair{1, 2})));
static_assert(fn::first(f, std::pair{1, 2}) == fn::bimap(f, fn::identity, std::pair{1, 2}));
static_assert(fn::second(f, std::pair{1, 2}) == fn::bimap(fn::identity, f, std::pair{1, 2}));
static_assert(fn::bimap(fn::identity, fn::identity, std::tuple{1, 2}) == std::tuple{1, 2});
static_assert(same(fn::bimap(fn::identity, fn::identity, quote(ok(3))), quote(ok(3))));
static_assert(fn::bimap(f, [](failure) { return 0; }, quote(ok(3))).unwrap_unchecked() == 9);

static_assert(fn::dimap(fn::identity, fn::identity, f)(4) == f(4));
static_assert(fn::dimap(fn::compose(f, g), fn::compose(neg, f), neg)(5) == fn::dimap(g, neg, fn::dimap(f, f, neg))(5));
static_assert(fn::lmap(g, f)(5) == fn::dimap(g, fn::identity, f)(5) && fn::rmap(g, f)(5) == fn::dimap(fn::identity, g, f)(5));

// ---- Endo, and mconcat as a fold --------------------------------------------------------------------------------------

constexpr auto inc = [](int x) { return x + 1; };
static_assert(fn::combine(fn::mempty<fn::endo<fn::detail::identity_fn>>(), fn::endo{f})(4) == f(4));
static_assert(fn::combine(fn::endo{f}, fn::mempty<fn::endo<fn::detail::identity_fn>>())(4) == f(4));
static_assert(fn::combine(fn::combine(fn::endo{f}, fn::endo{g}), fn::endo{inc})(4) ==
              fn::combine(fn::endo{f}, fn::combine(fn::endo{g}, fn::endo{inc}))(4));

template <class T, std::size_t N>
constexpr bool mconcat_is_a_fold(const std::array<T, N>& xs)
{
    T acc = fn::mempty<T>();
    for (std::size_t i = N; i-- > 0;) acc = fn::combine(xs[i], acc);
    return fn::mconcat(xs) == acc;
}
static_assert(mconcat_is_a_fold(std::array{fn::sum_of<int>{1}, fn::sum_of<int>{2}, fn::sum_of<int>{3}}));
static_assert(mconcat_is_a_fold(std::array{fn::first_of<int>{}, fn::first_of<int>{5}, fn::first_of<int>{6}}));
static_assert(mconcat_is_a_fold(std::array{fn::min_of<int>{3}, fn::min_of<int>{-4}}));
static_assert(mconcat_is_a_fold(std::array<fn::any_of, 0>{}));

TEST(FunctionalTypeclassLaws, FunctorAndMonoidLawsHoldForVectors)
{
    const std::vector<int> v{1, 2, 3};
    EXPECT_EQ(fn::fmap(fn::identity, v), v);
    EXPECT_EQ(fn::fmap(fn::compose(f, g), v), fn::fmap(f, fn::fmap(g, v)));
    const std::vector<int> a{1}, b{2, 3}, c{};
    EXPECT_EQ(fn::combine(fn::combine(a, b), c), fn::combine(a, fn::combine(b, c)));
    EXPECT_EQ(fn::combine(fn::mempty<std::vector<int>>(), b), b);
}

TEST(FunctionalTypeclassLaws, MonadLawsHoldForAResultWithAStringError)
{
    using parsed = result<int, std::string>;
    const auto k = [](int x) { return x > 0 ? parsed(ok(x - 1)) : parsed(err(std::string("negative"))); };
    const auto h = [](int x) { return parsed(ok(x * 2)); };
    for (const parsed& m : {parsed(ok(3)), parsed(ok(0)), parsed(err(std::string("bad")))}) {
        const auto lhs = fn::mbind(fn::mbind(m, k), h);
        const auto rhs = fn::mbind(m, [&](int x) { return fn::mbind(k(x), h); });
        ASSERT_EQ(lhs.is_ok(), rhs.is_ok());
        if (lhs.is_ok()) EXPECT_EQ(lhs.unwrap_unchecked(), rhs.unwrap_unchecked());
        else EXPECT_EQ(lhs.unwrap_err_unchecked(), rhs.unwrap_err_unchecked());
    }
}

} // namespace
