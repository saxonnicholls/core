//
//  state_laws_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The laws of the operations of <snicholls/core/functional/state.hpp>, at compile time:
//    State:  put s >> get == put s >> return s;  get >>= put == return ();  put a >> put b == put b;
//            get >>= \s -> get >>= k s == get >>= \s -> k s s;  modify f == get >>= put . f;  gets f == fmap f get
//    Reader: local id m == m;  local f . local g == local (g . f);  asks f == fmap f ask
//    Writer: tell a >> tell b == tell (a <> b);  tell mempty == return ()
//    Const:  ap accumulates in the monoid;  fmap leaves the constant alone
//  and the run helpers' agreement: eval == fst . run, exec == snd . run.
//

#include <snicholls/core/functional/instances.hpp>
#include <snicholls/core/functional/state.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace {

namespace fn = snicholls::fn;
using unit = std::monostate;

// ---- State ---------------------------------------------------------------------------------------------------------

template <class A, class B>
constexpr bool same_run(const A& a, const B& b, int s)
{
    return fn::run_state(a, s) == fn::run_state(b, s);
}

static_assert(same_run(fn::mthen(fn::put_state(7), fn::get_state), fn::mthen(fn::put_state(7), fn::pure<fn::state<>>(7)), 1));
static_assert(fn::run_state(fn::mbind(fn::get_state, fn::put_state), 4) == std::pair{unit{}, 4});
static_assert(same_run(fn::mthen(fn::put_state(1), fn::put_state(2)), fn::put_state(2), 9));
constexpr auto k2 = [](int a, int b) { return fn::pure<fn::state<>>(a * 10 + b); };
static_assert(same_run(fn::mbind(fn::get_state, [](int a) { return fn::mbind(fn::get_state, [a](int b) { return k2(a, b); }); }),
                       fn::mbind(fn::get_state, [](int a) { return k2(a, a); }), 3));
constexpr auto twice = [](int s) { return s * 2; };
static_assert(same_run(fn::modify(twice), fn::mbind(fn::get_state, [](int s) { return fn::put_state(twice(s)); }), 5));
static_assert(same_run(fn::gets(twice), fn::fmap(twice, fn::get_state), 5));

// The run helpers agree.
constexpr auto counter = fn::mbind(fn::get_state, [](int n) { return fn::mthen(fn::put_state(n + 1), fn::pure<fn::state<>>(n * n)); });
static_assert(fn::eval_state(counter, 6) == fn::run_state(counter, 6).first);
static_assert(fn::exec_state(counter, 6) == fn::run_state(counter, 6).second);
// Running is noexcept when the functions are, and the program is a value.
static_assert(noexcept(fn::run_state(fn::get_state, 1)));
static_assert(std::is_trivially_copyable_v<decltype(counter)>);
// A state of another type: a pair as the state.
constexpr auto swap_pair = fn::modify([](std::pair<int, int> p) { return std::pair{p.second, p.first}; });
static_assert(fn::exec_state(swap_pair, std::pair{1, 2}) == std::pair{2, 1});

// ---- Reader ----------------------------------------------------------------------------------------------------------

constexpr auto plus_one = [](int e) { return e + 1; };
static_assert(fn::run_reader(fn::local(fn::identity, fn::ask), 4) == fn::run_reader(fn::ask, 4));
static_assert(fn::run_reader(fn::local(plus_one, fn::local(twice, fn::ask)), 4) ==
              fn::run_reader(fn::local(fn::compose(twice, plus_one), fn::ask), 4));   // (4 + 1) * 2
static_assert(fn::run_reader(fn::asks(twice), 4) == fn::run_reader(fn::fmap(twice, fn::ask), 4));
static_assert(fn::run_reader(fn::mthen(fn::ask, fn::pure<fn::reader<>>('x')), 0) == 'x');

// ---- Writer ----------------------------------------------------------------------------------------------------------

using log_t = fn::sum_of<int>;
static_assert(fn::mthen(fn::tell(log_t{2}), fn::tell(log_t{3})).log == fn::tell(fn::combine(log_t{2}, log_t{3})).log);
static_assert(fn::tell(fn::mempty<log_t>()).log == fn::pure<fn::writer<log_t>>(unit{}).log);
static_assert(fn::run_writer(fn::writer<log_t, int>{4, {1}}) == std::pair{4, log_t{1}});
static_assert(fn::exec_writer(fn::mthen(fn::tell(log_t{5}), fn::writer<log_t, int>{0, {6}})) == log_t{11});
static_assert(fn::exec_writer(fn::writer<fn::max_of<int>, int>{1, {9}}).value == 9);

// ---- Identity and Const ----------------------------------------------------------------------------------------------

static_assert(fn::fmap(twice, fn::identity_of<int>{3}) == fn::identity_of<int>{6});
static_assert(fn::pure<fn::identity_of<>>(2) == fn::identity_of<int>{2});
static_assert(fn::fmap(twice, fn::const_of<int>{3}).value == 3);   // Const ignores the function
static_assert(fn::ap(fn::ap(fn::const_of<log_t>{{1}}, fn::const_of<log_t>{{2}}), fn::const_of<log_t>{{4}}).value == log_t{7});
static_assert(fn::pure<fn::const_of<log_t>>(42).value == fn::mempty<log_t>());

TEST(FunctionalStateLaws, AStringWriterAppendsInOrder)
{
    const auto w = fn::mthen(fn::tell(std::string("a")), fn::mthen(fn::tell(std::string("b")), fn::tell(std::string("c"))));
    EXPECT_EQ(fn::exec_writer(w), "abc");
    EXPECT_EQ(fn::exec_writer(fn::tell(fn::combine(std::string("a"), std::string("bc")))), "abc");
}

TEST(FunctionalStateLaws, AStringStateCarriesABuffer)
{
    const auto append = [](char c) { return fn::modify([c](std::string s) { return s + c; }); };
    const auto prog = fn::mthen(append('x'), fn::mthen(append('y'), fn::gets([](const std::string& s) { return s.size(); })));
    const auto [n, s] = fn::run_state(prog, std::string("w"));
    EXPECT_EQ(n, 3u);
    EXPECT_EQ(s, "wxy");
}

} // namespace
