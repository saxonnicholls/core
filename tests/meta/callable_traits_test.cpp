//
//  callable_traits_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/meta/callable_traits.hpp>. A pure type computation, so
//  every claim is a static_assert; the one TEST keeps the file a registered unit.
//

#include <snicholls/core/meta/callable_traits.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>

namespace {

using snicholls::callable_arguments_t;
using snicholls::callable_arity_v;
using snicholls::callable_result_t;
using snicholls::callable_traits;
using snicholls::single_signature_callable;
using snicholls::meta::type_list;

// Defined, and marked, because they are only ever named in unevaluated contexts.
[[maybe_unused]] int free_fn(int, double) { return 0; }
[[maybe_unused]] int free_noexcept(int) noexcept { return 0; }
[[maybe_unused]] int variadic(const char*, ...) { return 0; }

struct widget {
    int plain(int);
    int constant(int) const;
    int nothrow(int) const noexcept;
    int lvalue_only(int) &;
    int rvalue_only(int) &&;
    int volatile_ref(int) const volatile& noexcept;
};

struct adder {
    int operator()(int a, int b) const { return a + b; }
};
struct two_calls {
    int operator()(int);
    int operator()(double);
};

// function types, pointers and references
using fn_t = callable_traits<int(int, double)>;
static_assert(std::is_same_v<fn_t::result_type, int>);
static_assert(std::is_same_v<fn_t::arguments, type_list<int, double>>);
static_assert(std::is_same_v<fn_t::signature, int(int, double)>);
static_assert(fn_t::arity == 2 && !fn_t::is_noexcept && !fn_t::is_member);
static_assert(std::is_same_v<fn_t::argument_t<1>, double>);
static_assert(std::is_same_v<callable_traits<decltype(&free_fn)>::arguments, type_list<int, double>>);
static_assert(std::is_same_v<callable_traits<decltype(free_fn)>::arguments, type_list<int, double>>);   // the function type
static_assert(std::is_same_v<callable_traits<int (&)(int, double)>::arguments, type_list<int, double>>);   // a reference
static_assert(std::is_same_v<callable_traits<int (*const)(int, double)>::arguments, type_list<int, double>>);   // a const pointer
static_assert(callable_traits<decltype(&free_noexcept)>::is_noexcept);
static_assert(!single_signature_callable<decltype(&variadic)>);

// pointers to member functions, with their qualifications
static_assert(std::is_same_v<callable_traits<decltype(&widget::plain)>::class_type, widget>);
static_assert(callable_traits<decltype(&widget::plain)>::is_member);
static_assert(callable_traits<decltype(&widget::plain)>::arity == 1);
static_assert(!callable_traits<decltype(&widget::constant)>::is_noexcept);
static_assert(callable_traits<decltype(&widget::nothrow)>::is_noexcept);
static_assert(std::is_same_v<callable_traits<decltype(&widget::lvalue_only)>::result_type, int>);
static_assert(std::is_same_v<callable_traits<decltype(&widget::rvalue_only)>::arguments, type_list<int>>);
static_assert(callable_traits<decltype(&widget::volatile_ref)>::is_noexcept);
static_assert(std::is_same_v<callable_traits<decltype(&widget::volatile_ref)>::class_type, widget>);

// function objects and lambdas: the operator()'s signature, but not a member
static_assert(std::is_same_v<callable_traits<adder>::arguments, type_list<int, int>>);
static_assert(!callable_traits<adder>::is_member);
static_assert(std::is_same_v<callable_traits<adder>::class_type, adder>);

constexpr auto lambda = [](int a, const std::string& b) -> std::size_t { return static_cast<std::size_t>(a) + b.size(); };
using lambda_t = callable_traits<decltype(lambda)>;   // const object: cv stripped
static_assert(std::is_same_v<lambda_t::result_type, std::size_t>);
static_assert(std::is_same_v<lambda_t::argument_t<1>, const std::string&>);   // references kept
static_assert(lambda_t::arity == 2 && !lambda_t::is_member && !lambda_t::is_noexcept);
static_assert(std::is_same_v<callable_traits<const decltype(lambda)&>::arguments, type_list<int, const std::string&>>);

constexpr auto nothrow_lambda = [](int) noexcept { return 1; };
static_assert(callable_traits<decltype(nothrow_lambda)>::is_noexcept);

[[maybe_unused]] auto mutable_lambda = [n = 0](int) mutable { return ++n; };   // a non-const operator()
static_assert(callable_traits<decltype(mutable_lambda)>::arity == 1);

static_assert(std::is_same_v<callable_traits<std::function<void(int&)>>::arguments, type_list<int&>>);
static_assert(std::is_same_v<callable_traits<std::function<void(int&)>>::result_type, void>);

// no single signature: a generic lambda, an overloaded operator(), a non-callable
constexpr auto generic = [](auto x) { return x; };
static_assert(!single_signature_callable<decltype(generic)>);
static_assert(!single_signature_callable<two_calls>);
static_assert(!single_signature_callable<int>);
static_assert(!single_signature_callable<std::string>);
static_assert(single_signature_callable<adder> && single_signature_callable<decltype(lambda)> &&
              single_signature_callable<decltype(&free_fn)> && single_signature_callable<decltype(&widget::constant)>);

// the helpers
static_assert(std::is_same_v<callable_result_t<adder>, int>);
static_assert(std::is_same_v<callable_arguments_t<decltype(&free_fn)>, type_list<int, double>>);
static_assert(callable_arity_v<decltype(lambda)> == 2);
static_assert(callable_arity_v<int()> == 0);

TEST(CallableTraits, AreCheckedAtCompileTime)
{
    SUCCEED() << "every claim above is a static_assert";
}

} // namespace
