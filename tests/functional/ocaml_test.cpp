//
//  ocaml_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/functional/ocaml.hpp>: the Option, Result and List modules under OCaml's names.
//

#include <snicholls/core/functional/ocaml.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace {
namespace fn = snicholls::fn; namespace oc = snicholls::fn::ocaml;
enum class e { bad };
using R = snicholls::result<int, e>;
static_assert(oc::option::map([](int x) { return x + 1; }, std::optional<int>{1}) == std::optional<int>{2});
static_assert(oc::option::bind(std::optional<int>{1}, [](int x) { return std::optional<int>{x * 3}; }) == std::optional<int>{3});
static_assert(oc::option::value(std::optional<int>{}, 4) == 4 && oc::option::get(std::optional<int>{2}) == 2);
static_assert(oc::option::is_none(std::optional<int>{}) && oc::option::some(3) == std::optional<int>{3});
static_assert(oc::option::fold(0, [](int x) { return x * 2; }, std::optional<int>{3}) == 6);
static_assert(oc::result::map_error([](e) { return 1; }, R(snicholls::err(e::bad))).unwrap_err_unchecked() == 1);
static_assert(oc::result::fold([](int v) { return v; }, [](e) { return -1; }, R(snicholls::ok(5))) == 5);
static_assert(oc::result::get_ok(R(snicholls::ok(5))) == 5 && oc::result::is_ok(R(snicholls::ok(5))));
static_assert(*oc::result::to_option(R(snicholls::ok(5))) == 5);
static_assert(oc::list::fold_right([](int x, int acc) { return x - acc; }, std::array{1, 2, 3}, 0) == 2);
static_assert(oc::list::sort([](int a, int b) { return b - a; }, std::array{1, 3, 2}) == std::array{3, 2, 1});
static_assert(oc::list::init<3>([](std::size_t i) { return i * i; }) == std::array<std::size_t, 3>{0, 1, 4});
static_assert((2 | fn::then([](int x) { return x + 1; })) == fn::call([](int x) { return x + 1; }, 2));
static_assert(oc::option::join(std::optional<std::optional<int>>{std::optional<int>{2}}) == std::optional<int>{2});
static_assert(oc::result::is_error(R(snicholls::err(e::bad))) && oc::result::get_error(R(snicholls::err(e::bad))) == e::bad);
static_assert(oc::list::exists([](int x) { return x > 2; }, std::array{1, 3}) && oc::list::for_all([](int x) { return x > 0; }, std::array{1, 3}));
static_assert(oc::list::assoc_opt(2, std::array{std::pair{2, 'z'}}) == std::optional<char>{'z'});

TEST(FunctionalOcaml, OptionValueWithAStringDefault)
{
    const std::optional<std::string> venue;
    EXPECT_EQ(oc::option::value(venue, std::string("LMAX")), "LMAX");
}

} // namespace
