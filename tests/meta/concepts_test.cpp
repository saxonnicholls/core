//
//  concepts_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/meta/concepts.hpp>. Concepts are compile-time
//  facilities, so every claim is a static_assert; the one TEST keeps the file a
//  registered unit.
//

#include <snicholls/core/meta/concepts.hpp>

#include <gtest/gtest.h>

#include <string_view>

#include <list>

#include <array>

#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using namespace snicholls;

enum plain { one, two };
enum class scoped { a, b };
struct has_stream_operator {};
[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const has_stream_operator&) { return os; }   // named in concepts only
struct silent {};

// specialisation_of
static_assert(specialisation_of<std::optional<int>, std::optional>);
static_assert(specialisation_of<const std::vector<int>&, std::vector>);
static_assert(!specialisation_of<int, std::optional>);
static_assert(!specialisation_of<std::vector<int>, std::optional>);

// printable: print(std::ostream&) const or print(acout&) const, and only those
struct self_printing {
    void print(std::ostream&) const {}
};
struct prints_to_acout {
    void print(acout&) const;   // acout is only declared here; a reference needs no more
};
struct prints_both {
    void print(std::ostream&) const {}
    void print(acout&) const;
};
struct prints_mutably {
    void print(std::ostream&) {}
};
struct prints_elsewhere {
    void print(int) const {}
};
static_assert(printable_to_stream<self_printing> && !printable_to_acout<self_printing>);
static_assert(printable_to_acout<prints_to_acout> && !printable_to_stream<prints_to_acout>);
static_assert(printable_to_stream<prints_both> && printable_to_acout<prints_both>);
static_assert(printable<self_printing> && printable<const self_printing&> && printable<prints_to_acout> && printable<prints_both>);
static_assert(!printable<prints_mutably> && !printable<prints_elsewhere> && !printable<int>);

// streamable
static_assert(streamable<int>);
static_assert(streamable<const std::string&>);
static_assert(streamable<has_stream_operator>);
static_assert(!streamable<silent>);
static_assert(!streamable<std::vector<int>>);
static_assert(streamable_to<int, std::ostream>);

// enumerations
static_assert(enumeration<plain> && enumeration<scoped> && enumeration<const scoped&>);
static_assert(!enumeration<int>);
static_assert(scoped_enumeration<scoped> && !scoped_enumeration<plain>);
static_assert(unscoped_enumeration<plain> && !unscoped_enumeration<scoped>);

// characters are not integers, bool is not an integer, bytes are
static_assert(character<char> && character<wchar_t> && character<char8_t> && character<char16_t> && character<char32_t>);
static_assert(!character<signed char> && !character<unsigned char> && !character<int>);
static_assert(integer<int> && integer<unsigned long long> && integer<signed char> && integer<unsigned char> && integer<const int&>);
static_assert(!integer<bool> && !integer<char> && !integer<double> && !integer<scoped>);
static_assert(number<int> && number<double> && number<float>);
static_assert(!number<bool> && !number<char> && !number<std::string>);

// map_like and set_like
static_assert(map_like<std::map<int, int>> && map_like<std::unordered_map<std::string, int>> && map_like<std::multimap<int, int>>);
static_assert(map_like<const std::map<int, int>&>);
static_assert(!map_like<std::vector<std::pair<int, int>>> && !map_like<std::set<int>> && !map_like<int>);
static_assert(set_like<std::set<int>> && set_like<std::unordered_set<int>> && set_like<std::multiset<int>>);
static_assert(!set_like<std::map<int, int>> && !set_like<std::vector<int>>);

// always_false_v is false, and dependent
template <typename T>
constexpr bool branch_reached = always_false_v<T>;
static_assert(!branch_reached<int>);

TEST(Concepts, AreCheckedAtCompileTime)
{
    SUCCEED() << "every claim above is a static_assert";
}

// ---- added with the meta/ move ----------------------------------------------------------------
struct named_usd { static constexpr std::string_view name() { return "usd"; } };
struct named_aud { static constexpr const char* name() { return "aud"; } };
struct named_usd_again { static constexpr std::string_view name() { return "usd"; } };
struct throwing_hash_key {};

static_assert(snicholls::range_of<std::vector<int>, int> && !snicholls::range_of<std::vector<int>, long>);
static_assert(snicholls::std_hashable<int> && snicholls::std_hashable<std::string> && !snicholls::std_hashable<throwing_hash_key>);
static_assert(snicholls::nothrow_std_hashable<int>);
static_assert(snicholls::reservable<std::vector<int>> && snicholls::reservable<std::string> && !snicholls::reservable<std::list<int>>);
static_assert(snicholls::is_std_array_v<std::array<int, 3>> && snicholls::is_std_array_v<const std::array<int, 0>&>);
static_assert(!snicholls::is_std_array_v<std::vector<int>> && !snicholls::is_std_array_v<int[3]>);
static_assert(snicholls::has_static_name<named_usd> && snicholls::has_static_name<named_aud> && !snicholls::has_static_name<int>);
static_assert(snicholls::distinct_static_names_v<named_usd, named_aud>);
static_assert(!snicholls::distinct_static_names_v<named_usd, named_aud, named_usd_again>);
static_assert(snicholls::distinct_static_names_v<>);

} // namespace
