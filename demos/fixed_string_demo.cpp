//
//  fixed_string_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fixed_string.hpp>: compile-time string operations,
//  a string as a template parameter, and a fixed string keying an unordered
//  set. Exits non-zero if any result is not what it documents.
//

#include <snicholls/core/fixed_string.hpp>

#include <cstdio>
#include <string_view>
#include <unordered_set>

//! [nttp]
// A type parameterised by a name, with no run-time string anywhere
template <snicholls::basic_fixed_string Name>
struct field {
    static constexpr std::string_view name = Name;
    double value = 0;
};

using price = field<"price">;
using quantity = field<"quantity">;
//! [nttp]

int main()
{
    std::printf("snicholls-core fixed_string demo\n");
    bool ok = true;

    //! [basics]
    constexpr snicholls::fixed_string greeting = "hello";
    static_assert(greeting.size() == 5);
    static_assert(greeting == "hello");
    static_assert((greeting + snicholls::fixed_string(", world")) == "hello, world");
    static_assert(greeting.substr<1, 3>() == "ell");
    //! [basics]

    std::printf("  greeting        \"%s\" (%zu characters)\n", greeting.c_str(), greeting.size());
    const auto joined = greeting + snicholls::fixed_string(", world");
    std::printf("  concatenated    \"%s\"\n", joined.c_str());

    price p{101.5};
    quantity q{3};
    std::printf("  field names     %.*s = %g, %.*s = %g\n",
                static_cast<int>(price::name.size()), price::name.data(), p.value,
                static_cast<int>(quantity::name.size()), quantity::name.data(), q.value);
    ok &= price::name == "price" && quantity::name == "quantity";

    // Fixed strings compare with each other directly, at any lengths, and
    // key an unordered container with the default hash and equality
    std::unordered_set<snicholls::basic_fixed_string<char, 3>> seen{
        snicholls::basic_fixed_string("abc"), snicholls::basic_fixed_string("xyz"),
        snicholls::basic_fixed_string("abc")};
    static_assert(snicholls::fixed_string("abc") < snicholls::fixed_string("abcd"));
    ok &= snicholls::fixed_string("abc") != snicholls::fixed_string("abd");
    std::printf("  distinct keys   %zu\n", seen.size());
    ok &= seen.size() == 2;
    ok &= joined == "hello, world";

    std::printf(ok ? "fixed_string demo: ok\n" : "fixed_string demo: FAILED\n");
    return ok ? 0 : 1;
}
