//
//  concepts_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/meta/concepts.hpp>: the concepts as the conditions of a
//  first_match chain, and each one asked directly. Exits non-zero if any answer
//  is not the one documented.
//

#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/overloaded.hpp>

#include <cstdio>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

struct point {
    int x;
    int y;
};
std::ostream& operator<<(std::ostream& os, const point& p) { return os << '(' << p.x << ", " << p.y << ')'; }

enum class colour { red, green = 7 };

//! [chain]
// Each branch states its condition with a concept; the order of the list is the chain.
std::string text(const auto& value);

const auto text_chain = snicholls::first_match{
    [](std::string_view s) { return std::string{s}; },
    [](std::same_as<bool> auto b) { return std::string{b ? "true" : "false"}; },
    [](snicholls::character auto c) { return std::string(1, static_cast<char>(c)); },
    [](snicholls::number auto n) { return std::to_string(n); },
    [](snicholls::enumeration auto e) { return std::to_string(static_cast<std::underlying_type_t<decltype(e)>>(e)); },
    [](const snicholls::map_like auto& m) {
        std::string out{"{"};
        for (const auto& [key, value] : m)
            out += (out.size() > 1 ? ", " : "") + text(key) + ": " + text(value);
        return out + "}";
    },
    [](const std::ranges::input_range auto& r) {
        std::string out{"["};
        for (const auto& element : r)
            out += (out.size() > 1 ? ", " : "") + text(element);
        return out + "]";
    },
    [](const snicholls::streamable auto& x) { std::ostringstream os; os << x; return os.str(); },
    [](const auto&) { return std::string{"?"}; },
};

std::string text(const auto& value) { return text_chain(value); }
//! [chain]

} // namespace

int main()
{
    std::printf("snicholls-core concepts demo\n");
    bool ok = true;

    const std::map<std::string, int> ages{{"ann", 41}, {"bob", 7}};
    const std::set<int> primes{2, 3, 5};
    struct opaque {};

    std::printf("  text(\"abc\")         = %s\n", text("abc").c_str());
    std::printf("  text(true)          = %s\n", text(true).c_str());
    std::printf("  text('c')           = %s\n", text('c').c_str());
    std::printf("  text(42)            = %s\n", text(42).c_str());
    std::printf("  text(colour::green) = %s\n", text(colour::green).c_str());
    std::printf("  text(ages)          = %s\n", text(ages).c_str());
    std::printf("  text(primes)        = %s\n", text(primes).c_str());
    std::printf("  text(point{1, 2})   = %s\n", text(point{1, 2}).c_str());
    std::printf("  text(opaque{})      = %s\n", text(opaque{}).c_str());

    ok &= text("abc") == "abc";
    ok &= text(true) == "true";
    ok &= text('c') == "c";
    ok &= text(42) == "42";
    ok &= text(colour::green) == "7";
    ok &= text(ages) == "{ann: 41, bob: 7}";
    ok &= text(primes) == "[2, 3, 5]";
    ok &= text(point{1, 2}) == "(1, 2)";
    ok &= text(opaque{}) == "?";

    //! [direct]
    // Asked directly, each concept is a constant.
    static_assert(snicholls::integer<int> && !snicholls::integer<bool> && !snicholls::integer<char>);
    static_assert(snicholls::enumeration<colour> && snicholls::scoped_enumeration<colour>);
    static_assert(snicholls::streamable<point> && !snicholls::streamable<opaque>);
    static_assert(snicholls::map_like<decltype(ages)> && snicholls::set_like<decltype(primes)>);
    static_assert(snicholls::specialisation_of<std::optional<int>, std::optional>);
    //! [direct]

    std::printf(ok ? "concepts demo: ok\n" : "concepts demo: FAILED\n");
    return ok ? 0 : 1;
}
