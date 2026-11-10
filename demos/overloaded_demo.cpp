//
//  overloaded_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/overloaded.hpp>: visiting a variant with one handler per
//  alternative (overloaded), an if constexpr chain written as an ordered set of
//  constrained lambdas (first_match), and function pointers as elements. Exits
//  non-zero if any handler is not the one documented.
//

#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/overloaded.hpp>

#include <concepts>
#include <cstdio>
#include <format>
#include <map>
#include <ostream>
#include <ranges>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace {

//! [first_match]
// The conditions of the chain are core's concepts (snicholls/core/meta/concepts.hpp).
std::string text(const auto& value);   // the chain calls itself for the elements of a range

// A to_string written as one generic lambda has an if constexpr / else if constexpr
// chain inside it. Here every branch is its own callable and the order of the list is the
// chain: the first that accepts the value runs, and nothing is ambiguous. No if constexpr.
const auto text_chain = snicholls::first_match{
    [](std::string_view s) { return std::string{s}; },                              // std::string, const char*, string_view
    [](std::same_as<bool> auto b) { return std::string{b ? "true" : "false"}; },
    [](snicholls::character auto c) { return std::string(1, static_cast<char>(c)); },
    [](snicholls::integer auto n) { return std::to_string(n); },                    // not bool, not a character
    [](std::floating_point auto x) { return std::format("{}", x); },
    [](snicholls::enumeration auto e) { return std::to_string(static_cast<std::underlying_type_t<decltype(e)>>(e)); },
    [](const snicholls::map_like auto& m) {
        std::string out{"{"};
        for (const auto& [key, value] : m)
            out += (out.size() > 1 ? ", " : "") + text(key) + ": " + text(value);
        return out + "}";
    },
    [](const std::ranges::input_range auto& r) {   // after map_like: a map is a range too
        std::string out{"["};
        for (const auto& element : r)
            out += (out.size() > 1 ? ", " : "") + text(element);
        return out + "]";
    },
    [](const snicholls::streamable auto& x) { std::ostringstream os; os << x; return os.str(); },
    [](const auto&) { return std::string{"?"}; },                                   // the else
};

std::string text(const auto& value) { return text_chain(value); }
//! [first_match]

struct point {
    int x;
    int y;
};

std::ostream& operator<<(std::ostream& os, const point& p) { return os << '(' << p.x << ", " << p.y << ')'; }

enum class colour { red, green = 7 };

//! [fn_ptr]
int twice(int n) { return 2 * n; }
//! [fn_ptr]

} // namespace

int main()
{
    std::printf("snicholls-core overloaded demo\n");
    bool ok = true;

    //! [visit]
    using message = std::variant<int, double, std::string>;
    const std::vector<message> inbox{42, 2.5, std::string{"hello"}};

    const auto describe = snicholls::overloaded{
        [](int n) { return "int " + std::to_string(n); },
        [](const std::string& s) { return "string " + s; },
        [](const auto&) { return std::string{"something else"}; },   // catch-all
    };

    for (const auto& m : inbox)
        std::printf("  %s\n", std::visit(describe, m).c_str());
    //! [visit]

    ok &= std::visit(describe, inbox[0]) == "int 42";
    ok &= std::visit(describe, inbox[1]) == "something else";
    ok &= std::visit(describe, inbox[2]) == "string hello";

    // first_match: every branch of the former if constexpr chain, in order
    struct opaque {};
    const std::map<std::string, int> ages{{"ann", 41}, {"bob", 7}};
    const std::vector<std::vector<int>> nested{{1}, {2, 3}};
    std::printf("  text(\"abc\")          = %s\n", text("abc").c_str());
    std::printf("  text(true)           = %s\n", text(true).c_str());
    std::printf("  text('c')            = %s\n", text('c').c_str());
    std::printf("  text(42)             = %s\n", text(42).c_str());
    std::printf("  text(2.5)            = %s\n", text(2.5).c_str());
    std::printf("  text(colour::green)  = %s\n", text(colour::green).c_str());
    std::printf("  text(ages)           = %s\n", text(ages).c_str());
    std::printf("  text(nested)         = %s\n", text(nested).c_str());
    std::printf("  text(point{1, 2})    = %s\n", text(point{1, 2}).c_str());
    std::printf("  text(opaque{})       = %s\n", text(opaque{}).c_str());

    ok &= text("abc") == "abc";
    ok &= text(std::string{"s"}) == "s";
    ok &= text(std::string_view{"v"}) == "v";
    ok &= text(true) == "true";
    ok &= text('c') == "c";
    ok &= text(42) == "42";
    ok &= text(2.5) == "2.5";
    ok &= text(colour::green) == "7";
    ok &= text(ages) == "{ann: 41, bob: 7}";
    ok &= text(nested) == "[[1], [2, 3]]";
    ok &= text(point{1, 2}) == "(1, 2)";
    ok &= text(opaque{}) == "?";

    //! [fn_ptr]
    // A function name (or its address) is accepted as an element: it becomes a snicholls::fn_ptr
    // with the function's exact signature, so it takes part in overload resolution as the
    // function would.
    const auto scale = snicholls::overloaded{twice, [](double d) { return d / 2; }};
    std::printf("  scale(21) = %d, scale(3.0) = %g\n", scale(21), scale(3.0));
    //! [fn_ptr]

    ok &= scale(21) == 42;
    ok &= scale(3.0) == 1.5;

    std::printf(ok ? "overloaded demo: ok\n" : "overloaded demo: FAILED\n");
    return ok ? 0 : 1;
}
