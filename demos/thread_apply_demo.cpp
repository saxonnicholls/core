//
//  thread_apply_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/thread_apply.hpp>: Mathematica's Thread over tuples, with an
//  overload set choosing per element. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/overloaded.hpp>
#include <snicholls/core/thread_apply.hpp>

#include <array>
#include <cstdio>
#include <string>
#include <tuple>

int main()
{
    std::printf("snicholls-core thread_apply demo\n");
    bool ok = true;

    //! [thread_apply]
    // Thread[f[{a, b}, {x, y}]] is {f[a, x], f[b, y}]: corresponding elements, whatever their types.
    constexpr auto add = [](auto a, auto b) { return a + b; };
    static_assert(snicholls::thread_apply(add, std::tuple{1, 2.5}, std::tuple{10, 20}) == std::tuple{11, 22.5});

    // Thread[f[{a, b}, x]] is {f[a, x], f[b, x]}: an argument that is not a list is repeated.
    static_assert(snicholls::thread_apply(add, std::tuple{1, 2}, 10) == std::tuple{11, 12});

    // One overload per element's type, chosen at compile time.
    std::string log;
    snicholls::thread_apply(snicholls::overloaded{
                                [&log](int qty, const char* venue) { log += std::string(venue) + " qty " + std::to_string(qty) + "; "; },
                                [&log](double px, const char* venue) { log += std::string(venue) + " px " + std::to_string(px) + "; "; },
                            },
                            std::tuple{100, 189.25}, "NYSE");

    // whole() repeats a tuple rather than threading it.
    constexpr auto second_plus = [](int a, const std::tuple<int, int>& t) { return a + std::get<1>(t); };
    static_assert(snicholls::thread_apply(second_plus, std::array{1, 2}, snicholls::whole(std::tuple{0, 5})) == std::tuple{6, 7});
    //! [thread_apply]

    //! [outer]
    // Outer[f, {a, b}, {x, y}] is {{f[a, x], f[a, y]}, {f[b, x], f[b, y]}}: every combination, row-major.
    constexpr auto notional = [](double price, int quantity) { return price * quantity; };
    constexpr auto grid = snicholls::outer(notional, std::array{189.25, 420.50}, std::array{100, 200, 300});
    static_assert(grid[1][2] == 420.50 * 300);
    //! [outer]
    ok &= grid[0][0] == 18925.0;

    std::printf("  %s\n", log.c_str());
    ok &= log == "NYSE qty 100; NYSE px 189.250000; ";

    const auto [doubled, text] = snicholls::thread_apply(
        snicholls::overloaded{[](int v) { return std::to_string(v * 2); }, [](const std::string& v) { return v + v; }}, std::tuple{21, std::string{"ab"}});
    ok &= doubled == "42" && text == "abab";
    ok &= snicholls::thread_apply(add, 1, 2) == 3;   // nothing to thread: the plain call

    std::printf(ok ? "thread_apply demo: ok\n" : "thread_apply demo: FAILED\n");
    return ok ? 0 : 1;
}
