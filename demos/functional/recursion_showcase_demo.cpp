//
//  recursion_showcase_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Showcase for <snicholls/core/functional/recursion.hpp>: recursion schemes on classic problems, by the compiler:
//  a number's digits (ana) and its digital root (hylo), binary digits read back by a catamorphism, pairs counted by a
//  paramorphism, Euclid's algorithm and Ackermann's function on a trampoline with an explicit stack.
//  Exits non-zero on a mismatch.
//

#include <snicholls/core/functional/recursion.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <optional>
#include <utility>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}

// Ackermann's state: the pending values of m on a fixed stack, and the current n.
struct ack_state {
    std::array<int, 512> ms{};   // deep enough for A(3, 6); A(m, n) needs about A(m, n) + 1 slots
    std::size_t depth = 0;
    int n = 0;
};
} // namespace

int main()
{
    namespace fn = snicholls::fn;
    std::printf("snicholls-core functional/recursion showcase\n");

    //! [digits]
    // ana unfolds a number into its digits; hylo folds them as they come, never building the list.
    constexpr auto digits = [](long n) { return n == 0 ? std::nullopt : std::optional{std::pair{n % 10, n / 10}}; };
    static_assert(fn::ana<20>(digits, 2026L).size() == 4);
    constexpr auto digit_sum = [digits](long n) { return fn::hylo([](long acc, long d) { return acc + d; }, 0L, digits, n); };
    constexpr auto digital_root = [digit_sum](long n) {
        while (n >= 10) n = digit_sum(n);
        return n;
    };
    static_assert(digit_sum(987'654'321L) == 45 && digital_root(987'654'321L) == 9);
    //! [digits]
    check(digital_root(2026L) == 1, "2026's digital root is 1");

    //! [cata]
    // A catamorphism replaces the list's constructors: cons by an algebra, nil by a seed.
    // Here the binary digits, most significant first, become the number they spell.
    constexpr std::array bits{1, 0, 1, 1, 0, 1};
    constexpr auto value = fn::cata([](int bit, std::pair<int, int> acc) { return std::pair{acc.first + bit * acc.second, acc.second * 2}; },
                                    std::pair{0, 1}, bits).first;
    static_assert(value == 0b101101);
    //! [cata]
    check(value == 45, "101101 in binary is 45");

    //! [para]
    // A paramorphism also sees the rest of the list: count the later elements each one beats (a record count).
    constexpr std::array prices{5, 3, 8, 1, 4};
    constexpr int beaten = fn::para([](int x, auto rest, int acc) {
        for (int y : rest) acc += x > y ? 1 : 0;
        return acc;
    }, 0, prices);
    static_assert(beaten == 6);
    //! [para]
    check(beaten == 6, "six pairs where an earlier price beats a later one");

    //! [euclid]
    // Euclid's algorithm as a trampoline: each step is more(next) or done(result), in a loop.
    constexpr auto gcd_step = [](std::pair<long, long> s) -> fn::bounce<std::pair<long, long>, long> {
        if (s.second == 0) return fn::done(s.first);
        return fn::more(std::pair{s.second, s.first % s.second});
    };
    static_assert(fn::trampoline(gcd_step, std::pair{1'071L, 462L}) == 21);
    //! [euclid]
    check(fn::trampoline(gcd_step, std::pair{832'040L, 514'229L}) == 1, "neighbouring Fibonacci numbers are coprime");

    //! [ackermann]
    // Ackermann's function is not primitive recursive; with its pending calls on an explicit stack it is a loop.
    constexpr auto ack_step = [](ack_state s) -> fn::bounce<ack_state, int> {
        if (s.depth == 0) return fn::done(s.n);
        if (s.depth + 1 >= s.ms.size()) return fn::done(-1);   // out of stack: say so, never overflow
        const int m = s.ms[--s.depth];
        if (m == 0) {
            s.n += 1;
        } else if (s.n == 0) {
            s.ms[s.depth++] = m - 1;
            s.n = 1;
        } else {
            s.ms[s.depth++] = m - 1;
            s.ms[s.depth++] = m;
            s.n -= 1;
        }
        return fn::more(s);
    };
    constexpr auto ackermann = [ack_step](int m, int n) {
        ack_state s;
        s.ms[0] = m;
        s.depth = 1;
        s.n = n;
        return fn::trampoline(ack_step, s);
    };
    static_assert(ackermann(2, 3) == 9 && ackermann(3, 3) == 61);
    //! [ackermann]
    check(ackermann(3, 5) == 253, "A(3, 5) = 253, at run time");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
