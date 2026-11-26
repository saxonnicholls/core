//
//  state_machine_reachability_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/state_machine.hpp> with <snicholls/core/iterate.hpp> and
//  <snicholls/core/functional/rewrite.hpp>: questions about a machine answered by fixed points, at compile time. The
//  transition relation is read off the machine itself (restore each state, offer each event, see where it goes); then
//  "which states can be reached", "from which states can the order still be delivered" and "how many steps to each
//  state" are each the least fixed point of a function on sets of states, found by iterating it until it stops
//  changing. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/functional/rewrite.hpp>
#include <snicholls/core/iterate.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <tuple>

namespace {

namespace fn = snicholls::fn;

//! [machine]
enum class order : std::uint8_t { draft, placed, paid, shipped, delivered, cancelled, refunded, returned };
constexpr std::size_t state_count = 8;
struct place {};
struct pay {};
struct ship {};
struct deliver {};
struct cancel {};
struct refund {};
struct send_back {};
using events = std::tuple<place, pay, ship, deliver, cancel, refund, send_back>;

struct order_lifecycle {
    constexpr auto operator()() const
    {
        using namespace snicholls::fsm;
        constexpr auto draft = enum_state<order::draft>;
        constexpr auto placed = enum_state<order::placed>;
        constexpr auto paid = enum_state<order::paid>;
        constexpr auto shipped = enum_state<order::shipped>;
        constexpr auto delivered = enum_state<order::delivered>;
        constexpr auto cancelled = enum_state<order::cancelled>;
        constexpr auto refunded = enum_state<order::refunded>;
        constexpr auto returned = enum_state<order::returned>;
        return make_transition_table(
            *draft     + event<place>     = placed,
             placed    + event<pay>       = paid,
             placed    + event<cancel>    = cancelled,
             paid      + event<ship>      = shipped,
             paid      + event<cancel>    = refunded,
             shipped   + event<deliver>   = delivered,
             delivered + event<send_back> = returned,
             returned  + event<refund>    = refunded);
    }
};
using machine = snicholls::fsm::sm<order_lifecycle>;
//! [machine]

using set = std::uint32_t;   // a set of states, one bit per enumerator
constexpr set bit(order s) { return set{1} << static_cast<unsigned>(s); }

//! [relation]
// The successor sets, read from the machine: from each state, every event, where it lands.
constexpr std::array<set, state_count> successors()
{
    std::array<set, state_count> next{};
    for (std::size_t s = 0; s < state_count; ++s) {
        std::apply([&](auto... e) {
            ((
                 [&] {
                     machine m;
                     static_cast<void>(m.restore(static_cast<order>(s)));
                     if (m.process_event(e)) next[s] |= bit(m.current());
                 }()),
             ...);
        }, events{});
    }
    return next;
}
constexpr auto next = successors();

constexpr set image(set from)   // every state one step from a state in `from`
{
    set out = 0;
    for (std::size_t s = 0; s < state_count; ++s)
        if (from >> s & 1u) out |= next[s];
    return out;
}
constexpr set preimage(set to)   // every state with a step into `to`
{
    set out = 0;
    for (std::size_t s = 0; s < state_count; ++s)
        if (next[s] & to) out |= set{1} << s;
    return out;
}
//! [relation]

//! [fixed_points]
// Reachable from draft: the least set R with draft in R and image(R) within R. Iterate R -> R | image(R).
constexpr auto reachable = snicholls::fixed_point([](set r) { return r | image(r); }, bit(order::draft), 64);
static_assert(reachable.converged && reachable.value == (set{1} << state_count) - 1);   // every state is reachable

// Can still be delivered: the least set D with delivered in D and preimage(D) within D.
constexpr auto can_deliver = snicholls::fixed_point([](set d) { return d | preimage(d); }, bit(order::delivered), 64);
static_assert(can_deliver.value ==
              (bit(order::draft) | bit(order::placed) | bit(order::paid) | bit(order::shipped) | bit(order::delivered)));

// Terminal states: no successors at all.
constexpr set terminal = [] {
    set t = 0;
    for (std::size_t s = 0; s < state_count; ++s)
        if (next[s] == 0) t |= set{1} << s;
    return t;
}();
static_assert(terminal == (bit(order::cancelled) | bit(order::refunded)));
//! [fixed_points]

//! [rewrite]
// The same closure as rewriting: a rule that fires while some successor is missing, applied until it no longer
// fires. The count of rewrites is the number of breadth-first layers: the longest shortest path from draft.
constexpr auto grow = fn::rule([](set r) { return (r | image(r)) != r; }, [](set r) { return r | image(r); });
constexpr auto layers = fn::replace_repeated(bit(order::draft), grow, 64);
static_assert(layers.converged && layers.value == reachable.value);
static_assert(layers.iterations == 5);   // returned is five steps from draft: placed, paid, shipped, delivered, returned
//! [rewrite]

// The distance of every state from draft: the layer at which it first appears.
constexpr std::array<int, state_count> distances()
{
    std::array<int, state_count> d{};
    d.fill(-1);
    set seen = 0;
    set frontier = bit(order::draft);
    for (int layer = 0; frontier != 0; ++layer) {
        for (std::size_t s = 0; s < state_count; ++s)
            if (frontier >> s & 1u) d[s] = layer;
        seen |= frontier;
        frontier = image(frontier) & ~seen;
    }
    return d;
}
constexpr auto distance = distances();
static_assert(distance[static_cast<std::size_t>(order::refunded)] == 3);   // draft -> placed -> paid -> refunded

const char* name(std::size_t s)
{
    constexpr std::array<const char*, state_count> names{"draft",     "placed",    "paid",     "shipped",
                                                         "delivered", "cancelled", "refunded", "returned"};
    return names[s];
}

} // namespace

int main()
{
    std::printf("snicholls-core fsm: reachability by fixed points\n");
    bool ok = true;
    for (std::size_t s = 0; s < state_count; ++s)
        std::printf("  %-9s  distance %d  %s%s\n", name(s), distance[s], can_deliver.value >> s & 1u ? "can deliver" : "           ",
                    terminal >> s & 1u ? "  terminal" : "");
    ok &= std::popcount(reachable.value) == static_cast<int>(state_count);
    ok &= std::popcount(can_deliver.value) == 5 && std::popcount(terminal) == 2;
    ok &= layers.iterations == 5 && distance[static_cast<std::size_t>(order::returned)] == 5;
    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
