//
//  state_machine_fold_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/state_machine.hpp> with <snicholls/core/functional/>: a state machine is a fold. The
//  machine is a value, an event is a std::variant matched with fn::match, and running it over a list of events is
//  fn::foldl_of (the final state), fn::scanl (every state on the way) or a lazy fn::stream (an endless input taken
//  as far as wanted). Everything runs at compile time and is checked with static_assert. Exits non-zero if a result
//  is not the one documented.
//

#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/functional/fold.hpp>
#include <snicholls/core/functional/lazy.hpp>
#include <snicholls/core/functional/list.hpp>
#include <snicholls/core/functional/match.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <variant>

namespace {

namespace fn = snicholls::fn;

//! [machine]
// The classic turnstile: a coin unlocks it, a push locks it again.
enum class turn : std::uint8_t { locked, unlocked };
struct coin {};
struct push {};
using input = std::variant<coin, push>;

struct turnstile {
    constexpr auto operator()() const
    {
        using namespace snicholls::fsm;
        constexpr auto locked = enum_state<turn::locked>;
        constexpr auto unlocked = enum_state<turn::unlocked>;
        return make_transition_table(
            *locked   + event<coin> = unlocked,
             locked   + event<push> = locked,     // pushing a locked turnstile does nothing
             unlocked + event<push> = locked,
             unlocked + event<coin> = unlocked);  // a second coin is kept
    }
};
using machine = snicholls::fsm::sm<turnstile>;

// One step of the fold: the machine is taken by value and returned, as a pure function of (machine, event).
constexpr machine step(machine m, const input& e)
{
    fn::match(e, [&](const auto& ev) { static_cast<void>(m.process_event(ev)); });
    return m;
}
//! [machine]

constexpr turn state_of(const machine& m) { return m.current(); }

} // namespace

int main()
{
    std::printf("snicholls-core fsm: a state machine is a fold\n");
    bool ok = true;

    //! [fold]
    constexpr std::array<input, 6> day{coin{}, push{}, push{}, coin{}, coin{}, push{}};
    constexpr turn last = fn::foldl_of(step, machine{}, day).current();   // where the day ends
    constexpr auto trace = fn::map(state_of, fn::scanl(step, machine{}, day));   // every state, the start included
    static_assert(last == turn::locked);
    static_assert(trace == std::array{turn::locked, turn::unlocked, turn::locked, turn::locked, turn::unlocked,
                                      turn::unlocked, turn::locked});
    //! [fold]
    for (turn t : trace) std::printf("  %s\n", t == turn::locked ? "locked" : "unlocked");
    ok &= last == turn::locked && trace.size() == day.size() + 1;

    //! [paired]
    // The machine and what it has earned, folded together: a coin counts only when the machine takes it unlocked
    // from locked (the second coin in a row is kept, not counted).
    constexpr auto earn = [](std::pair<machine, int> acc, const input& e) {
        const bool was_locked = acc.first.current() == turn::locked;
        acc.first = step(acc.first, e);
        const bool paid = was_locked && acc.first.current() == turn::unlocked;
        return std::pair{acc.first, acc.second + (paid ? 1 : 0)};
    };
    constexpr auto takings = fn::foldl_of(earn, std::pair{machine{}, 0}, day);
    static_assert(takings.second == 2 && takings.first.current() == turn::locked);
    //! [paired]
    ok &= takings.second == 2;

    //! [stream]
    // An endless input, coin then push then push, forever: a lazy stream of machines taken as far as wanted.
    constexpr auto inputs = fn::stream::cycle(std::array<input, 3>{coin{}, push{}, push{}});
    constexpr auto states = fn::stream::map(state_of, fn::stream::scanl(step, machine{}, inputs));
    constexpr auto first_seven = fn::stream::take<7>(states);
    static_assert(first_seven == std::array{turn::locked, turn::unlocked, turn::locked, turn::locked, turn::unlocked,
                                            turn::locked, turn::locked});
    // How many coins until the turnstile has been unlocked ten times: the stream is cut where the count is reached.
    constexpr auto unlocks = fn::stream::scanl([](int n, turn t) { return n + (t == turn::unlocked ? 1 : 0); }, 0, states);
    static_assert(fn::stream::nth(fn::stream::drop_while([](int n) { return n < 10; }, unlocks), 0) == 10);
    //! [stream]
    ok &= first_seven[1] == turn::unlocked;

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
