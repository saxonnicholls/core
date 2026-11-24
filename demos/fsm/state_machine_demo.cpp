//
//  state_machine_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/state_machine.hpp>: what core adds to sml's design. States as enumerators, so the
//  machine's state is a value (current(), restore()); refusals as values (try_process_event); the compile-time checks
//  as values (table_diagnostics); and a snapshot of every region. sml's own examples are replicated in
//  state_machine_<example>_demo.cpp. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/fsm/state_machine.hpp>

#include <cstdint>
#include <cstdio>
#include <string_view>

namespace {

//! [enum]
enum class order_state : std::uint8_t { placed, paid, shipped, delivered, cancelled };

struct pay {
    long cents = 0;
};
struct ship {};
struct deliver {};
struct cancel {};

struct order {
    constexpr auto operator()() const   // constexpr: the machine can then run at compile time
    {
        using namespace snicholls::fsm;
        constexpr auto placed = enum_state<order_state::placed>;
        constexpr auto paid = enum_state<order_state::paid>;
        constexpr auto shipped = enum_state<order_state::shipped>;
        constexpr auto delivered = enum_state<order_state::delivered>;
        constexpr auto cancelled = enum_state<order_state::cancelled>;
        const auto paid_in_full = [](const pay& p) { return p.cents >= 1999; };
        return make_transition_table(
            *placed + event<pay> [paid_in_full] = paid,
             placed + event<cancel>             = cancelled,
             paid   + event<ship>               = shipped,
             paid   + event<cancel>             = cancelled,
             shipped + event<deliver>           = delivered);
    }
};
//! [enum]

struct broken {   // shadows its own second row: the compiler refuses sm<broken>; table_diagnostics says why
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"a"_s + event<ship> = "b"_s,
             "a"_s + event<ship> = "c"_s,
             "orphan"_s + event<ship> = "b"_s);
    }
};

std::string_view name(order_state s) { return snicholls::enum_name(s); }

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm demo\n");
    bool ok = true;

    sm<order> machine;
    static_assert(std::is_same_v<sm<order>::state_enum, order_state>);   // every state is an enumerator
    static_assert(sizeof(machine) == 1);                                     // the state is one byte

    // The state is a value: it goes on the order object, and comes back from it.
    const auto refused = machine.try_process_event(pay{500});
    ok &= refused.is_err() && refused.unwrap_err_unchecked() == fsm_errc::guard_rejected;
    ok &= machine.process_event(pay{1999});
    std::printf("  after pay: %.*s\n", static_cast<int>(name(machine.current()).size()), name(machine.current()).data());
    ok &= machine.current() == order_state::paid;

    const auto wrong = machine.try_process_event(deliver{});   // nothing leaves paid on deliver
    ok &= wrong.is_err() && wrong.unwrap_err_unchecked() == fsm_errc::no_transition;

    //! [restore]
    const order_state stored = machine.current();   // e.g. a column of the order's row
    sm<order> rebuilt;                              // after a restart
    ok &= rebuilt.restore(stored).is_ok();          // no action runs
    ok &= rebuilt.process_event(ship{}) && rebuilt.current() == order_state::shipped;
    //! [restore]

    // An enumerator the table does not use is refused, and nothing changes
    const auto bad = rebuilt.restore(static_cast<order_state>(42));
    ok &= bad.is_err() && bad.unwrap_err_unchecked() == fsm_errc::not_a_state && rebuilt.current() == order_state::shipped;

    // Every region as a number and a name, for persistence (fsm/persist.hpp writes these)
    const auto snap = rebuilt.snapshot();
    std::printf("  snapshot: %lld %.*s\n", static_cast<long long>(snap[0].value), static_cast<int>(snap[0].name.size()),
                snap[0].name.data());
    ok &= snap[0].value == 2 && snap[0].name == "shipped";
    sm<order> copy;
    ok &= copy.restore_snapshot(snap).is_ok() && copy.current() == order_state::shipped;

    // The compile-time checks, as values
    using diagnosis = table_diagnostics<broken>;
    static_assert(!diagnosis::ok);
    static_assert(diagnosis::shadowed_row == 1);                     // the second "a" + ship row never runs
    static_assert(diagnosis::state_names[diagnosis::unreachable_state] == "orphan");
    static_assert(table_diagnostics<order>::ok);
    std::printf("  broken: row %zu never runs; state \"%s\" is unreachable\n", diagnosis::shadowed_row,
                diagnosis::state_names[diagnosis::unreachable_state].data());

    // Everything is constexpr: the machine runs at compile time
    static_assert([] {
        sm<order> m;
        static_cast<void>(m.process_event(pay{2000}));
        static_cast<void>(m.process_event(ship{}));
        return m.current() == order_state::shipped;
    }());

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
