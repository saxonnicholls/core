//
//  state_machine_event_sourcing_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/state_machine.hpp> with <snicholls/core/functional/>: event sourcing. An account is
//  never stored, only its events; its state is replayed by folding the events over the lifecycle machine and the
//  balance together. The law that makes snapshots safe - a snapshot taken anywhere, restored and replayed with the
//  rest, gives the same account as replaying everything - is checked at compile time for every split point. The
//  snapshot then goes through fsm/persist.hpp and a key-value store. Exits non-zero if a result is not the one
//  documented.
//

#include <snicholls/core/fsm/persist.hpp>
#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/functional/fold.hpp>
#include <snicholls/core/functional/match.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <variant>

namespace {

namespace fn = snicholls::fn;

//! [events]
enum class account_state : std::uint8_t { open, frozen, closed };
struct deposit {
    long cents = 0;
};
struct withdraw {
    long cents = 0;
};
struct freeze {};
struct unfreeze {};
struct close {};
using account_event = std::variant<deposit, withdraw, freeze, unfreeze, close>;

struct lifecycle {
    constexpr auto operator()() const
    {
        using namespace snicholls::fsm;
        constexpr auto open = enum_state<account_state::open>;
        constexpr auto frozen = enum_state<account_state::frozen>;
        constexpr auto closed = enum_state<account_state::closed>;
        return make_transition_table(
            *open   + event<deposit>,                // internal: the state stays, the event is taken
             open   + event<withdraw>,
             open   + event<freeze>   = frozen,
             frozen + event<unfreeze> = open,
             open   + event<close>    = closed);
    }
};
using machine = snicholls::fsm::sm<lifecycle>;

// The account: the lifecycle and the money. An event the machine refuses changes nothing; an overdraft is refused.
struct account {
    machine life{};
    long balance = 0;
    int refused = 0;
    friend constexpr bool operator==(const account& a, const account& b)
    {
        return a.life.current() == b.life.current() && a.balance == b.balance && a.refused == b.refused;
    }
};

constexpr account apply(account a, const account_event& e)
{
    const long delta = fn::match(e, [](const deposit& d) { return d.cents; }, [](const withdraw& w) { return -w.cents; },
                                 fn::otherwise([](const auto&) { return 0L; }));
    const bool affordable = a.balance + delta >= 0;
    const bool taken = affordable && fn::match(e, [&](const auto& ev) { return a.life.process_event(ev); });
    if (taken) a.balance += delta;
    else ++a.refused;
    return a;
}

// Replay: the fold of apply over the events, from a given starting account.
constexpr account replay(account from, std::span<const account_event> events) { return fn::foldl_of(apply, from, events); }
//! [events]

constexpr std::array<account_event, 9> history{deposit{10'000}, withdraw{2'500}, freeze{},   withdraw{100}, unfreeze{},
                                       withdraw{9'000}, deposit{1'000},  withdraw{8'000}, close{}};

//! [law]
// A snapshot is the lifecycle state and the money; resuming restores the machine (no action runs) and replays the rest.
struct snapshot {
    account_state state;
    long balance;
    int refused;
};
constexpr snapshot take(const account& a) { return {a.life.current(), a.balance, a.refused}; }
constexpr account resume(const snapshot& s)
{
    account a{machine{}, s.balance, s.refused};
    static_cast<void>(a.life.restore(s.state));
    return a;
}

// The law, for every split point k: replay(all) == replay(resume(take(replay(first k))), rest).
constexpr bool snapshots_are_safe()
{
    const account whole = replay(account{}, history);
    for (std::size_t k = 0; k <= history.size(); ++k) {
        const std::span<const account_event> all{history};
        const account resumed = replay(resume(take(replay(account{}, all.first(k)))), all.subspan(k));
        if (!(resumed == whole)) return false;
    }
    return true;
}
static_assert(snapshots_are_safe());
//! [law]

constexpr account final_account = replay(account{}, history);
static_assert(final_account.life.current() == account_state::closed);
static_assert(final_account.balance == 10'000 - 2'500 + 1'000 - 8'000);   // 500: the 9,000 overdraft was refused
static_assert(final_account.refused == 2);                                 // the withdrawal while frozen, the overdraft

} // namespace

int main()
{
    std::printf("snicholls-core fsm: event sourcing, replay as a fold\n");
    bool ok = true;

    std::printf("  replayed: %s, balance %ld, %d events refused\n",
                final_account.life.current() == account_state::closed ? "closed" : "not closed", final_account.balance,
                final_account.refused);
    ok &= final_account.balance == 500 && final_account.refused == 2;

    //! [persist]
    // Halfway through, the machine is saved through fsm/persist.hpp; another process loads it and replays the rest.
    snicholls::reference_key_value_store store;
    const std::span<const account_event> all{history};
    const account halfway = replay(account{}, all.first(4));
    ok &= snicholls::fsm::save(store, "account/42", halfway.life, 0).is_ok();

    account reloaded{machine{}, halfway.balance, halfway.refused};   // the money travels with the snapshot
    ok &= snicholls::fsm::load(store, "account/42", reloaded.life).is_ok();
    ok &= reloaded.life.current() == account_state::frozen;
    ok &= replay(reloaded, all.subspan(4)) == final_account;
    //! [persist]
    std::printf("  saved frozen after four events, loaded, replayed the rest: the same account\n");

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
