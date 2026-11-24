//
//  state_machine_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  fsm::sm against a hand-written switch over an enumeration, the same transitions and the same event sequence:
//  sml's hello_world (four states and X) and the reference payment lifecycle (eight states, seven events). The events
//  are 4,096 codes drawn once from a fixed seed, valid and invalid mixed, so that neither side can fold the run away;
//  both sides decode a code by the same switch into one handler per event (process_event<E> on one side, a function
//  per event over the state on the other, as such code is written by hand). A final state is reset to the start, by
//  restore() or a fresh machine on one side and by assignment on the other. main() runs a fairness check first and refuses to benchmark if the two sides do not take
//  the same number of events and end in the same state (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/fsm/lifecycle.hpp>
#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

using namespace snicholls::fsm;

constexpr std::size_t n_events = 4096;

template <std::uint64_t Kinds>
std::array<std::uint8_t, n_events> make_codes(std::uint64_t seed)
{
    snicholls::stats::wyrand rng(seed);
    std::array<std::uint8_t, n_events> out{};
    for (auto& c : out)
        c = static_cast<std::uint8_t>(snicholls::stats::uniform_index(rng, Kinds));
    return out;
}

// ---- hello_world ----------------------------------------------------------------------------------------------

struct release {};
struct ack {};
struct fin {};
struct timeout {};

struct hello_world {
    constexpr auto operator()() const noexcept
    {
        const auto valid = [] { return true; };
        return make_transition_table(
            *"established"_s + event<release> = "fin wait 1"_s,
             "fin wait 1"_s + event<ack> [valid] = "fin wait 2"_s,
             "fin wait 2"_s + event<fin> [valid] = "timed wait"_s,
             "timed wait"_s + event<timeout> = X);
    }
};

enum class tcp : std::uint8_t { established, fin_wait_1, fin_wait_2, timed_wait, closed };

const auto hello_codes = make_codes<4>(0x243f6a8885a308d3ULL);

struct tally {
    std::size_t taken = 0;
    std::uint8_t last = 0;
};

tally hello_fsm()
{
    sm<hello_world> m;
    tally t;
    for (const auto code : hello_codes) {
        bool taken = false;
        switch (code) {
        case 0: taken = m.process_event(release{}); break;
        case 1: taken = m.process_event(ack{}); break;
        case 2: taken = m.process_event(fin{}); break;
        default: taken = m.process_event(timeout{}); break;
        }
        t.taken += taken ? 1 : 0;
        if (m.is(X))
            m = sm<hello_world>{};   // start again: one byte
    }
    t.last = static_cast<std::uint8_t>(m.snapshot()[0].value);
    return t;
}

// The hand-written machine as it is usually written: one handler per event, a switch (or an if) over the state.
bool on_release(tcp& s) { return s == tcp::established ? (s = tcp::fin_wait_1, true) : false; }
bool on_ack(tcp& s) { return s == tcp::fin_wait_1 ? (s = tcp::fin_wait_2, true) : false; }
bool on_fin(tcp& s) { return s == tcp::fin_wait_2 ? (s = tcp::timed_wait, true) : false; }
bool on_timeout(tcp& s) { return s == tcp::timed_wait ? (s = tcp::closed, true) : false; }

tally hello_switch()
{
    tcp s = tcp::established;
    tally t;
    for (const auto code : hello_codes) {
        bool taken = false;
        switch (code) {
        case 0: taken = on_release(s); break;
        case 1: taken = on_ack(s); break;
        case 2: taken = on_fin(s); break;
        default: taken = on_timeout(s); break;
        }
        t.taken += taken ? 1 : 0;
        if (s == tcp::closed)
            s = tcp::established;
    }
    t.last = static_cast<std::uint8_t>(s);   // the fsm's state index: established 0, fin wait 1 1, ...
    return t;
}

// ---- the payment lifecycle ------------------------------------------------------------------------------------

const auto payment_codes = make_codes<7>(0x13198a2e03707344ULL);

tally payment_fsm()
{
    using namespace lifecycle_event;
    sm<lifecycle> m;
    tally t;
    for (const auto code : payment_codes) {
        bool taken = false;
        switch (code) {
        case 0: taken = m.process_event(submit{}); break;
        case 1: taken = m.process_event(confirm{}); break;
        case 2: taken = m.process_event(settle{}); break;
        case 3: taken = m.process_event(fail{}); break;
        case 4: taken = m.process_event(cancel{}); break;
        case 5: taken = m.process_event(expire{}); break;
        default: taken = m.process_event(reverse{}); break;
        }
        t.taken += taken ? 1 : 0;
        if (is_final(m.current()))
            static_cast<void>(m.restore(lifecycle_state::requested));
    }
    t.last = static_cast<std::uint8_t>(m.current());
    return t;
}

using ls = lifecycle_state;

bool on_submit(ls& s) { return s == ls::requested ? (s = ls::pending, true) : false; }
bool on_confirm(ls& s) { return s == ls::pending ? (s = ls::confirmed, true) : false; }
bool on_settle(ls& s) { return s == ls::confirmed ? (s = ls::settled, true) : false; }
bool on_fail(ls& s)
{
    switch (s) {
    case ls::requested:
    case ls::pending:
    case ls::confirmed: s = ls::failed; return true;
    default: return false;
    }
}
bool on_cancel(ls& s)
{
    switch (s) {
    case ls::requested:
    case ls::pending: s = ls::canceled; return true;
    default: return false;
    }
}
bool on_expire(ls& s) { return s == ls::pending ? (s = ls::expired, true) : false; }
bool on_reverse(ls& s)
{
    switch (s) {
    case ls::confirmed:
    case ls::settled: s = ls::reversed; return true;
    default: return false;
    }
}

tally payment_switch()
{
    ls state = ls::requested;
    tally t;
    for (const auto code : payment_codes) {
        bool taken = false;
        switch (code) {
        case 0: taken = on_submit(state); break;
        case 1: taken = on_confirm(state); break;
        case 2: taken = on_settle(state); break;
        case 3: taken = on_fail(state); break;
        case 4: taken = on_cancel(state); break;
        case 5: taken = on_expire(state); break;
        default: taken = on_reverse(state); break;
        }
        t.taken += taken ? 1 : 0;
        if (is_final(state))
            state = ls::requested;
    }
    t.last = static_cast<std::uint8_t>(state);
    return t;
}

void BM_FsmHelloWorld_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(hello_fsm());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_events));
}
BENCHMARK(BM_FsmHelloWorld_Core);

void BM_FsmHelloWorld_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(hello_switch());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_events));
}
BENCHMARK(BM_FsmHelloWorld_HandWritten);

void BM_FsmPaymentLifecycle_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(payment_fsm());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_events));
}
BENCHMARK(BM_FsmPaymentLifecycle_Core);

void BM_FsmPaymentLifecycle_HandWritten(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(payment_switch());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n_events));
}
BENCHMARK(BM_FsmPaymentLifecycle_HandWritten);

} // namespace

int main(int argc, char** argv)
{
    const tally hf = hello_fsm();
    const tally hs = hello_switch();
    const tally pf = payment_fsm();
    const tally ps = payment_switch();
    if (hf.taken != hs.taken || hf.last != hs.last || pf.taken != ps.taken || pf.last != ps.last || hf.taken == 0 || pf.taken == 0) {
        std::fprintf(stderr,
                     "fsm benchmark: fairness check FAILED - hello_world %zu/%u vs %zu/%u, lifecycle %zu/%u vs %zu/%u; "
                     "not benchmarking\n",
                     hf.taken, hf.last, hs.taken, hs.last, pf.taken, pf.last, ps.taken, ps.last);
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
