//
//  state_machine_mealy_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/state_machine.hpp> with <snicholls/core/functional/state.hpp>: a Mealy machine (its
//  output depends on the state and the input) written twice, as a transition table whose actions emit the output,
//  and as a program in the State monad, `input -> State<gate, output>`. The compiler then runs both on every input
//  sequence of length 8 and proves they emit the same outputs. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/functional/fold.hpp>
#include <snicholls/core/functional/state.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace {

namespace fn = snicholls::fn;

enum class gate : std::uint8_t { locked, unlocked };
enum class output : std::uint8_t { unlock, refund, lock, alarm };
enum class input : std::uint8_t { coin, push };

// Where the machine's actions write: a fixed tape, so the whole run is constexpr.
struct tape {
    std::array<output, 8> out{};
    std::size_t size = 0;
    constexpr void emit(output o) { out[size++] = o; }
};

//! [table]
// The table form: each row's action emits the output; the tape is a dependency, matched by type.
struct coin {};
struct push {};
struct turnstile {
    constexpr auto operator()() const
    {
        using namespace snicholls::fsm;
        constexpr auto locked = enum_state<gate::locked>;
        constexpr auto unlocked = enum_state<gate::unlocked>;
        const auto emits = [](output o) { return [o](tape& t) { t.emit(o); }; };
        return make_transition_table(
            *locked   + event<coin> / emits(output::unlock) = unlocked,
             locked   + event<push> / emits(output::alarm),            // internal: stays locked
             unlocked + event<coin> / emits(output::refund),
             unlocked + event<push> / emits(output::lock)   = locked);
    }
};
//! [table]

//! [monad]
// The State form: one input is a State program over the gate whose value is the output. get, decide, put, return.
constexpr auto step(input i)
{
    return fn::mbind(fn::get_state, [i](gate g) {
        const bool is_coin = i == input::coin;
        const bool is_locked = g == gate::locked;
        const output o = is_coin ? (is_locked ? output::unlock : output::refund) : (is_locked ? output::alarm : output::lock);
        const gate next = is_coin ? gate::unlocked : gate::locked;
        return fn::mthen(fn::put_state(next), fn::pure<fn::state<>>(o));
    });
}

// Two steps composed with mbind: still one State program, run once.
constexpr auto coin_then_push =
    fn::mbind(step(input::coin), [](output a) { return fn::fmap([a](output b) { return std::pair{a, b}; }, step(input::push)); });
static_assert(fn::eval_state(coin_then_push, gate::locked) == std::pair{output::unlock, output::lock});
static_assert(fn::exec_state(coin_then_push, gate::locked) == gate::locked);
static_assert(fn::eval_state(coin_then_push, gate::unlocked) == std::pair{output::refund, output::lock});
//! [monad]

constexpr std::array<input, 8> inputs_of(unsigned bits)
{
    std::array<input, 8> in{};
    for (std::size_t k = 0; k < in.size(); ++k) in[k] = (bits >> k) & 1u ? input::push : input::coin;
    return in;
}

constexpr tape run_table(const std::array<input, 8>& in)
{
    tape t;
    snicholls::fsm::sm<turnstile> m{t};
    for (input i : in) static_cast<void>(i == input::coin ? m.process_event(coin{}) : m.process_event(push{}));
    return t;
}

// The State form over the whole sequence: a fold that runs each step's program from the gate the last one left.
constexpr tape run_monad(const std::array<input, 8>& in)
{
    const auto run = [](std::pair<gate, tape> acc, input i) {
        const auto [o, next] = fn::run_state(step(i), acc.first);
        acc.second.emit(o);
        return std::pair{next, acc.second};
    };
    return fn::foldl_of(run, std::pair{gate::locked, tape{}}, in).second;
}

//! [proof]
// Every one of the 256 input sequences of length 8: the table and the monad emit the same eight outputs.
constexpr bool same_machine()
{
    for (unsigned bits = 0; bits < 256; ++bits) {
        const tape a = run_table(inputs_of(bits));
        const tape b = run_monad(inputs_of(bits));
        if (a.size != 8 || b.size != 8 || a.out != b.out) return false;
    }
    return true;
}
static_assert(same_machine());
//! [proof]

const char* name(output o)
{
    constexpr std::array<const char*, 4> names{"unlock", "refund", "lock", "alarm"};
    return names[static_cast<std::size_t>(o)];
}

} // namespace

int main()
{
    std::printf("snicholls-core fsm: a Mealy machine as a table and as the State monad\n");
    bool ok = same_machine();

    const auto day = inputs_of(0b1001'0110u);   // coin push push coin push coin coin push
    const tape a = run_table(day);
    const tape b = run_monad(day);
    for (std::size_t k = 0; k < a.size; ++k) std::printf("  %-6s %-6s\n", name(a.out[k]), name(b.out[k]));
    ok &= a.out == b.out;
    ok &= a.out == std::array{output::unlock, output::lock, output::alarm, output::unlock, output::lock, output::unlock,
                              output::refund, output::lock};

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
