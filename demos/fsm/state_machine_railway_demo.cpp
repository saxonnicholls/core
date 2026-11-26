//
//  state_machine_railway_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/state_machine.hpp> with <snicholls/core/functional/>: railway-oriented checkout. A
//  payment machine's guards are built from functions rather than written as lambdas (fn::pointwise predicates,
//  fn::compose, fn::curry), and a checkout is one chain of Rust-style steps: an optional card becomes a result
//  (fn::rust::option::ok_or), each step runs only if the last succeeded (result::and_then), and the machine's own
//  refusals (try_process_event) join the same track through map_err. The first failure is the answer and nothing
//  after it runs. All of it is evaluated by the compiler. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/functional/combinators.hpp>
#include <snicholls/core/functional/curry.hpp>
#include <snicholls/core/functional/pointwise.hpp>
#include <snicholls/core/functional/rust.hpp>

#include <cstdint>
#include <cstdio>
#include <optional>

namespace {

namespace fn = snicholls::fn;
namespace option = snicholls::fn::rust::option;

enum class payment : std::uint8_t { created, authorised, captured, refunded };
struct authorise {
    long cents = 0;
};
struct capture {};
struct refund {
    long cents = 0;
};

//! [guards]
// Guards as values, built from pieces. A closure from fn goes into a table wrapped in fn::pointwise, and the pieces
// are combined here, outside the table: inside it, &&, || and ! are the table's own guard algebra, and combining two
// pointwise guards there would be ambiguous between the two.
constexpr bool at_most(long limit, long cents) { return cents <= limit; }
constexpr fn::pointwise amount{[](const authorise& a) { return a.cents; }};
constexpr auto authorisable = (amount > 0L) && (amount <= 50'000L);                                   // a predicate
constexpr fn::pointwise refundable{fn::compose(fn::curry(at_most)(20'000L), &refund::cents)};       // curried, composed
static_assert(authorisable(authorise{1'000}) && !authorisable(authorise{0}) && !authorisable(authorise{90'000}));
static_assert(refundable(refund{5'000}) && !refundable(refund{25'000}));

struct card_payment {
    constexpr auto operator()() const
    {
        using namespace snicholls::fsm;
        constexpr auto created = enum_state<payment::created>;
        constexpr auto authorised = enum_state<payment::authorised>;
        constexpr auto captured = enum_state<payment::captured>;
        constexpr auto refunded = enum_state<payment::refunded>;
        return make_transition_table(
            *created    + event<authorise> [authorisable] = authorised,
             authorised + event<capture>                  = captured,
             captured   + event<refund> [refundable]      = refunded);
    }
};
//! [guards]

enum class checkout_error : std::uint8_t { no_card, expired, declined, out_of_order };

struct card {
    int expiry_year = 0;
};

constexpr checkout_error from_machine(snicholls::fsm::fsm_errc e)
{
    return e == snicholls::fsm::fsm_errc::guard_rejected ? checkout_error::declined : checkout_error::out_of_order;
}

struct outcome {
    snicholls::result<void, checkout_error> result;
    payment state;
    int steps;   // how many steps ran: the chain stops at the first failure
};

//! [railway]
// One track: card -> valid card -> authorised -> captured. Each and_then runs only on success.
constexpr outcome checkout(std::optional<card> c, long cents, int this_year)
{
    snicholls::fsm::sm<card_payment> m;
    int steps = 0;
    auto r = option::ok_or(c, checkout_error::no_card)
                 .and_then([&](card k) -> snicholls::result<void, checkout_error> {
                     ++steps;
                     if (k.expiry_year < this_year) return snicholls::err(checkout_error::expired);
                     return snicholls::ok();
                 })
                 .and_then([&] {
                     ++steps;
                     return m.try_process_event(authorise{cents}).map_err(from_machine);
                 })
                 .and_then([&] {
                     ++steps;
                     return m.try_process_event(capture{}).map_err(from_machine);
                 });
    return {r, m.current(), steps};
}
//! [railway]

constexpr outcome paid = checkout(card{2030}, 1'999, 2026);
static_assert(paid.result.is_ok() && paid.state == payment::captured && paid.steps == 3);

constexpr outcome no_card = checkout(std::nullopt, 1'999, 2026);
static_assert(no_card.result.unwrap_err_unchecked() == checkout_error::no_card && no_card.steps == 0);

constexpr outcome expired = checkout(card{2020}, 1'999, 2026);
static_assert(expired.result.unwrap_err_unchecked() == checkout_error::expired && expired.steps == 1);
static_assert(expired.state == payment::created);   // the machine was never asked

constexpr outcome too_large = checkout(card{2030}, 90'000, 2026);
static_assert(too_large.result.unwrap_err_unchecked() == checkout_error::declined && too_large.steps == 2);

// Refunds after capture, through the curried guard; a second refund is out of order.
constexpr bool refunds_once()
{
    snicholls::fsm::sm<card_payment> m;
    const bool through = m.process_event(authorise{1'000}) && m.process_event(capture{});
    const bool too_much = m.try_process_event(refund{30'000}).unwrap_err_unchecked() == snicholls::fsm::fsm_errc::guard_rejected;
    const bool refunded = m.try_process_event(refund{1'000}).is_ok();
    const bool again = m.try_process_event(refund{1'000}).unwrap_err_unchecked() == snicholls::fsm::fsm_errc::no_transition;
    return through && too_much && refunded && again && m.current() == payment::refunded;
}
static_assert(refunds_once());

const char* name(checkout_error e)
{
    switch (e) {
    case checkout_error::no_card: return "no card";
    case checkout_error::expired: return "card expired";
    case checkout_error::declined: return "declined";
    case checkout_error::out_of_order: return "out of order";
    }
    return "?";
}

void show(const char* what, const outcome& o)
{
    if (o.result.is_ok()) std::printf("  %-22s captured after %d steps\n", what, o.steps);
    else std::printf("  %-22s %s after %d steps\n", what, name(o.result.unwrap_err_unchecked()), o.steps);
}

} // namespace

int main()
{
    std::printf("snicholls-core fsm: railway checkout over a payment machine\n");
    show("a good card", paid);
    show("no card", no_card);
    show("an expired card", expired);
    show("over the limit", too_large);

    // The same chain at run time, from input the compiler cannot see.
    volatile long cents = 2'500;
    const outcome live = checkout(card{2031}, cents, 2026);
    show("at run time", live);
    const bool ok = live.result.is_ok() && live.state == payment::captured && refunds_once();

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
