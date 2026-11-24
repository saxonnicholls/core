//
//  lifecycle_chain_transaction_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  A chain transaction on the provider-neutral lifecycle (fsm/lifecycle.hpp): broadcast, seen in the mempool, then
//  confirmed only once a block reports enough confirmations, a number the machine is given as a dependency. Fewer
//  confirmations are an internal transition that records the depth; a reorganisation before confirmation returns it
//  to pending, after it reverses it; a dropped transaction fails. Exits non-zero if a state is not the one documented.
//

#include <snicholls/core/fsm/lifecycle.hpp>
#include <snicholls/core/meta/enums.hpp>

#include <cstdint>
#include <cstdio>
#include <string_view>

namespace {

//! [chain]
struct broadcast {};
struct seen_in_mempool {};
struct block {
    std::uint32_t confirmations = 0;
};
struct reorg {};
struct dropped {};

struct policy {
    std::uint32_t required = 6;   // confirmations before the funds count
    std::uint32_t depth = 0;      // the deepest seen so far
};

struct chain_transaction {
    using state_enum = snicholls::fsm::lifecycle_state;

    auto operator()() const
    {
        using namespace snicholls::fsm;
        using s = lifecycle_state;
        const auto deep_enough = [](const block& b, const policy& p) { return b.confirmations >= p.required; };
        const auto record = [](const block& b, policy& p) { p.depth = b.confirmations; };
        const auto forget = [](policy& p) { p.depth = 0; };
        return make_transition_table(
            *enum_state<s::requested> + event<broadcast>                       = enum_state<s::pending>,
             enum_state<s::pending>   + event<block> [deep_enough] / record    = enum_state<s::confirmed>,
             enum_state<s::pending>   + event<block> / record,                // not deep enough yet: stay
             enum_state<s::pending>   + event<seen_in_mempool>,               // nothing to do
             enum_state<s::pending>   + event<reorg> / forget,                // its block was orphaned: count again
             enum_state<s::pending>   + event<dropped>                         = enum_state<s::failed>,
             enum_state<s::confirmed> + event<block> / record,                // deeper still
             enum_state<s::confirmed> + event<reorg>                           = enum_state<s::reversed>);
    }
};
//! [chain]

std::string_view name(snicholls::fsm::lifecycle_state s) { return snicholls::enum_name(s); }

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm: a chain transaction\n");
    bool ok = true;

    policy six{.required = 6};
    sm<chain_transaction> tx{six};
    ok &= tx.process_event(broadcast{}) && tx.current() == lifecycle_state::pending;
    ok &= tx.process_event(seen_in_mempool{});
    ok &= tx.process_event(block{2}) && tx.process_event(reorg{}) && six.depth == 0;   // orphaned: still pending
    for (std::uint32_t c = 1; c <= 7; ++c) {
        ok &= tx.process_event(block{c});
        std::printf("  block with %u confirmation%s: %.*s\n", c, c == 1 ? "" : "s", static_cast<int>(name(tx.current()).size()),
                    name(tx.current()).data());
        ok &= tx.current() == (c < 6 ? lifecycle_state::pending : lifecycle_state::confirmed);
    }
    ok &= six.depth == 7;

    policy one{.required = 1};
    sm<chain_transaction> fast{one};   // the same table with a different policy
    ok &= fast.process_event(broadcast{}) && fast.process_event(block{1}) && fast.current() == lifecycle_state::confirmed;
    ok &= fast.process_event(reorg{}) && fast.current() == lifecycle_state::reversed;

    policy any{};
    sm<chain_transaction> lost{any};
    ok &= lost.process_event(broadcast{}) && lost.process_event(dropped{}) && lost.current() == lifecycle_state::failed;
    ok &= !lost.process_event(block{9});   // nothing leaves failed

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
