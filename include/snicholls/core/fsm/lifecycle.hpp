//
//  lifecycle.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `fsm::lifecycle_state`: the provider-neutral lifecycle of a payment (requested, pending, confirmed,
/// settled, failed, canceled, expired, reversed), its presentation, and its reference machine.
/// \ingroup core
///
/// \overview Every money movement has a state attached to it, and every provider names its states differently: a card
/// network's authorised and captured, a bank's accepted and booked, a chain's in the mempool and six blocks deep. One
/// vocabulary above them lets notifications, reporting and ageing speak one language across products; each
/// provider's own machine maps its states to these.
///
/// - `requested`: created by us, not yet sent to the provider;
/// - `pending`: the provider has it and has not answered;
/// - `confirmed`: the provider accepted it; funds are committed but not yet in the account;
/// - `settled`: the funds moved;
/// - `failed`: the provider refused it, or it broke;
/// - `canceled`: withdrawn by us or the customer before it was confirmed;
/// - `expired`: no answer within its validity;
/// - `reversed`: undone after confirmation or settlement (a refund, a chargeback, a chain reorganisation).
///
/// `lifecycle` is the reference machine over it, `pending -> confirmed -> settled` with the failures and the reversal:
///
/// \snippet fsm/lifecycle_demo.cpp table
///
/// Its state is the enumerator, so the machine is rebuilt from the byte stored on the transaction:
///
/// \snippet fsm/lifecycle_demo.cpp restore
///
/// Use `lifecycle_state` on any object that moves money, and `lifecycle` as is or as the model for a provider's own
/// table (its events are the provider's webhooks). Not for the ledger's journal rows, which hold no state: a
/// transition that moves money posts a fact, and the state is the fold of the facts.
///
/// \relations `fsm/state_machine.hpp` (the machine), `fsm/persist.hpp` (save and load the state),
/// `fsm/diagram.hpp` (the table drawn), `meta/attributes.hpp` and `display.hpp` (the presentation: `colour`, `label`,
/// `tooltip`). Group: \ref core.
///
/// \impls One enumeration and one reference table. Providers' tables live with the providers' adapters, above core.
///
/// \contracts The enumerators are stable: their values never change and new ones are appended, so a stored value
/// keeps its meaning (`fsm/persist.hpp` stores the value). Colours, as core's presentation rules say: green for
/// settled (the money arrived), red for failed, grey while nothing has happened (requested, pending, canceled), blue
/// for confirmed, yellow for expired, magenta for reversed. `is_final` is `constexpr` and `noexcept`.
///
/// \errors None here; the machine's errors are `fsm_errc`.
///
/// \examples \snippet fsm/lifecycle_demo.cpp attributes
/// The payment demo (`fsm/lifecycle_payment_demo.cpp`) saves and loads the state through the reference key-value store,
/// with a simulated crash and a stale writer; the chain demo (`fsm/lifecycle_chain_transaction_demo.cpp`) confirms a
/// chain transaction after a number of blocks set by a dependency.
///
/// \perf `benchmarks/fsm/state_machine_benchmark.cpp` runs the lifecycle against the same machine written by hand:
/// 7.56 ns per random event against 7.49 (macOS x86-64, Apple Clang 17, Release, 2026-11-24, load 9); the same
/// instructions. `current()` is the stored byte, since the states are numbered by their enumerators.
///
/// \testing `tests/fsm/lifecycle_test.cpp`; `demos/fsm/lifecycle_demo.cpp`, `demos/fsm/lifecycle_payment_demo.cpp`,
/// `demos/fsm/lifecycle_chain_transaction_demo.cpp`.
///
/// \deps `<array>`, `<cstdint>`, and core's `fsm/state_machine.hpp`, `colour.hpp`, `display.hpp`,
/// `meta/attributes.hpp`.
///
/// \gotchas `reversed` follows `confirmed` or `settled`; a payment that failed is not reversed, it failed.
///
/// \status Experimental, new in 0.24.0.

#ifndef snicholls_core_fsm_lifecycle_hpp
#define snicholls_core_fsm_lifecycle_hpp

#include <snicholls/core/colour.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/meta/attributes.hpp>

#include <array>
#include <cstdint>

namespace snicholls::fsm {

    /// \brief Where a money movement stands, whatever the provider. Values are stable; new ones are appended.
    /// \ingroup core
    enum class lifecycle_state : std::uint8_t {
        requested,   ///< created by us, not yet with the provider
        pending,     ///< with the provider, no answer yet
        confirmed,   ///< accepted by the provider; the funds are committed, not yet moved
        settled,     ///< the funds moved
        failed,      ///< refused by the provider, or broken
        canceled,    ///< withdrawn before confirmation
        expired,     ///< no answer within its validity
        reversed     ///< undone after confirmation or settlement: a refund, a chargeback, a reorganisation
    };

    /// \brief The presentation of each lifecycle state: its colour, label and tooltip.
    /// \ingroup core
    /// \param s the state.
    /// \return its attribute set: `colour`, `label`, `tooltip`.
    constexpr const auto& attributes(lifecycle_state s) noexcept
    {
        return per_enumerator(s, [] {
            return std::array{
                attrs(colours::grey, label{"Requested"}, tooltip{"Created, not yet sent to the provider"}),
                attrs(colours::grey, label{"Pending"}, tooltip{"With the provider, no answer yet"}),
                attrs(colours::blue, label{"Confirmed"}, tooltip{"Accepted by the provider; funds committed, not yet moved"}),
                attrs(colours::green, label{"Settled"}, tooltip{"The funds moved"}),
                attrs(colours::red, label{"Failed"}, tooltip{"Refused by the provider, or broken"}),
                attrs(colours::grey, label{"Canceled"}, tooltip{"Withdrawn before confirmation"}),
                attrs(colours::yellow, label{"Expired"}, tooltip{"No answer within its validity"}),
                attrs(colours::magenta, label{"Reversed"}, tooltip{"Undone after confirmation or settlement"}),
            };
        });
    }

    /// \brief Whether nothing more happens to a movement in this state: failed, canceled, expired, reversed. A settled
    /// one can still be reversed.
    /// \ingroup core
    /// \param s the state.
    /// \return true for the final states.
    [[nodiscard]] constexpr bool is_final(lifecycle_state s) noexcept
    {
        return s == lifecycle_state::failed || s == lifecycle_state::canceled || s == lifecycle_state::expired ||
            s == lifecycle_state::reversed;
    }

    /// \brief The events of the reference lifecycle: what a provider's webhook or a timer reports.
    /// \ingroup core
    namespace lifecycle_event {
        /// \brief The movement was sent to the provider.
        struct submit {};
        /// \brief The provider accepted it.
        struct confirm {};
        /// \brief The funds moved.
        struct settle {};
        /// \brief The provider refused it, or it broke.
        struct fail {};
        /// \brief It was withdrawn.
        struct cancel {};
        /// \brief Its validity ran out.
        struct expire {};
        /// \brief It was undone: a refund, a chargeback, a reorganisation.
        struct reverse {};
    } // namespace lifecycle_event

    /// \brief The reference lifecycle machine: `requested -> pending -> confirmed -> settled`, failing, canceling or
    /// expiring on the way, and reversed after confirmation or settlement.
    /// \ingroup core
    struct lifecycle {
        /// \brief The machine's states are these enumerators: `current()` and `restore()` speak them.
        using state_enum = lifecycle_state;

        /// \brief The table.
        /// \return the transition table.
        constexpr auto operator()() const noexcept
        {
            using namespace lifecycle_event;
            constexpr auto requested = enum_state<lifecycle_state::requested>;
            constexpr auto pending = enum_state<lifecycle_state::pending>;
            constexpr auto confirmed = enum_state<lifecycle_state::confirmed>;
            constexpr auto settled = enum_state<lifecycle_state::settled>;
            constexpr auto failed = enum_state<lifecycle_state::failed>;
            constexpr auto canceled = enum_state<lifecycle_state::canceled>;
            constexpr auto expired = enum_state<lifecycle_state::expired>;
            constexpr auto reversed = enum_state<lifecycle_state::reversed>;
            return make_transition_table(
                *requested + event<submit>  = pending,
                 requested + event<cancel>  = canceled,
                 requested + event<fail>    = failed,
                 pending   + event<confirm> = confirmed,
                 pending   + event<fail>    = failed,
                 pending   + event<cancel>  = canceled,
                 pending   + event<expire>  = expired,
                 confirmed + event<settle>  = settled,
                 confirmed + event<fail>    = failed,
                 confirmed + event<reverse> = reversed,
                 settled   + event<reverse> = reversed);
        }
    };

} // namespace snicholls::fsm

#endif /* snicholls_core_fsm_lifecycle_hpp */
