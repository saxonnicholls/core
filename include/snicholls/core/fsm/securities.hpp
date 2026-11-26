//
//  securities.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The reference lifecycles of securities: a trade, a fund application or redemption, a corporate action and a
/// repo, each a state enumeration with its presentation and a machine in sml's notation.
/// \ingroup core
///
/// \overview A `security_fact` (`concepts/security_provider.hpp`) is an immutable event; the state of the object it is
/// about (an order, a dealing request, an action, a repo) is the fold of its facts through one of these tables, kept
/// as the enumerator on the object, so `fsm/persist.hpp` saves and loads it and the machine is rebuilt after a crash.
///
/// | Machine | States | Facts that drive it |
/// |---|---|---|
/// | `trade_lifecycle` | placed, filled, settling, settled, failed; cancelled before a fill | `buy` / `sell` (fill), the settlement instruction, `settle`, `settle_failed` |
/// | `dealing_lifecycle` | requested, priced, issued, settled; cancelled before issue | an `application` or `redemption` request, the dealing price, the units issued or redeemed on the register, the cash settled |
/// | `corporate_action_lifecycle` | announced, ex, record, paid; withdrawn before the record date | the announcement, the ex date, the record date (`dividend_entitlement`), the payment (`dividend_paid` and friends) |
/// | `repo_lifecycle` | open, accruing, closed | `repo_open` / `reverse_repo_open`, each `interest_accrual`, `repo_close` / `reverse_repo_close` |
///
/// \snippet fsm/securities_demo.cpp trade
///
/// \snippet fsm/securities_demo.cpp persist
///
/// For a redemption, `issued` is the units cancelled on the register at the dealing price: the register has changed,
/// and the cash is still to settle. Use these tables as they are or as the model for a provider's own; never a switch
/// over the states.
///
/// \relations `fsm/state_machine.hpp` (the machine), `fsm/persist.hpp` (save and load), `fsm/diagram.hpp` (the tables
/// drawn), `fsm/lifecycle.hpp` (the payment lifecycle, the model for these), `concepts/security_provider.hpp` (the
/// facts), `meta/attributes.hpp` and `display.hpp` (the presentation). Group: \ref core.
///
/// \impls Four enumerations and four reference tables. A broker's or registry's own states map to them in servers.
///
/// \contracts The enumerators are stable: values never change and new ones are appended, so a stored value keeps its
/// meaning. Every enumerator is a state of its table, so every `restore` of an enumerator succeeds. Colours: grey
/// while nothing has happened or after a cancellation, blue and cyan in progress, green when done (settled, paid,
/// closed), red for a failed settlement. `is_final` is `constexpr` and `noexcept`.
///
/// \errors None here; the machine's errors are `fsm_errc`.
///
/// \examples \snippet fsm/securities_demo.cpp dealing
///
/// \perf Not benchmarked on their own: each is a table of a few rows over a one-byte state, as `fsm/lifecycle.hpp`
/// (`benchmarks/fsm/state_machine_benchmark.cpp`).
///
/// \testing `tests/fsm/securities_test.cpp`; `demos/fsm/securities_demo.cpp`.
///
/// \deps `<array>`, `<cstdint>`, and core's `fsm/state_machine.hpp`, `colour.hpp`, `display.hpp`,
/// `meta/attributes.hpp`.
///
/// \gotchas A settlement that fails is final here: the retry is a new instruction on a new object, as CHESS rolls a
/// failed settlement into the next day's batch. A repo's accruals keep it `accruing` (a self-transition, which runs the
/// state's exit and entry, as sml's does).
///
/// \status Experimental, new in 0.27.0.

#ifndef snicholls_core_fsm_securities_hpp
#define snicholls_core_fsm_securities_hpp

#include <snicholls/core/colour.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/meta/attributes.hpp>

#include <array>
#include <cstdint>

namespace snicholls::fsm {

    /// \brief Where a trade stands. Values are stable; new ones are appended.
    /// \ingroup core
    enum class trade_state : std::uint8_t {
        placed,     ///< the order is with the broker
        filled,     ///< executed: the trade date, a `buy` or `sell` fact
        settling,   ///< the settlement instruction is in
        settled,    ///< the units and the cash moved: a `settle` fact
        failed,     ///< it did not settle on its date: a `settle_failed` fact
        cancelled,  ///< withdrawn before a fill
    };

    /// \brief Where a fund application or redemption stands. Values are stable; new ones are appended.
    /// \ingroup core
    enum class dealing_state : std::uint8_t {
        requested,  ///< received before the cut-off
        priced,     ///< the dealing price is struck
        issued,     ///< the units are issued (or, for a redemption, cancelled) on the register
        cancelled,  ///< withdrawn or rejected before issue
        settled,    ///< the cash settled
    };

    /// \brief Where a corporate action stands. Values are stable; new ones are appended.
    /// \ingroup core
    enum class action_state : std::uint8_t {
        announced,  ///< announced with its terms and dates
        ex,         ///< the ex date passed: the security trades without the entitlement
        record,     ///< the record date passed: the register is entitled
        paid,       ///< paid or allotted
        cancelled,  ///< withdrawn before the record date
    };

    /// \brief Where a repo stands. Values are stable; new ones are appended.
    /// \ingroup core
    enum class repo_state : std::uint8_t {
        open,      ///< opened: the securities and the cash exchanged
        accruing,  ///< interest is accruing
        closed,    ///< repurchased: the cash and the interest returned
    };

    /// \brief The presentation of each trade state.
    /// \ingroup core
    /// \param s the state.
    /// \return its attribute set: `colour`, `label`, `tooltip`.
    constexpr const auto& attributes(trade_state s) noexcept
    {
        return per_enumerator(s, [] {
            return std::array{
                attrs(colours::grey, label{"Placed"}, tooltip{"The order is with the broker"}),
                attrs(colours::blue, label{"Filled"}, tooltip{"Executed; not yet settling"}),
                attrs(colours::cyan, label{"Settling"}, tooltip{"The settlement instruction is in"}),
                attrs(colours::green, label{"Settled"}, tooltip{"The units and the cash moved"}),
                attrs(colours::red, label{"Failed"}, tooltip{"It did not settle on its date"}),
                attrs(colours::grey, label{"Cancelled"}, tooltip{"Withdrawn before a fill"}),
            };
        });
    }

    /// \brief The presentation of each dealing state.
    /// \ingroup core
    /// \param s the state.
    /// \return its attribute set: `colour`, `label`, `tooltip`.
    constexpr const auto& attributes(dealing_state s) noexcept
    {
        return per_enumerator(s, [] {
            return std::array{
                attrs(colours::grey, label{"Requested"}, tooltip{"Received before the cut-off"}),
                attrs(colours::blue, label{"Priced"}, tooltip{"The dealing price is struck"}),
                attrs(colours::cyan, label{"Issued"}, tooltip{"The register has changed; the cash is to settle"}),
                attrs(colours::grey, label{"Cancelled"}, tooltip{"Withdrawn or rejected before issue"}),
                attrs(colours::green, label{"Settled"}, tooltip{"The cash settled"}),
            };
        });
    }

    /// \brief The presentation of each corporate-action state.
    /// \ingroup core
    /// \param s the state.
    /// \return its attribute set: `colour`, `label`, `tooltip`.
    constexpr const auto& attributes(action_state s) noexcept
    {
        return per_enumerator(s, [] {
            return std::array{
                attrs(colours::grey, label{"Announced"}, tooltip{"Announced with its terms and dates"}),
                attrs(colours::blue, label{"Ex"}, tooltip{"Trading without the entitlement"}),
                attrs(colours::cyan, label{"Record"}, tooltip{"The register is entitled"}),
                attrs(colours::green, label{"Paid"}, tooltip{"Paid or allotted"}),
                attrs(colours::grey, label{"Cancelled"}, tooltip{"Withdrawn before the record date"}),
            };
        });
    }

    /// \brief The presentation of each repo state.
    /// \ingroup core
    /// \param s the state.
    /// \return its attribute set: `colour`, `label`, `tooltip`.
    constexpr const auto& attributes(repo_state s) noexcept
    {
        return per_enumerator(s, [] {
            return std::array{
                attrs(colours::blue, label{"Open"}, tooltip{"The securities and the cash exchanged"}),
                attrs(colours::cyan, label{"Accruing"}, tooltip{"Interest is accruing"}),
                attrs(colours::green, label{"Closed"}, tooltip{"Repurchased; the cash and the interest returned"}),
            };
        });
    }

    /// \brief Whether nothing more happens to a trade: settled, failed, cancelled.
    /// \ingroup core
    /// \param s the state.
    /// \return true for the final states.
    [[nodiscard]] constexpr bool is_final(trade_state s) noexcept
    {
        return s == trade_state::settled || s == trade_state::failed || s == trade_state::cancelled;
    }

    /// \brief Whether nothing more happens to a dealing request: cancelled, settled.
    /// \ingroup core
    /// \param s the state.
    /// \return true for the final states.
    [[nodiscard]] constexpr bool is_final(dealing_state s) noexcept
    {
        return s == dealing_state::cancelled || s == dealing_state::settled;
    }

    /// \brief Whether nothing more happens to a corporate action: paid, cancelled.
    /// \ingroup core
    /// \param s the state.
    /// \return true for the final states.
    [[nodiscard]] constexpr bool is_final(action_state s) noexcept
    {
        return s == action_state::paid || s == action_state::cancelled;
    }

    /// \brief Whether nothing more happens to a repo: closed.
    /// \ingroup core
    /// \param s the state.
    /// \return true for the final state.
    [[nodiscard]] constexpr bool is_final(repo_state s) noexcept { return s == repo_state::closed; }

    /// \brief The events of a trade.
    /// \ingroup core
    namespace trade_event {
        /// \brief Executed (a `buy` or `sell` fact).
        struct fill {};
        /// \brief The settlement instruction is in.
        struct instruct {};
        /// \brief It settled (a `settle` fact).
        struct settle {};
        /// \brief It failed to settle (a `settle_failed` fact).
        struct fail {};
        /// \brief The order was withdrawn.
        struct cancel {};
    } // namespace trade_event

    /// \brief The events of a fund application or redemption.
    /// \ingroup core
    namespace dealing_event {
        /// \brief The dealing price is struck.
        struct price {};
        /// \brief The units are issued or cancelled on the register.
        struct issue {};
        /// \brief The cash settled.
        struct settle {};
        /// \brief Withdrawn or rejected.
        struct cancel {};
    } // namespace dealing_event

    /// \brief The events of a corporate action.
    /// \ingroup core
    namespace action_event {
        /// \brief The ex date arrived.
        struct go_ex {};
        /// \brief The record date arrived (a `dividend_entitlement` fact).
        struct record_date {};
        /// \brief Paid or allotted.
        struct pay {};
        /// \brief Withdrawn by the issuer.
        struct withdraw {};
    } // namespace action_event

    /// \brief The events of a repo.
    /// \ingroup core
    namespace repo_event {
        /// \brief Interest accrued for a period (an `interest_accrual` fact).
        struct accrue {};
        /// \brief Repurchased (a `repo_close` or `reverse_repo_close` fact). Not `close`, which POSIX declares globally.
        struct repurchase {};
    } // namespace repo_event

    /// \brief The trade machine: `placed -> filled -> settling -> settled`, or `failed`; `cancelled` before a fill.
    /// \ingroup core
    struct trade_lifecycle {
        /// \brief The machine's states are these enumerators.
        using state_enum = trade_state;

        /// \brief The table.
        /// \return the transition table.
        constexpr auto operator()() const noexcept
        {
            using namespace trade_event;
            constexpr auto placed = enum_state<trade_state::placed>;
            constexpr auto filled = enum_state<trade_state::filled>;
            constexpr auto settling = enum_state<trade_state::settling>;
            constexpr auto settled = enum_state<trade_state::settled>;
            constexpr auto failed = enum_state<trade_state::failed>;
            constexpr auto cancelled = enum_state<trade_state::cancelled>;
            return make_transition_table(
                *placed   + event<fill>     = filled,
                 placed   + event<cancel>   = cancelled,
                 filled   + event<instruct> = settling,
                 settling + event<settle>   = settled,
                 settling + event<fail>     = failed);
        }
    };

    /// \brief The dealing machine: `requested -> priced -> issued -> settled`; `cancelled` before issue.
    /// \ingroup core
    struct dealing_lifecycle {
        /// \brief The machine's states are these enumerators.
        using state_enum = dealing_state;

        /// \brief The table.
        /// \return the transition table.
        constexpr auto operator()() const noexcept
        {
            using namespace dealing_event;
            constexpr auto requested = enum_state<dealing_state::requested>;
            constexpr auto priced = enum_state<dealing_state::priced>;
            constexpr auto issued = enum_state<dealing_state::issued>;
            constexpr auto cancelled = enum_state<dealing_state::cancelled>;
            constexpr auto settled = enum_state<dealing_state::settled>;
            return make_transition_table(
                *requested + event<price>  = priced,
                 requested + event<cancel> = cancelled,
                 priced    + event<issue>  = issued,
                 priced    + event<cancel> = cancelled,
                 issued    + event<settle> = settled);
        }
    };

    /// \brief The corporate-action machine: `announced -> ex -> record -> paid`; `cancelled` before the record date.
    /// \ingroup core
    struct corporate_action_lifecycle {
        /// \brief The machine's states are these enumerators.
        using state_enum = action_state;

        /// \brief The table.
        /// \return the transition table.
        constexpr auto operator()() const noexcept
        {
            using namespace action_event;
            constexpr auto announced = enum_state<action_state::announced>;
            constexpr auto ex = enum_state<action_state::ex>;
            constexpr auto record = enum_state<action_state::record>;
            constexpr auto paid = enum_state<action_state::paid>;
            constexpr auto cancelled = enum_state<action_state::cancelled>;
            return make_transition_table(
                *announced + event<go_ex>       = ex,
                 announced + event<withdraw>    = cancelled,
                 ex        + event<record_date> = record,
                 ex        + event<withdraw>    = cancelled,
                 record    + event<pay>         = paid);
        }
    };

    /// \brief The repo machine: `open -> accruing -> closed`, accruing again at each accrual; closed the day it opened
    /// as well.
    /// \ingroup core
    struct repo_lifecycle {
        /// \brief The machine's states are these enumerators.
        using state_enum = repo_state;

        /// \brief The table.
        /// \return the transition table.
        constexpr auto operator()() const noexcept
        {
            using namespace repo_event;
            constexpr auto open = enum_state<repo_state::open>;
            constexpr auto accruing = enum_state<repo_state::accruing>;
            constexpr auto closed = enum_state<repo_state::closed>;
            return make_transition_table(
                *open     + event<accrue>     = accruing,
                 open     + event<repurchase> = closed,
                 accruing + event<accrue>     = accruing,
                 accruing + event<repurchase> = closed);
        }
    };

} // namespace snicholls::fsm

#endif /* snicholls_core_fsm_securities_hpp */
