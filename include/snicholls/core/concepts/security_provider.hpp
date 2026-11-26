//
//  security_provider.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::security_fact` and `snicholls::security_provider`: what a broker, custodian or registry tells a
/// ledger about securities (trades and their settlement, corporate actions, repos, fund dealing), and the capability
/// that reports it.
/// \ingroup core
///
/// **Overview.** One ledger books crypto, fiat with its FX, and securities. The securities side reaches it as
/// `security_fact`s, a sibling of `payment_fact` (`concepts/payment_provider.hpp`) with the same identity (`source`,
/// `provider_account`, `event_id`, `fact_id`, booked once by `fact_id`) and the same dates (`fact_dates`), plus what a
/// security needs:
///
/// | Field | What |
/// |---|---|
/// | `kind` | `security_fact_kind`: trades and settlement, entitlements and distributions, reorganisations, rights, takeovers, repos, accruals and collateral, and fund dealing (application, redemption, switch, distribution reinvested) |
/// | `instrument`, `isin` | the security as an asset, `<SYMBOL>:<VENUE>` at its scale (`BHP:ASX` at 0, a unit trust at 4), and optionally its ISIN, checked by `ids/check_digits.hpp` |
/// | `quantity` | the signed, exact change in the holding, in `instrument` |
/// | `other_instrument`, `other_quantity`, `cost_apportionment` | the second security of a two-instrument action (a demerger's new shares, a scrip offer's bidder, a switch's target fund, a right's underlying) and the share of the cost base it takes |
/// | `new_for_old` | the ratio of a split, consolidation, bonus or rights issue (3-for-2 is 3/2) |
/// | `price`, `consideration`, `fees`, `taxes` | money: the price of one unit, the gross cash (signed: negative paid, positive received), fees, and taxes withheld or charged, with franking credits as a tax attribute |
/// | `dates`, `ex_date`, `record_date`, `pay_date` | `fact_dates` (trade, settlement, known), and a corporate action's dates as `std::chrono::sys_days` |
/// | `repo` | a repo's counterparty, agreement, exact annual rate, `finance::day_count_basis` and maturity |
/// | `party`, `refs` | our customer's uuid (never personal data) and the broker's or registry's ids |
///
/// `validate(security_fact)` is the check every fact passes before it leaves a provider; `cash_movement` is the cash it
/// moves (consideration less fees and taxes). The ledger owns the cost work (parcels, restatement, cost-base reduction,
/// rollover); core owns the contract only.
///
/// **The capability.** `security_provider` is the broker, custodian or registry side, with the shape of
/// `payment_provider` so servers' webhook inbox, worker and outbox serve both: `verify_webhook` authenticates a
/// delivery without I/O (fails closed), the worker calls `resolve(event)` for the authoritative facts, the reconcilers
/// read `statement(query)` page by page, and `holdings(as_of)` gives the positions the provider holds for us at the end
/// of a day, by trade date and by settlement, for the break check. Its webhook types (`webhook_delivery`,
/// `webhook_event`, `owned_delivery`) are `payment_provider`'s; an event about a security is
/// `event_kind::security_updated`. This is the concept half of a pair; the pure virtual half is
/// `security_provider_interface` (`interfaces/security_provider.hpp`), with `security_provider_adapter<P>` and
/// `security_provider_ref`.
///
/// **The contract** (`interfaces/security_provider_conformance.hpp` checks it):
/// - Every fact passes `validate`. Quantities, prices and cash are exact `money`; ratios are exact `finance::ratio`.
/// - **Amounts and positions never come from the webhook body**: `resolve` reads them from the provider's records.
///   An event that books nothing (`event_kind::other`, or a test delivery) resolves to no facts.
/// - A trade is booked on its trade date (`buy`, `sell`, with `dates.settles_on`) and settles by a later `settle` fact
///   carrying the same quantity, or fails by `settle_failed`. A fact with `dates.settled_at` moves the settled holding
///   on that day; `settle` is how a trade booked unsettled settles. `holdings` folds exactly that.
/// - Every statement line is exactly one fact, oldest first within a page, with `fact_id` the provider's line id.
/// - Errors follow the redaction rule of `capability_error.hpp`.
/// - Personal data stays out: a holder identification number, a name or an address is never in a fact. `party` is our
///   customer's uuid.
///
/// \snippet concepts/security_provider_demo.cpp fact
///
/// \snippet concepts/security_provider_demo.cpp concept
///
/// **Relationships.** `concepts/payment_provider.hpp` (the webhook types, the sibling contract), `capability_ids.hpp`
/// (`fact_dates`, the ids), `capability_error.hpp`, `money/money.hpp`, `finance/ratio.hpp`, `finance/day_count.hpp`,
/// `ids/check_digits.hpp`, `fsm/securities.hpp` (the lifecycles of trades, dealing, actions and repos); the twin
/// `interfaces/security_provider.hpp`, the reference `interfaces/security_provider_reference.hpp`, the kit
/// `interfaces/security_provider_conformance.hpp`. Above core: the securities library (instruments, corporate actions,
/// entitlements), the ledger and servers' broker, custodian and registry adapters. Group: \ref core.
///
/// **Implementations.** `memory_security_provider` (`interfaces/security_provider_reference.hpp`, for tests). Above
/// core: brokers, custodians and registries, in servers.
///
/// **Guarantees.** `validate`, `cash_movement`, `is_instrument_code` and the kind predicates are `constexpr` and
/// `noexcept`. Thread safety is the implementation's.
///
/// **Errors.** `capability_error`; `validate` returns `invalid_argument` with a fixed message naming the rule broken.
///
/// **Performance.** Not benchmarked: `validate` is a few comparisons per field.
///
/// **Tests and demos.** `tests/concepts/security_provider_test.cpp`, `demos/concepts/security_provider_demo.cpp`.
///
/// **Dependencies.** `<chrono>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<string_view>`, `<vector>`,
/// `capability_error.hpp`, `capability_ids.hpp`, `concepts/payment_provider.hpp`, `containers/inplace_string.hpp`,
/// `finance/day_count.hpp`, `finance/ratio.hpp`, `ids/check_digits.hpp`, `money/money.hpp`, `uuid.hpp`.
///
/// **Known issues.** The instrument code is `<SYMBOL>:<VENUE>`, as core's chain tokens (`USDC:8453`) and the
/// securities library's `asset_code` (`BHP:ASX`) are. `price` is per whole unit in the cash currency's code, at any
/// scale at least as fine as the currency's (a share priced at 0.5 cents is `AUD` at scale 3). Holdings are positions,
/// not cost: cost is the ledger's.
///
/// **Status.** New in 0.27.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_security_provider_hpp
#define snicholls_core_concepts_security_provider_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/concepts/payment_provider.hpp>
#include <snicholls/core/containers/inplace_string.hpp>
#include <snicholls/core/finance/day_count.hpp>
#include <snicholls/core/finance/ratio.hpp>
#include <snicholls/core/ids/check_digits.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/uuid.hpp>

#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief What a security fact books. Values are stable; new ones are appended.
    /// \ingroup core
    enum class security_fact_kind : std::uint8_t {
        buy = 1,                       ///< bought on the trade date; settles later
        sell = 2,                      ///< sold on the trade date; settles later
        settle = 3,                    ///< a trade settled: the units and the cash moved
        settle_failed = 4,             ///< a trade failed to settle on its date
        dividend_entitlement = 5,      ///< entitled to a dividend at the record date; paid later
        withholding = 6,               ///< tax withheld from a distribution
        dividend_paid = 7,             ///< a dividend or distribution paid in cash
        dividend_reinvested = 8,       ///< a dividend reinvested in new units (a DRP)
        capital_return = 9,            ///< cash returned from capital
        split = 10,                    ///< more units for each held
        consolidation = 11,            ///< fewer units for each held
        bonus = 12,                    ///< free units of the same security
        symbol_change = 13,            ///< the holding moved, unit for unit, to a new code
        merger = 14,                   ///< the holding replaced by the acquirer's units, cash, or both (a scheme)
        demerger = 15,                 ///< units of a new security, with part of the cost base
        rights_issue = 16,             ///< rights to subscribe, per unit held
        rights_exercised = 17,         ///< rights exercised: new units for the subscription price
        rights_lapsed = 18,            ///< rights not exercised, gone
        cash_takeover = 19,            ///< the holding taken for cash
        scrip_takeover = 20,           ///< the holding taken for the bidder's units
        repo_open = 21,                ///< securities sold under an agreement to repurchase: cash borrowed
        repo_close = 22,               ///< repurchased: the cash and the interest repaid
        reverse_repo_open = 23,        ///< securities bought under an agreement to resell: cash lent
        reverse_repo_close = 24,       ///< resold: the cash and the interest received
        interest_accrual = 25,         ///< interest accrued on a repo or a bond, not yet paid
        margin_call = 26,              ///< collateral called or posted against an exposure
        collateral_reuse = 27,         ///< collateral received and reused (rehypothecated)
        application = 28,              ///< units in a fund applied for and issued at the dealing price
        redemption = 29,               ///< units in a fund redeemed at the dealing price
        unit_switch = 30,              ///< units in one fund or class switched into another
        distribution_reinvested = 31,  ///< a fund's distribution reinvested in new units
    };

    /// \brief Whether the fact is a repo's (and so carries `repo_terms`): open, close, either side, margin and reuse.
    /// \ingroup core
    /// \param k The kind.
    /// \return `true` for the repo kinds.
    [[nodiscard]] constexpr bool is_repo(security_fact_kind k) noexcept
    {
        using enum security_fact_kind;
        return k == repo_open || k == repo_close || k == reverse_repo_open || k == reverse_repo_close ||
               k == margin_call || k == collateral_reuse;
    }

    /// \brief Whether the fact is a corporate action's (and so may carry ex, record and pay dates).
    /// \ingroup core
    /// \param k The kind.
    /// \return `true` for entitlements, distributions, reorganisations, rights and takeovers.
    [[nodiscard]] constexpr bool is_corporate_action(security_fact_kind k) noexcept
    {
        const auto v = static_cast<std::uint8_t>(k);
        return v >= static_cast<std::uint8_t>(security_fact_kind::dividend_entitlement) &&
               v <= static_cast<std::uint8_t>(security_fact_kind::scrip_takeover);
    }

    /// \brief Whether the action names a second instrument: symbol change, demerger, rights issue and exercise, scrip
    /// takeover, unit switch (a merger may, when it pays in scrip).
    /// \ingroup core
    /// \param k The kind.
    /// \return `true` when `other_instrument` is required.
    [[nodiscard]] constexpr bool needs_other_instrument(security_fact_kind k) noexcept
    {
        using enum security_fact_kind;
        return k == symbol_change || k == demerger || k == rights_issue || k == rights_exercised || k == scrip_takeover ||
               k == unit_switch;
    }

    /// \brief Whether the action is a ratio of units held: split, consolidation, bonus, rights issue.
    /// \ingroup core
    /// \param k The kind.
    /// \return `true` when `new_for_old` is required.
    [[nodiscard]] constexpr bool needs_ratio(security_fact_kind k) noexcept
    {
        using enum security_fact_kind;
        return k == split || k == consolidation || k == bonus || k == rights_issue;
    }

    /// \brief Whether `code` names a security: `<SYMBOL>:<VENUE>`, a symbol of asset-code characters other than `:`, one
    /// `:`, and a venue of upper-case letters and digits (a market identifier such as `ASX` or `XNYS`, a chain id).
    /// \ingroup core
    /// \param code The asset code.
    /// \return `true` when it does.
    [[nodiscard]] constexpr bool is_instrument_code(std::string_view code) noexcept
    {
        // A loop rather than find(): GCC 13 refuses string_view::find over an asset's own characters in a constant
        // expression.
        std::size_t colon = 0, colons = 0;
        for (std::size_t i = 0; i < code.size(); ++i)
            if (code[i] == ':') {
                colon = i;
                ++colons;
            }
        if (!is_asset_code(code) || colons != 1 || colon == 0 || colon + 1 == code.size())
            return false;
        for (std::size_t i = colon + 1; i < code.size(); ++i)
            if (!((code[i] >= 'A' && code[i] <= 'Z') || (code[i] >= '0' && code[i] <= '9')))
                return false;
        return true;
    }

    /// \brief An ISIN as text (twelve characters, checked by `validate_isin`).
    /// \ingroup core
    using isin_text = inplace_string<12>;

    /// \brief Taxes on a fact, each a non-negative amount in the fact's cash currency.
    /// \ingroup core
    struct security_taxes {
        std::optional<money> withheld;          ///< withheld at source (a non-resident's tax, a no-TFN amount)
        std::optional<money> duty;              ///< stamp duty or a financial transaction tax
        std::optional<money> franking_credit;   ///< an imputation credit attached to a franked dividend: a tax
                                                ///< attribute, never cash
        std::optional<finance::ratio> franked;  ///< the franked part of the dividend, 0 to 1

        /// \brief Equality.
        friend constexpr bool operator==(const security_taxes&, const security_taxes&) noexcept = default;
    };

    /// \brief A repo's terms: who, which agreement, the rate and how it accrues, and when it ends.
    /// \ingroup core
    struct repo_terms {
        snicholls::provider_id counterparty;  ///< the counterparty's id (an LEI, the provider's id); never a name
        snicholls::provider_id agreement_id;  ///< the agreement or the trade (a GMRA confirmation)
        finance::ratio rate;                  ///< the annual rate, exact; negative where rates are
        finance::day_count_basis day_count{finance::day_count_basis::act_360};  ///< how it accrues
        std::optional<std::chrono::sys_days> maturity;                          ///< none for an open repo

        /// \brief Equality.
        friend constexpr bool operator==(const repo_terms&, const repo_terms&) noexcept = default;
    };

    /// \brief The provider's ids a fact relates to (empty where none).
    /// \ingroup core
    struct security_refs {
        snicholls::provider_id order;          ///< the order
        snicholls::provider_id execution;      ///< the fill or execution
        snicholls::provider_id contract_note;  ///< the confirmation sent to the client
        snicholls::provider_id settlement;     ///< the settlement instruction (a CHESS or SWIFT reference)
        snicholls::provider_id action;         ///< the corporate action's id at the registry or custodian
        snicholls::provider_id dealing;        ///< a fund's application, redemption or switch request

        /// \brief Equality.
        friend constexpr bool operator==(const security_refs&, const security_refs&) noexcept = default;
    };

    /// \brief One fact about a security for the ledger: authoritative, from the provider's records, never from a
    /// webhook body. Booked once, by `fact_id`.
    /// \ingroup core
    struct security_fact {
        provider_name source;                          ///< the provider: `"memory"`, a broker, a registry
        snicholls::provider_account provider_account;  ///< opaque; the product maps it to a book
        snicholls::event_id event_id;                  ///< the delivery that led here; empty when read from a statement
        snicholls::fact_id fact_id;                    ///< the provider's line id
        security_fact_kind kind{security_fact_kind::buy};  ///< what it books
        asset instrument;                              ///< `<SYMBOL>:<VENUE>` at its scale
        std::optional<isin_text> isin;                 ///< the ISIN, when known
        money quantity;                                ///< the signed change in the holding, in `instrument`
        std::optional<asset> other_instrument;         ///< the second security of a two-instrument action
        std::optional<money> other_quantity;           ///< the signed change in it, in `other_instrument`
        std::optional<finance::ratio> cost_apportionment;  ///< the share of the cost base `other_instrument` takes, 0 to 1
        std::optional<finance::ratio> new_for_old;     ///< new units for each held: split, consolidation, bonus, rights
        std::optional<money> price;                    ///< the price of one whole unit, in the cash currency's code
        money consideration;                           ///< the gross cash: negative paid, positive received
        money fees;                                    ///< brokerage and the provider's fees, not negative
        security_taxes taxes;                          ///< withheld and charged, and franking
        fact_dates dates;                              ///< trade, settlement, known (`fact_dates`)
        std::optional<std::chrono::sys_days> ex_date;      ///< an action's ex date
        std::optional<std::chrono::sys_days> record_date;  ///< an action's record date: whose register is entitled
        std::optional<std::chrono::sys_days> pay_date;     ///< an action's payment or allotment date
        std::optional<repo_terms> repo;                ///< a repo's terms
        std::optional<uuid> party;                     ///< our customer's id; never personal data
        security_refs refs;                            ///< related provider ids

        /// \brief Equality.
        friend constexpr bool operator==(const security_fact&, const security_fact&) noexcept = default;
    };

    /// \brief The cash a fact moves: `consideration - fees - duty - withheld`.
    /// \ingroup core
    /// \param f The fact.
    /// \return The net cash, or `std::nullopt` when its amounts are of different assets or overflow.
    [[nodiscard]] constexpr std::optional<money> cash_movement(const security_fact& f) noexcept
    {
        auto net = subtract(f.consideration, f.fees);
        for (const auto& tax : {f.taxes.duty, f.taxes.withheld})
            if (tax && net)
                net = subtract(*net, *tax);
        if (!net)
            return std::nullopt;
        return *net;
    }

    namespace detail {
        constexpr bool cash_of(const std::optional<money>& m, const asset& currency) noexcept
        {
            return !m || (m->asset_of() == currency && !m->is_negative());
        }
        constexpr bool unit_interval(const std::optional<finance::ratio>& r) noexcept
        {
            return !r || (!r->is_negative() && *r <= finance::ratio{1, 1});
        }
    } // namespace detail

    /// \brief The checks every security fact passes before it leaves a provider.
    /// \ingroup core
    /// \param f The fact.
    /// \return Success, or `invalid_argument` with a fixed message naming the rule.
    [[nodiscard]] constexpr capability_result<void> validate(const security_fact& f) noexcept
    {
        using enum security_fact_kind;
        const auto bad = [](capability_message why) noexcept { return fail(capability_errc::invalid_argument, {}, why); };
        if (f.source.empty() || f.fact_id.empty())
            return bad("a fact needs its source and fact id");
        if (const auto k = static_cast<std::uint8_t>(f.kind); k < 1 || k > static_cast<std::uint8_t>(distribution_reinvested))
            return bad("not a security fact kind");
        if (auto dates = validate(f.dates); !dates)
            return dates;
        if (!is_instrument_code(f.instrument.code()))
            return bad("an instrument is <SYMBOL>:<VENUE>");
        if (f.quantity.asset_of() != f.instrument)
            return bad("the quantity is in the instrument");
        if (f.isin && !validate_isin(f.isin->view()))
            return bad("not a valid ISIN");
        if (needs_other_instrument(f.kind) && !f.other_instrument)
            return bad("this action names its other instrument");
        if (f.other_instrument && (!is_instrument_code(f.other_instrument->code()) || *f.other_instrument == f.instrument))
            return bad("the other instrument is another <SYMBOL>:<VENUE>");
        if (f.other_quantity && (!f.other_instrument || f.other_quantity->asset_of() != *f.other_instrument))
            return bad("the other quantity is in the other instrument");
        if (f.cost_apportionment && !f.other_instrument)
            return bad("a cost apportionment goes to the other instrument");
        if (!detail::unit_interval(f.cost_apportionment))
            return bad("a cost apportionment is from 0 to 1");
        if (f.kind == demerger && !f.cost_apportionment)
            return bad("a demerger apportions the cost base");
        if (needs_ratio(f.kind) && !f.new_for_old)
            return bad("this action has a new-for-old ratio");
        if (f.new_for_old) {
            const finance::ratio one{1, 1}, r = *f.new_for_old;
            if (r <= finance::ratio{} || (f.kind == split && r <= one) || (f.kind == consolidation && r >= one))
                return bad("a ratio is positive, a split's above 1, a consolidation's below");
        }
        const asset& currency = f.consideration.asset_of();
        if (f.fees.asset_of() != currency || f.fees.is_negative() || !detail::cash_of(f.taxes.withheld, currency) ||
            !detail::cash_of(f.taxes.duty, currency) || !detail::cash_of(f.taxes.franking_credit, currency))
            return bad("fees and taxes are not negative, in the cash currency");
        if (f.price && (f.price->is_negative() || f.price->asset_of().code() != currency.code() ||
                        f.price->asset_of().scale() < currency.scale()))
            return bad("a price is not negative, in the cash currency");
        if ((f.kind == buy && (f.quantity.signum() <= 0 || f.consideration.signum() > 0)) ||
            (f.kind == sell && (f.quantity.signum() >= 0 || f.consideration.is_negative())) ||
            (f.kind == application && (f.quantity.is_negative() || f.consideration.signum() > 0)) ||
            (f.kind == redemption && (f.quantity.signum() > 0 || f.consideration.is_negative())))
            return bad("a purchase adds units and pays; a sale removes and receives");
        if ((f.taxes.franking_credit || f.taxes.franked) && f.kind != dividend_entitlement && f.kind != dividend_paid &&
            f.kind != dividend_reinvested && f.kind != distribution_reinvested)
            return bad("franking is a dividend's");
        if (!detail::unit_interval(f.taxes.franked))
            return bad("the franked part is from 0 to 1");
        if ((f.ex_date || f.record_date || f.pay_date) && !is_corporate_action(f.kind) && f.kind != distribution_reinvested)
            return bad("ex, record and pay dates are a corporate action's");
        if ((f.ex_date && f.record_date && *f.ex_date > *f.record_date) ||
            (f.record_date && f.pay_date && *f.record_date > *f.pay_date) || (f.ex_date && f.pay_date && *f.ex_date > *f.pay_date))
            return bad("ex, then record, then pay");
        if ((f.kind == dividend_entitlement || f.kind == capital_return || needs_ratio(f.kind) || f.kind == demerger) &&
            !f.record_date)
            return bad("an entitlement has its record date");
        if (is_repo(f.kind) && !f.repo)
            return bad("a repo fact carries its terms");
        if (f.repo) {
            if (f.repo->counterparty.empty() || f.repo->agreement_id.empty())
                return bad("a repo names its counterparty and agreement");
            const auto basis = static_cast<std::uint8_t>(f.repo->day_count);
            if (basis < 1 || basis > static_cast<std::uint8_t>(finance::day_count_basis::act_act_isda))
                return bad("not a day-count basis");
            if (f.repo->maturity && *f.repo->maturity < f.dates.trade_at.days())
                return bad("a repo matures after it opens");
        }
        return ok();
    }

    /// \brief Which facts to read: a page from `cursor` (empty for the first), optionally one instrument's.
    /// \ingroup core
    struct security_statement_query {
        std::string_view cursor;          ///< from the previous page's `next_cursor`
        std::optional<asset> instrument;  ///< only this instrument's facts (either instrument of a two-instrument one)
        std::size_t limit{100};           ///< at most this many facts

        /// \brief Equality.
        friend constexpr bool operator==(const security_statement_query&, const security_statement_query&) noexcept = default;
    };

    /// \brief One page of statement lines, each a fact.
    /// \ingroup core
    struct security_statement_page {
        std::vector<security_fact> facts;    ///< oldest first within the page
        snicholls::provider_id next_cursor;  ///< empty on the last page

        /// \brief Equality.
        friend bool operator==(const security_statement_page&, const security_statement_page&) = default;
    };

    /// \brief One position the provider holds for us at the end of a day.
    /// \ingroup core
    struct security_holding {
        asset instrument;  ///< the security
        money traded;      ///< by trade date: every fact traded on or before the day
        money settled;     ///< by settlement: what has settled on or before the day

        /// \brief Equality.
        friend constexpr bool operator==(const security_holding&, const security_holding&) noexcept = default;
    };

    /// \brief What a security provider reports.
    /// \ingroup core
    struct security_capabilities {
        std::vector<security_fact_kind> kinds;         ///< the kinds of fact it reports
        bool holdings{false};                          ///< whether `holdings` answers (otherwise `unsupported`)
        snicholls::provider_account provider_account;  ///< the opaque id of our account at the provider

        /// \brief Equality.
        friend bool operator==(const security_capabilities&, const security_capabilities&) = default;
    };

    /// \brief A security provider (a broker, custodian or registry): `capabilities`, `verify_webhook`, `resolve`,
    /// `statement`, `holdings`.
    /// \ingroup core
    template <typename P>
    concept security_provider = requires(P& p, const P& cp, const webhook_delivery& w, const webhook_event& e,
                                         const security_statement_query& q, std::chrono::sys_days as_of) {
        { cp.capabilities() } -> std::same_as<security_capabilities>;
        { cp.verify_webhook(w) } -> std::same_as<capability_result<webhook_event>>;
        { p.resolve(e) } -> std::same_as<capability_result<std::vector<security_fact>>>;
        { p.statement(q) } -> std::same_as<capability_result<security_statement_page>>;
        { p.holdings(as_of) } -> std::same_as<capability_result<std::vector<security_holding>>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_security_provider_hpp
