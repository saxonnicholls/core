//
//  onramp_provider.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::onramp_provider`: anything that buys a stablecoin with fiat at a venue and withdraws it to an
/// address, with the venue's facts for a ledger.
/// \ingroup core
///
/// **Overview.** An on-ramp turns fiat held at a venue into a stablecoin and sends it on chain: `quote`, `buy`, then
/// `withdraw` to a whitelisted address, with `balances` at every step and `statement` for the ledger. Moving funds
/// between the venue's own accounts (an exchange's trading and funding accounts) is never hidden inside `withdraw`: an
/// implementation that needs it offers it as an explicit, idempotent operation of its own, and `withdraw` with too
/// little in the account it draws from is `insufficient_funds`. This is the concept half of a pair; the pure virtual
/// half is `onramp_provider_interface` (`interfaces/onramp_provider.hpp`), with `onramp_provider_adapter<O>` and
/// `onramp_provider_ref`.
///
/// **The contract** (`interfaces/onramp_provider_conformance.hpp` checks it):
/// - Every amount is an exact `money` (`money/money.hpp`) at its asset's scale (USDC 6, ETH 18, AUD 2); a price is an
///   exact `rate` (`money/rate.hpp`) from the asset to the quote currency, in whole units. Never a `double` or a string.
/// - `buy` is idempotent on `client_order_id`, `withdraw` on `client_reference` (each at most 64 characters,
///   `client_id`): the same id and request return the same order or withdrawal; a different request is `conflict`.
/// - A buy with `max_price` never fills above it (`rejected` when the venue is above it); more than the balance is
///   `insufficient_funds`.
/// - A withdrawal to a chain the asset does not support, or to a malformed address, is `invalid_argument`; more than
///   the available balance (the amount plus the network fee) is `insufficient_funds`.
/// - An unknown order or withdrawal id is `not_found`.
/// - `statement(query)` pages the venue's **facts** for the ledger (`onramp_fact`, booked once by `fact_id`): one per
///   fill or per withdrawal sent, newest page first, oldest first within a page. **A fact is never rounded**: it
///   carries the venue's stated amounts in the venue's own assets at the venue's precision, apart from the chain
///   asset. A withdrawal fact is the 1:1 move from the venue asset to the chain asset; an amount the chain asset
///   cannot hold exactly is an error, never a rounding. Every fact passes `validate`.
/// - Errors follow the redaction rule of `capability_error.hpp`.
///
/// \snippet concepts/onramp_provider_demo.cpp concept
///
/// **Relationships.** `money/money.hpp`, `money/rate.hpp`, `capability_error.hpp`, `capability_ids.hpp`,
/// `time/utc.hpp`; the twin `interfaces/onramp_provider.hpp`, the reference `interfaces/onramp_provider_reference.hpp`,
/// the kit `interfaces/onramp_provider_conformance.hpp`. Group: \ref core.
///
/// **Implementations.** `memory_onramp_provider` (`interfaces/onramp_provider_reference.hpp`, for tests). Above core:
/// an OKX venue.
///
/// **Guarantees.** `is_evm_address` and `same_value` are `constexpr` and `noexcept`; `validate` is `constexpr` and
/// `noexcept`. Thread safety is the implementation's.
///
/// **Errors.** `capability_error`, as the contract says; `validate` answers `invalid_argument` with a fixed message.
///
/// **Performance.** Not benchmarked: a venue is I/O bound.
///
/// **Tests and demos.** `tests/concepts/onramp_provider_test.cpp`, `demos/concepts/onramp_provider_demo.cpp`.
///
/// **Dependencies.** `<concepts>`, `<cstddef>`, `<cstdint>`, `<optional>`, `<string_view>`, `<vector>`,
/// `capability_error.hpp`, `capability_ids.hpp`, `money/money.hpp`, `money/rate.hpp`, `time/utc.hpp`, `uint256.hpp`.
///
/// **Known issues.** `onramp_quote` holds a `rate`, which has no default value, so the quote is made with all its
/// members (an aggregate with `.price` given). Selling (an off-ramp) and fiat withdrawal from the venue are not in the
/// contract.
///
/// **Status.** New in 0.23.0. Experimental until two real implementations pass the kit.

#ifndef snicholls_core_concepts_onramp_provider_hpp
#define snicholls_core_concepts_onramp_provider_hpp

#include <snicholls/core/capability_error.hpp>
#include <snicholls/core/capability_ids.hpp>
#include <snicholls/core/money/money.hpp>
#include <snicholls/core/money/rate.hpp>
#include <snicholls/core/time/utc.hpp>
#include <snicholls/core/uint256.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace snicholls {

    /// \brief One asset a venue can deliver.
    /// \ingroup core
    struct onramp_asset {
        asset unit;                            ///< the asset (code and scale)
        std::vector<std::uint64_t> chain_ids;  ///< the EVM chain ids it can be withdrawn on
        bool stablecoin{false};                ///< whether it is a stablecoin
        short_text peg;                        ///< the currency it is pegged to (`"USD"`), or empty

        /// \brief Equality.
        friend bool operator==(const onramp_asset&, const onramp_asset&) = default;
    };

    /// \brief A request for a price: spend `amount` of the quote currency, or receive `amount` of the asset, as
    /// `amount`'s asset says; any other asset is `invalid_argument`.
    /// \ingroup core
    struct onramp_quote_request {
        asset unit;            ///< what to buy (USDC, scale 6)
        asset quote_currency;  ///< what pays for it (AUD, scale 2; or USDT)
        money amount;          ///< in `quote_currency` (spend) or in `unit` (receive)

        /// \brief Equality.
        friend constexpr bool operator==(const onramp_quote_request&, const onramp_quote_request&) noexcept = default;
    };

    /// \brief A price for a ticket of that size, slippage included.
    /// \ingroup core
    struct onramp_quote {
        snicholls::rate price;  ///< from the asset to the quote currency, whole units, exact
        money spend;            ///< in the quote currency, fee included
        money receive;          ///< in the asset
        money fee;              ///< the venue's fee
        utc_s observed_at{};    ///< the venue's time
        utc_s expires_at{};     ///< after this, quote again

        /// \brief Equality.
        friend constexpr bool operator==(const onramp_quote&, const onramp_quote&) noexcept = default;
    };

    /// \brief A request to buy.
    /// \ingroup core
    struct buy_request {
        std::string_view client_order_id;  ///< required, at most 64 characters: the idempotency key (borrowed)
        asset unit;                        ///< what to buy
        asset quote_currency;              ///< what pays for it
        money amount;                      ///< spend (in `quote_currency`) or receive (in `unit`)
        std::optional<rate> max_price;     ///< never fill above this

        /// \brief Equality.
        friend constexpr bool operator==(const buy_request&, const buy_request&) noexcept = default;
    };

    /// \brief Where an order is.
    /// \ingroup core
    enum class order_status : std::uint8_t {
        open = 1,                    ///< accepted, not filled
        partially_filled = 2,        ///< filled in part
        filled = 3,                  ///< filled
        cancelled = 4,               ///< cancelled
        rejected = 5,                ///< refused by the venue
        filled_partially_final = 6,  ///< final with part filled: an IOC or market order whose rest the venue cancelled
    };

    /// \brief An order, as the venue reports it.
    /// \ingroup core
    struct order {
        snicholls::provider_id order_id;      ///< the venue's id
        snicholls::client_id client_order_id; ///< ours
        order_status status{order_status::open};  ///< where it is
        money filled;                         ///< of the asset
        money spent;                          ///< of the quote currency, fee included
        money fee;                            ///< the venue's fee
        utc_s updated_at{};                   ///< the venue's time

        /// \brief Equality.
        friend constexpr bool operator==(const order&, const order&) noexcept = default;
    };

    /// \brief A request to withdraw to an address.
    /// \ingroup core
    struct withdraw_request {
        std::string_view client_reference;  ///< required, at most 64 characters: the idempotency key (borrowed)
        money amount;                       ///< net to arrive
        std::uint64_t chain_id{1};          ///< the EVM chain id
        std::string_view address;           ///< `0x` and 40 hex digits; whitelisted at the venue (borrowed)
        std::string_view memo;              ///< where the chain needs one (borrowed)

        /// \brief Equality.
        friend constexpr bool operator==(const withdraw_request&, const withdraw_request&) noexcept = default;
    };

    /// \brief Where a withdrawal is.
    /// \ingroup core
    enum class withdrawal_status : std::uint8_t {
        pending = 1,    ///< accepted
        broadcast = 2,  ///< on the network
        confirmed = 3,  ///< confirmed
        failed = 4,     ///< will not arrive
        cancelled = 5,  ///< withdrawn
    };

    /// \brief A withdrawal, as the venue reports it.
    /// \ingroup core
    struct withdrawal {
        snicholls::provider_id withdrawal_id;     ///< the venue's id
        snicholls::client_id client_reference;    ///< ours
        withdrawal_status status{withdrawal_status::pending};  ///< where it is
        money amount;                             ///< net
        money fee;                                ///< the network fee charged
        std::uint64_t chain_id{1};                ///< the chain
        short_text address;                       ///< the destination
        snicholls::provider_id transaction_hash;  ///< once broadcast
        utc_s updated_at{};                       ///< the venue's time

        /// \brief Equality.
        friend constexpr bool operator==(const withdrawal&, const withdrawal&) noexcept = default;
    };

    /// \brief A balance at the venue.
    /// \ingroup core
    struct venue_balance {
        short_text account;  ///< the venue's account (`"trading"`, `"funding"`)
        money available;     ///< ready to move
        money total;         ///< including held

        /// \brief Equality.
        friend constexpr bool operator==(const venue_balance&, const venue_balance&) noexcept = default;
    };

    /// \brief Whether `s` is an EVM address: `0x` and 40 hex digits (the EIP-55 checksum is not checked).
    /// \ingroup core
    /// \param s The text.
    /// \return `true` when well formed.
    [[nodiscard]] constexpr bool is_evm_address(std::string_view s) noexcept
    {
        if (s.size() != 42 || s[0] != '0' || (s[1] != 'x' && s[1] != 'X'))
            return false;
        for (std::size_t i = 2; i < s.size(); ++i) {
            const char c = s[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
                return false;
        }
        return true;
    }

    /// \brief What a fact books.
    /// \ingroup core
    enum class onramp_fact_kind : std::uint8_t {
        trade_fill = 1,           ///< one fill of a buy
        withdrawal = 2,           ///< an on-chain withdrawal sent: the venue asset out, the chain asset to the address
        withdrawal_reversed = 3,  ///< a withdrawal that failed or was cancelled after it was booked
        internal_transfer = 4,    ///< between the venue's own accounts
    };

    /// \brief One fact for the ledger, in the venue's own assets at its precision, never rounded; booked once by
    /// `fact_id`.
    /// \ingroup core
    struct onramp_fact {
        provider_name source;                         ///< `"okx"`
        snicholls::provider_account provider_account; ///< opaque; the product maps it to a book
        snicholls::fact_id fact_id;                   ///< the venue's trade id, withdrawal id (or `<id>:reversed`), transfer id
        onramp_fact_kind kind{onramp_fact_kind::trade_fill};  ///< what it books
        utc_s occurred_at{};                          ///< the venue's time
        short_text account;                           ///< the venue account debited (`"trading"`, `"funding"`)
        short_text to_account;                        ///< an internal transfer's account credited
        money debit;   ///< what left `account`, in a venue asset (quote spent; amount withdrawn; amount moved)
        money credit;  ///< what arrived: the venue asset bought; for a withdrawal the same value in the chain asset
        money fee;     ///< the venue's fee in its venue asset; zero when none
        std::optional<rate> price;                    ///< a fill's exact price, quote per asset
        snicholls::provider_id order_id;              ///< the venue's order (fills)
        snicholls::client_id client_order_id;         ///< ours (fills)
        snicholls::provider_id withdrawal_id;         ///< the venue's withdrawal
        snicholls::client_id client_reference;        ///< ours (withdrawals, transfers)
        std::uint64_t chain_id{0};                    ///< withdrawals
        short_text address;                           ///< withdrawals: the destination
        snicholls::provider_id transaction_hash;      ///< withdrawals: once broadcast

        /// \brief Equality.
        friend constexpr bool operator==(const onramp_fact&, const onramp_fact&) noexcept = default;
    };

    /// \brief Whether two amounts of different scales are exactly the same value, sign included.
    /// \ingroup core
    /// \param a One amount.
    /// \param b The other.
    /// \return `true` when equal in value.
    [[nodiscard]] constexpr bool same_value(const money& a, const money& b) noexcept
    {
        if (a.is_negative() != b.is_negative())
            return false;
        const unsigned sa = a.asset_of().scale(), sb = b.asset_of().scale();
        const auto up_a = uint256::checked_mul(a.magnitude(), uint256::pow10(sa < sb ? sb - sa : 0u));
        const auto up_b = uint256::checked_mul(b.magnitude(), uint256::pow10(sb < sa ? sa - sb : 0u));
        return up_a && up_b && *up_a == *up_b;
    }

    /// \brief The checks every fact passes before it leaves a provider.
    /// \ingroup core
    /// \param f The fact.
    /// \return Success, or `invalid_argument` with a fixed message.
    [[nodiscard]] constexpr capability_result<void> validate(const onramp_fact& f) noexcept
    {
        if (f.source.empty() || f.fact_id.empty() || f.account.empty())
            return fail(capability_errc::invalid_argument, {}, "a fact needs its source, fact id and account");
        if (f.debit.is_negative() || f.credit.is_negative() || f.fee.is_negative())
            return fail(capability_errc::invalid_argument, {}, "a fact's amounts are not negative");
        switch (f.kind) {
        case onramp_fact_kind::trade_fill:
            if (f.debit.is_zero() || f.credit.is_zero() || !f.price || f.order_id.empty())
                return fail(capability_errc::invalid_argument, {}, "a fill needs both legs, a price and an order");
            if (f.debit.asset_of() == f.credit.asset_of())
                return fail(capability_errc::invalid_argument, {}, "a fill moves two assets");
            break;
        case onramp_fact_kind::withdrawal:
        case onramp_fact_kind::withdrawal_reversed:
            if (f.debit.is_zero() || f.chain_id == 0 || f.address.empty() || f.withdrawal_id.empty())
                return fail(capability_errc::invalid_argument, {}, "a withdrawal needs an amount, chain, address and id");
            if (f.credit.asset_of() == f.debit.asset_of() || !same_value(f.debit, f.credit))
                return fail(capability_errc::invalid_argument, {}, "a withdrawal moves one value 1:1 to the chain asset");
            break;
        case onramp_fact_kind::internal_transfer:
            if (f.debit.is_zero() || f.debit != f.credit || !f.fee.is_zero() || f.to_account.empty() || f.to_account == f.account)
                return fail(capability_errc::invalid_argument, {}, "a transfer moves one amount between two accounts");
            break;
        }
        return ok();
    }

    /// \brief Which facts a statement reads.
    /// \ingroup core
    enum class onramp_statement_source : std::uint8_t {
        fills = 1,        ///< one fact per fill
        withdrawals = 2,  ///< one fact per withdrawal sent
    };

    /// \brief A statement page request: from `cursor` (empty for the newest page), at most `limit` facts.
    /// \ingroup core
    struct onramp_statement_query {
        onramp_statement_source source{onramp_statement_source::fills};  ///< what to read
        std::string_view cursor;   ///< the previous page's `next_cursor` (borrowed)
        std::uint32_t limit{100};  ///< at most this many (1 to 100)

        /// \brief Equality.
        friend constexpr bool operator==(const onramp_statement_query&, const onramp_statement_query&) noexcept = default;
    };

    /// \brief One page of facts.
    /// \ingroup core
    struct onramp_statement_page {
        std::vector<onramp_fact> facts;    ///< oldest first within the page
        snicholls::provider_id next_cursor; ///< for the next (older) page; empty on the last

        /// \brief Equality.
        friend bool operator==(const onramp_statement_page&, const onramp_statement_page&) = default;
    };

    /// \brief An on-ramp provider: `assets`, `quote`, `buy`, `order_status`, `withdraw`, `withdrawal_status`,
    /// `balances`, `statement`.
    /// \ingroup core
    template <typename O>
    concept onramp_provider = requires(O& o, const O& co, const onramp_quote_request& q, const buy_request& b,
                                       const withdraw_request& w, std::string_view id, const onramp_statement_query& sq) {
        { co.assets() } -> std::same_as<std::vector<onramp_asset>>;
        { o.quote(q) } -> std::same_as<capability_result<onramp_quote>>;
        { o.buy(b) } -> std::same_as<capability_result<order>>;
        { o.order_status(id) } -> std::same_as<capability_result<order>>;
        { o.withdraw(w) } -> std::same_as<capability_result<withdrawal>>;
        { o.withdrawal_status(id) } -> std::same_as<capability_result<withdrawal>>;
        { o.balances() } -> std::same_as<capability_result<std::vector<venue_balance>>>;
        { o.statement(sq) } -> std::same_as<capability_result<onramp_statement_page>>;
    };

} // namespace snicholls

#endif // snicholls_core_concepts_onramp_provider_hpp
