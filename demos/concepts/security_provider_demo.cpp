//
//  security_provider_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/security_provider.hpp>: a trade, a rights issue and a reverse repo as facts, each
//  validated, the cash each moves, the refusals of facts a ledger could not book, and code written against the concept.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/security_provider.hpp>
#include <snicholls/core/interfaces/security_provider_reference.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/money/asset.hpp>

#include <chrono>
#include <cstdio>
#include <string>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

constexpr asset aud = checked_asset("AUD", 2), bhp = checked_asset("BHP:ASX", 0), rights = checked_asset("BHPR:ASX", 0);
const utc_s now = utc_s::from_unix_seconds(1'800'000'000);

//! [concept]
// Any provider, by concept: the reconciler reads every fact the provider has.
template <security_provider P>
std::size_t count_facts(P& provider)
{
    std::size_t n = 0;
    std::string cursor;
    for (;;) {
        const auto page = provider.statement({cursor, std::nullopt, 100});
        if (!page)
            return n;
        n += page->facts.size();
        if (page->next_cursor.empty())
            return n;
        cursor = std::string(page->next_cursor.view());
    }
}
//! [concept]
} // namespace

int main()
{
    std::printf("snicholls-core concepts/security_provider.hpp demo\n");
    //! [fact]
    security_fact trade;
    trade.source = "broker";
    trade.fact_id = "fill-20261126-1";
    trade.kind = security_fact_kind::buy;
    trade.instrument = bhp;                                   // BHP:ASX, whole shares
    trade.isin = isin_text{"AU000000BHP4"};                   // checked by validate
    trade.quantity = money::from_units(bhp, 100);
    trade.price = *parse_money(checked_asset("AUD", 3), "45.125");
    trade.consideration = -*parse_money(aud, "4512.50");      // paid
    trade.fees = *parse_money(aud, "9.95");
    trade.dates.trade_at = now;
    trade.dates.known_at = now;
    trade.dates.settles_on = now.days() + std::chrono::days{2};
    const bool bookable = validate(trade).is_ok();
    const auto cash = cash_movement(trade);                   // -4522.45
    //! [fact]
    check(bookable, "the buy validates");
    check(cash && *cash == -*parse_money(aud, "4522.45"), "it moves -4522.45 of cash, brokerage included");

    security_fact issue = trade;
    issue.fact_id = "action-rights-1";
    issue.kind = security_fact_kind::rights_issue;
    issue.quantity = money::zero(bhp);
    issue.price.reset();
    issue.consideration = money::zero(aud);
    issue.fees = money::zero(aud);
    issue.other_instrument = rights;
    issue.other_quantity = money::from_units(rights, 20);
    issue.new_for_old = finance::ratio{1, 5};                 // one right for every five shares
    issue.ex_date = now.days() + std::chrono::days{6};
    issue.record_date = now.days() + std::chrono::days{7};
    check(validate(issue).is_ok(), "a 1-for-5 rights issue validates");
    std::printf("  %s: %lld new for %lld\n", std::string(enum_name(issue.kind)).c_str(),
                static_cast<long long>(issue.new_for_old->numerator()), static_cast<long long>(issue.new_for_old->denominator()));

    security_fact repo = trade;
    repo.fact_id = "repo-1";
    repo.kind = security_fact_kind::reverse_repo_open;
    repo.price.reset();
    repo.consideration = -*parse_money(aud, "1000000.00");    // cash lent
    repo.fees = money::zero(aud);
    repo.repo = repo_terms{"5493001KJTIIGC8Y1R12", "gmra-77", *finance::ratio::parse("4.35%"),
                           finance::day_count_basis::act_365_fixed, now.days() + std::chrono::days{30}};
    check(validate(repo).is_ok() && is_repo(repo.kind), "a reverse repo validates with its terms");
    const auto interest = multiply(repo.repo->rate, finance::accrual_fraction(repo.repo->day_count, now.days(), *repo.repo->maturity));
    check(interest && *interest == finance::ratio{87 * 30, 2000 * 365}, "its interest per dollar lent is exact");

    security_fact wrong = trade;
    wrong.quantity = -money::from_units(bhp, 100);
    check(validate(wrong).unwrap_err().message.view() == "a purchase adds units and pays; a sale removes and receives",
          "a buy that removes units is refused");
    wrong = issue;
    wrong.new_for_old.reset();
    check(validate(wrong).unwrap_err().message.view() == "this action has a new-for-old ratio", "a rights issue needs its ratio");
    check(!is_instrument_code("BHP") && is_instrument_code("BHP:ASX"), "an instrument is <SYMBOL>:<VENUE>");
    check(is_corporate_action(security_fact_kind::demerger) && needs_other_instrument(security_fact_kind::demerger) &&
              !needs_ratio(security_fact_kind::demerger),
          "a demerger is an action with a second instrument and no ratio");

    memory_security_provider broker{secret<std::string>{"whsec_demo_only"}, [] { return now; }};
    static_cast<void>(broker.execute("ord-1", money::from_units(bhp, 100), -*parse_money(aud, "4512.50")));
    static_cast<void>(broker.settle("ord-1"));
    check(count_facts(broker) == 2, "code written against the concept reads the reference");
    return failures == 0 ? 0 : 1;
}
