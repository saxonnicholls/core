//
//  capability_ids_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/capability_ids.hpp>: the bounded identifiers of the contracts and make_id, which refuses an id that does not fit. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/capability_ids.hpp>

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
} // namespace

int main()
{
    std::printf("snicholls-core capability_ids.hpp demo\n");
    //! [ids]
    const auto key = make_id<client_id>("order-2026-11-24-0001");   // our idempotency key: at most 64
    const auto hash = make_id<provider_id>("0x" + std::string(64, 'f'));   // a transaction hash: 66 of 96
    const auto too_long = make_id<event_id>(std::string(65, 'e'));    // refused, never cut short
    //! [ids]
    check(key.is_ok() && key->view() == "order-2026-11-24-0001", "a client id fits");
    check(hash.is_ok() && hash->size() == 66, "a transaction hash fits a provider id");
    check(too_long.is_err() && too_long.unwrap_err().code == capability_errc::invalid_argument, "too long is invalid_argument");
    check(make_id<fact_id>("txn_1").is_ok() && make_id<provider_account>("acct_1").is_ok() &&
              make_id<provider_name>("stripe").is_ok() && make_id<short_text>("BSB 062-000").is_ok(),
          "every alias makes its id");
    check(utc_s::from_unix_seconds(60).unix_seconds() == 60, "time is utc_s");
    return failures == 0 ? 0 : 1;
}
