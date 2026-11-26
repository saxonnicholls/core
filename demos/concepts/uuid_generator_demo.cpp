//
//  uuid_generator_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/uuid_generator.hpp>: identifiers made by code written once against the concept.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/uuid_generator.hpp>
#include <snicholls/core/interfaces/uuid_generator_reference.hpp>

#include <cstdio>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

// Generic code: an order id from any generator.
template <uuid_generator G>
capability_result<uuid> new_order_id(G& ids)
{
    return ids.v7();   // time-ordered: rows insert in key order
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/uuid_generator.hpp demo\n");
    //! [concept]
    auto ids = make_reference_uuid_generator(2026, utc_ms::from_unix_millis(1'800'000'000'000));
    const auto first = new_order_id(ids);
    const auto second = new_order_id(ids);
    //! [concept]
    check(first.is_ok() && second.is_ok(), "two identifiers");
    check(*first < *second, "the second sorts after the first");
    check(first->version() == 7 && first->timestamp()->time_since_epoch().count() == 1'800'000'000'000, "version 7 at the clock's millisecond");
    std::printf("    %s\n", to_string(*first).c_str());
    return failures == 0 ? 0 : 1;
}
