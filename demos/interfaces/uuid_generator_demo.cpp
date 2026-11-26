//
//  uuid_generator_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/uuid_generator.hpp>: a generator chosen at run time through the virtual
//  interface, the adapter and the handle. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/uuid_generator.hpp>
#include <snicholls/core/interfaces/uuid_generator_reference.hpp>

#include <cstdio>
#include <memory>

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
    std::printf("snicholls-core interfaces/uuid_generator.hpp demo\n");
    //! [virtual]
    std::unique_ptr<uuid_generator_interface> chosen = std::make_unique<uuid_generator_adapter<reference_uuid_generator>>(
        make_reference_uuid_generator(1, utc_ms::from_unix_millis(1'800'000'000'000)));
    uuid_generator_ref ids{*chosen};   // copyable, models the concept
    const auto session = ids.v4();
    //! [virtual]
    check(session.is_ok() && session->version() == 4, "a version 4 through the handle");
    check(chosen->v7()->version() == 7, "a version 7 through the interface");
    auto& adapter = static_cast<uuid_generator_adapter<reference_uuid_generator>&>(*chosen);
    check(adapter.get().clock().now().unix_millis() == 1'800'000'000'000, "the adapter reaches the model");
    return failures == 0 ? 0 : 1;
}
