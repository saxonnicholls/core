//
//  kv_health_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/kv_health.hpp>: a supervisor that holds only kv_health_interface
//  references to the stores it watches, fed by kv_health_adapter, with a model's exception turned into an
//  error. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/interfaces/kv_health.hpp>

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>

using namespace snicholls;

namespace {
struct throwing_probe {   // a health check whose driver throws
    result<void, kv_errc> ping() const { throw std::runtime_error("timeout"); }
    result<kv_diagnostics, kv_errc> diagnostics() const { throw std::runtime_error("timeout"); }
};
} // namespace

int main()
{
    std::printf("snicholls-core kv_health_interface demo\n");
    bool ok = true;
    reference_key_value_store primary, replica;
    static_cast<void>(kv_put(primary, "k", "v"));
    replica.set_unavailable(true);

    //! [probe]
    // The supervisor sees only the interface; adapters put any model of kv_health behind it.
    const kv_health_adapter<reference_key_value_store&> primary_health{primary};
    const kv_health_adapter<reference_key_value_store&> replica_health{replica};
    const kv_health_adapter driver_health{throwing_probe{}};
    const std::vector<std::reference_wrapper<const kv_health_interface>> watched{primary_health, replica_health, driver_health};
    std::size_t up = 0;
    for (const kv_health_interface& store : watched) {
        const auto ping = store.ping();
        const auto d = store.diagnostics();
        std::printf("  %s, %llu keys\n", ping.is_ok() ? "up" : "down",
                    static_cast<unsigned long long>(d.is_ok() ? d->keys.value_or(0) : 0));
        up += ping.is_ok();
    }
    //! [probe]

    ok &= up == 1;
    ok &= driver_health.ping().unwrap_err() == kv_errc::backend;
    ok &= primary_health.diagnostics()->keys == 1u && &primary_health.get() == &primary;

    std::printf(ok ? "kv_health_interface demo: ok\n" : "kv_health_interface demo: FAILED\n");
    return ok ? 0 : 1;
}
