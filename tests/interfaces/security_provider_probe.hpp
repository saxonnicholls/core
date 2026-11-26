//
//  security_provider_probe.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The probe the security provider kit runs the reference through, in every form (capability_probes.hpp's forms).
//  Shared by tests/interfaces/security_provider_test.cpp, _conformance_test.cpp and _reference_test.cpp.
//

#ifndef snicholls_core_tests_interfaces_security_provider_probe_hpp
#define snicholls_core_tests_interfaces_security_provider_probe_hpp

#include <snicholls/core/interfaces/security_provider_reference.hpp>

#include "capability_probes.hpp"

#include <chrono>
#include <string>
#include <string_view>

namespace snicholls::test {

    inline constexpr asset bhp = checked_asset("BHP:ASX", 0);

    inline memory_security_provider make_security_provider()
    {
        return memory_security_provider{secret<std::string>{"whsec_test_only"}, [] { return probe_time; }};
    }

    template <form F>
    struct security_provider_probe {
        forms<memory_security_provider, security_provider_interface, security_provider_adapter, security_provider_ref, F> f{
            make_security_provider()};
        auto& provider() { return f.get(); }
        static money quantity() { return money::from_units(bhp, 250); }
        static money consideration() { return -*parse_money(aud, "11281.25"); }
        void trade(std::string_view order, const money& q, const money& c) { static_cast<void>(f.impl.execute(order, q, c)); }
        void settle(std::string_view order) { static_cast<void>(f.impl.settle(order)); }
        owned_delivery deliver(const webhook_event& e) { return f.impl.signed_webhook(e); }
        owned_delivery deliver_late(const webhook_event& e) { return f.impl.signed_webhook(e, probe_time - std::chrono::minutes{6}); }
        void provider_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

} // namespace snicholls::test

#endif // snicholls_core_tests_interfaces_security_provider_probe_hpp
