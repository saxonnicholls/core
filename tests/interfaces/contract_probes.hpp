//
//  contract_probes.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The probes and test sources of the 0.27 contracts (uuid_generator, time_zone, task_executor, encryptor,
//  sql_connection, http_client), in the three forms of capability_probes.hpp. Shared by tests/interfaces/<capability>_test.cpp
//  and <capability>_conformance_test.cpp.
//

#ifndef snicholls_core_tests_interfaces_contract_probes_hpp
#define snicholls_core_tests_interfaces_contract_probes_hpp

#include "capability_probes.hpp"

#include <snicholls/core/interfaces/encryptor_reference.hpp>
#include <snicholls/core/interfaces/http_client_reference.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>
#include <snicholls/core/interfaces/sql_connection_reference.hpp>
#include <snicholls/core/interfaces/time_zone_reference.hpp>
#include <snicholls/core/interfaces/uuid_generator_reference.hpp>

#include <chrono>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>

namespace snicholls::test {

    /// A wyrand that fails on command: the operating system's source, out of entropy.
    struct failing_random {
        stats::wyrand r{42};
        std::shared_ptr<bool> fails = std::make_shared<bool>(false);
        std::optional<std::uint64_t> operator()() noexcept
        {
            if (*fails)
                return std::nullopt;
            return r();
        }
    };

    /// A wall clock in milliseconds that the test may move either way, as a system clock may.
    struct settable_clock {
        std::shared_ptr<utc_ms> t = std::make_shared<utc_ms>(utc_ms::from_unix_millis(1'800'000'000'000));
        [[nodiscard]] utc_ms now() const noexcept { return *t; }
    };

    using test_uuid_generator = basic_uuid_generator<failing_random, settable_clock>;

    template <form F>
    struct uuid_generator_probe {
        forms<test_uuid_generator, uuid_generator_interface, uuid_generator_adapter, uuid_generator_ref, F> f{
            test_uuid_generator{failing_random{}, settable_clock{}}};
        auto& generator() { return f.get(); }
        std::int64_t clock_ms() { return f.impl.clock().now().unix_millis(); }
        void clock_steps_back() { *f.impl.clock().t -= std::chrono::seconds{1}; }
        void randomness_fails() { *f.impl.random().fails = true; }
    };

    /// The kit's window: 2025 to 2027, three daylight-saving years.
    inline constexpr std::chrono::sys_seconds zone_window_from{std::chrono::sys_days{std::chrono::year{2025} / 1 / 1}};
    inline constexpr std::chrono::sys_seconds zone_window_to{std::chrono::sys_days{std::chrono::year{2028} / 1 / 1}};

    inline fixed_offset_zone make_test_zone(std::type_identity<fixed_offset_zone>)
    {
        return fixed_offset_zone{"UTC+05:30", std::chrono::minutes{330}, "+0530"};
    }
    inline two_rule_zone make_test_zone(std::type_identity<two_rule_zone>) { return sydney_like_zone(); }

    template <typename Zone, form F>
    struct time_zone_probe {
        forms<Zone, time_zone_interface, time_zone_adapter, time_zone_ref, F> f;
        time_zone_probe() : f(make_test_zone(std::type_identity<Zone>{})) {}
        auto& zone() { return f.get(); }
        static std::chrono::sys_seconds from() { return zone_window_from; }
        static std::chrono::sys_seconds to() { return zone_window_to; }
    };

    /// The test double with two keys, "k1" and "k2".
    inline insecure_test_encryptor make_test_encryptor()
    {
        insecure_test_encryptor e;
        std::array<std::byte, 32> k{};
        for (std::size_t i = 0; i < k.size(); ++i)
            k[i] = static_cast<std::byte>(i * 7 + 1);
        static_cast<void>(e.add_key("k1", k));
        k[0] = std::byte{0xff};
        static_cast<void>(e.add_key("k2", k));
        return e;
    }

    template <form F>
    struct encryptor_probe {
        forms<insecure_test_encryptor, encryptor_interface, encryptor_adapter, encryptor_ref, F> f{make_test_encryptor()};
        auto& encryptor() { return f.get(); }
        static std::string_view key() { return "k1"; }
        static std::string_view other_key() { return "k2"; }
        void forget(std::string_view id) { f.impl.remove_key(id); }
    };

    template <form F>
    struct sql_connection_probe {
        forms<memory_sql_connection, sql_connection_interface, sql_connection_adapter, sql_connection_ref, F> f{memory_sql_connection{}};
        auto& connection() { return f.get(); }
        void connection_breaks() { f.impl.break_connection(); }
    };

    /// The kit's server: GET /echo echoes the body and X-Echo; anything else is 404.
    inline route_table echo_server()
    {
        route_table t;
        static_cast<void>(t.add(http_method::get, "/echo", [](const http_request& r) {
            http_response out = respond(200, r.body, "application/octet-stream");
            if (const auto h = find_header(r.headers, "X-Echo"))
                out.headers.push_back({"X-Echo", std::string(*h)});
            return out;
        }));
        return t;
    }

    using test_http_client = loopback_http_client<route_table>;

    template <form F>
    struct http_client_probe {
        forms<test_http_client, http_client_interface, http_client_adapter, http_client_ref, F> f{test_http_client{echo_server()}};
        auto& client() { return f.get(); }
        void transport_fails(capability_errc c, std::string_view r) { f.impl.fail_next(c, r); }
    };

} // namespace snicholls::test

#endif // snicholls_core_tests_interfaces_contract_probes_hpp
