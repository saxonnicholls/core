//
//  utc_format_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/utc_format.hpp>.
//

// Demo: interfaces/utc/utc_format.hpp - std::format with snicholls::utc.

#include <chrono>
#include <format>
#include <iostream>

#include <snicholls/core/time/utc_format.hpp>

//! [demo]
int main()
{
    using namespace std::chrono;
    using snicholls::utc;
    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << '\n';
        failures += !ok;
    };

    const utc<> t = utc<>::from_unix_nanos(1714486985123456789);
    std::cout << std::format("{{}}          -> {}\n{{:%a %d %b}} -> {:%a %d %b}\n", t, t);
    check("{} is the RFC 3339 fast path", std::format("{}", t) == "2024-04-30T14:23:05.123456789Z");
    check("chrono specs are forwarded to std::formatter<sys_time>", std::format("{:%F %H:%M}", t) == "2024-04-30 14:23");
    try {
        (void)std::format("{}", utc<seconds>(sys_days{year(12000) / January / 1}));
        check("year 12000 throws std::format_error", false);
    } catch (const std::format_error& e) {
        std::cout << "    " << e.what() << '\n';
        check("year 12000 throws std::format_error", true);
    }

    std::cout << (failures == 0 ? "OK\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
//! [demo]
