//
//  utc_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/time/utc.hpp>.
//

// Demo: interfaces/utc/utc.hpp - snicholls::utc, a zero-overhead UTC instant (one std::chrono::sys_time).

#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <unordered_set>

#include <snicholls/core/time/rfc3339.hpp>
#include <snicholls/core/time/utc.hpp>

namespace demo {
// Hides a value from the optimiser, so the demo runs the library at run time.
#if defined(_MSC_VER)
#define SNICHOLLS_DEMO_NOINLINE __declspec(noinline)
#else
#define SNICHOLLS_DEMO_NOINLINE [[gnu::noinline]]
#endif
template <typename T>
SNICHOLLS_DEMO_NOINLINE T opaque(T v)
{
    return v;
}

} // namespace demo

//! [demo]
int main()
{
    using namespace std::chrono;
    using namespace std::chrono_literals;
    using snicholls::utc;
    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << '\n';
        failures += !ok;
    };

    std::cout << "sizeof(snicholls::utc<>) = " << sizeof(utc<>) << ", trivially copyable: " << std::boolalpha
              << std::is_trivially_copyable_v<utc<>> << '\n';
    check("zero overhead: same size as sys_time<ns>", sizeof(utc<>) == sizeof(sys_time<nanoseconds>));

    // Construction: the epoch by default; now(); from std::chrono; from Unix numbers (market data is Unix ms).
    const utc<> now = utc<>::now();
    const utc<> massive = demo::opaque(utc<>::from_unix_millis(1714486985123));
    const utc<> from_calendar = sys_days{2024y / April / 30} + 14h + 23min + 5s + 123ms;
    std::cout << "now = " << now << "\nMassive 1714486985123 ms = " << massive << '\n';
    check("default is the epoch, never now", demo::opaque(utc<>{}) == utc<>::epoch() && utc<>{}.unix_nanos() == 0);
    check("now() is after 2024", now > massive);
    check("from_unix_millis == calendar expression", massive == from_calendar);
    check("from_unix_seconds / micros / nanos", utc<>::from_unix_seconds(demo::opaque(1)).unix_nanos() == 1000000000 &&
                                                   utc<>::from_unix_micros(demo::opaque(1)).unix_nanos() == 1000 &&
                                                   utc<>::from_unix_nanos(7).unix_nanos() == 7);
    check("from_unix(1h)", utc<>::from_unix(1h).unix_seconds() == 3600);
    check("is_representable: 9223372036 s fits in ns, 9223372037 s does not",
          utc<>::is_representable(seconds(9223372036)) && !utc<>::is_representable(seconds(9223372037)));
    check("is_representable: any ns fits in utc<s>; a wide finer rep is checked",
          utc<seconds>::is_representable(nanoseconds::max()) && utc<days>::is_representable(seconds(86400)));
    std::cout << "range: " << utc<>::min() << " .. " << utc<>::max() << '\n';
    check("min/max are the duration limits", utc<>::min().time_since_epoch() == nanoseconds::min() &&
                                                utc<>::max().time_since_epoch() == nanoseconds::max());

    // Unix numbers floor: 1 ns before the epoch is second -1, not 0.
    const utc<> before = demo::opaque(utc<>::from_unix_nanos(-1));
    check("unix_seconds / millis / micros floor", before.unix_seconds() == -1 && before.unix_millis() == -1 &&
                                                     before.unix_micros() == -1);
    check("unix_millis round-trips", massive.unix_millis() == 1714486985123 && massive.to_unix<milliseconds>() == 1714486985123ms);
    const std::timespec ts = before.to_timespec();
    check("timespec: {-1, 999999999} and back", ts.tv_sec == -1 && ts.tv_nsec == 999999999 && utc<>::from_timespec(ts) == before);

    // std::chrono interop.
    const sys_time<nanoseconds> tp = massive;   // implicit
    check("converts to sys_time and back", tp == massive.sys_time() && utc<>(tp) == massive);
    const utc<> widened = utc<milliseconds>::from_unix_millis(5);   // lossless widening converts implicitly
    check("utc<ms> -> utc<ns> implicitly", widened.unix_nanos() == 5000000);
    check("sys_seconds -> utc<> implicitly", utc<>(sys_seconds(10s)).unix_seconds() == 10);

    // Calendar, in std::chrono types.
    std::cout << "date " << snicholls::to_string(massive.date()) << ", weekday " << massive.weekday().c_encoding()
              << ", hour " << massive.time_of_day().hours().count() << '\n';
    check("date / weekday / days / time_of_day", massive.date() == 2024y / April / 30 && massive.weekday() == Tuesday &&
                                                    massive.days() == sys_days{2024y / April / 30} &&
                                                    massive.time_of_day().subseconds() == 123ms);
    check("floor / ceil / round keep the type", massive.floor<days>() == utc<>(sys_days{2024y / April / 30}) &&
                                                   massive.ceil<days>() == utc<>(sys_days{2024y / May / 1}) &&
                                                   massive.round<seconds>() == utc<>(sys_days{2024y / April / 30} + 14h + 23min + 5s));

    // Arithmetic and comparison, like std::chrono::time_point.
    utc<> t = massive;
    t += 1s;
    t -= 500ms;
    const auto ms_plus_ns = utc<milliseconds>::from_unix_millis(1) + 1ns;   // utc<ns>
    check("+= and -=", t - massive == 500ms);
    check("utc<ms> + 1ns is a utc<ns>", std::is_same_v<decltype(ms_plus_ns), const utc<nanoseconds>> && ms_plus_ns.unix_nanos() == 1000001);
    check("duration + utc, utc - duration", (1h + demo::opaque(utc<>{})).unix_seconds() == 3600 && (utc<>{} - 1h).unix_seconds() == -3600);
    check("compare across resolutions", demo::opaque(utc<milliseconds>::from_unix_millis(1)) == utc<>::from_unix_nanos(1000000) &&
                                           demo::opaque(utc<milliseconds>::from_unix_millis(1)) < utc<>::from_unix_nanos(1000001) &&
                                           (utc<>{} <=> massive) == std::strong_ordering::less);

    // Hashing.
    std::unordered_set<utc<>> seen{massive, massive, t};
    check("std::hash: duplicates collapse", seen.size() == 2);

    std::cout << (failures == 0 ? "OK\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
//! [demo]
