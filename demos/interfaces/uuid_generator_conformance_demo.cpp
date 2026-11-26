//
//  uuid_generator_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/uuid_generator_conformance.hpp>: the kit run on the reference, with a probe whose
//  random source can fail and whose clock can step back. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/uuid_generator_conformance.hpp>
#include <snicholls/core/interfaces/uuid_generator_reference.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

struct source {
    stats::wyrand r{1};
    bool fails = false;
    std::optional<std::uint64_t> operator()() noexcept { return fails ? std::nullopt : std::optional<std::uint64_t>{r()}; }
};
struct clock {
    utc_ms t = utc_ms::from_unix_millis(1'800'000'000'000);
    [[nodiscard]] utc_ms now() const noexcept { return t; }
};

struct probe {
    basic_uuid_generator<source, clock> g{source{}, clock{}};
    auto& generator() { return g; }
    std::int64_t clock_ms() { return g.clock().now().unix_millis(); }
    void clock_steps_back() { g.clock().t -= std::chrono::seconds{1}; }
    void randomness_fails() { g.random().fails = true; }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/uuid_generator_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_uuid_generator_conformance<probe>();
    //! [kit]
    check(report.ok(), "the reference passes");
    check(report.checks == 5, "five checks ran");
    return failures == 0 ? 0 : 1;
}
