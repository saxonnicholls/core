//
//  payment_provider_reference_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Webhook verification, the payment provider's hot path: the reference's verify_webhook (the header, HMAC,
//  a constant-time compare, the time window, the body) against the hand-written check of the tag alone, on one
//  delivery. The difference is the price of the parsing. The fairness check refuses to run unless both accept.
//

#include <snicholls/core/hash/hmac.hpp>
#include <snicholls/core/hash/secure.hpp>
#include <snicholls/core/hash/sha2.hpp>
#include <snicholls/core/interfaces/payment_provider_reference.hpp>
#include <snicholls/core/money/asset.hpp>

#include <benchmark/benchmark.h>

#include <cstdio>
#include <string>
#include <string_view>

namespace {

using namespace snicholls;

const utc_s at = utc_s::from_unix_seconds(1'800'000'000);
const std::string secret_text = "whsec_bench_only";

memory_payment_provider make() { return {{checked_asset("AUD", 2)}, secret<std::string>{secret_text}, [] { return at; }}; }

owned_delivery delivery(const memory_payment_provider& p)
{
    webhook_event e;
    e.event_id = "evt_1NG8Du2eZvKYlo2CUI79vXWy";
    e.kind = event_kind::payment_updated;
    e.type = "checkout.session.completed";
    e.payment_id = "pi_3NG8Du2eZvKYlo2C0ZKZ8ZqL";
    e.our_reference = "order-123456";
    e.occurred_at = at;
    return p.signed_webhook(e);
}

bool hand_written(const owned_delivery& d)
{
    const std::string_view h = d.headers[0].value;
    const auto comma = h.find(',');
    const std::string_view t = h.substr(2, comma - 2), tag = h.substr(comma + 4);
    hmac<sha256> m{secret_text};
    m.update(t);
    m.update(std::string_view{"."});
    m.update(d.body);
    return constant_time_equal(m.finish().hex().view(), tag);
}

bool fair()
{
    const auto p = make();
    const auto d = delivery(p);
    return p.verify_webhook(d.view()).is_ok() && hand_written(d);
}

void bm_verify_webhook(benchmark::State& state)
{
    const auto p = make();
    const auto d = delivery(p);
    for (auto _ : state) {
        auto e = p.verify_webhook(d.view());
        benchmark::DoNotOptimize(e);
    }
}
BENCHMARK(bm_verify_webhook);

void bm_hand_written_tag_check(benchmark::State& state)
{
    const auto p = make();
    const auto d = delivery(p);
    for (auto _ : state) {
        bool ok = hand_written(d);
        benchmark::DoNotOptimize(ok);
    }
}
BENCHMARK(bm_hand_written_tag_check);

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "payment_provider_reference benchmark: fairness check FAILED - the two checks disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
