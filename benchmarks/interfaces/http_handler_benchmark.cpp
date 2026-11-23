//
//  http_handler_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  HTTP dispatch three ways on identical requests: the reference route table by the concept, the same table
//  through the virtual twin (one virtual call more), and a hand-written if-chain. The fairness check refuses to run
//  unless all three answer every request identically.
//

#include <snicholls/core/interfaces/http_handler.hpp>
#include <snicholls/core/interfaces/http_handler_reference.hpp>

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>

namespace {

using namespace snicholls;

route_table make_table()
{
    route_table t;
    static_cast<void>(t.add(http_method::get, "/health", [](const http_request&) { return respond(200, "ok"); }));
    static_cast<void>(t.add(http_method::get, "/v1/me", [](const http_request&) { return respond(200, "me"); }));
    static_cast<void>(t.add(http_method::get, "/v1/payments", [](const http_request&) { return respond(200, "payments"); }));
    return t;
}

http_response hand_written(const http_request& r)
{
    if (r.method == http_method::get && r.path == "/health")
        return respond(200, "ok");
    if (r.method == http_method::get && r.path == "/v1/me")
        return respond(200, "me");
    if (r.method == http_method::get && r.path == "/v1/payments")
        return respond(200, "payments");
    return respond(404, "not found");
}

std::array<http_request, 4> requests()
{
    const std::array<std::string, 4> paths{"/health", "/v1/me", "/v1/payments", "/nowhere"};
    std::array<http_request, 4> out;
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i].path = paths[i];
    return out;
}

bool fair()
{
    auto table = make_table();
    http_handler_adapter<route_table&> virt{table};
    for (const http_request& r : requests())
        if (table.handle(r) != hand_written(r) || virt.handle(r) != hand_written(r))
            return false;
    return true;
}

template <http_handler H>
void run(benchmark::State& state, H& h)
{
    const auto rs = requests();
    std::size_t i = 0;
    for (auto _ : state) {
        auto out = h.handle(rs[i++ & 3]);
        benchmark::DoNotOptimize(out);
    }
}

void bm_concept(benchmark::State& state)
{
    auto table = make_table();
    run(state, table);
}
BENCHMARK(bm_concept);

void bm_virtual(benchmark::State& state)
{
    auto table = make_table();
    http_handler_adapter<route_table&> adapter{table};
    http_handler_interface& h = adapter;
    run(state, h);
}
BENCHMARK(bm_virtual);

struct hand {
    http_response handle(const http_request& r) { return hand_written(r); }
};

void bm_hand_written(benchmark::State& state)
{
    hand h;
    run(state, h);
}
BENCHMARK(bm_hand_written);

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "http_handler benchmark: fairness check FAILED - the forms disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
