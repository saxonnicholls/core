//
//  acout_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::acout against a plain, unsynchronised stream, and against
//  std::osyncstream where the library ships it, writing the same 4,096 messages
//  (a label, an integer, a double) to a string stream, single-threaded: the cost
//  of the atomic write itself. main() runs a fairness check first and refuses to
//  benchmark if the texts differ (CLAUDE.md "Zero overhead").
//

#include <snicholls/core/acout.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <version>

#if defined(__cpp_lib_syncbuf)
#include <syncstream>
#endif

namespace {

struct message {
    std::string label;
    int count;
    double price;
};

std::vector<message> make_messages()
{
    std::vector<message> out(4096);
    std::uint64_t x = 0x243f6a8885a308d3ULL;   // a fixed xorshift stream: identical data for all three
    for (auto& m : out) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        m.label = "order-" + std::to_string(x % 1000);
        m.count = static_cast<int>(x % 500);
        m.price = static_cast<double>(x % 100000) / 100.0;
    }
    return out;
}

const std::vector<message> messages = make_messages();

std::string via_plain_stream()
{
    std::ostringstream out;
    for (const auto& m : messages)
        out << m.label << ' ' << m.count << " @ " << m.price << '\n';
    return out.str();
}

std::string via_acout()
{
    std::ostringstream out;
    for (const auto& m : messages)
        snicholls::acout(out) << m.label << ' ' << m.count << " @ " << m.price << '\n';
    return out.str();
}

// std::osyncstream where the library ships it (libstdc++, MSVC; libc++ only as experimental)
#if defined(__cpp_lib_syncbuf)
std::string via_osyncstream()
{
    std::ostringstream out;
    for (const auto& m : messages)
        std::osyncstream(out) << m.label << ' ' << m.count << " @ " << m.price << '\n';
    return out.str();
}
#else
std::string via_osyncstream() { return via_plain_stream(); }
#endif

std::string via_print()
{
    std::ostringstream out;
    for (const auto& m : messages)
        snicholls::println(out, "{} {} @ {}", m.label, m.count, m.price);
    return out.str();
}

void BM_Acout_Core(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(via_acout());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(messages.size()));
}
BENCHMARK(BM_Acout_Core);

#if defined(__cpp_lib_syncbuf)
void BM_Acout_Osyncstream(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(via_osyncstream());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(messages.size()));
}
BENCHMARK(BM_Acout_Osyncstream);
#endif

void BM_Acout_PlainStreamUnsafe(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(via_plain_stream());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(messages.size()));
}
BENCHMARK(BM_Acout_PlainStreamUnsafe);

void BM_Acout_Println(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(via_print());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(messages.size()));
}
BENCHMARK(BM_Acout_Println);

// The engine on a container: a vector of 16 ints per message, against the same loop written out
const std::vector<std::vector<int>> rows = [] {
    std::vector<std::vector<int>> out(messages.size());
    int n = 0;
    for (auto& row : out)
        for (int i = 0; i < 16; ++i)
            row.push_back(n++ % 1000);
    return out;
}();

std::string rows_via_engine()
{
    std::ostringstream out;
    for (const auto& row : rows) {
        snicholls::print_to(out, row);
        out << '\n';
    }
    return out.str();
}

std::string rows_by_hand()
{
    std::ostringstream out;
    for (const auto& row : rows) {
        out << '[';
        const char* separator = "";
        for (const int v : row) {
            out << separator << v;
            separator = ", ";
        }
        out << "]\n";
    }
    return out.str();
}

void BM_Acout_EngineVector(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(rows_via_engine());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows.size()));
}
BENCHMARK(BM_Acout_EngineVector);

void BM_Acout_HandWrittenVector(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(rows_by_hand());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(rows.size()));
}
BENCHMARK(BM_Acout_HandWrittenVector);

// A coloured label per message, against the same escape codes written out by hand
std::string coloured_via_acout()
{
    std::ostringstream out;
    for (const auto& m : messages)
        snicholls::acout(out) << snicholls::styled(snicholls::colours::green, m.label) << ' ' << m.count << " @ " << m.price << '\n';
    return out.str();
}

std::string coloured_by_hand()
{
    const std::string_view green = "\x1b[32m";
    std::ostringstream out;
    for (const auto& m : messages)
        out << green << m.label << snicholls::ansi_reset << ' ' << m.count << " @ " << m.price << '\n';
    return out.str();
}

void BM_Acout_Coloured(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(coloured_via_acout());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(messages.size()));
}
BENCHMARK(BM_Acout_Coloured);

void BM_Acout_ColouredByHand(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(coloured_by_hand());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(messages.size()));
}
BENCHMARK(BM_Acout_ColouredByHand);

bool fair()
{
    const std::string expected = via_plain_stream();
    return via_acout() == expected && via_osyncstream() == expected && via_print() == expected &&
           rows_via_engine() == rows_by_hand() && coloured_via_acout() == coloured_by_hand();
}

} // namespace

int main(int argc, char** argv)
{
    snicholls::set_colour_mode(snicholls::colour_mode::always);   // the targets are string streams
    if (!fair()) {
        std::fprintf(stderr, "acout benchmark: fairness check FAILED - the texts differ; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
