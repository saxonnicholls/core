//
//  stages_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/pipeline/stages.hpp>.
//

// Demo: components/emitter_observer/stages.hpp - zero-cost, compile-time observer chains.
#include <cstddef>
#include <cstdio>
#include <functional>
#include <type_traits>
#include <vector>

#include <snicholls/core/pipeline/stages.hpp>

// Print a step, and remember any result that is not what the demo documents.
namespace demo {
inline int failures = 0;
inline void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
inline int result() { return failures == 0 ? 0 : 1; }
} // namespace demo

using namespace snicholls::pipeline;

//! [demo]
int main()
{
    std::printf("filter | transform | tap | sink: one flat struct of inlined calls\n");
    std::vector<int> out;
    int tapped = 0;
    auto chain = filter([](int v) { return v % 2 == 0; }) | transform([](int v) { return v * 10; }) |
                 tap([&](int) { ++tapped; }) | [&](int v) { out.push_back(v); };
    for (int i = 0; i < 6; ++i) chain(i);
    demo::check(out == std::vector<int>{0, 20, 40} && tapped == 3, "evens, times ten, tapped three times");
    static_assert(!std::is_polymorphic_v<decltype(chain)>);

    std::printf("batch<T>(n): vectors of n, flush() for the remainder\n");
    std::vector<std::size_t> sizes;
    auto batched = batch<int>(4) | [&](const std::vector<int>& b) { sizes.push_back(b.size()); };
    for (int i = 0; i < 10; ++i) batched(i);
    batched.flush();
    demo::check(sizes == std::vector<std::size_t>{4, 4, 2}, "batches of 4, 4 and 2");
    auto ones = batch<int>(0) | [&](const std::vector<int>& b) { sizes.push_back(b.size()); };
    ones(1);
    demo::check(sizes.back() == 1, "batch size 0 means 1");

    std::printf("a non-copyable sink by reference\n");
    struct counter
    {
        int n = 0;
        void operator()(int) { ++n; }
    } c;
    auto by_ref = filter([](int v) { return v > 0; }) | std::ref(c);
    by_ref(1);
    by_ref(-1);
    demo::check(c.n == 1, "one positive counted");
    return demo::result();
}
//! [demo]
