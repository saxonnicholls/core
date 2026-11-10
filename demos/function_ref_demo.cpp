//
//  function_ref_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/function_ref.hpp>: a non-template function that
//  takes any callable as a parameter, with no allocation and no copy of the
//  callable. Exits non-zero if any result is not the one documented.
//

#include <snicholls/core/function_ref.hpp>

#include <cstddef>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {

//! [parameter]
// Not a template: one definition, whatever callable the caller passes.
int count_if(const std::vector<int>& values, snicholls::function_ref<bool(int)> keep)
{
    int n = 0;
    for (int v : values)
        if (keep(v))
            ++n;
    return n;
}
//! [parameter]

// A void signature discards the callable's result.
void for_each_word(std::string_view text, snicholls::function_ref<void(std::string_view)> visit)
{
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = text.find(' ', start);
        if (end == std::string_view::npos)
            end = text.size();
        if (end > start)
            visit(text.substr(start, end - start));
        start = end + 1;
    }
}

struct tally {
    int calls = 0;
    std::size_t operator()(std::string_view word) { ++calls; return word.size(); }   // result is discarded
};

} // namespace

int main()
{
    std::printf("snicholls-core function_ref demo\n");
    bool ok = true;

    const std::vector<int> values{1, 5, 8, 12, 3, 20};

    // A lambda written in the argument list: valid for exactly that call
    const int big = count_if(values, [](int v) { return v > 4; });
    std::printf("  values > 4: %d\n", big);
    ok &= big == 4;

    // A capturing lambda: the reference sees the captured state, not a copy of it
    int threshold = 10;
    const auto above = [&threshold](int v) { return v > threshold; };
    ok &= count_if(values, above) == 2;
    threshold = 0;
    ok &= count_if(values, above) == 6;
    std::printf("  above a threshold that changes between calls: 2, then 6\n");

    // A stateful function object: every copy of the function_ref calls the same object
    tally t;
    for_each_word("the quick brown fox", t);
    std::printf("  words visited: %d\n", t.calls);
    ok &= t.calls == 4;

    // Copies refer to the same callable
    const snicholls::function_ref<void(std::string_view)> a = t;
    const snicholls::function_ref<void(std::string_view)> b = a;
    b("jumps");
    ok &= t.calls == 5;

    // A result convertible to R is converted
    const auto length = [](std::string_view s) { return static_cast<int>(s.size()); };
    const snicholls::function_ref<long(std::string_view)> as_long = length;
    ok &= as_long("four") == 4L;

    std::printf(ok ? "function_ref demo: ok\n" : "function_ref demo: FAILED\n");
    return ok ? 0 : 1;
}
