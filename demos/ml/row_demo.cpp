//
//  row_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/row.hpp>: a function that takes any row, called with a vector, an array
//  and a span. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/row.hpp>

#include <array>
#include <cstdio>
#include <span>
#include <vector>

//! [row]
// A function over rows takes any contiguous range of numbers and works on a span inside.
template <snicholls::ml::row R>
constexpr auto total(const R& r)
{
    snicholls::ml::row_value_t<R> s{};
    for (const auto x : snicholls::ml::as_row(r))
        s += x;
    return s;
}

template <snicholls::ml::mutable_row R>
constexpr void scale(R&& r, snicholls::ml::row_value_t<R> by)
{
    for (auto& x : snicholls::ml::as_mutable_row(r))
        x *= by;
}
//! [row]

int main()
{
    std::printf("snicholls-core row demo\n");
    bool ok = true;

    std::vector<float> v{1, 2, 3};
    constexpr std::array<double, 4> a{0.5, 0.25, 0.125, 0.125};
    scale(v, 2.0f);
    const std::span<const float> s(v.data() + 1, 2);
    std::printf("  vector %.1f, array %.3f, span %.1f\n", static_cast<double>(total(v)), total(a), static_cast<double>(total(s)));
    ok &= total(v) == 12.0f && total(a) == 1.0 && total(s) == 10.0f;
    static_assert(total(std::array<double, 2>{1.5, 2.5}) == 4.0);   // and at compile time

    std::printf(ok ? "row demo: ok\n" : "row demo: FAILED\n");
    return ok ? 0 : 1;
}
