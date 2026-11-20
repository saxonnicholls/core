//
//  mixer_fixture.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Shared by the tests of the linear-time sequence mixers (ml/recurrence.hpp, ml/short_conv.hpp, ml/ssd.hpp,
//  ml/gla.hpp, ml/rwkv.hpp, ml/delta_rule.hpp, ml/mixer_layer.hpp): deterministic data, the largest difference between
//  two vectors, and a central difference of a loss in one element of a vector.
//

#ifndef snicholls_core_tests_support_mixer_fixture_hpp
#define snicholls_core_tests_support_mixer_fixture_hpp

#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace snicholls::core_testing
{

/// Uniform values in [lo, hi), from a fixed stream.
template <typename T>
std::vector<T> uniform(std::size_t n, std::uint64_t seed, double lo = -1.0, double hi = 1.0)
{
    auto g = snicholls::stats::wyrand::stream(seed, 0);
    std::vector<T> v(n);
    for (auto& e : v)
        e = static_cast<T>(lo + (hi - lo) * snicholls::stats::uniform_real(g));
    return v;
}

/// The largest absolute difference, scaled by the largest magnitude when that exceeds 1.
template <typename A, typename B>
double max_relative_difference(const A& a, const B& b)
{
    double d = 0, m = 1;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
        m = std::max(m, std::fabs(static_cast<double>(a[i])));
    }
    return d / m;
}

/// `(loss(v + h e_i) - loss(v - h e_i)) / 2h`, `v` restored.
template <typename F>
double central_difference(std::vector<double>& v, std::size_t i, F&& loss, double h = 1e-6)
{
    const double keep = v[i];
    v[i] = keep + h;
    const double up = loss();
    v[i] = keep - h;
    const double down = loss();
    v[i] = keep;
    return (up - down) / (2 * h);
}

/// `Σ a_i b_i`.
template <typename A, typename B>
double weighted(const A& a, const B& b)
{
    double s = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return s;
}

} // namespace snicholls::core_testing

#endif // snicholls_core_tests_support_mixer_fixture_hpp
