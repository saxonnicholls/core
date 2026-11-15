//
//  normal_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/stats/normal.hpp>: standard normal draws by the inverse CDF and by Box-Muller from the same
//  generator, their moments, and a sampler of the caller's own behind the same concept. Exits non-zero if a result is
//  not what it documents.
//

#include <snicholls/core/stats/normal.hpp>

#include <snicholls/core/stats/random.hpp>

#include <cmath>
#include <cstdio>

namespace {

namespace st = snicholls::stats;

//! [normal]
// Any method is a value behind one concept, so the code that draws does not change when the method does
template <st::normal_sampler<st::wyrand> S>
void moments(S sampler, double& mean, double& variance, int n)
{
    st::wyrand g(7);   // the same generator, the same seed
    double m1 = 0, m2 = 0;
    for (int i = 0; i < n; ++i) {
        const double z = sampler(g);
        m1 += z;
        m2 += z * z;
    }
    mean = m1 / n;
    variance = m2 / n - mean * mean;
}

// A method of the caller's own: draws clipped at 4 sigma (a truncated normal for stress scenarios, say)
struct clipped {
    st::normal_inverse_cdf base{};
    double operator()(st::wyrand& g) const
    {
        const double z = base(g);
        return z > 4 ? 4 : z < -4 ? -4 : z;
    }
};
//! [normal]

} // namespace

int main()
{
    std::printf("snicholls-core normal demo\n");
    bool ok = true;
    constexpr int n = 200000;
    double mi = 0, vi = 0, mb = 0, vb = 0, mc = 0, vc = 0;
    moments(st::normal_inverse_cdf{}, mi, vi, n);
    moments(st::normal_box_muller{}, mb, vb, n);
    moments(clipped{}, mc, vc, n);
    std::printf("  inverse CDF: mean %+.4f, variance %.4f\n", mi, vi);
    std::printf("  Box-Muller:  mean %+.4f, variance %.4f\n", mb, vb);
    std::printf("  clipped:     mean %+.4f, variance %.4f\n", mc, vc);
    for (const double m : {mi, mb, mc})
        ok &= std::abs(m) < 0.01;
    for (const double v : {vi, vb, vc})
        ok &= std::abs(v - 1) < 0.02;
    std::printf(ok ? "normal demo: ok\n" : "normal demo: FAILED\n");
    return ok ? 0 : 1;
}
