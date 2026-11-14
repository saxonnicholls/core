//
//  muon_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/muon.hpp>: an ill-conditioned gradient orthogonalised (its singular values pulled
//  together), and Muon steps fitting a matrix. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/muon.hpp>

#include <snicholls/core/linalg/reference.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core muon demo\n");
    namespace ml = snicholls::ml;
    namespace la = snicholls::linalg;
    bool ok = true;

    //! [muon]
    // A 2 x 3 gradient whose rows differ in size by 100: plain SGD would move almost only along the first
    const std::vector<float> g{10.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f};
    std::vector<float> x(g.size()), work(ml::muon_work_size(2, 3));
    ml::muon_orthogonalise(la::reference, std::span<const float>(g), 2, 3, std::span<float>(x), std::span<float>(work));

    // Fitting W to a target by Muon steps on the gradient of 1/2 |W - target|^2, which is W - target
    const std::vector<float> target{1.0f, -1.0f, 0.5f, 0.0f, 2.0f, -0.5f};
    std::vector<float> w(6, 0.0f), momentum(6, 0.0f), grad(6);
    auto distance = [&] {
        float d = 0.0f;
        for (std::size_t i = 0; i < w.size(); ++i)
            d += (w[i] - target[i]) * (w[i] - target[i]);
        return std::sqrt(d);
    };
    const float before = distance();
    for (int step = 0; step < 30; ++step) {
        for (std::size_t i = 0; i < w.size(); ++i)
            grad[i] = w[i] - target[i];
        ml::muon_step(la::reference, std::span<float>(w), std::span<const float>(grad), std::span<float>(momentum), 2, 3,
                      ml::muon_config<float>{.lr = 0.1f, .momentum = 0.9f}, std::span<float>(work));
    }
    const float after = distance();
    //! [muon]

    std::printf("  gradient rows 10 and 0.1 become %.3f and %.3f\n", static_cast<double>(x[0]), static_cast<double>(x[4]));
    std::printf("  distance to the target: %.4f before, %.4f after 30 steps\n", static_cast<double>(before), static_cast<double>(after));
    ok &= x[0] > 0.6f && x[0] < 1.25f && x[4] > 0.6f && x[4] < 1.25f;   // both singular values near 1
    ok &= after < 0.25f * before;
    std::printf(ok ? "muon demo: ok\n" : "muon demo: FAILED\n");
    return ok ? 0 : 1;
}
