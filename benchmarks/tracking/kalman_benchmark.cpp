//
//  kalman_benchmark.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  snicholls::tracking::predict and update for a 2-D constant-velocity target, (x, ẋ, y, ẏ), with a position sensor,
//  against the same arithmetic written by hand over raw double arrays, in the same order: F and Q formed once for the
//  shared dt, then F x and F P Fᵀ + Q (the lower triangle mirrored), then S = H P Hᵀ + R, P Hᵀ, the 2 x 2 Cholesky of S, the gain, the innovation, the
//  squared distance, the log-likelihood, x + K ν and P - K S Kᵀ symmetrised, each written to the update record that
//  tracking::update returns. 4096 tracks per iteration. main() runs a fairness check first and refuses to benchmark
//  unless every record agrees to the bit.
//

#include <snicholls/core/tracking/kalman.hpp>

#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>
#include <vector>

namespace {

namespace la = snicholls::linalg;
namespace st = snicholls::stats;
namespace tr = snicholls::tracking;
using cv2 = tr::combined<tr::constant_velocity<double>, tr::constant_velocity<double>>;
using position = tr::linear_gaussian<double, 4, std::array<std::size_t, 2>{0, 2}>;
using state = tr::gaussian<double, 4>;
using record = tr::kalman_update<double, 4, 2>;
constexpr std::size_t n = 4096;
constexpr double dt = 0.5, q = 0.05, r = 0.25;

struct scenario {
    std::vector<state> priors;
    std::vector<la::fixed_vector<double, 2>> z;
};

const scenario& data()
{
    static const scenario s = [] {
        scenario out{std::vector<state>(n), std::vector<la::fixed_vector<double, 2>>(n)};
        st::wyrand g(9);
        for (std::size_t k = 0; k < n; ++k) {
            out.priors[k].mean = la::make_vector(st::uniform_real(g) * 100, st::uniform_real(g), st::uniform_real(g) * 100, st::uniform_real(g));
            out.priors[k].covariance = la::diagonal_matrix(la::make_vector(1.0 + st::uniform_real(g), 0.5, 1.0 + st::uniform_real(g), 0.5));
            out.z[k] = la::make_vector(out.priors[k].mean[0] + st::uniform_real(g), out.priors[k].mean[2] - st::uniform_real(g));
        }
        return out;
    }();
    return s;
}

void core(std::vector<record>& out)
{
    const auto& d = data();
    const cv2 model = tr::combine(tr::constant_velocity<double>{q}, tr::constant_velocity<double>{q});
    const position sensor{la::diagonal_matrix(la::make_vector(r, r))};
    // Every track steps over the same dt, so F and Q are formed once, as the hand-written loop forms them
    const auto f = model.matrix(dt);
    const auto qm = model.covariance(dt);
    for (std::size_t k = 0; k < n; ++k)
        if (const auto u = tr::update(tr::predict(d.priors[k], f, qm), d.z[k], sensor))
            out[k] = *u;
}

// x ← A x for 4 x 4 A; P ← A P Aᵀ with the header's sandwich order (A P first, then the lower triangle mirrored)
void by_hand(std::vector<record>& out)
{
    const auto& d = data();
    const double f[16] = {1, dt, 0, 0, 0, 1, 0, 0, 0, 0, 1, dt, 0, 0, 0, 1};
    // constant velocity's Q per axis, as the model forms it: q dt³/3, q dt²/2, q dt
    const double q00 = q * (dt * dt * dt) / (1.0 * 1.0 * 3.0), q01 = q * (dt * dt) / (1.0 * 1.0 * 2.0), q11 = q * dt / (1.0 * 1.0 * 1.0);
    const double qm[16] = {q00, q01, 0, 0, q01, q11, 0, 0, 0, 0, q00, q01, 0, 0, q01, q11};
    const double log_2pi = std::log(2.0 * static_cast<double>(std::numbers::pi_v<long double>));
    for (std::size_t k = 0; k < n; ++k) {
        const double* x0 = d.priors[k].mean.elements.data();
        const double* p0 = d.priors[k].covariance.elements.data();
        // predict
        double x[4] = {};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                x[i] += f[i * 4 + j] * x0[j];
        double fp[16] = {};
        for (int i = 0; i < 4; ++i)
            for (int c = 0; c < 4; ++c) {
                const double a = f[i * 4 + c];
                for (int j = 0; j < 4; ++j)
                    fp[i * 4 + j] += a * p0[c * 4 + j];
            }
        double p[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j <= i; ++j) {
                double s = 0;
                for (int c = 0; c < 4; ++c)
                    s += fp[i * 4 + c] * f[j * 4 + c];
                p[i * 4 + j] = s;
                p[j * 4 + i] = s;
            }
        for (int i = 0; i < 16; ++i)
            p[i] += qm[i];
        // S = H P Hᵀ + R and P Hᵀ, H selecting components 0 and 2
        const int h[2] = {0, 2};
        double s2[4];
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j <= i; ++j) {
                // H P computed as the product with a 0/1 matrix, as fixed_matrix's product does
                double hp[4] = {};
                for (int c = 0; c < 4; ++c) {
                    const double a = c == h[i] ? 1.0 : 0.0;
                    for (int jj = 0; jj < 4; ++jj)
                        hp[jj] += a * p[c * 4 + jj];
                }
                double v = 0;
                for (int c = 0; c < 4; ++c)
                    v += hp[c] * (c == h[j] ? 1.0 : 0.0);
                s2[i * 2 + j] = v;
                s2[j * 2 + i] = v;
            }
        s2[0] += r;
        s2[3] += r;
        double pht[8] = {};
        for (int i = 0; i < 4; ++i)
            for (int c = 0; c < 4; ++c) {
                const double a = p[i * 4 + c];
                for (int j = 0; j < 2; ++j)
                    pht[i * 2 + j] += a * (c == h[j] ? 1.0 : 0.0);
            }
        // Cholesky of S
        const double scale = std::fabs(s2[0]) > std::fabs(s2[3]) ? std::fabs(s2[0]) : std::fabs(s2[3]);
        const double tol = std::numeric_limits<double>::epsilon() * 2.0 * scale;
        double l[4] = {};
        {
            double d0 = s2[0];
            if (!(d0 > tol))
                continue;
            l[0] = std::sqrt(d0);
            l[2] = s2[2] / l[0];
            double d1 = s2[3] - l[2] * l[2];
            if (!(d1 > tol))
                continue;
            l[3] = std::sqrt(d1);
        }
        // K = (S⁻¹ (P Hᵀ)ᵀ)ᵀ: forward then back substitution on each of the 4 columns of (P Hᵀ)ᵀ
        double kt[8];  // 2 x 4
        for (int j = 0; j < 4; ++j) {
            double y0 = pht[j * 2 + 0] / l[0];
            double y1 = (pht[j * 2 + 1] - l[2] * y0) / l[3];
            const double z1 = y1 / l[3];
            const double z0 = (y0 - l[2] * z1) / l[0];
            kt[0 * 4 + j] = z0;
            kt[1 * 4 + j] = z1;
        }
        double kg[8];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 2; ++j)
                kg[i * 2 + j] = kt[j * 4 + i];
        // innovation (linear), its whitened norm and log-likelihood (computed and kept, as the header does)
        const double zp[2] = {x[0], x[2]};
        const double nu[2] = {d.z[k][0] - zp[0], d.z[k][1] - zp[1]};
        const double w0 = nu[0] / l[0];
        const double w1 = (nu[1] - l[2] * w0) / l[3];
        const double d2 = w0 * w0 + w1 * w1;
        const double ll = -(d2 + 2.0 * (std::log(l[0]) + std::log(l[3])) + 2.0 * log_2pi) / 2.0;
        // the whole record, as update returns it
        record& rec = out[k];
        rec.innovation = la::make_vector(nu[0], nu[1]);
        std::memcpy(rec.innovation_covariance.elements.data(), s2, sizeof s2);
        std::memcpy(rec.gain.elements.data(), kg, sizeof kg);
        rec.distance2 = d2;
        rec.log_likelihood = ll;
        state& o = rec.posterior;
        for (int i = 0; i < 4; ++i) {
            double s = 0;
            for (int j = 0; j < 2; ++j)
                s += kg[i * 2 + j] * nu[j];
            o.mean[static_cast<std::size_t>(i)] = x[i] + s;
        }
        double ks[8] = {};
        for (int i = 0; i < 4; ++i)
            for (int c = 0; c < 2; ++c) {
                const double a = kg[i * 2 + c];
                for (int j = 0; j < 2; ++j)
                    ks[i * 2 + j] += a * s2[c * 2 + j];
            }
        double kskt[16];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j <= i; ++j) {
                double s = 0;
                for (int c = 0; c < 2; ++c)
                    s += ks[i * 2 + c] * kg[j * 2 + c];
                kskt[i * 4 + j] = s;
                kskt[j * 4 + i] = s;
            }
        double pc[16];
        for (int i = 0; i < 16; ++i)
            pc[i] = p[i] - kskt[i];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                o.covariance(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = i == j ? pc[i * 4 + j] : (j < i ? (pc[i * 4 + j] + pc[j * 4 + i]) / 2.0 : (pc[j * 4 + i] + pc[i * 4 + j]) / 2.0);
    }
}

template <void (*Run)(std::vector<record>&)>
void BM_Kalman(benchmark::State& st_)
{
    std::vector<record> out(n);
    data();
    for (auto _ : st_) {
        Run(out);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }
    st_.SetItemsProcessed(st_.iterations() * std::int64_t(n));
}
BENCHMARK(BM_Kalman<core>)->Name("BM_KalmanPredictUpdateCV2D_Core")->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Kalman<by_hand>)->Name("BM_KalmanPredictUpdateCV2D_HandWritten")->Unit(benchmark::kMicrosecond);

bool fair()
{
    std::vector<record> a(n), b(n);
    core(a);
    by_hand(b);
    return std::memcmp(a.data(), b.data(), n * sizeof(record)) == 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "kalman benchmark: fairness check FAILED - the pairs disagree; not benchmarking\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
