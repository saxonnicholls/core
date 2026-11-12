//
//  reference_benchmark.cpp
//  snicholls-core
//
//  The concept path against hand-written code: snicholls::linalg::dot and axpy (4096 elements), gemv, gemm and
//  syrk (64 x 64), triangular_solve, lu_factor + lu_solve, cholesky_factor + cholesky_solve and ldlt_factor +
//  ldlt_solve (order 64), least_squares and qr_q (96 x 32),
//  symmetric_eigen (16) and svd (24 x 16), called through the free functions with the reference backend, against
//  the same algorithms over raw pointers, identical arithmetic in the same order. The fairness check refuses to
//  benchmark unless every pair's results are bit-for-bit equal.
//

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/stats/random.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace la = snicholls::linalg;

constexpr std::size_t vector_length = 4096;
std::size_t n = 64;   // a run-time order, as a caller's is: neither side gets it as a constant

std::vector<double> filled(std::size_t count, double salt)
{
    std::vector<double> v(count);
    for (std::size_t i = 0; i < count; ++i)
        v[i] = std::sin(static_cast<double>(i) * 0.37 + salt);
    return v;
}

std::vector<double> x = filled(vector_length, 0.1), y = filled(vector_length, 0.2);   // not const: nothing folds
std::vector<double> a = filled(64 * 64, 0.3), b = filled(64 * 64, 0.4), c(64 * 64);
std::vector<double> system0 = [] {
    auto s = filled(64 * 64, 0.5);
    for (std::size_t i = 0; i < 64; ++i)
        s[i * 64 + i] += 4.0;
    return s;
}();
std::vector<double> rhs0 = filled(64, 0.6), lu(64 * 64), sol(64);
std::vector<la::pivot_index> piv(64);

// ---- dot ----------------------------------------------------------------------------------------------------

double dot_core()
{
    return la::dot(la::reference, la::vector_view<const double, la::layout_right>(x.data(), {vector_length}),
                   la::vector_view<const double, la::layout_right>(y.data(), {vector_length}))
        .unwrap_or(0.0);
}
double dot_hand()
{
    double s = 0;
    for (std::size_t i = 0; i < vector_length; ++i)
        s += x[i] * y[i];
    return s;
}

// ---- gemm ---------------------------------------------------------------------------------------------------

void gemm_core()
{
    static_cast<void>(la::gemm(la::reference, 1.0, la::matrix_view<const double, la::layout_right>(a.data(), {n, n}),
                               la::matrix_view<const double, la::layout_right>(b.data(), {n, n}), 0.0,
                               la::matrix_view<double, la::layout_right>(c.data(), {n, n})));
}
void gemm_hand()
{
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0;
            for (std::size_t p = 0; p < n; ++p)
                s += a[i * n + p] * b[p * n + j];
            c[i * n + j] = 1.0 * s;
        }
}

// ---- LU -----------------------------------------------------------------------------------------------------

void restore()
{
    std::memcpy(lu.data(), system0.data(), lu.size() * sizeof(double));
    std::memcpy(sol.data(), rhs0.data(), sol.size() * sizeof(double));
}

bool lu_core()
{
    restore();
    const la::matrix_view<double, la::layout_right> m(lu.data(), {n, n});
    return la::lu_factor(la::reference, m, std::span<la::pivot_index>(piv)) &&
           la::lu_solve(la::reference, m, std::span<const la::pivot_index>(piv), la::vector_view<double, la::layout_right>(sol.data(), {n}));
}

bool lu_hand()
{
    restore();
    double* m = lu.data();
    double scale = 0;
    for (std::size_t i = 0; i < n * n; ++i) {
        const double v = std::fabs(m[i]);
        if (v > scale || v != v)
            scale = v;
    }
    const double tolerance = std::numeric_limits<double>::epsilon() * static_cast<double>(n) * scale;
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t p = k;
        double largest = std::fabs(m[k * n + k]);
        for (std::size_t i = k + 1; i < n; ++i)
            if (std::fabs(m[i * n + k]) > largest) {
                largest = std::fabs(m[i * n + k]);
                p = i;
            }
        piv[k] = static_cast<la::pivot_index>(p);
        if (!(largest > tolerance))
            return false;
        if (p != k)
            for (std::size_t j = 0; j < n; ++j) {
                const double t = m[k * n + j];
                m[k * n + j] = m[p * n + j];
                m[p * n + j] = t;
            }
        const double pivot = m[k * n + k];
        if (std::fabs(pivot) >= std::numeric_limits<double>::min()) {
            const double inverse = 1.0 / pivot;
            for (std::size_t i = k + 1; i < n; ++i)
                m[i * n + k] *= inverse;
        } else {
            for (std::size_t i = k + 1; i < n; ++i)
                m[i * n + k] = m[i * n + k] / pivot;
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double l = m[i * n + k];
            for (std::size_t j = k + 1; j < n; ++j)
                m[i * n + j] -= l * m[k * n + j];
        }
    }
    for (std::size_t k = 0; k < n; ++k) {
        const auto p = static_cast<std::size_t>(piv[k]);
        if (p != k) {
            const double t = sol[k];
            sol[k] = sol[p];
            sol[p] = t;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        double s = sol[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= m[i * n + j] * sol[j];
        sol[i] = s;
    }
    for (std::size_t i = n; i-- > 0;) {
        double s = sol[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= m[i * n + j] * sol[j];
        sol[i] = s / m[i * n + i];
    }
    return true;
}

// ---- Cholesky (order 64) ------------------------------------------------------------------------------------

std::vector<double> spd0 = [] {   // G G^T + 64 I
    const auto g = filled(64 * 64, 0.7);
    std::vector<double> s(64 * 64);
    for (std::size_t i = 0; i < 64; ++i)
        for (std::size_t j = 0; j < 64; ++j) {
            double t = 0;
            for (std::size_t k = 0; k < 64; ++k)
                t += g[i * 64 + k] * g[j * 64 + k];
            s[i * 64 + j] = t + (i == j ? 64.0 : 0.0);
        }
    return s;
}();
std::vector<double> chol(64 * 64), chol_sol(64);

void restore_cholesky()
{
    std::memcpy(chol.data(), spd0.data(), chol.size() * sizeof(double));
    std::memcpy(chol_sol.data(), rhs0.data(), chol_sol.size() * sizeof(double));
}

bool cholesky_core()
{
    restore_cholesky();
    const la::matrix_view<double, la::layout_right> m(chol.data(), {n, n});
    return la::cholesky_factor(la::reference, m) &&
           la::cholesky_solve(la::reference, m, la::vector_view<double, la::layout_right>(chol_sol.data(), {n}));
}

bool cholesky_hand()
{
    restore_cholesky();
    double* m = chol.data();
    double* b = chol_sol.data();
    double scale = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = m[i * n + i];
        if (d > scale || d != d)
            scale = d;
    }
    const double tolerance = std::numeric_limits<double>::epsilon() * static_cast<double>(n) * scale * 1.0;
    for (std::size_t j = 0; j < n; ++j) {
        double d = m[j * n + j];
        for (std::size_t k = 0; k < j; ++k)
            d -= m[j * n + k] * m[j * n + k];
        if (!(d > tolerance))
            return false;
        const double root = std::sqrt(d);
        m[j * n + j] = root;
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = m[i * n + j];
            for (std::size_t k = 0; k < j; ++k)
                s -= m[i * n + k] * m[j * n + k];
            m[i * n + j] = s / root;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= m[i * n + j] * b[j];
        b[i] = s / m[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= m[j * n + i] * b[j];
        b[i] = s / m[i * n + i];
    }
    return true;
}

// ---- QR least squares (96 x 32) -----------------------------------------------------------------------------

std::size_t qm = 96, qn = 32;
std::vector<double> design0 = [] {   // random, full rank (a sine of a linear index would have rank 2)
    snicholls::stats::wyrand g(96);
    std::vector<double> d(96 * 32);
    for (auto& v : d)
        v = 2.0 * snicholls::stats::uniform_real(g) - 1.0;
    return d;
}();
std::vector<double> obs0 = filled(96, 0.9), qr(96 * 32), qr_obs(96), qr_tau(32);

void restore_qr()
{
    std::memcpy(qr.data(), design0.data(), qr.size() * sizeof(double));
    std::memcpy(qr_obs.data(), obs0.data(), qr_obs.size() * sizeof(double));
}

bool qr_core()
{
    restore_qr();
    return static_cast<bool>(la::least_squares(la::reference, la::matrix_view<double, la::layout_right>(qr.data(), {qm, qn}), std::span<double>(qr_tau),
                                               la::vector_view<double, la::layout_right>(qr_obs.data(), {qm})));
}

// The column norm below row `first`, scaled, as the reference backend computes it.
double column_norm_hand(const double* a, std::size_t m, std::size_t cols, std::size_t first, std::size_t column)
{
    double scale = 0;
    for (std::size_t i = first; i < m; ++i)
        if (std::fabs(a[i * cols + column]) > scale)
            scale = std::fabs(a[i * cols + column]);
    if (scale == 0)
        return 0;
    double sum = 0;
    for (std::size_t i = first; i < m; ++i) {
        const double v = a[i * cols + column] / scale;
        sum += v * v;
    }
    return scale * std::sqrt(sum);
}

double hypot_hand(double a, double b)
{
    const double x = std::fabs(a), y = std::fabs(b);
    const double big = x > y ? x : y, small = x > y ? y : x;
    if (big == 0)
        return 0;
    const double r = small / big;
    return big * std::sqrt(1.0 + r * r);
}

bool qr_hand()
{
    restore_qr();
    double* a = qr.data();
    double* b = qr_obs.data();
    double* tau = qr_tau.data();
    const std::size_t m = qm, cols = qn;
    for (std::size_t i = 0; i < m * cols; ++i)
        if (a[i] != a[i] || std::fabs(a[i]) == std::numeric_limits<double>::infinity())
            return false;
    for (std::size_t k = 0; k < cols; ++k) {
        const double x_norm = k + 1 < m ? column_norm_hand(a, m, cols, k + 1, k) : 0.0;
        const double alpha = a[k * cols + k];
        if (x_norm == 0) {
            tau[k] = 0;
            continue;
        }
        const double norm = hypot_hand(alpha, x_norm);
        const double beta = alpha >= 0 ? -norm : norm;
        tau[k] = 1.0 - alpha / beta;
        const double denominator = alpha / norm - beta / norm;
        for (std::size_t i = k + 1; i < m; ++i)
            a[i * cols + k] = a[i * cols + k] / norm / denominator;
        a[k * cols + k] = beta;
        for (std::size_t j = k + 1; j < cols; ++j) {
            double w = a[k * cols + j];
            for (std::size_t i = k + 1; i < m; ++i)
                w += a[i * cols + k] * a[i * cols + j];
            a[k * cols + j] = (a[k * cols + j] - w) + (1.0 - tau[k]) * w;
            for (std::size_t i = k + 1; i < m; ++i)
                a[i * cols + j] -= (tau[k] * a[i * cols + k]) * w;
        }
    }
    double largest = 0;
    for (std::size_t k = 0; k < cols; ++k)
        if (std::fabs(a[k * cols + k]) > largest || a[k * cols + k] != a[k * cols + k])
            largest = std::fabs(a[k * cols + k]);
    const double tolerance = std::numeric_limits<double>::epsilon() * static_cast<double>(m > cols ? m : cols) * largest * 1.0;
    for (std::size_t k = 0; k < cols; ++k)
        if (!(std::fabs(a[k * cols + k]) > tolerance))
            return false;
    for (std::size_t i = 0; i < cols; ++i)
        for (std::size_t j = i + 1; j < cols; ++j)
            if (a[i * cols + j] != a[i * cols + j] || std::fabs(a[i * cols + j]) == std::numeric_limits<double>::infinity())
                return false;
    for (std::size_t k = 0; k < cols; ++k) {
        double w = b[k];
        for (std::size_t i = k + 1; i < m; ++i)
            w += a[i * cols + k] * b[i];
        b[k] = (b[k] - w) + (1.0 - tau[k]) * w;
        for (std::size_t i = k + 1; i < m; ++i)
            b[i] -= (tau[k] * a[i * cols + k]) * w;
    }
    for (std::size_t i = cols; i-- > 0;) {
        double s = b[i];
        for (std::size_t j = i + 1; j < cols; ++j)
            s -= a[i * cols + j] * b[j];
        b[i] = s / a[i * cols + i];
    }
    return true;
}

// ---- Jacobi: symmetric eigen (16) and SVD (24 x 16) ---------------------------------------------------------

std::size_t en = 16, sm = 24, sn = 16;
std::vector<double> sym0 = [] {
    std::vector<double> s(design0.begin(), design0.begin() + 16 * 16);
    for (std::size_t i = 0; i < 16; ++i)
        for (std::size_t j = 0; j < i; ++j)
            s[j * 16 + i] = s[i * 16 + j];
    return s;
}();
std::vector<double> sym(16 * 16), eig_v(16 * 16), eig_w(16), svd0 = design0, svd_u(24 * 16), svd_v(16 * 16), svd_s(16);

void rotation_hand(double theta, double& c, double& s)
{
    const double root = std::fabs(theta) > 1.0 ? std::fabs(theta) * std::sqrt(1.0 + 1.0 / (theta * theta)) : std::sqrt(1.0 + theta * theta);
    const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + root);
    c = 1.0 / std::sqrt(1.0 + t * t);
    s = t * c;
}

void swap_columns_hand(double* m, std::size_t rows, std::size_t cols, std::size_t i, std::size_t j)
{
    for (std::size_t r = 0; r < rows; ++r) {
        const double t = m[r * cols + i];
        m[r * cols + i] = m[r * cols + j];
        m[r * cols + j] = t;
    }
}

bool eigen_core()
{
    std::memcpy(sym.data(), sym0.data(), sym.size() * sizeof(double));
    return static_cast<bool>(la::symmetric_eigen(la::reference, la::matrix_view<double, la::layout_right>(sym.data(), {en, en}), std::span<double>(eig_w),
                                                 la::matrix_view<double, la::layout_right>(eig_v.data(), {en, en})));
}

bool eigen_hand()
{
    std::memcpy(sym.data(), sym0.data(), sym.size() * sizeof(double));
    double* a = sym.data();
    double* v = eig_v.data();
    double* w = eig_w.data();
    const std::size_t m = en;
    double scale = 0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            const double x = std::fabs(a[i * m + j]);
            if (x != x || x == std::numeric_limits<double>::infinity())
                return false;
            if (x > scale)
                scale = x;
        }
    double norm2 = 0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            if (scale > 0)
                a[i * m + j] = j > i ? a[j * m + i] / scale : a[i * m + j] / scale;
            else if (j > i)
                a[i * m + j] = a[j * m + i];
            norm2 += a[i * m + j] * a[i * m + j];
            v[i * m + j] = i == j ? 1.0 : 0.0;
        }
    const double eps = std::numeric_limits<double>::epsilon();
    bool converged = false;
    for (int sweep = 0; sweep <= la::reference_backend::max_sweeps && !converged; ++sweep) {
        double off = 0;
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = i + 1; j < m; ++j)
                off += a[i * m + j] * a[i * m + j];
        if (!(off > eps * eps * norm2)) {
            converged = off == off;
            break;
        }
        if (sweep == la::reference_backend::max_sweeps)
            break;
        for (std::size_t p = 0; p < m; ++p)
            for (std::size_t q = p + 1; q < m; ++q) {
                if (a[p * m + q] == 0)
                    continue;
                double c = 1, s = 0;
                rotation_hand((a[q * m + q] - a[p * m + p]) / (2.0 * a[p * m + q]), c, s);
                for (std::size_t k = 0; k < m; ++k) {
                    const double akp = a[k * m + p], akq = a[k * m + q];
                    a[k * m + p] = c * akp - s * akq;
                    a[k * m + q] = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < m; ++k) {
                    const double apk = a[p * m + k], aqk = a[q * m + k];
                    a[p * m + k] = c * apk - s * aqk;
                    a[q * m + k] = s * apk + c * aqk;
                }
                for (std::size_t k = 0; k < m; ++k) {
                    const double vkp = v[k * m + p], vkq = v[k * m + q];
                    v[k * m + p] = c * vkp - s * vkq;
                    v[k * m + q] = s * vkp + c * vkq;
                }
            }
    }
    for (std::size_t i = 0; i < m; ++i)
        w[i] = scale > 0 ? a[i * m + i] * scale : a[i * m + i];
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t smallest = i;
        for (std::size_t j = i + 1; j < m; ++j)
            if (w[j] < w[smallest])
                smallest = j;
        if (smallest != i) {
            const double t = w[i];
            w[i] = w[smallest];
            w[smallest] = t;
            swap_columns_hand(v, m, m, i, smallest);
        }
    }
    return converged;
}

bool svd_core()
{
    std::memcpy(svd_u.data(), svd0.data(), svd_u.size() * sizeof(double));
    return static_cast<bool>(la::svd(la::reference, la::matrix_view<double, la::layout_right>(svd_u.data(), {sm, sn}), std::span<double>(svd_s),
                                     la::matrix_view<double, la::layout_right>(svd_v.data(), {sn, sn})));
}

bool svd_hand()
{
    std::memcpy(svd_u.data(), svd0.data(), svd_u.size() * sizeof(double));
    double* a = svd_u.data();
    double* v = svd_v.data();
    double* s = svd_s.data();
    const std::size_t m = sm, cols = sn;
    double scale = 0;
    for (std::size_t i = 0; i < m * cols; ++i) {
        const double x = std::fabs(a[i]);
        if (x != x || x == std::numeric_limits<double>::infinity())
            return false;
        if (x > scale)
            scale = x;
    }
    if (scale > 0)
        for (std::size_t i = 0; i < m * cols; ++i)
            a[i] = a[i] / scale;
    for (std::size_t i = 0; i < cols; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            v[i * cols + j] = i == j ? 1.0 : 0.0;
    const double eps = std::numeric_limits<double>::epsilon();
    bool converged = false;
    for (int sweep = 0; sweep < la::reference_backend::max_sweeps && !converged; ++sweep) {
        bool rotated = false;
        for (std::size_t p = 0; p < cols; ++p)
            for (std::size_t q = p + 1; q < cols; ++q) {
                double alpha = 0, beta = 0, gamma = 0;
                for (std::size_t i = 0; i < m; ++i) {
                    alpha += a[i * cols + p] * a[i * cols + p];
                    beta += a[i * cols + q] * a[i * cols + q];
                    gamma += a[i * cols + p] * a[i * cols + q];
                }
                if (gamma != gamma || alpha != alpha || beta != beta)
                    return false;
                if (!(std::fabs(gamma) > eps * std::sqrt(alpha) * std::sqrt(beta)))
                    continue;
                rotated = true;
                double c = 1, sn_ = 0;
                rotation_hand((beta - alpha) / (2.0 * gamma), c, sn_);
                for (std::size_t i = 0; i < m; ++i) {
                    const double ap = a[i * cols + p], aq = a[i * cols + q];
                    a[i * cols + p] = c * ap - sn_ * aq;
                    a[i * cols + q] = sn_ * ap + c * aq;
                }
                for (std::size_t i = 0; i < cols; ++i) {
                    const double vp = v[i * cols + p], vq = v[i * cols + q];
                    v[i * cols + p] = c * vp - sn_ * vq;
                    v[i * cols + q] = sn_ * vp + c * vq;
                }
            }
        converged = !rotated;
    }
    for (std::size_t j = 0; j < cols; ++j) {
        const double norm = column_norm_hand(a, m, cols, 0, j);
        for (std::size_t i = 0; i < m; ++i)
            a[i * cols + j] = norm > 0 ? a[i * cols + j] / norm : 0.0;
        s[j] = norm * scale;
    }
    for (std::size_t i = 0; i < cols; ++i) {
        std::size_t largest = i;
        for (std::size_t j = i + 1; j < cols; ++j)
            if (s[j] > s[largest])
                largest = j;
        if (largest != i) {
            const double t = s[i];
            s[i] = s[largest];
            s[largest] = t;
            swap_columns_hand(a, m, cols, i, largest);
            swap_columns_hand(v, cols, cols, i, largest);
        }
    }
    return converged;
}

// ---- axpy, gemv, syrk, triangular solve, LDLT, qr_q ---------------------------------------------------------

std::vector<double> ax_y(vector_length), gv_y(64), sy_c(64 * 64), tr_b(64), ldlt_m(64 * 64), ldlt_b(64), q_out(96 * 32);
const std::vector<double> lower0 = [] {   // a well-conditioned lower triangle
    auto l = filled(64 * 64, 1.7);
    for (std::size_t i = 0; i < 64; ++i) {
        for (std::size_t j = i + 1; j < 64; ++j)
            l[i * 64 + j] = 0;
        l[i * 64 + i] += 8.0;
    }
    return l;
}();
std::vector<double> lower = lower0;

void axpy_core()
{
    std::memcpy(ax_y.data(), y.data(), ax_y.size() * sizeof(double));
    static_cast<void>(la::axpy(la::reference, 0.75, la::vector_view<const double, la::layout_right>(x.data(), {vector_length}),
                               la::vector_view<double, la::layout_right>(ax_y.data(), {vector_length})));
}
void axpy_hand()
{
    std::memcpy(ax_y.data(), y.data(), ax_y.size() * sizeof(double));
    for (std::size_t i = 0; i < vector_length; ++i)
        ax_y[i] += 0.75 * x[i];
}

void gemv_core()
{
    std::memcpy(gv_y.data(), rhs0.data(), gv_y.size() * sizeof(double));
    static_cast<void>(la::gemv(la::reference, 1.5, la::matrix_view<const double, la::layout_right>(a.data(), {n, n}),
                               la::vector_view<const double, la::layout_right>(rhs0.data(), {n}), 0.5,
                               la::vector_view<double, la::layout_right>(gv_y.data(), {n})));
}
void gemv_hand()
{
    std::memcpy(gv_y.data(), rhs0.data(), gv_y.size() * sizeof(double));
    for (std::size_t i = 0; i < n; ++i) {
        double s = 0;
        for (std::size_t j = 0; j < n; ++j)
            s += a[i * n + j] * rhs0[j];
        gv_y[i] = 1.5 * s + 0.5 * gv_y[i];
    }
}

void syrk_core()
{
    static_cast<void>(la::syrk(la::reference, 1.0, la::matrix_view<const double, la::layout_right>(a.data(), {n, n}), 0.0,
                               la::matrix_view<double, la::layout_right>(sy_c.data(), {n, n}), la::triangle::lower));
}
void syrk_hand()
{
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double s = 0;
            for (std::size_t p = 0; p < n; ++p)
                s += a[i * n + p] * a[j * n + p];
            sy_c[i * n + j] = 1.0 * s;
        }
}

bool triangular_core()
{
    std::memcpy(tr_b.data(), rhs0.data(), tr_b.size() * sizeof(double));
    return static_cast<bool>(la::triangular_solve(la::reference, la::matrix_view<const double, la::layout_right>(lower.data(), {n, n}),
                                                  la::vector_view<double, la::layout_right>(tr_b.data(), {n}), la::triangle::lower));
}
bool triangular_hand()
{
    std::memcpy(tr_b.data(), rhs0.data(), tr_b.size() * sizeof(double));
    for (std::size_t i = 0; i < n; ++i)
        if (lower[i * n + i] == 0)
            return false;
    for (std::size_t i = 0; i < n; ++i) {
        double s = tr_b[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= lower[i * n + j] * tr_b[j];
        tr_b[i] = s / lower[i * n + i];
    }
    return true;
}

void restore_ldlt()
{
    std::memcpy(ldlt_m.data(), spd0.data(), ldlt_m.size() * sizeof(double));
    std::memcpy(ldlt_b.data(), rhs0.data(), ldlt_b.size() * sizeof(double));
}
bool ldlt_core()
{
    restore_ldlt();
    const la::matrix_view<double, la::layout_right> m(ldlt_m.data(), {n, n});
    return la::ldlt_factor(la::reference, m) && la::ldlt_solve(la::reference, m, la::vector_view<double, la::layout_right>(ldlt_b.data(), {n}));
}
bool ldlt_hand()
{
    restore_ldlt();
    double* m = ldlt_m.data();
    double* b2 = ldlt_b.data();
    double scale = 0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            const double v = std::fabs(m[i * n + j]);
            if (v > scale || v != v)
                scale = v;
        }
    const double tolerance = std::numeric_limits<double>::epsilon() * static_cast<double>(n) * scale * 1.0;
    for (std::size_t j = 0; j < n; ++j) {
        double d = m[j * n + j];
        for (std::size_t k = 0; k < j; ++k)
            d -= m[j * n + k] * (m[j * n + k] * m[k * n + k]);
        if (!(std::fabs(d) > tolerance))
            return false;
        m[j * n + j] = d;
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = m[i * n + j];
            for (std::size_t k = 0; k < j; ++k)
                s -= m[i * n + k] * (m[j * n + k] * m[k * n + k]);
            m[i * n + j] = s / d;
        }
    }
    double growth = 0;
    for (std::size_t i = 0; i < n; ++i) {
        double g = std::fabs(m[i * n + i]);
        for (std::size_t k = 0; k < i; ++k)
            g += m[i * n + k] * m[i * n + k] * std::fabs(m[k * n + k]);
        if (g > growth || g != g)
            growth = g;
    }
    if (!(growth <= static_cast<double>(n) / std::sqrt(std::sqrt(std::numeric_limits<double>::epsilon())) * scale))
        return false;
    for (std::size_t i = 0; i < n; ++i) {
        double s = b2[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= m[i * n + j] * b2[j];
        b2[i] = s;
    }
    for (std::size_t i = 0; i < n; ++i)
        b2[i] = b2[i] / m[i * n + i];
    for (std::size_t i = n; i-- > 0;) {
        double s = b2[i];
        for (std::size_t j = i + 1; j < n; ++j)
            s -= m[j * n + i] * b2[j];
        b2[i] = s;
    }
    return true;
}

// qr_q of the least-squares design's factors (factored once by qr_core in the fairness check and at start-up).
void qr_q_core()
{
    static_cast<void>(la::qr_q(la::reference, la::matrix_view<const double, la::layout_right>(qr.data(), {qm, qn}), std::span<const double>(qr_tau),
                               la::matrix_view<double, la::layout_right>(q_out.data(), {qm, qn})));
}
void qr_q_hand()
{
    const double* h = qr.data();
    double* q = q_out.data();
    const std::size_t m = qm, cols = qn;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            q[i * cols + j] = i == j ? 1.0 : 0.0;
    for (std::size_t k = cols; k-- > 0;)
        for (std::size_t j = 0; j < cols; ++j) {
            double w = q[k * cols + j];
            for (std::size_t i = k + 1; i < m; ++i)
                w += h[i * cols + k] * q[i * cols + j];
            q[k * cols + j] = (q[k * cols + j] - w) + (1.0 - qr_tau[k]) * w;
            for (std::size_t i = k + 1; i < m; ++i)
                q[i * cols + j] -= (qr_tau[k] * h[i * cols + k]) * w;
        }
}

// ---- The benchmarks -----------------------------------------------------------------------------------------

void BM_LinalgDot_Core(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(dot_core());
    }
    s.SetItemsProcessed(static_cast<std::int64_t>(s.iterations() * vector_length));
}
void BM_LinalgDot_ByHand(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(x.data());
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(dot_hand());
    }
    s.SetItemsProcessed(static_cast<std::int64_t>(s.iterations() * vector_length));
}
template <void (*F)()>
void run_gemm(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(a.data());
        F();
        benchmark::ClobberMemory();
    }
    s.SetItemsProcessed(static_cast<std::int64_t>(s.iterations() * n * n * n));
}
void BM_LinalgGemm_Core(benchmark::State& s) { run_gemm<gemm_core>(s); }
void BM_LinalgGemm_ByHand(benchmark::State& s) { run_gemm<gemm_hand>(s); }
template <bool (*F)()>
void run_lu(benchmark::State& s)
{
    for (auto _ : s) {
        benchmark::DoNotOptimize(F());
        benchmark::ClobberMemory();
    }
    s.SetItemsProcessed(static_cast<std::int64_t>(s.iterations()));
}
void BM_LinalgLuSolve_Core(benchmark::State& s) { run_lu<lu_core>(s); }
void BM_LinalgLuSolve_ByHand(benchmark::State& s) { run_lu<lu_hand>(s); }
template <void (*F)()>
void run_void(benchmark::State& s)
{
    for (auto _ : s) {
        F();
        benchmark::ClobberMemory();
    }
    s.SetItemsProcessed(static_cast<std::int64_t>(s.iterations()));
}
void BM_LinalgAxpy_Core(benchmark::State& s) { run_void<axpy_core>(s); }
void BM_LinalgAxpy_ByHand(benchmark::State& s) { run_void<axpy_hand>(s); }
void BM_LinalgGemv_Core(benchmark::State& s) { run_void<gemv_core>(s); }
void BM_LinalgGemv_ByHand(benchmark::State& s) { run_void<gemv_hand>(s); }
void BM_LinalgSyrk_Core(benchmark::State& s) { run_void<syrk_core>(s); }
void BM_LinalgSyrk_ByHand(benchmark::State& s) { run_void<syrk_hand>(s); }
void BM_LinalgTriangularSolve_Core(benchmark::State& s) { run_lu<triangular_core>(s); }
void BM_LinalgTriangularSolve_ByHand(benchmark::State& s) { run_lu<triangular_hand>(s); }
void BM_LinalgLdltSolve_Core(benchmark::State& s) { run_lu<ldlt_core>(s); }
void BM_LinalgLdltSolve_ByHand(benchmark::State& s) { run_lu<ldlt_hand>(s); }
void BM_LinalgQrQ_Core(benchmark::State& s) { run_void<qr_q_core>(s); }
void BM_LinalgQrQ_ByHand(benchmark::State& s) { run_void<qr_q_hand>(s); }
void BM_LinalgCholeskySolve_Core(benchmark::State& s) { run_lu<cholesky_core>(s); }
void BM_LinalgCholeskySolve_ByHand(benchmark::State& s) { run_lu<cholesky_hand>(s); }
void BM_LinalgLeastSquares_Core(benchmark::State& s) { run_lu<qr_core>(s); }
void BM_LinalgLeastSquares_ByHand(benchmark::State& s) { run_lu<qr_hand>(s); }
void BM_LinalgSymmetricEigen_Core(benchmark::State& s) { run_lu<eigen_core>(s); }
void BM_LinalgSymmetricEigen_ByHand(benchmark::State& s) { run_lu<eigen_hand>(s); }
void BM_LinalgSvd_Core(benchmark::State& s) { run_lu<svd_core>(s); }
void BM_LinalgSvd_ByHand(benchmark::State& s) { run_lu<svd_hand>(s); }

BENCHMARK(BM_LinalgDot_Core);
BENCHMARK(BM_LinalgDot_ByHand);
BENCHMARK(BM_LinalgGemm_Core);
BENCHMARK(BM_LinalgGemm_ByHand);
BENCHMARK(BM_LinalgLuSolve_Core);
BENCHMARK(BM_LinalgLuSolve_ByHand);
BENCHMARK(BM_LinalgAxpy_Core);
BENCHMARK(BM_LinalgAxpy_ByHand);
BENCHMARK(BM_LinalgGemv_Core);
BENCHMARK(BM_LinalgGemv_ByHand);
BENCHMARK(BM_LinalgSyrk_Core);
BENCHMARK(BM_LinalgSyrk_ByHand);
BENCHMARK(BM_LinalgTriangularSolve_Core);
BENCHMARK(BM_LinalgTriangularSolve_ByHand);
BENCHMARK(BM_LinalgLdltSolve_Core);
BENCHMARK(BM_LinalgLdltSolve_ByHand);
BENCHMARK(BM_LinalgQrQ_Core);
BENCHMARK(BM_LinalgQrQ_ByHand);
BENCHMARK(BM_LinalgCholeskySolve_Core);
BENCHMARK(BM_LinalgCholeskySolve_ByHand);
BENCHMARK(BM_LinalgLeastSquares_Core);
BENCHMARK(BM_LinalgLeastSquares_ByHand);
BENCHMARK(BM_LinalgSymmetricEigen_Core);
BENCHMARK(BM_LinalgSymmetricEigen_ByHand);
BENCHMARK(BM_LinalgSvd_Core);
BENCHMARK(BM_LinalgSvd_ByHand);

bool differs(const char* what)
{
    std::fprintf(stderr, "linalg reference benchmark: %s differs from the hand-written version\n", what);
    return false;
}

bool fair()
{
    const auto same = [](const std::vector<double>& x, const std::vector<double>& y) {
        return std::memcmp(x.data(), y.data(), x.size() * sizeof(double)) == 0;
    };
    if (dot_core() != dot_hand())
        return differs("dot");
    gemm_core();
    const auto c1 = c;
    gemm_hand();
    if (!same(c1, c))
        return differs("gemm");
    if (!lu_core())
        return differs("lu (core failed)");
    const auto s1 = sol;
    const auto p1 = piv;
    if (!lu_hand() || !same(s1, sol) || p1 != piv)
        return differs("lu");
    if (!cholesky_core())
        return differs("cholesky (core failed)");
    const auto c2 = chol_sol;
    if (!cholesky_hand() || !same(c2, chol_sol))
        return differs("cholesky");
    if (!qr_core())
        return differs("least squares (core failed)");
    const auto q1 = qr_obs, q2 = qr;
    if (!qr_hand() || !same(q1, qr_obs) || !same(q2, qr))
        return differs("least squares");
    axpy_core();
    const auto ax1 = ax_y;
    axpy_hand();
    if (!same(ax1, ax_y))
        return differs("axpy");
    gemv_core();
    const auto gv1 = gv_y;
    gemv_hand();
    if (!same(gv1, gv_y))
        return differs("gemv");
    syrk_core();
    const auto sy1 = sy_c;
    std::fill(sy_c.begin(), sy_c.end(), 0.0);
    syrk_hand();
    if (!same(sy1, sy_c))
        return differs("syrk");
    if (!triangular_core())
        return differs("triangular solve (core failed)");
    const auto tr1 = tr_b;
    if (!triangular_hand() || !same(tr1, tr_b))
        return differs("triangular solve");
    if (!ldlt_core())
        return differs("ldlt (core failed)");
    const auto ld1 = ldlt_b, ld2 = ldlt_m;
    if (!ldlt_hand() || !same(ld1, ldlt_b) || !same(ld2, ldlt_m))
        return differs("ldlt");
    if (!qr_core())   // the factors qr_q reads
        return differs("least squares (core failed)");
    qr_q_core();
    const auto q1o = q_out;
    qr_q_hand();
    if (!same(q1o, q_out))
        return differs("qr_q");
    if (!eigen_core())
        return differs("symmetric eigen (core failed)");
    const auto w1 = eig_w, v1 = eig_v;
    if (!eigen_hand() || !same(w1, eig_w) || !same(v1, eig_v))
        return differs("symmetric eigen");
    if (!svd_core())
        return differs("svd (core failed)");
    const auto sv1 = svd_s, u1 = svd_u, vv1 = svd_v;
    if (!svd_hand() || !same(sv1, svd_s) || !same(u1, svd_u) || !same(vv1, svd_v))
        return differs("svd");
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (!fair()) {
        std::fprintf(stderr, "linalg reference benchmark: fairness check FAILED\n");
        return 1;
    }
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
