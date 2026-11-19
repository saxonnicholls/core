//
//  hadamard_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/hadamard.hpp>: the fast transform against Sylvester's matrix, the orthonormal form
//  undoing itself, a block transform of a width that is not a power of two, and a randomised rotation folded into
//  weights before per-row int8 quantisation: the product is unchanged and the quantisation error falls. Exits non-zero if
//  a result is not what it documents.
//

#include <snicholls/core/ml/hadamard.hpp>
#include <snicholls/core/ml/quantize.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

int main()
{
    std::printf("snicholls-core hadamard demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    // The fast transform is Sylvester's matrix times the row, in n log n additions
    std::array<double, 8> x{1, 2, 3, 4, 5, 6, 7, 8};
    const auto x0 = x;
    ml::fwht(x);
    double row1 = 0;
    for (std::size_t j = 0; j < 8; ++j)
        row1 += ml::hadamard_entry(1, j) * x0[j];
    std::printf("  fwht(1..8) = %g %g %g ... ; row 1 of H times x = %g\n", x[0], x[1], x[2], row1);
    ok &= x[0] == 36 && x[1] == row1;

    // The orthonormal form is its own inverse; a 96-wide row takes blocks of 32
    std::array<float, 4> y{1.0f, -2.0f, 0.5f, 3.0f};
    ml::hadamard(y);
    ml::hadamard(y);
    std::array<float, 96> wide{};
    wide[0] = 1.0f;
    ml::block_hadamard(wide, 32);
    std::printf("  hadamard twice: %g %g %g %g; block transform of e0, 96 wide: %g ... %g, then %g\n", double(y[0]), double(y[1]),
                double(y[2]), double(y[3]), double(wide[0]), double(wide[31]), double(wide[32]));
    ok &= y[1] == -2.0f && y[3] == 3.0f && std::fabs(wide[31] - 1.0f / std::sqrt(32.0f)) < 1e-7f && wide[32] == 0.0f;

    //! [hadamard]
    // Weights with an outlier column, and an input: rotate both by H D (QuaRot's randomised Hadamard)
    constexpr std::size_t rows = 4, cols = 256;
    std::vector<float> w(rows * cols), a(cols);
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = 0.02f * std::sin(0.37f * static_cast<float>(i)) + (i % cols == 17 ? 1.5f : 0.0f);
    for (std::size_t j = 0; j < cols; ++j)
        a[j] = std::cos(0.11f * static_cast<float>(j));

    std::vector<std::int8_t> signs(cols);
    ml::hadamard_signs(signs, 2026);       // D: the same seed gives the same signs everywhere
    std::vector<float> wf = w, af = a;
    ml::rotate_rows(wf, cols, signs, cols);  // W_f = W D H, folded once, offline
    ml::random_hadamard(af, signs, cols);    // x' = H (D x), at inference

    // Quantise both to int8 with one scale per row (per-channel): the folded rows have no outlier left to stretch it
    const auto int8_error = [](const std::vector<float>& v, std::size_t width) {
        double e = 0;
        std::vector<std::int8_t> q(width);
        for (std::size_t r = 0; r < v.size(); r += width) {
            const std::span<const float> row(v.data() + r, width);
            const float scale = ml::quantize_int8(row, q);
            for (std::size_t j = 0; j < width; ++j)
                e += (double(q[j]) * double(scale) - double(row[j])) * (double(q[j]) * double(scale) - double(row[j]));
        }
        return std::sqrt(e);
    };
    const double plain_error = int8_error(w, cols), folded_error = int8_error(wf, cols);
    //! [hadamard]

    double plain = 0, folded = 0;
    for (std::size_t j = 0; j < cols; ++j) {
        plain += double(w[j]) * double(a[j]);
        folded += double(wf[j]) * double(af[j]);
    }
    std::printf("  row 0 of W x: %.6f, folded %.6f\n", plain, folded);
    std::printf("  per-row int8 error of W: %.5f, of W D H: %.5f\n", plain_error, folded_error);
    ok &= std::fabs(plain - folded) < 1e-4 && folded_error < 0.5 * plain_error;

    std::vector<float> back = af;
    ml::random_hadamard_inverse(back, signs, cols);
    float worst = 0.0f;
    for (std::size_t j = 0; j < cols; ++j)
        worst = std::fmax(worst, std::fabs(back[j] - a[j]));
    std::printf("  inverse rotation: largest difference %.2g\n", double(worst));
    ok &= worst < 1e-5f;

    std::vector<float> m(16);
    ml::hadamard_matrix<float>(4, m);
    std::printf("  the 4 x 4 orthonormal matrix, row 3: %g %g %g %g\n", double(m[12]), double(m[13]), double(m[14]), double(m[15]));
    ok &= m[12] == 0.5f && m[13] == -0.5f && m[15] == 0.5f && ml::is_hadamard_order(4) && !ml::is_hadamard_order(96);

    std::printf(ok ? "hadamard demo: ok\n" : "hadamard demo: FAILED\n");
    return ok ? 0 : 1;
}
