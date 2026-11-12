//
//  linalg_backend_missing_member.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: calling linalg::lu_factor with a backend that has the BLAS operations but no
//  lu_factor member. Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed in
//  tests/CMakeLists.txt, which names the missing member; without it, the positive control (dot, which
//  the backend has) must compile.
//

#include <snicholls/core/linalg/reference.hpp>

#include <array>
#include <span>

namespace {

namespace la = snicholls::linalg;

// The BLAS half of a backend only, forwarding to the reference backend.
struct blas_only {
    template <la::scalar T>
    T dot(la::vector_view<const T> x, la::vector_view<const T> y) const noexcept
    {
        return la::reference.dot(x, y);
    }
};

} // namespace

int main()
{
    std::array<double, 4> a{2, 0, 0, 2};
    const la::matrix_view<double, la::layout_right> m(a.data(), {2, 2});
#ifdef SNICHOLLS_COMPILE_FAIL
    std::array<la::pivot_index, 2> pivots{};
    return la::lu_factor(blas_only{}, m, std::span<la::pivot_index>(pivots)) ? 0 : 1;
#else
    const la::vector_view<const double> d(a.data(), {2}, {3});   // the diagonal
    return la::dot(blas_only{}, d, d).unwrap_or(0.0) == 8.0 ? 0 : 1;
#endif
}
