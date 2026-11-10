//
//  constexpr_math_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/math/constexpr_math.hpp>: a discount curve and a table of normal
//  quantiles built at compile time, in double, in float for a GPU table, and in long double. Exits
//  non-zero if a value is not the one documented.
//

#include <snicholls/core/math/constexpr_math.hpp>

#include <array>
#include <cstdio>
#include <limits>

int main()
{
    std::printf("snicholls-core constexpr_math demo\n");
    namespace math = snicholls::math;
    bool ok = true;

    //! [math]
    // Discount factors for 1 to 5 years at 4% continuously compounded, computed by the compiler.
    constexpr auto discount = [] {
        std::array<double, 5> out{};
        for (int y = 1; y <= 5; ++y)
            out[static_cast<std::size_t>(y - 1)] = math::exp(-0.04 * y);
        return out;
    }();

    // The quantiles a VaR report needs, as constants.
    constexpr double z95 = math::normal_quantile(0.95);
    constexpr double z99 = math::normal_quantile(0.99);
    static_assert(math::near(z99, 2.3263478740408408, 1e-14));
    //! [math]

    //! [math_types]
    // The same functions in the argument's type: float for a table a GPU kernel reads, long double where
    // its extra digits matter. Each type's limits are a math_limits specialisation, as numeric_limits are.
    constexpr std::array<float, 3> z_gpu{math::normal_quantile(0.90f), math::normal_quantile(0.95f), math::normal_quantile(0.99f)};
    constexpr long double e_wide = math::exp(1.0L);
    static_assert(z_gpu[2] == static_cast<float>(math::normal_quantile(static_cast<double>(0.99f))));   // in double, rounded once
    static_assert(math::near<long double>(e_wide, 2.718281828459045235360287471352662498L, 4 * std::numeric_limits<long double>::epsilon()));
    //! [math_types]

    std::printf("  5-year discount %.6f, z95 %.6f, z99 %.6f\n", discount[4], z95, z99);
    ok &= math::near(discount[4], 0.8187307530779818, 1e-14) && math::near(z95, 1.6448536269514722, 1e-14);
    std::printf("  float z90 %.6f, long double e %.18Lf (%d bits)\n", static_cast<double>(z_gpu[0]), e_wide, std::numeric_limits<long double>::digits);
    ok &= math::near<float>(z_gpu[0], 1.2815516f, 1e-6f);

    std::printf(ok ? "constexpr_math demo: ok\n" : "constexpr_math demo: FAILED\n");
    return ok ? 0 : 1;
}
