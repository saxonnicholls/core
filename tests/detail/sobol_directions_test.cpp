//
//  sobol_directions_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for the generated <snicholls/core/detail/sobol_directions.hpp>: rows checked against
//  Joe and Kuo's published new-joe-kuo-6.21201, and the table's invariants for every dimension.
//

#include <snicholls/core/detail/sobol_directions.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

namespace d = snicholls::detail;

static_assert(d::sobol_max_dimensions == 1024);
static_assert(sizeof(d::sobol_degree) == 1023 && sizeof(d::sobol_coefficients) / sizeof(d::sobol_coefficients[0]) == 1023);

std::vector<std::uint16_t> row(std::size_t dimension)
{
    const std::size_t i = dimension - 2;
    return {d::sobol_m + d::sobol_m_offset[i], d::sobol_m + d::sobol_m_offset[i] + d::sobol_degree[i]};
}

TEST(SobolDirections, RowsMatchJoeAndKuosPublishedTable)
{
    EXPECT_EQ(d::sobol_degree[0], 1);   // d = 2: s 1, a 0, m 1
    EXPECT_EQ(d::sobol_coefficients[0], 0u);
    EXPECT_EQ(row(2), (std::vector<std::uint16_t>{1}));
    EXPECT_EQ(d::sobol_degree[7 - 2], 4);   // d = 7: s 4, a 4, m 1 3 5 13
    EXPECT_EQ(d::sobol_coefficients[7 - 2], 4u);
    EXPECT_EQ(row(7), (std::vector<std::uint16_t>{1, 3, 5, 13}));
    EXPECT_EQ(row(38), (std::vector<std::uint16_t>{1, 3, 1, 15, 31, 13, 49, 245}));   // d = 38: s 8, a 14
    EXPECT_EQ(d::sobol_coefficients[38 - 2], 14u);
    EXPECT_EQ(row(65), (std::vector<std::uint16_t>{1, 1, 1, 11, 21, 53, 125, 249, 293}));   // d = 65: s 9, a 67
    EXPECT_EQ(d::sobol_coefficients[65 - 2], 67u);
}

TEST(SobolDirections, EveryDirectionNumberIsOddAndBelowTwoToItsIndex)
{
    std::size_t previous_degree = 1;
    for (std::size_t dim = 2; dim <= d::sobol_max_dimensions; ++dim) {
        const std::size_t s = d::sobol_degree[dim - 2];
        EXPECT_GE(s, previous_degree) << "dimension " << dim;   // polynomials come in order of degree
        EXPECT_LT(d::sobol_coefficients[dim - 2], 1u << (s - 1)) << "dimension " << dim;
        previous_degree = s;
        const auto m = row(dim);
        for (std::size_t k = 1; k <= m.size(); ++k) {
            EXPECT_EQ(m[k - 1] % 2, 1) << "dimension " << dim << ", m_" << k;
            EXPECT_LT(static_cast<unsigned>(m[k - 1]), 1u << k) << "dimension " << dim << ", m_" << k;
        }
    }
}

} // namespace
