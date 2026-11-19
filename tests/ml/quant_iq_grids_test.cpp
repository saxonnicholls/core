//
//  quant_iq_grids_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/ml/quant_iq_grids.hpp>: every table, built by the compiler from the compact index
//  lists, has the FNV-1a hash of ggml-common.h's table (static_assert, and again against the "table" lines of
//  tests/fixtures/ml/quant_iq_fixture.txt, which ggml's generator wrote); the levels of the grids; the sign table's
//  parity; iq_grid_find as ggml's kmap; iq_grid_coordinate as the quantiser's 2l + 1.
//

#include <snicholls/core/ml/quant_iq_grids.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace {

namespace ml = snicholls::ml;

// FNV-1a over the little-endian bytes of each entry: the bytes of ggml's tables in memory on x86 and ARM
template <typename T, std::size_t N>
constexpr std::uint64_t fnv(const std::array<T, N>& a)
{
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (const T v : a)
        for (std::size_t b = 0; b < sizeof(T); ++b) {
            h ^= static_cast<std::uint8_t>(static_cast<std::uint64_t>(v) >> (8 * b));
            h *= 0x100000001b3ULL;
        }
    return h;
}

// The hashes ggml's generator recorded (the "table" lines of the fixture)
static_assert(fnv(ml::iq2xxs_grid) == 0xbb4ee025b5ac6e8eULL);
static_assert(fnv(ml::iq2xs_grid) == 0xc9b1ee61e79909bdULL);
static_assert(fnv(ml::iq2s_grid) == 0x123edd38a3b62b90ULL);
static_assert(fnv(ml::iq3xxs_grid) == 0xdfa5dc83d6a155d5ULL);
static_assert(fnv(ml::iq3s_grid) == 0xfa37020c25b44829ULL);
static_assert(fnv(ml::iq1s_grid) == 0x6703ed863501ae2eULL);
static_assert(fnv(ml::iq2_signs) == 0xf19ba8f8c3292ba5ULL);
static_assert(fnv(ml::iq4nl_values) == 0x055d06d2d8fa0f75ULL);

// ggml's first entries, by hand
static_assert(ml::iq2xxs_grid[0] == 0x0808080808080808ULL && ml::iq2xxs_grid[1] == 0x080808080808082bULL);
static_assert(ml::iq3xxs_grid[0] == 0x04040404u && ml::iq3s_grid[0] == 0x01010101u);
static_assert(ml::iq1s_grid[0] == 0xffffffffffffffffULL && ml::iq1s_grid[1] == 0xffffffffffffff01ULL);
static_assert(ml::iq2_signs[1] == 129 && ml::iq2_signs[3] == 3 && ml::iq2_signs[127] == 255);

// kmap and the quantiser's coordinates
static_assert(ml::iq_grid_find(ml::iq2xxs_grid_index, 0) == 0 && ml::iq_grid_find(ml::iq2xxs_grid_index, 2) == 1
              && ml::iq_grid_find(ml::iq2xxs_grid_index, 1) == -1 && ml::iq_grid_find(ml::iq2xxs_grid_index, 0xffff) == -1);
static_assert(ml::iq_grid_coordinate(ml::iq2xxs_grid_index, 2, 1, 0) == 5 && ml::iq_grid_coordinate(ml::iq2xxs_grid_index, 2, 1, 1) == 1);
static_assert(noexcept(ml::iq_grid_find(ml::iq3s_grid_index, 7)) && noexcept(ml::iq_grid_coordinate(ml::iq3s_grid_index, 3, 0, 0)));

TEST(MlQuantIqGrids, EveryTableHasTheHashGgmlsGeneratorRecorded)
{
    std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/quant_iq_fixture.txt");
    ASSERT_TRUE(in.good()) << "missing tests/fixtures/ml/quant_iq_fixture.txt";
    std::map<std::string, std::uint64_t> recorded;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("table ", 0) != 0)
            continue;
        std::istringstream ls(line);
        std::string word, name, size, hash;
        ls >> word >> name >> size >> hash;
        recorded[name] = std::stoull(hash.substr(4), nullptr, 16);
    }
    ASSERT_EQ(recorded.size(), 9u);
    EXPECT_EQ(fnv(ml::iq2xxs_grid), recorded["iq2xxs_grid"]);
    EXPECT_EQ(fnv(ml::iq2xs_grid), recorded["iq2xs_grid"]);
    EXPECT_EQ(fnv(ml::iq2s_grid), recorded["iq2s_grid"]);
    EXPECT_EQ(fnv(ml::iq3xxs_grid), recorded["iq3xxs_grid"]);
    EXPECT_EQ(fnv(ml::iq3s_grid), recorded["iq3s_grid"]);
    EXPECT_EQ(fnv(ml::iq1s_grid), recorded["iq1s_grid"]);
    EXPECT_EQ(fnv(ml::iq2_signs), recorded["ksigns_iq2xs"]);
    EXPECT_EQ(fnv(ml::iq4nl_values), recorded["kvalues_iq4nl"]);
    constexpr std::array<std::uint8_t, 8> mask{1, 2, 4, 8, 16, 32, 64, 128};   // ggml's kmask_iq2xs: bit j
    EXPECT_EQ(fnv(mask), recorded["kmask_iq2xs"]);
}

TEST(MlQuantIqGrids, EveryCoordinateIsOneOfTheGridsLevels)
{
    const auto levels = [](const auto& grid, int bytes) {
        std::set<int> s;
        for (const auto p : grid)
            for (int j = 0; j < bytes; ++j)
                s.insert(int((std::uint64_t(p) >> (8 * j)) & 0xff));
        return s;
    };
    EXPECT_EQ(levels(ml::iq2xxs_grid, 8), (std::set<int>{8, 25, 43}));
    EXPECT_EQ(levels(ml::iq2s_grid, 8), (std::set<int>{8, 25, 43}));
    EXPECT_EQ(levels(ml::iq1s_grid, 8), (std::set<int>{0x00, 0x01, 0xff}));
    EXPECT_EQ(levels(ml::iq3xxs_grid, 4), (std::set<int>{4, 12, 20, 28, 36, 44, 52, 62}));
    EXPECT_EQ(levels(ml::iq3s_grid, 4), (std::set<int>{1, 3, 5, 7, 9, 11, 13, 15}));
}

TEST(MlQuantIqGrids, TheSignTableHasAnEvenNumberOfMinusSigns)
{
    for (std::size_t i = 0; i < 128; ++i) {
        EXPECT_EQ(std::popcount(unsigned(ml::iq2_signs[i])) % 2, 0) << i;
        EXPECT_EQ(ml::iq2_signs[i] & 127u, i);
    }
}

TEST(MlQuantIqGrids, FindIsTheInverseOfTheIndexList)
{
    for (std::size_t g = 0; g < ml::iq1s_grid_index.size(); ++g)
        EXPECT_EQ(ml::iq_grid_find(ml::iq1s_grid_index, ml::iq1s_grid_index[g]), int(g));
    for (std::size_t g = 0; g < ml::iq3s_grid_index.size(); ++g)
        EXPECT_EQ(ml::iq_grid_find(ml::iq3s_grid_index, ml::iq3s_grid_index[g]), int(g));
    int found = 0;
    for (std::uint32_t u = 0; u < 4096; ++u)
        found += ml::iq_grid_find(ml::iq3xxs_grid_index, static_cast<std::uint16_t>(u)) >= 0 ? 1 : 0;
    EXPECT_EQ(found, 256);   // every other packed point is off the grid
}

TEST(MlQuantIqGrids, CoordinatesAreTwiceTheLevelPlusOne)
{
    for (std::size_t g = 0; g < ml::iq2s_grid_index.size(); ++g)
        for (int k = 0; k < 8; ++k) {
            const int c = ml::iq_grid_coordinate(ml::iq2s_grid_index, 2, g, k);
            const int byte = int((ml::iq2s_grid[g] >> (8 * k)) & 0xff);
            EXPECT_EQ(byte, c == 1 ? 8 : c == 3 ? 25 : 43);
        }
}

} // namespace
