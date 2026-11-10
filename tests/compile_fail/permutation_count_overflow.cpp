//
//  permutation_count_overflow.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: a permutation count too large for std::size_t. Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed
//  in tests/CMakeLists.txt; without it, the positive control must compile.
//

#include <snicholls/core/combinatorics/permutations.hpp>

#ifdef SNICHOLLS_COMPILE_FAIL
inline constexpr unsigned long long count = snicholls::permutation_count_v<sizeof(unsigned long long) * 3>;
#else
inline constexpr unsigned long long count = snicholls::permutation_count_v<5>;
#endif

int main() { return count == 0 ? 1 : 0; }
