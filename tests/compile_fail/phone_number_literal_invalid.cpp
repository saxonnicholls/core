//
//  phone_number_literal_invalid.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: an _e164 literal that is not an E.164 number (no country code). Built with
//  SNICHOLLS_COMPILE_FAIL it must fail with the message listed in tests/CMakeLists.txt; without it, the positive
//  control must compile.
//

#include <snicholls/core/phone_number.hpp>

using namespace snicholls::literals;

#ifdef SNICHOLLS_COMPILE_FAIL
inline constexpr snicholls::phone_number number = "0412 345 678"_e164;
#else
inline constexpr snicholls::phone_number number = "+61 412 345 678"_e164;
#endif

int main() { return number.country_code() == 61 ? 0 : 1; }
