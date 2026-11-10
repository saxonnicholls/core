//
//  string_access_unterminated_array.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: a character array is a string literal, so its last
//  element must be NUL; in constant evaluation an unterminated one is a compile
//  error naming snicholls_character_array_is_not_nul_terminated. Built with
//  SNICHOLLS_COMPILE_FAIL it must fail to compile; without it (a terminated
//  array), it is the positive control and must compile.
//

#include <snicholls/core/reflect/string_access.hpp>

#ifdef SNICHOLLS_COMPILE_FAIL
constexpr char letters[3] = {'a', 'b', 'c'};
#else
constexpr char letters[3] = {'a', 'b', '\0'};
#endif

constexpr auto length = snicholls::length_of(letters);

int main()
{
    return static_cast<int>(length) - 2;
}
