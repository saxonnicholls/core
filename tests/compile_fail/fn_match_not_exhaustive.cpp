//
//  fn_match_not_exhaustive.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: snicholls::fn::match over a variant whose double alternative has only a guarded case. Built
//  with SNICHOLLS_COMPILE_FAIL it must fail with the message below; without it the unguarded case is there, and it
//  is the positive control and must compile.
//

#include <snicholls/core/functional/match.hpp>

#include <concepts>
#include <variant>

int main()
{
    const std::variant<int, double> v = 2.0;
    const int r = snicholls::fn::match(v, [](std::same_as<int> auto i) { return i; },
                                       snicholls::fn::guard([](double d) { return d > 1; }, [](double) { return 1; })
#ifndef SNICHOLLS_COMPILE_FAIL
                                           ,
                                       [](double) { return 0; }
#endif
    );
    return r == 1 ? 0 : 1;
}
