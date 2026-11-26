//
//  fn_curry_generic_lambda.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: snicholls::fn::curry(f) of a generic lambda, which has no arity of its own. Built with
//  SNICHOLLS_COMPILE_FAIL it must fail with the message below; without it the arity is given, curry<2>(f), and it
//  is the positive control and must compile.
//

#include <snicholls/core/functional/curry.hpp>

int main()
{
    constexpr auto add = [](auto a, auto b) { return a + b; };
#ifdef SNICHOLLS_COMPILE_FAIL
    const auto curried = snicholls::fn::curry(add);
#else
    const auto curried = snicholls::fn::curry<2>(add);
#endif
    return curried(1)(2) == 3 ? 0 : 1;
}
