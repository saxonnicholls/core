//
//  invoke_with_generic_lambda.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: invoke_any on a set holding a generic lambda, whose parameter
//  types cannot be read. Built with SNICHOLLS_COMPILE_FAIL it must fail to compile
//  with the message below; without it (the same set with a concrete lambda), it is
//  the positive control and must compile.
//

#include <snicholls/core/invoke_with.hpp>

#include <any>

int main()
{
#ifdef SNICHOLLS_COMPILE_FAIL
    const auto f = snicholls::overloaded{[](const auto& x) { return x; }};
#else
    const auto f = snicholls::overloaded{[](const int& x) { return x; }};
#endif
    return std::any_cast<int>(snicholls::invoke_any(f, {0}).unwrap());
}
