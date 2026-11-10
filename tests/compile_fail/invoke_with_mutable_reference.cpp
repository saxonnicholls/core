//
//  invoke_with_mutable_reference.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: invoke_any on a callable that takes a non-const lvalue
//  reference, which cannot bind the object inside a const std::any. Built with
//  SNICHOLLS_COMPILE_FAIL it must fail to compile with the message below; without it
//  (the same callable taking a const reference), it is the positive control and must
//  compile.
//

#include <snicholls/core/invoke_with.hpp>

#include <any>

int main()
{
#ifdef SNICHOLLS_COMPILE_FAIL
    const auto f = [](int& x) { return x; };
#else
    const auto f = [](const int& x) { return x; };
#endif
    return std::any_cast<int>(snicholls::invoke_any(f, {0}).unwrap());
}
