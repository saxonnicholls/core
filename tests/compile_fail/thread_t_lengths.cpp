//
//  thread_t_lengths.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: threading over lists of different lengths. Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed
//  in tests/CMakeLists.txt; without it, the positive control must compile.
//

#include <snicholls/core/meta/type_list.hpp>

using namespace snicholls::meta;
template <typename...> struct f;

#ifdef SNICHOLLS_COMPILE_FAIL
using threaded = thread_t<f<type_list<int, char>, type_list<bool>>>;
#else
using threaded = thread_t<f<type_list<int, char>, type_list<bool, long>>>;
#endif

int main() { return threaded::size == 2 ? 0 : 1; }
