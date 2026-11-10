//
//  thread_t_position.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Compile-fail case: a position specification beyond the arguments. Built with SNICHOLLS_COMPILE_FAIL it must fail with the message listed
//  in tests/CMakeLists.txt; without it, the positive control must compile.
//

#include <snicholls/core/meta/type_list.hpp>

using namespace snicholls::meta;
template <typename...> struct f;

#ifdef SNICHOLLS_COMPILE_FAIL
using threaded = thread_t<f<type_list<int, char>>, type_list, thread_at<2>>;
#else
using threaded = thread_t<f<type_list<int, char>>, type_list, thread_at<1>>;
#endif

int main() { return threaded::size == 2 ? 0 : 1; }
