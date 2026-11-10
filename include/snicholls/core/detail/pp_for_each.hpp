//
//  pp_for_each.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

// SNICHOLLS_PP_FOR_EACH(macro, data, a, b, c) -> macro(data, a) macro(data, b) macro(data, c)
//
// The C++20 __VA_OPT__ recursion (deferred re-invocation, forced by repeated rescans): no fixed-arity tables, no
// compiler extensions. SNICHOLLS_PP_EXPAND rescans 342 times, so up to 342 elements; more leaves an unexpanded
// SNICHOLLS_PP_FOR_EACH_HELPER in the output, a compile error that names it. Not re-entrant: `macro` must not itself
// use SNICHOLLS_PP_FOR_EACH.

#ifndef snicholls_core_detail_pp_for_each_hpp
#define snicholls_core_detail_pp_for_each_hpp

#define SNICHOLLS_PP_PARENS ()

#define SNICHOLLS_PP_EXPAND(...) SNICHOLLS_PP_EXPAND4(SNICHOLLS_PP_EXPAND4(SNICHOLLS_PP_EXPAND4(SNICHOLLS_PP_EXPAND4(__VA_ARGS__))))
#define SNICHOLLS_PP_EXPAND4(...) SNICHOLLS_PP_EXPAND3(SNICHOLLS_PP_EXPAND3(SNICHOLLS_PP_EXPAND3(SNICHOLLS_PP_EXPAND3(__VA_ARGS__))))
#define SNICHOLLS_PP_EXPAND3(...) SNICHOLLS_PP_EXPAND2(SNICHOLLS_PP_EXPAND2(SNICHOLLS_PP_EXPAND2(SNICHOLLS_PP_EXPAND2(__VA_ARGS__))))
#define SNICHOLLS_PP_EXPAND2(...) SNICHOLLS_PP_EXPAND1(SNICHOLLS_PP_EXPAND1(SNICHOLLS_PP_EXPAND1(SNICHOLLS_PP_EXPAND1(__VA_ARGS__))))
#define SNICHOLLS_PP_EXPAND1(...) __VA_ARGS__

#define SNICHOLLS_PP_FOR_EACH(macro, data, ...) __VA_OPT__(SNICHOLLS_PP_EXPAND(SNICHOLLS_PP_FOR_EACH_HELPER(macro, data, __VA_ARGS__)))
#define SNICHOLLS_PP_FOR_EACH_HELPER(macro, data, a1, ...)                                                                \
    macro(data, a1) __VA_OPT__(SNICHOLLS_PP_FOR_EACH_AGAIN SNICHOLLS_PP_PARENS(macro, data, __VA_ARGS__))
#define SNICHOLLS_PP_FOR_EACH_AGAIN() SNICHOLLS_PP_FOR_EACH_HELPER

#endif /* snicholls_core_detail_pp_for_each_hpp */
