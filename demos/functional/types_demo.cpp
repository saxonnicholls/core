//
//  types_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/functional/types.hpp>: the same list operations on types, at compile time. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/functional/types.hpp>

#include <cstdio>
#include <type_traits>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    namespace ty = snicholls::fn::types;
    std::printf("snicholls-core functional/types demo\n");

    //! [types]
    using fields = ty::list<int, double, char, double>;
    using pointers = ty::map_t<fields, std::add_pointer_t>;
    using integral = ty::filter_t<fields, std::is_integral>;
    static_assert(std::is_same_v<integral, ty::list<int, char>>);
    static_assert(std::is_same_v<ty::nub_t<fields>, ty::list<int, double, char>>);
    static_assert(ty::length_v<pointers> == 4 && ty::elem_v<pointers, char*>);
    //! [types]
    check(ty::length_v<ty::reverse_t<fields>> == 4, "reversed, still four");

    std::printf("%s\n", failures == 0 ? "all ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
