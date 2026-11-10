//
//  type_name_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/meta/type_name.hpp>: type names as constants, in a
//  message and in a static_assert. Exits non-zero if a portable name is not the
//  one documented.
//

#include <snicholls/core/meta/type_name.hpp>

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace trading {
struct order {};
enum class side { buy, sell };
} // namespace trading

int main()
{
    std::printf("snicholls-core type_name demo\n");
    bool ok = true;

    //! [names]
    // A type's name, as a constant: in a static_assert, or printed.
    static_assert(snicholls::type_name<trading::order>() == "trading::order");
    constexpr std::string_view side_name = snicholls::type_name<trading::side>();

    std::printf("  %s, %s, %s\n", snicholls::type_name<int>().data(), side_name.data(),
                std::string(snicholls::type_name<std::vector<trading::order>>()).c_str());
    //! [names]

    ok &= snicholls::type_name<int>() == "int";
    ok &= side_name == "trading::side";
    ok &= snicholls::type_name<std::vector<trading::order>>().starts_with("std::vector<trading::order");
    ok &= snicholls::type_name(3.5) == "double";

    std::printf(ok ? "type_name demo: ok\n" : "type_name demo: FAILED\n");
    return ok ? 0 : 1;
}
