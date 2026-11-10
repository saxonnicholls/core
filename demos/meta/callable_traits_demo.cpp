//
//  callable_traits_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/meta/callable_traits.hpp>: reading what a callable takes,
//  building something from its argument list, and refusing a callable with no
//  single signature at compile time. Exits non-zero if any reading is not the
//  one documented.
//

#include <snicholls/core/meta/callable_traits.hpp>

#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace {

//! [traits]
// A handler's parameter list, read from its type: a registry can record what each one takes.
template <snicholls::single_signature_callable F>
std::string describe(const F&)
{
    using traits = snicholls::callable_traits<F>;
    std::string out = std::to_string(traits::arity) + " argument(s)";
    snicholls::meta::for_each_type<typename traits::arguments>([&]<typename Argument>() {
        out += std::string{", "} + typeid(Argument).name();
    });
    out += traits::is_noexcept ? ", noexcept" : "";
    return out;
}

struct account {
    double balance(const std::string& currency) const noexcept;
};
//! [traits]

double account::balance(const std::string&) const noexcept { return 0; }

} // namespace

int main()
{
    std::printf("snicholls-core callable_traits demo\n");
    bool ok = true;

    const auto deposit = [](int account_id, double amount) { return account_id > 0 && amount > 0; };
    std::printf("  deposit lambda:    %s\n", describe(deposit).c_str());
    std::printf("  account::balance:  %s\n", describe(&account::balance).c_str());

    using deposit_traits = snicholls::callable_traits<decltype(deposit)>;
    static_assert(deposit_traits::arity == 2);
    static_assert(std::is_same_v<deposit_traits::result_type, bool>);
    static_assert(std::is_same_v<deposit_traits::argument_t<1>, double>);
    ok &= describe(deposit).rfind("2 argument(s)", 0) == 0;

    using balance_traits = snicholls::callable_traits<decltype(&account::balance)>;
    static_assert(balance_traits::is_member && balance_traits::is_noexcept);
    static_assert(std::is_same_v<balance_traits::class_type, account>);
    ok &= describe(&account::balance).find("noexcept") != std::string::npos;

    // A generic lambda has no single signature: describe() does not accept it, at compile time.
    const auto generic = [](auto x) { return x; };
    static_assert(!snicholls::single_signature_callable<decltype(generic)>);
    std::printf("  generic lambda:    no single signature\n");

    std::printf(ok ? "callable_traits demo: ok\n" : "callable_traits demo: FAILED\n");
    return ok ? 0 : 1;
}
