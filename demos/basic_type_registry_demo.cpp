//
//  basic_type_registry_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/basic_type_registry.hpp>: messages arrive as a
//  (type name, payload) pair; the registry builds the named type with a
//  caller-supplied format and hands it to a per-type handler. Exits non-zero
//  if any result is not the one documented.
//

#include <snicholls/core/basic_type_registry.hpp>
#include <snicholls/core/overloaded.hpp>

#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct Celsius { int value; };
struct Label   { std::string text; };

// The caller's format: here, plain text.
struct TextFormat {
    Celsius operator()(std::type_identity<Celsius>, std::string_view s) const { return Celsius{ std::stoi(std::string(s)) }; }
    Label operator()(std::type_identity<Label>, std::string_view s) const { return Label{ std::string(s) }; }
};

} // namespace

int main()
{
    std::printf("snicholls-core basic_type_registry demo\n");
    bool ok = true;

    //! [dispatch]
    using Registry = snicholls::basic_type_registry<
        snicholls::named<"celsius", Celsius>,
        snicholls::named<"label", Label>>;

    static_assert(Registry::name_of<Label>() == "label");   // an unregistered type would not compile

    int degrees = 0;
    std::string text;
    const auto handler = snicholls::overloaded{
        [&](Celsius&& c) { degrees = c.value; },
        [&](Label&& l) { text = std::move(l.text); },
    };

    const std::vector<std::pair<std::string, std::string>> inbox{
        {"celsius", "21"}, {"label", "hello"}, {"kelvin", "294"}};
    int handled = 0;
    for (const auto& [name, payload] : inbox)
        if (Registry::dispatch(name, payload, handler, TextFormat{}))
            ++handled;
    //! [dispatch]

    std::printf("  handled %d of %zu: degrees=%d text='%s'\n", handled, inbox.size(), degrees, text.c_str());
    ok &= handled == 2 && degrees == 21 && text == "hello";
    ok &= Registry::count == 2 && Registry::contains("label") && !Registry::contains("kelvin");

    for (const auto name : Registry::names)
        std::printf("  registered: %.*s\n", static_cast<int>(name.size()), name.data());

    std::printf(ok ? "basic_type_registry demo: ok\n" : "basic_type_registry demo: FAILED\n");
    return ok ? 0 : 1;
}
