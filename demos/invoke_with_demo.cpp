//
//  invoke_with_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/invoke_with.hpp>: a handler set called with arguments that
//  exist only at run time, first as std::any (invoke_any), then from the words of a
//  command line through a source of the demo's own (invoke_with), and a table of such
//  sets by name. Exits non-zero if any call does not go where documented.
//

#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/invoke_with.hpp>

#include <any>
#include <charconv>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

//! [source]
// A source of the demo's own: the arguments are the words of a command line, converted on
// request. get<T>(i) returns std::optional<T>: a value when word i reads as a T, empty to refuse.
// Which conversion applies to which T is a first_match over a type tag, not an if constexpr chain.
// Each branch's constraint admits only what its body can take: std::from_chars takes no bool
// and no character type, and snicholls::integer excludes exactly those.
struct token_source {
    std::vector<std::string_view> words;

    [[nodiscard]] std::size_t size() const noexcept { return words.size(); }

    template <typename T>
    [[nodiscard]] std::optional<T> get(std::size_t i) const
    {
        const std::string_view word = words[i];
        return snicholls::first_match{
            [word](std::same_as<bool> auto*) -> std::optional<bool> {
                if (word == "true" || word == "on") return true;
                if (word == "false" || word == "off") return false;
                return std::nullopt;
            },
            [word](snicholls::integer auto* tag) {
                using number = std::remove_pointer_t<decltype(tag)>;
                number value{};
                const auto [end, error] = std::from_chars(word.data(), word.data() + word.size(), value);
                return error == std::errc{} && end == word.data() + word.size() ? std::optional<number>{value} : std::nullopt;
            },
            [word](std::string*) { return std::optional<std::string>{std::string{word}}; },
            [](auto* tag) { return std::optional<std::remove_pointer_t<decltype(tag)>>{}; },   // refused
        }(static_cast<T*>(nullptr));
    }
};
//! [source]

} // namespace

int main()
{
    std::printf("snicholls-core invoke_with demo\n");
    bool ok = true;

    //! [commands]
    // A handler set whose callables have concrete parameter lists, which invoke_any reads.
    const auto volume = snicholls::overloaded{
        [](int level) { return "volume " + std::to_string(level); },
        [](int level, bool fade) { return "volume " + std::to_string(level) + (fade ? " with fade" : ""); },
        [](const std::string& preset) { return "preset " + preset; },
    };

    // Arguments that exist only at run time: here written out, in practice read off a socket or
    // handed over by a script. Each std::any holds exactly a parameter's type.
    const std::vector<std::any> line{42};
    auto r = snicholls::invoke_any(volume, line);   // snicholls::result<std::any, snicholls::overload_mismatch>
    if (r)
        std::printf("  %s\n", std::any_cast<std::string>(r.unwrap()).c_str());   // volume 42

    r = snicholls::invoke_any(volume, {7, true});               // the (int, bool) handler
    std::printf("  %s\n", std::any_cast<std::string>(r.unwrap()).c_str());
    r = snicholls::invoke_any(volume, {std::string{"loud"}});   // the (const std::string&) handler
    std::printf("  %s\n", std::any_cast<std::string>(r.unwrap()).c_str());

    // Every handler returns a std::string, so ask for that instead of a std::any.
    const auto typed = snicholls::invoke_any<std::string>(volume, {3});
    std::printf("  %s\n", typed.unwrap().c_str());

    const auto three = snicholls::invoke_any(volume, {1, 2, 3});   // nothing takes three arguments
    const auto wrong = snicholls::invoke_any(volume, {2.5});       // one argument, but no handler takes a double
    std::printf("  {1, 2, 3}: %s\n", snicholls::error_text(three.unwrap_err()).c_str());
    std::printf("  {2.5}:     %s\n", snicholls::error_text(wrong.unwrap_err()).c_str());
    //! [commands]

    ok &= std::any_cast<std::string>(snicholls::invoke_any(volume, line).unwrap()) == "volume 42";
    ok &= std::any_cast<std::string>(snicholls::invoke_any(volume, {7, true}).unwrap()) == "volume 7 with fade";
    ok &= std::any_cast<std::string>(snicholls::invoke_any(volume, {std::string{"loud"}}).unwrap()) == "preset loud";
    ok &= typed.unwrap() == "volume 3";
    ok &= three.is_err() && three.unwrap_err() == snicholls::overload_mismatch::arity;
    ok &= wrong.is_err() && wrong.unwrap_err() == snicholls::overload_mismatch::types;

    // The same set from words: "7 on" reaches (int, bool), "loud" reaches (const std::string&),
    // "7" reaches (int) because it is written before the string handler.
    const auto words = [&](std::initializer_list<std::string_view> w) {
        return snicholls::invoke_with<std::string>(volume, token_source{std::vector<std::string_view>{w}});
    };
    std::printf("  words \"7 on\":   %s\n", words({"7", "on"}).unwrap().c_str());
    std::printf("  words \"loud\":   %s\n", words({"loud"}).unwrap().c_str());
    std::printf("  words \"7\":      %s\n", words({"7"}).unwrap().c_str());
    std::printf("  words \"7 maybe\": %s\n", snicholls::error_text(words({"7", "maybe"}).unwrap_err()).c_str());

    ok &= words({"7", "on"}).unwrap() == "volume 7 with fade";
    ok &= words({"loud"}).unwrap() == "preset loud";
    ok &= words({"7"}).unwrap() == "volume 7";
    ok &= words({"7", "maybe"}).unwrap_err() == snicholls::overload_mismatch::types;
    ok &= words({}).unwrap_err() == snicholls::overload_mismatch::arity;

    //! [table]
    // Sets of different types behind one signature: a command table.
    using outcome = snicholls::result<std::any, snicholls::overload_mismatch>;
    std::map<std::string, std::function<outcome(std::span<const std::any>)>> commands;
    commands["volume"] = [&volume](std::span<const std::any> args) { return snicholls::invoke_any(volume, args); };
    commands["mute"] = [](std::span<const std::any> args) {
        return snicholls::invoke_any([](bool on) { return std::string{on ? "muted" : "unmuted"}; }, args);
    };

    const std::vector<std::any> muted{true};
    const auto m = commands.at("mute")(muted);
    std::printf("  mute {true}: %s\n", std::any_cast<std::string>(m.unwrap()).c_str());
    //! [table]

    ok &= std::any_cast<std::string>(m.unwrap()) == "muted";
    ok &= std::any_cast<std::string>(commands.at("volume")(line).unwrap()) == "volume 42";
    ok &= commands.at("mute")(line).unwrap_err() == snicholls::overload_mismatch::types;

    std::printf(ok ? "invoke_with demo: ok\n" : "invoke_with demo: FAILED\n");
    return ok ? 0 : 1;
}
