//
//  colour_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/colour.hpp>: one colour value, two renderings (CSS hex for user
//  interfaces, ANSI for terminals). Exits non-zero if a rendering is not the one documented.
//

#include <snicholls/core/colour.hpp>

#include <iostream>
#include <string_view>

//! [palette]
int main()
{
    int failures = 0;
    const auto show = [&](const char* name, snicholls::colour c) {
        std::cout << "  " << snicholls::ansi_start(c).view() << name << snicholls::ansi_reset << "  " << snicholls::hex(c).view() << '\n';
    };
    std::cout << "== palette (16-colour terminal code + RGB)\n";
    show("red", snicholls::colours::red);
    show("green", snicholls::colours::green);
    show("yellow", snicholls::colours::yellow);
    show("orange", snicholls::colours::orange);
    show("blue", snicholls::colours::blue);
    show("magenta", snicholls::colours::magenta);
    show("cyan", snicholls::colours::cyan);
    show("grey", snicholls::colours::grey);
    std::cout << "== any RGB colour (24-bit terminal sequence)\n";
    show("teal", snicholls::rgb(0x008080));
    failures += snicholls::hex(snicholls::rgb(0x008080)).view() == "#008080" ? 0 : 1;
    failures += snicholls::ansi_start(snicholls::rgb(0x008080)).view() == "\x1b[38;2;0;128;128m" ? 0 : 1;
    failures += snicholls::ansi_start(snicholls::colours::red).view() == "\x1b[31m" ? 0 : 1;
    snicholls::colour_text<4> t;
    t.append("abcdef");
    t.append('g');
    t.append_decimal(1);
    failures += t.view() == "abcd" && std::string_view(t) == "abcd" ? 0 : 1;   // bounded, never overflows
    std::cout << (failures ? "FAILED\n" : "all checks passed\n");
    return failures ? 1 : 0;
}
//! [palette]
