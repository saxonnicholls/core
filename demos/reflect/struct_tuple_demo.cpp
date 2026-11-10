//
//  struct_tuple_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/reflect/struct_tuple.hpp>: plain structs <-> tuples, with no member count
//  anywhere. Exits non-zero if any result is not the one documented.
//

#include <snicholls/core/reflect/struct_tuple.hpp>

#include <cstdio>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

namespace demo {

// Plain structs, e.g. from another API or a database row - no N anywhere.
struct leg_row { double px; int qty; };
struct order_row { long id; leg_row leg; std::string note; };
struct holder { std::unique_ptr<int> p; std::string label; };

void print(const order_row& o)
{
    std::printf("  order_row{id=%ld, px=%g, qty=%d, note='%s'}\n", o.id, o.leg.px, o.leg.qty, o.note.c_str());
}

} // namespace demo

int main()
{
    using namespace demo;
    std::printf("snicholls-core struct_tuple demo\n");
    bool ok = true;

    order_row o{7, {9.5, 100}, "vwap"};

    // tie: references to the members, assignable as a whole; a nested struct stays a struct.
    //! [tie]
    auto refs = snicholls::tie(o);
    std::get<1>(refs).qty = 200;
    std::get<2>(refs) = "twap";
    snicholls::tie(o) = std::make_tuple(8L, leg_row{10.25, 300}, std::string("close"));
    //! [tie]
    std::printf("after tie assignment:\n");
    print(o);
    ok &= o.id == 8 && o.leg.qty == 300 && o.note == "close";

    //! [to_tuple]
    // to_tuple / from_tuple: values out, a struct back in. Strict: sizes must match, no narrowing.
    const auto values = snicholls::to_tuple(o);   // std::tuple<long, leg_row, std::string>
    const order_row copy = snicholls::from_tuple<order_row>(values);
    const leg_row from_pair = snicholls::from_tuple<leg_row>(std::make_pair(1.5, 2));
    //! [to_tuple]
    std::printf("to_tuple -> from_tuple:\n");
    print(copy);
    ok &= std::get<0>(values) == 8 && copy.note == "close" && from_pair.qty == 2;

    // Move-only members: to_tuple moves from an rvalue, from_tuple moves from an rvalue tuple.
    holder h{std::make_unique<int>(42), "answer"};
    std::get<1>(snicholls::tie(h)) = "the answer";
    auto moved = snicholls::to_tuple(std::move(h));
    const holder back = snicholls::from_tuple<holder>(std::move(moved));
    std::printf("struct round-trip: *p=%d label='%s'\n", *back.p, back.label.c_str());
    ok &= *back.p == 42 && back.label == "the answer";

    // Everything is constexpr.
    static_assert(std::get<1>(snicholls::to_tuple(leg_row{1.5, 2})) == 2);
    static_assert(snicholls::from_tuple<leg_row>(std::make_tuple(3.5, 4)).qty == 4);

    // noexcept is exact: no allocation -> noexcept; std::string copy -> may throw; move -> noexcept.
    static_assert(noexcept(snicholls::tie(o)));
    static_assert(noexcept(snicholls::to_tuple(leg_row{})));
    static_assert(!noexcept(snicholls::to_tuple(std::as_const(o))));
    static_assert(noexcept(snicholls::to_tuple(std::move(o))));

    std::printf(ok ? "struct_tuple demo: ok\n" : "struct_tuple demo: FAILED\n");
    return ok ? 0 : 1;
}
