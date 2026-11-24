//
//  lifecycle_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/lifecycle.hpp>: the provider-neutral payment lifecycle, its presentation (colour,
//  label, tooltip, read by acout), and the reference machine taking a payment pending -> confirmed -> settled, with
//  its state on the payment itself. Exits non-zero if a state is not the one documented.
//

#include <snicholls/core/acout.hpp>
#include <snicholls/core/display.hpp>
#include <snicholls/core/fsm/lifecycle.hpp>
#include <snicholls/core/meta/enums.hpp>

#include <cstdio>

namespace {

struct payment {
    long amount_cents = 0;
    snicholls::fsm::lifecycle_state state = snicholls::fsm::lifecycle_state::requested;   // the state, on the object
};

} // namespace

int main()
{
    using namespace snicholls::fsm;
    using namespace snicholls::fsm::lifecycle_event;
    std::printf("snicholls-core fsm lifecycle demo\n");
    bool ok = true;

    //! [table]
    sm<lifecycle> machine;   // requested
    payment p{12'50};
    for (const auto step : {0, 1, 2}) {
        const bool moved = step == 0 ? machine.process_event(submit{})
                         : step == 1 ? machine.process_event(confirm{})
                                     : machine.process_event(settle{});
        ok &= moved;
        p.state = machine.current();   // pending, then confirmed, then settled
        snicholls::println(p.state);   // acout reads the state's label and colour
    }
    //! [table]
    ok &= p.state == lifecycle_state::settled;

    //! [restore]
    sm<lifecycle> later;   // another process, another day
    ok &= later.restore(p.state).is_ok();
    ok &= later.process_event(reverse{}) && later.current() == lifecycle_state::reversed;   // a refund
    ok &= is_final(later.current()) && !is_final(lifecycle_state::settled);
    ok &= !later.process_event(settle{});   // nothing leaves reversed
    //! [restore]

    //! [attributes]
    for (const auto s : snicholls::enum_values_v<lifecycle_state>) {
        const auto& a = attributes(s);
        std::printf("  %-9.*s %06x  %.*s\n", static_cast<int>(a.get<snicholls::label>().text.size()),
                    a.get<snicholls::label>().text.data(), static_cast<unsigned>(a.get<snicholls::colour>().rgb),
                    static_cast<int>(a.get<snicholls::tooltip>().text.size()), a.get<snicholls::tooltip>().text.data());
    }
    //! [attributes]
    ok &= attributes(lifecycle_state::settled).get<snicholls::colour>() == snicholls::colours::green;
    ok &= attributes(lifecycle_state::failed).get<snicholls::colour>() == snicholls::colours::red;
    ok &= attributes(lifecycle_state::pending).get<snicholls::colour>() == snicholls::colours::grey;

    // The paths that do not settle
    sm<lifecycle> declined;
    ok &= declined.process_event(submit{}) && declined.process_event(fail{}) && declined.current() == lifecycle_state::failed;
    sm<lifecycle> abandoned;
    ok &= abandoned.process_event(submit{}) && abandoned.process_event(expire{}) &&
          abandoned.current() == lifecycle_state::expired;
    sm<lifecycle> withdrawn;
    ok &= withdrawn.process_event(cancel{}) && withdrawn.current() == lifecycle_state::canceled;

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
