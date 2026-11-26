//
//  persist_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/fsm/persist.hpp>: a machine with two regions and a sub-machine saved to the reference
//  key-value store and loaded into a fresh one, the record it writes, a stale writer refused, and the records that
//  are refused on load (corrupt, a state the table no longer has). Exits non-zero if a result is not the one
//  documented.
//

#include <snicholls/core/fsm/persist.hpp>
#include <snicholls/core/fsm/state_machine.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <cstdio>
#include <string>

namespace {

struct go {};
struct dig {};

struct inner {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(*"shallow"_s + event<dig> = "deep"_s);
    }
};

struct outer {
    auto operator()() const
    {
        using namespace snicholls::fsm;
        return make_transition_table(
            *"parked"_s + event<go> = state<inner>,
            *"lights off"_s + event<go> = "lights on"_s);
    }
};

} // namespace

int main()
{
    using namespace snicholls::fsm;
    std::printf("snicholls-core fsm persist demo\n");
    bool ok = true;
    snicholls::reference_key_value_store store;

    //! [save_load]
    sm<outer> machine;
    ok &= machine.process_event(go{}) && machine.process_event(dig{});
    const auto v1 = save(store, "job/7", machine, 0);   // 0: a new key; the result is the version written
    ok &= v1.is_ok() && *v1 == 1;

    sm<outer> restored;                                 // another process
    const auto read = load(store, "job/7", restored);   // every region, the sub-machine's included
    ok &= read.is_ok() && *read == 1;
    ok &= restored.is(state<inner>, "lights on"_s) && restored.is<inner>("deep"_s);

    const auto stale = save(store, "job/7", restored, 0);   // claims there was nothing: someone saved version 1
    ok &= stale.is_err() && stale.unwrap_err_unchecked() == fsm_errc::stale;
    ok &= save(store, "job/7", restored, *read).is_ok();    // with the version it read: version 2
    //! [save_load]

    std::string record;
    static_cast<void>(snicholls::kv_get(store, "job/7", record));
    std::printf("%s", record.c_str());
    ok &= record == "snicholls.fsm.v1\nmachine outer\nversion 2\nregions 3\n1 inner\n3 lights on\n1 deep\n";

    // Records that are refused, leaving the machine as it was
    sm<outer> fresh;
    ok &= snicholls::kv_put(store, "job/8", std::string("not a record")).is_ok();
    ok &= load(store, "job/8", fresh).unwrap_err_unchecked() == fsm_errc::corrupt;
    ok &= snicholls::kv_put(store, "job/9", std::string("snicholls.fsm.v1\nmachine outer\nversion 1\nregions 3\n"
                                                         "0 parked\n0 retired\n0 shallow\n"))
              .is_ok();
    ok &= load(store, "job/9", fresh).unwrap_err_unchecked() == fsm_errc::unknown_state;   // no state "retired"
    ok &= load(store, "job/10", fresh).unwrap_err_unchecked() == fsm_errc::not_found;
    ok &= snicholls::kv_put(store, "job/11", std::string("snicholls.fsm.v1\nmachine elsewhere\nversion 1\nregions 3\n"
                                                          "0 parked\n0 lights off\n0 shallow\n"))
              .is_ok();
    ok &= load(store, "job/11", fresh).unwrap_err_unchecked() == fsm_errc::wrong_machine;   // another machine's record
    ok &= fresh.is("parked"_s, "lights off"_s);
    std::printf("  corrupt, unknown_state, wrong_machine and not_found leave the machine where it was\n");

    // The record is also available without a store
    const std::string text = to_record(restored, 5);
    sm<outer> from_text;
    ok &= from_record(text, from_text).is_ok() && from_text.is<inner>("deep"_s);
    ok &= version_key("job/7", 2) == "job/7/v/00000000000000000002";

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
