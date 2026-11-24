//
//  lifecycle_payment_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  A payment's lifecycle (fsm/lifecycle.hpp) kept in a key-value store (fsm/persist.hpp): every transition saved
//  with the next version, a process that dies between confirmation and settlement and the payment restored by the
//  next one, two workers racing on the same payment (the stale one refused, reloads, and finds the work done), a
//  store that is down, the history read back, and a failed and a reversed payment. Exits non-zero if any result is
//  not the one documented.
//

#include <snicholls/core/fsm/lifecycle.hpp>
#include <snicholls/core/fsm/persist.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>

#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace {

using namespace snicholls::fsm;
using namespace snicholls::fsm::lifecycle_event;

const char* name(lifecycle_state s)
{
    static std::string text;
    text = snicholls::enum_name(s);
    return text.c_str();
}

// One worker: a machine and the version it last read or wrote.
struct worker {
    sm<lifecycle> machine{};
    std::uint64_t version = 0;

    template <class Event>
    snicholls::result<std::uint64_t, fsm_errc> handle(snicholls::reference_key_value_store& store, const char* key,
                                                      const Event& e)
    {
        if (!machine.process_event(e))
            return snicholls::err(fsm_errc::no_transition);
        auto saved = save(store, key, machine, version);
        if (saved.is_ok())
            version = *saved;
        return saved;
    }
};

} // namespace

int main()
{
    std::printf("snicholls-core fsm: a payment lifecycle, persisted\n");
    bool ok = true;
    snicholls::reference_key_value_store store;
    const char* key = "payment/2026-11-24/0001";

    //! [persist]
    {
        worker w;   // the first process
        ok &= w.handle(store, key, submit{}).is_ok();    // version 1: pending
        ok &= w.handle(store, key, confirm{}).is_ok();   // version 2: confirmed
        std::printf("  process 1: %s at version %llu, then it crashes\n", name(w.machine.current()),
                    static_cast<unsigned long long>(w.version));
    }   // the process dies here: nothing in memory survives

    worker restarted;   // the next process
    const auto loaded = load(store, key, restarted.machine);
    ok &= loaded.is_ok() && *loaded == 2 && restarted.machine.current() == lifecycle_state::confirmed;
    restarted.version = *loaded;
    std::printf("  process 2: loaded %s at version %llu\n", name(restarted.machine.current()),
                static_cast<unsigned long long>(restarted.version));
    //! [persist]

    // Two workers race on the settlement webhook: both loaded version 2
    worker racer;
    racer.version = *load(store, key, racer.machine);
    ok &= restarted.handle(store, key, settle{}).is_ok();   // version 3: settled
    const auto lost = racer.handle(store, key, settle{});    // also tries to write version 3
    ok &= lost.is_err() && lost.unwrap_err_unchecked() == fsm_errc::stale;
    std::printf("  the second worker's save: stale, so it reloads\n");
    racer.version = *load(store, key, racer.machine);
    ok &= racer.version == 3 && racer.machine.current() == lifecycle_state::settled;   // the work is done
    ok &= !racer.machine.process_event(settle{});                                     // and is not done twice

    // The store goes down: a save fails as a value, and nothing is written
    store.set_unavailable(true);
    sm<lifecycle> probe;
    const auto down = save(store, "payment/2026-11-24/0002", probe, 0);
    ok &= down.is_err() && down.unwrap_err_unchecked() == fsm_errc::store_unavailable;
    store.set_unavailable(false);
    ok &= load(store, "payment/2026-11-24/0002", probe).unwrap_err_unchecked() == fsm_errc::not_found;

    // The history: every version is kept under key/v/<version>
    for (std::uint64_t v = 1; v <= 3; ++v) {
        sm<lifecycle> then;
        ok &= load_version(store, key, v, then).is_ok();
        std::printf("  version %llu: %s\n", static_cast<unsigned long long>(v), name(then.current()));
        ok &= then.current() == (v == 1 ? lifecycle_state::pending : v == 2 ? lifecycle_state::confirmed : lifecycle_state::settled);
    }

    // A refund reverses the settled payment; a declined one fails
    ok &= restarted.version == 3;
    restarted.version = 3;
    ok &= restarted.handle(store, key, reverse{}).is_ok() && restarted.machine.current() == lifecycle_state::reversed;
    worker declined;
    ok &= declined.handle(store, "payment/2026-11-24/0003", submit{}).is_ok();
    ok &= declined.handle(store, "payment/2026-11-24/0003", fail{}).is_ok();
    ok &= declined.machine.current() == lifecycle_state::failed;
    ok &= declined.handle(store, "payment/2026-11-24/0003", settle{}).unwrap_err_unchecked() == fsm_errc::no_transition;

    std::string record;
    static_cast<void>(snicholls::kv_get(store, key, record));
    std::printf("  the record under %s:\n%s", key, record.c_str());
    ok &= record == "snicholls.fsm.v1\nmachine lifecycle\nversion 4\nregions 1\n7 reversed\n";

    std::printf("%s\n", ok ? "ok" : "FAILED");
    return ok ? 0 : 1;
}
