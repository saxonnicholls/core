//
//  persist_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  fsm::save and fsm::load over the reference key-value store: round trips of one region, of several and of
//  sub-machines; the record's text; versions and the compare-and-set (a stale writer refused, nothing written); the
//  history; every error of load (not_found, corrupt in each line, wrong_machine, shape_mismatch, unknown_state,
//  state_name_mismatch, the store down); a record that stays valid when enumerators are appended, and one refused when
//  they are renamed or reordered. And the property: for random event sequences, a machine
//  saved and loaded at a random point behaves exactly as the one that never was.
//

#include <snicholls/core/fsm/lifecycle.hpp>
#include <snicholls/core/fsm/persist.hpp>
#include <snicholls/core/interfaces/key_value_store_reference.hpp>
#include <snicholls/core/stats/random.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace {

using namespace snicholls::fsm;
using namespace snicholls::fsm::lifecycle_event;
using store_t = snicholls::reference_key_value_store;

static_assert(snicholls::kv_writable<store_t>);

std::string read(const store_t& store, std::string_view key)
{
    std::string out;
    EXPECT_TRUE(snicholls::kv_get(store, key, out).is_ok());
    return out;
}

TEST(FsmPersist, SaveThenLoadRestoresTheState)
{
    store_t store;
    sm<lifecycle> m;
    ASSERT_TRUE(m.process_event(submit{}));
    const auto v = save(store, "p/1", m, 0);
    ASSERT_TRUE(v.is_ok());
    EXPECT_EQ(*v, 1u);
    EXPECT_EQ(read(store, "p/1"), "snicholls.fsm.v1\nmachine lifecycle\nversion 1\nregions 1\n1 pending\n");
    EXPECT_EQ(read(store, "p/1/v/00000000000000000001"), read(store, "p/1"));
    sm<lifecycle> back;
    const auto loaded = load(store, "p/1", back);
    ASSERT_TRUE(loaded.is_ok());
    EXPECT_EQ(*loaded, 1u);
    EXPECT_EQ(back.current(), lifecycle_state::pending);
}

TEST(FsmPersist, AStaleWriterIsRefusedAndWritesNothing)
{
    store_t store;
    sm<lifecycle> a;
    ASSERT_TRUE(save(store, "p/2", a, 0).is_ok());   // version 1
    sm<lifecycle> b;
    ASSERT_TRUE(load(store, "p/2", b).is_ok());
    ASSERT_TRUE(a.process_event(cancel{}));
    ASSERT_TRUE(save(store, "p/2", a, 1).is_ok());   // version 2
    ASSERT_TRUE(b.process_event(submit{}));
    const auto stale = save(store, "p/2", b, 1);
    ASSERT_TRUE(stale.is_err());
    EXPECT_EQ(stale.unwrap_err_unchecked(), fsm_errc::stale);
    sm<lifecycle> check;
    EXPECT_EQ(*load(store, "p/2", check), 2u);
    EXPECT_EQ(check.current(), lifecycle_state::canceled);   // a's write stands
    EXPECT_EQ(save(store, "p/2", b, 0).unwrap_err_unchecked(), fsm_errc::stale);   // "new" when it is not
}

TEST(FsmPersist, EveryVersionIsKeptAndCanBeLoaded)
{
    store_t store;
    sm<lifecycle> m;
    std::uint64_t v = 0;
    for (int step = 0; step < 3; ++step) {
        const bool moved = step == 0 ? m.process_event(submit{}) : step == 1 ? m.process_event(confirm{}) : m.process_event(settle{});
        ASSERT_TRUE(moved);
        v = *save(store, "p/3", m, v);
    }
    EXPECT_EQ(v, 3u);
    sm<lifecycle> then;
    EXPECT_EQ(*load_version(store, "p/3", 2, then), 2u);
    EXPECT_EQ(then.current(), lifecycle_state::confirmed);
    EXPECT_EQ(load_version(store, "p/3", 9, then).unwrap_err_unchecked(), fsm_errc::not_found);
    EXPECT_EQ(version_key("k", 12345), "k/v/00000000000000012345");
}

TEST(FsmPersist, LoadErrorsLeaveTheMachineUnchanged)
{
    store_t store;
    sm<lifecycle> m;
    ASSERT_TRUE(m.process_event(submit{}));
    EXPECT_EQ(load(store, "absent", m).unwrap_err_unchecked(), fsm_errc::not_found);
    const std::string good = "snicholls.fsm.v1\nmachine lifecycle\nversion 4\nregions 1\n3 settled\n";
    const std::string bad[] = {
        "",
        "snicholls.fsm.v2\nmachine lifecycle\nversion 4\nregions 1\n3 settled\n",
        "snicholls.fsm.v1\nmachin lifecycle\nversion 4\nregions 1\n3 settled\n",
        "snicholls.fsm.v1\nmachine lifecycle\nversion x\nregions 1\n3 settled\n",
        "snicholls.fsm.v1\nmachine lifecycle\nversion 4\nregion 1\n3 settled\n",
        "snicholls.fsm.v1\nmachine lifecycle\nversion 4\nregions 1\nsettled\n",
        "snicholls.fsm.v1\nmachine lifecycle\nversion 4\nregions 1\n3x settled\n",
        "snicholls.fsm.v1\nmachine lifecycle\nversion 4\nregions 1\n3 settled\nextra\n",
    };
    for (const auto& record : bad) {
        ASSERT_TRUE(snicholls::kv_put(store, "bad", record).is_ok());
        EXPECT_EQ(load(store, "bad", m).unwrap_err_unchecked(), fsm_errc::corrupt) << record;
    }
    ASSERT_TRUE(snicholls::kv_put(store, "shape", std::string("snicholls.fsm.v1\nmachine lifecycle\nversion 1\nregions 2\n0 a\n0 b\n")).is_ok());
    EXPECT_EQ(load(store, "shape", m).unwrap_err_unchecked(), fsm_errc::shape_mismatch);
    ASSERT_TRUE(snicholls::kv_put(store, "unknown", std::string("snicholls.fsm.v1\nmachine lifecycle\nversion 1\nregions 1\n99 gone\n")).is_ok());
    EXPECT_EQ(load(store, "unknown", m).unwrap_err_unchecked(), fsm_errc::unknown_state);
    EXPECT_EQ(m.current(), lifecycle_state::pending);
    ASSERT_TRUE(snicholls::kv_put(store, "good", good).is_ok());
    EXPECT_EQ(*load(store, "good", m), 4u);
    EXPECT_EQ(m.current(), lifecycle_state::settled);
}

TEST(FsmPersist, AStoreThatFailsIsAnErrorAndNothingIsWritten)
{
    store_t store;
    sm<lifecycle> m;
    store.set_unavailable(true);
    EXPECT_EQ(save(store, "p/4", m, 0).unwrap_err_unchecked(), fsm_errc::store_unavailable);
    EXPECT_EQ(load(store, "p/4", m).unwrap_err_unchecked(), fsm_errc::store_unavailable);
    store.set_unavailable(false);
    store.inject_failure(snicholls::kv_errc::io, 1);   // the second write of the batch fails: all or nothing
    EXPECT_EQ(save(store, "p/4", m, 0).unwrap_err_unchecked(), fsm_errc::store_failed);
    EXPECT_EQ(load(store, "p/4", m).unwrap_err_unchecked(), fsm_errc::not_found);
    EXPECT_TRUE(save(store, "p/4", m, 0).is_ok());
}

// Appending an enumerator keeps old records valid: they hold values. The machine keeps its name across the versions.
enum class v1_state : std::uint8_t { open, closed };
enum class v2_state : std::uint8_t { open, closed, archived };
enum class v3_state : std::uint8_t { open, shut, archived };       // `closed` renamed
enum class v4_state : std::uint8_t { closed, open, archived };     // values reordered
struct go {};
namespace v1 {
    struct door {
        constexpr auto operator()() const noexcept
        {
            return make_transition_table(*enum_state<v1_state::open> + event<go> = enum_state<v1_state::closed>);
        }
    };
} // namespace v1
namespace v2 {
    struct door {
        constexpr auto operator()() const noexcept
        {
            return make_transition_table(*enum_state<v2_state::open> + event<go> = enum_state<v2_state::closed>,
                                         enum_state<v2_state::closed> + event<go> = enum_state<v2_state::archived>);
        }
    };
} // namespace v2
namespace v3 {
    struct door {
        constexpr auto operator()() const noexcept
        {
            return make_transition_table(*enum_state<v3_state::open> + event<go> = enum_state<v3_state::shut>,
                                         enum_state<v3_state::shut> + event<go> = enum_state<v3_state::archived>);
        }
    };
} // namespace v3
namespace v4 {
    struct door {
        constexpr auto operator()() const noexcept
        {
            return make_transition_table(*enum_state<v4_state::open> + event<go> = enum_state<v4_state::closed>,
                                         enum_state<v4_state::closed> + event<go> = enum_state<v4_state::archived>);
        }
    };
} // namespace v4
struct gate {   // another machine, with the same enumeration and states as v1::door
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*enum_state<v1_state::open> + event<go> = enum_state<v1_state::closed>);
    }
};

TEST(FsmPersist, ARecordStaysValidWhenEnumeratorsAreAppended)
{
    store_t store;
    sm<v1::door> old;
    ASSERT_TRUE(old.process_event(go{}));
    ASSERT_TRUE(save(store, "door", old, 0).is_ok());
    sm<v2::door> upgraded;
    ASSERT_TRUE(load(store, "door", upgraded).is_ok());
    EXPECT_EQ(upgraded.current(), v2_state::closed);
    EXPECT_TRUE(upgraded.process_event(go{}));
    EXPECT_EQ(upgraded.current(), v2_state::archived);
}

TEST(FsmPersist, ARecordOfAnotherMachineIsRefusedAndChangesNothing)
{
    store_t store;
    sm<v1::door> door;
    ASSERT_TRUE(door.process_event(go{}));
    ASSERT_TRUE(save(store, "door", door, 0).is_ok());
    sm<gate> other;   // same enumeration, same values, same names: only the machine's name differs
    EXPECT_EQ(load(store, "door", other).unwrap_err_unchecked(), fsm_errc::wrong_machine);
    EXPECT_EQ(other.current(), v1_state::open);
    sm<lifecycle> life;
    ASSERT_TRUE(life.process_event(submit{}));
    EXPECT_EQ(load(store, "door", life).unwrap_err_unchecked(), fsm_errc::wrong_machine);
    EXPECT_EQ(from_record(read(store, "door"), life).unwrap_err_unchecked(), fsm_errc::wrong_machine);
    EXPECT_EQ(life.current(), lifecycle_state::pending);
    // the name is the definition's unqualified name, so the same name in another namespace is the same machine
    sm<v2::door> same_name;
    EXPECT_TRUE(load(store, "door", same_name).is_ok());
}

TEST(FsmPersist, AStateWhoseNameDisagreesWithItsValueIsRefused)
{
    store_t store;
    sm<v1::door> door;
    ASSERT_TRUE(door.process_event(go{}));
    ASSERT_TRUE(save(store, "door", door, 0).is_ok());   // "1 closed"

    sm<v3::door> renamed;   // value 1 is now `shut`
    EXPECT_EQ(load(store, "door", renamed).unwrap_err_unchecked(), fsm_errc::state_name_mismatch);
    EXPECT_EQ(renamed.current(), v3_state::open);
    sm<v4::door> reordered;   // value 1 is now `open`: by value alone the door would be restored open
    EXPECT_EQ(load(store, "door", reordered).unwrap_err_unchecked(), fsm_errc::state_name_mismatch);
    EXPECT_EQ(reordered.current(), v4_state::open);
    EXPECT_TRUE(reordered.process_event(go{}));   // unchanged, and still working

    sm<lifecycle> life;
    ASSERT_TRUE(snicholls::kv_put(store, "p", std::string("snicholls.fsm.v1\nmachine lifecycle\nversion 1\nregions 1\n"
                                                          "3 pending\n")).is_ok());
    EXPECT_EQ(load(store, "p", life).unwrap_err_unchecked(), fsm_errc::state_name_mismatch);
    // an unknown value is unknown_state, before any name is compared
    ASSERT_TRUE(snicholls::kv_put(store, "p", std::string("snicholls.fsm.v1\nmachine lifecycle\nversion 1\nregions 1\n"
                                                          "99 pending\n")).is_ok());
    EXPECT_EQ(load(store, "p", life).unwrap_err_unchecked(), fsm_errc::unknown_state);
    // a state recorded without a name is restored by its value
    ASSERT_TRUE(snicholls::kv_put(store, "p", std::string("snicholls.fsm.v1\nmachine lifecycle\nversion 1\nregions 1\n"
                                                          "3 \n")).is_ok());
    EXPECT_TRUE(load(store, "p", life).is_ok());
    EXPECT_EQ(life.current(), lifecycle_state::settled);
}

// A composite whose sub-machine has a state enumeration: its slots are checked by name too.
enum class job_state : std::uint8_t { queued, running, done };
struct job {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(*enum_state<job_state::queued> + event<go> = enum_state<job_state::running>,
                                     enum_state<job_state::running> + event<go> = enum_state<job_state::done>);
    }
};
struct worker {
    constexpr auto operator()() const noexcept { return make_transition_table(*"idle"_s + event<go> = state<job>); }
};

TEST(FsmPersist, ASubMachinesStatesAreCheckedByNameToo)
{
    store_t store;
    sm<worker> w;
    ASSERT_TRUE(w.process_event(go{}));
    ASSERT_TRUE(w.process_event(go{}));
    ASSERT_TRUE(save(store, "w", w, 0).is_ok());
    EXPECT_EQ(read(store, "w"), "snicholls.fsm.v1\nmachine worker\nversion 1\nregions 2\n1 job\n1 running\n");
    sm<worker> back;
    ASSERT_TRUE(load(store, "w", back).is_ok());
    EXPECT_EQ(back.snapshot(), w.snapshot());
    ASSERT_TRUE(snicholls::kv_put(store, "w", std::string("snicholls.fsm.v1\nmachine worker\nversion 1\nregions 2\n"
                                                          "1 job\n1 done\n")).is_ok());
    sm<worker> fresh;
    EXPECT_EQ(load(store, "w", fresh).unwrap_err_unchecked(), fsm_errc::state_name_mismatch);
    EXPECT_EQ(fresh.snapshot(), sm<worker>{}.snapshot());
}

// ---- the property: a machine saved and loaded behaves as the one that never was -----------------------------------

struct a {};
struct b {};
struct c {};
struct d {};

struct leaf {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(
            "l1"_s(H) + event<a> = "l2"_s,
             "l2"_s + event<b> = "l3"_s,
             "l3"_s + event<a> = "l1"_s,
            *"m1"_s + event<c> = "m2"_s,
             "m2"_s + event<c> = "m1"_s);
    }
};
struct rich {
    constexpr auto operator()() const noexcept
    {
        return make_transition_table(
            *"idle"_s + event<a> = state<leaf>,
             state<leaf> + event<d> = "parked"_s,
             "parked"_s + event<b> = state<leaf>,
             "parked"_s + event<c> = "idle"_s,
            *"r1"_s + event<b> [([](const b&, const int& n) { return n % 2 == 0; })] = "r2"_s,
             "r2"_s + event<d> / [](int& n) { ++n; } = "r1"_s);
    }
};

template <class Machine, class Pick>
void check_property(std::uint64_t seed, Pick pick, const char* key)
{
    snicholls::stats::wyrand rng(seed);
    for (int round = 0; round < 64; ++round) {
        store_t store;
        int counter_original = 0;
        int counter_restored = 0;
        Machine original{counter_original};
        const int length = 40;
        const int cut = static_cast<int>(snicholls::stats::uniform_index(rng, static_cast<std::uint64_t>(length)));
        std::uint64_t draws[40];
        for (auto& x : draws)
            x = snicholls::stats::uniform_index(rng, 4);
        for (int i = 0; i < cut; ++i)
            static_cast<void>(pick(original, draws[i]));
        ASSERT_TRUE(save(store, key, original, 0).is_ok());
        Machine restored{counter_restored};
        ASSERT_TRUE(load(store, key, restored).is_ok());
        counter_restored = counter_original;   // the dependency is the caller's: it comes back with the caller's object
        for (int i = cut; i < length; ++i) {
            const bool x = pick(original, draws[i]);
            const bool y = pick(restored, draws[i]);
            ASSERT_EQ(x, y) << "seed " << seed << " round " << round << " step " << i;
            ASSERT_EQ(original.snapshot(), restored.snapshot()) << "seed " << seed << " round " << round << " step " << i;
        }
    }
}

TEST(FsmPersistProperty, ARestoredMachineBehavesAsTheOneNeverSaved)
{
    const auto rich_pick = [](auto& m, std::uint64_t k) {
        return k == 0 ? m.process_event(a{}) : k == 1 ? m.process_event(b{}) : k == 2 ? m.process_event(c{}) : m.process_event(d{});
    };
    for (std::uint64_t seed = 1; seed <= 8; ++seed)
        check_property<sm<rich>>(seed, rich_pick, "rich");

    const auto lifecycle_pick = [](auto& m, std::uint64_t k) {
        switch (k) {
        case 0: return m.process_event(submit{}) || m.process_event(confirm{});
        case 1: return m.process_event(settle{});
        case 2: return m.process_event(fail{}) || m.process_event(expire{});
        default: return m.process_event(reverse{}) || m.process_event(cancel{});
        }
    };
    struct lifecycle_with_int : sm<lifecycle> {
        explicit lifecycle_with_int(int&) {}
    };
    for (std::uint64_t seed = 11; seed <= 18; ++seed)
        check_property<lifecycle_with_int>(seed, lifecycle_pick, "life");
}

} // namespace
