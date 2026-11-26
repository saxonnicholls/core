//
//  random_source_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The random_source interface: a source chosen at run time, and the adapters both ways.
//

#include <snicholls/core/interfaces/random_source.hpp>
#include <snicholls/core/interfaces/random_source_reference.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <cstdio>

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++failures;
}
} // namespace

int main()
{
    using namespace snicholls;
    std::puts("random_source_interface");
    //! [interface]
    const bool testing = true;   // a configuration decides which source the process uses
    any_random_source source = testing ? any_random_source::of(test_random_source{7})
                                       : any_random_source{};   // production: ts-moveables' source here
    std::array<std::byte, 32> key{};
    const std::errc e = source.fill(key.data(), key.size());
    //! [interface]
    check(!random_failed(e), "a 32-byte key from the source chosen at run time");
    test_random_source mine{7};
    random_source_adapter<test_random_source&> borrowed{mine};
    random_source_interface& as_interface = borrowed;
    std::array<std::byte, 32> again{};
    check(!random_failed(as_interface.fill(again.data(), again.size())) && again == key, "the same seed, borrowed through the interface");
    any_random_source empty;
    check(snicholls::random_failed(empty.fill(key.data(), key.size())), "an empty source fails closed");
    std::printf("%s\n", failures == 0 ? "all checks passed" : "FAILED");
    return failures == 0 ? 0 : 1;
}
