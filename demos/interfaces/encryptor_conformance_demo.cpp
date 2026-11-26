//
//  encryptor_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/encryptor_conformance.hpp>: the kit run on the test double. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/encryptor_conformance.hpp>
#include <snicholls/core/interfaces/encryptor_reference.hpp>

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string_view>
#include <vector>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}
} // namespace

namespace {
insecure_test_encryptor with_keys()
{
    insecure_test_encryptor e;   // NOT encryption: a test double
    std::array<std::byte, 32> k{};
    k.fill(std::byte{7});
    static_cast<void>(e.add_key("k-2026", k));
    k.fill(std::byte{8});
    static_cast<void>(e.add_key("k-2027", k));
    return e;
}
} // namespace

namespace {
struct probe {
    insecure_test_encryptor e = with_keys();
    auto& encryptor() { return e; }
    static std::string_view key() { return "k-2026"; }
    static std::string_view other_key() { return "k-2027"; }
    void forget(std::string_view id) { e.remove_key(id); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/encryptor_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_encryptor_conformance<probe>();
    //! [kit]
    check(report.ok(), "the test double honours the contract (which says nothing of strength)");
    check(report.checks == 8, "eight checks ran");
    return failures == 0 ? 0 : 1;
}
