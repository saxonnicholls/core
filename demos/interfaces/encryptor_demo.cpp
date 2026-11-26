//
//  encryptor_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/encryptor.hpp>: an encryptor chosen at run time through the virtual interface, the adapter and the handle. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/encryptor.hpp>
#include <snicholls/core/interfaces/encryptor_reference.hpp>

#include <memory>
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
std::vector<std::byte> bytes_of(std::string_view s)
{
    std::vector<std::byte> out;
    for (const char c : s)
        out.push_back(static_cast<std::byte>(c));
    return out;
}
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

int main()
{
    std::printf("snicholls-core interfaces/encryptor.hpp demo\n");
    //! [virtual]
    std::unique_ptr<encryptor_interface> chosen = std::make_unique<encryptor_adapter<insecure_test_encryptor>>(with_keys());
    encryptor_ref cipher{*chosen};   // copyable, models the concept
    const auto sealed = cipher.encrypt("k-2027", bytes_of("seed words"), {});
    //! [virtual]
    check(sealed.is_ok(), "sealed through the handle");
    check(chosen->decrypt(*sealed, {}).is_ok(), "opened through the interface");
    check(cipher.algorithm() == chosen->algorithm(), "the algorithm through both");
    return failures == 0 ? 0 : 1;
}
