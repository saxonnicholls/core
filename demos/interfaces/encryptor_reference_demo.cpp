//
//  encryptor_reference_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/encryptor_reference.hpp>: the test double (NOT encryption) sealing, rotating keys and refusing a tampered envelope. Exits non-zero if a result is not the one documented.
//

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
    std::printf("snicholls-core interfaces/encryptor_reference.hpp demo\n");
    //! [reference]
    insecure_test_encryptor e = with_keys();                  // a test double: it protects nothing
    const auto old_row = *e.encrypt("k-2026", bytes_of("v1"), {});
    const auto new_row = *e.encrypt("k-2027", bytes_of("v2"), {});   // rotation: new rows on the new key
    std::vector<std::byte> tampered = new_row;
    tampered.back() ^= std::byte{1};
    //! [reference]
    check(e.decrypt(old_row, {}).is_ok() && e.decrypt(new_row, {}).is_ok(), "old and new keys both open");
    check(e.decrypt(tampered, {}).unwrap_err().code == capability_errc::signature_invalid, "a tampered envelope is refused");
    std::printf("    algorithm: %.*s\n", static_cast<int>(e.algorithm().size()), e.algorithm().data());
    check(e.algorithm().find("not encryption") != std::string_view::npos, "it says what it is");
    return failures == 0 ? 0 : 1;
}
