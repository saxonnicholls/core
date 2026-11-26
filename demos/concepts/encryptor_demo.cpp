//
//  encryptor_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/concepts/encryptor.hpp>: a field sealed by code written once against the concept, bound to its row by the associated data. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/concepts/encryptor.hpp>
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

namespace {
// Generic code: seal a card number under the active key, bound to the user's row.
template <encryptor E>
capability_result<std::vector<std::byte>> seal_card(E& e, std::string_view user_row, std::string_view card)
{
    return e.encrypt("k-2026", bytes_of(card), bytes_of(user_row));
}
} // namespace

int main()
{
    std::printf("snicholls-core concepts/encryptor.hpp demo\n");
    //! [concept]
    insecure_test_encryptor e = with_keys();
    const auto sealed = seal_card(e, "users/42", "4111 1111 1111 1111");
    const auto opened = e.decrypt(*sealed, bytes_of("users/42"));
    const auto moved = e.decrypt(*sealed, bytes_of("users/43"));     // pasted into another row: refused
    //! [concept]
    check(opened.is_ok() && *opened == bytes_of("4111 1111 1111 1111"), "the row opens its own field");
    check(moved.is_err() && moved.unwrap_err().code == capability_errc::signature_invalid, "another row cannot");
    check(sealed_key_id(*sealed).is_ok() && *sealed_key_id(*sealed) == "k-2026", "the envelope names its key");
    check(is_key_id("k-2026") && !is_key_id("k 2026"), "is_key_id");
    check(sealed_header("k").size() == 3, "sealed_header");
    return failures == 0 ? 0 : 1;
}
