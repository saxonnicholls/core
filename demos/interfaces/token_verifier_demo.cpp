//
//  token_verifier_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/token_verifier.hpp>: a verifier chosen at run time, behind the virtual
//  interface. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/token_verifier.hpp>
#include <snicholls/core/interfaces/token_verifier_reference.hpp>

#include <cstdio>
#include <string>
#include <utility>

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

int main()
{
    std::printf("snicholls-core interfaces/token_verifier.hpp demo\n");
    auto reference = memory_token_verifier::deterministic();
    token_claims c;
    c.subject = uuid::from_u64(7, 7);
    const std::string token = reference.issue(c);
    //! [virtual]
    token_verifier_adapter<memory_token_verifier&> adapter{reference};   // borrows
    const token_verifier_interface& chosen = adapter;
    const token_verifier_ref handle{chosen};
    const auto who = handle.verify(token);
    //! [virtual]
    check(who.is_ok() && who->subject == c.subject, "verified through the interface and the handle");
    check(&adapter.get() == &reference, "the adapter borrows");
    token_verifier_adapter owned{std::move(reference)};
    check(owned.verify(token).is_ok(), "an owning adapter verifies what it was given");
    return failures == 0 ? 0 : 1;
}
