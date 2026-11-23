//
//  signer_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/signer_conformance.hpp>: the kit run over the reference signer. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/signer_conformance.hpp>
#include <snicholls/core/interfaces/signer_reference.hpp>

#include <cstddef>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

using namespace snicholls;

namespace {
int failures = 0;
void check(bool ok, const char* what)
{
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok)
        ++failures;
}

struct probe {
    memory_signer impl = memory_signer::deterministic();
    auto& signer() { return impl; }
    std::string active_key() { return std::string(impl.create({signature_algorithm::hs256, "jwt"})->key_id.view()); }
    std::string retired_key()
    {
        std::string id = active_key();
        static_cast<void>(impl.retire(id));
        return id;
    }
    bool verify(const signature& s, std::span<const std::byte> payload) { return impl.verify(s, payload); }
    bool deterministic() { return true; }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/signer_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_signer_conformance<probe>();
    //! [kit]
    check(report.ok() && report.checks == 7, "the reference passes all seven checks");
    return failures == 0 ? 0 : 1;
}
