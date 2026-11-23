//
//  key_store_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/key_store_conformance.hpp>: the kit run over the reference key store. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/key_store_conformance.hpp>
#include <snicholls/core/interfaces/key_store_reference.hpp>

#include <cstdio>
#include <optional>
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
    memory_key_store impl = memory_key_store::deterministic();
    auto& store() { return impl; }
    key_spec supported() { return {signature_algorithm::hs256, "jwt"}; }
    std::optional<key_spec> unsupported() { return key_spec{signature_algorithm::es256, "jwt"}; }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/key_store_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_key_store_conformance<probe>();
    //! [kit]
    check(report.ok() && report.checks == 8, "the reference passes all eight checks");
    return failures == 0 ? 0 : 1;
}
