//
//  signature_verifier_conformance_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/interfaces/signature_verifier_conformance.hpp>: the kit run over the reference verifier.
//  Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/interfaces/signature_verifier_conformance.hpp>
#include <snicholls/core/interfaces/signature_verifier_reference.hpp>
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
    memory_signer keys = memory_signer::deterministic();
    std::string kid{keys.create({signature_algorithm::hs256, "jwt"})->key_id.view()};
    memory_signature_verifier impl{keys};
    auto& verifier() { return impl; }
    signed_sample sign(std::span<const std::byte> message)
    {
        const auto s = keys.sign({kid, message, payload_kind::message});
        return {signature_algorithm::hs256, kid, s->key_version, {}, s->bytes};
    }
    static signature_algorithm unsupported() { return signature_algorithm::eddsa; }
    void provider_fails(capability_errc code, std::string_view response) { impl.fail_next(code, response); }
};
} // namespace

int main()
{
    std::printf("snicholls-core interfaces/signature_verifier_conformance.hpp demo\n");
    //! [kit]
    const capability_conformance_report report = run_signature_verifier_conformance<probe>();
    //! [kit]
    const signed_sample sample{};
    check(sample.request({}).signature.empty(), "signed_sample::request borrows the sample");
    check(report.ok() && report.checks == 6, "the reference passes all six checks");
    return failures == 0 ? 0 : 1;
}
