//
//  http_header_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/http_header.hpp>: a header field, ASCII case-insensitive comparison and the lookup a webhook route uses. Exits non-zero if a result is not the one documented.
//

#include <snicholls/core/http_header.hpp>

#include <cstdio>
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

int main()
{
    std::printf("snicholls-core http_header.hpp demo\n");
    //! [http_header]
    const std::vector<http_header> headers{{"Content-Type", "application/json"}, {"Stripe-Signature", "t=1,v1=ab"}};
    const auto signature = find_header(headers, "stripe-signature");   // RFC 9110: names ignore case
    //! [http_header]
    check(signature && *signature == "t=1,v1=ab", "found whatever the case");
    check(!find_header(headers, "X-Missing"), "a missing header is nullopt");
    check(ascii_iequals("ETag", "etag") && !ascii_iequals("ETag", "e-tag"), "ascii_iequals");
    return failures == 0 ? 0 : 1;
}
