//
//  sha1_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/hash/sha1.hpp>: the WebSocket handshake (RFC 6455), the one place SHA-1 is still
//  everyday, and Git's object name. Exits non-zero if any value is not the one documented.
//

#include <snicholls/core/hash/sha1.hpp>

#include <cstdio>
#include <string>

namespace {

//! [sha1]
// RFC 6455 4.2.2: the server answers the client's key with SHA-1(key + GUID). The protocol fixes SHA-1; nothing
// here relies on collision resistance. (The answer is sent base64-encoded, by base-encode-decode.)
snicholls::digest<20> websocket_accept(const std::string& client_key)
{
    snicholls::sha1 h;
    h.update(client_key).update("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    return h.finish();
}
//! [sha1]

} // namespace

int main()
{
    std::printf("snicholls-core sha1 demo\n");
    bool ok = true;

    // RFC 6455's example: key dGhlIHNhbXBsZSBub25jZQ== is answered by s3pPLMBiTxaQ9kYGzzhZRbK+xOo= (b37a4f2c...)
    const auto accept = websocket_accept("dGhlIHNhbXBsZSBub25jZQ==");
    std::printf("  websocket accept  = %s\n", accept.hex().c_str());
    ok &= accept.hex().view() == "b37a4f2cc0624f1690f64606cf385945b2bec4ea";

    // Git names a blob by SHA-1("blob <size>\0<content>")
    const auto blob = snicholls::sha1::hash(std::string("blob 6\0hello\n", 13));
    std::printf("  git blob(hello\\n) = %s\n", blob.hex().c_str());
    ok &= blob.hex().view() == "ce013625030ba8dba906f756967f9e9ca394464a";

    static_assert(snicholls::sha1::hash("abc").hex().view() == "a9993e364706816aba3e25717850c26c9cd0d89d");
    ok &= snicholls::sha1::digest_size == 20 && snicholls::sha1::block_size == 64;

    std::printf(ok ? "sha1 demo: ok\n" : "sha1 demo: FAILED\n");
    return ok ? 0 : 1;
}
