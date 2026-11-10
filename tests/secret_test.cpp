//
//  secret_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/secret.hpp>.
//

#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <snicholls/core/secret.hpp>

namespace {

using snicholls::secret;

TEST(Secret, PrintsMaskNotValue)
{
    const secret<std::string> s("hunter2");
    std::ostringstream out;
    out << s;
    EXPECT_EQ(out.str(), "********");
    EXPECT_EQ(s.reveal(), "hunter2");
}

TEST(Secret, CopyMoveCompare)
{
    secret<std::string> a("k");
    secret<std::string> b = a;
    EXPECT_EQ(a, b);
    secret<std::string> c = std::move(b);
    EXPECT_EQ(c.reveal(), "k");
    c.reveal() = "z";
    EXPECT_FALSE(a == c);
    static_assert(snicholls::is_secret_v<const secret<int>&>);
    static_assert(!snicholls::is_secret_v<std::string>);
    static_assert(std::is_nothrow_move_constructible_v<secret<std::string>>);
}

TEST(Secret, KeyHeuristicMasksConventionalNames)
{
    for (const char* key : {"DB_PASSWORD", "api_key", "AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "PRIVATE_KEY_PEM",
                            "AUTH_HEADER", "SESSION_COOKIE", "KEY", "SIGNING_KEY", "passphrase", "DB_PWD"})
        EXPECT_TRUE(snicholls::is_secret_key(key)) << key;
    for (const char* key : {"PORT", "HOST", "LOG_LEVEL", "KEYBOARD_LAYOUT", "MONKEY"})
        EXPECT_FALSE(snicholls::is_secret_key(key)) << key;
}

TEST(Secret, UrlPasswordsAreMasked)
{
    EXPECT_EQ(snicholls::mask_url_credentials("postgres://user:pw@db:5432/x"), "postgres://user:********@db:5432/x");
    EXPECT_EQ(snicholls::mask_url_credentials("https://host/path?a=b:c@d"), "https://host/path?a=b:c@d");
    EXPECT_EQ(snicholls::mask_url_credentials("redis://user@host"), "redis://user@host");
    EXPECT_EQ(snicholls::mask_url_credentials("plain"), "plain");
    EXPECT_EQ(snicholls::mask_url_credentials("amqp://u:p:q@h"), "amqp://u:********@h");
}

} // namespace
