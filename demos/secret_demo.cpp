//
//  secret_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/secret.hpp>: a value that prints as a mask, the key-name heuristic and URL
//  credential masking. Exits non-zero if any result is not the one documented.
//

#include <snicholls/core/secret.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <utility>

int main()
{
    int failures = 0;
    const auto check = [&](const char* what, bool ok) {
        std::cout << (ok ? "  ok   " : "  FAIL ") << what << '\n';
        failures += !ok;
    };

    //! [secret]
    snicholls::secret<std::string> key{"hunter2"};
    std::ostringstream log;
    log << "connecting with key " << key;                     // "connecting with key ********"
    const std::string& value = key.reveal();                  // the only way to the value: easy to grep for
    //! [secret]
    check("streams as the mask, never the value", log.str() == "connecting with key ********");
    check("reveal() gives the value", value == "hunter2");
    check("masked_text is eight stars", snicholls::masked_text == "********");

    snicholls::secret<std::string> copy = key;
    check("copies compare equal", copy == key);
    copy.reveal() = "other";
    check("a changed copy compares unequal", !(copy == key));
    snicholls::secret<std::string> moved = std::move(copy);
    check("moves keep the value", moved.reveal() == "other");
    check("is_secret_v sees through cv and references",
          snicholls::is_secret_v<const snicholls::secret<int>&> && !snicholls::is_secret_v<std::string>);

    //! [masking]
    const bool masked = snicholls::is_secret_key("db_password");   // PASSWORD, TOKEN, *_KEY, ... (case-insensitive)
    const std::string url = snicholls::mask_url_credentials("postgres://app:pw@db:5432/ledger");
    //! [masking]
    check("is_secret_key: db_password", masked);
    check("is_secret_key: *_KEY and KEY", snicholls::is_secret_key("SIGNING_KEY") && snicholls::is_secret_key("key"));
    check("is_secret_key: PORT and MONKEY are not", !snicholls::is_secret_key("PORT") && !snicholls::is_secret_key("MONKEY"));
    check("mask_url_credentials hides the password", url == "postgres://app:********@db:5432/ledger");
    check("no scheme, no user info, or no password: unchanged",
          snicholls::mask_url_credentials("plain") == "plain" &&
              snicholls::mask_url_credentials("redis://user@host") == "redis://user@host" &&
              snicholls::mask_url_credentials("https://host/p?a=b:c@d") == "https://host/p?a=b:c@d");

    std::cout << (failures == 0 ? "OK\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
