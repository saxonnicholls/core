//
//  time_fixtures.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Reads the portability fixtures in tests/fixtures/time/ (written by
//  scripts/generate_time_fixtures.py from Python's datetime and exact integers,
//  never from core). The same files are checked by the same tests on every
//  platform, so a platform that disagrees with them - or with another
//  platform - fails.
//

#ifndef snicholls_core_tests_support_time_fixtures_hpp
#define snicholls_core_tests_support_time_fixtures_hpp

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef SNICHOLLS_CORE_FIXTURE_DIR
#error "SNICHOLLS_CORE_FIXTURE_DIR must name tests/fixtures (set by tests/CMakeLists.txt)"
#endif

namespace snicholls::test {

/// One fixture line, split on tabs.
struct fixture_row
{
    std::size_t line = 0;
    std::vector<std::string> fields;
};

/// Every case of tests/fixtures/time/<name>, skipping '#' comments. Throws if the file is missing or empty, so a
/// fixture that is not found can never pass by testing nothing.
inline std::vector<fixture_row> read_time_fixture(const std::string& name)
{
    const std::string path = std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/time/" + name;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("time fixture not found: " + path);
    std::vector<fixture_row> rows;
    std::string text;
    std::size_t line = 0;
    while (std::getline(in, text)) {
        ++line;
        if (!text.empty() && text.back() == '\r') text.pop_back();   // a checkout with CRLF line endings
        if (text.empty() || text.front() == '#') continue;
        fixture_row row{line, {}};
        std::size_t start = 0;
        for (;;) {
            const std::size_t tab = text.find('\t', start);
            row.fields.push_back(text.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
            if (tab == std::string::npos) break;
            start = tab + 1;
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::runtime_error("time fixture is empty: " + path);
    return rows;
}

/// A signed 64-bit integer field (std::stoll is `long long`, 64 bits on every supported platform).
inline std::int64_t to_i64(const std::string& s) { return static_cast<std::int64_t>(std::stoll(s)); }

/// The fixture spelling of an error code.
inline std::string errc_name(std::errc ec)
{
    if (ec == std::errc{}) return "ok";
    if (ec == std::errc::invalid_argument) return "invalid_argument";
    if (ec == std::errc::result_out_of_range) return "result_out_of_range";
    if (ec == std::errc::value_too_large) return "value_too_large";
    return "errc " + std::to_string(static_cast<int>(ec));
}

} // namespace snicholls::test

#endif /* snicholls_core_tests_support_time_fixtures_hpp */
