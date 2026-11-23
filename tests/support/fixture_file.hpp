//
//  fixture_file.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Reads an oracle fixture under tests/fixtures/ (tab-separated, '#' comments), written by a script in scripts/
//  from an independent implementation, never from core. Throws if the file is missing or empty, so a fixture that
//  is not found can never pass by testing nothing.
//

#ifndef snicholls_core_tests_support_fixture_file_hpp
#define snicholls_core_tests_support_fixture_file_hpp

#include <chrono>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SNICHOLLS_CORE_FIXTURE_DIR
#error "SNICHOLLS_CORE_FIXTURE_DIR must name tests/fixtures (set by tests/CMakeLists.txt)"
#endif

namespace snicholls::test {

/// One fixture line, split on tabs, with its line number for the failure message.
struct fixture_line
{
    std::size_t line = 0;
    std::vector<std::string> fields;
};

/// Every line of tests/fixtures/<path>, skipping '#' comments and empty lines.
inline std::vector<fixture_line> read_fixture(const std::string& path)
{
    const std::string full = std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/" + path;
    std::ifstream in(full, std::ios::binary);
    if (!in) throw std::runtime_error("fixture not found: " + full);
    std::vector<fixture_line> rows;
    std::string text;
    std::size_t line = 0;
    while (std::getline(in, text)) {
        ++line;
        if (!text.empty() && text.back() == '\r') text.pop_back();   // a checkout with CRLF line endings
        if (text.empty() || text.front() == '#') continue;
        fixture_line row{line, {}};
        std::size_t start = 0;
        for (;;) {
            const std::size_t tab = text.find('\t', start);
            row.fields.push_back(text.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
            if (tab == std::string::npos) break;
            start = tab + 1;
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty()) throw std::runtime_error("fixture is empty: " + full);
    return rows;
}

/// A `YYYY-MM-DD` field as a day.
inline std::chrono::sys_days fixture_date(const std::string& s)
{
    if (s.size() != 10 || s[4] != '-' || s[7] != '-') throw std::runtime_error("not a YYYY-MM-DD date: " + s);
    const std::chrono::year_month_day ymd{std::chrono::year{std::stoi(s.substr(0, 4))},
                                          std::chrono::month{static_cast<unsigned>(std::stoi(s.substr(5, 2)))},
                                          std::chrono::day{static_cast<unsigned>(std::stoi(s.substr(8, 2)))}};
    if (!ymd.ok()) throw std::runtime_error("not a valid date: " + s);
    return std::chrono::sys_days{ymd};
}

} // namespace snicholls::test

#endif /* snicholls_core_tests_support_fixture_file_hpp */
