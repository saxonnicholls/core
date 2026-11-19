//
//  signal_fixture.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Reads tests/fixtures/signal/signal_fixture.txt (written by scripts/generate_signal_fixture.py from numpy, scipy
//  and librosa, never from core): one record per line, a name and its numbers.
//

#ifndef snicholls_core_tests_support_signal_fixture_hpp
#define snicholls_core_tests_support_signal_fixture_hpp

#include <snicholls/core/signal/complex.hpp>

#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SNICHOLLS_CORE_FIXTURE_DIR
#error "SNICHOLLS_CORE_FIXTURE_DIR must name tests/fixtures (set by tests/CMakeLists.txt)"
#endif

namespace snicholls::test {

/// Every record of the signal fixture, read once. Throws if the file is missing, so a test can never pass by
/// testing nothing.
inline const std::map<std::string, std::vector<double>>& signal_records()
{
    static const std::map<std::string, std::vector<double>> records = [] {
        const std::string path = std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/signal/signal_fixture.txt";
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("signal fixture not found: " + path);
        std::map<std::string, std::vector<double>> out;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line.front() == '#')
                continue;
            std::istringstream ls(line);
            std::string name;
            ls >> name;
            std::vector<double> v;
            std::string word;
            while (ls >> word)
                v.push_back(std::stod(word));
            out.emplace(name, std::move(v));
        }
        return out;
    }();
    return records;
}

/// One record's numbers; throws if it is absent.
inline const std::vector<double>& signal_record(const std::string& name)
{
    const auto& r = signal_records();
    const auto it = r.find(name);
    if (it == r.end())
        throw std::runtime_error("signal fixture has no record " + name);
    return it->second;
}

/// A record of interleaved real and imaginary parts, as complex numbers.
template <typename T = double>
std::vector<signal::complex<T>> signal_complex(const std::string& name)
{
    const auto& v = signal_record(name);
    std::vector<signal::complex<T>> out(v.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = {static_cast<T>(v[2 * i]), static_cast<T>(v[2 * i + 1])};
    return out;
}

/// A sparse record (rows, columns, then row, column, value triples) as a dense row-major matrix.
inline std::vector<double> signal_dense(const std::string& name, std::size_t& rows, std::size_t& cols)
{
    const auto& v = signal_record(name);
    rows = static_cast<std::size_t>(v.at(0));
    cols = static_cast<std::size_t>(v.at(1));
    std::vector<double> out(rows * cols, 0.0);
    for (std::size_t i = 2; i + 2 < v.size(); i += 3)
        out[static_cast<std::size_t>(v[i]) * cols + static_cast<std::size_t>(v[i + 1])] = v[i + 2];
    return out;
}

} // namespace snicholls::test

#endif // snicholls_core_tests_support_signal_fixture_hpp
