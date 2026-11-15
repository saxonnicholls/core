//
//  error_sink.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  The conformance suite of snicholls::error_sink (include/snicholls/core/traced.hpp): a typed GoogleTest suite that
//  every sink registers in, core's text_error_sink and a sink from another library (super-log's) alike. Test-only:
//  it is not part of the library, includes GoogleTest, and is exported by CMake as the INTERFACE target
//  snicholls::core_conformance (include directory tests/support), never installed.
//
//  A probe P is default constructible and has:
//      snicholls::error_sink& sink();              the sink under test
//      std::vector<std::string> written() const;   one entry per report written, in the sink's own format
//
//  Use, in exactly one translation unit of a test binary that links GTest, at namespace scope:
//
//      #include <snicholls/core/conformance/error_sink.hpp>
//
//      SNICHOLLS_ERROR_SINK_CONFORMANCE(SuperLog, my_probe);
//
//  The checks: a traced error is written once with its whole message (context included); a plain error with its
//  text; a success writes nothing; a field that hides behind reveal() is never written; concurrent writers are all
//  written.
//

#ifndef snicholls_core_conformance_error_sink_hpp
#define snicholls_core_conformance_error_sink_hpp

#include <snicholls/core/traced.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace snicholls::conformance {

/// A value that hides itself behind reveal(), like a credential type.
struct contract_secret
{
    std::string value;
    [[nodiscard]] const std::string& reveal() const noexcept { return value; }
};

enum class contract_error
{
    disk_full = 28,
};

inline snicholls::result<int, snicholls::traced<contract_error>> contract_failing_load()
{
    return snicholls::traced_err(contract_error::disk_full);
}

template <class Probe>
class ErrorSinkContract : public testing::Test
{
protected:
    Probe probe;
};

TYPED_TEST_SUITE_P(ErrorSinkContract);

TYPED_TEST_P(ErrorSinkContract, ATracedErrorIsWrittenOnceWithItsWholeMessage)
{
    auto r = contract_failing_load().context("saving book {}", 7).log_err(this->probe.sink(), snicholls::error_level::error);
    EXPECT_TRUE(r.is_err());
    const auto lines = this->probe.written();
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_NE(lines[0].find("saving book 7: error 28"), std::string::npos) << lines[0];
}

TYPED_TEST_P(ErrorSinkContract, APlainErrorIsWrittenWithItsText)
{
    snicholls::result<int, std::string> r = snicholls::err(std::string("socket closed"));
    (void)r.log_err(this->probe.sink());
    const auto lines = this->probe.written();
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_NE(lines[0].find("socket closed"), std::string::npos) << lines[0];
}

TYPED_TEST_P(ErrorSinkContract, ASuccessWritesNothing)
{
    snicholls::result<int, std::string> r = snicholls::ok(1);
    (void)r.log_err(this->probe.sink(), snicholls::error_level::critical);
    EXPECT_TRUE(this->probe.written().empty());
}

TYPED_TEST_P(ErrorSinkContract, ASecretFieldIsNeverWritten)
{
    (void)contract_failing_load()
        .with_field("credential", contract_secret{"hunter2"})
        .log_err(this->probe.sink(), snicholls::error_level::error);
    const auto lines = this->probe.written();
    ASSERT_EQ(lines.size(), 1u);
    EXPECT_EQ(lines[0].find("hunter2"), std::string::npos) << lines[0];
    EXPECT_NE(lines[0].find(snicholls::masked_field_text), std::string::npos) << lines[0];
}

TYPED_TEST_P(ErrorSinkContract, ConcurrentWritersAreAllWritten)
{
    constexpr int threads = 4;
    constexpr int per_thread = 50;
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&] {
            for (int i = 0; i < per_thread; ++i)
                (void)contract_failing_load().log_err(this->probe.sink(), snicholls::error_level::warning);
        });
    for (auto& t : pool)
        t.join();
    EXPECT_EQ(this->probe.written().size(), static_cast<std::size_t>(threads * per_thread));
}

REGISTER_TYPED_TEST_SUITE_P(ErrorSinkContract, ATracedErrorIsWrittenOnceWithItsWholeMessage,
                            APlainErrorIsWrittenWithItsText, ASuccessWritesNothing, ASecretFieldIsNeverWritten,
                            ConcurrentWritersAreAllWritten);

} // namespace snicholls::conformance

/// Register the suite: `SNICHOLLS_ERROR_SINK_CONFORMANCE(Prefix, probes...)`, at namespace scope.
#define SNICHOLLS_ERROR_SINK_CONFORMANCE(prefix, ...)                                                                  \
    namespace snicholls::conformance {                                                                                 \
        using prefix##_ErrorSinkProbes = ::testing::Types<__VA_ARGS__>;                                                \
        INSTANTIATE_TYPED_TEST_SUITE_P(prefix, ErrorSinkContract, prefix##_ErrorSinkProbes);                           \
    }                                                                                                                  \
    static_assert(true, "")

#endif /* snicholls_core_conformance_error_sink_hpp */
