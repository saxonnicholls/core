//
//  error_sink_contract.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Tests for <snicholls/core/traced.hpp>.
//

// The snicholls::error_sink contract (include/snicholls/core/traced.hpp): every implementation registers here with a probe.
//
// A probe P is default constructible and has:
//   snicholls::error_sink& sink();                  the sink under test
//   std::vector<std::string> written() const;  one entry per report written, in the sink's own format
//
// Register:  using my_probes = testing::Types<my_probe>;
//            INSTANTIATE_TYPED_TEST_SUITE_P(MySink, ErrorSinkContract, my_probes);

#include <snicholls/core/traced.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace snicholls_test {

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

} // namespace snicholls_test
