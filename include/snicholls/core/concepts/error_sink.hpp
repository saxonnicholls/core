//
//  error_sink.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::error_sink_like`: anything that receives a structured error report at a level.
/// \ingroup core
///
/// **Overview.** Where logged errors go. A sink receives one `snicholls::error_report` at an `snicholls::error_level`
/// through `write(level, report)`. This is the concept half of a pair; the pure virtual half is
/// `snicholls::error_sink` (`interfaces/error_sink.hpp`), the hook that `result::log_err` calls. A sink chosen at
/// compile time (a test probe, a sink for one program) models the concept and needs no virtual call; one chosen at
/// run time derives from the interface or is wrapped in `snicholls::error_sink_model`.
///
/// \snippet concepts/error_sink_demo.cpp concept
///
/// **Relationships.** `interfaces/error_sink.hpp` (the virtual twin, `text_error_sink`, the adapter
/// `error_sink_model`), `traced.hpp` (`error_report`, `error_level`, `make_error_report`), `result.hpp`
/// (`log_err`). Group: \ref core.
///
/// **Implementations.** `snicholls::error_sink` and everything derived from it (`text_error_sink`, a super-log
/// adapter in the consuming library); any type with the member.
///
/// **Guarantees.** A concept: no code, no cost. It names `error_level` and `error_report` by declaration only, so it
/// costs no include; a caller that writes a report includes `traced.hpp`.
///
/// **Errors.** None.
///
/// **Performance.** Not applicable.
///
/// **Tests and demos.** `tests/concepts/error_sink_test.cpp`, `demos/concepts/error_sink_demo.cpp`.
///
/// **Dependencies.** `<cstdint>`.
///
/// **Known issues.** None.
///
/// **Status.** New in 0.23.0.

#ifndef snicholls_core_concepts_error_sink_hpp
#define snicholls_core_concepts_error_sink_hpp

#include <cstdint>

namespace snicholls {

    enum class error_level : std::uint8_t;
    struct error_report;

    /// \brief A type whose objects accept `write(level, report)` with an `error_level` and a const `error_report`.
    template <typename S>
    concept error_sink_like = requires(S& s, error_level level, const error_report& report) {
        s.write(level, report);
    };

} // namespace snicholls

#endif // snicholls_core_concepts_error_sink_hpp
