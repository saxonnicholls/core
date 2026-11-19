//
//  recurrence_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/recurrence.hpp>: an exponential moving average written as a recurrence (a step and
//  its backward pass), run over two sequences with checkpoints every three tokens, differentiated through time from
//  them, and decoded one token at a time. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/recurrence.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <span>

namespace {

//! [recurrence]
// s <- a s + x_t, y_t = s: the state is one number; a is a parameter
struct moving_average
{
    using value_type = double;
    static constexpr std::size_t state_size = 1, work_size = 0;
    std::span<const double> x{};
    double a = 0.5;
    std::span<double> y{};
    std::span<const double> dy{};   // the backward pass's views
    std::span<double> dx{};
    double* da = nullptr;

    constexpr void advance(std::size_t t, std::span<double> s) const { s[0] = a * s[0] + x[t]; }
    constexpr void step(std::size_t t, std::span<double> s) const
    {
        advance(t, s);
        y[t] = s[0];
    }
    constexpr void step_backward(std::size_t t, std::span<const double> prev, std::span<double> ds, std::span<double>) const
    {
        const double d = ds[0] + dy[t];   // the gradient in s_t: from the output and from the next state
        dx[t] = d;
        *da += d * prev[0];
        ds[0] = d * a;                    // into s_{t-1}
    }
};
//! [recurrence]

} // namespace

int main()
{
    std::printf("snicholls-core recurrence demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [recurrence_run]
    constexpr std::size_t n = 5, sequences = 2, chunk = 3;
    const std::array<double, n * sequences> x{1, 0, 0, 0, 0, 2, 2, 2, 2, 2};
    std::array<double, n * sequences> y{}, dx{};
    std::array<double, sequences> states{};   // zeros: two new sequences
    std::array<double, sequences * ml::recurrence_checkpoints(n, chunk)> checkpoints{};
    ml::recurrence_forward(moving_average{x, 0.5, y, {}, {}, nullptr}, n, sequences, chunk, std::span<double>(states),
                           std::span<double>(checkpoints));

    // The loss is the sum of every output: its gradient in each output is 1
    const std::array<double, n * sequences> ones{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double da = 0;
    std::array<double, sequences> dstates{};   // the loss does not read the final states
    std::array<double, ml::recurrence_backward_work<moving_average>(chunk)> work{};
    ml::recurrence_backward(moving_average{x, 0.5, {}, ones, dx, &da}, n, sequences, chunk, std::span<const double>(checkpoints),
                            std::span<double>(dstates), std::span<double>(work));

    // Decoding: the step itself, one token at a time with the sequence's state
    std::array<double, 1> state{};
    std::array<double, n> decoded{};
    for (std::size_t t = 0; t < n; ++t)
        moving_average{std::span<const double>(x).subspan(n, n), 0.5, decoded, {}, {}, nullptr}.step(t, state);
    //! [recurrence_run]

    std::printf("  impulse response: %.4f %.4f %.4f %.4f %.4f\n", y[0], y[1], y[2], y[3], y[4]);
    std::printf("  d(sum)/dx_0 = %.4f, d(sum)/da = %.4f\n", dx[0], da);
    ok &= y[4] == 0.0625 && dx[0] == 1 + 0.5 + 0.25 + 0.125 + 0.0625 && dx[4] == 1;
    ok &= checkpoints[1] == y[2] && states[1] == y[9];   // the state at token 3 of the first sequence; the second's final state
    ok &= decoded[4] == y[9];
    // d/da of Σ y over the impulse sequence is Σ_t t a^{t-1} = 1 + 1 + 0.75 + 0.5; the second sequence adds its own
    double expected = 0, s = 0, ds = 0;
    for (std::size_t t = 0; t < n; ++t) {   // forward-mode: ds/da = s_{t-1} + a ds_{t-1}/da
        ds = s + 0.5 * ds;
        s = 0.5 * s + x[n + t];
        expected += ds;
    }
    ok &= std::fabs(da - (1 + 1 + 0.75 + 0.5 + expected)) < 1e-12;
    std::printf(ok ? "recurrence demo: ok\n" : "recurrence demo: FAILED\n");
    return ok ? 0 : 1;
}
