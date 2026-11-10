//
//  stages.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief snicholls::pipeline stages - compile-time composable observer chains.
/// \ingroup core
///
/// snicholls::pipeline stages - compile-time composable observer chains. Std only.
///
/// \code
///   auto chain = filter([](const quote& q) { return q.bid > 0; })
///              | transform([](const quote& q) { return mid_of(q); })
///              | tap([&](double m) { last = m; })
///              | sink;                                     // any callable
///   chain(q);                                              // all inlined: no allocation, no virtual call
/// \endcode
///
/// A *stage spec* (filter, transform, tap, batch\<T>) describes a step; `spec | spec` composes specs, and
/// `spec | callable` binds the chain to its final sink, giving a plain callable struct whose members are the specs'
/// functions and the sink. Every call is a direct, inlinable call (the zero-cost form; benchmarked against a hand-written
/// loop). For run-time plug-and-play, connect any chain to a run-time emitter or observer in the consuming library.

#ifndef snicholls_core_pipeline_stages_hpp
#define snicholls_core_pipeline_stages_hpp

#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls::pipeline {

template <typename T>
concept stage_spec = requires { typename std::remove_cvref_t<T>::is_stage_spec; };

// --- bound stages: callables ---------------------------------------------------------------------------------------

/// \brief snicholls::pipeline stages - compile-time composable observer chains.
/// \ingroup core
///
/// \overview
/// The header's description above gives the usage and the rules in full.
///
/// \relations
/// - See the file description above for the related types and functions.
/// - Group: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// Stated in the file description above where they apply (thread safety, exceptions, lifetime); otherwise the usual value-type rules: no internal synchronisation.
///
/// \errors
/// See the `\throws` of the members; none otherwise.
///
/// \examples
/// \snippet stages_demo.cpp demo
///
/// \perf
/// benchmarks/pipeline/stages_benchmark.cpp, a filter | transform | sink chain against the hand-written loop over
/// 1024 quotes, fairness-checked (identical sums): medians 8.43 us (chain) and 8.55 us (hand-written), equal within
/// noise (CV 9-15%, interleaved, 10 repetitions; macOS x86-64 Xeon W-3245, Apple Clang, Release, 2026-11-02, on a
/// shared machine).
///
/// \testing
/// - Tests: tests/pipeline/stages_test.cpp.
/// - Demo: demos/pipeline/stages_demo.cpp.
///
/// \deps
/// CMake target `snicholls::core`.
///
/// \gotchas
/// None known.
///
/// \status
/// Stable, pre-1.0.
template <typename Pred, typename Next>
struct filter_stage
{
    /// \brief The pred.
    [[no_unique_address]] Pred pred;
    /// \brief The next.
    [[no_unique_address]] Next next;
    /// \brief The operator.
    /// \param v the value.
    template <typename T>
    void operator()(const T& v)
    {
        if (std::invoke(pred, v)) std::invoke(next, v);
    }
};

/// \brief The transform stage.
template <typename F, typename Next>
struct transform_stage
{
    /// \brief The f.
    [[no_unique_address]] F f;
    /// \brief The next.
    [[no_unique_address]] Next next;
    /// \brief The operator.
    /// \param v the value.
    template <typename T>
    void operator()(const T& v)
    {
        std::invoke(next, std::invoke(f, v));
    }
};

/// \brief The tap stage.
template <typename F, typename Next>
struct tap_stage
{
    /// \brief The f.
    [[no_unique_address]] F f;
    /// \brief The next.
    [[no_unique_address]] Next next;
    /// \brief The operator.
    /// \param v the value.
    template <typename T>
    void operator()(const T& v)
    {
        std::invoke(f, v);
        std::invoke(next, v);
    }
};

/// \brief Collects n values, then passes the next stage a const std::vector\<T>& (and clears).
///
/// flush() passes a partial batch.
template <typename T, typename Next>
struct batch_stage
{
    /// \brief The number of elements.
    std::size_t size;
    /// \brief The next.
    Next next;
    /// \brief The items.
    std::vector<T> items{};

    /// \brief The operator.
    /// \param v the value.
    void operator()(const T& v)
    {
        if (items.capacity() < size) items.reserve(size);
        items.push_back(v);
        if (items.size() >= size) flush();
    }
    /// \brief Flushes buffered data.
    void flush()
    {
        if (items.empty()) return;
        std::invoke(next, std::as_const(items));
        items.clear();
    }
};

// --- specs ----------------------------------------------------------------------------------------------------------

/// \brief The filter spec.
template <typename Pred>
struct filter_spec
{
    /// \brief Whether the value is stage spec.
    using is_stage_spec = void;
    /// \brief The pred.
    Pred pred;
    /// \brief The bind.
    /// \param next the next.
    /// \return the bind.
    template <typename Next>
    auto bind(Next next) const&
    {
        return filter_stage<Pred, Next>{pred, std::move(next)};
    }
};

/// \brief The transform spec.
template <typename F>
struct transform_spec
{
    /// \brief Whether the value is stage spec.
    using is_stage_spec = void;
    /// \brief The f.
    F f;
    /// \brief The bind.
    /// \param next the next.
    /// \return the bind.
    template <typename Next>
    auto bind(Next next) const&
    {
        return transform_stage<F, Next>{f, std::move(next)};
    }
};

/// \brief The tap spec.
template <typename F>
struct tap_spec
{
    /// \brief Whether the value is stage spec.
    using is_stage_spec = void;
    /// \brief The f.
    F f;
    /// \brief The bind.
    /// \param next the next.
    /// \return the bind.
    template <typename Next>
    auto bind(Next next) const&
    {
        return tap_stage<F, Next>{f, std::move(next)};
    }
};

/// \brief The batch spec.
template <typename T>
struct batch_spec
{
    /// \brief Whether the value is stage spec.
    using is_stage_spec = void;
    /// \brief The number of elements.
    std::size_t size;
    /// \brief The bind.
    /// \param next the next.
    /// \return the bind.
    template <typename Next>
    auto bind(Next next) const&
    {
        return batch_stage<T, Next>{size == 0 ? 1 : size, std::move(next)};
    }
};

/// \brief Two specs in sequence.
template <typename First, typename Second>
struct chain_spec
{
    /// \brief Whether the value is stage spec.
    using is_stage_spec = void;
    /// \brief The first.
    First first;
    /// \brief The second.
    Second second;
    /// \brief The bind.
    /// \param next the next.
    /// \return the bind.
    template <typename Next>
    auto bind(Next next) const&
    {
        return first.bind(second.bind(std::move(next)));
    }
};

/// \brief The filter.
/// \param p the pointer.
/// \return the filter.
template <typename Pred>
filter_spec<std::decay_t<Pred>> filter(Pred&& p)
{
    return {std::forward<Pred>(p)};
}
/// \brief The transform.
/// \param f the callable.
/// \return the transform.
template <typename F>
transform_spec<std::decay_t<F>> transform(F&& f)
{
    return {std::forward<F>(f)};
}
/// \brief The tap.
/// \param f the callable.
/// \return the tap.
template <typename F>
tap_spec<std::decay_t<F>> tap(F&& f)
{
    return {std::forward<F>(f)};
}
/// \brief The batch.
/// \param n the count.
/// \return the batch.
template <typename T>
batch_spec<T> batch(std::size_t n)
{
    return {n};
}

/// \brief The operator|.
/// \param a the left operand.
/// \param b the right operand.
/// \return the result.
template <stage_spec A, stage_spec B>
chain_spec<std::remove_cvref_t<A>, std::remove_cvref_t<B>> operator|(A&& a, B&& b)
{
    return {std::forward<A>(a), std::forward<B>(b)};
}

/// \brief Binds a chain to its sink.
///
/// Pass std::ref(sink) to keep a non-copyable sink by reference.
/// \ingroup core
/// \param a the left operand.
/// \param sink the sink.
/// \return the result.
template <stage_spec A, typename Sink>
    requires(!stage_spec<Sink>)
auto operator|(A&& a, Sink&& sink)
{
    return a.bind(std::forward<Sink>(sink));
}

} // namespace snicholls::pipeline

#endif /* snicholls_core_pipeline_stages_hpp */
