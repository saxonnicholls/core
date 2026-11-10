//
//  tabular.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief The game of a model at one row of a dataset: a feature's absence is its value in background rows.
/// \ingroup core
///
/// **Overview.** To explain why a model gave `f(x)` for one row `x`, each feature is a player. A coalition
/// keeps its features at `x`'s values and replaces the others by their values in a background row, and its
/// worth is the model's average over the background. That is the interventional game SHAP's permutation
/// explainer samples; give it to `shapley` (exact, up to about 20 features) or `shapley_sampled`.
///
/// \snippet game/tabular_demo.cpp tabular
///
/// Any model, any dataset: the instance is a random-access range of feature values; the background
/// is any range of rows like it; the model is called with a `std::span<const F>` of one hybrid row.
/// The features may be anything a row holds (`double`, an enum, a struct) and the result any
/// `game_value`.
///
/// **Relationships.** `game/shapley.hpp`; `game/change.hpp` for one point moving to another, the
/// special case of a single background row. Group: \ref core.
///
/// **Guarantees.** `constexpr`. One evaluation is one model call per background row. The game refers to
/// the model, instance and background: they must outlive it. It keeps one row buffer, so it is not
/// thread-safe; give each thread its own.
///
/// **Errors.** None of its own; the background must not be empty and each row must have as many
/// features as the instance (asserted).
///
/// **Performance.** Not benchmarked: the model calls are the cost.
///
/// **Tests and demos.** `tests/game/tabular_test.cpp`, `demos/game/tabular_demo.cpp`.
///
/// **Dependencies.** `<cassert>`, `<cstddef>`, `<functional>`, `<ranges>`, `<span>`, `<type_traits>`,
/// `<vector>`, `game/coalition.hpp`.
///
/// **Known issues.** The interventional game breaks correlations between features: a feature is
/// replaced independently of the others, as in SHAP's default. Conditional expectations need a model
/// of the data, which is the caller's to build.
///
/// **Status.** New in 0.1.0, experimental.

#ifndef snicholls_core_game_tabular_hpp
#define snicholls_core_game_tabular_hpp

#include <snicholls/core/game/coalition.hpp>

#include <cassert>
#include <cstddef>
#include <functional>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

namespace snicholls::game {

    /// \brief The interventional game of `model` at `instance` over `background`.
    /// \ingroup core
    /// \tparam Model      Callable with `std::span<const F>` of one row, returning a `game_value`.
    /// \tparam Instance   A random-access range of the features `F`.
    /// \tparam Background A range of rows, each a random-access range of values convertible to `F`.
    template <typename Model, std::ranges::random_access_range Instance, std::ranges::input_range Background>
        requires game_value<std::remove_cvref_t<std::invoke_result_t<const Model&, std::span<const std::ranges::range_value_t<Instance>>>>>
    class interventional_game {
    public:
        /// \brief The feature type.
        using feature = std::ranges::range_value_t<Instance>;
        /// \brief The model's result.
        using value = std::remove_cvref_t<std::invoke_result_t<const Model&, std::span<const feature>>>;

        /// \brief The game of `model` at `instance`.
        /// \param model      The model; it must outlive the game.
        /// \param instance   The row explained; it must outlive the game.
        /// \param background The rows a missing feature is drawn from; not empty; it must outlive the game.
        constexpr interventional_game(const Model& model, const Instance& instance, const Background& background)
            : model_(&model), instance_(&instance), background_(&background),
              row_(static_cast<std::size_t>(std::ranges::size(instance)))
        {
        }

        /// \brief The number of players: the instance's features.
        /// \return The count.
        [[nodiscard]] constexpr std::size_t players() const noexcept { return row_.size(); }

        /// \brief The model's average over the background with the features in `s` held at the instance's values.
        /// \param s The coalition.
        /// \return The value.
        /// \throws Whatever the model throws.
        constexpr value operator()(coalition s) const
        {
            value sum{};
            std::size_t count = 0;
            for (const auto& background_row : *background_) {
                assert(static_cast<std::size_t>(std::ranges::size(background_row)) == row_.size());
                for (std::size_t j = 0; j < row_.size(); ++j)
                    row_[j] = contains(s, j) ? static_cast<feature>((*instance_)[j]) : static_cast<feature>(background_row[j]);
                sum = sum + std::invoke(*model_, std::span<const feature>(row_));
                ++count;
            }
            assert(count > 0 && "snicholls::game::interventional_game: the background is empty");
            return sum * (game_scalar_t<value>(1) / static_cast<game_scalar_t<value>>(count));
        }

    private:
        const Model* model_;
        const Instance* instance_;
        const Background* background_;
        mutable std::vector<feature> row_;
    };

    /// \brief The interventional game of a model at one row: `shapley(game.players(), interventional(model, x, data))`.
    /// \ingroup core
    /// \tparam Model      The model's type.
    /// \tparam Instance   The row's type.
    /// \tparam Background The background's type.
    /// \param model      The model, called with `std::span<const F>`.
    /// \param instance   The row explained.
    /// \param background The rows a missing feature is drawn from.
    /// \return The game; it refers to all three.
    template <typename Model, std::ranges::random_access_range Instance, std::ranges::input_range Background>
    [[nodiscard]] constexpr interventional_game<Model, Instance, Background> interventional(const Model& model, const Instance& instance,
                                                                                         const Background& background)
    {
        return {model, instance, background};
    }

} // namespace snicholls::game

#endif /* snicholls_core_game_tabular_hpp */
