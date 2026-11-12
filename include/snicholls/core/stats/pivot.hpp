//
//  pivot.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `stats::pivot`: a pivot table whose rows, columns and measures are types, filled in one pass.
/// \ingroup core
///
/// **Overview.** "P&L and mean return by currency and index" is a pivot: group the rows by two keys and
/// aggregate two values in each group. Here the shape is a type and the work is one pass over the rows:
///
/// \snippet stats/pivot_demo.cpp pivot
///
/// Keys and cells print with their display attributes (`display.hpp`): declared once beside the key types and the
/// measure's getter, the grid shows each currency in its colour and every P&L green or red through `acout`:
///
/// \snippet stats/pivot_demo.cpp pivot_attributes
///
/// - **Keys.** `Rows` and `Columns` are `meta::type_list`s. Each element is a key type, read from a row by
///   `std::get<K>(row)` (a `std::tuple` of strong types); a getter, any default-constructible type callable
///   with the row (a lambda's type, a struct reading a member); a name, `by_name<"currency">`, read by
///   `get<"currency">(row)` as an object model with named fields provides; or a tag struct with a static `get(row)`. A key that is
///   a reflectable enumeration, a `bool`, or a type with a declared `pivot_domain` has a fixed set of
///   values, so its cells are placed by arithmetic, with no hashing; any other key that can be hashed is
///   given a place when first seen.
/// - **Measures.** `measure<Value, Accumulator>`: the value, again a type read with `get` or a getter, and
///   an accumulator from `stats/accumulators.hpp` (or any type with `add` and `value`). Every measure of every
///   row is taken in the same pass. `histogram` gives each cell its distribution, `collect` keeps each cell's values
///   for any statistic afterwards, and `bootstrapped<Acc, B>` gives each cell a Poisson bootstrap of its
///   statistic in the same pass, each cell drawing its own stream.
/// - **Parallel.** Each worker fills its own pivot from its share of the rows; `merge` combines them, exactly
///   for counts, sums, minima and maxima, and to rounding for moments. Core has no threads; the caller
///   brings them.
///
/// **Relationships.** `stats/accumulators.hpp`; `meta/enums.hpp` for enumeration keys and their names;
/// `meta/type_list.hpp`. Group: \ref core.
///
/// **Guarantees.** `add` costs one lookup per key (a binary search for an enumeration, a hash for a key
/// found at run time) and one `add` per measure; no allocation unless a new key widens a dimension. Cells
/// are stored densely: their number is the product of the dimensions' sizes, so dimensions of modest size
/// (currencies, indices, sectors, books) suit it, and an identifier per row does not. A row whose
/// enumeration key is not an enumerator is counted in `dropped()` and not aggregated.
///
/// **Errors.** Compile errors with a message for a key that cannot be placed. `std::bad_alloc`.
///
/// **Performance.** `benchmarks/stats/pivot_benchmark.cpp`, P&L sum and return mean by currency and index
/// over 1,000,000 rows, against the same loop written out, after a fairness check. Release, Apple clang 17,
/// Xeon W-3245, 2026-11-07: by enumerations, 4.76 ms (210 million rows a
/// second) against 4.67 ms by hand; by 32 desk names found at run time, 36.4 ms against 33.5 ms for a
/// hand-written `std::unordered_map`.
///
/// **Tests and demos.** `tests/stats/pivot_test.cpp`, `demos/stats/pivot_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<cassert>`, `<concepts>`, `<cstddef>`, `<cstdint>`, `<functional>`,
/// `<optional>`, `<ostream>`, `<ranges>`, `<tuple>`, `<type_traits>`, `<unordered_map>`,
/// `<utility>`, `<vector>`, `meta/concepts.hpp`, `meta/enums.hpp`, `meta/type_list.hpp`,
/// `stats/accumulators.hpp`.
///
/// **Known issues.** Not thread-safe: one pivot per worker, then `merge`.
///
/// **Status.** New in 0.5.0, experimental.

#ifndef snicholls_core_stats_pivot_hpp
#define snicholls_core_stats_pivot_hpp

#include <snicholls/core/acout.hpp>
#include <snicholls/core/fixed_string.hpp>
#include <snicholls/core/hash/wyhash.hpp>
#include <snicholls/core/meta/concepts.hpp>
#include <snicholls/core/meta/enums.hpp>
#include <snicholls/core/meta/type_list.hpp>
#include <snicholls/core/stats/accumulators.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace snicholls::stats {

    /// \brief A getter that reads the element of type `T` from a row: `std::get<T>(row)`, or a `get<T>` found by argument-dependent lookup.
    /// \ingroup core
    /// \tparam T The element's type.
    template <typename T>
    struct by_type {
        /// \brief Read the element.
        /// \tparam Row The row's type.
        /// \param row The row.
        /// \return The element.
        template <typename Row>
        constexpr decltype(auto) operator()(const Row& row) const
        {
            using std::get;
            return get<T>(row);
        }
    };

    /// \brief A getter that reads by compile-time name: `get<"pnl">(row)` found by argument-dependent lookup (objects with
    /// named fields), or a `row.get<"pnl">()` member.
    /// \ingroup core
    /// \tparam Name The name.
    template <basic_fixed_string Name>
    struct by_name {
        /// \brief Read the named element, by a free `get<Name>`.
        /// \tparam Row The row's type.
        /// \param row The row.
        /// \return The element.
        template <typename Row>
            requires requires(const Row& r) { get<Name>(r); }
        constexpr decltype(auto) operator()(const Row& row) const
        {
            return get<Name>(row);
        }
        /// \brief Read the named element, by a `get<Name>()` member.
        /// \tparam Row The row's type.
        /// \param row The row.
        /// \return The element.
        template <typename Row>
            requires(!requires(const Row& r) { get<Name>(r); }) && requires(const Row& r) { r.template get<Name>(); }
        constexpr decltype(auto) operator()(const Row& row) const
        {
            return row.template get<Name>();
        }
    };

    /// \brief Declare the values a key type can take, so that its cells are placed by arithmetic, as an enumeration's are.
    ///
    /// Specialise it with `static constexpr std::array<K, N> values{...};`, in the order the pivot should show them.
    /// \ingroup core
    /// \tparam K The key type.
    template <typename K>
    struct pivot_domain;

    /// \brief A measure of a pivot: a value read from each row and the accumulator that aggregates it in each cell.
    /// \ingroup core
    /// \tparam Value       A type read with `get`, or a getter type.
    /// \tparam Accumulator The accumulator, such as `stats::sum<double>`.
    template <typename Value, typename Accumulator>
    struct measure {};

    namespace detail {

        template <typename Spec, typename Row>
        struct getter_for {
            using type = by_type<Spec>;
        };
        template <typename Spec, typename Row>
            requires std::invocable<const Spec&, const Row&>
        struct getter_for<Spec, Row> {
            using type = Spec;
        };

        // A tag struct with a static get(row): nested-struct field tags, without an object model
        template <typename Spec>
        struct static_getter {
            template <typename Row>
            constexpr decltype(auto) operator()(const Row& row) const
            {
                return Spec::get(row);
            }
        };
        template <typename Spec, typename Row>
            requires(!std::invocable<const Spec&, const Row&>) && requires(const Row& r) { Spec::get(r); }
        struct getter_for<Spec, Row> {
            using type = static_getter<Spec>;
        };

        template <typename A>
        concept reseedable = requires(A a, std::uint64_t s) { a.reseed(s); };

        template <typename A>
        constexpr void reseed_cell(A& a, std::uint64_t seed) noexcept
        {
            if constexpr (reseedable<A>)
                a.reseed(seed);
        }

        template <typename Spec, typename Row>
        using getter_t = typename getter_for<Spec, Row>::type;

        template <typename Spec, typename Row>
        using key_t = std::remove_cvref_t<std::invoke_result_t<const getter_t<Spec, Row>&, const Row&>>;

        template <typename K>
        concept declared_domain = requires { pivot_domain<K>::values; };

        // A dimension: the keys seen in one position of the pivot, and each one's index.
        template <typename K>
        class dimension {
            static_assert(std_hashable<K> && std::equality_comparable<K>,
                          "snicholls::stats::pivot: a key must be an enumeration, a bool, have a pivot_domain, or be hashable");

        public:
            static constexpr bool fixed = false;
            std::optional<std::size_t> index(const K& k) const
            {
                const auto found = index_.find(k);
                return found == index_.end() ? std::nullopt : std::optional<std::size_t>(found->second);
            }
            std::optional<std::size_t> index_or_add(const K& k)
            {
                const auto [it, added] = index_.try_emplace(k, keys_.size());
                if (added)
                    keys_.push_back(k);
                return it->second;
            }
            std::size_t size() const noexcept { return keys_.size(); }
            const K& key(std::size_t i) const noexcept { return keys_[i]; }

        private:
            std::unordered_map<K, std::size_t> index_;
            std::vector<K> keys_;
        };

        template <typename K>
            requires(reflectable_enum<K> && !declared_domain<K>)
        class dimension<K> {
        public:
            static constexpr bool fixed = true;
            static constexpr std::optional<std::size_t> index(K k) noexcept { return enum_index(k); }
            static constexpr std::optional<std::size_t> index_or_add(K k) noexcept { return enum_index(k); }
            static constexpr std::size_t size() noexcept { return enum_count_v<K>; }
            static constexpr K key(std::size_t i) noexcept { return enum_values_v<K>[i]; }
        };

        template <>
        class dimension<bool> {
        public:
            static constexpr bool fixed = true;
            static constexpr std::optional<std::size_t> index(bool k) noexcept { return k ? 1u : 0u; }
            static constexpr std::optional<std::size_t> index_or_add(bool k) noexcept { return k ? 1u : 0u; }
            static constexpr std::size_t size() noexcept { return 2; }
            static constexpr bool key(std::size_t i) noexcept { return i != 0; }
        };

        template <typename K>
            requires declared_domain<K>
        class dimension<K> {
        public:
            static constexpr bool fixed = true;
            static constexpr std::optional<std::size_t> index(const K& k) noexcept
            {
                for (std::size_t i = 0; i < pivot_domain<K>::values.size(); ++i)
                    if (pivot_domain<K>::values[i] == k)
                        return i;
                return std::nullopt;
            }
            static constexpr std::optional<std::size_t> index_or_add(const K& k) noexcept { return index(k); }
            static constexpr std::size_t size() noexcept { return pivot_domain<K>::values.size(); }
            static constexpr const K& key(std::size_t i) noexcept { return pivot_domain<K>::values[i]; }
        };

        // A cell under its measure's attributes: those of the value type or getter, on a default-constructed one,
        // so a P&L getter with sign_colours makes every P&L cell green or red. An empty optional is "-".
        template <std::default_initializable Getter>
        constexpr auto measure_attributes()
        {
            return attributes_of(Getter{});
        }
        template <typename Getter>
        constexpr attribute_set<> measure_attributes() noexcept
        {
            return {};
        }
        template <typename Getter, typename V>
        void print_cell(std::ostream& os, const V& v)
        {
            print_displayed(os, measure_attributes<Getter>(), v);
        }
        template <typename Getter, typename V>
        void print_cell(std::ostream& os, const std::optional<V>& v)
        {
            if (v)
                print_cell<Getter>(os, *v);
            else
                os << '-';
        }

    } // namespace detail

    /// \brief A pivot table of `Row`s: keys `Rows` down, `Columns` across, and each `Measure` in every cell.
    /// \ingroup core
    /// \tparam Row      The row type: a tuple of strong types, a struct, anything the getters read.
    /// \tparam Rows     A `meta::type_list` of row keys (key types or getter types).
    /// \tparam Columns  A `meta::type_list` of column keys.
    /// \tparam Measures `measure<Value, Accumulator>`s, at least one.
    template <typename Row, typename Rows, typename Columns, typename... Measures>
    class pivot;

    /// \brief The pivot table; see the file description.
    /// \ingroup core
    /// \tparam Row The row type.
    /// \tparam Rs  The row keys.
    /// \tparam Cs  The column keys.
    /// \tparam Vs  The measures' values.
    /// \tparam As  The measures' accumulators.
    template <typename Row, typename... Rs, typename... Cs, typename... Vs, typename... As>
    class pivot<Row, meta::type_list<Rs...>, meta::type_list<Cs...>, measure<Vs, As>...> {
        static_assert(sizeof...(As) > 0, "snicholls::stats::pivot: at least one measure");

        using specs = meta::type_list<Rs..., Cs...>;
        static constexpr std::size_t row_dims = sizeof...(Rs);
        static constexpr std::size_t dims = sizeof...(Rs) + sizeof...(Cs);

        template <std::size_t D>
        using spec = meta::at_t<specs, D>;

    public:
        /// \brief The keys of a cell: the row keys, then the column keys.
        using keys_type = std::tuple<detail::key_t<Rs, Row>..., detail::key_t<Cs, Row>...>;
        /// \brief A cell: one accumulator per measure.
        using cell_type = std::tuple<As...>;
        /// \brief Measure 0's getter or value type: `print` shows its cells under its attributes (`meta/attributes.hpp`).
        using first_getter = std::tuple_element_t<0, std::tuple<Vs...>>;

        /// \brief An empty pivot whose cells start as default-constructed accumulators.
        pivot() = default;

        /// \brief An empty pivot whose cells start as copies of these: for accumulators with parameters (an `ewma`'s alpha, a histogram's edges).
        /// \param prototypes One accumulator per measure.
        explicit pivot(As... prototypes) : prototype_(std::move(prototypes)...) {}

        /// \brief Aggregate one row.
        /// \param row The row.
        /// \throws std::bad_alloc when a new key widens a dimension.
        void add(const Row& row)
        {
            ++rows_;
            std::array<std::size_t, dims> at{};
            if (!place(row, at, std::make_index_sequence<dims>{})) {
                ++dropped_;
                return;
            }
            const std::size_t i = flat(at);
            if (touched_[i] == 0) {   // a new cell: give each bootstrap its own stream
                touched_[i] = 1;
                reseed(cells_[i], std::index_sequence_for<As...>{});
            }
            add_measures(cells_[i], row, std::index_sequence_for<As...>{});
        }

        /// \brief Aggregate every row of a range, in one pass.
        /// \tparam R The range's type.
        /// \param rows The rows.
        /// \throws std::bad_alloc.
        template <std::ranges::input_range R>
            requires std::convertible_to<std::ranges::range_reference_t<R>, const Row&>
        void add_all(const R& rows)
        {
            for (const Row& row : rows)
                add(row);
        }

        /// \brief Combine with another pivot of the same shape, built from other rows: as if one pivot had seen all of them.
        /// \param other The other.
        /// \throws std::bad_alloc.
        void merge(const pivot& other)
            requires(mergeable<As> && ...)
        {
            rows_ += other.rows_;
            dropped_ += other.dropped_;
            other.for_each([&](const keys_type& keys, const cell_type& cell) {
                std::array<std::size_t, dims> at{};
                place_keys(keys, at, std::make_index_sequence<dims>{});
                const std::size_t i = flat(at);
                touched_[i] = 1;
                merge_cells(cells_[i], cell, std::index_sequence_for<As...>{});
            });
        }

        /// \brief The cell at these keys, or null when no row has reached it.
        /// \param keys The row keys, then the column keys.
        /// \return The cell.
        [[nodiscard]] const cell_type* find(const keys_type& keys) const
        {
            std::array<std::size_t, dims> at{};
            if (!find_keys(keys, at, std::make_index_sequence<dims>{}))
                return nullptr;
            const std::size_t i = flat(at);
            return touched_[i] != 0 ? &cells_[i] : nullptr;
        }

        /// \brief Measure `M`'s value at these keys, or none when no row has reached the cell.
        /// \tparam M The measure's position.
        /// \param keys The row keys, then the column keys.
        /// \return The value.
        template <std::size_t M>
        [[nodiscard]] auto value(const keys_type& keys) const
        {
            const cell_type* c = find(keys);
            using V = std::remove_cvref_t<decltype(std::get<M>(*c).value())>;
            return c != nullptr ? std::optional<V>(std::get<M>(*c).value()) : std::nullopt;
        }

        /// \brief Call `f(keys, cell)` for every cell some row reached, rows-major in each dimension's order.
        /// \tparam F The callable's type.
        /// \param f Called as `f(const keys_type&, const cell_type&)`.
        template <typename F>
        void for_each(F&& f) const
        {
            const auto caps = capacity_;
            for (std::size_t i = 0; i < cells_.size(); ++i) {
                if (touched_[i] == 0)
                    continue;
                std::array<std::size_t, dims> at{};
                std::size_t rest = i;
                for (std::size_t d = dims; d-- > 0;) {
                    at[d] = rest % caps[d];
                    rest /= caps[d];
                }
                f(keys_at(at, std::make_index_sequence<dims>{}), cells_[i]);
            }
        }

        /// \brief Every cell merged: the grand total of each measure.
        /// \return The total.
        [[nodiscard]] cell_type total() const
            requires(mergeable<As> && ...)
        {
            cell_type out = prototype_;
            for_each([&](const keys_type&, const cell_type& c) { merge_cells(out, c, std::index_sequence_for<As...>{}); });
            return out;
        }

        /// \brief The total of every cell whose row keys are these: the right-hand margin.
        /// \param row_keys The row keys.
        /// \return The total.
        [[nodiscard]] cell_type row_total(const std::tuple<detail::key_t<Rs, Row>...>& row_keys) const
            requires(mergeable<As> && ...)
        {
            cell_type out = prototype_;
            for_each([&](const keys_type& keys, const cell_type& c) {
                if (row_part(keys, std::make_index_sequence<row_dims>{}) == row_keys)
                    merge_cells(out, c, std::index_sequence_for<As...>{});
            });
            return out;
        }

        /// \brief The total of every cell whose column keys are these: the bottom margin.
        /// \param column_keys The column keys.
        /// \return The total.
        [[nodiscard]] cell_type column_total(const std::tuple<detail::key_t<Cs, Row>...>& column_keys) const
            requires(mergeable<As> && ...)
        {
            cell_type out = prototype_;
            for_each([&](const keys_type& keys, const cell_type& c) {
                if (column_part(keys, std::make_index_sequence<sizeof...(Cs)>{}) == column_keys)
                    merge_cells(out, c, std::index_sequence_for<As...>{});
            });
            return out;
        }

        /// \brief The keys of dimension `D` (rows first, then columns), in the order the pivot shows them.
        /// \tparam D The dimension.
        /// \return The keys: an enumeration's in order of value, a declared domain's in its order, others as first seen.
        /// \throws std::bad_alloc.
        template <std::size_t D>
        [[nodiscard]] std::vector<detail::key_t<spec<D>, Row>> labels() const
        {
            const auto& dim = std::get<D>(dims_);
            std::vector<detail::key_t<spec<D>, Row>> out;
            for (std::size_t i = 0; i < dim.size(); ++i)
                out.push_back(dim.key(i));
            return out;
        }

        /// \brief Rows added, including dropped ones.
        /// \return The count.
        [[nodiscard]] std::uint64_t rows() const noexcept { return rows_; }

        /// \brief Rows not aggregated because an enumeration key was not an enumerator.
        /// \return The count.
        [[nodiscard]] std::uint64_t dropped() const noexcept { return dropped_; }

        /// \brief Print measure 0 as a grid: a line per combination of row keys, a column per combination of column keys.
        ///
        /// Keys print as `acout` prints them, so an enumeration with display attributes shows its label in its colour;
        /// cells take the attributes of measure 0's getter, so a P&L getter with `sign_colours` makes them green or red.
        /// Colour is written when `os` is an `acout` message that wants it (`acout{} << table;`), and into a plain stream only
        /// under `colour_mode::always`.
        /// \param os The stream.
        void print(std::ostream& os) const
        {
            std::vector<std::tuple<detail::key_t<Rs, Row>...>> row_keys;
            std::vector<std::tuple<detail::key_t<Cs, Row>...>> column_keys;
            for_each([&](const keys_type& keys, const cell_type&) {
                const auto r = row_part(keys, std::make_index_sequence<row_dims>{});
                const auto c = column_part(keys, std::make_index_sequence<sizeof...(Cs)>{});
                if (std::find(row_keys.begin(), row_keys.end(), r) == row_keys.end())
                    row_keys.push_back(r);
                if (std::find(column_keys.begin(), column_keys.end(), c) == column_keys.end())
                    column_keys.push_back(c);
            });
            const auto print_keys = [&os](const auto& t) {
                std::apply(
                    [&os](const auto&... k) {
                        const char* separator = "";
                        ((os << separator, print_to(os, k), separator = "/"), ...);
                    },
                    t);
            };
            for (const auto& c : column_keys) {
                os << '\t';
                print_keys(c);
            }
            os << '\n';
            for (const auto& r : row_keys) {
                print_keys(r);
                for (const auto& c : column_keys) {
                    os << '\t';
                    if (const auto v = value<0>(std::tuple_cat(r, c)))
                        detail::print_cell<first_getter>(os, *v);
                }
                os << '\n';
            }
        }

    private:
        template <std::size_t... Ds>
        bool place(const Row& row, std::array<std::size_t, dims>& at, std::index_sequence<Ds...>)
        {
            bool ok = true;
            ((ok = ok && place_one<Ds>(detail::getter_t<spec<Ds>, Row>{}(row), at)), ...);
            return ok;
        }

        template <std::size_t D, typename K>
        bool place_one(const K& key, std::array<std::size_t, dims>& at)
        {
            const std::optional<std::size_t> i = std::get<D>(dims_).index_or_add(key);
            if (!i)
                return false;
            if constexpr (!fixed_dimension<D>)   // a fixed dimension never grows
                if (*i >= capacity_[D])
                    grow(D, *i + 1);
            at[D] = *i;
            return true;
        }

        template <std::size_t D>
        static constexpr bool fixed_dimension = detail::dimension<detail::key_t<spec<D>, Row>>::fixed;

        static constexpr bool all_fixed = [] {
            return []<std::size_t... Ds>(std::index_sequence<Ds...>) { return (true && ... && fixed_dimension<Ds>); }(std::make_index_sequence<dims>{});
        }();

        template <std::size_t... Ds>
        void place_keys(const keys_type& keys, std::array<std::size_t, dims>& at, std::index_sequence<Ds...>)
        {
            (static_cast<void>(place_one<Ds>(std::get<Ds>(keys), at)), ...);
        }

        template <std::size_t... Ds>
        bool find_keys(const keys_type& keys, std::array<std::size_t, dims>& at, std::index_sequence<Ds...>) const
        {
            bool ok = true;
            const auto one = [&]<std::size_t D>(std::integral_constant<std::size_t, D>) {
                const auto i = std::get<D>(dims_).index(std::get<D>(keys));
                if (!i || *i >= capacity_[D])
                    return false;
                at[D] = *i;
                return true;
            };
            ((ok = ok && one(std::integral_constant<std::size_t, Ds>{})), ...);
            return ok;
        }

        template <std::size_t... Ds>
        keys_type keys_at(const std::array<std::size_t, dims>& at, std::index_sequence<Ds...>) const
        {
            return keys_type{std::get<Ds>(dims_).key(at[Ds])...};
        }

        template <std::size_t... Is>
        static auto row_part(const keys_type& keys, std::index_sequence<Is...>)
        {
            return std::tuple<detail::key_t<Rs, Row>...>{std::get<Is>(keys)...};
        }

        template <std::size_t... Is>
        static auto column_part(const keys_type& keys, std::index_sequence<Is...>)
        {
            return std::tuple<detail::key_t<Cs, Row>...>{std::get<row_dims + Is>(keys)...};
        }

        template <std::size_t... Ms>
        static void add_measures(cell_type& cell, const Row& row, std::index_sequence<Ms...>)
        {
            (std::get<Ms>(cell).add(detail::getter_t<Vs, Row>{}(row)), ...);
        }

        template <std::size_t... Ms>
        void reseed(cell_type& cell, std::index_sequence<Ms...>) noexcept
        {
            ++cells_used_;
            (detail::reseed_cell(std::get<Ms>(cell), splitmix64(cells_used_ * 0x9e3779b97f4a7c15ull + Ms)), ...);
        }

        template <std::size_t... Ms>
        static void merge_cells(cell_type& into, const cell_type& from, std::index_sequence<Ms...>)
        {
            (std::get<Ms>(into).merge(std::get<Ms>(from)), ...);
        }

        // Row-major position; with every dimension fixed the strides are constants.
        std::size_t flat(const std::array<std::size_t, dims>& at) const noexcept
        {
            const std::array<std::size_t, dims>& caps = all_fixed ? fixed_capacity : capacity_;
            std::size_t i = 0;
            for (std::size_t d = 0; d < dims; ++d)
                i = i * caps[d] + at[d];
            return i;
        }

        // Widen dimension d to hold `needed` keys, doubling, and move every cell to its new place.
        void grow(std::size_t d, std::size_t needed)
        {
            std::array<std::size_t, dims> caps = capacity_;
            std::size_t wanted = caps[d] == 0 ? 4 : caps[d];
            while (wanted < needed)
                wanted *= 2;
            caps[d] = wanted;
            std::size_t total = 1;
            for (const std::size_t c : caps)
                total *= c;
            std::vector<cell_type> cells(total, prototype_);
            std::vector<std::uint8_t> touched(total, 0);
            for (std::size_t i = 0; i < cells_.size(); ++i) {
                if (touched_[i] == 0)
                    continue;
                std::size_t rest = i, j = 0, scale = 1;
                std::array<std::size_t, dims> at{};
                for (std::size_t k = dims; k-- > 0;) {
                    at[k] = rest % capacity_[k];
                    rest /= capacity_[k];
                }
                for (std::size_t k = dims; k-- > 0;) {
                    j += at[k] * scale;
                    scale *= caps[k];
                }
                cells[j] = std::move(cells_[i]);
                touched[j] = 1;
            }
            capacity_ = caps;
            cells_ = std::move(cells);
            touched_ = std::move(touched);
        }

        static constexpr std::array<std::size_t, dims> initial_capacity() noexcept
        {
            return initial_capacity_of(std::make_index_sequence<dims>{});
        }
        template <std::size_t... Ds>
        static constexpr std::array<std::size_t, dims> initial_capacity_of(std::index_sequence<Ds...>) noexcept
        {
            return {(detail::dimension<detail::key_t<spec<Ds>, Row>>::fixed ? fixed_size<Ds>() : std::size_t{0})...};
        }
        template <std::size_t D>
        static constexpr std::size_t fixed_size() noexcept
        {
            if constexpr (detail::dimension<detail::key_t<spec<D>, Row>>::fixed)
                return detail::dimension<detail::key_t<spec<D>, Row>>::size();
            else
                return 0;
        }
        static constexpr std::array<std::size_t, dims> fixed_capacity = initial_capacity();

        static std::size_t product(const std::array<std::size_t, dims>& caps) noexcept
        {
            std::size_t total = 1;
            for (const std::size_t c : caps)
                total *= c;
            return total;
        }

        std::tuple<detail::dimension<detail::key_t<Rs, Row>>..., detail::dimension<detail::key_t<Cs, Row>>...> dims_{};
        cell_type prototype_{};
        std::array<std::size_t, dims> capacity_ = initial_capacity();
        std::vector<cell_type> cells_ = std::vector<cell_type>(product(capacity_), prototype_);
        std::vector<std::uint8_t> touched_ = std::vector<std::uint8_t>(product(capacity_), 0);
        std::uint64_t rows_ = 0;
        std::uint64_t dropped_ = 0;
        std::uint64_t cells_used_ = 0;
    };

    /// \brief A pivot of `rows` in one pass: `make_pivot<trade, rows, columns, measures...>(trades)`.
    /// \ingroup core
    /// \tparam Row      The row type.
    /// \tparam Rows     The row keys.
    /// \tparam Columns  The column keys.
    /// \tparam Measures The measures.
    /// \tparam R        The range's type.
    /// \param rows The rows.
    /// \return The pivot.
    /// \throws std::bad_alloc.
    template <typename Row, typename Rows, typename Columns, typename... Measures, std::ranges::input_range R>
    [[nodiscard]] pivot<Row, Rows, Columns, Measures...> make_pivot(const R& rows)
    {
        pivot<Row, Rows, Columns, Measures...> p;
        p.add_all(rows);
        return p;
    }

} // namespace snicholls::stats

#endif /* snicholls_core_stats_pivot_hpp */
