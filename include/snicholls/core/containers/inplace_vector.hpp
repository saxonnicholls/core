//
//  inplace_vector.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  C++26's std::inplace_vector for C++20.
//

/// \file
/// \brief `containers::inplace_vector<T, N>`: a vector of at most `N` elements stored in place, never allocating, `constexpr`.
/// \ingroup core
///
/// **Overview.** On a hot path an allocation is the cost to avoid: the legs of an order, the levels of a
/// book snapshot, the fills of one message are bounded by a number known when the program is written.
/// `inplace_vector<T, N>` holds up to `N` elements inside itself, behaves like a `std::vector` up to that
/// capacity, and is `constexpr`, so tables of variable length can be built at compile time.
///
/// \snippet containers/inplace_vector_demo.cpp inplace_vector
///
/// Past capacity, `push_back` and `emplace_back` throw `std::length_error` (C++26's `std::bad_alloc`
/// would mislead: nothing is allocated); `try_push_back` returns `false` instead and is `noexcept`.
///
/// **Relationships.** C++26's `std::inplace_vector`, whose interface this follows where it can;
/// `containers/static_map.hpp`; the other containers. Group: \ref core.
///
/// **Guarantees.** No allocation, ever. `sizeof` is `N * sizeof(T)` plus the size, rounded to alignment;
/// trivially copyable when `T` is. Every operation is `constexpr`; iterators are pointers; erasure moves
/// later elements down, keeping order.
///
/// **Errors.** `std::length_error` past capacity (`push_back`, `emplace_back`, `resize`, `insert`);
/// `std::out_of_range` from `at`.
///
/// **Performance.** `benchmarks/containers/inplace_vector_benchmark.cpp`: 16 doubles per message for 4,096
/// messages, Release, Apple clang 17, Xeon W-3245, 2026-11-07: 64.8 us, against 64.6 us for a plain array
/// written by hand and 85.8 us for a reserved and reused `std::vector`.
///
/// **Tests and demos.** `tests/containers/inplace_vector_test.cpp`, `demos/containers/inplace_vector_demo.cpp`.
///
/// **Dependencies.** `<algorithm>`, `<array>`, `<compare>`, `<concepts>`, `<cstddef>`, `<iterator>`,
/// `<stdexcept>`, `<type_traits>`, `<utility>`.
///
/// **Known issues.** `T` must be default-constructible: the unused capacity holds default-constructed
/// elements, which is what makes the container `constexpr` in C++20 (C++26's uses raw storage).
///
/// **Status.** New in 0.3.0.

#ifndef snicholls_core_containers_inplace_vector_hpp
#define snicholls_core_containers_inplace_vector_hpp

#include <algorithm>
#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace snicholls::containers {

    /// \brief A vector of at most `N` elements, stored in place.
    /// \ingroup core
    /// \tparam T The element type, default-constructible.
    /// \tparam N The capacity.
    template <std::default_initializable T, std::size_t N>
    class inplace_vector {
    public:
        using value_type = T;                       ///< The element type.
        using size_type = std::size_t;              ///< The size type.
        using difference_type = std::ptrdiff_t;     ///< The difference type.
        using reference = T&;                       ///< A reference.
        using const_reference = const T&;           ///< A const reference.
        using pointer = T*;                         ///< A pointer.
        using const_pointer = const T*;             ///< A const pointer.
        using iterator = T*;                        ///< An iterator.
        using const_iterator = const T*;            ///< A const iterator.

        /// \brief Empty.
        constexpr inplace_vector() noexcept(std::is_nothrow_default_constructible_v<T>) = default;

        /// \brief From a list of values.
        /// \param values At most `N` of them.
        /// \throws std::length_error with more than `N`.
        constexpr inplace_vector(std::initializer_list<T> values)
        {
            if (values.size() > N)
                throw std::length_error("snicholls::containers::inplace_vector: more values than capacity");
            for (const T& v : values)
                data_[size_++] = v;
        }

        /// \brief The capacity.
        /// \return `N`.
        [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }
        /// \brief The capacity, as `std::vector` names it.
        /// \return `N`.
        [[nodiscard]] static constexpr size_type max_size() noexcept { return N; }
        /// \brief The number of elements.
        /// \return It.
        [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
        /// \brief Whether there are none.
        /// \return It.
        [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
        /// \brief Whether there is no room for another.
        /// \return It.
        [[nodiscard]] constexpr bool full() const noexcept { return size_ == N; }

        /// \brief Element `i`, unchecked.
        /// \param i Below `size()`.
        /// \return It.
        [[nodiscard]] constexpr reference operator[](size_type i) noexcept { return data_[i]; }
        /// \brief Element `i`, unchecked.
        /// \param i Below `size()`.
        /// \return It.
        [[nodiscard]] constexpr const_reference operator[](size_type i) const noexcept { return data_[i]; }
        /// \brief Element `i`, checked.
        /// \param i The index.
        /// \return It.
        /// \throws std::out_of_range when `i >= size()`.
        [[nodiscard]] constexpr reference at(size_type i)
        {
            if (i >= size_)
                throw std::out_of_range("snicholls::containers::inplace_vector::at");
            return data_[i];
        }
        /// \brief Element `i`, checked.
        /// \param i The index.
        /// \return It.
        /// \throws std::out_of_range when `i >= size()`.
        [[nodiscard]] constexpr const_reference at(size_type i) const
        {
            if (i >= size_)
                throw std::out_of_range("snicholls::containers::inplace_vector::at");
            return data_[i];
        }
        /// \brief The first element.
        /// \return It.
        [[nodiscard]] constexpr reference front() noexcept { return data_[0]; }
        /// \brief The first element.
        /// \return It.
        [[nodiscard]] constexpr const_reference front() const noexcept { return data_[0]; }
        /// \brief The last element.
        /// \return It.
        [[nodiscard]] constexpr reference back() noexcept { return data_[size_ - 1]; }
        /// \brief The last element.
        /// \return It.
        [[nodiscard]] constexpr const_reference back() const noexcept { return data_[size_ - 1]; }
        /// \brief The elements, contiguous.
        /// \return A pointer to the first.
        [[nodiscard]] constexpr pointer data() noexcept { return data_.data(); }
        /// \brief The elements, contiguous.
        /// \return A pointer to the first.
        [[nodiscard]] constexpr const_pointer data() const noexcept { return data_.data(); }

        /// \brief The first element's iterator.
        /// \return It.
        [[nodiscard]] constexpr iterator begin() noexcept { return data_.data(); }
        /// \brief The first element's iterator.
        /// \return It.
        [[nodiscard]] constexpr const_iterator begin() const noexcept { return data_.data(); }
        /// \brief One past the last element.
        /// \return It.
        [[nodiscard]] constexpr iterator end() noexcept { return data_.data() + size_; }
        /// \brief One past the last element.
        /// \return It.
        [[nodiscard]] constexpr const_iterator end() const noexcept { return data_.data() + size_; }

        /// \brief Append a copy, or return false when full.
        /// \param v The value.
        /// \return Whether it was appended.
        constexpr bool try_push_back(const T& v) noexcept(std::is_nothrow_copy_assignable_v<T>)
        {
            if (size_ == N)
                return false;
            data_[size_++] = v;
            return true;
        }

        /// \brief Append a copy.
        /// \param v The value.
        /// \throws std::length_error when full.
        constexpr void push_back(const T& v)
        {
            if (!try_push_back(v))
                throw std::length_error("snicholls::containers::inplace_vector: full");
        }

        /// \brief Append a value built from `args`.
        /// \tparam Args The arguments' types.
        /// \param args The arguments.
        /// \return The new element.
        /// \throws std::length_error when full; whatever constructing a `T` throws.
        template <typename... Args>
        constexpr reference emplace_back(Args&&... args)
        {
            if (size_ == N)
                throw std::length_error("snicholls::containers::inplace_vector: full");
            data_[size_] = T(std::forward<Args>(args)...);
            return data_[size_++];
        }

        /// \brief Remove the last element.
        /// \pre Not empty.
        constexpr void pop_back() noexcept(std::is_nothrow_default_constructible_v<T> && std::is_nothrow_move_assignable_v<T>)
        {
            data_[--size_] = T{};
        }

        /// \brief Remove every element.
        constexpr void clear() noexcept(std::is_nothrow_default_constructible_v<T> && std::is_nothrow_move_assignable_v<T>)
        {
            while (size_ > 0)
                data_[--size_] = T{};
        }

        /// \brief Grow with default values or shrink to `n` elements.
        /// \param n At most `N`.
        /// \throws std::length_error when `n > N`.
        constexpr void resize(size_type n)
        {
            if (n > N)
                throw std::length_error("snicholls::containers::inplace_vector: resize past capacity");
            while (size_ > n)
                data_[--size_] = T{};
            size_ = n;
        }

        /// \brief Insert a value before `pos`, moving later elements up.
        /// \param pos The position.
        /// \param v   The value.
        /// \return The new element's position.
        /// \throws std::length_error when full.
        constexpr iterator insert(const_iterator pos, const T& v)
        {
            if (size_ == N)
                throw std::length_error("snicholls::containers::inplace_vector: full");
            const auto i = static_cast<size_type>(pos - begin());
            std::move_backward(data_.data() + i, data_.data() + size_, data_.data() + size_ + 1);
            data_[i] = v;
            ++size_;
            return data_.data() + i;
        }

        /// \brief Remove the element at `pos`, moving later elements down.
        /// \param pos The position.
        /// \return The position of the element that followed it.
        constexpr iterator erase(const_iterator pos)
        {
            const auto i = static_cast<size_type>(pos - begin());
            std::move(data_.data() + i + 1, data_.data() + size_, data_.data() + i);
            data_[--size_] = T{};
            return data_.data() + i;
        }

        /// \brief Equal when the elements are.
        /// \param a One.
        /// \param b Another.
        /// \return Whether they are.
        friend constexpr bool operator==(const inplace_vector& a, const inplace_vector& b)
        {
            return std::equal(a.begin(), a.end(), b.begin(), b.end());
        }

        /// \brief Ordered lexicographically.
        /// \param a One.
        /// \param b Another.
        /// \return The ordering.
        friend constexpr auto operator<=>(const inplace_vector& a, const inplace_vector& b)
            requires std::three_way_comparable<T>
        {
            return std::lexicographical_compare_three_way(a.begin(), a.end(), b.begin(), b.end());
        }

    private:
        std::array<T, N> data_{};
        size_type size_ = 0;
    };

    /// \brief Remove every element for which `pred` holds; returns how many.
    /// \ingroup core
    /// \tparam T    The element type.
    /// \tparam N    The capacity.
    /// \tparam Pred The predicate's type.
    /// \param v    The vector.
    /// \param pred The predicate.
    /// \return The number removed.
    template <typename T, std::size_t N, typename Pred>
    constexpr std::size_t erase_if(inplace_vector<T, N>& v, Pred pred)
    {
        const auto kept = std::remove_if(v.begin(), v.end(), pred);
        const auto removed = static_cast<std::size_t>(v.end() - kept);
        v.resize(v.size() - removed);
        return removed;
    }

} // namespace snicholls::containers

#endif /* snicholls_core_containers_inplace_vector_hpp */
