//
//  offset_ptr.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief A pointer stored as the distance from its own address, so a structure stays linked when its region moves.
/// \ingroup core

#ifndef snicholls_core_containers_offset_ptr_hpp
#define snicholls_core_containers_offset_ptr_hpp

#include <cstddef>
#include <cstdint>

namespace snicholls::containers {

/// \brief A pointer to `T` held as the byte offset from the `offset_ptr` itself to its target.
/// \ingroup core
///
/// \overview
/// A linked structure (a list, a tree, a hash table with chains) built inside a memory-mapped file breaks when the
/// file is mapped at another address: every stored pointer still names the old one. An `offset_ptr` stores
/// `target - this` instead, which stays correct when the whole region, pointer and target together, is mapped,
/// copied or shared at any address. Dereferencing costs one addition.
///
/// Use it for links inside one region that moves as a whole. Not for a pointer from outside the region into it (the
/// offset would change whenever the region moved), and not where an index into an array will do: `index_pool`'s
/// indices are half the size and trivially copyable.
///
/// \relations
/// - `index_pool`, the index-addressed alternative for fixed-size records.
/// - ts-moveables' `mapped_memory_resource`, which allocates the nodes such a structure links.
/// - Module: \ref core.
///
/// \impls
/// None.
///
/// \contracts
/// - **Thread safety:** as a raw pointer.
/// - **Exception safety:** everything is `noexcept`.
/// - **Complexity:** O(1); `get()` is one addition and one comparison.
/// - **Ownership and lifetime:** non-owning, as `T*`.
/// - **Invariants:** `get()` returns what was last assigned, as long as the `offset_ptr` and its target have moved
///   together (or not at all).
///
/// \errors
/// None.
///
/// \examples
/// \snippet offset_ptr_demo.cpp list
///
/// \perf
/// benchmarks/containers/offset_ptr_benchmark.cpp walks a linked list through `offset_ptr` against the same list
/// through `T*`, after a fairness check: 4,096 nodes in shuffled order. Release, Apple Clang 17, Intel Xeon W-3245
/// (macOS 26), 2026-11-12, median of 5 on a loaded machine: 16.4 µs against 13.1 µs, about 0.8 ns more per hop, the
/// null test and the addition on the critical path of a pointer chase.
///
/// \testing
/// - Tests: tests/containers/offset_ptr_test.cpp.
/// - Demo: demos/containers/offset_ptr_demo.cpp.
/// - Benchmark: benchmarks/containers/offset_ptr_benchmark.cpp.
///
/// \deps
/// Standard library only.
///
/// \gotchas
/// - Not trivially copyable: copying one recomputes the offset for its new address, which is what makes `a = b`
///   work. A byte-for-byte copy of a *whole region* keeps every link; a byte-for-byte copy of one object to
///   somewhere else does not.
/// - Null is stored as offset 1 (a one-byte offset cannot name an aligned `T` other than a `char`), so an
///   `offset_ptr<char>` cannot point at the byte just past itself; 0 means "points at itself", which is allowed.
/// - Not `constexpr`: the offset is computed from integer addresses.
/// - Both addresses must be in one object or one mapping for the distance to mean anything, which is how it is used.
///
/// \status
/// Experimental.
///
/// \tparam T the pointee type.
template <class T>
class offset_ptr
{
    static constexpr std::ptrdiff_t null_offset = 1;

public:
    using element_type = T;   ///< The pointee type.

    /// \brief Null.
    offset_ptr() noexcept = default;
    /// \brief Null.
    offset_ptr(std::nullptr_t) noexcept {}
    /// \brief Point at `p`.
    /// \param p the target, or null.
    offset_ptr(T* p) noexcept : off_(offset_to(p)) {}   // NOLINT: implicit, as a raw pointer converts
    /// \brief Point at what `o` points at.
    /// \param o the source.
    offset_ptr(const offset_ptr& o) noexcept : off_(offset_to(o.get())) {}
    /// \brief Point at what `o` points at.
    /// \param o the source.
    /// \return `*this`.
    offset_ptr& operator=(const offset_ptr& o) noexcept
    {
        off_ = offset_to(o.get());
        return *this;
    }
    /// \brief Point at `p`.
    /// \param p the target, or null.
    /// \return `*this`.
    offset_ptr& operator=(T* p) noexcept
    {
        off_ = offset_to(p);
        return *this;
    }
    ~offset_ptr() = default;   ///< Trivial.

    /// \brief The target.
    /// \return the address, or null.
    [[nodiscard]] T* get() const noexcept
    {
        if (off_ == null_offset) return nullptr;
        return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(this) + static_cast<std::uintptr_t>(off_));
    }
    /// \brief The target.
    /// \return `*get()`.
    [[nodiscard]] T& operator*() const noexcept { return *get(); }
    /// \brief The target.
    /// \return `get()`.
    [[nodiscard]] T* operator->() const noexcept { return get(); }
    /// \brief Whether it points anywhere.
    /// \return `get() != nullptr`.
    [[nodiscard]] explicit operator bool() const noexcept { return off_ != null_offset; }
    /// \brief The stored offset (1 for null).
    /// \return bytes from this object to its target.
    [[nodiscard]] std::ptrdiff_t offset() const noexcept { return off_; }

    /// \brief Same target.
    friend bool operator==(const offset_ptr& a, const offset_ptr& b) noexcept { return a.get() == b.get(); }
    /// \brief Same target.
    friend bool operator==(const offset_ptr& a, const T* b) noexcept { return a.get() == b; }
    /// \brief Null.
    friend bool operator==(const offset_ptr& a, std::nullptr_t) noexcept { return !a; }

private:
    std::ptrdiff_t offset_to(const T* p) const noexcept
    {
        if (p == nullptr) return null_offset;
        return static_cast<std::ptrdiff_t>(reinterpret_cast<std::uintptr_t>(p) - reinterpret_cast<std::uintptr_t>(this));
    }

    std::ptrdiff_t off_ = null_offset;
};

} // namespace snicholls::containers

#endif /* snicholls_core_containers_offset_ptr_hpp */
