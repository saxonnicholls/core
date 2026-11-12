//
//  object_pool.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief A chunked object pool: stable addresses, O(1) allocate and release, no per-object heap allocation.
/// \ingroup core

#ifndef snicholls_core_containers_object_pool_hpp
#define snicholls_core_containers_object_pool_hpp

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace snicholls::containers {

/// \brief A pool of `T` slots allocated in fixed-size chunks, with an intrusive free list.
/// \ingroup core
///
/// \overview
/// Hot paths that create and destroy many small objects (orders, price levels) pay for `new`/`delete` and lose
/// locality. The pool allocates `ChunkSize` slots at a time, hands them out in address order, and recycles released
/// slots last-in first-out through a free list threaded through the slots themselves (no side vector, unlike the
/// reference implementations' `ObjectPool`). Addresses are stable for the pool's lifetime, so intrusive
/// containers may link the objects.
///
/// Use it for many objects of one type whose lifetimes end in arbitrary order. Not for objects that must be
/// returned to the operating system individually (memory is released only by `release_all()` or destruction).
///
/// \relations
/// - Used by limit order books (orders, stop levels and price levels).
/// - Module: \ref core.
///
/// \impls
/// None: a single implementation.
///
/// \contracts
/// - **Thread safety:** none internal; one owner thread.
/// - **Exception safety:** `create` is strong: if a chunk allocation or `T`'s constructor throws, the pool is
///   unchanged (the slot goes back to the free list). `destroy` is `noexcept` when `T`'s destructor is.
/// - **Complexity:** `create` / `destroy` O(1); a new chunk every `ChunkSize` live objects.
/// - **Ownership and lifetime:** the pool owns the memory. Neither `release_all()` nor the destructor runs the
///   destructors of live objects: destroy them first, or use a trivially destructible `T`.
/// - **Invariants:** `live() == created - destroyed`.
///
/// \errors
/// `std::bad_alloc` when a new chunk cannot be allocated; whatever `T`'s constructor throws.
///
/// \examples
/// \snippet object_pool_demo.cpp pool
///
/// \perf
/// Measured inside limit order books.
/// In core, benchmarks/containers/object_pool_benchmark.cpp compares it like for like with `new` / `delete`, after a fairness
/// check; not yet measured in Release.
///
/// \testing
/// - Tests: tests/containers/object_pool_test.cpp.
/// - Demo: demos/containers/object_pool_demo.cpp.
/// - Benchmark: benchmarks/containers/object_pool_benchmark.cpp.
///
/// \deps
/// CMake target `snicholls::core`; standard library only.
///
/// \gotchas
/// - Destroying the pool with live non-trivially-destructible objects leaks their resources (their destructors do
///   not run); a debug assertion is deliberately absent so the pool stays usable as an arena.
/// - A double `destroy(p)` is not detected: it runs the destructor twice and puts the slot on the free list twice,
///   so two later `create` calls return the same address. `destroy` only objects this pool created, once each.
///
/// \status
/// Experimental.
///
/// \tparam T the object type.
/// \tparam ChunkSize slots per chunk.
template <class T, std::size_t ChunkSize = 4096>
class object_pool
{
    static_assert(ChunkSize > 0, "snicholls: object_pool needs a non-zero chunk size");

    union slot
    {
        slot* next_free;
        alignas(T) unsigned char storage[sizeof(T)];
    };

public:
    using value_type = T;   ///< The object type.

    /// \brief An empty pool (allocates on first `create`).
    object_pool() noexcept = default;
    object_pool(const object_pool&) = delete;              ///< Not copyable.
    object_pool& operator=(const object_pool&) = delete;   ///< Not copyable.
    /// \brief Take over `other`'s chunks (addresses stay valid).
    /// \param other the source, left empty.
    object_pool(object_pool&& other) noexcept
        : chunks_(std::move(other.chunks_)), free_(std::exchange(other.free_, nullptr)),
          next_(std::exchange(other.next_, nullptr)), end_(std::exchange(other.end_, nullptr)),
          live_(std::exchange(other.live_, 0))
    {
    }
    /// \brief Take over `other`'s chunks.
    /// \param other the source, left empty.
    /// \return `*this`.
    object_pool& operator=(object_pool&& other) noexcept
    {
        chunks_ = std::move(other.chunks_);
        free_ = std::exchange(other.free_, nullptr);
        next_ = std::exchange(other.next_, nullptr);
        end_ = std::exchange(other.end_, nullptr);
        live_ = std::exchange(other.live_, 0);
        return *this;
    }
    /// \brief Frees every chunk (see the gotcha about live objects).
    ~object_pool() = default;

    /// \brief Construct a `T` in a free slot.
    /// \param args constructor arguments.
    /// \return the object (stable address).
    /// \throws std::bad_alloc when a chunk cannot be allocated; whatever `T`'s constructor throws (pool unchanged).
    template <class... Args>
    [[nodiscard]] T* create(Args&&... args)
    {
        slot* s = acquire();
        if constexpr (std::is_nothrow_constructible_v<T, Args...>) {
            T* p = ::new (static_cast<void*>(s->storage)) T(std::forward<Args>(args)...);
            ++live_;
            return p;
        } else {
            try {
                T* p = ::new (static_cast<void*>(s->storage)) T(std::forward<Args>(args)...);
                ++live_;
                return p;
            } catch (...) {
                s->next_free = free_;
                free_ = s;
                throw;
            }
        }
    }

    /// \brief Destroy `p` and recycle its slot.
    /// \param p an object created by this pool and not yet destroyed.
    void destroy(T* p) noexcept(std::is_nothrow_destructible_v<T>)
    {
        p->~T();
        slot* s = reinterpret_cast<slot*>(p);
        s->next_free = free_;
        free_ = s;
        --live_;
    }

    /// \brief Forget every object (destructors are not run) and keep the chunks for reuse.
    void release_all() noexcept
    {
        free_ = nullptr;
        next_ = chunks_.empty() ? nullptr : chunks_.front().get();
        end_ = next_ ? next_ + ChunkSize : nullptr;
        // Chained chunks after the first are reused through the free list.
        for (std::size_t c = 1; c < chunks_.size(); ++c) {
            slot* base = chunks_[c].get();
            for (std::size_t i = ChunkSize; i-- > 0;) {
                base[i].next_free = free_;
                free_ = &base[i];
            }
        }
        live_ = 0;
    }

    /// \brief Pre-allocate chunks for at least `n` objects in total.
    /// \param n the capacity wanted.
    /// \throws std::bad_alloc on allocation failure.
    void reserve(std::size_t n)
    {
        while (capacity() < n) {
            auto chunk = std::make_unique<slot[]>(ChunkSize);
            slot* base = chunk.get();
            chunks_.push_back(std::move(chunk));
            for (std::size_t i = ChunkSize; i-- > 0;) {
                base[i].next_free = free_;
                free_ = &base[i];
            }
        }
    }

    /// \brief Live objects.
    /// \return the count.
    [[nodiscard]] std::size_t live() const noexcept { return live_; }
    /// \brief Slots allocated (live + free).
    /// \return the capacity.
    [[nodiscard]] std::size_t capacity() const noexcept { return chunks_.size() * ChunkSize; }

private:
    slot* acquire()
    {
        if (free_) {
            slot* s = free_;
            free_ = s->next_free;
            return s;
        }
        if (next_ == end_) {
            // Own the chunk first, then point at it: if push_back threw after next_ was set, next_ would point into
            // the chunk the unique_ptr just freed, and the next create() would hand out a dangling slot.
            auto chunk = std::make_unique<slot[]>(ChunkSize);
            slot* base = chunk.get();
            chunks_.push_back(std::move(chunk));
            next_ = base;
            end_ = base + ChunkSize;
        }
        return next_++;
    }

    std::vector<std::unique_ptr<slot[]>> chunks_;
    slot* free_ = nullptr;
    slot* next_ = nullptr;
    slot* end_ = nullptr;
    std::size_t live_ = 0;
};

} // namespace snicholls::containers

#endif /* snicholls_core_containers_object_pool_hpp */
