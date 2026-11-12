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
#include <memory_resource>
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
/// Chunks come from a `std::pmr::memory_resource` named at construction, `std::pmr::new_delete_resource()` unless
/// another is given: an arena, a `monotonic_buffer_resource`, or a resource over a memory-mapped file (ts-moveables'
/// `mapped_memory_resource`), so the objects can live outside the heap. Only the chunks come from it; the short list
/// of chunk pointers stays on the default heap.
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
/// - **Ownership and lifetime:** the pool owns the chunks and returns each to its resource, with the size and
///   alignment it was allocated with, on destruction; the resource must outlive the pool. Neither `release_all()` nor the destructor runs the
///   destructors of live objects: destroy them first, or use a trivially destructible `T`.
/// - **Invariants:** `live() == created - destroyed`.
///
/// \errors
/// `std::bad_alloc` when a new chunk cannot be allocated (or whatever the resource's `allocate` throws); whatever `T`'s
/// constructor throws.
///
/// \examples
/// \snippet object_pool_demo.cpp pool
/// Chunks from a resource of the caller's:
/// \snippet object_pool_demo.cpp resource
///
/// \perf
/// Measured inside limit order books.
/// In core, benchmarks/containers/object_pool_benchmark.cpp compares it like for like with `new` / `delete`, after a fairness
/// check. Release, Apple Clang 17, Intel Xeon W-3245 (macOS 26), 2026-11-12, median of 5 on a loaded machine: 33.7 µs
/// for 4,096 creates and their destroys, against 596 µs with `new` / `delete`. With chunks from a
/// `std::pmr::unsynchronized_pool_resource` it is 33.4 µs: the resource is called once a chunk, so it costs nothing
/// measurable.
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
/// - Moving a pool moves its resource pointer with it; moving one pool into another frees the target's chunks to the
///   target's resource first, so two pools over different resources may be moved between freely.
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

    /// \brief An empty pool (allocates on first `create`), its chunks from `resource`.
    /// \param resource where chunks come from; it must outlive the pool. Null means `std::pmr::new_delete_resource()`.
    explicit object_pool(std::pmr::memory_resource* resource = std::pmr::new_delete_resource()) noexcept
        : resource_(resource ? resource : std::pmr::new_delete_resource())
    {
    }
    object_pool(const object_pool&) = delete;              ///< Not copyable.
    object_pool& operator=(const object_pool&) = delete;   ///< Not copyable.
    /// \brief Take over `other`'s chunks (addresses stay valid).
    /// \param other the source, left empty.
    object_pool(object_pool&& other) noexcept
        : resource_(other.resource_), chunks_(std::move(other.chunks_)), free_(std::exchange(other.free_, nullptr)),
          next_(std::exchange(other.next_, nullptr)), end_(std::exchange(other.end_, nullptr)),
          live_(std::exchange(other.live_, 0))
    {
        other.chunks_.clear();
    }
    /// \brief Take over `other`'s chunks.
    /// \param other the source, left empty.
    /// \return `*this`.
    object_pool& operator=(object_pool&& other) noexcept
    {
        if (this == &other) return *this;
        free_chunks();
        resource_ = other.resource_;
        chunks_ = std::move(other.chunks_);
        other.chunks_.clear();
        free_ = std::exchange(other.free_, nullptr);
        next_ = std::exchange(other.next_, nullptr);
        end_ = std::exchange(other.end_, nullptr);
        live_ = std::exchange(other.live_, 0);
        return *this;
    }
    /// \brief Returns every chunk to the resource (see the gotcha about live objects).
    ~object_pool() { free_chunks(); }

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
        next_ = chunks_.empty() ? nullptr : chunks_.front();
        end_ = next_ ? next_ + ChunkSize : nullptr;
        // Chained chunks after the first are reused through the free list.
        for (std::size_t c = 1; c < chunks_.size(); ++c) {
            slot* base = chunks_[c];
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
            slot* base = new_chunk();
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
    /// \brief Where the chunks come from.
    /// \return the resource (never null).
    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept { return resource_; }

private:
    slot* acquire()
    {
        if (free_) {
            slot* s = free_;
            free_ = s->next_free;
            return s;
        }
        if (next_ == end_) {
            // Own the chunk first, then point at it: if recording it threw after next_ was set, next_ would point
            // into a chunk already given back, and the next create() would hand out a dangling slot.
            slot* base = new_chunk();
            next_ = base;
            end_ = base + ChunkSize;
        }
        return next_++;
    }

    static constexpr std::size_t chunk_bytes = ChunkSize * sizeof(slot);

    // A chunk from the resource, recorded in chunks_. Strong: the list grows first, so once the chunk exists nothing
    // can throw, and a failed allocation leaves only spare capacity in the list.
    slot* new_chunk()
    {
        if (chunks_.size() == chunks_.capacity()) chunks_.reserve(chunks_.empty() ? 4 : 2 * chunks_.size());
        void* raw = resource_->allocate(chunk_bytes, alignof(slot));
        slot* base = static_cast<slot*>(raw);
        for (std::size_t i = 0; i < ChunkSize; ++i) ::new (static_cast<void*>(base + i)) slot;
        chunks_.push_back(base);   // cannot throw: capacity reserved above
        return base;
    }

    void free_chunks() noexcept
    {
        for (slot* c : chunks_) resource_->deallocate(c, chunk_bytes, alignof(slot));
        chunks_.clear();
    }

    std::pmr::memory_resource* resource_ = std::pmr::new_delete_resource();
    std::vector<slot*> chunks_;
    slot* free_ = nullptr;
    slot* next_ = nullptr;
    slot* end_ = nullptr;
    std::size_t live_ = 0;
};

} // namespace snicholls::containers

#endif /* snicholls_core_containers_object_pool_hpp */
