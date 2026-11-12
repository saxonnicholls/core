//
//  mapped_header.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief A fixed 64-byte file header for persisted containers: magic, version, element shape, count, capacity, checksum.
/// \ingroup core

#ifndef snicholls_core_containers_mapped_header_hpp
#define snicholls_core_containers_mapped_header_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include <snicholls/core/hash/wyhash.hpp>
#include <snicholls/core/result.hpp>

namespace snicholls::containers {

/// \brief Bytes in an encoded `mapped_header`; a container's data may start here.
inline constexpr std::size_t mapped_header_size = 64;
/// \brief The format version `encode_header` writes and `decode_header` accepts.
inline constexpr std::uint32_t mapped_header_version = 1;
/// \brief The magic number at offset 0 ("SNMAPHDR" as little-endian bytes).
inline constexpr std::uint64_t mapped_header_magic = 0x524448504d414e53ull;

/// \brief Why a header was refused.
/// \ingroup core
enum class header_error : std::uint8_t
{
    too_short,             ///< fewer than `mapped_header_size` bytes
    bad_magic,             ///< not a header (or not this format)
    bad_version,           ///< a format version this build does not read
    bad_size,              ///< the recorded header size is not `mapped_header_size`
    bad_checksum,          ///< the header's bytes do not match its checksum (torn write or corruption)
    count_over_capacity,   ///< the count exceeds the capacity
    element_mismatch,      ///< the element size or alignment differs from the reader's
    type_mismatch,         ///< the type tag differs from the reader's
};

/// \brief The description of a persisted container that leads its file: what the elements are and how many there are.
/// \ingroup core
///
/// \overview
/// A container persisted to a file (ts-moveables' `mapped_vector`, an `index_pool` in a mapped file) must refuse a
/// file that is not one of its own, was written for another element type, or was torn by a crash. The header is
/// 64 bytes at offset 0: a magic number and a format version, the element's `sizeof` and `alignof`, a type tag the
/// caller chooses, the element count and capacity, one word for the container's own use, and a wyhash checksum of
/// the 56 bytes before it. `encode_header` writes it, `decode_header` reads and checks it, and `validate_header`
/// also checks it against the reader's own element shape.
///
/// Encoded layout, every field little-endian whatever the machine:
///
/// | offset | size | field |
/// |---:|---:|---|
/// | 0 | 8 | magic `mapped_header_magic` |
/// | 8 | 4 | format version |
/// | 12 | 4 | header size, 64 |
/// | 16 | 8 | `type_tag` |
/// | 24 | 4 | `element_size` |
/// | 28 | 4 | `element_align` |
/// | 32 | 8 | `count` |
/// | 40 | 8 | `capacity` |
/// | 48 | 8 | `user` |
/// | 56 | 8 | wyhash of bytes 0-55, seeded with the magic |
///
/// The header itself is portable; the elements after it are in the writer's native layout, which is what the
/// element shape and the type tag guard.
///
/// \relations
/// - `wyhash` (hash/wyhash.hpp), the checksum.
/// - `index_pool`, a region that can follow the header.
/// - ts-moveables' `mapped_vector`, which keeps its size and capacity here.
/// - Module: \ref core.
///
/// \impls
/// None.
///
/// \contracts
/// - **Thread safety:** pure functions.
/// - **Exception safety:** everything is `noexcept`.
/// - **Complexity:** O(1): 64 bytes and one 56-byte hash.
/// - **Invariants:** a header `decode_header` accepts has `count <= capacity` and the recorded checksum.
///
/// \errors
/// `header_error`, as returned by `encode_header` (`too_short`, `count_over_capacity`), `decode_header` (every
/// structural error) and `validate_header` (those plus `element_mismatch` and `type_mismatch`).
///
/// \examples
/// \snippet mapped_header_demo.cpp header
///
/// \perf
/// benchmarks/containers/mapped_header_benchmark.cpp measures an encode and a decode against a hand-written memcpy
/// and wyhash, after a fairness check. Release, Apple Clang 17, Intel Xeon W-3245 (macOS 26), 2026-11-12, median of 5:
/// 19.9 ns per round trip against 19.3 ns by hand. The cost is the two wyhashes of 56 bytes; the explicit byte order
/// is close to free.
///
/// \testing
/// - Tests: tests/containers/mapped_header_test.cpp.
/// - Demo: demos/containers/mapped_header_demo.cpp.
/// - Benchmark: benchmarks/containers/mapped_header_benchmark.cpp.
///
/// \deps
/// `hash/wyhash.hpp`, `result.hpp`; standard library only. No I/O: reading and writing the file is the caller's.
///
/// \gotchas
/// - The checksum makes a torn header detectable, not the data: a container that must survive a crash writes its
///   data, makes it durable, and only then rewrites the header (ts-moveables' `mapped_vector::flush` does).
/// - The type tag is the caller's: a hash of a type name and a schema version (`fnv1a_64("order v2")`) is usual. Zero
///   is a valid tag, so a reader that does not care passes the tag it finds.
///
/// \status
/// Experimental.
struct mapped_header
{
    std::uint64_t type_tag = 0;        ///< The caller's tag for the element type and schema.
    std::uint32_t element_size = 0;    ///< `sizeof` of the element.
    std::uint32_t element_align = 0;   ///< `alignof` of the element.
    std::uint64_t count = 0;           ///< Elements in use.
    std::uint64_t capacity = 0;        ///< Elements the file holds.
    std::uint64_t user = 0;            ///< One word for the container's own use.

    /// \brief Field-by-field equality.
    friend constexpr bool operator==(const mapped_header&, const mapped_header&) noexcept = default;
};

/// \brief The header for elements of type `T`.
/// \ingroup core
/// \tparam T the element type.
/// \param type_tag the caller's tag.
/// \param count elements in use.
/// \param capacity elements the file holds.
/// \return the header, with `T`'s size and alignment.
template <class T>
[[nodiscard]] constexpr mapped_header mapped_header_for(std::uint64_t type_tag, std::uint64_t count = 0,
                                                        std::uint64_t capacity = 0) noexcept
{
    return mapped_header{type_tag, static_cast<std::uint32_t>(sizeof(T)), static_cast<std::uint32_t>(alignof(T)),
                         count, capacity, 0};
}

namespace detail {
    template <class U>
    constexpr void put_le(std::span<std::byte> out, std::size_t at, U v) noexcept
    {
        for (std::size_t i = 0; i < sizeof(U); ++i) out[at + i] = static_cast<std::byte>((v >> (8 * i)) & 0xff);
    }
    template <class U>
    constexpr U get_le(std::span<const std::byte> in, std::size_t at) noexcept
    {
        U v = 0;
        for (std::size_t i = 0; i < sizeof(U); ++i) v |= static_cast<U>(static_cast<U>(in[at + i]) << (8 * i));
        return v;
    }
    constexpr std::uint64_t header_checksum(std::span<const std::byte> in) noexcept
    {
        return wyhash(in.first(56), mapped_header_magic);
    }
} // namespace detail

/// \brief Write `h` into the first `mapped_header_size` bytes of `out`.
/// \ingroup core
/// \param h the header.
/// \param out at least `mapped_header_size` bytes.
/// \return `ok()`, or `too_short` / `count_over_capacity` (nothing is written then).
[[nodiscard]] constexpr result<void, header_error> encode_header(const mapped_header& h, std::span<std::byte> out) noexcept
{
    if (out.size() < mapped_header_size) return err(header_error::too_short);
    if (h.count > h.capacity) return err(header_error::count_over_capacity);
    detail::put_le(out, 0, mapped_header_magic);
    detail::put_le(out, 8, mapped_header_version);
    detail::put_le(out, 12, static_cast<std::uint32_t>(mapped_header_size));
    detail::put_le(out, 16, h.type_tag);
    detail::put_le(out, 24, h.element_size);
    detail::put_le(out, 28, h.element_align);
    detail::put_le(out, 32, h.count);
    detail::put_le(out, 40, h.capacity);
    detail::put_le(out, 48, h.user);
    detail::put_le(out, 56, detail::header_checksum(out));
    return ok();
}

/// \brief Read and check the header at the start of `in`: magic, version, size, checksum, count against capacity.
/// \ingroup core
/// \param in at least `mapped_header_size` bytes.
/// \return the header, or the first `header_error` found, in the order of the enumeration.
[[nodiscard]] constexpr result<mapped_header, header_error> decode_header(std::span<const std::byte> in) noexcept
{
    if (in.size() < mapped_header_size) return err(header_error::too_short);
    if (detail::get_le<std::uint64_t>(in, 0) != mapped_header_magic) return err(header_error::bad_magic);
    if (detail::get_le<std::uint32_t>(in, 8) != mapped_header_version) return err(header_error::bad_version);
    if (detail::get_le<std::uint32_t>(in, 12) != mapped_header_size) return err(header_error::bad_size);
    if (detail::get_le<std::uint64_t>(in, 56) != detail::header_checksum(in)) return err(header_error::bad_checksum);
    mapped_header h{detail::get_le<std::uint64_t>(in, 16), detail::get_le<std::uint32_t>(in, 24),
                    detail::get_le<std::uint32_t>(in, 28), detail::get_le<std::uint64_t>(in, 32),
                    detail::get_le<std::uint64_t>(in, 40), detail::get_le<std::uint64_t>(in, 48)};
    if (h.count > h.capacity) return err(header_error::count_over_capacity);
    return ok(h);
}

/// \brief Check the header at the start of `in` and that it describes `expected`'s elements.
/// \ingroup core
/// \param in at least `mapped_header_size` bytes.
/// \param expected the reader's element shape and type tag (its count, capacity and user word are ignored).
/// \return `ok()`, or the `decode_header` error, then `element_mismatch` or `type_mismatch`.
[[nodiscard]] constexpr result<void, header_error> validate_header(std::span<const std::byte> in,
                                                                  const mapped_header& expected) noexcept
{
    auto h = decode_header(in);
    if (h.is_err()) return err(h.unwrap_err_unchecked());
    if (h->element_size != expected.element_size || h->element_align != expected.element_align)
        return err(header_error::element_mismatch);
    if (h->type_tag != expected.type_tag) return err(header_error::type_mismatch);
    return ok();
}

} // namespace snicholls::containers

#endif /* snicholls_core_containers_mapped_header_hpp */
