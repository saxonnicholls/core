//
//  rwkv.hpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief `snicholls::ml::rwkv_encode`: RWKV's "world" tokenizer, greedy longest match over a trie of byte strings, and `rwkv_unescape`, which reads its vocabulary's escaped pieces.
/// \ingroup core
///
/// **Overview.** RWKV's tokenizer has no merges and no scores: from the start of the text, take the longest
/// vocabulary entry that is a prefix, and repeat. Its vocabulary holds arbitrary bytes, which GGUF stores escaped
/// (`\n`, `\t`, `\r`, `\xNN`, and a backslash before anything else taken literally); `rwkv_unescape` reads one
/// back, once, when the vocabulary is loaded. Decoding is the concatenation of the pieces (`bpe_decode` in
/// `ml/bpe.hpp`).
///
/// \snippet ml/rwkv_demo.cpp rwkv
///
/// **Relationships.** `ml/token_trie.hpp` (the vocabulary over the unescaped pieces), `ml/bpe.hpp` (`bpe_decode`).
/// Group: \ref core.
///
/// **Implementations.** One, llama.cpp's `llm_tokenizer_rwkv_session`, except that a byte no entry starts with is
/// the unknown token and the walk moves on one byte (llama.cpp would loop: its vocabulary has every byte, so it
/// never meets one).
///
/// **Guarantees.** `constexpr`, `noexcept`, no allocation. Complexity: O(n · L log V).
///
/// **Errors.** `rwkv_error::output_too_small` (fewer slots than tokens; `text.size()` always suffice),
/// `bad_escape` (a truncated `\x`).
///
/// **Performance.** Not benchmarked separately: the work is `token_trie::longest_prefix`
/// (`benchmarks/ml/token_trie_benchmark.cpp`).
///
/// **Tests and demos.** `tests/ml/rwkv_test.cpp` (greedy matches, the unknown byte, escapes, at compile time),
/// `demos/ml/rwkv_demo.cpp`. No RWKV vocabulary ships with llama.cpp's test vectors, so there is no token-for-token
/// fixture.
///
/// **Dependencies.** `<cstddef>`, `<cstdint>`, `<span>`, `<string_view>`, `snicholls/core/result.hpp`,
/// `ml/token_trie.hpp`.
///
/// **Known issues.** None known.
///
/// **Status.** Experimental, new in 0.17.0.

#ifndef snicholls_core_ml_rwkv_hpp
#define snicholls_core_ml_rwkv_hpp

#include <snicholls/core/ml/token_trie.hpp>
#include <snicholls/core/result.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace snicholls::ml
{

/// \brief Why `rwkv_encode` or `rwkv_unescape` failed.
/// \ingroup core
enum class rwkv_error
{
    output_too_small,   ///< The output is shorter than the result.
    bad_escape,         ///< A `\x` escape without two hex digits.
};

/// \brief Reads an escaped RWKV vocabulary piece back to its bytes.
/// \param escaped The piece as GGUF stores it.
/// \param out Receives the bytes: `escaped.size()` always suffice.
/// \return The bytes written, or `bad_escape` / `output_too_small`.
constexpr result<std::size_t, rwkv_error> rwkv_unescape(std::string_view escaped, std::span<char> out) noexcept
{
    const auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        return -1;
    };
    std::size_t at = 0;
    for (std::size_t i = 0; i < escaped.size(); ++i) {
        char c = escaped[i];
        if (c == '\\') {
            if (++i == escaped.size())
                break;   // a trailing backslash escapes nothing, as llama.cpp reads it
            const char e = escaped[i];
            if (e == 'x') {
                if (i + 2 >= escaped.size())
                    return err(rwkv_error::bad_escape);
                const int hi = hex(escaped[i + 1]), lo = hex(escaped[i + 2]);
                if (hi < 0 || lo < 0)
                    return err(rwkv_error::bad_escape);
                c = static_cast<char>(hi * 16 + lo);
                i += 2;
            } else {
                c = e == 't' ? '\t' : e == 'n' ? '\n' : e == 'r' ? '\r' : e;
            }
        }
        if (at == out.size())
            return err(rwkv_error::output_too_small);
        out[at++] = c;
    }
    return ok(at);
}

/// \brief Tokenizes text by greedy longest match, as RWKV's tokenizer does.
/// \param text The text.
/// \param vocab The vocabulary (unescaped pieces) as a trie.
/// \param unknown The token for a byte no entry starts with.
/// \param out Receives the tokens: `text.size()` slots always suffice.
/// \return The number of tokens written, or `output_too_small`.
constexpr result<std::size_t, rwkv_error> rwkv_encode(std::string_view text, const token_trie& vocab, std::uint32_t unknown,
                                                      std::span<std::uint32_t> out) noexcept
{
    std::size_t count = 0;
    for (std::size_t at = 0; at < text.size();) {
        std::size_t n = 0;
        std::uint32_t id = vocab.longest_prefix(text.substr(at), n);
        if (id == no_token) {
            id = unknown;
            n = 1;
        }
        if (count == out.size())
            return err(rwkv_error::output_too_small);
        out[count++] = id;
        at += n;
    }
    return ok(count);
}

} // namespace snicholls::ml

#endif // snicholls_core_ml_rwkv_hpp
