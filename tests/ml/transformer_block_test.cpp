//
//  transformer_block_test.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/core/ml/transformer_block.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace ml = snicholls::ml;
using W = ml::dense_weights<float>;

// The fixture: per section, the arrays the generator filled (regenerated here from the same splitmix64 stream, in the
// file's order) and the arrays it computed (float bits, or doubles in hex)
struct fixture {
    std::map<std::string, std::map<std::string, std::vector<float>>> filled;
    std::map<std::string, std::map<std::string, std::vector<float>>> floats;
    std::map<std::string, std::map<std::string, std::vector<double>>> doubles;
};

const fixture& data()
{
    static const fixture f = [] {
        fixture out;
        std::ifstream in(std::string(SNICHOLLS_CORE_FIXTURE_DIR) + "/ml/transformer_fixture.txt");
        EXPECT_TRUE(in.good()) << "missing tests/fixtures/ml/transformer_fixture.txt";
        std::uint64_t state = 0x5eed2026;
        std::string line, section;
        while (std::getline(in, line)) {
            std::istringstream s(line);
            std::string word, name;
            s >> word;
            if (word.empty() || word[0] == '#')
                continue;
            if (word == "section") {
                s >> section;
                continue;
            }
            std::size_t n = 0;
            if (word == "fill") {
                std::string scale_text;
                s >> name >> n >> scale_text;
                const double scale = std::strtod(scale_text.c_str(), nullptr);
                std::vector<float> v(n);
                for (float& x : v) {
                    std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
                    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
                    z ^= z >> 31;
                    x = float((2.0 * (double(z >> 40) * 0x1p-24) - 1.0) * scale);
                }
                out.filled[section][name] = std::move(v);
                continue;
            }
            s >> n;
            std::vector<std::string> items(n);
            for (auto& item : items)
                s >> item;
            if (section == "gpt2") {
                std::vector<float> v(n);
                for (std::size_t i = 0; i < n; ++i)
                    v[i] = std::bit_cast<float>(static_cast<std::uint32_t>(std::stoul(items[i], nullptr, 16)));
                out.floats[section][word] = std::move(v);
            } else {
                std::vector<double> v(n);
                for (std::size_t i = 0; i < n; ++i)
                    v[i] = std::strtod(items[i].c_str(), nullptr);
                out.doubles[section][word] = std::move(v);
            }
        }
        return out;
    }();
    return f;
}

const std::vector<float>& filled(const std::string& section, const std::string& name)
{
    return data().filled.at(section).at(name);
}

std::span<const float> maybe(const std::string& section, const std::string& name)
{
    const auto& s = data().filled.at(section);
    const auto it = s.find(name);
    return it == s.end() ? std::span<const float>{} : std::span<const float>(it->second);
}

// The largest error relative to max(1, |reference|)
template <typename R>
double worst(const std::vector<float>& got, const std::vector<R>& want)
{
    EXPECT_EQ(got.size(), want.size());
    double e = 0;
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i)
        e = std::max(e, std::abs(double(got[i]) - double(want[i])) / std::max(1.0, std::abs(double(want[i]))));
    return e;
}

// GPT-2's block at llm.c's test size: C = 32, two heads of 16, learned positions outside the block
constexpr ml::transformer_config gpt2_tiny{.d_model = 32, .n_heads = 2, .n_kv_heads = 2, .head_dim = 16, .d_ff = 128,
                                           .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .mlp = ml::mlp_kind::gelu_tanh, .bias = true};

// The textbook blocks of the generator, field for field
constexpr ml::transformer_config llama_tiny{.d_model = 32, .n_heads = 4, .n_kv_heads = 2, .head_dim = 8, .d_ff = 64,
                                            .norm = ml::norm_kind::rms, .norm_eps = 1e-5, .mlp = ml::mlp_kind::swiglu,
                                            .rope = ml::rope_kind::half, .rope_base = 10000};
constexpr ml::transformer_config nanochat_tiny{.d_model = 32, .n_heads = 4, .n_kv_heads = 4, .head_dim = 8, .d_ff = 64,
                                               .norm = ml::norm_kind::rms, .norm_eps = 1.1920928955078125e-07,
                                               .qk_norm = ml::qk_norm_kind::after_rope, .mlp = ml::mlp_kind::relu_squared,
                                               .rope = ml::rope_kind::half, .rope_base = 10000, .window = 3, .attention_softcap = 5};
constexpr ml::transformer_config mqa_tiny{.d_model = 32, .n_heads = 4, .n_kv_heads = 1, .head_dim = 8, .d_ff = 64,
                                          .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .qk_norm = ml::qk_norm_kind::before_rope,
                                          .mlp = ml::mlp_kind::gelu, .rope = ml::rope_kind::interleaved, .rope_base = 500, .bias = true};

ml::block_weights<W> gpt2_weights()
{
    constexpr std::size_t c = 32, f = 128;
    const std::string g = "gpt2";
    const std::span<const float> qkvw(filled(g, "qkvw")), qkvb(filled(g, "qkvb"));
    ml::block_weights<W> w{};
    w.norm1_weight = filled(g, "ln1w");
    w.norm1_bias = filled(g, "ln1b");
    w.attention = {W{qkvw.subspan(0, c * c), c, c}, W{qkvw.subspan(c * c, c * c), c, c}, W{qkvw.subspan(2 * c * c, c * c), c, c},
                   W{filled(g, "attprojw"), c, c}, qkvb.subspan(0, c), qkvb.subspan(c, c), qkvb.subspan(2 * c, c), filled(g, "attprojb")};
    w.norm2_weight = filled(g, "ln2w");
    w.norm2_bias = filled(g, "ln2b");
    w.mlp = {W{filled(g, "fcw"), f, c}, W{}, W{filled(g, "fcprojw"), c, f}, filled(g, "fcb"), {}, filled(g, "fcprojb")};
    return w;
}

template <ml::transformer_config C>
ml::block_weights<W> textbook_weights(const std::string& s)
{
    ml::block_weights<W> w{};
    w.norm1_weight = filled(s, "norm1_weight");
    w.norm1_bias = maybe(s, "norm1_bias");
    w.attention = {W{filled(s, "wq"), C.q_size(), C.d_model}, W{filled(s, "wk"), C.kv_size(), C.d_model},
                   W{filled(s, "wv"), C.kv_size(), C.d_model}, W{filled(s, "wo"), C.d_model, C.q_size()},
                   maybe(s, "bq"), maybe(s, "bk"), maybe(s, "bv"), maybe(s, "bo"), maybe(s, "q_norm"), maybe(s, "k_norm")};
    w.norm2_weight = filled(s, "norm2_weight");
    w.norm2_bias = maybe(s, "norm2_bias");
    const auto gate = maybe(s, "gate");
    w.mlp = {W{filled(s, "up"), C.d_ff, C.d_model}, W{gate, gate.empty() ? 0 : C.d_ff, gate.empty() ? 0 : C.d_model},
             W{filled(s, "down"), C.d_model, C.d_ff}, maybe(s, "up_bias"), {}, maybe(s, "down_bias")};
    return w;
}

// A run of tokens through one block from position 0, with a cache of `capacity` slots
template <ml::transformer_config C, typename R>
std::vector<float> run(const ml::block_weights<W>& w, std::vector<float> x, std::size_t capacity, const R& rope_src)
{
    std::vector<float> k(ml::kv_cache_view<C>::required_size(1, capacity)), v(k.size());
    const ml::kv_cache_view<C> cache(k, v, 1, capacity);
    ml::block_prefill<C>(w, x, 0, cache, 0, rope_src);
    return x;
}

template <ml::transformer_config C>
std::vector<float> textbook(const std::string& s, std::size_t capacity)
{
    static const ml::rope_table<float, 8, C.head_dim> table(ml::rope_base_v<C, float>);
    return run<C>(textbook_weights<C>(s), filled(s, "x"), capacity, table);
}

TEST(MlTransformerBlock, Gpt2BlockIsLlmcsWithinTolerance)
{
    const auto out = run<gpt2_tiny>(gpt2_weights(), filled("gpt2", "x"), 4, ml::no_rope);
    // llm.c computes attention in three passes with expf and core in one with its own exp, so the bits may differ;
    // measured, they agree within one float epsilon (1.2e-7). The bound is eight.
    EXPECT_LT(worst(out, data().floats.at("gpt2").at("out")), 1e-6);
}

TEST(MlTransformerBlock, TextbookBlocksMatchTheDoublePrecisionReference)
{
    const double llama = worst(textbook<llama_tiny>("llama", 6), data().doubles.at("llama").at("out"));
    const double nanochat = worst(textbook<nanochat_tiny>("nanochat", 3), data().doubles.at("nanochat").at("out"));
    const double mqa = worst(textbook<mqa_tiny>("mqa", 6), data().doubles.at("mqa").at("out"));
    // float against double: measured 1.3e-7 to 1.8e-7, a float epsilon or so; the bound is eight
    EXPECT_LT(llama, 1e-6);
    EXPECT_LT(nanochat, 1e-6);
    EXPECT_LT(mqa, 1e-6);
}

// Prefill is block_forward token by token, so a prompt prefilled and the same prompt decoded step by step agree
TEST(MlTransformerBlock, PrefillEqualsDecodingStepByStep)
{
    const auto w = textbook_weights<llama_tiny>("llama");
    static const ml::rope_table<float, 8, 8> table(ml::rope_base_v<llama_tiny, float>);
    const auto prefilled = run<llama_tiny>(w, filled("llama", "x"), 6, table);
    std::vector<float> stepped = filled("llama", "x");
    std::vector<float> k(ml::kv_cache_view<llama_tiny>::required_size(1, 6)), v(k.size());
    const ml::kv_cache_view<llama_tiny> cache(k, v, 1, 6);
    for (std::size_t t = 0; t < 6; ++t)
        ml::block_forward<llama_tiny>(w, std::span<float>(stepped).subspan(t * 32, 32), t, cache, 0, table);
    EXPECT_EQ(prefilled, stepped);
}

// The llama block with every matrix in Q8_0: close to the float result, within the quantisation error
TEST(MlTransformerBlock, Q8WeightsStayNearTheFloatResult)
{
    using Q = ml::quantized_weights<ml::block_q8_0>;
    const std::string s = "llama";
    std::map<std::string, std::vector<ml::block_q8_0>> blocks;
    auto q8 = [&](const std::string& name, std::size_t rows, std::size_t cols) {
        auto& b = blocks[name];
        b.resize(rows * cols / 32);
        ml::quantize(filled(s, name), std::span<ml::block_q8_0>(b));
        return Q{b, rows, cols};
    };
    constexpr auto c = llama_tiny;
    ml::block_weights<Q> w{};
    w.norm1_weight = filled(s, "norm1_weight");
    w.norm2_weight = filled(s, "norm2_weight");
    w.attention = {q8("wq", c.q_size(), c.d_model), q8("wk", c.kv_size(), c.d_model), q8("wv", c.kv_size(), c.d_model), q8("wo", c.d_model, c.q_size())};
    w.mlp = {q8("up", c.d_ff, c.d_model), q8("gate", c.d_ff, c.d_model), q8("down", c.d_model, c.d_ff)};
    static const ml::rope_table<float, 8, 8> table(ml::rope_base_v<llama_tiny, float>);
    std::vector<float> x = filled(s, "x");
    std::vector<float> k(ml::kv_cache_view<c>::required_size(1, 6)), v(k.size());
    ml::block_prefill<c>(w, x, 0, ml::kv_cache_view<c>(k, v, 1, 6), 0, table);
    EXPECT_LT(worst(x, data().doubles.at("llama").at("out")), 0.02);
}

constexpr bool compile_time()
{
    constexpr ml::transformer_config c{.d_model = 2, .n_heads = 1, .n_kv_heads = 1, .head_dim = 2, .d_ff = 2, .norm = ml::norm_kind::rms,
                                       .mlp = ml::mlp_kind::relu_squared};
    std::array<double, 4> eye{1, 0, 0, 1}, zero{};
    using D = ml::dense_weights<double>;
    ml::block_weights<D, double> w{};
    w.attention = {D{eye, 2, 2}, D{eye, 2, 2}, D{eye, 2, 2}, D{zero, 2, 2}};   // the attention adds nothing
    w.mlp = {D{zero, 2, 2}, {}, D{zero, 2, 2}};                               // nor does the MLP
    std::array<double, 4> k{}, v{};
    std::array<double, 2> x{3, 4};
    ml::block_forward<c>(w, x, 0, ml::kv_cache_view<c, double>(k, v, 1, 2), 0, ml::no_rope);
    return x[0] == 3 && x[1] == 4 && k[0] != 0;   // the residual stream is unchanged, the cache is written
}
static_assert(compile_time());

} // namespace
