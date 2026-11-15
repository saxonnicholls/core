//
//  muon_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/muon.hpp>: an ill-conditioned gradient orthogonalised (its singular values pulled
//  together), Muon steps fitting a matrix, and a tiny GPT-2 trained end to end: token and position embeddings, two
//  blocks, the final norm, a tied head, Muon on the blocks' matrices and AdamW on the rest, the loss falling and the
//  same bits on a second run. Exits non-zero if a result is not what it documents.
//

#include <snicholls/core/ml/muon.hpp>

#include <snicholls/core/linalg/reference.hpp>
#include <snicholls/core/ml/embedding.hpp>
#include <snicholls/core/ml/lm_head.hpp>
#include <snicholls/core/ml/norm.hpp>
#include <snicholls/core/ml/transformer_block.hpp>
#include <snicholls/core/stats/random.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

namespace ml = snicholls::ml;
namespace la = snicholls::linalg;

//! [train_gpt2]
// GPT-2 in miniature: 32 channels, 2 heads of 16, an MLP of 128, layer norm, tanh GELU, biases
constexpr ml::transformer_config tiny{.d_model = 32, .n_heads = 2, .n_kv_heads = 2, .head_dim = 16, .d_ff = 128,
                                      .norm = ml::norm_kind::layer, .norm_eps = 1e-5, .mlp = ml::mlp_kind::gelu_tanh, .bias = true};
constexpr std::size_t vocab = 16, d = tiny.d_model, f = tiny.d_ff, n = 16, layers = 2;

// One parameter tensor with its gradient and its optimiser's state: Muon's momentum, or AdamW's two moments
struct tensor {
    std::size_t rows, cols;
    bool muon;   // a matrix inside a block
    std::vector<float> w, g, m, v;
    tensor(std::size_t r, std::size_t c, bool matrix, float scale, float offset, snicholls::stats::wyrand& rng)
        : rows(r), cols(c), muon(matrix), w(r * c), g(r * c), m(r * c), v(r * c)
    {
        for (float& e : w)
            e = offset + scale * static_cast<float>(2 * snicholls::stats::uniform_real(rng) - 1);
    }
    ml::dense_weights<float> dense() const { return {w, rows, cols}; }
};

struct block {
    tensor n1w, n1b, wq, wk, wv, wo, bq, bk, bv, bo, n2w, n2b, up, bup, down, bdown;
    explicit block(snicholls::stats::wyrand& r)
        : n1w(1, d, false, 0, 1, r), n1b(1, d, false, 0, 0, r), wq(d, d, true, 0.3f, 0, r), wk(d, d, true, 0.3f, 0, r),
          wv(d, d, true, 0.3f, 0, r), wo(d, d, true, 0.1f, 0, r), bq(1, d, false, 0, 0, r), bk(1, d, false, 0, 0, r),
          bv(1, d, false, 0, 0, r), bo(1, d, false, 0, 0, r), n2w(1, d, false, 0, 1, r), n2b(1, d, false, 0, 0, r),
          up(f, d, true, 0.3f, 0, r), bup(1, f, false, 0, 0, r), down(d, f, true, 0.05f, 0, r), bdown(1, d, false, 0, 0, r)
    {
    }
    std::array<tensor*, 16> all() { return {&n1w, &n1b, &wq, &wk, &wv, &wo, &bq, &bk, &bv, &bo, &n2w, &n2b, &up, &bup, &down, &bdown}; }
    ml::block_weights<ml::dense_weights<float>> weights() const
    {
        return {n1w.w, n1b.w, {wq.dense(), wk.dense(), wv.dense(), wo.dense(), bq.w, bk.w, bv.w, bo.w}, n2w.w, n2b.w,
                {up.dense(), {}, down.dense(), bup.w, {}, bdown.w}};
    }
};

// What one block's forward pass saves, and the working space of its backward pass (shared by the blocks)
struct saved {
    std::vector<float> x0 = std::vector<float>(n * d), h1 = x0, x1 = x0, h2 = x0, q = x0, k = x0, v = x0, y = x0,
                       lse = std::vector<float>(n * tiny.n_heads), up = std::vector<float>(n * f);
    ml::block_activations<float> act() { return {x0, h1, x1, h2, {q, k, v, {}, {}, y, lse}, up, {}}; }
};

struct model {
    snicholls::stats::wyrand rng{2026};
    tensor wte{vocab, d, false, 0.05f, 0, rng}, wpe{n, d, false, 0.1f, 0, rng}, lnfw{1, d, false, 0, 1, rng}, lnfb{1, d, false, 0, 0, rng};
    std::vector<block> blocks = [this] {
        std::vector<block> b;
        for (std::size_t l = 0; l < layers; ++l)
            b.emplace_back(rng);   // each block its own draws
        return b;
    }();
    std::vector<saved> acts = std::vector<saved>(layers);
    std::vector<float> dq = std::vector<float>(n * d), dk = dq, dv = dq, scratch = dq, work = std::vector<float>(ml::muon_work_size(f, d));

    // One training step on a sequence: forward, loss, backward, then every tensor's update. Returns the loss.
    float step(const std::vector<std::size_t>& text, std::uint64_t t)
    {
        const std::span<const std::size_t> in(text.data(), n), next(text.data() + 1, n);
        std::vector<float> x(n * d), xf(n * d), h(n * d), dh(n * d), logits(vocab);
        std::vector<ml::norm_stats<float>> stats(n);
        for (std::size_t p = 0; p < n; ++p)
            ml::embed_with_position(wte.w, in[p], wpe.w, p, std::span<float>(x).subspan(p * d, d));
        for (std::size_t l = 0; l < layers; ++l)
            ml::block_forward_sequence<tiny>(blocks[l].weights(), x, ml::no_rope, acts[l].act());
        xf = x;   // the final norm's input
        for (std::size_t p = 0; p < n; ++p)
            stats[p] = ml::layer_norm(std::span<const float>(xf).subspan(p * d, d), lnfw.w, lnfb.w, std::span<float>(h).subspan(p * d, d));
        const float loss = ml::lm_head_loss(wte.dense(), h, next, logits, dh, wte.g);   // the head is wte, tied

        for (std::size_t p = 0; p < n; ++p)
            ml::layer_norm_backward(std::span<const float>(dh).subspan(p * d, d), std::span<const float>(xf).subspan(p * d, d), lnfw.w,
                                    stats[p], std::span<float>(x).subspan(p * d, d), lnfw.g, lnfb.g);
        for (std::size_t l = layers; l-- > 0;) {
            block& b = blocks[l];
            const ml::block_gradients<float> grads{
                b.n1w.g, b.n1b.g, b.n2w.g, b.n2b.g,
                {dq, dk, dv, b.wq.g, b.wk.g, b.wv.g, b.wo.g, b.bq.g, b.bk.g, b.bv.g, b.bo.g, {}, {}},
                {b.up.g, {}, b.down.g, b.bup.g, {}, b.bdown.g},
                scratch};
            ml::block_backward<tiny>(b.weights(), ml::no_rope, acts[l].act(), x, grads);
        }
        for (std::size_t p = 0; p < n; ++p) {
            ml::embedding_backward(std::span<const float>(x).subspan(p * d, d), in[p], wte.g);
            ml::embedding_backward(std::span<const float>(x).subspan(p * d, d), p, wpe.g);
        }

        std::vector<tensor*> all{&wte, &wpe, &lnfw, &lnfb};
        for (block& b : blocks)
            for (tensor* e : b.all())
                all.push_back(e);
        for (tensor* e : all) {
            if (e->muon)
                ml::muon_step(la::reference, std::span<float>(e->w), std::span<const float>(e->g), std::span<float>(e->m), e->rows,
                              e->cols, ml::muon_config<float>{.lr = 0.02f}, std::span<float>(work));
            else
                ml::adamw_step(e->w, e->g, e->m, e->v, t, ml::adamw_config<float>{.learning_rate = 0.01f, .beta2 = 0.95f});
            std::fill(e->g.begin(), e->g.end(), 0.0f);   // gradients accumulate: clear them for the next step
        }
        return loss;
    }
};

std::vector<float> train(const std::vector<std::size_t>& text, int steps)
{
    model m;
    std::vector<float> losses;
    for (int t = 1; t <= steps; ++t)
        losses.push_back(m.step(text, static_cast<std::uint64_t>(t)));
    return losses;
}
//! [train_gpt2]

} // namespace

int main()
{
    std::printf("snicholls-core muon demo\n");
    bool ok = true;

    //! [muon]
    // A 2 x 3 gradient whose rows differ in size by 100: plain SGD would move almost only along the first
    const std::vector<float> g{10.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f};
    std::vector<float> x(g.size()), work(ml::muon_work_size(2, 3));
    ml::muon_orthogonalise(la::reference, std::span<const float>(g), 2, 3, std::span<float>(x), std::span<float>(work));

    // Fitting W to a target by Muon steps on the gradient of 1/2 |W - target|^2, which is W - target
    const std::vector<float> target{1.0f, -1.0f, 0.5f, 0.0f, 2.0f, -0.5f};
    std::vector<float> w(6, 0.0f), momentum(6, 0.0f), grad(6);
    auto distance = [&] {
        float dist = 0.0f;
        for (std::size_t i = 0; i < w.size(); ++i)
            dist += (w[i] - target[i]) * (w[i] - target[i]);
        return std::sqrt(dist);
    };
    const float before = distance();
    for (int step = 0; step < 30; ++step) {
        for (std::size_t i = 0; i < w.size(); ++i)
            grad[i] = w[i] - target[i];
        ml::muon_step(la::reference, std::span<float>(w), std::span<const float>(grad), std::span<float>(momentum), 2, 3,
                      ml::muon_config<float>{.lr = 0.1f, .momentum = 0.9f}, std::span<float>(work));
    }
    const float after = distance();
    //! [muon]

    std::printf("  gradient rows 10 and 0.1 become %.3f and %.3f\n", static_cast<double>(x[0]), static_cast<double>(x[4]));
    std::printf("  distance to the target: %.4f before, %.4f after 30 steps\n", static_cast<double>(before), static_cast<double>(after));
    ok &= x[0] > 0.6f && x[0] < 1.25f && x[4] > 0.6f && x[4] < 1.25f;   // both singular values near 1
    ok &= after < 0.25f * before;

    // The tiny GPT-2 learns a sequence of 17 tokens by heart: next-token prediction at each of 16 positions
    std::vector<std::size_t> text(n + 1);
    snicholls::stats::wyrand pick(7);
    for (std::size_t& tok : text)
        tok = snicholls::stats::uniform_index(pick, vocab);
    constexpr int steps = 30;
    const auto losses = train(text, steps), again = train(text, steps);
    std::printf("  tiny GPT-2, loss by step:");
    for (int t = 0; t < steps; t += 5)
        std::printf(" %d: %.3f", t + 1, static_cast<double>(losses[static_cast<std::size_t>(t)]));
    std::printf(" %d: %.4f\n", steps, static_cast<double>(losses.back()));
    std::printf("  log(16) = %.3f is a uniform guess; a second run %s\n", std::log(16.0), losses == again ? "gives the same bits" : "DIFFERS");
    ok &= std::abs(losses.front() - std::log(16.0f)) < 0.5f;   // it starts near a uniform guess
    ok &= losses.back() < 0.05f * losses.front();               // and learns the sequence
    ok &= losses == again;                                       // reproducibly
    std::printf(ok ? "muon demo: ok\n" : "muon demo: FAILED\n");
    return ok ? 0 : 1;
}
