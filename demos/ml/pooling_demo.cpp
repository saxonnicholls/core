//
//  pooling_demo.cpp
//  snicholls-core
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//  Demo for <snicholls/core/ml/pooling.hpp>: two packed sequences pooled by mean, first and last token, the backward
//  pass of the mean, and a reranker's head turning the first token's row into a relevance score. Exits non-zero if a
//  result is not what it documents.
//

#include <snicholls/core/ml/pooling.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>

int main()
{
    std::printf("snicholls-core pooling demo\n");
    namespace ml = snicholls::ml;
    bool ok = true;

    //! [pooling]
    // Two sequences packed end to end, as a batch holds them: rows 0 to 2 and row 3, two values a row
    const std::array<float, 8> rows{1, 2, 3, 4, 5, 6, 10, 20};
    const std::array<std::size_t, 3> starts{0, 3, 4};
    std::array<float, 4> mean{}, cls{}, last{};
    ml::pool_rows<float>(ml::pooling_kind::mean, rows, 2, starts, mean);   // sentence-transformers
    ml::pool_rows<float>(ml::pooling_kind::cls, rows, 2, starts, cls);     // BERT, BGE
    ml::pool_rows<float>(ml::pooling_kind::last, rows, 2, starts, last);   // Qwen3-Embedding
    std::array<float, 8> none{}, drows{};
    ml::pool_rows<float>(ml::pooling_kind::none, rows, 2, starts, none);
    ml::pool_rows_backward<float>(ml::pooling_kind::mean, std::array<float, 4>{3, 3, 1, 1}, 2, starts, drows);

    // A reranker: the first token's row through a dense layer, tanh, and a one-score projection
    const std::array<float, 4> dense{1, 0, 0, 1};
    const std::array<float, 2> project{2, -1};
    const ml::rank_head_weights<float, float> head{.cls = {dense, 2, 2}, .out = {project, 1, 2}};
    std::array<float, 2> scratch{};
    std::array<float, 1> score{};
    ml::rank_head(head, std::span<const float>(cls).first(2), scratch, score);
    //! [pooling]

    std::printf("  mean [%.1f %.1f | %.1f %.1f], cls [%.0f %.0f], last [%.0f %.0f]\n", static_cast<double>(mean[0]), static_cast<double>(mean[1]),
                static_cast<double>(mean[2]), static_cast<double>(mean[3]), static_cast<double>(cls[0]), static_cast<double>(cls[1]),
                static_cast<double>(last[0]), static_cast<double>(last[1]));
    ok &= mean[0] == 3.0f && mean[1] == 4.0f && mean[2] == 10.0f && cls[0] == 1.0f && last[1] == 6.0f && cls[2] == 10.0f && none == rows;
    ok &= drows[0] == 1.0f && drows[6] == 1.0f;   // a mean's gradient is spread evenly: 3 / 3
    std::printf("  rerank score %.4f\n", static_cast<double>(score[0]));
    ok &= std::abs(score[0] - (2.0f * std::tanh(1.0f) - std::tanh(2.0f))) < 1e-6f;

    std::printf(ok ? "pooling demo: ok\n" : "pooling demo: FAILED\n");
    return ok ? 0 : 1;
}
