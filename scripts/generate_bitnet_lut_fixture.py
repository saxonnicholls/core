#!/usr/bin/env python3
#
#  generate_bitnet_lut_fixture.py
#  snicholls-core
#
#  Copyright 2010-2026 Saxon Herschel Nicholls
#
#  Writes tests/fixtures/ml/bitnet_lut_fixture.txt: the bytes of BitNet's TL1 and TL2 weight layouts, from the
#  layout functions of microsoft/BitNet 0b341e5 utils/convert-hf-to-gguf-bitnet.py (process_tl1,
#  preprocess_weights_tl1, preprocess_two_weights_tl2, preprocess_three_weights_tl2, preprocess_weights_tl2),
#  transcribed below unchanged except that the tile sizes are arguments instead of a lookup in
#  include/kernel_config.ini. Needs numpy; not part of the build.
#
#    python3 scripts/generate_bitnet_lut_fixture.py > tests/fixtures/ml/bitnet_lut_fixture.txt
#
#  Each line: tl1|tl2 M K BM BK bm, then the FNV-1a 64 hash of the input ternary matrix (int8, row-major) and of the
#  packed bytes, and their count. The matrix comes from the LCG below, which tests/ml/bitnet_kernels_test.cpp repeats.

import numpy as np

MASK = (1 << 64) - 1


def ternary(m, k, seed):
    s = (seed * 0x9E3779B97F4A7C15 + 1) & MASK
    out = np.zeros(m * k, dtype=np.int8)
    for i in range(m * k):
        s = (s * 6364136223846793005 + 1442695040888963407) & MASK
        out[i] = (s >> 33) % 3 - 1
    return out.reshape(m, k)


def fnv(data):
    h = 0xCBF29CE484222325
    for b in bytes(data):
        h ^= b
        h = (h * 0x100000001B3) & MASK
    return h


# --- transcribed from BitNet utils/convert-hf-to-gguf-bitnet.py ---

def process_tl1(weight, BM, BY, bm, by, M, K):
    weight = weight.reshape((M, K // 2)).astype(np.uint8)
    weight = weight.reshape((M // BM, BM, K // 2)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 2, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 2)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 2, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm // 16, 16, by // 2)).transpose(0, 1, 2, 3, 4, 6, 5)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm // 16, by // 4, 4 // 2, 16)).transpose(0, 1, 2, 3, 4, 5, 7, 6)
    weight = weight.reshape((M * K // 16 // 4, 16, 4 // 2))
    weight_0 = weight[:, :, 0] << 4
    weight_1 = weight[:, :, 1]
    weight = weight_0 + weight_1
    return weight


def preprocess_weights_tl1(w, BM, BY, bm):
    M, K = w.shape
    weight = w
    weight = np.where(np.abs(weight) < 1e-6, 0, weight).astype(np.float32)
    weight = np.sign(weight)
    weight_num = np.prod(weight.shape)
    by = 256 // bm
    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]
    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)
    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)
    weight = process_tl1(weight, BM, BY, bm, by, M, K)
    return weight


def preprocess_two_weights_tl2(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]
    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)
    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)
    weight = weight.reshape((M // BM, BM, K // 2)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 2, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 2)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 2, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm, by // 2))
    weight_0 = weight[:, :, :, :, :, 0]
    weight_1 = weight[:, :, :, :, :, 1]
    weight_0 = weight_0 << 4
    weight_1 = weight_1
    weight = weight_0 + weight_1
    weight = weight.reshape((M * K // bm // by, bm // 8, 8))
    weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :]
    weight = weight.reshape(M * K // bm // by, bm)
    for i in range(weight.shape[0]):
        final_weight.append(weight[i, :])


def preprocess_three_weights_tl2(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 3, 3))
    split_weights = np.split(weight, 3, axis=1)
    first_weight = np.multiply(split_weights[0], 9)
    second_weight = np.multiply(split_weights[1], 3)
    third_weight = split_weights[2]
    weight = np.reshape((first_weight + second_weight + third_weight), weight_num // 3)
    sign_weight = np.sign(weight) + 2
    sign_weight = np.where(sign_weight > 1, 0, sign_weight)
    weight = np.abs(weight)
    weight = np.reshape(weight, (M, K // 3)).astype(np.uint8)
    sign_weight = np.reshape(sign_weight, (M, K // 3)).astype(np.uint8)
    weight = weight.reshape((M // BM, BM, K // 3)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 3, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 3)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 3, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm, by // 3))
    weight_0 = weight[:, :, :, :, :, 0]
    weight_1 = weight[:, :, :, :, :, 1]
    weight_0 = weight_0 << 4
    weight_1 = weight_1
    weight = weight_0 + weight_1
    weight = weight.reshape((M * K // bm // by, bm // 8, 8))
    weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :]
    weight = weight.reshape(M * K // bm // by, bm)
    for i in range(weight.shape[0]):
        final_weight.append(weight[i, :])
    sign_weight = sign_weight.reshape((M // BM, BM, K // 3)).transpose(0, 2, 1)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BY // 3, BM)).transpose(0, 1, 3, 2)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 3)).transpose(0, 1, 2, 4, 3)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), by // 3 * 4, bm)).transpose(0, 1, 2, 3, 5, 4)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), bm, by // 3 * 4)).transpose(0, 1, 2, 3, 5, 4)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), by // 3 * 8, bm // 2)).astype(np.uint16)
    combine_weight = np.zeros((M // BM, K // BY, BM // bm, BY // (by * 4), bm // 2), dtype=np.uint16)
    for i in range(16):
        temp_weight = sign_weight[:, :, :, :, i, :] << 15 - i
        combine_weight += temp_weight
    combine_weight = combine_weight.view(np.uint8)
    combine_weight = combine_weight.reshape((M * K // bm // (by * 4)), bm)
    for i in range(combine_weight.shape[0]):
        final_weight.append(combine_weight[i, :])


def preprocess_weights_tl2(w, BM, BY, bm):
    M, K = w.shape
    weight = w
    weight = np.where(np.abs(weight) < 1e-6, 0, weight).astype(np.float32)
    weight = np.sign(weight)
    weight_num = np.prod(weight.shape)
    by = 192 // bm
    if (weight.shape[1] % BY != 0):
        slice_k_idx = weight.shape[1] - weight.shape[1] % BY
        slice_weights = np.split(weight, [slice_k_idx], axis=1)
        three_weight = slice_weights[0]
        two_weight = slice_weights[1]
    else:
        three_weight = weight
    final_weight = []
    preprocess_three_weights_tl2(three_weight.shape[0], three_weight.shape[1], three_weight.shape[0] * three_weight.shape[1],
                                 BM, BY, bm, by, three_weight, final_weight)
    if (weight.shape[1] % BY != 0):
        preprocess_two_weights_tl2(two_weight.shape[0], two_weight.shape[1], two_weight.shape[0] * two_weight.shape[1],
                                   BM, 32, 32, 4, two_weight, final_weight)
    weight = np.array(final_weight, dtype=np.uint8).reshape(-1)
    weight = np.pad(weight, (0, (K - 256) * M // 3 * 5 // 8 + 256 * M // 2 * 4 // 8 - weight.shape[0]), mode='constant', constant_values=0)
    return weight

# --- end of the transcription ---


def main():
    print("# BitNet's TL1 and TL2 weight layouts (microsoft/BitNet 0b341e5), from scripts/generate_bitnet_lut_fixture.py")
    print("# <layout> M K BM BK bm in=<ternary matrix hash> out=<packed bytes hash> bytes=<count, hex>")
    # TL1: (M, K, BM, BK, bm); the last two as in preset_kernels/bitnet_b1_58-large/kernel_config_tl1.ini, one row tile
    for (m, k, BM, BK, bm) in [(64, 128, 32, 64, 32), (128, 256, 64, 128, 64), (256, 4096, 256, 128, 32), (128, 1536, 128, 64, 64)]:
        t = ternary(m, k, m * 7 + k)
        out = preprocess_weights_tl1(t.astype(np.float32), BM, BK, bm).astype(np.uint8).reshape(-1)
        print(f"tl1 {m} {k} {BM} {BK} {bm} in={fnv(t.astype(np.int8).tobytes()):016x} out={fnv(out.tobytes()):016x} bytes={out.size:x}")
    # TL2: as preset_kernels/bitnet_b1_58-3B/kernel_config_tl2.ini (K = 3200: a 32-column two-weight tail), one row tile
    for (m, k, BM, BK, bm) in [(64, 320, 32, 96, 32), (64, 384, 64, 192, 32), (320, 3200, 320, 96, 32), (160, 8640, 160, 96, 32)]:
        t = ternary(m, k, m * 7 + k)
        out = preprocess_weights_tl2(t.astype(np.float32), BM, BK, bm).astype(np.uint8).reshape(-1)
        print(f"tl2 {m} {k} {BM} {BK} {bm} in={fnv(t.astype(np.int8).tobytes()):016x} out={fnv(out.tobytes()):016x} bytes={out.size:x}")


if __name__ == "__main__":
    main()
