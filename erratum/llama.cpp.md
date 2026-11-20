# llama.cpp errata

Defects found in llama.cpp while checking core against it. They are upstream's to fix; core records what it does
instead and tests it. Each entry names the llama.cpp commit it was found at.

## src/llama-kv-cells.h: `pos_div` stores the K-shift with the wrong sign (39039aa)

- Evidence: `pos_add` accumulates `p_new - p_old` into a cell's shift; `pos_div` accumulates `p_old - p_new`, the
  opposite sign. The K-shift then rotates each cached key by the shift as stored.
- Impact: after `llama_memory_seq_div` (used by context extension such as self-extend), the K-shift rotates a divided
  key away from its new position instead of to it, so attention sees keys at the wrong positions.
- Core: `ml/kv_cells.hpp` records `p_new - p_old` for both `seq_add` and `seq_div`, so `kv_shift_keys` always moves a key
  to its new position; `tests/ml/kv_cells_test.cpp` checks the K-shift after `seq_div` against keys rotated directly to
  the divided positions. Documented under Known issues in `ml/kv_cells.hpp` and in the 0.17.0 CHANGELOG.
- Upstream: not reported.
