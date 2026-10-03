// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Types for the fused T=1 (decode step) gated delta rule ttnn op.
// One Tensix core per (B*H) head; the whole recurrent step
// (L2-norm q/k, decay, v_read, delta, rank-1 write, o = q@h) runs in one
// reader/compute/writer program, matching the python graph
// `recurrent_gated_delta_rule_decode_ttnn`.

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct DecodeGatedDeltaRuleParams {
    uint32_t B;   // batch
    uint32_t H;   // heads (== v heads; no GQA in the decode graph)
    uint32_t BH;  // B * H (one core per head)
    uint32_t K;   // key dim (multiple of 32)
    uint32_t V;   // value dim (multiple of 32)
    bool has_initial_state;
    bool inplace_state;  // write new state back into initial_state's buffer
    float scale;         // folded into q's L2-norm factor (K**-0.5 by default)
    tt::tt_metal::MemoryConfig output_mem_config;
    // Packed mode (K's milestone 2): q/k/v are the SAME [1,B,C] TILE tensor -- the conv+gates
    // kernel's output, channels [q | k | v] head-major -- and beta/g are [1,B,H] TILE. The
    // reader gathers head (b,h)'s rows straight out of that layout (row b, tile column
    // off + head*Dt) and takes q/k from GQA source head h / rf, so the model's slices,
    // reshapes and repeat_interleaves disappear. Compute and writer are unchanged.
    bool packed = false;
    uint32_t Ct = 0;       // C / 32
    uint32_t q_off_t = 0;  // tile column where q starts (0)
    uint32_t k_off_t = 0;  // Nk*K / 32
    uint32_t v_off_t = 0;  // 2*Nk*K / 32
    uint32_t rf = 1;       // H / Nk (GQA expansion factor)
    uint32_t Nvt = 1;      // ceil(H / 32): beta/g tile columns
    // Fused output norm + gate (K's last slice): the op returns gated [1,B,H*V] TILE =
    // rms_norm(o) * norm_w * silu(z) instead of o, assembled across cores (L1 scatter +
    // semaphore per (batch-tile, head)). Requires packed mode.
    bool fuse_ng = false;
    uint32_t Wt_z = 0;          // z's width in tiles
    uint32_t z_off_t = 0;       // tile column of head 0 in z
    uint32_t ng_eps_bits = 0;   // fp32 bits of V*epsilon   (x / sqrt(mean + eps) == x*sqrt(V)/sqrt(sum + V*eps))
    uint32_t ng_scale_bits = 0; // fp32 bits of sqrt(V)
};

// All inputs are python-facing shapes, TILE layout, same dtype (bf16 or fp32),
// on device. T=1 inputs ([B,1,H,*]) share TILE pages across 32 heads; the
// reader gathers each head's row out of the shared pages. Outputs: o comes
// back ROW_MAJOR (page bh == head bh's [V] stick; full-page writes only), the
// new state is TILE [B,H,K,V].
struct DecodeGatedDeltaRuleInputs {
    Tensor q;                             // [B,1,H,K]
    Tensor k;                             // [B,1,H,K]
    Tensor v;                             // [B,1,H,V]
    Tensor beta;                          // [B,1,H]
    Tensor g;                             // [B,1,H]  log-space decay
    std::optional<Tensor> initial_state;  // [B,H,K,V] (absent => zeros)
    std::optional<Tensor> z;              // [1,Bz,W] TILE (fuse_ng): output gate input, column window
    std::optional<Tensor> norm_w;         // [1,1,V]  TILE (fuse_ng): rms_norm weight
};

}  // namespace ttnn::prim
