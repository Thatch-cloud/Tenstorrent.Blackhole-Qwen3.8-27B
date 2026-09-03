"""K's last slice: fold the output norm + gate into decode_gated_delta_rule (packed mode).

With z and norm_w given, each head core finishes its own o row -- rms_norm over V, times
norm_w, times silu(z[b, z_off + h*V : +V]) -- and scatters the row's 16-element chunks by
NOC into the L1 of the assembler core for (batch-tile, head), then bumps that core's
semaphore. After its own instances, the assembler waits for the row count and writes the
Vt full output tiles of gated [1, B, H*V] TILE. L1 sub-page writes are fine; only DRAM
sub-page writes are the thing this op avoids. The op then returns (gated, new_state).

Applies on top of patch_packed.py. Idempotent; run from kwork/.
"""
import io
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode_gated_delta_rule")


def rw(rel, fn):
    p = os.path.join(ROOT, rel)
    s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    s2 = fn(s)
    assert s2 != s, p
    io.open(p, "w", encoding="utf-8", newline="\n").write(s2)
    print("patched", rel)


def types_hpp(s):
    old = """    uint32_t rf = 1;       // H / Nk (GQA expansion factor)
    uint32_t Nvt = 1;      // ceil(H / 32): beta/g tile columns
};"""
    new = """    uint32_t rf = 1;       // H / Nk (GQA expansion factor)
    uint32_t Nvt = 1;      // ceil(H / 32): beta/g tile columns
    // Fused output norm + gate (K's last slice): the op returns gated [1,B,H*V] TILE =
    // rms_norm(o) * norm_w * silu(z) instead of o, assembled across cores (L1 scatter +
    // semaphore per (batch-tile, head)). Requires packed mode.
    bool fuse_ng = false;
    uint32_t Wt_z = 0;          // z's width in tiles
    uint32_t z_off_t = 0;       // tile column of head 0 in z
    uint32_t ng_eps_bits = 0;   // fp32 bits of V*epsilon   (x / sqrt(mean + eps) == x*sqrt(V)/sqrt(sum + V*eps))
    uint32_t ng_scale_bits = 0; // fp32 bits of sqrt(V)
};"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    Tensor g;                             // [B,1,H]  log-space decay
    std::optional<Tensor> initial_state;  // [B,H,K,V] (absent => zeros)
};"""
    new = """    Tensor g;                             // [B,1,H]  log-space decay
    std::optional<Tensor> initial_state;  // [B,H,K,V] (absent => zeros)
    std::optional<Tensor> z;              // [1,Bz,W] TILE (fuse_ng): output gate input, column window
    std::optional<Tensor> norm_w;         // [1,1,V]  TILE (fuse_ng): rms_norm weight
};"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def devop_hpp(s):
    old = """    uint32_t head_k,
    uint32_t head_v,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim"""
    new = """    uint32_t head_k,
    uint32_t head_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& z = std::nullopt,
    const std::optional<Tensor>& norm_w = std::nullopt,
    uint32_t z_col_offset = 0,
    float epsilon = 1e-6f);

}  // namespace ttnn::prim"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def devop_cpp(s):
    old = """        TT_FATAL(
            bs.rank() == 3 && bs[0] == 1 && bs[1] == attrs.B && bs[2] == attrs.H && gs == bs,
            "decode_gated_delta_rule (packed): beta and g must be [1,B,H]");
    } else {"""
    new = """        TT_FATAL(
            bs.rank() == 3 && bs[0] == 1 && bs[1] == attrs.B && bs[2] == attrs.H && gs == bs,
            "decode_gated_delta_rule (packed): beta and g must be [1,B,H]");
        if (attrs.fuse_ng) {
            TT_FATAL(in.z.has_value() && in.norm_w.has_value(), "decode_gated_delta_rule (fused norm+gate): z and norm_w required");
            check(*in.z, "z");
            check(*in.norm_w, "norm_w");
            const auto& zs = in.z->logical_shape();
            TT_FATAL(
                zs.rank() == 3 && zs[0] == 1 && zs[1] >= attrs.B && zs[2] >= attrs.z_off_t * TILE_WIDTH + attrs.H * attrs.V,
                "decode_gated_delta_rule (fused norm+gate): z must be [1,>=B,W] with W >= z_col_offset + H*V");
            const auto& ws = in.norm_w->logical_shape();
            TT_FATAL(ws[-1] == attrs.V && in.norm_w->logical_volume() == attrs.V, "decode_gated_delta_rule (fused norm+gate): norm_w must be [1,1,V]");
        }
    } else {
        TT_FATAL(!attrs.fuse_ng, "decode_gated_delta_rule: fused norm+gate requires packed mode");"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    const auto layout_rm = TensorLayout(dt, PageConfig(Layout::ROW_MAJOR), attrs.output_mem_config);
    const auto layout_tile = TensorLayout(dt, PageConfig(Layout::TILE), attrs.output_mem_config);
    return {
        tt::tt_metal::TensorSpec(ttnn::Shape({attrs.B, 1, attrs.H, attrs.V}), layout_rm),
        tt::tt_metal::TensorSpec(ttnn::Shape({attrs.B, attrs.H, attrs.K, attrs.V}), layout_tile)};"""
    new = """    const auto layout_rm = TensorLayout(dt, PageConfig(Layout::ROW_MAJOR), attrs.output_mem_config);
    const auto layout_tile = TensorLayout(dt, PageConfig(Layout::TILE), attrs.output_mem_config);
    if (attrs.fuse_ng) {
        // gated [1,B,H*V] TILE: the assembler cores write full tiles (see the writer).
        return {
            tt::tt_metal::TensorSpec(ttnn::Shape({1, attrs.B, attrs.H * attrs.V}), layout_tile),
            tt::tt_metal::TensorSpec(ttnn::Shape({attrs.B, attrs.H, attrs.K, attrs.V}), layout_tile)};
    }
    return {
        tt::tt_metal::TensorSpec(ttnn::Shape({attrs.B, 1, attrs.H, attrs.V}), layout_rm),
        tt::tt_metal::TensorSpec(ttnn::Shape({attrs.B, attrs.H, attrs.K, attrs.V}), layout_tile)};"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    uint32_t head_k,
    uint32_t head_v,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = DecodeGatedDeltaRuleDeviceOperation;
    using namespace tt::constants;
    const auto& qs = qkv.logical_shape();  // [1,B,C]"""
    new = """    uint32_t head_k,
    uint32_t head_v,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<Tensor>& z,
    const std::optional<Tensor>& norm_w,
    uint32_t z_col_offset,
    float epsilon) {
    using OperationType = DecodeGatedDeltaRuleDeviceOperation;
    using namespace tt::constants;
    const auto& qs = qkv.logical_shape();  // [1,B,C]"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """        .rf = num_v_heads / num_k_heads,
        .Nvt = (num_v_heads + TILE_WIDTH - 1) / TILE_WIDTH,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = qkv,
        .k = qkv,
        .v = qkv,
        .beta = beta,
        .g = g,
        .initial_state = initial_state,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}"""
    new = """        .rf = num_v_heads / num_k_heads,
        .Nvt = (num_v_heads + TILE_WIDTH - 1) / TILE_WIDTH,
    };
    if (z.has_value()) {
        TT_FATAL(z_col_offset % TILE_WIDTH == 0, "decode_gated_delta_rule (fused norm+gate): z_col_offset must be a multiple of 32");
        attrs.fuse_ng = true;
        attrs.Wt_z = (static_cast<uint32_t>(z->logical_shape()[2]) + TILE_WIDTH - 1) / TILE_WIDTH;
        attrs.z_off_t = z_col_offset / TILE_WIDTH;
        attrs.ng_eps_bits = std::bit_cast<uint32_t>(epsilon * static_cast<float>(head_v));
        attrs.ng_scale_bits = std::bit_cast<uint32_t>(std::sqrt(static_cast<float>(head_v)));
    }
    auto tensor_args = OperationType::tensor_args_t{
        .q = qkv,
        .k = qkv,
        .v = qkv,
        .beta = beta,
        .g = g,
        .initial_state = initial_state,
        .z = z,
        .norm_w = norm_w,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    if "#include <bit>" not in s:
        s = s.replace("#include <tt-metalium/constants.hpp>\n", "#include <bit>\n#include <cmath>\n#include <tt-metalium/constants.hpp>\n", 1)
    return s


def api_hpp(s):
    old = """    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool inplace_state = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::transformer"""
    new = """    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool inplace_state = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& z = std::nullopt,
    const std::optional<ttnn::Tensor>& norm_w = std::nullopt,
    uint32_t z_col_offset = 0,
    float epsilon = 1e-6f);

}  // namespace ttnn::transformer"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """ * and reads head (b,h)'s q/k from GQA source head h / (H/Nk). Same math, same outputs as
 * decode_gated_delta_rule; no host-side slicing, reshaping or head expansion.
 */"""
    new = """ * and reads head (b,h)'s q/k from GQA source head h / (H/Nk). Same math, same outputs as
 * decode_gated_delta_rule; no host-side slicing, reshaping or head expansion.
 *
 * With z [1,Bz,W] and norm_w [1,1,V] given (fused norm + gate), the first output is instead
 *   gated [1, B, H*V] TILE = rms_norm(o) * norm_w * silu(z[:, :, z_col_offset + h*V ...])
 * assembled across cores, i.e. what the GDN out-projection consumes.
 */"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def api_cpp(s):
    old = """    std::optional<float> scale,
    const std::optional<ttnn::Tensor>& initial_state,
    bool inplace_state,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const float s = scale.value_or(1.0f / std::sqrt(static_cast<float>(head_k)));
    auto results = ttnn::prim::decode_gated_delta_rule_packed(
        qkv,
        beta,
        g,
        initial_state,
        inplace_state,
        s,
        num_k_heads,
        num_v_heads,
        head_k,
        head_v,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));"""
    new = """    std::optional<float> scale,
    const std::optional<ttnn::Tensor>& initial_state,
    bool inplace_state,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<ttnn::Tensor>& z,
    const std::optional<ttnn::Tensor>& norm_w,
    uint32_t z_col_offset,
    float epsilon) {
    const float s = scale.value_or(1.0f / std::sqrt(static_cast<float>(head_k)));
    auto results = ttnn::prim::decode_gated_delta_rule_packed(
        qkv,
        beta,
        g,
        initial_state,
        inplace_state,
        s,
        num_k_heads,
        num_v_heads,
        head_k,
        head_v,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        z,
        norm_w,
        z_col_offset,
        epsilon);"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def nanobind_cpp(s):
    old = """        nb::arg("num_k_heads"),
        nb::arg("num_v_heads"),
        nb::arg("head_k"),
        nb::arg("head_v"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("inplace_state") = false,
        nb::arg("memory_config") = nb::none());
}"""
    new = """        nb::arg("num_k_heads"),
        nb::arg("num_v_heads"),
        nb::arg("head_k"),
        nb::arg("head_v"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("inplace_state") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("z") = nb::none(),
        nb::arg("norm_w") = nb::none(),
        nb::arg("z_col_offset") = 0,
        nb::arg("epsilon") = 1e-6f);
}"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """        Keyword Args:
            scale, initial_state, inplace_state, memory_config: as decode_gated_delta_rule.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: o [B,1,H,V] ROW_MAJOR, new_state [B,H,K,V] TILE."""
    new = """        Keyword Args:
            scale, initial_state, inplace_state, memory_config: as decode_gated_delta_rule.
            z (ttnn.Tensor, optional): [1, Bz, W] TILE. With norm_w, fuses the output norm and
                gate: the first output becomes gated [1, B, H*V] TILE =
                rms_norm(o) * norm_w * silu(z[:, :, z_col_offset + h*V ...]).
            norm_w (ttnn.Tensor, optional): [1, 1, V] TILE.
            z_col_offset (int): column of head 0 in z (multiple of 32). epsilon (float): 1e-6.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: o [B,1,H,V] ROW_MAJOR (or gated [1,B,H*V] TILE),
            new_state [B,H,K,V] TILE."""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def factory_cpp(s):
    old = """constexpr uint32_t scratch = tt::CBIndex::c_27;  // max(Kt,Vt) io staging pages (full-page DMA)
}  // namespace cbd"""
    new = """constexpr uint32_t scratch = tt::CBIndex::c_27;  // max(Kt,Vt) io staging pages (full-page DMA); fused: the assembly buffer
// Fused norm + gate (fuse_ng): the two free indices hold norm_w (io) and its fp32 copy; the
// chain's other operands reuse rings that are dead once o exists (see the compute kernel).
constexpr uint32_t w = tt::CBIndex::c_30;   // [Vt] io   norm_w row-0 tiles (read once per core)
constexpr uint32_t wf = tt::CBIndex::c_31;  // [Vt] fp32 norm_w (persistent front)
}  // namespace cbd"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    const uint32_t ncores = grid.x * grid.y;
    const uint32_t per_core = (BH + ncores - 1) / ncores;"""
    new = """    const uint32_t ncores = grid.x * grid.y;
    const uint32_t per_core = (BH + ncores - 1) / ncores;
    const uint32_t fng = attrs.fuse_ng ? 1u : 0u;
    const uint32_t KVt = std::max(Kt, Vt);"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    add_cb(cbd::q, Kt, 1, df_io);
    add_cb(cbd::k, Kt, 1, df_io);
    add_cb(cbd::v, Vt, 1, df_io);"""
    new = """    add_cb(cbd::q, Kt, 1, df_io);
    add_cb(cbd::k, Kt, 1, df_io);
    add_cb(cbd::v, Vt, fng ? 2 : 1, df_io);  // fused: z rides in this ring after v"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    add_cb(cbd::qsq, Kt, 1, tt::DataFormat::Float32);"""
    new = """    add_cb(cbd::qsq, fng ? KVt : Kt, 1, tt::DataFormat::Float32);  // fused: reused for o*o"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    add_cb(cbd::qn, Kt, 1, tt::DataFormat::Float32);
    add_cb(cbd::kn, Kt, 1, tt::DataFormat::Float32);"""
    new = """    add_cb(cbd::qn, fng ? KVt : Kt, 1, tt::DataFormat::Float32);  // fused: reused for o*factor
    add_cb(cbd::kn, fng ? KVt : Kt, 1, tt::DataFormat::Float32);  // fused: reused for (o*factor)*w"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    add_cb(cbd::scratch, 2, 1, df_io);
"""
    new = """    add_cb(cbd::scratch, fng ? per_core * Vt : 2, 1, df_io);  // fused: per_core assembly slots of Vt tiles
    if (fng) {
        add_cb(cbd::w, Vt, 1, df_io);
        add_cb(cbd::wf, Vt, 1, tt::DataFormat::Float32);
        // one semaphore per assembly slot, on every core (ids 0..per_core-1)
        for (uint32_t sidx = 0; sidx < per_core; sidx++) {
            desc.semaphores.push_back(SemaphoreDescriptor{.id = sidx, .core_ranges = cores, .initial_value = 0});
        }
    }
"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    const std::vector<uint32_t> ct_args = {Kt, Vt, has_s0, eps_bits, scale_bits};"""
    new = """    // Compute: {Kt, Vt, has_s0, eps, scale, FNG, NG_EPS, NG_SCALE}
    const std::vector<uint32_t> ct_args = {
        Kt, Vt, has_s0, eps_bits, scale_bits, fng, attrs.ng_eps_bits, attrs.ng_scale_bits};"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """        attrs.packed ? 1u : 0u,
        attrs.Ct,
        attrs.q_off_t,
        attrs.k_off_t,
        attrs.v_off_t,
        attrs.rf,
        attrs.Nvt};"""
    new = """        attrs.packed ? 1u : 0u,
        attrs.Ct,
        attrs.q_off_t,
        attrs.k_off_t,
        attrs.v_off_t,
        attrs.rf,
        attrs.Nvt,
        fng,
        attrs.Wt_z,
        attrs.z_off_t};"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {Kt, Vt};"""
    new = """    TensorAccessorArgs(in.initial_state.has_value() ? in.initial_state->buffer() : nullptr).append_to(reader_ct);
    TensorAccessorArgs(in.z.has_value() ? in.z->buffer() : nullptr).append_to(reader_ct);
    TensorAccessorArgs(in.norm_w.has_value() ? in.norm_w->buffer() : nullptr).append_to(reader_ct);

    // Writer: {Kt, Vt, FNG, per_core, H, B} + accessors (o-or-gated, new_state)
    std::vector<uint32_t> writer_ct = {Kt, Vt, fng, per_core, attrs.H, attrs.B};"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    auto* o_buf = outputs[0].buffer();
    auto* s1_buf = outputs[1].buffer();
"""
    new = """    auto* o_buf = outputs[0].buffer();
    auto* s1_buf = outputs[1].buffer();
    auto* z_buf = in.z.has_value() ? in.z->buffer() : nullptr;
    auto* w_buf = in.norm_w.has_value() ? in.norm_w->buffer() : nullptr;
"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """        reader.emplace_runtime_args(core, {start, n_inst, q_buf, k_buf, v_buf, beta_buf, g_buf, s0_buf});
        // o is ROW_MAJOR: pass its stick page size (page bh == head bh's row).
        writer.emplace_runtime_args(core, {start, n_inst, o_buf, static_cast<uint32_t>(o_buf->page_size()), s1_buf});
        compute.emplace_runtime_args(core, {n_inst});"""
    new = """        using RtArg = std::variant<uint32_t, Buffer*>;
        std::vector<RtArg> rargs = {start, n_inst, q_buf, k_buf, v_buf, beta_buf, g_buf, s0_buf};
        if (fng) {
            rargs.emplace_back(z_buf);
            rargs.emplace_back(w_buf);
        }
        reader.emplace_runtime_args(core, rargs);
        // o is ROW_MAJOR: pass its stick page size (page bh == head bh's row). In fused mode
        // output 0 is the TILE gated tensor and the page is a tile.
        std::vector<RtArg> wargs = {start, n_inst, o_buf, static_cast<uint32_t>(o_buf->page_size()), s1_buf};
        if (fng) {
            // per instance: the physical NOC coordinates of its assembler core -- the core that
            // owns instance ((b - b%32) * H + h), the row-0 head of the same batch tile.
            for (uint32_t i = 0; i < n_inst; i++) {
                const uint32_t bh = start + i;
                const uint32_t b = bh / attrs.H;
                const uint32_t h = bh % attrs.H;
                const uint32_t bh0 = (b - (b % 32)) * attrs.H + h;
                const uint32_t ac = bh0 / per_core;
                const CoreCoord phys = device->worker_core_from_logical_core(active_cores[ac]);
                wargs.emplace_back(static_cast<uint32_t>(phys.x));
                wargs.emplace_back(static_cast<uint32_t>(phys.y));
            }
        }
        writer.emplace_runtime_args(core, wargs);
        compute.emplace_runtime_args(core, {n_inst});"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    if "#include <algorithm>" not in s:
        s = s.replace("#include <bit>\n", "#include <algorithm>\n#include <bit>\n", 1)
    if "#include <variant>" not in s:
        s = s.replace("#include <set>\n", "#include <set>\n#include <variant>\n", 1)
    return s


def reader_cpp(s):
    old = """    constexpr uint32_t NVT = get_compile_time_arg_val(12);

    constexpr auto q_a = TensorAccessorArgs<13>();"""
    new = """    constexpr uint32_t NVT = get_compile_time_arg_val(12);
    // Fused norm + gate: z's row for head (b,h) (same gather as v's, in the v ring) and norm_w
    // (row-0 tiles, once per core).
    constexpr uint32_t FNG = get_compile_time_arg_val(13);
    constexpr uint32_t WTZ = get_compile_time_arg_val(14);
    constexpr uint32_t ZOT = get_compile_time_arg_val(15);

    constexpr auto q_a = TensorAccessorArgs<16>();"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    constexpr auto s0_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
"""
    new = """    constexpr auto s0_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto z_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<z_a.next_compile_time_args_offset()>();
"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    const uint32_t s0_addr = get_arg_val<uint32_t>(7);
"""
    new = """    const uint32_t s0_addr = get_arg_val<uint32_t>(7);
    const uint32_t z_addr = FNG ? get_arg_val<uint32_t>(8) : 0;
    const uint32_t w_addr = FNG ? get_arg_val<uint32_t>(9) : 0;
"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb_io);
"""
    new = """    const auto s0_acc = TensorAccessor(s0_a, s0_addr, tb_io);
    const auto z_acc = TensorAccessor(z_a, z_addr, tb_io);
    const auto w_acc = TensorAccessor(w_a, w_addr, tb_io);
    constexpr uint32_t cb_w = 30;
"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    // Per-instance loop: this core owns the contiguous chunk
    // [bh_start, bh_start + n_inst) of head-instances.
    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {"""
    new = """    if constexpr (FNG) {
        // norm_w [1,1,V]: pages 0..Vt-1, row 0 -- gathered once, held for the whole kernel.
        gather_row(w_acc, cb_w, Vt, 0, 0);
    }

    // Per-instance loop: this core owns the contiguous chunk
    // [bh_start, bh_start + n_inst) of head-instances.
    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """        } else {
            zero(cbs.get_write_ptr(), kv * tb_io / 4);
        }
        cbs.push_back(kv);
    }
}"""
    new = """        } else {
            zero(cbs.get_write_ptr(), kv * tb_io / 4);
        }
        cbs.push_back(kv);

        if constexpr (FNG) {
            // z's row for head (b,h): [1,Bz,W] TILE, row b, tile column ZOT + h*Vt -- the same
            // gather as v's, into the v ring (the compute has consumed v by the time it needs z).
            gather_row(z_acc, cb_v, Vt, (b / 32) * WTZ + ZOT + h * Vt, b % 32);
        }
    }
}"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def compute_cpp(s):
    old = """    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);
    (void)has_s0;
"""
    new = """    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(4);
    // Fused norm + gate: gated = rms_norm(o) * norm_w * silu(z), on this head's row-0 tiles,
    // packed to cb_out in place of o. Operand rings reused (all dead once o exists):
    //   of = vread, sq = qsq, sum = sc, fac = sc2, xn = qn, xw = kn, zf = vf, zs = delta,
    //   z arrives in cb_v after v; norm_w in cb_w (30) -> fp32 cb_wf (31), once per core.
    constexpr uint32_t FNG = get_compile_time_arg_val(5);
    constexpr uint32_t NG_EPS_BITS = get_compile_time_arg_val(6);
    constexpr uint32_t NG_SCALE_BITS = get_compile_time_arg_val(7);
    constexpr uint32_t cb_w = 30, cb_wf = 31;
    (void)has_s0;
"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    for (uint32_t it = 0; it < n_inst; ++it) {"""
    new = """    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    if constexpr (FNG) {
        WAIT(cb_w, Vt);
        copy_tiles(cb_w, cb_wf, Vt);
        POP(cb_w, Vt);
        WAIT(cb_wf, Vt);  // persistent front for the whole kernel
    }

    for (uint32_t it = 0; it < n_inst; ++it) {"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """        // ---- o = qn @ new_h (fp32 x fp32; packed straight to io) ----
        mm(cb_qn, cb_snew, cb_out, 1, Kt, Vt, false);
        POP(cb_qn, Kt);
"""
    new = """        if constexpr (FNG) {
            // ---- o (fp32) -> rms_norm * norm_w * silu(z) -> cb_out (io) ----
            mm(cb_qn, cb_snew, cb_vread, 1, Kt, Vt, false);  // of = o, fp32
            POP(cb_qn, Kt);
            WAIT(cb_vread, Vt);
            ew(cb_vread, cb_vread, cb_qsq, Vt, 2);  // o^2
            WAIT(cb_qsq, Vt);
            rowsum_k(cb_qsq, cb_sc, Vt);  // [0,*] = sum o^2
            WAIT(cb_sc, 1);
            POP(cb_qsq, Vt);
            inv_rms(cb_sc, cb_sc2, NG_EPS_BITS, NG_SCALE_BITS, true);  // sqrt(V)/sqrt(sum + V*eps)
            POP(cb_sc, 1);
            WAIT(cb_sc2, 1);
            bcast_cols_mul(cb_vread, cb_sc2, cb_qn, 1, Vt);  // xn = o * factor
            WAIT(cb_qn, Vt);
            POP(cb_vread, Vt);
            POP(cb_sc2, 1);
            ew(cb_qn, cb_wf, cb_kn, Vt, 2);  // xw = xn * norm_w  (row-0 tiles: plain elementwise)
            WAIT(cb_kn, Vt);
            POP(cb_qn, Vt);
            WAIT(cb_v, Vt);  // z (gathered after v)
            copy_tiles(cb_v, cb_vf, Vt);  // zf fp32
            POP(cb_v, Vt);
            WAIT(cb_vf, Vt);
            silu_tiles(cb_vf, cb_delta, Vt);  // zs = silu(z)
            WAIT(cb_delta, Vt);
            POP(cb_vf, Vt);
            ew(cb_kn, cb_delta, cb_out, Vt, 2);  // gated = xw * zs -> io
            POP(cb_kn, Vt);
            POP(cb_delta, Vt);
        } else {
            // ---- o = qn @ new_h (fp32 x fp32; packed straight to io) ----
            mm(cb_qn, cb_snew, cb_out, 1, Kt, Vt, false);
            POP(cb_qn, Kt);
        }
"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """// out = S * scalar[0,0], n tiles.
void bcast_scalar_mul("""
    new = """// out = silu(in), n tiles (copy to fp32 DST, SFPU silu).
void silu_tiles(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    silu_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        silu_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = S * scalar[0,0], n tiles.
void bcast_scalar_mul("""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    if '#include "api/compute/compute_kernel_api.h"' not in s:
        s = s.replace('#include "api/compute/reconfig_data_format.h"\n', '#include "api/compute/reconfig_data_format.h"\n#include "api/compute/compute_kernel_api.h"\n', 1)
    return s


def writer_cpp(s):
    old = """    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);

    constexpr auto o_a = TensorAccessorArgs<2>();"""
    new = """    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    // Fused norm + gate: each head's gated row is scattered by NOC into the L1 assembly slot
    // of its (batch-tile, head) assembler core and counted on that core's semaphore; the
    // assembler writes the Vt full output tiles once all rows are in.
    constexpr uint32_t FNG = get_compile_time_arg_val(2);
    constexpr uint32_t PER_CORE = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);

    constexpr auto o_a = TensorAccessorArgs<6>();"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    Noc noc;
    uint32_t stage = 0;  // staged o stick base (state write below is untouched)
"""
    new = """    Noc noc;
    uint32_t stage = 0;  // staged o stick base (state write below is untouched)

    // Fused mode: the assembly buffer (cb_scratch: PER_CORE slots of Vt tiles) has the same L1
    // address on every core, so a remote slot is base + slot*Vt*tb on the assembler's L1.
    const uint32_t asm_base = FNG ? CircularBuffer(cb_scratch).get_write_ptr() : 0;
"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    // Per-instance loop: this core owns [bh_start, bh_start + n_inst).
    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
        // o: stage head bh's [V] stick in scratch, then ONE full-page write to
        // o page bh via the o accessor's page_id form (the same write shape the
        // new-state write below uses and that ttsim/silicon both land). Tile t's
        // row 0 holds o cols 32t..32t+31: face 0 (cols 0-15) at element offset 0,
        // face 1 (cols 16-31) at element offset 256.
        {
"""
    assert s.count(old) == 1, "loop head anchor"
    new = """    if constexpr (FNG) {
        // Rows at or beyond B of my own assembly slots stay zero: nobody writes them.
        for (uint32_t s = 0; s < n_inst; s++) {
            const uint32_t bh = bh_start + s;
            const uint32_t b = bh / H;
            if (b % 32 != 0) {
                continue;  // not an assembler slot
            }
            const uint32_t rows = (B - b < 32) ? (B - b) : 32;
            for (uint32_t r = rows; r < 32; r++) {
                const uint32_t e0 = ((r / 16) * 2) * 256 + (r % 16) * 16;
                for (uint32_t t = 0; t < Vt; t++) {
                    const uint32_t tile = asm_base + (s * Vt + t) * tb_io;
                    zero(tile + e0 * elem, 16 * elem / 4);
                    zero(tile + (e0 + 256) * elem, 16 * elem / 4);
                }
            }
        }
    }

    // Per-instance loop: this core owns [bh_start, bh_start + n_inst).
    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
        if constexpr (FNG) {
            // gated row (row 0 of Vt tiles in cb_out) -> row r of the assembler's slot tiles.
            CircularBuffer cb(cb_out);
            cb.wait_front(Vt);
            const uint32_t src = cb.get_read_ptr();
            const uint32_t b = bh / H;
            const uint32_t h = bh % H;
            const uint32_t r = b % 32;
            const uint32_t bh0 = (b - r) * H + h;
            const uint32_t slot = bh0 % PER_CORE;
            const uint32_t i = bh - bh_start;
            const uint32_t ax = get_arg_val<uint32_t>(5 + 2 * i);
            const uint32_t ay = get_arg_val<uint32_t>(6 + 2 * i);
            const uint32_t e0 = ((r / 16) * 2) * 256 + (r % 16) * 16;
            const uint32_t chunk = 16 * elem;
            for (uint32_t t = 0; t < Vt; t++) {
                const uint32_t dst_tile = asm_base + (slot * Vt + t) * tb_io;
                noc_async_write(src + t * tb_io, get_noc_addr(ax, ay, dst_tile + e0 * elem), chunk);
                noc_async_write(src + t * tb_io + 256 * elem, get_noc_addr(ax, ay, dst_tile + (e0 + 256) * elem), chunk);
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(ax, ay, get_semaphore(slot)), 1);
            cb.pop_front(Vt);
        } else {
        // o: stage head bh's [V] stick in scratch, then ONE full-page write to
        // o page bh via the o accessor's page_id form (the same write shape the
        // new-state write below uses and that ttsim/silicon both land). Tile t's
        // row 0 holds o cols 32t..32t+31: face 0 (cols 0-15) at element offset 0,
        // face 1 (cols 16-31) at element offset 256.
        {
"""
    s = s.replace(old, new, 1)
    old = """            noc.async_write(CoreLocalMem<uint32_t>(stage), o_acc, o_page, {}, {.page_id = bh});
            noc.async_write_barrier();
            scb.pop_front(1);
            cb.pop_front(Vt);
        }
"""
    assert s.count(old) == 1
    new = old + "        }  // !FNG\n"
    s = s.replace(old, new, 1)
    old = """            noc.async_write_barrier();
            cb.pop_front(kv);
        }
    }  // per-instance loop
}"""
    assert s.count(old) == 1
    new = """            noc.async_write_barrier();
            cb.pop_front(kv);
        }
    }  // per-instance loop

    if constexpr (FNG) {
        // Assembler duty: for each of my slots that is the row-0 head of a batch tile, wait for
        // that tile's rows, write the Vt full output tiles of gated [1,B,H*Vt*32], reset.
        for (uint32_t s = 0; s < n_inst; s++) {
            const uint32_t bh = bh_start + s;
            const uint32_t b = bh / H;
            if (b % 32 != 0) {
                continue;
            }
            const uint32_t h = bh % H;
            const uint32_t bt = b / 32;
            const uint32_t rows = (B - b < 32) ? (B - b) : 32;
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(s));
            noc_semaphore_wait(sem, rows);
            asm volatile("" ::: "memory");
            for (uint32_t t = 0; t < Vt; t++) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(asm_base + (s * Vt + t) * tb_io),
                    o_acc,
                    o_page,
                    {},
                    {.page_id = bt * (H * Vt) + h * Vt + t});
            }
            noc.async_write_barrier();
            noc_semaphore_set(sem, 0);
        }
    }
}"""
    return s.replace(old, new, 1)


if __name__ == "__main__":
    marker = os.path.join(ROOT, "device", "decode_gated_delta_rule_device_operation_types.hpp")
    if "fuse_ng" in io.open(marker, encoding="utf-8").read():
        print("already applied")
        sys.exit(0)
    rw("device/decode_gated_delta_rule_device_operation_types.hpp", types_hpp)
    rw("device/decode_gated_delta_rule_device_operation.hpp", devop_hpp)
    rw("device/decode_gated_delta_rule_device_operation.cpp", devop_cpp)
    rw("decode_gated_delta_rule.hpp", api_hpp)
    rw("decode_gated_delta_rule.cpp", api_cpp)
    rw("decode_gated_delta_rule_nanobind.cpp", nanobind_cpp)
    rw("device/decode_gated_delta_rule_program_factory.cpp", factory_cpp)
    rw("device/kernels/dataflow/reader_decode_gated_delta_rule.cpp", reader_cpp)
    rw("device/kernels/compute/decode_gated_delta_rule.cpp", compute_cpp)
    rw("device/kernels/dataflow/writer_decode_gated_delta_rule.cpp", writer_cpp)
    print("all patched")
