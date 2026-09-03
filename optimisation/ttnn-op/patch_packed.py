"""Apply the packed-layout (K milestone 2) changes to a clean copy of decode_gated_delta_rule.
Run from the directory holding decode_gated_delta_rule/. Idempotence: refuses if already applied."""
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
    old = """    float scale;         // folded into q's L2-norm factor (K**-0.5 by default)
    tt::tt_metal::MemoryConfig output_mem_config;
};"""
    new = """    float scale;         // folded into q's L2-norm factor (K**-0.5 by default)
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
};"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def devop_hpp(s):
    old = """    float scale,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim"""
    new = """    float scale,
    const tt::tt_metal::MemoryConfig& output_mem_config);

// Packed-layout dispatch: qkv [1,B,C] (= q|k|v head-major), beta/g [1,B,H].
std::vector<Tensor> decode_gated_delta_rule_packed(
    const Tensor& qkv,
    const Tensor& beta,
    const Tensor& g,
    const std::optional<Tensor>& initial_state,
    bool inplace_state,
    float scale,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    uint32_t head_k,
    uint32_t head_v,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def devop_cpp(s):
    old = """    check(in.q, "q");
    check(in.k, "k");
    check(in.v, "v");
    check(in.beta, "beta");
    check(in.g, "g");
    TT_FATAL(in.q.logical_shape()[1] == 1, "decode_gated_delta_rule: q must be T=1 [B,1,H,K]");
    TT_FATAL(in.v.logical_shape()[2] == attrs.H, "decode_gated_delta_rule: v heads must equal q heads (no GQA)");
    TT_FATAL(attrs.K % TILE_WIDTH == 0, "decode_gated_delta_rule: K must be a multiple of 32");
    TT_FATAL(attrs.V % TILE_WIDTH == 0, "decode_gated_delta_rule: V must be a multiple of 32");"""
    new = """    check(in.q, "q");
    check(in.k, "k");
    check(in.v, "v");
    check(in.beta, "beta");
    check(in.g, "g");
    TT_FATAL(attrs.K % TILE_WIDTH == 0, "decode_gated_delta_rule: K must be a multiple of 32");
    TT_FATAL(attrs.V % TILE_WIDTH == 0, "decode_gated_delta_rule: V must be a multiple of 32");
    if (attrs.packed) {
        const auto& qs = in.q.logical_shape();  // [1,B,C]
        TT_FATAL(
            qs.rank() == 3 && qs[0] == 1 && qs[1] == attrs.B && qs[2] == attrs.Ct * TILE_WIDTH,
            "decode_gated_delta_rule (packed): qkv must be [1,B,C] with C == Ct*32");
        TT_FATAL(
            attrs.v_off_t * TILE_WIDTH + attrs.H * attrs.V == attrs.Ct * TILE_WIDTH,
            "decode_gated_delta_rule (packed): C must equal 2*Nk*K + H*V");
        TT_FATAL(
            attrs.rf >= 1 && attrs.H % attrs.rf == 0, "decode_gated_delta_rule (packed): H must be a multiple of rf");
        const auto& bs = in.beta.logical_shape();
        const auto& gs = in.g.logical_shape();
        TT_FATAL(
            bs.rank() == 3 && bs[0] == 1 && bs[1] == attrs.B && bs[2] == attrs.H && gs == bs,
            "decode_gated_delta_rule (packed): beta and g must be [1,B,H]");
    } else {
        TT_FATAL(in.q.logical_shape()[1] == 1, "decode_gated_delta_rule: q must be T=1 [B,1,H,K]");
        TT_FATAL(in.v.logical_shape()[2] == attrs.H, "decode_gated_delta_rule: v heads must equal q heads (no GQA)");
    }"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old2 = """    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim"""
    new2 = """    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

std::vector<Tensor> decode_gated_delta_rule_packed(
    const Tensor& qkv,
    const Tensor& beta,
    const Tensor& g,
    const std::optional<Tensor>& initial_state,
    bool inplace_state,
    float scale,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    uint32_t head_k,
    uint32_t head_v,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = DecodeGatedDeltaRuleDeviceOperation;
    using namespace tt::constants;
    const auto& qs = qkv.logical_shape();  // [1,B,C]
    TT_FATAL(qs.rank() == 3, "decode_gated_delta_rule (packed): qkv must be rank 3 [1,B,C]");
    TT_FATAL(
        num_k_heads >= 1 && num_v_heads % num_k_heads == 0, "decode_gated_delta_rule (packed): H must be a multiple of Nk");
    TT_FATAL(
        (num_k_heads * head_k) % TILE_WIDTH == 0, "decode_gated_delta_rule (packed): Nk*K must be a multiple of 32");
    const uint32_t B = qs[1];
    auto attrs = OperationType::operation_attributes_t{
        .B = B,
        .H = num_v_heads,
        .BH = B * num_v_heads,
        .K = head_k,
        .V = head_v,
        .has_initial_state = initial_state.has_value(),
        .inplace_state = inplace_state,
        .scale = scale,
        .output_mem_config = output_mem_config,
        .packed = true,
        .Ct = static_cast<uint32_t>(qs[2]) / TILE_WIDTH,
        .q_off_t = 0,
        .k_off_t = (num_k_heads * head_k) / TILE_WIDTH,
        .v_off_t = (2 * num_k_heads * head_k) / TILE_WIDTH,
        .rf = num_v_heads / num_k_heads,
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
}

}  // namespace ttnn::prim"""
    assert s.count(old2) == 1
    return s.replace(old2, new2, 1)


def factory_cpp(s):
    old = """    std::vector<uint32_t> reader_ct = {Kt, Vt, has_s0, eps_bits, scale_bits, attrs.H};"""
    new = """    // Packed geometry follows H (all zero/unused when !packed); accessor args start at index 13.
    std::vector<uint32_t> reader_ct = {
        Kt,
        Vt,
        has_s0,
        eps_bits,
        scale_bits,
        attrs.H,
        attrs.packed ? 1u : 0u,
        attrs.Ct,
        attrs.q_off_t,
        attrs.k_off_t,
        attrs.v_off_t,
        attrs.rf,
        attrs.Nvt};"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    s = s.replace(
        "    // Reader compile args: {Kt,Vt,has_s0,eps,scale,H} + TensorAccessorArgs per",
        "    // Reader compile args: {Kt,Vt,has_s0,eps,scale,H, packed,Ct,qot,kot,vot,rf,Nvt} + TensorAccessorArgs per",
        1,
    )
    return s


def api_hpp(s):
    old = """    bool inplace_state = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::transformer"""
    new = """    bool inplace_state = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

/**
 * Packed-layout variant (K's milestone 2). Takes the conv+gates kernel's outputs directly:
 *   qkv      [1, B, C]  TILE, channels [q | k | v] head-major (C = 2*Nk*K + H*V)
 *   beta, g  [1, B, H]  TILE
 * and reads head (b,h)'s q/k from GQA source head h / (H/Nk). Same math, same outputs as
 * decode_gated_delta_rule; no host-side slicing, reshaping or head expansion.
 */
std::tuple<ttnn::Tensor, ttnn::Tensor> decode_gated_delta_rule_packed(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    uint32_t head_k,
    uint32_t head_v,
    std::optional<float> scale = std::nullopt,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    bool inplace_state = false,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::transformer"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def api_cpp(s):
    old = """    return {results[0], results[1]};
}

}  // namespace ttnn::transformer"""
    new = """    return {results[0], results[1]};
}

std::tuple<ttnn::Tensor, ttnn::Tensor> decode_gated_delta_rule_packed(
    const ttnn::Tensor& qkv,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& g,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    uint32_t head_k,
    uint32_t head_v,
    std::optional<float> scale,
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
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
    return {results[0], results[1]};
}

}  // namespace ttnn::transformer"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def nanobind_cpp(s):
    old = """        nb::arg("inplace_state") = false,
        nb::arg("memory_config") = nb::none());
}"""
    new = """        nb::arg("inplace_state") = false,
        nb::arg("memory_config") = nb::none());

    const auto* doc_packed =
        R"doc(
        Packed-layout variant of decode_gated_delta_rule: takes the conv+gates kernel's
        outputs directly (qkv [1,B,C] TILE with channels [q|k|v] head-major, beta/g [1,B,H]
        TILE) and gathers each head's rows in the reader, GQA included. Same outputs.

        Args:
            qkv (ttnn.Tensor):  [1, B, C] TILE, C = 2*num_k_heads*head_k + num_v_heads*head_v
            beta, g (ttnn.Tensor): [1, B, num_v_heads] TILE
            num_k_heads, num_v_heads, head_k, head_v (int)

        Keyword Args:
            scale, initial_state, inplace_state, memory_config: as decode_gated_delta_rule.

        Returns:
            tuple[ttnn.Tensor, ttnn.Tensor]: o [B,1,H,V] ROW_MAJOR, new_state [B,H,K,V] TILE.
        )doc";

    ttnn::bind_function<"decode_gated_delta_rule_packed", "ttnn.transformer.">(
        mod,
        doc_packed,
        &ttnn::transformer::decode_gated_delta_rule_packed,
        nb::arg("qkv").noconvert(),
        nb::arg("beta").noconvert(),
        nb::arg("g").noconvert(),
        nb::arg("num_k_heads"),
        nb::arg("num_v_heads"),
        nb::arg("head_k"),
        nb::arg("head_v"),
        nb::kw_only(),
        nb::arg("scale") = nb::none(),
        nb::arg("initial_state") = nb::none(),
        nb::arg("inplace_state") = false,
        nb::arg("memory_config") = nb::none());
}"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def reader_cpp(s):
    old = """    constexpr uint32_t H = get_compile_time_arg_val(5);  // heads (bh = b*H + h)

    constexpr auto q_a = TensorAccessorArgs<6>();"""
    new = """    constexpr uint32_t H = get_compile_time_arg_val(5);  // heads (bh = b*H + h)
    // Packed mode (K's milestone 2): q/k/v are one [1,B,C] tensor (channels q|k|v, head-major),
    // beta/g are [1,B,H]; head (b,h) reads row b, tile column off + head*Dt, q/k from GQA
    // source head h / RF. All zero when !PACKED (the original [B,1,H,*] gather runs instead).
    constexpr uint32_t PACKED = get_compile_time_arg_val(6);
    constexpr uint32_t Ct = get_compile_time_arg_val(7);
    constexpr uint32_t QOT = get_compile_time_arg_val(8);
    constexpr uint32_t KOT = get_compile_time_arg_val(9);
    constexpr uint32_t VOT = get_compile_time_arg_val(10);
    constexpr uint32_t RF = get_compile_time_arg_val(11);
    constexpr uint32_t NVT = get_compile_time_arg_val(12);

    constexpr auto q_a = TensorAccessorArgs<13>();"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    auto gather_row = [&](const auto& acc, uint32_t cb_id, uint32_t n_tiles, uint32_t physrow) {
        const uint32_t first_page = (physrow / 32) * n_tiles;
        const uint32_t r = physrow % 32;"""
    new = """    auto gather_row = [&](const auto& acc, uint32_t cb_id, uint32_t n_tiles, uint32_t first_page, uint32_t r) {"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    auto gather_scalar = [&](const auto& acc, uint32_t cb_id, uint32_t bh) {
        const uint32_t b = bh / H;
        const uint32_t hcol = bh % H;
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        zero(base, tb_io / 4);
        noc.async_read(acc, cb, tb_io, {.page_id = b * ppb + hcol / 32}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t cc = hcol % 32;
        const uint32_t eoff = (cc / 16) * 256 + (cc % 16);"""
    new = """    auto gather_scalar = [&](const auto& acc, uint32_t cb_id, uint32_t page, uint32_t r, uint32_t cc) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        zero(base, tb_io / 4);
        noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t eoff = ((r / 16) * 2 + (cc / 16)) * 256 + (r % 16) * 16 + (cc % 16);"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """        // Head (b,h)'s physical row in the [B,1,H,D] TILE tensors.
        const uint32_t physrow = (bh / H) * padH + (bh % H);

        gather_row(q_acc, cb_q, Kt, physrow);
        gather_row(k_acc, cb_k, Kt, physrow);
        gather_row(v_acc, cb_v, Vt, physrow);
        gather_scalar(beta_acc, cb_beta, bh);
        gather_scalar(g_acc, cb_g, bh);"""
    new = """        const uint32_t b = bh / H;
        const uint32_t h = bh % H;
        if constexpr (PACKED) {
            // [1,B,C] TILE: row b of tile-row b/32; q/k from GQA source head h/RF.
            const uint32_t row_page0 = (b / 32) * Ct;
            const uint32_t r = b % 32;
            const uint32_t hq = h / RF;
            gather_row(q_acc, cb_q, Kt, row_page0 + QOT + hq * Kt, r);
            gather_row(k_acc, cb_k, Kt, row_page0 + KOT + hq * Kt, r);
            gather_row(v_acc, cb_v, Vt, row_page0 + VOT + h * Vt, r);
            // beta/g [1,B,H] TILE: page (b/32)*NVT + h/32, row b%32, col h%32.
            gather_scalar(beta_acc, cb_beta, (b / 32) * NVT + h / 32, r, h % 32);
            gather_scalar(g_acc, cb_g, (b / 32) * NVT + h / 32, r, h % 32);
        } else {
            // Head (b,h)'s physical row in the [B,1,H,D] TILE tensors.
            const uint32_t physrow = b * padH + h;
            gather_row(q_acc, cb_q, Kt, (physrow / 32) * Kt, physrow % 32);
            gather_row(k_acc, cb_k, Kt, (physrow / 32) * Kt, physrow % 32);
            gather_row(v_acc, cb_v, Vt, (physrow / 32) * Vt, physrow % 32);
            // beta/g [B,1,H]: (b,hcol) at page b*ppb + hcol/32, row 0, col hcol%32.
            gather_scalar(beta_acc, cb_beta, b * ppb + h / 32, 0, h % 32);
            gather_scalar(g_acc, cb_g, b * ppb + h / 32, 0, h % 32);
        }"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    s = s.replace(
        "// Compile args: {Kt, Vt, has_s0, eps_bits, scale_bits, H} + accessor args per",
        "// Compile args: {Kt, Vt, has_s0, eps_bits, scale_bits, H, PACKED, Ct, QOT, KOT, VOT, RF, NVT} + accessor args per",
        1,
    )
    return s


if __name__ == "__main__":
    marker = os.path.join(ROOT, "device", "decode_gated_delta_rule_device_operation_types.hpp")
    if "packed" in io.open(marker, encoding="utf-8").read():
        print("already applied")
        sys.exit(0)
    rw("device/decode_gated_delta_rule_device_operation_types.hpp", types_hpp)
    rw("device/decode_gated_delta_rule_device_operation.hpp", devop_hpp)
    rw("device/decode_gated_delta_rule_device_operation.cpp", devop_cpp)
    rw("device/decode_gated_delta_rule_program_factory.cpp", factory_cpp)
    rw("decode_gated_delta_rule.hpp", api_hpp)
    rw("decode_gated_delta_rule.cpp", api_cpp)
    rw("decode_gated_delta_rule_nanobind.cpp", nanobind_cpp)
    rw("device/kernels/dataflow/reader_decode_gated_delta_rule.cpp", reader_cpp)
    print("all patched")
