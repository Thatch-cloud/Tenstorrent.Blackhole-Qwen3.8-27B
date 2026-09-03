"""K milestone 4: gdn_decode_conv_gates reads the fused projection output directly.

x may be wider than the conv channels (channels= gives C; pages index by x's real width) and
a / b may be column windows of a wider tensor at any element offset (a_col=, b_col=): the
gate reader gathers the Nv columns per tile from the one or two source tiles that hold them.
Removes the qkv / ab / a / b slices from the model's decode path (z still needs one).
Idempotent; run from kwork/.
"""
import io
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdn_conv_gates")


def rw(rel, fn):
    p = os.path.join(ROOT, rel)
    s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    s2 = fn(s)
    assert s2 != s, p
    io.open(p, "w", encoding="utf-8", newline="\n").write(s2)
    print("patched", rel)


def types_hpp(s):
    old = """    uint32_t K;     // conv taps (== conv_states.size() == taps.size())
    tt::tt_metal::MemoryConfig output_mem_config;
};"""
    new = """    uint32_t K;     // conv taps (== conv_states.size() == taps.size())
    tt::tt_metal::MemoryConfig output_mem_config;
    // Direct read of the fused projection output (milestone 4): x may be wider than C (pages
    // index by Wt = its real width in tiles) and a / b may be column windows of a wider
    // tensor at any element offset -- the gate reader gathers them per element.
    uint32_t Wt = 0;      // x's width in tiles (== C/32 when x is exactly the conv channels)
    bool gates_packed = false;
    uint32_t AWt = 0;     // a's width in tiles
    uint32_t a_col = 0;   // element column of head 0 in a
    uint32_t BWt = 0;     // b's width in tiles
    uint32_t b_col = 0;   // element column of head 0 in b
};"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def api_hpp(s):
    old = """    std::optional<uint32_t> batch = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);"""
    new = """    std::optional<uint32_t> batch = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    std::optional<uint32_t> channels = std::nullopt,
    uint32_t a_col = 0,
    uint32_t b_col = 0);"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """ * Rows of x at or beyond `batch` (default: x's row count) are treated as zero when they"""
    new = """ * Direct read (milestone 4): x may be the whole fused projection output [1, B, W] with
 * `channels` = C (the conv reads its first C columns), and a / b may be that same tensor
 * with `a_col` / `b_col` naming the element column where head 0 sits -- no slices needed.
 *
 * Rows of x at or beyond `batch` (default: x's row count) are treated as zero when they"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def api_cpp(s):
    old = """    std::optional<uint32_t> batch,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const uint32_t B = batch.value_or(x.logical_shape()[1]);
    auto results = ttnn::prim::gdn_conv_gates(
        x, conv_states, taps, a, b, dt_bias, neg_exp_A, B, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));"""
    new = """    std::optional<uint32_t> batch,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<uint32_t> channels,
    uint32_t a_col,
    uint32_t b_col) {
    const uint32_t B = batch.value_or(x.logical_shape()[1]);
    const uint32_t C = channels.value_or(x.logical_shape()[2]);
    auto results = ttnn::prim::gdn_conv_gates(
        x, conv_states, taps, a, b, dt_bias, neg_exp_A, B, C, a_col, b_col,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def nanobind_cpp(s):
    old = """        nb::arg("batch") = nb::none(),
        nb::arg("memory_config") = nb::none());"""
    new = """        nb::arg("batch") = nb::none(),
        nb::arg("memory_config") = nb::none(),
        nb::arg("channels") = nb::none(),
        nb::arg("a_col") = 0,
        nb::arg("b_col") = 0);"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """            memory_config (ttnn.MemoryConfig, optional): placement of the three outputs."""
    new = """            memory_config (ttnn.MemoryConfig, optional): placement of the three outputs.
            channels (int, optional): C when x is wider (the fused projection output).
            a_col, b_col (int): element column of head 0 in a / b when they are column
                windows of a wider tensor (e.g. the projection output). Default 0."""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def devop_hpp(s):
    old = """    uint32_t batch,
    const tt::tt_metal::MemoryConfig& output_mem_config);"""
    new = """    uint32_t batch,
    uint32_t channels,
    uint32_t a_col,
    uint32_t b_col,
    const tt::tt_metal::MemoryConfig& output_mem_config);"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def devop_cpp(s):
    old = """    const auto& xs = in.x.logical_shape();
    TT_FATAL(xs.rank() == 3 && xs[0] == 1 && xs[2] == attrs.C, "gdn_decode_conv_gates: x must be [1,B,C]");"""
    new = """    const auto& xs = in.x.logical_shape();
    TT_FATAL(xs.rank() == 3 && xs[0] == 1 && xs[2] >= attrs.C, "gdn_decode_conv_gates: x must be [1,B,W] with W >= C");"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    const auto& as = in.a.logical_shape();
    const auto& bs = in.b.logical_shape();
    TT_FATAL(
        as.rank() == 3 && as[0] == 1 && as[1] == attrs.B && as[2] == attrs.Nv && bs == as,
        "gdn_decode_conv_gates: a and b must be [1,B,Nv] with B == batch");"""
    new = """    const auto& as = in.a.logical_shape();
    const auto& bs = in.b.logical_shape();
    TT_FATAL(
        as.rank() == 3 && as[0] == 1 && as[1] >= attrs.B && as[2] >= attrs.a_col + attrs.Nv,
        "gdn_decode_conv_gates: a must be [1,>=B,W] with W >= a_col + Nv");
    TT_FATAL(
        bs.rank() == 3 && bs[0] == 1 && bs[1] >= attrs.B && bs[2] >= attrs.b_col + attrs.Nv,
        "gdn_decode_conv_gates: b must be [1,>=B,W] with W >= b_col + Nv");"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    uint32_t batch,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = GdnConvGatesDeviceOperation;
    const auto& xs = x.logical_shape();  // [1,Bx,C]"""
    new = """    uint32_t batch,
    uint32_t channels,
    uint32_t a_col,
    uint32_t b_col,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = GdnConvGatesDeviceOperation;
    using namespace tt::constants;
    const auto& xs = x.logical_shape();  // [1,Bx,W]"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """        .C = xs[2],
        .Nv = as[2],
        .K = static_cast<uint32_t>(conv_states.size()),
        .output_mem_config = output_mem_config,
    };"""
    new = """        .C = channels,
        .Nv = static_cast<uint32_t>(as[2]) - a_col < static_cast<uint32_t>(as[2]) ? 0u : 0u,  // set below
        .K = static_cast<uint32_t>(conv_states.size()),
        .output_mem_config = output_mem_config,
    };
    // Nv comes from dt_bias (always exactly [1,1,Nv]); a / b may be wider windows.
    attrs.Nv = static_cast<uint32_t>(dt_bias.logical_shape()[-1]);
    attrs.Wt = (static_cast<uint32_t>(xs[2]) + TILE_WIDTH - 1) / TILE_WIDTH;
    attrs.AWt = (static_cast<uint32_t>(as[2]) + TILE_WIDTH - 1) / TILE_WIDTH;
    attrs.BWt = (static_cast<uint32_t>(b.logical_shape()[2]) + TILE_WIDTH - 1) / TILE_WIDTH;
    attrs.a_col = a_col;
    attrs.b_col = b_col;
    attrs.gates_packed = (a_col != 0) || (b_col != 0) || (static_cast<uint32_t>(as[2]) != attrs.Nv) ||
                         (static_cast<uint32_t>(b.logical_shape()[2]) != attrs.Nv);"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def factory_cpp(s):
    old = """    // Reader compile args: {K, Ct, Nvt, B, xBt} + accessors (x, st0..3, tap0..3, a, b, dt_bias, neg_exp_A).
    std::vector<uint32_t> reader_ct = {K, Ct, Nvt, attrs.B, xBt};"""
    new = """    // Reader compile args: {K, Ct, Nvt, B, xBt, Wt, GP, AWt, ACOL, BWt, BCOL} + accessors
    // (x, st0..3, tap0..3, a, b, dt_bias, neg_exp_A). Wt == Ct unless x is the wider
    // projection output; GP selects the per-element gate gather.
    std::vector<uint32_t> reader_ct = {
        K, Ct, Nvt, attrs.B, xBt, attrs.Wt, attrs.gates_packed ? 1u : 0u, attrs.AWt, attrs.a_col, attrs.BWt, attrs.b_col};"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """constexpr uint32_t g = tt::CBIndex::c_12;     // 1 io
}  // namespace cbcg"""
    new = """constexpr uint32_t g = tt::CBIndex::c_12;     // 1 io
constexpr uint32_t gsrc = tt::CBIndex::c_13;  // 2 io   source tiles for the packed gate gather
}  // namespace cbcg"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    add_cb(cbcg::g, 1, 2, df_io);
"""
    new = """    add_cb(cbcg::g, 1, 2, df_io);
    add_cb(cbcg::gsrc, 2, 1, df_io);
"""
    assert s.count(old) == 1
    return s.replace(old, new, 1)


def reader_cpp(s):
    old = """    constexpr uint32_t xBt = get_compile_time_arg_val(4);
    static_assert(K == 4, "reader is written for K == 4 taps");

    constexpr auto x_a = TensorAccessorArgs<5>();"""
    new = """    constexpr uint32_t xBt = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);    // x's width in tiles (Ct when not wider)
    constexpr uint32_t GP = get_compile_time_arg_val(6);    // 1: gather a/b per element from windows
    constexpr uint32_t AWt = get_compile_time_arg_val(7);
    constexpr uint32_t ACOL = get_compile_time_arg_val(8);
    constexpr uint32_t BWt = get_compile_time_arg_val(9);
    constexpr uint32_t BCOL = get_compile_time_arg_val(10);
    static_assert(K == 4, "reader is written for K == 4 taps");

    constexpr auto x_a = TensorAccessorArgs<11>();"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """            if (have_x) {
                noc.async_read(x_acc, cb, tb, {.page_id = bt * Ct + cc}, {.offset_bytes = 3 * tb});
            }"""
    new = """            if (have_x) {
                noc.async_read(x_acc, cb, tb, {.page_id = bt * Wt + cc}, {.offset_bytes = 3 * tb});
            }"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """    // gates: one tile per instance gi = bt_g*Nvt + t
    for (uint32_t gi = 0; gi < g_n; ++gi) {
        const uint32_t t = gi % Nvt;
        auto one = [&](const auto& acc, uint32_t cb_id, uint32_t page) {
            CircularBuffer cb(cb_id);
            cb.reserve_back(1);
            noc.async_read(acc, cb, tb, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb.push_back(1);
        };
        one(a_acc, cb_a, gi);
        one(b_acc, cb_b, gi);
        one(dtb_acc, cb_dtb, t);
        one(nega_acc, cb_nega, t);
    }"""
    new = """    // Gather gate tile t (output cols h in [32t, 32t+32) ∩ [0,Nv)) from a window that starts
    // at element column `col0` of a tensor `wt` tiles wide: the source columns col0+h span one
    // or two source tiles. Full-page reads into the scratch CB, then per-element L1 copies
    // (16-bit at bf16, 32-bit at fp32) into the zeroed target tile.
    auto gather_gate = [&](const auto& acc, uint32_t cb_id, uint32_t wt, uint32_t col0, uint32_t bt_g, uint32_t t) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t dst = cb.get_write_ptr();
        zero_words(dst, tb / 4);
        const uint32_t h0 = t * 32;
        const uint32_t h1 = (h0 + 32 < Nvt * 32) ? h0 + 32 : Nvt * 32;  // exclusive; Nv itself bounds via ACOL window
        const uint32_t c0 = col0 + h0;
        const uint32_t c1 = col0 + h1 - 1;
        const uint32_t p0 = c0 / 32;
        const uint32_t p1 = c1 / 32;
        CircularBuffer scb(cb_gsrc);
        scb.reserve_back(2);
        const uint32_t sbase = scb.get_write_ptr();
        for (uint32_t p = p0; p <= p1; p++) {
            noc.async_read(acc, scb, tb, {.page_id = bt_g * wt + p}, {.offset_bytes = (p - p0) * tb});
        }
        noc.async_read_barrier();
        asm volatile("" ::: "memory");
        for (uint32_t h = h0; h < h1; h++) {
            const uint32_t c = col0 + h;
            const uint32_t sp = sbase + (c / 32 - p0) * tb;
            const uint32_t sc = c % 32;
            const uint32_t dc = h - h0;
            for (uint32_t r = 0; r < 32; r++) {
                const uint32_t se = ((r / 16) * 2 + (sc / 16)) * 256 + (r % 16) * 16 + (sc % 16);
                const uint32_t de = ((r / 16) * 2 + (dc / 16)) * 256 + (r % 16) * 16 + (dc % 16);
                if (elem == 2) {
                    auto sv = CoreLocalMem<volatile uint16_t>(sp + se * 2);
                    auto dv = CoreLocalMem<volatile uint16_t>(dst + de * 2);
                    dv[0] = sv[0];
                } else {
                    auto sv = CoreLocalMem<volatile uint32_t>(sp + se * 4);
                    auto dv = CoreLocalMem<volatile uint32_t>(dst + de * 4);
                    dv[0] = sv[0];
                }
            }
        }
        asm volatile("" ::: "memory");
        scb.push_back(2);
        scb.pop_front(2);
        cb.push_back(1);
    };

    // gates: one tile per instance gi = bt_g*Nvt + t
    for (uint32_t gi = 0; gi < g_n; ++gi) {
        const uint32_t t = gi % Nvt;
        const uint32_t bt_g = gi / Nvt;
        auto one = [&](const auto& acc, uint32_t cb_id, uint32_t page) {
            CircularBuffer cb(cb_id);
            cb.reserve_back(1);
            noc.async_read(acc, cb, tb, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb.push_back(1);
        };
        if constexpr (GP) {
            gather_gate(a_acc, cb_a, AWt, ACOL, bt_g, t);
            gather_gate(b_acc, cb_b, BWt, BCOL, bt_g, t);
        } else {
            one(a_acc, cb_a, gi);
            one(b_acc, cb_b, gi);
        }
        one(dtb_acc, cb_dtb, t);
        one(nega_acc, cb_nega, t);
    }"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    old = """constexpr uint32_t cb_a = 6, cb_b = 7, cb_dtb = 8, cb_nega = 9;"""
    new = """constexpr uint32_t cb_a = 6, cb_b = 7, cb_dtb = 8, cb_nega = 9, cb_gsrc = 13;"""
    assert s.count(old) == 1
    s = s.replace(old, new, 1)
    s = s.replace(
        "// Compile args: {K, Ct, Nvt, B, xBt} + accessor args (x, st0..st3, tap0..tap3, a, b,",
        "// Compile args: {K, Ct, Nvt, B, xBt, Wt, GP, AWt, ACOL, BWt, BCOL} + accessor args (x, st0..st3, tap0..tap3, a, b,",
        1,
    )
    return s


if __name__ == "__main__":
    marker = os.path.join(ROOT, "device", "gdn_conv_gates_device_operation_types.hpp")
    if "gates_packed" in io.open(marker, encoding="utf-8").read():
        print("already applied")
        sys.exit(0)
    rw("device/gdn_conv_gates_device_operation_types.hpp", types_hpp)
    rw("gdn_conv_gates.hpp", api_hpp)
    rw("gdn_conv_gates.cpp", api_cpp)
    rw("gdn_conv_gates_nanobind.cpp", nanobind_cpp)
    rw("device/gdn_conv_gates_device_operation.hpp", devop_hpp)
    rw("device/gdn_conv_gates_device_operation.cpp", devop_cpp)
    rw("device/gdn_conv_gates_program_factory.cpp", factory_cpp)
    rw("device/kernels/dataflow/reader_gdn_conv_gates.cpp", reader_cpp)
    print("all patched")
