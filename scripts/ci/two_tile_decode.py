"""Two-tile attention and MLP bindings for a verify block wider than one 32-row tile.

Read from the image's sources (probe run 35503727180: attention/tp.py, gdn/tp.py, mlp.py,
tp_common.py, model_config.py), what the model's own decode does with a 64-row input once
the norms are two tiles high (two_tile_norm.py):

ATTENTION (attention/tp.py). forward_decode:514 takes its fused prep path - the fused
[q|k|v|gate] projection `_qkv_raw_decode`, ttnn.transformer.attn_decode_prep, then
`_decode_from_prep` (the K/V write and the SDPA the block replaces) - only while
`x.shape[-2] <= ttnn.TILE_SIZE` (:526-531). At 64 rows it falls to `_qkv` (:175), whose
first arm is the PREFILL fused all-gather-matmul (`_fuse_agmm and x.shape[-2] > TILE_SIZE`,
:186): it gathers an input the decode norm has already gathered, and neither the block's
K/V writer nor its readers would run. So the wide block binds `forward_decode` on every
full-attention layer to the prep path itself (TwoTileAttentionDecode: :533-557 without
the one-tile gate), which is the path the 32-row block runs under the image's
QWEN_ATTN_PREP switch. Inside it, `_qkv_raw_decode` (:162-173) runs the 1D decode matmul
with `args.attn_qkv_decode_1d_progcfg` UNCONDITIONALLY - a
MatmulMultiCoreReuseMultiCast1DProgramConfig built at M = 1 (model_config.py:210-212,
tp_common.create_matmul_1d_decode_progcfg:137-166), per_core_M 1. A 1D mcast_in0 matmul
computes every M tile on every core, so the block rebuilds that config at its rows:
per_core_M = rows / 32 and out_subblock_h by the builder's own rule, everything else kept
(two_tile_matmul_1d_progcfg), bound on the model args around the wide forward. One pass
over the interleaved fused weight (attention/tp.py:65-72); run 35507675630 (image v51)
ran it on the device at per_core_M 2.

Two ops of that path are one tile on the device itself, both found by v51:

- `attn_decode_prep` at batch 64 HANGS: the watcher shows its reader, writer and compute
  kernels resident with 23 worker cores per chip waiting on a circular buffer the reader
  never fills at that batch. `prep_by_tile` runs the op once per 32-row tile - the fused
  projection sliced at row 32 (tile-aligned), the rotary tables sliced per whole tile,
  `_kv_shard_cfg(32)` (the 8x4 height shard, one user per core), batch=32: the 32-row
  block's exact op twice - then joins the outputs on the user axis: q and the gate
  concatenated in DRAM; K and V, height-sharded 32 users per call, each returned to
  interleaved DRAM and concatenated into the (1, 64, 32, 256) DRAM tensor
  `_decode_from_prep` hands the block's K/V writer (packed_cache_writer.py), which is the
  layout that writer converted the 64-shard tensor to anyway before cutting it into its
  two 32-row tiles; the conversion is now the only step that disappears. The halves are
  joined rather than handed to the writer directly because `_decode_from_prep` frees its
  K/V through the real ttnn (attention/tp.py:722-723; attention_batch.serial_tail's
  Overlay replaces only the `experimental` and `transformer` namespaces), so what the
  tail receives must be one real tensor.
- `nlp_concat_heads_decode` refuses batch 64 on the host (TT_FATAL
  nlp_concat_heads_decode_device_operation.cpp:39: input_shape[1] <= 32).
  `TwoTileConcatHeads` binds `_concat_heads_decode` on the attention instance for the
  wide block - `_decode_from_prep` looks it up on `self` at call time (:749) - and runs
  the model's own method (:375-409) once per 32-user half of the gated SDPA output (a
  whole-tile slice on the user axis in L1): gx 8 and the 8x4 grid, the 32-row block's
  exact op twice, the (1, 32, 3072) outputs concatenated on the user axis in L1. The
  native contract is kept: the input is consumed and freed here, the joined output is
  the caller's.

The rest of the prep path is row-generic: `_decode_from_prep` (:715-764) runs the
block's writer and readers, then `_wo_proj` (:230-270), which at 64 rows takes its own
prefill arm - `create_prefill_mlp_matmul_program_config` at M = 64 on the interleaved wo,
L1 output - one pass (the same builder result the GDN out-projection ran on v51 in every
layer before layer 3); then tt_all_reduce (ccl.py:169-191 on this mesh: one
reduce_scatter_minimal_async, shape-agnostic; v51 ran it at 64 rows).

GDN (gdn/tp.py). The block already replaces `forward_decode` whole
(model_batch.gdn_forward). The two model-owned projections it calls route themselves
at 64 rows: `_project_qkvzab_raw` (:1032-1043) leaves the one-tile 1D arm at
`S <= TILE_SIZE` for `_col_proj` -> tp_common.sharded_decode_matmul (:600-634), whose
`seq > TILE_SIZE` branch is the generic 2D prefill config (`args.prefill_progcfg`) on the
interleaved qkvzab weight (gdn/tp.py:97-104), DRAM output; `_row_proj` (:379-411) leaves
its 1D arm the same way for the prefill 2D config on the interleaved out weight
(:135-142). One pass each. Nothing to bind.

MLP (mlp.py). `_forward_tp` (:214-350) has the same trap as the attention: its first arm
is the PREFILL fused all-gather + SwiGLU (`_fuse_gateup_agmm and x.shape[-2] > TILE_SIZE
and w.w_gate_up is not None`, :227), and w_gate_up is loaded whenever TP > 1 (:67-78,
tp_common.mlp_gateup_agmm_enabled). It would gather the gathered input. With that flag
off for the call, 64 rows take the unfused prefill arm (:274-296): w1 and w3 through
`create_prefill_mlp_matmul_program_config` at M = 64 with SiLU fused, L1 outputs, the
product in DRAM (:308), w2 through the same builder (:324-333) with its L1 output (:336),
then tt_all_reduce. One pass over w1, w3 and w2 - the bytes the 32-row block's 1D decode
arm reads. So the wide block binds `feed_forward.forward` on every layer to a wrapper
that runs `_forward_tp` with `_fuse_gateup_agmm` False for exactly that call
(TwoTileMLPForward).

INPUT LAYOUT. Both prefill arms and the GDN's 2D branch take the activation as ttnn.linear
finds it; at prefill that is DRAM-interleaved (layer.py:180-181). The one-tile decode arms
interleave the norm's width shard themselves (tp_common.matmul_1d_decode:172, mlp.py:257).
So the wide block's norm binding hands every consumer an L1-interleaved activation
(two_tile_norm: output_mem_config), and the 1D attention arm's own interleave is then a
no-op.

WHERE IT RUNS. All of it only for a block wider than one tile, applied through
model_batch.instance_overrides around that block's forward, and counted: ModelBatch.run
demands one attention forward per full-attention layer and one MLP forward per layer.
The 32-row block builds none of it.

UNVERIFIED ON HARDWARE until the four-user gate: the two prep calls and the two head
concats inside one 64-row forward and the joins between them; wo at M = 64. Run on the
device by v51 before layer 3's prep: the fused QKV matmul at per_core_M 2, and the prefill
2D program configs at M = 64 for the GDN in and out projections and w1, w3, w2.
"""

from model_batch import instance_overrides

TILE = 32
SUBBLOCK_CAP = 4  # fp32 dest accumulation: the builder's cap on out_subblock_h * out_subblock_w


def validate_two_tile_rows(rows):
    if type(rows) is not int or rows <= TILE or rows % TILE:
        raise ValueError('Two-tile decode bindings serve blocks of whole %d-row tiles beyond one tile; %r rows given'
                         % (TILE, rows))
    return rows // TILE


def two_tile_matmul_1d_progcfg(config, rows, operations):
    """The model's one-tile 1D decode matmul config (tp_common.create_matmul_1d_decode_progcfg
    at M = 1) rebuilt at M = rows: per_core_M = rows / 32, out_subblock_h by the builder's rule
    (the largest divisor of the tile count whose subblock stays within the fp32 cap), grid,
    in0_block_w, out_subblock_w, per_core_N and the fused activation kept. Refuses anything
    but a one-tile mcast_in0 config with fused batch."""
    tiles = validate_two_tile_rows(rows)
    names = ('compute_with_storage_grid_size', 'in0_block_w', 'out_subblock_h', 'out_subblock_w', 'per_core_M',
             'per_core_N', 'fuse_batch', 'fused_activation', 'mcast_in0')
    missing = [name for name in names if not hasattr(config, name)]
    if missing:
        raise ValueError('The 1D decode matmul config does not expose %s; it cannot be rebuilt two tiles high'
                         % ', '.join(missing))
    if config.per_core_M != 1 or config.out_subblock_h != 1:
        raise ValueError('The 1D decode matmul config is expected at one tile (per_core_M 1, out_subblock_h 1), '
                         'found per_core_M %r, out_subblock_h %r: the model changed' % (config.per_core_M, config.out_subblock_h))
    if config.mcast_in0 is not True or config.fuse_batch is not True:
        raise ValueError('The 1D decode matmul config is expected to multicast in0 with fused batch: the model changed')
    subblock_w = config.out_subblock_w
    if type(subblock_w) is not int or not 1 <= subblock_w <= SUBBLOCK_CAP:
        raise ValueError('The 1D decode matmul config carries an out_subblock_w of %r, outside the fp32 cap' % (subblock_w,))
    subblock_h = max(i for i in range(1, SUBBLOCK_CAP + 1) if tiles % i == 0 and i * subblock_w <= SUBBLOCK_CAP)
    grid = config.compute_with_storage_grid_size
    return operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y), in0_block_w=config.in0_block_w,
        out_subblock_h=subblock_h, out_subblock_w=subblock_w, per_core_M=tiles, per_core_N=config.per_core_N,
        fuse_batch=True, fused_activation=config.fused_activation, mcast_in0=True)


def prep_by_tile(operations, attention, qkv_raw, cos, sin, rows):
    """ttnn.transformer.attn_decode_prep over the block as one call per 32-row tile, its
    outputs joined on the user axis (attention/tp.py:535-548 at batch 32, twice).

    The projection (1, 1, rows, width) is sliced at every tile row (aligned starts and
    ends), the rotary tables (1, rows, 1, rope_dim) per whole tile, and each call takes
    `_kv_shard_cfg(32)` at batch 32: the 32-row block's exact op. q and the gate come back
    (1, 32, NH, HD) in DRAM and are concatenated to (1, rows, NH, HD); K and V come back
    height-sharded one user per core, are returned to interleaved DRAM and concatenated to
    (1, rows, 32, HD) - the layout the block's segmented K/V writer converts its input to
    before cutting it into 32-row tiles, so that conversion becomes a no-op. Everything
    made here but the four joined outputs is freed here; the joined outputs are the
    caller's, as the native prep's are (`_decode_from_prep` frees them).
    """
    validate_two_tile_rows(rows)
    dram = operations.DRAM_MEMORY_CONFIG
    shape, rope = tuple(qkv_raw.shape), tuple(cos.shape)
    if len(shape) != 4 or shape[:3] != (1, 1, rows):
        raise ValueError('The fused QKV projection of a %d-row block must be (1, 1, %d, width); %r given' % (rows, rows, shape))
    if len(rope) != 4 or rope[:2] != (1, rows) or tuple(sin.shape) != rope:
        raise ValueError('The rotary tables of a %d-row block must be a (1, %d, 1, rope_dim) pair; %r and %r given'
                         % (rows, rows, rope, tuple(sin.shape)))
    config = attention._kv_shard_cfg(TILE)
    owned, parts, joined = [], ([], [], [], []), []
    try:
        for first in range(0, rows, TILE):
            last = first + TILE
            piece = operations.slice(qkv_raw, (0, 0, first, 0), (1, 1, last, shape[3]), memory_config=dram)
            owned.append(piece)
            cos_piece = operations.slice(cos, (0, first, 0, 0), (1, last, rope[2], rope[3]), memory_config=dram)
            owned.append(cos_piece)
            sin_piece = operations.slice(sin, (0, first, 0, 0), (1, last, rope[2], rope[3]), memory_config=dram)
            owned.append(sin_piece)
            outputs = operations.transformer.attn_decode_prep(
                piece, cos_piece, sin_piece, attention.tw['q_norm'], attention.tw['k_norm'],
                attention.NH, attention.NKV, attention.HD, attention.rope_dim, config,
                batch=TILE, memory_config=dram)
            owned.extend(outputs)
            q, gate, k_sh, v_sh = outputs
            parts[0].append(q)
            parts[1].append(gate)
            for index, sharded in ((2, k_sh), (3, v_sh)):
                interleaved = operations.sharded_to_interleaved(sharded, dram)
                owned.append(interleaved)
                parts[index].append(interleaved)
        for pieces in parts:
            joined.append(operations.concat(pieces, dim=1, memory_config=dram))
        return tuple(joined)
    except BaseException:
        for value in joined:
            operations.deallocate(value)
        raise
    finally:
        for value in owned:
            operations.deallocate(value)


class TwoTileConcatHeads:
    """One full-attention layer's `_concat_heads_decode` at the block's users, as two
    32-user halves through the model's own method (attention/tp.py:375-409), whose op
    refuses batch 64 on the host. Bound on the attention instance for the wide block;
    `_decode_from_prep` looks it up on `self` at call time (:749)."""

    def __init__(self, attention, rows, operations):
        validate_two_tile_rows(rows)
        native = getattr(type(attention), '_concat_heads_decode', None)
        if not callable(native):
            raise ValueError('The attention class no longer defines _concat_heads_decode; the two-tile head concat cannot mirror it')
        if not getattr(attention, '_use_nlp_decode_heads', False):
            raise ValueError('The two-tile head concat serves the nlp decode-heads path the serving model runs')
        self.attention, self.rows, self.operations, self.native = attention, rows, operations, native
        self.calls = 0

    def __call__(self, gated, B):
        operations = self.operations
        shape = tuple(gated.shape)
        if B != self.rows or len(shape) != 4 or shape[1] != self.rows:
            raise ValueError('The two-tile head concat was bound for %d users; batch %r over %r given' % (self.rows, B, shape))
        l1 = operations.L1_MEMORY_CONFIG
        halves = []
        try:
            for first in range(0, self.rows, TILE):
                halves.append(operations.slice(gated, (0, first, 0, 0), (1, first + TILE, shape[2], shape[3]), memory_config=l1))
        except BaseException:
            for half in halves:
                operations.deallocate(half)
            raise
        # The native contract: the input is consumed and freed by the concat.
        operations.deallocate(gated)
        outputs = []
        try:
            for index, half in enumerate(halves):
                # The native method frees the half it was given; one that fails leaves its
                # half in an unknown state, so only the halves not yet handed over are freed.
                try:
                    outputs.append(self.native(self.attention, half, TILE))
                except BaseException:
                    for remaining in halves[index + 1:]:
                        operations.deallocate(remaining)
                    raise
            joined = operations.concat(outputs, dim=1, memory_config=l1)
        finally:
            for value in outputs:
                operations.deallocate(value)
        self.calls += 1
        return joined


class TwoTileAttentionDecode:
    """One full-attention layer's forward_decode at the block's rows: the model's fused prep
    path (attention/tp.py:533-557) - its projection under the rebuilt config, its prep op
    once per 32-row tile (prep_by_tile), then its own `_decode_from_prep`, which the block
    has bound to its K/V writer and readers and which must take the layer's two-tile head
    concat (`concat`) exactly once."""

    def __init__(self, attention, rows, operations, progcfg):
        validate_two_tile_rows(rows)
        self.rows, self.operations, self.progcfg = rows, operations, progcfg
        args = getattr(attention, 'args', None)
        if args is None or not getattr(args, 'proj_1d_decode', False):
            raise ValueError('The two-tile attention forward serves the 1D decode projection the model config selects')
        if not getattr(attention, 'use_paged', False) or not getattr(attention, '_fused_qkv', False):
            raise ValueError('The two-tile attention forward needs the paged fused-QKV attention the serving model runs')
        for name in ('_qkv_raw_decode', '_kv_shard_cfg', '_decode_from_prep'):
            if not callable(getattr(attention, name, None)):
                raise ValueError('The attention module no longer exposes %s; the two-tile forward cannot mirror its prep path' % name)
        weights = getattr(attention, 'tw', None)
        if not isinstance(weights, dict) or 'q_norm' not in weights or 'k_norm' not in weights:
            raise ValueError('The attention weights no longer carry q_norm and k_norm')
        dims = [getattr(attention, name, None) for name in ('NH', 'NKV', 'HD', 'rope_dim')]
        if any(type(value) is not int or value <= 0 for value in dims):
            raise ValueError('The attention module no longer exposes integer NH, NKV, HD and rope_dim')
        if not callable(getattr(getattr(operations, 'transformer', None), 'attn_decode_prep', None)):
            raise ValueError('ttnn.transformer.attn_decode_prep is required for the two-tile attention forward')
        if any(not callable(getattr(operations, name, None)) for name in ('slice', 'concat', 'sharded_to_interleaved')):
            raise ValueError('ttnn slice, concat and sharded_to_interleaved are required to join the per-tile prep outputs')
        self.concat = TwoTileConcatHeads(attention, rows, operations)
        self.attention = attention
        self.calls = 0

    def __call__(self, x, cur_pos_tt, cos_tt, sin_tt, page_table=None):
        attention, operations = self.attention, self.operations
        if page_table is None:
            raise ValueError('The two-tile attention forward serves the paged decode only')
        if tuple(x.shape)[-2] != self.rows:
            raise ValueError('The two-tile attention forward was bound for %d rows; %r given' % (self.rows, tuple(x.shape)))
        if getattr(attention.args, 'attn_qkv_decode_1d_progcfg', None) is not self.progcfg:
            raise AssertionError('The two-tile QKV projection config is not bound on the model args')
        rows = self.rows
        qkv_raw = attention._qkv_raw_decode(x)
        try:
            q, gate, k_sh, v_sh = prep_by_tile(operations, attention, qkv_raw, cos_tt, sin_tt, rows)
        finally:
            operations.deallocate(qkv_raw)
        concats = self.concat.calls
        result = attention._decode_from_prep(q, gate, k_sh, v_sh, cur_pos_tt, page_table, rows)
        if self.concat.calls - concats != 1:
            raise AssertionError("The wide block's attention tail must take its two-tile head concat exactly once; "
                                 "%d taken" % (self.concat.calls - concats))
        self.calls += 1
        return result


class TwoTileAttentionBinding:
    """The wide block's attention bindings: the rebuilt fused-QKV config on the model args
    and, per full-attention layer, the two-tile forward and its head concat on the
    attention instance; `calls` counts the forwards (each of which counted its concat)."""

    label = 'full-attention forward'

    def __init__(self, model, rows, operations):
        validate_two_tile_rows(rows)
        layers = [layer for layer in getattr(model, 'layers', ()) if getattr(layer, 'is_full_attention', False)]
        args = getattr(model, 'args', None)
        if not layers or args is None:
            raise ValueError('A model with args and at least one full-attention layer is required')
        attentions = [layer.attention for layer in layers]
        if any(getattr(attention, 'args', None) is not args for attention in attentions):
            raise ValueError('Every full-attention layer must share the model args the config is bound on')
        native = getattr(args, 'attn_qkv_decode_1d_progcfg', None)
        if native is None:
            raise ValueError('The model args carry no attn_qkv_decode_1d_progcfg to rebuild; the model changed')
        self.rows = rows
        self.progcfg = two_tile_matmul_1d_progcfg(native, rows, operations)
        self.forwards = [TwoTileAttentionDecode(attention, rows, operations, self.progcfg) for attention in attentions]
        self.bindings = [(args, 'attn_qkv_decode_1d_progcfg', self.progcfg)]
        for attention, forward in zip(attentions, self.forwards):
            self.bindings.append((attention, 'forward_decode', forward))
            self.bindings.append((attention, '_concat_heads_decode', forward.concat))
        self.expected_calls = len(self.forwards)

    @property
    def calls(self):
        return sum(forward.calls for forward in self.forwards)


class TwoTileMLPForward:
    """One layer's MLP forward at the block's rows: the model's `_forward_tp` with its prefill
    all-gather fusion off for exactly this call, so 64 rows take the unfused prefill arm
    (one pass over w1, w3 and w2) instead of gathering an already gathered input."""

    def __init__(self, mlp, rows):
        validate_two_tile_rows(rows)
        if getattr(mlp, 'num_devices', 1) <= 1 or not callable(getattr(mlp, '_forward_tp', None)):
            raise ValueError('The two-tile MLP forward serves the tensor-parallel MLP')
        if '_fuse_gateup_agmm' not in getattr(mlp, '__dict__', {}):
            raise ValueError('The MLP no longer keeps its gate/up all-gather fusion switch on the instance; the model changed')
        self.mlp, self.rows = mlp, rows
        self.calls = 0

    def __call__(self, x):
        if tuple(x.shape)[-2] != self.rows:
            raise ValueError('The two-tile MLP forward was bound for %d rows; %r given' % (self.rows, tuple(x.shape)))
        with instance_overrides([(self.mlp, '_fuse_gateup_agmm', False)]):
            result = self.mlp._forward_tp(x)
        self.calls += 1
        return result


class TwoTileMLPBinding:
    """The wide block's MLP bindings: one two-tile forward per layer; `calls` counts them."""

    label = 'MLP forward'

    def __init__(self, model, rows):
        validate_two_tile_rows(rows)
        layers = list(getattr(model, 'layers', ()))
        if not layers or any(getattr(layer, 'feed_forward', None) is None for layer in layers):
            raise ValueError('A model whose every layer carries a feed_forward MLP is required')
        self.rows = rows
        self.forwards = [TwoTileMLPForward(layer.feed_forward, rows) for layer in layers]
        self.bindings = [(layer.feed_forward, 'forward', forward) for layer, forward in zip(layers, self.forwards)]
        self.expected_calls = len(self.forwards)

    @property
    def calls(self):
        return sum(forward.calls for forward in self.forwards)


def bind_two_tile_attention(model, rows, operations):
    return TwoTileAttentionBinding(model, rows, operations)


def bind_two_tile_mlp(model, rows):
    return TwoTileMLPBinding(model, rows)
