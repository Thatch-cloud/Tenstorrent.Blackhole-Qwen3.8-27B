"""quad_draft at any served width: the four-card twin of the four-user, 64-row draft pass (QWEN_FAST_QUAD_DRAFT).

quad_draft.py stays as it is: it is written for the pair's chips (16 query / 4 KV heads per chip, two chips, a 1, 2 mesh, the
(1, 4, 2048, 128) K/V bank), it is in both image copy lists at the bytes test_quad_draft holds, and the pair's evidence
(CONV_KERNEL_SHA256, the card-B Q0 proofs) is about those bytes. tp_addresses.install() puts THIS module in its place in
sys.modules at QWEN_FAST_TP=4 only (MODULE_TWINS: the coordinator imports quad_draft lazily, so it reaches the twin; the pair
never installs and keeps the pinned module).

WHAT CHANGES, and nothing else. Everything below is the pinned function with its literal width read from tp_shapes; at two
chips each is call for call the pinned one (test_quad_draft_tp4 holds that):
  - the per-chip heads: 8 query / 2 KV at four cards, so the quad's four-way fold runs the SDPA at 32 query / 8 KV heads
    (exactly the pair fold's program at the pair, already proven bit-exact against single-user drafting on card B), and the
    folded head map 16h + 4u + j -> KV head 4h + u holds with the group of four;
  - the 64-row head helpers (split_projected_heads, project_key_value, concatenate_query_heads): 1,024-wide query and
    256-wide K/V projections, nlp_create_qkv_heads at 8 / 2 heads;
  - the fused convolution: the mesh is the (1, tp) mesh and each chip gets the same 110-worker program (the learned
    convolution is replicated at full width on every chip, so quad_conv_io.cpp, sha-pinned, and its page math are unchanged);
  - the readback: tp chips x two 32,768-column vocabulary chunks (62,080 per chip) x values and indices, and the replicated
    selector features read from every chip;
  - the K/V banks, in two modes. Without QWEN_FAST_FUSED_COMMIT_LIVE_BANKS its bucket owns four users' placeholder banks,
    (1, 2, 2048, 128) per layer and k / v, and copies each user's active bank into them every round: what the pair does without
    the flag (dflash_proposal_trace.PreparedPackedDFlashProposal._update); the copy reads whichever bank is active, so it is
    swap-parity agnostic. With the flag - accepted only beside QWEN_FAST_FUSED_COMMIT=1 and _INPLACE=1, the four-card fused
    commit (fused_commit_tp), whose in-place slides never move a live bank - the bucket binds the pool's ACTIVE banks
    (fused_commit.live_bank_history per pair, width free) and a round copies only where a device's live bank sits on the pool's
    spare side (an odd swap count before its first in-place commit), into the pool's active bank, logging the pair's
    `[PACKED-PROPOSE] live banks ... normalised=N`. That is 40 fewer bank copies a round and no placeholders (40 MiB a chip);
  - the engage-time refusal: the flags the four-card path can serve (the live-banks flag only with the fused commit and its
    in-place slides) and the (1, 4) mesh.

WHAT DOES NOT CHANGE. The K/V plan (twelve pieces, each user's pad from its own pair's rows), the host inputs and RoPE, the
seams, the coordinator's contract (has_pending / finish / collect / adopt), the shadow audit against the two pair traces
(QWEN_FAST_QUAD_DRAFT_AUDIT), every log line and marker (the gate's regexes), the flags QWEN_FAST_QUAD_SDPA / _CONV and the
give-up rule. The names not redefined here are the pinned module's own objects (module __getattr__), so the coordinator and
the tests read one set of markers, lines and constants.

WHY IT IS EXACT (per user, against drafting alone). The quad's every (query head, KV head) work unit is that user's
single-user unit: the same 16 query rows at the same tile rows, the same K/V at every visible key in the same order, the same
mask tiles and the same kernel; only the compile-time head counts differ (32 / 8 here). The row-local ops, the matmul programs
at per_core_M = 2, the readback merges and the selector are the pair's and the single's, applied to tile-aligned halves. Held
on the CPU (test_quad_draft_tp4: the fold at 8 / 2 heads against each user's single, K/V layout bytes, the readback against
four singles' rows, shape runs at four chips, the coordinator at four cards); held on hardware by the in-model singles audit
(QWEN_FAST_DRAFT_SINGLES_AUDIT, draft_singles_audit.py) and by the per-request accepted-prefix comparison. NOT qualified: no
card has run this module.

Stdlib and torch only at call time (torch inside functions), importable on py 3.7 beside the pinned module.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import quad_draft as _pinned
import tp_shapes
from quad_draft import (HIDDEN, PAIRS, conv_kernel_path, conv_pages, core_ranges, grid_fits, log_line, seam_words,
                        validate_conv_shapes)

# The pinned geometry (one assignment there: the image's copy-closure check reads simple assignments), read off the module.
CONTEXT, BLOCK, SPAN = _pinned.CONTEXT, _pinned.BLOCK, _pinned.SPAN
USERS, ROWS, HEAD_DIM = _pinned.USERS, _pinned.ROWS, _pinned.HEAD_DIM

# What the four-card quad serves. The live-banks flag is not required; set, it is accepted only with the four-card fused commit and its
# in-place slides (an F4-bound quad reads banks that must never move: the eager four-card slide swaps them every commit).
REQUIRED_FLAGS = ('QWEN_FAST_PACKED_PROPOSAL', 'QWEN_FAST_PAIR_ROW_EXACT', 'QWEN_FAST_ROUND_B1')
LIVE_BANKS_FLAG = 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'
LIVE_BANKS_NEEDS = ('QWEN_FAST_FUSED_COMMIT', 'QWEN_FAST_FUSED_COMMIT_INPLACE')
TP4 = 4
GROUP_USERS = 2                   # users per half: the pair fold's two


def __getattr__(name):
    """Every name this twin does not define is the pinned module's own (markers, line formats, flag readers, the K/V plan,
    the audit, the constants), so the two modules cannot describe different quads."""
    if name.startswith('__'):
        raise AttributeError(name)
    try:
        return getattr(_pinned, name)
    except AttributeError:
        raise AttributeError('module %r has no attribute %r' % (__name__, name)) from None


# ---------------------------------------------------------------------------------------------
# The width.
# ---------------------------------------------------------------------------------------------

def heads():
    """(query heads, KV heads, GQA group, quad query heads, quad KV heads) per chip at the width this process serves at:
    (16, 4, 4, 64, 16) at the pair, (8, 2, 4, 32, 8) at four cards."""
    found = tp_shapes.active()
    query, key = found.draft_heads, found.draft_kv_heads
    return query, key, query // key, USERS * query, USERS * key


def live_banks_requested(environ=None):
    """QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1 (refusal() holds the flags it needs)."""
    return (os.environ if environ is None else environ).get(LIVE_BANKS_FLAG) == '1'


def live_banks_missing(environ=None):
    """The flags the live-banks mode needs that are not 1 (in LIVE_BANKS_NEEDS order); empty when it is off."""
    environ = os.environ if environ is None else environ
    if not live_banks_requested(environ):
        return []
    return [name for name in LIVE_BANKS_NEEDS if environ.get(name) != '1']


def missing_requirements(environ=None):
    """The required flags not set to 1, in REQUIRED_FLAGS order."""
    environ = os.environ if environ is None else environ
    return [name for name in REQUIRED_FLAGS if environ.get(name) != '1']


def note(slots, sdpa, conv, *, log=None):
    """MARKER once per process, at the first quad bucket that captured and replayed (the pinned module's once-flag:
    whichever module logs it, it is logged once)."""
    noted = _pinned._NOTED
    if noted:
        return False
    noted.append(tuple(slots))
    query, key = heads()[0], heads()[1]
    text = ('%d/%d' % heads()[3:5]) if sdpa == 'fold' else '%d/%dx2' % (2 * query, 2 * key)
    (log or log_line)('%s slots=[%s] heads=%s rows=%d sdpa=%s conv=%s' % (
        _pinned.MARKER, ','.join(str(slot) for slot in slots), text, ROWS, sdpa, conv))
    return True


# ---------------------------------------------------------------------------------------------
# The draft SDPA: the four-way fold, or the pair fold per half.
# ---------------------------------------------------------------------------------------------

def validate_quad(operations, query, key, value, mask):
    query_heads, key_heads = heads()[:2]
    if (tuple(query.shape) != (1, query_heads, ROWS, HEAD_DIM)
            or tuple(key.shape) != (1, key_heads, USERS * SPAN, HEAD_DIM) or tuple(value.shape) != tuple(key.shape)
            or tuple(mask.shape) != (1, 1, 32, SPAN)):
        raise ValueError('The quad fold takes the (1, %d, 64, 128) query, the four-segment (1, %d, 8320, 128) keys '
                         'and values and the single-user (1, 1, 32, 2080) mask' % (query_heads, key_heads))
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 quad draft attention operands required')
    if any(tensor.layout != operations.TILE_LAYOUT or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG
           for tensor in (query, key, value, mask)):
        raise ValueError('Interleaved tiled DRAM operands required')


def quad_fold_query(operations, query, retain):
    """(1, q, 64, 128) -> (1, 4q, 32, 128). Each tile-aligned half is folded by the pair fold (pair_row_exact_tp.fold_query:
    head (2q/kv)h + 4u' + j), read as (kv, 2 * group, 32, 128), and the halves are concatenated on dim 1, so folded head
    (4q/kv)h + group * u + j with u = 2p + u'. The GQA group then maps it to KV head 4h + u: user u's segment of head h."""
    from pair_row_exact_tp import fold_query

    query_heads, key_heads, group = heads()[:3]
    memory = operations.DRAM_MEMORY_CONFIG
    grouped = []
    for pair in range(GROUP_USERS):
        half = retain(operations.slice(query, (0, 0, 32 * pair, 0), (1, query_heads, 32 * (pair + 1), HEAD_DIM)))
        folded = fold_query(operations, half, retain)
        grouped.append(retain(operations.reshape(folded, (key_heads, 2 * group, 32, HEAD_DIM))))
    combined = retain(operations.concat(grouped, dim=1, memory_config=memory))
    return retain(operations.reshape(combined, (1, heads()[3], 32, HEAD_DIM)))


def quad_fold_keys(operations, tensor, retain):
    """(1, kv, 8320, 128) -> (1, 4 * kv, 2080, 128), a view: KV head 4h + u is user u's segment of head h."""
    return retain(operations.reshape(tensor, (1, heads()[4], SPAN, HEAD_DIM)))


def quad_unfold_output(operations, output, retain):
    """(1, 4q, 32, 128) -> (1, q, 64, 128): per half p, folded heads [(4q/kv)h + 2 * group * p, ... + 2 * group) are pair
    p's (1, 2q, 32, 128) fold output, which the pair unfold turns into the pair's packed rows."""
    from pair_row_exact_tp import unfold_output

    query_heads, key_heads, group = heads()[:3]
    grouped = retain(operations.reshape(output, (key_heads, heads()[3] // key_heads, 32, HEAD_DIM)))
    halves = []
    for pair in range(GROUP_USERS):
        part = retain(operations.slice(grouped, (0, 2 * group * pair, 0, 0),
                                       (key_heads, 2 * group * (pair + 1), 32, HEAD_DIM)))
        folded = retain(operations.reshape(part, (1, 2 * query_heads, 32, HEAD_DIM)))
        halves.append(unfold_output(operations, folded, retain))
    return retain(operations.concat(halves, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))


def fold_attention(operations, query, key, value, mask, retain, *, mask_validated=False):
    """QWEN_FAST_QUAD_SDPA=fold: every (Q head, KV head) unit is the pair fold's unit for that user, byte for byte in Q, K,
    V and mask (given the plan's pads); only the compile-time NQH / NKH differ (32 / 8 at four cards)."""
    from pair_row_exact import folded_sdpa

    if mask_validated is not True:
        raise ValueError('Validate the single-user host mask before upload and replay')
    validate_quad(operations, query, key, value, mask)
    folded = quad_fold_query(operations, query, retain)
    keys = quad_fold_keys(operations, key, retain)
    values = quad_fold_keys(operations, value, retain)
    attention = retain(folded_sdpa(operations, folded, keys, values, mask))
    return quad_unfold_output(operations, attention, retain)


def pairs_attention(operations, query, key, value, mask, retain, *, mask_validated=False):
    """QWEN_FAST_QUAD_SDPA=pairs: the pair fold on each half - pair p's 32 query rows and its two key segments, keys
    [4160p, 4160p + 4160), which are byte for byte its own pair assembly."""
    from pair_row_exact import folded_sdpa
    from pair_row_exact_tp import fold_keys, fold_query, unfold_output, validate_fold

    if mask_validated is not True:
        raise ValueError('Validate the single-user host mask before upload and replay')
    validate_quad(operations, query, key, value, mask)
    query_heads, key_heads = heads()[:2]
    halves = []
    for pair in range(GROUP_USERS):
        rows = retain(operations.slice(query, (0, 0, 32 * pair, 0), (1, query_heads, 32 * (pair + 1), HEAD_DIM)))
        keys, values = (retain(operations.slice(tensor, (0, 0, 2 * SPAN * pair, 0),
                                                (1, key_heads, 2 * SPAN * (pair + 1), HEAD_DIM)))
                        for tensor in (key, value))
        validate_fold(operations, rows, keys, values, mask)
        folded = fold_query(operations, rows, retain)
        attention = retain(folded_sdpa(operations, folded, fold_keys(operations, keys, retain),
                                       fold_keys(operations, values, retain), mask))
        halves.append(unfold_output(operations, attention, retain))
    return retain(operations.concat(halves, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))


def quad_head_map():
    """{folded query head: (h, u, j, kv head)}: the fold's head arithmetic at this width."""
    query_heads, key_heads, group, quad_query, _ = heads()
    span = quad_query // key_heads
    return {span * h + group * u + j: (h, u, j, USERS * h + u)
            for h in range(key_heads) for u in range(USERS) for j in range(group)}


# ---------------------------------------------------------------------------------------------
# The 64-row head helpers.
# ---------------------------------------------------------------------------------------------

def _tiled_dram(operations, tensors):
    return all(tensor.dtype == operations.bfloat16 and tensor.layout == operations.TILE_LAYOUT
               and tensor.memory_config() == operations.DRAM_MEMORY_CONFIG for tensor in tensors)


def split_projected_heads(operations, query, key, value, retain):
    """draft_head_layout_tp.split_projected_heads at 64 rows: query, keys and values share the 64 rows, so there is no query
    pad and no slice back."""
    found = tp_shapes.active()
    if (tuple(query.shape) != (1, 1, ROWS, found.draft_query)
            or tuple(key.shape) != (1, 1, ROWS, found.draft_kv_heads * HEAD_DIM)
            or tuple(value.shape) != tuple(key.shape) or not _tiled_dram(operations, (query, key, value))):
        raise ValueError('64-row BF16 DRAM projections for %d query and %d KV heads required'
                         % (found.draft_heads, found.draft_kv_heads))
    combined_kv = retain(operations.concat([key, value], dim=3, memory_config=operations.DRAM_MEMORY_CONFIG))
    made = operations.experimental.nlp_create_qkv_heads(query, combined_kv, num_heads=found.draft_heads,
        num_kv_heads=found.draft_kv_heads, transpose_k_heads=False, memory_config=operations.DRAM_MEMORY_CONFIG)
    query_heads, key_heads, value_heads = (retain(tensor) for tensor in made)
    return dict(q=query_heads, k=key_heads, v=value_heads)


def project_key_value(operations, inputs, query, cosine_sine, retain, *, parameters):
    """draft_kv_projection_tp.project_key_value at 64 rows: the same program (per_core_M = rows // 32 = 2), the 64-row head
    split, then today's k norm, rotary and typecast."""
    if (parameters.get('operations') is not operations or parameters.get('native_head_layout') is not True
            or tuple(inputs.shape) != (1, 1, ROWS, HIDDEN) or tuple(query.shape) != (1, 1, ROWS, tp_shapes.active().draft_query)
            or len(cosine_sine) != 2 or any(tuple(table.shape) != (1, 1, ROWS, HEAD_DIM) for table in cosine_sine)
            or not _tiled_dram(operations, (inputs, query, *cosine_sine))):
        raise ValueError('Owned 64-row BF16 tiled K/V rows, the live-key rotary tables and native head parameters '
                         'required')
    kernel = parameters['kernel']
    program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 8),
        in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=ROWS // 32,
        per_core_N=1, fuse_batch=True, fused_activation=None, mcast_in0=True)
    flat = {}
    for name in ('k', 'v'):
        projected = retain(operations.matmul(inputs, parameters['projections'][name], dtype=operations.float32,
            compute_kernel_config=kernel, program_config=program, memory_config=operations.DRAM_MEMORY_CONFIG))
        flat[name] = retain(operations.typecast(projected, operations.bfloat16))
    made = split_projected_heads(operations, query, flat['k'], flat['v'], retain)
    normalized = retain(operations.rms_norm(made['k'], epsilon=1e-6, weight=parameters['head_norms']['k'],
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    wide = [retain(operations.typecast(value, operations.float32)) for value in (normalized, *cosine_sine)]
    rotated = retain(operations.experimental.rotary_embedding_hf(*wide, is_decode_mode=False,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    return dict(q=made['q'], k=retain(operations.typecast(rotated, operations.bfloat16)), v=made['v'])


def concatenate_query_heads(operations, value, retain):
    """draft_head_layout_tp.concatenate_query_heads at 64 rows."""
    if tuple(value.shape) != (1, heads()[0], ROWS, HEAD_DIM) or not _tiled_dram(operations, (value,)):
        raise ValueError('64-row BF16 DRAM query heads required')
    return retain(operations.experimental.nlp_concat_heads(value, memory_config=operations.DRAM_MEMORY_CONFIG))


# ---------------------------------------------------------------------------------------------
# The fused convolution.
# ---------------------------------------------------------------------------------------------

def quad_fused_convolution(operations, mesh, hidden, dynamic, base, *, boundaries, conv='110'):
    """quad_draft.quad_fused_convolution over the (1, tp) mesh: the promoted I/O kernel reads each page's tiles and seam
    word, the served compute kernel does the per-page arithmetic, one compile arg per group of workers taking the same
    number of pages. Per chip, runtime args are the six buffer addresses + [rows, worker, workers, low seams, high seams]
    - the convolution is replicated at full width, so every chip runs the same program on its own operands."""
    validate_conv_shapes(hidden, dynamic, base)
    if conv not in _pinned.CONV_VARIANTS:
        raise ValueError('The quad conv program runs on 110 or 80 workers, not %r' % (conv,))
    low, high = seam_words(boundaries, ROWS)
    tensors = [hidden, *dynamic, *base]
    chips = tp_shapes.mesh_width(mesh)
    if chips is None or not _tiled_dram(operations, tensors):
        raise ValueError('%s-chip interleaved DRAM BF16 convolution operands required' % tp_shapes.count_word())
    parts = [operations.get_device_tensors(value) for value in tensors]
    if any(len(shards) != chips for shards in parts):
        raise ValueError('%s operand shards required' % tp_shapes.all_chips())
    kernel = conv_kernel_path()
    spec = _pinned.CONV_VARIANTS[conv]
    workers = spec['workers']
    coordinates = [spec['core'](worker) for worker in range(workers)]
    pages = conv_pages(workers)
    output = operations.empty(tuple(hidden.shape), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
        device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    tensors.append(output)
    parts.append(operations.get_device_tensors(output))
    cores = core_ranges(operations, coordinates)
    buffers = [operations.CBDescriptor(total_size=2048 * count, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
            page_size=2048, tile=operations.TileDescriptor(operations.Tile([32, 32])))])
        for index, count in ((0, 7), (1, 2), (16, 1))]
    groups = {}
    for worker, owned in pages.items():
        groups.setdefault(len(owned), []).append(coordinates[worker])
    computes = [operations.KernelDescriptor(kernel_source=str(Path(_pinned.__file__).with_name(_pinned.CONV_COMPUTE)),
        core_ranges=core_ranges(operations, group), compile_time_args=[count],
        config=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                  math_approx_mode=False))
        for count, group in sorted(groups.items())]
    program = operations.MeshProgramDescriptor()
    try:
        for chip in range(chips):
            local = [shards[chip] for shards in parts]
            if local[-1].buffer_address() in {value.buffer_address() for value in local[:-1]}:
                raise ValueError('Convolution output must not alias borrowed inputs')
            runtime = operations.RuntimeArgs()
            for worker, (x, y) in enumerate(coordinates):
                runtime[x][y] = [value.buffer_address() for value in local] + [ROWS, worker, workers, low, high]
            reader = operations.KernelDescriptor(kernel_source=str(kernel), core_ranges=cores,
                compile_time_args=[argument for value in local
                                   for argument in operations.TensorAccessorArgs(value).get_compile_time_args()],
                runtime_args=runtime, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
            coordinate = operations.MeshCoordinate(0, chip)
            program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[reader, *computes], cbs=buffers)
        operations.generic_op(tensors, program)
    except BaseException:
        operations.deallocate(output)
        raise
    return output


def halves_convolution(operations, mesh, hidden, dynamic, base, *, boundaries, retain):
    """QWEN_FAST_QUAD_CONV=halves (C0): the served 32-row fused call on each tile-aligned half, then a concat - at four cards
    the four-card twin of that call (draft_convolution_fused_tp)."""
    from draft_convolution_fused_tp import fused_convolution

    validate_conv_shapes(hidden, dynamic, base)
    seam_words(boundaries, ROWS)
    spans = tuple(tuple(span) for span in boundaries)
    parts = []
    for half in range(GROUP_USERS):
        start, stop = 32 * half, 32 * (half + 1)
        rows = retain(operations.slice(hidden, (0, 0, start, 0), (1, 1, stop, HIDDEN)))
        kernels = [retain(operations.slice(value, (0, 0, start, 0), (1, 1, stop, 320))) for value in dynamic]
        local = tuple((low - start, high - start) for low, high in spans if start <= low and high <= stop)
        parts.append(retain(fused_convolution(operations, mesh, rows, kernels, base, boundaries=local)))
    return operations.concat(parts, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------------------------------------
# The head.
# ---------------------------------------------------------------------------------------------

def head_candidates(operations, model, normalized, owned, retain):
    """H0: draft_shared_head_tp.shared_head_candidates on each tile-aligned 32-row half, then the candidate concat."""
    from draft_shared_head_tp import shared_head_candidates

    if tuple(normalized.shape) != (1, 1, ROWS, HIDDEN):
        raise ValueError('The 64-row learned-normalized block required')
    halves = []
    for half in range(GROUP_USERS):
        block = retain(operations.slice(normalized, (0, 0, 32 * half, 0), (1, 1, 32 * (half + 1), HIDDEN)))
        halves.append(shared_head_candidates(operations, model, block, owned))
    return _pinned.concat_candidates(operations, halves[0], halves[1], retain)


class QuadPass(_pinned.QuadPass):
    """What execute_proposal and the two branches take from the quad (their `quad` keyword), at this width: the pinned pass's
    64 rows and K/V plan, with the SDPA, the head helpers, the conv and the head of this module."""

    project_key_value = staticmethod(project_key_value)
    concatenate_query_heads = staticmethod(concatenate_query_heads)
    head_candidates = staticmethod(head_candidates)

    def attention(self, operations, query, key, value, mask, retain, *, mask_validated=False):
        attend = fold_attention if self.sdpa == 'fold' else pairs_attention
        return attend(operations, query, key, value, mask, retain, mask_validated=mask_validated)

    def convolution(self, operations, mesh, hidden, dynamic, base, *, fp32_intermediates=False,
                    retain_temporaries=None, boundaries=None):
        if fp32_intermediates is not True or not callable(retain_temporaries) or boundaries is None:
            raise ValueError('The quad conv requires exact FP32 arithmetic, a lifetime owner and the user seams')
        if self.conv == 'halves':
            output = halves_convolution(operations, mesh, hidden, dynamic, base, boundaries=boundaries,
                                        retain=retain_temporaries)
        else:
            output = quad_fused_convolution(operations, mesh, hidden, dynamic, base, boundaries=boundaries,
                                            conv=self.conv)
        retain_temporaries(output)
        return output


# ---------------------------------------------------------------------------------------------
# The readback.
# ---------------------------------------------------------------------------------------------

def read_quad_outputs(device, outputs):
    """read_device_outputs for the quad at this width: the reads (chunks x chips x values and indices, plus the replicated
    features), each half merged by the four-card merge_chunk_candidates(block_rows=32) exactly as its pair merges,
    stitched [half 0 | a dummy row | half 1] (the dummy is merged index 31 - block row 32, user 2's anchor - which no user
    slice reads) and split with split_selection(block_width=64): four users' (features, candidates, scores)."""
    import torch

    from dflash_packed_proposal import report_rejected_outputs, split_selection
    from draft_shared_head_tp import merge_chunk_candidates

    operations = device.operations
    chips = tp_shapes.chip_count()
    halves = ([], [])
    for chunk in outputs.chunks:
        values = operations.get_device_tensors(chunk['values'])
        indices = operations.get_device_tensors(chunk['indices'])
        if len(values) != chips or len(indices) != chips:
            raise AssertionError('%s learned head shards required' % tp_shapes.all_chips())
        for chip in range(chips):
            host_values = operations.to_torch(values[chip]).float().reshape(ROWS, 16)
            host_indices = operations.to_torch(indices[chip]).long().reshape(ROWS, 16)
            for half in range(GROUP_USERS):
                rows = slice(32 * half, 32 * (half + 1))
                halves[half].append(dict(chip=chip, start=chunk['start'], stop=chunk['stop'],
                                         values=host_values[rows], indices=host_indices[rows]))

    def merged(host_chunks):
        # S2 v86: a refused half is reported per chip (dflash_packed_proposal.REJECTED_OUTPUTS_LINE) and raised unchanged.
        try:
            return merge_chunk_candidates(host_chunks, block_rows=32)
        except ValueError as failure:
            report_rejected_outputs(device, outputs, host_chunks, failure)
            raise

    merged_halves = [merged(list(part)) for part in halves]
    candidates = torch.cat([merged_halves[0][0], torch.zeros_like(merged_halves[0][0][:, :1]), merged_halves[1][0]], dim=1)
    unary = torch.cat([merged_halves[0][1], torch.zeros_like(merged_halves[0][1][:, :1]), merged_halves[1][1]], dim=1)
    parts = [operations.to_torch(value) for value in operations.get_device_tensors(outputs.projected)]
    if len(parts) != chips or any(not torch.equal(parts[0], other) for other in parts[1:]):
        raise AssertionError('Replicated learned selector features differ')
    hidden = parts[0].reshape(1, ROWS, 256)
    return split_selection(hidden, candidates, unary, USERS, BLOCK, block_width=ROWS)


def select_quad_outputs(device, outputs, seeds, counts):
    """read_quad_outputs, then today's per-user selector (select_packed): the flag-off selection."""
    from dflash_packed_proposal import select_packed

    return select_packed(read_quad_outputs(device, outputs), seeds, counts, device.predecessors, device.successors)


# ---------------------------------------------------------------------------------------------
# The trace.
# ---------------------------------------------------------------------------------------------

class PreparedQuadDFlashProposal(_pinned.PreparedQuadDFlashProposal):
    """The pinned four-user 64-row trace with this module's width: one bucket, (2048,) * 4, built lazily at the first
    prepare_device; placeholders ids (1, 64), the single-user mask (1, 1, 32, 2080) (the pool's pre-trace mask when it holds
    one), rope.q and live_k 2 x (1, 1, 64, 128), and four users' PLACEHOLDER K/V banks (see the module docstring), then an
    eager warm-up, the capture, one blocking replay, the marker and the quad_built ledger point. What it inherits is width
    independent: prepare_device, has_pending, adopt, the mask audit, the shadow audit against the pair traces, close."""

    def __init__(self, devices, *, sdpa=None, conv=None):
        super().__init__(devices, sdpa=sdpa, conv=conv)
        self.quad = QuadPass(self.quad.sdpa, self.quad.conv)

    def _placeholder_banks(self):
        """cached_history: per user (device order) and layer, a k / v pair of zeros, (1, kv, 2048, 128) - the pair's own
        placeholders without live banks. _update copies each user's active bank into them every round. (With live banks the
        bucket binds the pool's active banks instead, the pinned _live_banks.)"""
        import torch

        kv = tp_shapes.active().draft_kv_heads
        return [[{name: self._upload(torch.zeros((1, kv, CONTEXT, HEAD_DIM), dtype=torch.bfloat16)) for name in ('k', 'v')}
                 for _ in device.kv_history.active] for device in self.devices]

    def _bucket(self):
        key = (CONTEXT,) * USERS
        bucket = self.buckets.get(key)
        if bucket is not None:
            self.last_built = False
            return bucket
        from dflash_packed_proposal import packed_identifiers
        from dflash_proposal_trace import borrow_pooled_mask, pool_outputs, traced_pass
        from gdn_multitoken_conv import addresses, release_owned
        quad_host_mask, quad_rope = _pinned.quad_host_mask, _pinned.quad_rope
        operations, device = self.operations, self.devices[0]
        placeholder_mark = len(self.owned)
        bind_live = live_banks_requested()
        try:
            host_mask = quad_host_mask()
            query, live = quad_rope([dict(position=CONTEXT, history_rows=CONTEXT)] * USERS)
            # S2: the pool's pre-trace mask for slots 0-3, the host mask copied in (dflash_proposal_trace
            # .POOLED_MASK_LINE); None without one, and the mask is uploaded below as before.
            pooled_mask = borrow_pooled_mask(device, self.pair_label(), host_mask, operations, self.mesh,
                                             log=log_line)
            bucket = SimpleNamespace(context=key, host_mask=host_mask,
                identifiers=self._upload(packed_identifiers([0] * USERS, BLOCK, block_width=ROWS), identifiers=True),
                mask=pooled_mask if pooled_mask is not None else self._upload(host_mask),
                rope=dict(q=tuple(self._upload(value) for value in query),
                          live_k=tuple(self._upload(value) for value in live)),
                cached_history=self._live_banks() if bind_live else self._placeholder_banks(), trace=None, outputs=None,
                owned=[], tokens=None, consumed=set(), parts=None, live_banks=bind_live)
            if pooled_mask is not None:
                # Borrowed, never in self.owned: protected like the lent banks (_protected), never released.
                bucket.lent_mask = (pooled_mask,)
            bucket.inputs = [bucket.identifiers, bucket.mask, *bucket.rope['q'], *bucket.rope['live_k'],
                *(value for cache in bucket.cached_history for layer in cache for value in layer.values())]
            bucket.addresses = [addresses(operations, value) for value in bucket.inputs]
            device.validated_native_proposal_masks.add(addresses(operations, bucket.mask))
            self.buckets[key] = bucket
            if bind_live:
                from fused_commit import note_live_banks

                note_live_banks(self.pair_label(), bucket.context)
            self._update(bucket, (0,) * USERS)
            transient, retain = device.temporaries(self._protected(bucket))
            try:
                warm = self._execute(bucket, transient, retain)
                # S2 v86: the pool's output set for slots 0-3 (dflash_proposal_trace.POOLED_OUTPUTS_LINE), which
                # every pass then ends by copying into; None without one, and the quad reads its own as before.
                pooled_outputs = pool_outputs(device, self.pair_label(), warm, operations, log=log_line)
                operations.synchronize_device(self.mesh)
            finally:
                release_owned(operations, transient)
            bucket.owned, retain = device.temporaries(self._protected(bucket))
            from attention_batch import capture_operation

            bucket.trace, bucket.outputs = capture_operation(operations, self.mesh,
                lambda: traced_pass(operations, lambda: self._execute(bucket, bucket.owned, retain),
                                    pooled_outputs))
            operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=True)
            note(self.pair_label(), self.quad.sdpa, self.quad.conv)
            import memory_ledger

            # The quad's own buffers only (its placeholders and the capture's retained intermediates): the delta
            # against the previous point is what a fresh quad costs (QUAD_CAPTURE_BYTES_EST until measured).
            memory_ledger.record('quad_built', point='slots=%s' % ','.join(str(slot) for slot in self.pair_label()),
                                 quad_placeholders=list(self.owned), quad_intermediates=list(bucket.owned))
        except BaseException:
            self.buckets.pop(key, None)
            built = locals().get('bucket')
            if built is not None:
                device.validated_native_proposal_masks.discard(addresses(operations, built.mask))
                if built.trace is not None:
                    operations.release_trace(self.mesh, built.trace)
                    built.trace = None
                release_owned(operations, built.owned)
            leaked = self.owned[placeholder_mark:]
            del self.owned[placeholder_mark:]
            release_owned(operations, leaked)
            raise
        self.last_built = True
        return bucket

    def _update(self, bucket, seeds, *, defer_finish=False):
        from dflash_packed_proposal import note_round_b1, packed_identifiers, round_b1_audit_enabled
        from dflash_proposal_trace import _audit_live_key_rope, pair_mask_audit_enabled, pair_mask_refresh_enabled
        from gdn_multitoken_conv import addresses, release_owned
        quad_rope, quad_users = _pinned.quad_rope, _pinned.quad_users
        devices, operations = self.devices, self.operations
        if any(device.history_rows != CONTEXT for device in devices):
            raise ValueError("Quad proposal replay requires every user at the bucket's committed context")
        if any(device.kv_history.pending is not None or device.position - device.history_rows < 0
               for device in devices):
            raise ValueError('Quad proposal replay requires a fully committed matching K/V frontier')
        users = quad_users(devices)
        if pair_mask_audit_enabled():
            self.audit_mask(bucket)
        note_round_b1('quad-update')
        query, live = quad_rope(users)
        if round_b1_audit_enabled():
            # C8, per pair: the uploaded live-key rows [32p, 32p + 32) against live_key_rope's own build.
            for pair, (first, second) in enumerate(PAIRS):
                rows = tuple(table[:, :, 32 * pair:32 * (pair + 1)] for table in live)
                _audit_live_key_rope(rows, [users[first], users[second]], BLOCK)
        sources = [packed_identifiers(list(seeds), BLOCK, block_width=ROWS), *query, *live]
        destinations = [bucket.identifiers, *bucket.rope['q'], *bucket.rope['live_k']]
        if pair_mask_refresh_enabled():
            sources.append(bucket.host_mask)
            destinations.append(bucket.mask)
            self._note_refresh(bucket)
        for value, destination in zip(sources, destinations, strict=True):
            payload = operations.from_torch(value, dtype=destination.dtype, layout=destination.layout,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(payload, destination)
        protected = self._protected()
        for device in devices:
            protected.extend((*device.kv_history.owned, *device.kv_history.borrowed))
        protected.extend(getattr(bucket, 'lent_mask', ()))
        owned, retain = devices[0].temporaries(protected)
        kv = tp_shapes.active().draft_kv_heads

        live = getattr(bucket, 'live_banks', False)

        def copy_cache():
            # Without live banks: every user's active bank (whichever side of the four-card slide's swap it is on) into the
            # quad's own placeholder for that user, layer and k / v: the pair's copy_cache without live banks. With them: the
            # pinned copy_cache, at this width - a bank that IS the pool's active one is read in place, and only a device whose
            # live bank sits on the pool's spare side (an odd swap count before its first in-place commit) is copied into the
            # pool's active bank, which the quad reads (the pair's own normalisation, dflash_proposal_trace._update).
            normalised = 0
            for device, cache in zip(devices, bucket.cached_history, strict=True):
                for active, destination in zip(device.kv_history.active, cache, strict=True):
                    for name in ('k', 'v'):
                        if live and active[name] is destination[name]:
                            continue
                        value = retain(operations.slice(active[name], (0, 0, 0, 0), (1, kv, CONTEXT, HEAD_DIM)))
                        operations.copy(value, destination[name])
                        normalised += 1
            if live and normalised:
                from fused_commit import LIVE_BANKS_NORMALISED, log_line as fused_log

                fused_log('%s pair=%s normalised=%d' % (LIVE_BANKS_NORMALISED, self.pair_label(), normalised))
        if defer_finish:
            try:
                copy_cache()
            except BaseException:
                release_owned(operations, owned)
                raise
            return owned
        try:
            copy_cache()
            operations.synchronize_device(self.mesh)
            if [addresses(operations, value) for value in bucket.inputs] != bucket.addresses:
                raise AssertionError('Prepared quad proposal input addresses moved')
        finally:
            release_owned(operations, owned)

    def finish(self, which, count):
        """The pair's finish() for four users: the first call checks the addresses, releases the transients and selects
        (unless the round's batched selection was adopted); each user reads its own tokens."""
        if self._pending is None:
            raise ValueError('No prepared quad proposal is pending')
        seeds, bucket, owned = self._pending
        if bucket.tokens is None:
            from gdn_multitoken_conv import addresses, release_owned

            operations = self.operations
            moved = [addresses(operations, value) for value in bucket.inputs] != bucket.addresses
            release_owned(operations, owned)
            if moved:
                raise AssertionError('Prepared quad proposal input addresses moved')
            bucket.tokens = select_quad_outputs(self.devices[0], bucket.outputs, seeds, (BLOCK - 1,) * USERS)
        tokens = bucket.tokens[which]
        bucket.consumed.add(which)
        if len(bucket.consumed) == USERS:
            self._pending = None
            bucket.tokens, bucket.consumed, bucket.parts = None, set(), None
        return tokens[:count]

    def collect(self):
        """QWEN_FAST_ROUND_B1 (C1), as the pair's: the address check, the transient release and the readback, with the seeds
        and counts finish() would select with. The parts are kept for the shadow audit."""
        if self._pending is None:
            raise ValueError('No prepared quad proposal is pending')
        seeds, bucket, owned = self._pending
        if bucket.tokens is not None or bucket.consumed:
            raise ValueError('This quad proposal was already selected')
        from gdn_multitoken_conv import addresses, release_owned
        read_audit = _pinned.read_audit
        operations = self.operations
        self._pending = (seeds, bucket, [])
        moved = [addresses(operations, value) for value in bucket.inputs] != bucket.addresses
        release_owned(operations, owned)
        if moved:
            raise AssertionError('Prepared quad proposal input addresses moved')
        bucket.parts = read_quad_outputs(self.devices[0], bucket.outputs)
        if isinstance(self._audit, list):
            # QWEN_FAST_QUAD_DRAFT_AUDIT: the pairs' readback and every raw read the audit compares, here and not after
            # the selection (quad_draft.PreparedQuadDFlashProposal.collect). A failure is the audit's verdict.
            try:
                self._audit_reads = read_audit(self, self._audit)
            except Exception as failure:  # noqa: BLE001 - run_audit logs it as equal=0
                self._audit_reads = 'read-error:%s' % type(failure).__name__
        return dict(parts=bucket.parts, seeds=seeds, counts=(BLOCK - 1,) * USERS)

    def audit_selection(self):
        """QWEN_FAST_ROUND_B1_AUDIT (C1): the tokens finish() would have selected itself."""
        if self._pending is None:
            raise ValueError('No prepared quad proposal is pending')
        seeds, bucket, _ = self._pending
        return select_quad_outputs(self.devices[0], bucket.outputs, seeds, (BLOCK - 1,) * USERS)


# ---------------------------------------------------------------------------------------------
# The coordinator's engage-time checks.
# ---------------------------------------------------------------------------------------------

def refusal(devices, batched, environ=None):
    """Why the four-card quad cannot serve these four packable devices (a permanent configuration reason: the coordinator
    disables it with the reason), or None."""
    environ = os.environ if environ is None else environ
    if tp_shapes.chip_count(environ) != TP4:
        return 'the four-card quad serves QWEN_FAST_TP=4 only'
    missing = missing_requirements(environ)
    if missing:
        return 'requires ' + ','.join('%s=1' % name for name in missing)
    needs = live_banks_missing(environ)
    if needs:
        return '%s=1 needs %s (the live bank must never move)' % (LIVE_BANKS_FLAG, ','.join('%s=1' % name for name in needs))
    if batched is None:
        return 'requires the batched selection (QWEN_FAST_ROUND_B1=1)'
    first = devices[0]
    if any(device.operations is not first.operations or device.mesh is not first.mesh for device in devices):
        return 'the four devices do not share one mesh'
    if tp_shapes.mesh_width(first.mesh, environ) is None:
        return 'the four devices are not on the (1, %d) mesh' % TP4
    if any(getattr(device, 'block_rows', None) != BLOCK for device in devices):
        return 'the four devices are not T16'
    if any(not getattr(device, 'fused_convolution', False) for device in devices):
        return 'requires the fused learned convolution'
    if any(device.layers is not first.layers or device.predecessors is not first.predecessors
           or device.successors is not first.successors for device in devices):
        return 'the four devices do not share one draft weight set'
    conv = _pinned.conv_mode(environ)
    if grid_fits(first.mesh, conv) is False:
        return 'the compute grid cannot hold QWEN_FAST_QUAD_CONV=%s' % conv
    _pinned.sdpa_mode(environ)
    return None
