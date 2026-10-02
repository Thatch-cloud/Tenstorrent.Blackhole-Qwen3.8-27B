"""The drafter's 5,120-wide RMS norms on a wide core grid at four cards (QWEN_FAST_TP4_DRAFT_WIDE, default off).

WHY. The only TP4 device profile (v170, trace 0 and the round after it) shows the two drafter pair passes at
24.75 ms per four-user round, 640 ops each at about 20 us, against a weight-read floor of about 2.15 ms per pass.
The ops' own core counts are the cheap part of that: the drafter's hidden-width RMS norm (one per attention branch,
one per MLP branch, the feature norm and the final norm: 21 per pass) runs interleaved, where the program gives one
core per 32-row tile row, so a 32 or 64-row block over 5,120 columns (160 tiles per row) sits on one or two cores:
LayerNorm 21 ops, 0.943 ms per pass (45 us each). The per-head norms of the query and key heads (128 wide, one tile
row per head) are already one core per head, so they are not touched here.

WHAT. rms_norm() is the call the drafter modules made (operations.rms_norm with the same arguments) when the flag is
off, byte for byte. With QWEN_FAST_TP4_DRAFT_WIDE=1 a (1, 1, rows, 5120) bfloat16 tiled block with the drafter's
row-major (1, 1, 160, 32) weight runs the sharded program instead: the block is width-sharded in L1 over WIDE_GRID
(20 cores, 8 tiles of the row each), LayerNormShardedMultiCoreProgramConfig with block_h = rows / 32, and the result
comes back to the interleaved DRAM layout the next op expects. Anything else (another width, dtype, layout or row
count, a weight in another layout) takes the plain call and logs why, once.

EXACTNESS. A drafter change never changes the served text, because the target verifies every draft token; what it
can change is the proposals (the sharded norm sums its squares per core and combines the partial sums, where the
interleaved op sums them in one core, so the last bit of a row's statistic can differ), hence the acceptance rate.
So the check is hardware only: the job runs the same coding prompts with the flag on and off and compares the
accepted-prefix sequences (speed_window_compare --strict-concurrent-prefixes: exit 0 means every proposal that was
verified is the same) and the [ACCEPT] mean. CPU tests prove the plan covers each shape exactly and that the flag-off
path is the plain call.

NOT DONE. The drafter's head ops (NlpCreateHeads and NLPConcatHeads, one core, 20 us each, 10 per pass) are left as
they are: they have no multi-core variant for an interleaved input, and every exact replacement is more launches
than the 0.17 ms per pass they cost. The 80-core fused conv grid is hard-coded in the kernel (D2 is the norms only).

Stdlib only, py 3.7. The flag is strict (unset or 0 is off, 1 is on, anything else raises) and, like the vglue
levers, refused at the pair: this exists for four cards, and the pair's launches stay what they were.
"""

import os
import sys

import tp_shapes

FLAG = 'QWEN_FAST_TP4_DRAFT_WIDE'
TILE = 32
HIDDEN = 5120
WEIGHT_SHAPE = (1, 1, HIDDEN // TILE, TILE)
# The sharded grid (x, y): 20 cores, 160 tiles / 20 = 8 tiles (block_w) per core.
WIDE_GRID = (5, 4)
# The row counts the plan accepts: the drafter's proposal block is 32 rows (one user) or 64 (a pair of packed users); the
# feature projection runs 32-row chunks. 128 rows of one core's 8 tiles is 64 KB of bfloat16 input plus as much output.
MAX_ROWS = 128
# The destination register limit of the sharded norm's per-subblock work (subblock_w x 1 tiles).
SUBBLOCK_CAP = 4

ENGAGED = '[PINDIAG] tp4 draft wide engaged'
FALLBACK = '[PINDIAG] tp4 draft wide fell back'

RUNTIME_FILES = ('draft_wide_tp.py',)

# What this process did, by site: how many calls ran wide, how many fell back (a test reads and clears it).
STATS = {'wide': 0, 'plain': 0, 'fallback': 0}
_LOGGED = set()


def enabled(environ=None):
    """QWEN_FAST_TP4_DRAFT_WIDE: strict 0 or 1; on at the pair raises."""
    source = os.environ if environ is None else environ
    value = source.get(FLAG)
    if value is None or value == '0':
        return False
    if value != '1':
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    return True


def plan(rows, width=HIDDEN, grid=WIDE_GRID):
    """The sharded norm's geometry for a (rows, width) block on `grid`, or ValueError. block_w x cores is exactly the row's
    tiles (no padded core), the shard is (rows, block_w * 32), and subblock_w is the largest divisor of block_w within
    SUBBLOCK_CAP."""
    if type(rows) is not int or rows < TILE or rows % TILE or rows > MAX_ROWS:
        raise ValueError('rows must be a whole number of %d-row tiles up to %d, got %r' % (TILE, MAX_ROWS, rows))
    if type(width) is not int or width <= 0 or width % TILE:
        raise ValueError('width must be a whole number of tiles, got %r' % (width,))
    x, y = grid
    cores = x * y
    tiles = width // TILE
    if cores < 1 or tiles % cores:
        raise ValueError('%d tiles do not split evenly over %d cores' % (tiles, cores))
    block_w = tiles // cores
    subblock_w = max(d for d in range(1, min(block_w, SUBBLOCK_CAP) + 1) if block_w % d == 0)
    return dict(rows=rows, width=width, grid=(x, y), cores=cores, block_h=rows // TILE, block_w=block_w,
                subblock_w=subblock_w, shard=(rows, block_w * TILE))


def ineligible(operations, tensor, weight):
    """Why this call cannot take the wide path (a short reason), or None."""
    shape = tuple(tensor.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != HIDDEN:
        return 'shape %r is not (1, 1, rows, %d)' % (shape, HIDDEN)
    try:
        plan(shape[2])
    except ValueError as failure:
        return str(failure)
    if tensor.dtype != operations.bfloat16 or tensor.layout != operations.TILE_LAYOUT:
        return 'input is not bfloat16 tiled'
    if weight is None or tuple(weight.shape) != WEIGHT_SHAPE or weight.layout != operations.ROW_MAJOR_LAYOUT:
        return 'weight is not the row-major %r layout' % (WEIGHT_SHAPE,)
    return None


def diagnostic(text):
    """One [PINDIAG] line into the server log: loguru where it exists, stderr otherwise (packed_verifier.diagnostic's way). Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


def note(kind, site, text):
    """One marker line per (kind, site, text), so a loop of 21 norms prints its three or four distinct lines."""
    key = (kind, site, text)
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    diagnostic('%s site=%s %s' % (kind, site, text))


def sharded_memory_config(operations, found):
    x, y = found['grid']
    cores = operations.CoreRangeSet({operations.CoreRange(operations.CoreCoord(0, 0), operations.CoreCoord(x - 1, y - 1))})
    return operations.MemoryConfig(operations.TensorMemoryLayout.WIDTH_SHARDED, operations.BufferType.L1,
                                   operations.ShardSpec(cores, list(found['shard']), operations.ShardOrientation.ROW_MAJOR))


def program_config(operations, found):
    return operations.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=found['grid'], subblock_w=found['subblock_w'], block_h=found['block_h'],
        block_w=found['block_w'], inplace=False)


def rms_norm(operations, tensor, *, epsilon, weight, compute_kernel_config, memory_config=None, site='hidden', environ=None):
    """operations.rms_norm(tensor, epsilon=..., weight=..., compute_kernel_config=..., memory_config=...) for the drafter's
    hidden-width norms: the same call with the flag off, the wide sharded program with it on (see the module text)."""
    def plain():
        return operations.rms_norm(tensor, epsilon=epsilon, weight=weight, compute_kernel_config=compute_kernel_config,
                                   memory_config=memory_config)

    if not enabled(environ):
        return plain()
    reason = ineligible(operations, tensor, weight)
    if reason is None and memory_config != operations.DRAM_MEMORY_CONFIG:
        reason = 'the caller asked for a layout other than interleaved DRAM'
    if reason is not None:
        STATS['fallback'] += 1
        note(FALLBACK, site, reason)
        return plain()
    found = plan(tuple(tensor.shape)[2])
    sharded = sharded_memory_config(operations, found)
    program = program_config(operations, found)
    local = operations.to_memory_config(tensor, sharded)
    normalized = None
    try:
        normalized = operations.rms_norm(local, epsilon=epsilon, weight=weight, compute_kernel_config=compute_kernel_config,
                                         program_config=program, memory_config=sharded)
        result = operations.to_memory_config(normalized, memory_config)
    finally:
        if normalized is not None:
            operations.deallocate(normalized)
        operations.deallocate(local)
    STATS['wide'] += 1
    note(ENGAGED, site, 'rows=%d cores=%d block_w=%d subblock_w=%d' % (found['rows'], found['cores'], found['block_w'],
                                                                       found['subblock_w']))
    return result

