"""grid13_tp (QWEN_FAST_GRID13, M1): the matmul program grids re-tuned for the 13 x 10 compute grid. Held here, on the CPU, on a fake ttnn that needs no torch:

  - the flags: strict, the sub-switches, the CFG grammar, everything at the pair refused, the audit and the CFG refused without the lever;
  - the plans: the M block that minimises the busiest core's rows (swept over every row count), the strict-gain rule, a device narrower than 13 x 10 or unreadable;
  - the prefill shim: the PINDED tp_common function (transcribed below, byte for byte in its calls) is called as it is, and the one device call it makes differs from the
    unwrapped call in exactly the keywords the plan names (config, num_workers_per_link) and in exactly those fields of the config; on 11 x 10 it differs in nothing;
  - the verify wrapper: the pinned builder makes both configs, the fields the T1 #11 rule keeps are checked, other sites and the 32-row configs are not touched;
  - the drafter commit helper: only compute_with_storage_grid_size moves;
  - the audit: a fake whose output does not depend on the partition is exact and logs the passing line, a fake whose output does is a logged mismatch and an AssertionError
    (the negative control), the budget, the freed temporaries, and nothing synchronized or read back inside a trace capture (capture_rules);
  - flag off: nothing is wrapped, install() returns [], the pinned objects are the ones that were there.

What this cannot hold is the device: that a 12-wide minimal matmul at M block 6 runs, and is bit-equal to the 8-wide one, is the audited attach (QWEN_FAST_GRID13_AUDIT=1).

    python3 -B -m unittest test_grid13_tp      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_rules  # noqa: E402
import grid13_smoke  # noqa: E402
import grid13_tp as grid13  # noqa: E402

TP4 = {'QWEN_FAST_TP': '4'}


# ---------------------------------------------------------------------------------------------
# The fake ttnn.
# ---------------------------------------------------------------------------------------------

class Coord(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return isinstance(other, Coord) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return 'Coord(%d, %d)' % (self.x, self.y)


class Record(object):
    """A config object with the keyword arguments as attributes (MinimalMatmulConfig, the 1D matmul program config)."""

    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)

    def fields(self):
        return dict(vars(self))


class MinimalMatmulConfig(Record):
    pass


class Program1D(Record):
    def __init__(self, **fields):
        grid = fields['compute_with_storage_grid_size']
        fields['compute_with_storage_grid_size'] = grid if isinstance(grid, Coord) else Coord(*grid)
        Record.__init__(self, **fields)


class Mesh(object):
    def __init__(self, columns=13, rows=10):
        self.columns, self.rows = columns, rows

    def compute_with_storage_grid_size(self):
        return Coord(self.columns, self.rows)


class Host(object):
    def __init__(self, value):
        self.value = value
        self.shape = (len(value),)

    def tolist(self):
        return list(self.value)


class Tensor(object):
    dtype = 'bf16'
    layout = 'tile'

    counter = [0]

    def __init__(self, shape, chips, mesh, name='t'):
        Tensor.counter[0] += 1
        self.shape = tuple(shape)
        self.chips = list(chips)                    # one list of ints per chip
        self.mesh = mesh
        self.name = name
        self.freed = False
        self.serial = Tensor.counter[0]

    def device(self):
        return self.mesh

    def memory_config(self):
        return 'dram'

    def __eq__(self, other):
        return isinstance(other, Tensor) and (self.shape, self.chips) == (other.shape, other.chips)

    def __ne__(self, other):
        return not self == other

    __hash__ = object.__hash__


def dummy(mesh, shape, seed):
    return Tensor(shape, [[seed + chip, chip] for chip in range(4)], mesh)


class FakeTTNN(ModuleType):
    """The slice of ttnn the transcribed tp_common functions and the lever use, over host values that do not depend on the grid (unless `inexact`)."""

    def __init__(self):
        ModuleType.__init__(self, 'fake_ttnn')
        self.inexact = False
        self.calls = []                 # every all_gather_minimal_matmul_async call's keywords
        self.linears = []               # every linear's (program_config,)
        self.matmuls = []
        self.freed = []
        self.synchronized = 0
        self.captured = False
        self.DRAM_MEMORY_CONFIG = 'dram'
        self.L1_MEMORY_CONFIG = 'l1'
        self.bfloat16 = 'bf16'
        self.float32 = 'fp32'
        self.MinimalMatmulConfig = MinimalMatmulConfig
        self.CoreCoord = Coord
        self.MatmulMultiCoreReuseMultiCast1DProgramConfig = Program1D
        self.experimental = SimpleNamespace(all_gather_minimal_matmul_async=self.agmm)

    def agmm(self, **kw):
        self.calls.append(dict(kw))
        x, weight, config = kw['input_tensor'], kw['weight_tensor'], kw['config']
        seed = (tuple(map(tuple, x.chips)), tuple(map(tuple, weight.chips)), config.K_block_size, bool(kw.get('fuse_swiglu')), str(kw.get('fused_activation')))
        if self.inexact:
            seed += (config.M_block_size,)
        width = weight.shape[-1] // (2 if kw.get('fuse_swiglu') else 1)
        return [Tensor((1, 1, x.shape[-2], width), [[hash(seed + (chip,)) % 65521] for chip in range(4)], x.mesh, 'agmm')]

    def reshape(self, x, shape):
        return Tensor(shape, x.chips, x.mesh, x.name)

    def to_memory_config(self, x, config):
        return x

    def linear(self, x, weight, compute_kernel_config=None, program_config=None, memory_config=None):
        self.linears.append(program_config)
        seed = (tuple(map(tuple, x.chips)), tuple(map(tuple, weight.chips)), program_config.in0_block_w)
        if self.inexact:
            seed += (program_config.per_core_N,)
        return Tensor((1, 1, x.shape[-2], weight.shape[-1]), [[hash(seed + (chip,)) % 65521] for chip in range(4)], x.mesh, 'linear')

    def matmul(self, x, weight, dtype=None, program_config=None, compute_kernel_config=None, memory_config=None):
        self.matmuls.append(program_config)
        seed = (tuple(map(tuple, x.chips)), tuple(map(tuple, weight.chips)), program_config.in0_block_w)
        if self.inexact:
            seed += (program_config.per_core_N, tuple((program_config.compute_with_storage_grid_size.x, program_config.compute_with_storage_grid_size.y)))
        return Tensor((1, 1, x.shape[-2], weight.shape[-1]), [[hash(seed + (chip,)) % 65521] for chip in range(4)], x.mesh, 'matmul')

    def deallocate(self, tensor):
        tensor.freed = True
        self.freed.append(tensor)

    def synchronize_device(self, mesh=None):
        self.synchronized += 1

    def get_device_tensors(self, tensor):
        return [Host(chip) for chip in tensor.chips]

    def to_torch(self, shard):
        return shard

    def begin_trace_capture(self, *args, **kwargs):
        self.captured = True
        return 1

    def end_trace_capture(self, *args, **kwargs):
        self.captured = False


# The pinned tp_common functions as they are in the image, transcribed for this fake (their calls are the ones the shim is held against).
TP_COMMON_SOURCE = '''
import math

TILE_SIZE = 32


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def create_matmul_1d_decode_progcfg(m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8):
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    per_core_k = _find_largest_divisor(k_tiles)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=per_core_k,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def matmul_1d_decode(x, weight, decode_1d_progcfg, compute_cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG):
    x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.linear(
        x_il,
        weight,
        compute_kernel_config=compute_cfg,
        program_config=decode_1d_progcfg,
        memory_config=out_memory_config,
    )
    if x_il is not x:
        ttnn.deallocate(x_il)
    return out


def agmm_k_block_size(k_local, default=8):
    k_tiles = k_local // TILE_SIZE
    b = 1 << (min(default, max(1, k_tiles)).bit_length() - 1)
    while b > 1 and k_tiles % b:
        b //= 2
    return b


def all_gather_matmul_prefill(
    x,
    weight,
    tt_ccl,
    compute_cfg,
    topology,
    grid=(7, 9),
    cluster_axis=1,
    fused_activation=None,
    out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    num_links = 2
    grid = (8, grid[1])
    workers = grid[0] // num_links
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=4,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=8,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        fused_activation=fused_activation,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
    )[0]

    return out


def all_gather_swiglu_prefill(
    x, weight, tt_ccl, compute_cfg, topology, grid=(7, 9), cluster_axis=1, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    num_links = 2
    grid = (8, grid[1])
    workers = grid[0] // num_links
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=16,
        subblock_h=1,
        subblock_w=4,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    return ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        compute_kernel_config=compute_cfg,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=True,
        num_workers_per_link=workers,
        num_buffers_per_channel=8,
        fuse_swiglu=True,
    )[0]
'''


class Collectives(object):
    def get_and_cycle_ag_semaphore_handles(self, axis):
        return ('semaphores', axis)


def make_tp_common(ops):
    module = ModuleType('fake_tp_common')
    module.ttnn = ops
    exec(compile(TP_COMMON_SOURCE, 'fake_tp_common.py', 'exec'), module.__dict__)
    return module


class Rig(object):
    """A fake device + tp_common + the lever's environment, with every logged line collected."""

    def __init__(self, columns=13, rows=10, environ=None, sync_guard=False):
        self.mesh = Mesh(columns, rows)
        self.ops = FakeTTNN()
        self.rules = None
        self.target = self.ops
        if sync_guard:
            self.target = capture_rules.guard(self.ops)
            self.rules = self.target.rules
        self.tp_common = make_tp_common(self.target)
        self.environ = dict(TP4)
        self.environ.update(environ or {})
        self.lines = []
        self.patchers = [patch.object(grid13, 'log_line', side_effect=self.lines.append), patch.dict(os.environ, self.environ, clear=True)]

    def __enter__(self):
        for patcher in self.patchers:
            patcher.start()
        grid13.reset()
        return self

    def __exit__(self, *exc):
        for patcher in reversed(self.patchers):
            patcher.stop()
        grid13.reset()
        return False

    def install(self):
        return grid13.install(module=self.tp_common)

    def matching(self, text):
        return [line for line in self.lines if text in line]

    def x(self, rows=2048, width=1280):
        return dummy(self.mesh, (1, rows, width), 7)

    def weight(self, width):
        return dummy(self.mesh, (1, 1, 5120, width), 11)

    def prefill(self, name='all_gather_matmul_prefill', rows=2048, width=4128, **kwargs):
        function = getattr(self.tp_common, name)
        return function(self.x(rows), self.weight(width), Collectives(), 'cfg', 'ring', **kwargs)

    def last_call(self):
        return self.ops.calls[-1]


def unwrapped(function):
    return getattr(function, 'grid13_of', function)


# ---------------------------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_off_is_none_and_nothing_else_is_read(self):
        self.assertIsNone(grid13.parse({}))
        self.assertIsNone(grid13.parse({grid13.FLAG: '0'}))
        self.assertFalse(grid13.enabled({}))

    def test_one_is_every_sub_switch_and_a_list_names_some(self):
        self.assertEqual(grid13.parse(dict(TP4, **{grid13.FLAG: '1'})).subs, frozenset(grid13.SUBSWITCHES))
        self.assertEqual(grid13.parse(dict(TP4, **{grid13.FLAG: 'prefill'})).subs, frozenset(['prefill']))
        self.assertEqual(grid13.parse(dict(TP4, **{grid13.FLAG: 'verify,prefill'})).subs, frozenset(['verify', 'prefill']))

    def test_malformed_values_are_refused(self):
        for bad in ('2', 'yes', '', 'bogus', 'prefill,prefill', 'prefill,', ',prefill', 'prefill,bogus', 'ALL'):
            with self.assertRaises(ValueError, msg=repr(bad)):
                grid13.parse(dict(TP4, **{grid13.FLAG: bad}))

    def test_the_pair_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'TP4 lever'):
            grid13.parse({grid13.FLAG: '1'})

    def test_audit_and_cfg_need_the_lever_and_the_audit_is_strict(self):
        with self.assertRaisesRegex(ValueError, 'audit nothing'):
            grid13.parse(dict(TP4, **{grid13.AUDIT_FLAG: '1'}))
        with self.assertRaisesRegex(ValueError, 'configure nothing'):
            grid13.parse(dict(TP4, **{grid13.CFG_FLAG: 'out=3'}))
        self.assertIsNone(grid13.parse(dict(TP4, **{grid13.AUDIT_FLAG: '0'})))
        for bad in ('2', 'true', ''):
            if bad == '':
                continue
            with self.assertRaises(ValueError):
                grid13.parse(dict(TP4, **{grid13.FLAG: '1', grid13.AUDIT_FLAG: bad}))
        self.assertTrue(grid13.parse(dict(TP4, **{grid13.FLAG: '1', grid13.AUDIT_FLAG: '1'})).audit)

    def test_the_cfg_grammar(self):
        self.assertEqual(grid13.parse_config(''), grid13.DEFAULT_CONFIG)
        config = grid13.parse_config('agmm=12x6,out=4,gdn_in=2,attn_in=3,width=13')
        self.assertEqual((config.agmm_cols, config.agmm_m, config.out, config.gdn_in, config.attn_in, config.width), (12, 6, 4, 2, 3, 13))
        self.assertEqual(grid13.parse_config('agmm=10').agmm_cols, 10)
        self.assertIsNone(grid13.parse_config('agmm=10').agmm_m)
        for bad in ('agmm=7', 'agmm=9', 'agmm=12x', 'agmm=12x0', 'agmm=x6', 'out=0', 'out=3,out=4', 'bogus=1', 'out', 'out=a', 'agmm=12,agmm=10', 'width=-1', 'agmm=6'):
            with self.assertRaises(ValueError, msg=bad):
                grid13.parse_config(bad)


class PlanTests(unittest.TestCase):
    def test_the_m_block_for_a_2048_row_chunk_is_six_on_twelve_columns(self):
        self.assertEqual(grid13.pick_m_block(64, 12), 6)
        self.assertEqual(grid13.rows_per_core(64, 12, 6), 6)
        self.assertEqual(grid13.rows_per_core(64, 8, 4), 8)           # the served gdn_in / attn_in
        self.assertEqual(grid13.rows_per_core(64, 8, 8), 8)           # the served SwiGLU

    def test_the_pick_is_the_minimum_over_every_block_for_every_row_count(self):
        for m_tiles in range(1, 130):
            for columns in (8, 10, 12):
                block = grid13.pick_m_block(m_tiles, columns)
                self.assertTrue(1 <= block <= grid13.AGMM_M_CAP)
                best = min(grid13.rows_per_core(m_tiles, columns, other) for other in range(1, grid13.AGMM_M_CAP + 1))
                self.assertEqual(grid13.rows_per_core(m_tiles, columns, block), best, (m_tiles, columns))
                # of equal rows, the largest block
                for other in range(block + 1, grid13.AGMM_M_CAP + 1):
                    self.assertGreater(grid13.rows_per_core(m_tiles, columns, other), best)

    def test_the_plan_for_each_served_call_on_the_unlocked_device(self):
        for served_block, name in ((4, 'gdn_in / attn_in'), (8, 'SwiGLU')):
            plan = grid13.agmm_plan(64, 8, 9, served_block, (13, 10))
            self.assertNotIsInstance(plan, str, name)
            self.assertEqual((plan.columns, plan.rows, plan.m_block, plan.workers), (12, 9, 6, 6), name)
            self.assertEqual(plan.rows_per_core, 6)
            self.assertEqual(plan.served_rows_per_core, 8)
            self.assertEqual(plan.columns, plan.workers * grid13.LINKS)

    def test_a_plan_is_strictly_fewer_rows_a_core_or_a_declined_reason(self):
        for m_tiles in range(1, 130):
            for served_block in (4, 8):
                plan = grid13.agmm_plan(m_tiles, 8, 9, served_block, (13, 10))
                served = grid13.rows_per_core(m_tiles, 8, served_block)
                if isinstance(plan, str):
                    self.assertIsInstance(plan, grid13.Declined, (m_tiles, served_block))          # nothing to gain is not a refusal
                    self.assertTrue(m_tiles < grid13.MIN_M_TILES or 'no gain' in plan, plan)
                    if m_tiles >= grid13.MIN_M_TILES:
                        self.assertLessEqual(grid13.rows_per_core(m_tiles, 12, grid13.pick_m_block(m_tiles, 12)), served)
                else:
                    self.assertGreaterEqual(m_tiles, grid13.MIN_M_TILES)
                    self.assertLess(plan.rows_per_core, served)
                    self.assertEqual(plan.rows_per_core, grid13.rows_per_core(m_tiles, plan.columns, plan.m_block))

    def test_a_narrow_short_or_unreadable_device_has_no_plan(self):
        for grid in ((11, 10), (12, 10), (13, 9), (8, 10), None):
            plan = grid13.agmm_plan(64, 8, 9, 4, grid)
            self.assertIsInstance(plan, str, grid)
            self.assertNotIsInstance(plan, grid13.Declined, grid)             # a refusal: it fails a smoke
        self.assertIn('11x10', grid13.agmm_plan(64, 8, 9, 4, (11, 10)))
        self.assertIn('cannot be read', grid13.agmm_plan(64, 8, 9, 4, None))

    def test_the_cfg_overrides_the_columns_and_the_block(self):
        plan = grid13.agmm_plan(64, 8, 9, 4, (13, 10), grid13.parse_config('agmm=10x7'))
        self.assertEqual((plan.columns, plan.m_block, plan.workers, plan.rows_per_core), (10, 7, 5, 7))
        for refused in (grid13.agmm_plan(64, 8, 9, 4, (13, 10), grid13.parse_config('agmm=14')),         # wider than the device
                        grid13.agmm_plan(64, 8, 10, 4, (13, 10))):                                         # no row left for the mux cores
            self.assertIsInstance(refused, str)
            self.assertNotIsInstance(refused, grid13.Declined)

    def test_wide_grid_is_rows_as_wide_as_the_device(self):
        self.assertEqual(grid13.wide_grid((13, 10), 54), (13, 5))
        self.assertEqual(grid13.wide_grid((13, 10), 80), (13, 7))
        self.assertEqual(grid13.wide_grid((13, 10), 7), (7, 1))
        with self.assertRaises(ValueError):
            grid13.wide_grid((13, 10), 131)
        with self.assertRaises(ValueError):
            grid13.wide_grid((13, 10), 0)
        for cores in range(1, 131):
            width, rows = grid13.wide_grid((13, 10), cores)
            self.assertGreaterEqual(width * rows, cores)
            self.assertLess(width * (rows - 1), cores)


class FlagOffTests(unittest.TestCase):
    def test_nothing_is_wrapped_and_install_returns_nothing(self):
        with Rig() as rig:
            before = dict((name, getattr(rig.tp_common, name)) for name in ('all_gather_matmul_prefill', 'all_gather_swiglu_prefill', 'matmul_1d_decode'))
            self.assertEqual(rig.install(), [])
            for name, function in before.items():
                self.assertIs(getattr(rig.tp_common, name), function, name)
            rig.prefill()
            self.assertEqual(rig.lines, [])
            call = rig.last_call()
            self.assertEqual((call['config'].M_block_size, call['num_workers_per_link'], call['config'].compute_with_storage_grid_size), (4, 4, Coord(8, 9)))

    def test_flag_zero_is_off_too(self):
        with Rig(environ={grid13.FLAG: '0'}) as rig:
            self.assertEqual(rig.install(), [])

    def test_the_model_tree_missing_is_not_an_error(self):
        with Rig(environ={grid13.FLAG: '1'}):
            with patch('importlib.import_module', side_effect=ImportError('no model tree', name='models')):
                self.assertEqual(grid13.install(), [])


class PrefillShimTests(unittest.TestCase):
    def call_both(self, rig, name, **kwargs):
        """The wrapped function's device call, and the pinned function's own, on the very same tensors."""
        function = getattr(rig.tp_common, name)
        x, weight = rig.x(), rig.weight(kwargs.pop('width', 4128))
        rig.ops.calls[:] = []
        function(x, weight, Collectives(), 'cfg', 'ring', **kwargs)
        mine = rig.last_call()
        rig.ops.calls[:] = []
        unwrapped(function)(x, weight, Collectives(), 'cfg', 'ring', **kwargs)
        return mine, rig.last_call()

    def test_install_wraps_the_two_prefill_functions_and_a_second_install_wraps_nothing(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            originals = [rig.tp_common.all_gather_matmul_prefill, rig.tp_common.all_gather_swiglu_prefill]
            changed = rig.install()
            self.assertEqual(sorted(name for _namespace, name, _old in changed), ['all_gather_matmul_prefill', 'all_gather_swiglu_prefill'])
            self.assertEqual([old for _namespace, _name, old in sorted(changed, key=lambda row: row[1])], originals)
            self.assertIs(rig.tp_common.all_gather_matmul_prefill.grid13_of, originals[0])
            self.assertEqual(rig.install(), [])
            for namespace, name, old in changed:                              # what tp_addresses.uninstall does
                namespace[name] = old
            self.assertIs(rig.tp_common.all_gather_matmul_prefill, originals[0])

    def test_only_the_named_sub_switch_is_wrapped(self):
        with Rig(environ={grid13.FLAG: 'verify'}) as rig:
            original = rig.tp_common.all_gather_matmul_prefill
            rig.install()
            self.assertIs(rig.tp_common.all_gather_matmul_prefill, original)
            self.assertIsNotNone(getattr(rig.tp_common.matmul_1d_decode, 'grid13_of', None))
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            decode = rig.tp_common.matmul_1d_decode
            rig.install()
            self.assertIs(rig.tp_common.matmul_1d_decode, decode)

    def test_the_in_projection_call_differs_in_the_two_keywords_and_the_config_fields_the_plan_names(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            mine, served = self.call_both(rig, 'all_gather_matmul_prefill')
            self.assertEqual(set(mine), set(served))
            self.assertEqual(sorted(name for name in mine if mine[name] != served[name] and not hasattr(mine[name], 'fields')), ['num_workers_per_link'])
            self.assertEqual((served['num_workers_per_link'], mine['num_workers_per_link']), (4, 6))
            for name in mine:
                if name not in ('config', 'num_workers_per_link', 'input_tensor'):
                    self.assertEqual(mine[name], served[name], name)
            left, right = mine['config'].fields(), served['config'].fields()
            self.assertEqual(sorted(name for name in left if left[name] != right[name]), ['M_block_size', 'compute_with_storage_grid_size'])
            self.assertEqual((right['M_block_size'], right['compute_with_storage_grid_size']), (4, Coord(8, 9)))
            self.assertEqual((left['M_block_size'], left['compute_with_storage_grid_size']), (6, Coord(12, 9)))
            self.assertEqual(sorted(left), sorted(right))
            self.assertIsInstance(mine['config'], MinimalMatmulConfig)
            self.assertIsInstance(mine['config'].compute_with_storage_grid_size, Coord)
            self.assertEqual(mine['num_workers_per_link'] * mine['num_links'], mine['config'].compute_with_storage_grid_size.x)

    def test_the_swiglu_call_differs_the_same_way(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            mine, served = self.call_both(rig, 'all_gather_swiglu_prefill')
            left, right = mine['config'].fields(), served['config'].fields()
            self.assertEqual(sorted(name for name in left if left[name] != right[name]), ['M_block_size', 'compute_with_storage_grid_size'])
            self.assertEqual((right['M_block_size'], right['N_block_size'], left['M_block_size'], left['N_block_size']), (8, 16, 6, 16))
            self.assertTrue(mine['fuse_swiglu'] and served['fuse_swiglu'])
            self.assertEqual(mine['num_workers_per_link'], 6)

    def test_the_engaged_line_names_the_site_the_grids_and_the_device(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            rig.prefill(width=4128)
            rig.prefill(width=3584)
            rig.prefill('all_gather_swiglu_prefill', width=8704)
            rig.prefill(width=4352)
            sites = sorted(line.split('site=')[1].split()[0] for line in rig.matching(grid13.ENGAGED))
            self.assertEqual(sites, ['attn_in', 'gate_up', 'gdn_in', 'mlp_gate_or_up'])
            line = rig.matching('site=gdn_in')[0]
            for part in ('rows=2048', 'grid=12x9', 'served_grid=8x9', 'm_block=6', 'served_m_block=4', 'workers=6', 'device=13x10'):
                self.assertIn(part, line)
            self.assertEqual(rig.matching(grid13.FELL_BACK), [])
            self.assertEqual(grid13.STATS['agmm'], 4)

    def test_an_11_by_10_device_keeps_the_served_call_and_says_why(self):
        with Rig(columns=11, environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            mine, served = self.call_both(rig, 'all_gather_matmul_prefill')
            self.assertEqual(sorted(name for name in mine if name != 'config' and mine[name] != served[name]), [])
            self.assertEqual(mine['config'].fields(), served['config'].fields())
            fell = rig.matching(grid13.FELL_BACK)
            self.assertEqual(len(fell), 1)
            self.assertIn('reason=', fell[0])
            self.assertIn('11x10', fell[0])
            self.assertEqual(rig.matching(grid13.ENGAGED), [])

    def test_a_small_chunk_keeps_the_served_call_and_is_information_not_a_fall_back(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            mine, served = self.call_both(rig, 'all_gather_matmul_prefill')
            rig.lines[:] = []
            function = rig.tp_common.all_gather_matmul_prefill
            function(rig.x(rows=256), rig.weight(4128), Collectives(), 'cfg', 'ring')
            self.assertEqual(rig.last_call()['config'].M_block_size, 4)
            self.assertEqual(rig.last_call()['num_workers_per_link'], 4)
            self.assertEqual(rig.matching(grid13.FELL_BACK), [])
            self.assertEqual(rig.matching(grid13.ENGAGED), [])
            lines = rig.matching(grid13.UNCHANGED)
            self.assertEqual(len(lines), 1)
            self.assertIn('site=gdn_in', lines[0])
            self.assertEqual(grid13_smoke.problems({grid13.FLAG: 'prefill'}, '\n'.join(rig.lines + [SmokeRuleTests.engaged_line(site) for site in ('gdn_in', 'attn_in', 'gate_up')])), [])

    def test_an_unreadable_device_grid_keeps_the_served_call(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            rig.mesh.compute_with_storage_grid_size = lambda: (_ for _ in ()).throw(RuntimeError('no grid'))
            rig.prefill()
            self.assertEqual(rig.last_call()['config'].M_block_size, 4)
            self.assertIn('cannot be read', rig.matching(grid13.FELL_BACK)[0])

    def test_a_call_that_is_not_the_census_runs_unchanged(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            module = ModuleType('fake_tp_common_untransposed')
            module.ttnn = rig.ops
            exec(compile(TP_COMMON_SOURCE.replace('force_transpose=True', 'force_transpose=False'), 'fake_tp_common.py', 'exec'), module.__dict__)
            rig.tp_common = module
            rig.install()
            rig.prefill()
            self.assertEqual(rig.last_call()['config'].M_block_size, 4)
            self.assertIn('force_transpose', rig.matching(grid13.FELL_BACK)[0])
            self.assertEqual(rig.matching(grid13.ENGAGED), [])

    def test_a_different_served_grid_runs_unchanged(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            module = ModuleType('fake_tp_common_wider')
            module.ttnn = rig.ops
            exec(compile(TP_COMMON_SOURCE.replace('grid = (8, grid[1])', 'grid = (10, grid[1])'), 'fake_tp_common.py', 'exec'), module.__dict__)
            rig.tp_common = module
            rig.install()
            rig.prefill()
            self.assertEqual(rig.last_call()['config'].compute_with_storage_grid_size, Coord(10, 9))
            self.assertIn('served grid width is 10', rig.matching(grid13.FELL_BACK)[0])

    def test_the_ttnn_global_is_back_after_the_call_and_after_a_failure(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            rig.prefill()
            self.assertIs(rig.tp_common.__dict__['ttnn'], rig.ops)
            real = rig.ops.agmm

            def boom(**kw):
                raise RuntimeError('device lost')
            rig.ops.experimental.all_gather_minimal_matmul_async = boom
            with self.assertRaises(RuntimeError):
                rig.prefill()
            self.assertIs(rig.tp_common.__dict__['ttnn'], rig.ops)
            rig.ops.experimental.all_gather_minimal_matmul_async = real

    def test_the_cfg_changes_the_grid(self):
        with Rig(environ={grid13.FLAG: 'prefill', grid13.CFG_FLAG: 'agmm=10'}) as rig:
            rig.install()
            rig.prefill()
            call = rig.last_call()
            self.assertEqual((call['config'].compute_with_storage_grid_size, call['num_workers_per_link']), (Coord(10, 9), 5))

    def test_the_result_is_the_pinned_functions_own(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            mine = rig.prefill()
            served = unwrapped(rig.tp_common.all_gather_matmul_prefill)(rig.x(), rig.weight(4128), Collectives(), 'cfg', 'ring')
            self.assertEqual(mine.chips, served.chips)


class AuditTests(unittest.TestCase):
    environment = {grid13.FLAG: 'prefill', grid13.AUDIT_FLAG: '1'}

    def test_an_exact_pair_logs_the_passing_line_and_frees_the_served_result(self):
        with Rig(environ=self.environment) as rig:
            rig.install()
            lever = rig.prefill('all_gather_swiglu_prefill', width=8704)
            lines = rig.matching(grid13.AUDIT_LINE)
            self.assertEqual(len(lines), 1)
            self.assertIn('n=1 exact=True site=gate_up rows=2048', lines[0])
            self.assertNotIn('mismatch', lines[0])
            self.assertEqual(len(rig.ops.freed), 1)
            self.assertIsNot(rig.ops.freed[0], lever)
            self.assertFalse(lever.freed)
            self.assertGreaterEqual(rig.ops.synchronized, 1)

    def test_a_partition_dependent_result_is_a_logged_mismatch_and_an_error(self):
        with Rig(environ=self.environment) as rig:
            rig.install()
            rig.ops.inexact = True                                           # the negative control: the op's bytes depend on the M block
            with self.assertRaises(AssertionError) as caught:
                rig.prefill()
            self.assertIn(grid13.AUDIT_MISMATCH, str(caught.exception))
            lines = rig.matching(grid13.AUDIT_LINE)
            self.assertTrue(any('exact=mismatch' in line for line in lines))
            self.assertTrue(any(line.startswith(grid13.AUDIT_MISMATCH) for line in lines))
            self.assertFalse(any('exact=True' in line for line in lines))
            self.assertTrue(all(grid13_smoke.MISMATCH in line or 'exact=mismatch' in line for line in lines))

    def test_only_the_first_calls_of_each_site_and_row_count_are_audited(self):
        with Rig(environ=self.environment) as rig:
            rig.install()
            for _ in range(grid13.AUDIT_CALLS + 3):
                rig.prefill()
            self.assertEqual(len(rig.matching('exact=True')), grid13.AUDIT_CALLS)
            rig.prefill(rows=1024)                                           # a new row count has its own budget
            self.assertEqual(len(rig.matching('exact=True')), grid13.AUDIT_CALLS + 1)
            self.assertEqual(len(rig.ops.calls), 2 * (grid13.AUDIT_CALLS + 1) + 3)

    def test_no_audit_without_the_flag(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            rig.install()
            rig.prefill()
            self.assertEqual(rig.matching(grid13.AUDIT_LINE), [])
            self.assertEqual(len(rig.ops.calls), 1)
            self.assertEqual(rig.ops.synchronized, 0)

    def test_nothing_is_synchronized_or_read_back_inside_a_capture_and_the_lever_program_was_warm(self):
        with Rig(environ=self.environment, sync_guard=True) as rig:
            rig.install()
            rig.prefill()                                                    # the eager warm pass: lever, audit
            self.assertEqual(len(rig.matching('exact=True')), 1)
            rig.target.begin_trace_capture(0)
            try:
                for _ in range(3):
                    rig.prefill()                                            # captured: the lever's program again, no audit
            finally:
                rig.target.end_trace_capture(0, 1)
            rig.rules.assert_clean()
            self.assertEqual(rig.rules.attempts, [])
            self.assertEqual(len(rig.matching('exact=True')), 1)
            self.assertEqual(len(rig.matching(grid13.SKIPPED)), 1)

    def test_a_program_the_warm_pass_did_not_run_is_found_by_the_rules(self):
        # the control of the test above: a captured call with a shape the warm pass never ran is a violation the model reports
        with Rig(environ={grid13.FLAG: 'prefill'}, sync_guard=True) as rig:
            rig.install()
            rig.prefill(rows=2048)
            rig.target.begin_trace_capture(0)
            try:
                with self.assertRaises(capture_rules.CaptureViolation):
                    rig.prefill(rows=1024)
            finally:
                rig.target.end_trace_capture(0, 1)
            self.assertTrue(rig.rules.problems())


class VerifyTests(unittest.TestCase):
    """The 64-row projections: the model's served config object goes in, the lever's config is what the matmul gets."""

    def served(self, rig, k, n, cores, grid_w=13, **kwargs):
        return rig.tp_common.create_matmul_1d_decode_progcfg(64, k, n, num_cores=cores, grid_w=grid_w, **kwargs)

    def run_projection(self, rig, served, k, n, rows=64):
        """One matmul_1d_decode call on `served`; -> the program config the matmul got and the output."""
        x, weight = dummy(rig.mesh, (1, 1, rows, k), 3), dummy(rig.mesh, (1, 1, k, n), 5)
        out = rig.tp_common.matmul_1d_decode(x, weight, served, 'cfg', out_memory_config='dram')
        return rig.ops.linears[-1], out

    def test_the_output_projection_goes_to_three_columns_a_core_on_the_wide_grid(self):
        with Rig(environ={grid13.FLAG: 'verify'}) as rig:
            rig.install()
            served = self.served(rig, 1536, 5120, 33)
            used, _out = self.run_projection(rig, served, 1536, 5120)
            self.assertEqual((served.per_core_N, served.compute_with_storage_grid_size), (5, Coord(13, 3)))
            self.assertEqual((used.per_core_N, used.compute_with_storage_grid_size), (3, Coord(13, 5)))
            left, right = used.fields(), served.fields()
            self.assertEqual(sorted(name for name in left if left[name] != right[name]), ['compute_with_storage_grid_size', 'out_subblock_h', 'out_subblock_w', 'per_core_N'])
            for field in grid13.KEPT_FIELDS:
                self.assertEqual(left[field], right[field], field)
            self.assertEqual(-(-160 // used.per_core_N), 54)
            lines = rig.matching(grid13.ENGAGED)
            self.assertEqual(len(lines), 1)
            for part in ('site=verify.out', 'per_core_N=3', 'served_per_core_N=5', 'cores=54', 'served_cores=32', 'grid=13x5', 'served_grid=13x3'):
                self.assertIn(part, lines[0])
            self.assertEqual(rig.matching(grid13.FELL_BACK), [])

    def test_a_served_config_gets_one_lever_config_for_its_whole_life(self):
        # the warm pass and the captured launch of a layer carry the same served object and must get the same program
        with Rig(environ={grid13.FLAG: 'verify'}) as rig:
            rig.install()
            served = self.served(rig, 1536, 5120, 33)
            first, _ = self.run_projection(rig, served, 1536, 5120)
            second, _ = self.run_projection(rig, served, 1536, 5120)
            self.assertIs(first, second)
            other = self.served(rig, 1536, 5120, 33)                       # the GDN out's own object: its own lever config, the same shape
            third, _ = self.run_projection(rig, other, 1536, 5120)
            self.assertIsNot(third, first)
            self.assertEqual(third.fields(), first.fields())
            self.assertEqual(len(rig.matching(grid13.ENGAGED)), 1)

    def test_the_other_sites_the_32_row_calls_and_the_mlp_get_the_served_config(self):
        with Rig(environ={grid13.FLAG: 'verify'}) as rig:
            rig.install()
            cases = [(5120, 4128, 44, 64), (5120, 3584, 44, 64), (5120, 3584, 64, 64), (5120, 4352, 88, 64), (4352, 5120, 33, 64), (1536, 5120, 33, 32), (5120, 4128, 44, 32)]
            for k, n, cores, rows in cases:
                served = self.served(rig, k, n, cores, grid_w=13 if cores != 64 else 8)
                used, _ = self.run_projection(rig, served, k, n, rows=rows)
                self.assertIs(used, served, (k, n, rows))
            self.assertEqual(rig.lines, [])

    def test_the_in_projections_only_when_the_cfg_names_them(self):
        with Rig(environ={grid13.FLAG: 'verify', grid13.CFG_FLAG: 'out=3,gdn_in=2,attn_in=3'}) as rig:
            rig.install()
            served = self.served(rig, 5120, 4128, 44)
            gdn, _ = self.run_projection(rig, served, 5120, 4128)
            self.assertEqual((gdn.per_core_N, gdn.compute_with_storage_grid_size), (2, Coord(13, 5)))
            served = self.served(rig, 5120, 3584, 44)
            attn, _ = self.run_projection(rig, served, 5120, 3584)
            self.assertEqual((served.per_core_N, served.compute_with_storage_grid_size), (3, Coord(13, 4)))
            self.assertEqual((attn.per_core_N, attn.compute_with_storage_grid_size), (3, Coord(13, 3)))      # the same 38 cores in 3 rows, not 4
            sites = sorted(line.split('site=')[1].split()[0] for line in rig.matching(grid13.ENGAGED))
            self.assertEqual(sites, ['verify.attn_in', 'verify.gdn_in'])

    def test_a_cfg_that_reproduces_the_served_config_is_not_engaged(self):
        with Rig(environ={grid13.FLAG: 'verify', grid13.CFG_FLAG: 'gdn_in=3'}) as rig:
            rig.install()
            served = self.served(rig, 5120, 4128, 44)
            used, _ = self.run_projection(rig, served, 5120, 4128)
            self.assertIs(used, served)
            self.assertEqual(rig.matching(grid13.FELL_BACK), [])
            self.assertIn('already 3 output columns', rig.matching(grid13.UNCHANGED)[0])
            self.assertEqual(rig.matching(grid13.ENGAGED), [])

    def test_a_narrower_device_keeps_the_served_config_and_says_why(self):
        with Rig(columns=11, environ={grid13.FLAG: 'verify'}) as rig:
            rig.install()
            served = self.served(rig, 1536, 5120, 33, grid_w=11)
            used, _ = self.run_projection(rig, served, 1536, 5120)
            self.assertIs(used, served)
            fell = rig.matching(grid13.FELL_BACK)
            self.assertEqual(len(fell), 1)
            self.assertIn('11x10', fell[0])
            self.assertEqual(rig.matching(grid13.ENGAGED), [])

    def test_a_partition_that_would_move_the_k_loop_is_refused(self):
        with Rig(environ={grid13.FLAG: 'verify'}) as rig:
            pinned = rig.tp_common.create_matmul_1d_decode_progcfg

            def moving(m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8):
                config = pinned(m, k, n, num_cores, fused_activation=fused_activation, fp32_acc=fp32_acc, grid_w=grid_w)
                if num_cores != 33:
                    config.in0_block_w = 4                                  # a builder that changes the K block with the core count
                return config
            rig.tp_common.create_matmul_1d_decode_progcfg = moving
            rig.install()
            served = self.served(rig, 1536, 5120, 33)
            used, _ = self.run_projection(rig, served, 1536, 5120)
            self.assertIs(used, served)
            self.assertIn('in0_block_w', rig.matching(grid13.FELL_BACK)[0])

    def test_an_audited_call_runs_the_served_config_too_and_compares(self):
        with Rig(environ={grid13.FLAG: 'verify', grid13.AUDIT_FLAG: '1'}) as rig:
            rig.install()
            served = self.served(rig, 1536, 5120, 33)
            used, out = self.run_projection(rig, served, 1536, 5120)
            self.assertEqual([program.per_core_N for program in rig.ops.linears], [3, 5])
            self.assertIs(rig.ops.linears[1], served)
            self.assertIn('n=1 exact=True site=verify.out rows=64', rig.matching(grid13.AUDIT_LINE)[0])
            self.assertEqual(len(rig.ops.freed), 1)
            self.assertFalse(out.freed)
            for _ in range(grid13.AUDIT_CALLS + 2):
                self.run_projection(rig, served, 1536, 5120)
            self.assertEqual(len(rig.matching('exact=True')), grid13.AUDIT_CALLS)

    def test_a_partition_dependent_projection_is_a_mismatch(self):
        with Rig(environ={grid13.FLAG: 'verify', grid13.AUDIT_FLAG: '1'}) as rig:
            rig.install()
            served = self.served(rig, 1536, 5120, 33)
            rig.ops.inexact = True
            with self.assertRaises(AssertionError):
                self.run_projection(rig, served, 1536, 5120)
            self.assertTrue(any(line.startswith(grid13.AUDIT_MISMATCH) for line in rig.lines))
            self.assertTrue(any('exact=mismatch' in line for line in rig.lines))

    def test_the_captured_launch_uses_the_program_config_the_warm_pass_ran_and_nothing_is_read_back(self):
        with Rig(environ={grid13.FLAG: 'verify', grid13.AUDIT_FLAG: '1'}, sync_guard=True) as rig:
            rig.install()
            served = self.served(rig, 1536, 5120, 33)
            self.run_projection(rig, served, 1536, 5120)                   # warm: lever + audited served
            rig.target.begin_trace_capture(0)
            try:
                for _ in range(2):
                    self.run_projection(rig, served, 1536, 5120)
            finally:
                rig.target.end_trace_capture(0, 1)
            rig.rules.assert_clean()
            self.assertEqual(rig.rules.attempts, [])
            self.assertEqual(len(rig.matching('exact=True')), 1)
            self.assertEqual(len(rig.matching(grid13.SKIPPED)), 1)

    def test_the_wrapper_is_the_pinned_function_when_verify_is_not_asked_for(self):
        with Rig(environ={grid13.FLAG: 'prefill'}) as rig:
            original = rig.tp_common.matmul_1d_decode
            rig.install()
            self.assertIs(rig.tp_common.matmul_1d_decode, original)


class CommitProgramTests(unittest.TestCase):
    def served(self, rig):
        return Program1D(compute_with_storage_grid_size=(8, 10), in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=2,
                         fuse_batch=True, fused_activation=None, mcast_in0=True)

    def test_only_the_grid_moves(self):
        with Rig(environ={grid13.FLAG: 'drafter'}) as rig:
            served = self.served(rig)
            wide = grid13.commit_program(rig.ops, rig.mesh, served, weight=dummy(rig.mesh, (1, 1, 15360, 5120), 1))
            self.assertEqual(wide.compute_with_storage_grid_size, Coord(13, 7))
            left, right = wide.fields(), served.fields()
            self.assertEqual(sorted(name for name in left if left[name] != right[name]), ['compute_with_storage_grid_size'])
            self.assertIn('site=drafter.commit cores=80 grid=13x7 served_grid=8x10', rig.matching(grid13.ENGAGED)[0])

    def test_off_other_sub_switches_and_a_narrow_device_keep_the_served_program(self):
        with Rig() as rig:
            served = self.served(rig)
            self.assertIs(grid13.commit_program(rig.ops, rig.mesh, served), served)
        with Rig(environ={grid13.FLAG: 'prefill,verify'}) as rig:
            served = self.served(rig)
            self.assertIs(grid13.commit_program(rig.ops, rig.mesh, served), served)
            self.assertEqual(rig.lines, [])
        with Rig(columns=11, environ={grid13.FLAG: 'drafter'}) as rig:
            served = self.served(rig)
            self.assertIs(grid13.commit_program(rig.ops, rig.mesh, served), served)
            self.assertIn('11x10', rig.matching(grid13.FELL_BACK)[0])

    def test_the_audit_runs_both_programs_on_the_same_operands(self):
        with Rig(environ={grid13.FLAG: 'drafter', grid13.AUDIT_FLAG: '1'}) as rig:
            served = self.served(rig)
            value, weight = dummy(rig.mesh, (1, 1, 32, 15360), 1), dummy(rig.mesh, (1, 1, 15360, 5120), 2)
            wide = grid13.commit_program(rig.ops, rig.mesh, served, value=value, weight=weight, kernel='k')
            self.assertEqual([program.compute_with_storage_grid_size for program in rig.ops.matmuls], [Coord(13, 7), Coord(8, 10)])
            self.assertEqual(wide.compute_with_storage_grid_size, Coord(13, 7))
            self.assertIn('n=1 exact=True site=drafter.commit rows=32', rig.matching(grid13.AUDIT_LINE)[0])
            self.assertEqual(len(rig.ops.freed), 2)

    def test_a_grid_dependent_result_is_a_mismatch(self):
        with Rig(environ={grid13.FLAG: 'drafter', grid13.AUDIT_FLAG: '1'}) as rig:
            rig.ops.inexact = True
            with self.assertRaises(AssertionError):
                grid13.commit_program(rig.ops, rig.mesh, self.served(rig), value=dummy(rig.mesh, (1, 1, 32, 15360), 1),
                                      weight=dummy(rig.mesh, (1, 1, 15360, 5120), 2), kernel='k')


class SmokeRuleTests(unittest.TestCase):
    @staticmethod
    def engaged_line(site, extra=''):
        return '%s site=%s rows=2048 grid=12x9 device=13x10%s' % (grid13.ENGAGED, site, extra)

    def engaged(self, site, extra=''):
        return self.engaged_line(site, extra)

    def audit(self, site, count=1):
        return '%s n=%d exact=True site=%s rows=2048' % (grid13.AUDIT_LINE, count, site)

    def log(self, audited=False):
        sites = ['gdn_in', 'attn_in', 'gate_up', 'verify.out']
        lines = [self.engaged(site) for site in sites]
        if audited:
            lines += [self.audit(site, count) for count, site in enumerate(sites, 1)]
        return '\n'.join(lines)

    def test_the_constants_equal_the_modules(self):
        self.assertEqual((grid13_smoke.FLAG, grid13_smoke.AUDIT_FLAG, grid13_smoke.CFG_FLAG), (grid13.FLAG, grid13.AUDIT_FLAG, grid13.CFG_FLAG))
        self.assertEqual((grid13_smoke.ENGAGED, grid13_smoke.FELL_BACK, grid13_smoke.AUDIT, grid13_smoke.MISMATCH),
                         (grid13.ENGAGED, grid13.FELL_BACK, grid13.AUDIT_LINE, grid13.AUDIT_MISMATCH))
        self.assertEqual(grid13_smoke.SUBSWITCHES, grid13.SUBSWITCHES)
        self.assertNotIn(grid13_smoke.FELL_BACK, grid13.UNCHANGED)       # the fall-back prefix must not match an unchanged line

    def test_the_markers_the_lever_logs_are_the_ones_the_rule_reads(self):
        with Rig(environ={grid13.FLAG: '1', grid13.AUDIT_FLAG: '1'}) as rig:
            rig.install()
            rig.prefill(width=4128)
            rig.prefill(width=3584)
            rig.prefill('all_gather_swiglu_prefill', width=8704)
            served = rig.tp_common.create_matmul_1d_decode_progcfg(64, 1536, 5120, num_cores=33, grid_w=13)
            rig.tp_common.matmul_1d_decode(dummy(rig.mesh, (1, 1, 64, 1536), 3), dummy(rig.mesh, (1, 1, 1536, 5120), 5), served, 'cfg', out_memory_config='dram')
            self.assertEqual(grid13_smoke.problems({grid13.FLAG: '1', grid13.AUDIT_FLAG: '1'}, '\n'.join(rig.lines)), [])

    def test_a_complete_log_passes_and_a_missing_site_fails_by_name(self):
        env = {grid13.FLAG: '1'}
        self.assertEqual(grid13_smoke.problems(env, self.log()), [])
        for site in ('gdn_in', 'attn_in', 'verify.out'):
            text = '\n'.join(line for line in self.log().splitlines() if 'site=%s ' % site not in line)
            found = grid13_smoke.problems(env, text)
            self.assertEqual(len(found), 1, site)
            self.assertIn(site, found[0])
        text = '\n'.join(line for line in self.log().splitlines() if 'site=gate_up ' not in line)
        self.assertEqual(len(grid13_smoke.problems(env, text)), 1)
        self.assertEqual(grid13_smoke.problems(env, text + '\n' + self.engaged('mlp_gate_or_up')), [])

    def test_sub_switches_ask_only_for_their_sites(self):
        text = '\n'.join(self.engaged(site) for site in ('gdn_in', 'attn_in', 'gate_up'))
        self.assertEqual(grid13_smoke.problems({grid13.FLAG: 'prefill'}, text), [])
        self.assertEqual(len(grid13_smoke.problems({grid13.FLAG: 'verify'}, text)), 1)
        self.assertEqual(grid13_smoke.problems({grid13.FLAG: 'drafter'}, ''), [])
        found = grid13_smoke.problems({grid13.FLAG: 'verify', grid13.CFG_FLAG: 'out=3,gdn_in=2'}, self.engaged('verify.out'))
        self.assertEqual(len(found), 1)
        self.assertIn('verify.gdn_in', found[0])

    def test_a_fall_back_and_a_mismatch_fail_the_arm(self):
        env = {grid13.FLAG: '1'}
        fell = '%s reason=the device grid is 11x10, the plans need 13x10 or wider site=gdn_in' % grid13.FELL_BACK
        self.assertEqual(len(grid13_smoke.problems(env, self.log() + '\n' + fell)), 1)
        bad = '%s site=gdn_in rows=2048 chips=[0]' % grid13.AUDIT_MISMATCH
        self.assertEqual(len(grid13_smoke.problems(env, self.log() + '\n' + bad)), 1)

    def test_the_audit_needs_a_passing_line_for_every_required_site(self):
        env = {grid13.FLAG: '1', grid13.AUDIT_FLAG: '1'}
        self.assertEqual(grid13_smoke.problems(env, self.log(audited=True)), [])
        self.assertEqual(len(grid13_smoke.problems(env, self.log())), 4)
        text = '\n'.join(line for line in self.log(audited=True).splitlines() if 'exact=True site=attn_in' not in line)
        found = grid13_smoke.problems(env, text)
        self.assertEqual(len(found), 1)
        self.assertIn('attn_in', found[0])

    def test_an_unchanged_line_is_information(self):
        env = {grid13.FLAG: '1'}
        info = '%s reason=only 8 tile rows, below the 16 the plan is derived for site=gdn_in' % grid13.UNCHANGED
        self.assertEqual(grid13_smoke.problems(env, self.log() + '\n' + info), [])

    def test_a_lever_off_profile_must_leave_no_line(self):
        self.assertEqual(grid13_smoke.problems({}, 'nothing of ours here'), [])
        self.assertEqual(len(grid13_smoke.problems({}, self.log())), 4)
        self.assertEqual(grid13_smoke.problems({grid13.FLAG: '0'}, ''), [])


class HygieneTests(unittest.TestCase):
    def test_the_new_files_name_no_host_address_or_path(self):
        import re

        pattern = re.compile(r'(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|/home/|/tmp/|C:\\|spark-|\.local\b|zot\.|ghcr\.)')
        for name in ('grid13_tp.py', 'grid13_smoke.py', 'test_grid13_tp.py'):
            text = (HERE / name).read_text(encoding='utf-8')
            found = [line for line in text.splitlines() if pattern.search(line) and 'pattern = re.compile' not in line]
            self.assertEqual(found, [], name)

    def test_runtime_files_exist(self):
        for name in grid13.RUNTIME_FILES:
            self.assertTrue((HERE / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
