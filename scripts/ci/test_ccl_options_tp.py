"""WP5 on the CPU: the named per-call options of the packed verify's collectives (QWEN_FAST_CCL_OPTIONS), their grammar, their table of what
changes bytes, their wiring into the unit-major reduce-scatter and the norms' gather, the audit and the smoke rule.

Nothing here ran on four cards. The exactness argument is the pinned source's (ccl_options_tp.py says which lines) and the audit arm is what proves
it again on the serving image."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

import ccl_options_smoke  # noqa: E402
import ccl_options_tp as options  # noqa: E402
import distributed_norm_gather_tp as gather_tp  # noqa: E402
import tile_collective_tp as collective  # noqa: E402

CHIPS = 4
WIDTH = 5120
FOUR = {'QWEN_FAST_TP': '4'}
UNIT_MAJOR = {'QWEN_FAST_TP': '4', 'QWEN_FAST_TP4_RS_UNIT_MAJOR': '1'}


class Memory:
    def __init__(self, name, sharded=False):
        self.name, self.sharded = name, sharded

    def is_sharded(self):
        return self.sharded

    def __eq__(self, other):
        return isinstance(other, Memory) and (self.name, self.sharded) == (other.name, other.sharded)

    def __hash__(self):
        return hash((self.name, self.sharded))

    def __repr__(self):
        return 'Memory(%s)' % self.name


DRAM, L1, SHARDED = Memory('DRAM'), Memory('L1'), Memory('SHARDED', True)


class Tensor:
    counter = 0

    def __init__(self, chips, memory=DRAM, dtype='bf16', layout='tile', buffer=None):
        self.chips, self.memory, self.dtype, self.layout = chips, memory, dtype, layout
        if buffer is None:
            Tensor.counter += 1
            buffer = Tensor.counter
        self.buffer = buffer
        self.freed = False

    @property
    def shape(self):
        return tuple(self.chips[0].shape)

    def memory_config(self):
        return self.memory

    def buffer_address(self):
        return self.buffer


def partials(rows, seed=0, width=WIDTH):
    generator = torch.Generator().manual_seed(seed)
    return [(torch.randn(1, 1, rows, width, generator=generator) * torch.exp(torch.randn(1, 1, rows, width, generator=generator))
             ).to(torch.bfloat16) for _ in range(CHIPS)]


def reduce_scatter(chips):
    total = sum(chip.to(torch.float32) for chip in chips).to(torch.bfloat16)
    width = total.shape[-1] // CHIPS
    return [total[..., k * width:(k + 1) * width].contiguous() for k in range(CHIPS)]


class Mesh:
    shape = (1, CHIPS)

    def get_num_devices(self):
        return CHIPS


class Collective:
    def __init__(self, links=2):
        self.links, self.rs_cycles, self.ag_cycles, self.barrier_cycles = links, 0, 0, 0

    def get_num_links(self, axis):
        return self.links

    def get_and_cycle_rs_semaphore_handles(self):
        self.rs_cycles += 1
        return ('rs', self.rs_cycles)

    def get_and_cycle_ag_semaphore_handles(self):
        self.ag_cycles += 1
        return ('ag', self.ag_cycles)

    def get_and_cycle_barrier_semaphore_handle(self):
        self.barrier_cycles += 1
        return ('barrier', self.barrier_cycles)


class Operations:
    """The ttnn calls the unit-major reduce-scatter, the audit and the gather shim make. A set whose `poison` option is present flips one bit of
    chip 1's result (the stand-in for an option that is not exact)."""
    bfloat16, float32, TILE_LAYOUT = 'bf16', 'f32', 'tile'
    DRAM_MEMORY_CONFIG = DRAM

    class Topology:
        Ring, Linear = 'Ring', 'Linear'

    def __init__(self, poison=None):
        self.calls = []
        self.poison = poison
        self.experimental = types.SimpleNamespace(reduce_scatter_minimal_async=self.reduce_scatter_minimal_async,
                                                  all_gather_async=self.all_gather_async)

    def reshape(self, tensor, shape):
        assert not tensor.freed
        self.calls.append(('reshape', tuple(tensor.shape), tuple(shape)))
        return Tensor([chip.reshape(*shape) for chip in tensor.chips], tensor.memory, tensor.dtype, tensor.layout, buffer=tensor.buffer)

    def reduce_scatter_minimal_async(self, tensor, **kwargs):
        assert not tensor.freed
        self.calls.append(('reduce_scatter', tuple(tensor.shape), kwargs))
        out = reduce_scatter(tensor.chips)
        if self.poison is not None and self.poison in str(kwargs):
            out[1] = out[1].clone()
            out[1].view(-1)[0] = out[1].view(-1)[0] + 1
        return Tensor(out, kwargs['memory_config'])

    def all_gather_async(self, tensor, **kwargs):
        assert not tensor.freed
        self.calls.append(('all_gather', tuple(tensor.shape), kwargs))
        full = torch.cat(tensor.chips, dim=3)
        out = [full.clone() for _ in range(CHIPS)]
        if self.poison is not None and self.poison in str(kwargs):
            out[2].view(-1)[3] = out[2].view(-1)[3] + 1
        return Tensor(out, kwargs['memory_config'])

    def clone(self, tensor, memory_config=None):
        assert not tensor.freed
        self.calls.append(('clone', tuple(tensor.shape)))
        return Tensor([chip.clone() for chip in tensor.chips], memory_config or tensor.memory, tensor.dtype, tensor.layout)

    def to_memory_config(self, tensor, memory_config):
        assert not tensor.freed
        self.calls.append(('to_memory_config', tuple(tensor.shape), memory_config))
        return Tensor([chip.clone() for chip in tensor.chips], memory_config, tensor.dtype, tensor.layout)

    def slice(self, tensor, start, stop, **kwargs):
        self.calls.append(('slice', start[2], stop[2]))
        return Tensor([chip[:, :, start[2]:stop[2], :].contiguous() for chip in tensor.chips],
                      kwargs.get('memory_config', tensor.memory), tensor.dtype, tensor.layout)

    def concat(self, tensors, dim, **kwargs):
        self.calls.append(('concat', len(tensors)))
        return Tensor([torch.cat([tensor.chips[k] for tensor in tensors], dim=2) for k in range(CHIPS)],
                      kwargs.get('memory_config', tensors[0].memory), tensors[0].dtype, tensors[0].layout)

    def deallocate(self, tensor):
        assert not tensor.freed, 'freed twice'
        self.calls.append(('deallocate', tuple(tensor.shape)))
        tensor.freed = True

    def get_device_tensors(self, tensor):
        return list(tensor.chips)

    def to_torch(self, chip):
        return chip

    def of(self, name):
        return [call for call in self.calls if call[0] == name]


class ModelAllReduce:
    def __init__(self, operations):
        self.operations = operations
        self.calls = []

    def __call__(self, tensor, mesh, ccl, cluster_axis=0, dim=3, topology='Ring', memory_config=DRAM):
        assert not tensor.freed
        self.calls.append((tuple(tensor.shape), memory_config))
        out = Tensor(reduce_scatter(tensor.chips), memory_config)
        tensor.freed = True
        return out


def census_kwargs():
    return dict(cluster_axis=0, dim=3, topology='Ring', memory_config=DRAM)


class TheLeverOffLoadsNothing(unittest.TestCase):
    """Default off is byte-identical off, and nothing of the lever is even imported: the served module reaches the WP5 modules by name only when a flag of the lever
    is in the environment, so the image-list closure tests (which read import statements) never see a static import of them."""

    def test_tile_collective_tp_has_no_import_statement_of_the_lever_modules(self):
        import ast
        with open(os.path.join(HERE, 'tile_collective_tp.py'), encoding='utf-8') as handle:
            tree = ast.parse(handle.read())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or '').split('.')[0])
        self.assertEqual(names & {'ccl_options_tp', 'distributed_norm_gather_tp', 'ccl_options_smoke'}, set())

    def test_a_forward_with_the_flags_unset_never_loads_them(self):
        import subprocess
        code = (
            'import os, sys\n'
            'for name in list(os.environ):\n'
            '    if name.startswith("QWEN_FAST_CCL_OPTIONS"):\n'
            '        del os.environ[name]\n'
            'import tile_collective_tp as c\n'
            'assert c.ccl_options_settings() is None\n'
            'assert c.install_gather_options() == []\n'
            'assert c.current_ccl_plan() is None\n'
            'with c.block_scope(64):\n'
            '    assert c._STATE["ccl"] is None\n'
            'assert "ccl_options_tp" not in sys.modules and "distributed_norm_gather_tp" not in sys.modules, sorted(m for m in sys.modules if "ccl" in m)\n'
            'os.environ["QWEN_FAST_CCL_OPTIONS_AUDIT"] = "1"\n'
            'try:\n'
            '    c.ccl_options_settings()\n'
            'except ValueError:\n'
            '    pass\n'
            'else:\n'
            '    raise SystemExit("the audit flag alone must be refused")\n'
            'assert "ccl_options_tp" in sys.modules\n')
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=HERE, env=dict(os.environ, PYTHONPATH=HERE), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class TheGrammar(unittest.TestCase):
    def test_served_is_the_empty_set(self):
        selection = options.parse_set('served')
        self.assertEqual((selection.name, selection.rs, selection.ag), ('served', {}, {}))

    def test_tokens_name_one_option_each_and_the_name_is_canonical(self):
        selection = options.parse_set('rs-w1+ag-c5+rs-c3+ag-linear+ag-bcast+rs-b3+rs-l1+ag-w4')
        self.assertEqual(selection.rs, {'num_workers_per_link': 1, 'chunks_per_sync': 3, 'num_buffers_per_channel': 3, 'num_links': 1})
        self.assertEqual(selection.ag, {'chunks_per_sync': 5, 'topology': 'Linear', 'use_broadcast': True, 'num_workers_per_link': 4})
        self.assertEqual(selection.name, 'ag-bcast+ag-c5+ag-linear+ag-w4+rs-b3+rs-c3+rs-l1+rs-w1')
        self.assertEqual(options.parse_set(selection.name), selection)
        self.assertEqual(options.parse_set('rs-c3+rs-w1').name, options.parse_set('rs-w1+rs-c3').name)

    def test_everything_outside_the_grammar_is_a_value_error(self):
        for text in ('', ' ', 'rs-w1 ', ' rs-w1', 'rs-w0', 'rs-w5', 'rs-w01', 'rs-l3', 'ag-l0', 'rs-c0', 'rs-c101', 'rs-b5', 'rs-linear', 'ag-linear2', 'rs-bcast',
                     'rs-x1', 'rs', 'w1', 'rs-w1+', '+rs-w1', 'rs-w1++rs-c1', 'rs-w1+rs-w2', 'ag-c1+ag-c2', 'served+rs-w1', 'Served', 'rs-W1', 'rs-w-1', 'rs-w1.5', 1,
                     None, 'rs-topology', 'rs-nobarrier', 'ag-pbuf', 'rs-pbuf', 'ag-nobar'):
            with self.assertRaises(ValueError, msg=repr(text)):
                options.parse_set(text)

    def test_the_same_option_twice_is_refused_even_at_the_same_value(self):
        with self.assertRaises(ValueError):
            options.parse_set('rs-w2+rs-w2')

    def test_a_token_naming_the_served_value_is_allowed_and_changes_nothing_extra(self):
        self.assertEqual(options.parse_set('rs-w2+rs-c10+rs-b2+rs-l2+ag-w2+ag-c10+ag-b2+ag-l2').rs,
                         {'num_workers_per_link': 2, 'chunks_per_sync': 10, 'num_buffers_per_channel': 2, 'num_links': 2})


class TheTableOfWhatChangesBytes(unittest.TestCase):
    def test_every_offered_option_is_exact_or_a_copy_in_the_table(self):
        offered = options.offered_options()
        self.assertTrue(offered)
        for op, name in offered:
            self.assertIn(options.option_effect(op, name), (options.EXACT, options.COPY), (op, name))
            self.assertEqual(options.option_effect(op, name), options.EXACT if op == 'rs' else options.COPY, (op, name))

    def test_what_changes_bytes_or_removes_synchronisation_is_not_offered(self):
        offered = set(options.offered_options())
        dangerous = [(option.op, option.name) for option in options.OPTIONS
                     if option.effect in (options.BYTES, options.SYNC, options.PLACEMENT, options.UNPROVEN)]
        self.assertTrue(dangerous)
        self.assertEqual(offered & set(dangerous), set())
        for needle in (('rs', 'topology'), ('rs', 'compute_kernel_config'), ('rs', 'dim'), ('ag', 'reverse_order'), ('ag', 'dim'),
                       ('rs', 'barrier_semaphore'), ('rs', 'persistent_output_buffers'), ('ag', 'persistent_output_buffer'), ('ag', 'barrier_semaphore')):
            self.assertIn(needle, dangerous)

    def test_every_option_is_listed_once_and_every_op_is_known(self):
        keys = [(option.op, option.name) for option in options.OPTIONS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(option.op for option in options.OPTIONS), {'rs', 'ag', 'process'})
        self.assertEqual(set(option.effect for option in options.OPTIONS) - {options.EXACT, options.COPY, options.BYTES, options.SYNC, options.PLACEMENT,
                                                                           options.INERT, options.IDENTITY, options.UNPROVEN}, set())

    def test_the_served_values_are_the_models_literals(self):
        served = dict(((option.op, option.name), option.served) for option in options.OPTIONS)
        for op in ('rs', 'ag'):
            self.assertEqual((served[(op, 'chunks_per_sync')], served[(op, 'num_workers_per_link')], served[(op, 'num_buffers_per_channel')],
                              served[(op, 'num_links')], served[(op, 'topology')]), (10, 2, 2, 2, 'Ring'))
        self.assertEqual(served[('rs', 'intermediate_memory_config')], 'DRAM')

    def test_a_payload_below_two_pages_moves_the_chunk_parity_and_the_table_says_so(self):
        payload = [option for option in options.OPTIONS if option.name == 'max_packet_payload_size_bytes'][0]
        self.assertIn('4096', payload.why)

        def tile_granularity(packet, page=2048, max_dst=8):
            return min(4 * min(4, packet // page), max_dst)
        self.assertEqual([tile_granularity(packet) for packet in (4352, 8192, 15232, 4096)], [8, 8, 8, 8])
        self.assertEqual(tile_granularity(2048), 4)
        self.assertEqual(tile_granularity(4352, max_dst=4), 4)      # fp32_dest_acc_en


class TheSettings(unittest.TestCase):
    def test_unset_and_zero_are_off(self):
        self.assertIsNone(options.settings({}))
        self.assertIsNone(options.settings({options.OPTIONS_FLAG: '0'}))
        self.assertIsNone(options.settings({'QWEN_FAST_TP': '4'}))

    def test_the_audit_flags_alone_are_refused(self):
        for source in ({options.AUDIT_FLAG: '1'}, {options.AUDIT_CALLS_FLAG: '4'}, {options.OPTIONS_FLAG: '0', options.AUDIT_FLAG: '1'}):
            with self.assertRaises(ValueError, msg=source):
                options.settings(dict(source, **FOUR))

    def test_the_pair_never_takes_it(self):
        with self.assertRaises(ValueError):
            options.settings({options.OPTIONS_FLAG: 'served', 'QWEN_FAST_TP': '2'})

    def test_a_reduce_scatter_token_needs_the_unit_major_lever(self):
        with self.assertRaises(ValueError):
            options.settings(dict(FOUR, **{options.OPTIONS_FLAG: 'rs-w1'}))
        with self.assertRaises(ValueError):
            options.settings(dict(FOUR, **{options.OPTIONS_FLAG: 'rs-w1', 'QWEN_FAST_TP4_RS_UNIT_MAJOR': '0'}))
        self.assertEqual(options.settings(dict(UNIT_MAJOR, **{options.OPTIONS_FLAG: 'rs-w1'})).selection.rs, {'num_workers_per_link': 1})

    def test_gather_only_and_served_sets_do_not_need_it(self):
        self.assertEqual(options.settings(dict(FOUR, **{options.OPTIONS_FLAG: 'ag-w1+ag-linear'})).selection.name, 'ag-linear+ag-w1')
        self.assertEqual(options.settings(dict(FOUR, **{options.OPTIONS_FLAG: 'served'})).selection.name, 'served')

    def test_the_audit(self):
        base = dict(FOUR, **{options.OPTIONS_FLAG: 'ag-w1'})
        self.assertEqual(options.settings(base).audit_calls, 0)
        self.assertEqual(options.settings(dict(base, **{options.AUDIT_FLAG: '1'})).audit_calls, options.DEFAULT_AUDIT_CALLS)
        self.assertEqual(options.settings(dict(base, **{options.AUDIT_FLAG: '1', options.AUDIT_CALLS_FLAG: '8'})).audit_calls, 8)
        for bad in ('0', '1', '3', '-2', '08', '2.0', '', 'x'):
            with self.assertRaises(ValueError, msg=bad):
                options.settings(dict(base, **{options.AUDIT_FLAG: '1', options.AUDIT_CALLS_FLAG: bad}))
        with self.assertRaises(ValueError):
            options.settings(dict(base, **{options.AUDIT_CALLS_FLAG: '8'}))
        with self.assertRaises(ValueError):
            options.settings(dict(base, **{options.AUDIT_FLAG: '2'}))


class Fixture(unittest.TestCase):
    def setUp(self):
        collective._HELD.clear()
        collective._STATE['reasons'].clear()
        collective._STATE['owners'] = 0
        collective._STATE['replayed'] = None
        collective._STATE['rows'] = None
        options._STATE['plan'] = None
        options._STATE['reasons'].clear()
        self.addCleanup(lambda: options._STATE.update(plan=None))
        self.operations = Operations()
        self.model = ModelAllReduce(self.operations)
        self.wrapper = collective.TileSplitAllReduce(self.model, self.operations)
        self.mesh, self.ccl = Mesh(), Collective()
        self.lines = []

    def log(self, text, *values):
        self.lines.append(text.format(*values))

    def pindiag(self, text):
        self.lines.append(text)

    def call(self, chips):
        tensor = Tensor(chips, DRAM)
        return self.wrapper(tensor, self.mesh, self.ccl, **census_kwargs())

    def settings(self, text, audit=0):
        return options.Settings(options.parse_set(text), audit)

    def scope(self, text=None, audit=0, unit_major=True, u1_audit=0, layers=None, expected=None):
        configured = None if text is None else self.settings(text, audit)
        return collective.block_scope(64, expected=expected, log=self.log, unit_major=unit_major, audit_calls=u1_audit, ccl=configured, layers=layers)

    def reduce_kwargs(self):
        return [call[2] for call in self.operations.of('reduce_scatter')]


SERVED_KWARGS = dict(persistent_output_buffers=None, dim=3, num_links=2, memory_config=DRAM, intermediate_memory_config=DRAM, topology='Ring',
                     chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)


def plain(kwargs):
    return dict((key, value) for key, value in kwargs.items() if key not in ('multi_device_global_semaphore', 'barrier_semaphore'))


class TheReduceScatterWiring(Fixture):
    def test_with_the_lever_off_the_call_is_the_x1_spikes_to_the_keyword(self):
        with self.scope():
            self.call(partials(64))
        self.assertEqual([plain(kwargs) for kwargs in self.reduce_kwargs()], [SERVED_KWARGS])
        self.assertEqual(self.reduce_kwargs()[0]['multi_device_global_semaphore'], ('rs', 1))
        self.assertEqual(self.reduce_kwargs()[0]['barrier_semaphore'], ('barrier', 1))

    def test_a_set_changes_exactly_the_named_keywords(self):
        with self.scope('rs-w1+rs-c3+rs-b4'):
            self.call(partials(64))
        expected = dict(SERVED_KWARGS, num_workers_per_link=1, chunks_per_sync=3, num_buffers_per_channel=4)
        self.assertEqual([plain(kwargs) for kwargs in self.reduce_kwargs()], [expected])

    def test_links_ride_the_override_not_the_collectives_answer(self):
        with self.scope('rs-l1'):
            self.call(partials(64))
        self.assertEqual(self.reduce_kwargs()[0]['num_links'], 1)

    def test_the_served_and_gather_only_sets_leave_the_reduce_scatter_alone(self):
        for text in ('served', 'ag-w1+ag-linear'):
            self.operations.calls.clear()
            with self.scope(text):
                self.call(partials(64))
            self.assertEqual(plain(self.reduce_kwargs()[0]), SERVED_KWARGS, text)

    def test_the_result_is_the_same_bits_with_and_without_the_set(self):
        source = partials(64, seed=3)
        with self.scope():
            base = self.call([chip.clone() for chip in source])
        with self.scope('rs-w1+rs-c1'):
            other = self.call([chip.clone() for chip in source])
        for left, right in zip(base.chips, other.chips):
            self.assertTrue(torch.equal(left.view(torch.int16), right.view(torch.int16)))

    def test_a_scope_counts_the_calls_that_carried_the_set_and_logs_the_engaged_line_after_the_guard(self):
        gathers = []
        with self.scope('rs-w1', layers=1) as _:
            for _ in range(2):
                self.call(partials(64))
            plan = options.current()
            plan.ag_engaged += 2
            gathers.append(plan)
        self.assertEqual(gathers[0].rs_engaged, 2)
        self.assertIsNone(options.current())
        engaged = [line for line in self.lines if options.ENGAGED_MARKER in line]
        self.assertEqual(engaged, ['[PINDIAG] tp4 ccl options engaged set=rs-w1 rows=64 rs=2 ag=2 fallbacks=0 audited=0'])

    def test_the_guard_refuses_a_forward_the_gather_wrapper_never_saw(self):
        with self.assertRaises(AssertionError) as raised:
            with self.scope('rs-w1+ag-w1', layers=64):
                self.call(partials(64))
        self.assertIn('saw 0 norm gathers', str(raised.exception))
        self.assertIsNone(options.current())
        self.assertFalse([line for line in self.lines if options.ENGAGED_MARKER in line])

    def test_a_reduce_scatter_only_set_does_not_ask_for_the_gather_wrapper(self):
        for text, active in (('rs-c1', False), ('rs-w1+rs-c1', False), ('served', True), ('ag-w1', True), ('rs-c1+ag-c1', True)):
            self.assertEqual(options.gather_active(options.parse_set(text)), active, text)
        with self.scope('rs-c1', layers=64):
            self.call(partials(64))
            self.assertIsNone(options.current().layers)
        self.assertTrue([line for line in self.lines if options.ENGAGED_MARKER in line][-1].endswith('rs=1 ag=0 fallbacks=0 audited=0'))

    def test_the_guard_accepts_the_final_norm_or_not(self):
        for gathers in (128, 129):
            with self.scope('served', layers=64):
                options.current().ag_engaged += gathers
        for gathers in (127, 130):
            with self.assertRaises(AssertionError):
                with self.scope('served', layers=64):
                    options.current().ag_engaged += gathers

    def test_an_error_in_the_forward_leaves_the_plan_closed_and_is_not_replaced_by_the_guard(self):
        with self.assertRaises(RuntimeError):
            with self.scope('rs-w1', layers=64):
                raise RuntimeError('forward failed')
        self.assertIsNone(options.current())

    def test_scopes_do_not_open_a_second_plan(self):
        with self.scope('served'):
            with self.assertRaises(ValueError):
                options.begin(self.settings('served'), 64, 1)
            options.current().ag_engaged += 2
        self.assertIsNone(options.current())

    def test_a_reduce_scatter_set_needs_the_unit_major_scope(self):
        with self.assertRaises(ValueError):
            with self.scope('rs-w1', unit_major=False):
                pass
        with self.scope('ag-w1', unit_major=False):
            pass

    def test_a_call_outside_the_census_is_served_by_the_split_with_the_models_values_and_counted(self):
        bad = Tensor(partials(64, width=2560), DRAM)
        with self.scope('rs-w1'):
            self.wrapper(bad, self.mesh, self.ccl, **census_kwargs())
            plan = options.current()
            self.assertEqual(plan.fallback_counts, {'rs': 1, 'ag': 0})
            self.assertEqual(plan.rs_engaged, 0)
        self.assertEqual(self.operations.of('reduce_scatter'), [])
        self.assertEqual(len([line for line in self.lines if options.FALLBACK_MARKER in line]), 1)
        # a gather-only set does not count a reduce-scatter fall-back as its own
        with self.scope('ag-w1'):
            self.wrapper(Tensor(partials(64, width=2560), DRAM), self.mesh, self.ccl, **census_kwargs())
            self.assertEqual(options.current().fallback_counts['rs'], 0)


class TheReduceScatterAudit(Fixture):
    def run_forward(self, text, calls, audit, u1_audit=0, poison=None):
        self.operations.poison = poison
        with self.scope(text, audit=audit, u1_audit=u1_audit):
            outputs = [self.call(partials(64, seed=index)) for index in range(calls)]
            options.current().ag_engaged += 0
        return outputs

    def test_the_first_calls_run_twice_and_the_models_values_are_served(self):
        self.operations.calls.clear()
        with self.scope('rs-w1', audit=2):
            outputs = [self.call(partials(64, seed=index)) for index in range(3)]
        kwargs = self.reduce_kwargs()
        self.assertEqual(len(kwargs), 3 + 2)
        self.assertEqual([kw['num_workers_per_link'] for kw in kwargs], [1, 2, 1, 2, 1])
        self.assertEqual(len(collective._HELD), 2)
        self.assertEqual(set((pair['op'], pair['marker']) for pair in collective._HELD), {('rs', options.AUDIT_MARKER)})
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(not output.freed for output in outputs))

    def test_the_replay_compare_logs_the_ccl_line_and_the_u1_line_is_untouched(self):
        with self.scope('rs-c1', audit=2, u1_audit=1):
            for index in range(3):
                self.call(partials(64, seed=index))
        owner = object()
        self.assertEqual(collective.audit_claim(owner, 'warm'), 3)
        collective.audit_round(self.operations, owner, 0, log=self.lines.append)
        lines = [line for line in self.lines if 'audit shape' in line]
        self.assertEqual(sorted(line.split(' shape=')[0] for line in lines), sorted([options.AUDIT_MARKER, '[PINDIAG] tp4 u1 audit']))
        ccl = [line for line in lines if line.startswith(options.AUDIT_MARKER)][0]
        self.assertIn('shape=64x5120 owner=warm1 round=0 op=rs calls=2 chips=4', ccl)
        self.assertTrue(ccl.endswith('exact=True'))
        u1 = [line for line in lines if line.startswith('[PINDIAG] tp4 u1 audit')][0]
        self.assertNotIn(' op=', u1)

    def test_an_option_that_changed_a_bit_is_a_mismatch_with_its_own_marker(self):
        with self.scope('rs-w1', audit=2):
            self.operations.poison = "'num_workers_per_link': 1"
            for index in range(2):
                self.call(partials(64, seed=index))
        owner = object()
        collective.audit_claim(owner, 'capture')
        with self.assertRaises(AssertionError) as raised:
            collective.audit_round(self.operations, owner, 0, log=self.lines.append)
        self.assertTrue(str(raised.exception).startswith(options.AUDIT_MISMATCH_MARKER))
        self.assertIn('op=rs', str(raised.exception))
        self.assertIn(options.AUDIT_MISMATCH_MARKER, self.lines[-1])

    def test_the_u1_audit_with_a_set_compares_the_options_against_the_split(self):
        self.operations.calls.clear()
        with self.scope('rs-w1', u1_audit=1):
            self.call(partials(64))
        kwargs = self.reduce_kwargs()
        self.assertEqual([kw['num_workers_per_link'] for kw in kwargs], [1] + [])     # the split's two tile calls go through the model's own
        self.assertEqual(len(self.model.calls), 2)

    def test_the_forward_keeps_an_even_number_of_reduce_scatters(self):
        # 3 calls, quota 2: 5 scatters, odd -> refused; the shipped forward has 128 plus the quota's extra ones (even), so the parity guard can see it
        with self.assertRaises(AssertionError) as raised:
            with self.scope('rs-w1', audit=2, expected=3):
                for index in range(3):
                    self.call(partials(64, seed=index))
        self.assertIn('odd', str(raised.exception))
        with self.scope('rs-w1', audit=2, expected=4):
            for index in range(4):
                self.call(partials(64, seed=index))

    def test_clones_are_freed_with_the_owner(self):
        with self.scope('rs-w1', audit=2):
            for index in range(2):
                self.call(partials(64, seed=index))
        owner = object()
        collective.audit_claim(owner, 'warm')
        self.assertEqual(collective.audit_release(self.operations, owner), 2)
        self.assertEqual(collective._HELD, [])


def gather_call_kwargs(handles, links=2, memory=SHARDED):
    return dict(persistent_output_buffer=None, dim=3, multi_device_global_semaphore=handles, num_links=links, topology='Ring', memory_config=memory,
                barrier_semaphore=('barrier', 0), chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2, subdevice_id=None)


NORM_SOURCE = '''
class Mode:
    DECODE = 'decode'
    PREFILL = 'prefill'


class Args:
    is_multichip = True
    mesh_device = None

    def is_distributed_norm(self, mode):
        return False

    def ccl_topology(self):
        return 'Ring'


class DistributedNorm:
    def __init__(self, norm, args, tt_ccl, TG=False):
        self.norm, self.args, self.tt_ccl, self.TG = norm, args, tt_ccl, TG

    def forward(self, x, mode, norm_config=None):
        input_mem_cfg = norm_config['sharded_output_config']
        if self.args.is_multichip and not self.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=self.tt_ccl.get_num_links(1),
                topology=self.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
                subdevice_id=None,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)
        return self.norm(x)
'''


class TheGatherShim(Fixture):
    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, 'fake_distributed_norm_for_wp5.py')
        with open(path, 'w') as handle:
            handle.write(NORM_SOURCE)
        import importlib.util
        spec = importlib.util.spec_from_file_location('fake_distributed_norm_for_wp5', path)
        self.norm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.norm_module)
        self.ttnn = types.SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', Topology=Operations.Topology, DRAM_MEMORY_CONFIG=DRAM,
                                          experimental=self.operations.experimental, to_memory_config=self.operations.to_memory_config,
                                          clone=self.operations.clone, deallocate=self.operations.deallocate,
                                          get_device_tensors=self.operations.get_device_tensors, to_torch=self.operations.to_torch)
        self.norm_module.ttnn = self.ttnn
        self.args = self.norm_module.Args()
        self.args.mesh_device = Mesh()
        self.norm = self.norm_module.DistributedNorm(lambda x: x, self.args, self.ccl)
        self.original = self.norm_module.DistributedNorm.forward
        self.addCleanup(lambda: setattr(self.norm_module.DistributedNorm, 'forward', self.original))
        self.wrapped = gather_tp.scoped_forward(self.original)

    def forward(self, chips=None, rows=64, mode='decode', norm=None):
        tensor = Tensor(chips or partials(rows, width=1280), L1)
        return self.wrapped(norm or self.norm, tensor, mode, norm_config={'sharded_output_config': SHARDED})

    def gathers(self):
        return [call[2] for call in self.operations.of('all_gather')]

    def test_outside_a_scope_the_wrapper_is_the_original(self):
        result = self.forward()
        self.assertEqual(len(self.gathers()), 1)
        self.assertEqual(plain(self.gathers()[0])['num_workers_per_link'], 2)
        self.assertEqual(result.shape, (1, 1, 64, 5120))
        self.assertIs(self.norm_module.__dict__['ttnn'], self.ttnn)

    def test_inside_a_scope_the_gather_carries_the_set_and_nothing_else_changes(self):
        with self.scope('ag-w1+ag-c5+ag-b3+ag-linear'):
            self.forward()
            self.assertEqual(options.current().ag_engaged, 1)
        kwargs = self.gathers()[0]
        self.assertEqual(plain(kwargs), dict(persistent_output_buffer=None, dim=3, num_links=2, topology='Linear', memory_config=SHARDED, chunks_per_sync=5,
                                             num_workers_per_link=1, num_buffers_per_channel=3, subdevice_id=None))
        self.assertEqual(kwargs['multi_device_global_semaphore'], ('ag', 1))
        self.assertEqual(kwargs['barrier_semaphore'], ('barrier', 1))

    def test_broadcast_and_links_ride_the_same_override(self):
        with self.scope('ag-bcast+ag-l1'):
            self.forward()
        kwargs = self.gathers()[0]
        self.assertTrue(kwargs['use_broadcast'])
        self.assertEqual(kwargs['num_links'], 1)

    def test_a_reduce_scatter_only_and_the_served_set_leave_the_gather_as_the_model_calls_it(self):
        for text in ('rs-w1', 'served'):
            self.operations.calls.clear()
            with self.scope(text):
                self.forward()
                self.assertEqual(options.current().ag_engaged, 1)
            self.assertEqual(plain(self.gathers()[0]), dict(persistent_output_buffer=None, dim=3, num_links=2, topology='Ring', memory_config=SHARDED,
                                                             chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2, subdevice_id=None), text)

    def test_the_module_global_is_put_back_also_when_the_forward_raises(self):
        broken = self.norm_module.DistributedNorm(lambda x: 1 / 0, self.args, self.ccl)
        with self.assertRaises(ZeroDivisionError):
            with self.scope('ag-w1'):
                self.forward(norm=broken)
        self.assertIs(self.norm_module.__dict__['ttnn'], self.ttnn)

    def test_prefill_and_other_shapes_pass_through_and_the_shim_is_not_left_on_ttnn(self):
        with self.scope('ag-w1'):
            self.forward(mode='prefill')
            self.assertEqual(options.current().ag_engaged, 0)
            self.forward(rows=32)
            plan = options.current()
            self.assertEqual((plan.ag_engaged, plan.fallback_counts['ag']), (0, 1))
        self.assertEqual([plain(kwargs)['num_workers_per_link'] for kwargs in self.gathers()], [2, 2])
        self.assertTrue(self.ttnn.experimental.all_gather_async == self.operations.all_gather_async)

    def test_the_census_names_every_reason(self):
        plan = options.begin(self.settings('ag-w1'), 64, 0, log=self.log)
        self.addCleanup(lambda: options.close(plan))
        shim = gather_tp.GatherShim(plan, self.norm, self.ttnn)
        good = gather_call_kwargs(('ag', 1))
        tensor = Tensor(partials(64, width=1280), L1)
        self.assertIsNone(shim.refusal(tensor, (), good))
        cases = [
            ('positional', (1, ), good),
            ('keywords', (), dict((key, value) for key, value in good.items() if key != 'subdevice_id')),
            ('keywords', (), dict(good, extra=1)),
            ('persistent', (), dict(good, persistent_output_buffer=object())),
            ('sub-device', (), dict(good, subdevice_id=1)),
            ('dim', (), dict(good, dim=2)),
            ('topology', (), dict(good, topology='Linear')),
            ('num_links', (), dict(good, num_links=1)),
            ('chunks_per_sync', (), dict(good, chunks_per_sync=5)),
            ('num_workers_per_link', (), dict(good, num_workers_per_link=1)),
            ('num_buffers_per_channel', (), dict(good, num_buffers_per_channel=1)),
        ]
        for needle, args, kwargs in cases:
            reason = shim.refusal(tensor, args, kwargs)
            self.assertIsNotNone(reason, needle)
            self.assertIn(needle, reason)
        for bad_tensor, needle in ((Tensor(partials(32, width=1280), L1), 'rows'), (Tensor(partials(64, width=640), L1), 'width'),
                                   (Tensor(partials(64, width=1280), L1, dtype='f32'), 'dtype'),
                                   (Tensor(partials(64, width=1280), L1, layout='row'), 'layout')):
            self.assertIn(needle, shim.refusal(bad_tensor, (), good))
        for bad_mesh in (None, types.SimpleNamespace(get_num_devices=lambda: 2, shape=(1, 2)), types.SimpleNamespace(get_num_devices=lambda: 4, shape=(2, 2))):
            self.args.mesh_device = bad_mesh
            self.assertIn('mesh', shim.refusal(tensor, (), good))

    def test_a_refused_gather_runs_with_the_models_own_values_and_is_counted_once_per_reason(self):
        with self.scope('ag-w1'):
            for _ in range(3):
                self.forward(rows=64, chips=partials(64, width=640))
            plan = options.current()
            self.assertEqual((plan.ag_engaged, plan.fallback_counts['ag']), (0, 3))
        self.assertEqual([plain(kwargs)['num_workers_per_link'] for kwargs in self.gathers()], [2, 2, 2])
        self.assertEqual(len([line for line in self.lines if options.FALLBACK_MARKER in line]), 1)

    def test_the_audit_runs_the_gather_twice_serves_the_models_values_and_holds_the_pair(self):
        with self.scope('ag-w1', audit=2):
            for index in range(3):
                self.forward(chips=partials(64, seed=index, width=1280))
        kwargs = self.gathers()
        self.assertEqual([plain(kw)['num_workers_per_link'] for kw in kwargs], [1, 2, 1, 2, 1])
        self.assertEqual(len(collective._HELD), 2)
        self.assertEqual(set((pair['op'], pair['marker']) for pair in collective._HELD), {('ag', options.AUDIT_MARKER)})
        # each audited call converts the options result and the model's own result into DRAM (the results are sharded), twice in all
        self.assertEqual(len(self.operations.of('to_memory_config')), 2 * 2)
        self.assertEqual(self.operations.of('clone'), [])
        owner = object()
        self.assertEqual(collective.audit_claim(owner, 'capture'), 2)
        collective.audit_round(self.operations, owner, 0, log=self.lines.append)
        found = [line for line in self.lines if line.startswith(options.AUDIT_MARKER)]
        self.assertEqual(len(found), 1)
        self.assertIn('shape=64x5120 owner=capture1 round=0 op=ag calls=2 chips=4', found[0])

    def test_the_audit_uses_fresh_semaphores_for_the_second_call(self):
        with self.scope('ag-w1', audit=2):
            self.forward()
        kwargs = self.gathers()
        self.assertEqual([kw['multi_device_global_semaphore'] for kw in kwargs], [('ag', 1), ('ag', 2)])
        self.assertEqual([kw['barrier_semaphore'] for kw in kwargs], [('barrier', 1), ('barrier', 2)])

    def test_a_gather_option_that_changed_a_bit_is_caught(self):
        with self.scope('ag-linear', audit=2):
            self.operations.poison = "'topology': 'Linear'"
            self.forward()
        owner = object()
        collective.audit_claim(owner, 'warm')
        with self.assertRaises(AssertionError) as raised:
            collective.audit_round(self.operations, owner, 0, log=self.lines.append)
        self.assertTrue(str(raised.exception).startswith(options.AUDIT_MISMATCH_MARKER))
        self.assertIn('op=ag', str(raised.exception))

    def test_a_failing_second_call_frees_the_first_clone(self):
        real = self.operations.all_gather_async
        calls = []

        def fail_second(tensor, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError('no memory')
            return real(tensor, **kwargs)
        self.ttnn.experimental = types.SimpleNamespace(all_gather_async=fail_second)
        with self.assertRaises(RuntimeError):
            with self.scope('ag-w1', audit=2):
                self.forward()
        self.assertEqual(collective._HELD, [])
        clones = [call for call in self.operations.of('to_memory_config')]
        frees = self.operations.of('deallocate')
        self.assertEqual(len(clones), 1)
        self.assertGreaterEqual(len(frees), 2)


class TheInstall(Fixture):
    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, 'fake_distributed_norm_install.py')
        with open(path, 'w') as handle:
            handle.write(NORM_SOURCE)
        import importlib.util
        spec = importlib.util.spec_from_file_location('fake_distributed_norm_install', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module.ttnn = types.SimpleNamespace()
        self.original = self.module.DistributedNorm.forward
        self.addCleanup(lambda: setattr(self.module.DistributedNorm, 'forward', self.original))

    def test_with_the_lever_off_nothing_is_wrapped(self):
        with patch.dict(os.environ, FOUR, clear=False):
            os.environ.pop(options.OPTIONS_FLAG, None)
            self.assertEqual(gather_tp.install(self.module), [])
        self.assertIs(self.module.DistributedNorm.forward, self.original)

    def test_with_the_lever_on_the_pinned_source_is_wrapped_and_a_second_install_changes_nothing(self):
        digest = gather_tp.forward_digest(self.original)
        with patch.dict(os.environ, dict(FOUR, **{options.OPTIONS_FLAG: 'ag-w1'})), patch.object(gather_tp, 'FORWARD_SHA256', digest):
            changed = gather_tp.install(self.module)
            self.assertEqual(len(changed), 1)
            namespace, name, old = changed[0]
            self.assertEqual((name, old), ('forward', self.original))
            self.assertIs(self.module.DistributedNorm.forward.gather_scope_of, self.original)
            self.assertEqual(gather_tp.install(self.module), [])
            namespace[name] = old
        self.assertIs(self.module.DistributedNorm.forward, self.original)

    def test_a_forward_that_is_not_the_pinned_source_is_refused(self):
        with patch.dict(os.environ, dict(FOUR, **{options.OPTIONS_FLAG: 'ag-w1'})):
            with self.assertRaises(RuntimeError) as raised:
                gather_tp.install(self.module)
        self.assertIn('not the pinned source', str(raised.exception))
        self.assertIs(self.module.DistributedNorm.forward, self.original)

    def test_a_set_of_reduce_scatter_options_alone_leaves_the_norm_unwrapped(self):
        digest = gather_tp.forward_digest(self.original)
        with patch.dict(os.environ, dict(UNIT_MAJOR, **{options.OPTIONS_FLAG: 'rs-c1'})), patch.object(gather_tp, 'FORWARD_SHA256', digest):
            self.assertEqual(gather_tp.install(self.module), [])
        self.assertIs(self.module.DistributedNorm.forward, self.original)
        for text in ('served', 'ag-c1', 'rs-c1+ag-c1'):
            with patch.dict(os.environ, dict(UNIT_MAJOR, **{options.OPTIONS_FLAG: text})), patch.object(gather_tp, 'FORWARD_SHA256', digest):
                changed = gather_tp.install(self.module)
                self.assertEqual(len(changed), 1, text)
                self.module.DistributedNorm.forward = self.original

    def test_without_the_model_tree_nothing_is_bound(self):
        with patch.dict(os.environ, dict(FOUR, **{options.OPTIONS_FLAG: 'ag-w1'})), patch.dict(sys.modules, {'models': None}):
            self.assertEqual(gather_tp.install(), [])

    def test_the_collective_installer_binds_the_gather_shim_with_the_scope_and_uninstall_puts_both_back(self):
        import model_batch
        ccl = types.ModuleType('fake_ccl_for_wp5')
        ccl.tt_all_reduce = lambda *args, **kwargs: None
        sys.modules['fake_ccl_for_wp5'] = ccl
        sys.modules['fake_dn_for_wp5'] = self.module
        run_before = model_batch.ModelBatch.run
        self.addCleanup(lambda: (sys.modules.pop('fake_ccl_for_wp5', None), sys.modules.pop('fake_dn_for_wp5', None),
                                 setattr(model_batch.ModelBatch, 'run', run_before)))
        digest = gather_tp.forward_digest(self.original)
        env = dict(FOUR, **{options.OPTIONS_FLAG: 'ag-w1'})
        with patch.dict(os.environ, env), patch.object(gather_tp, 'FORWARD_SHA256', digest), patch.object(gather_tp, 'NORM_MODULE', 'fake_dn_for_wp5'):
            changed = collective.install(ccl, scope=True)
            try:
                names = sorted(type(namespace).__name__ + ':' + name for namespace, name, _ in changed)
                self.assertIn('ClassAttributes:forward', names)
                self.assertIn('ClassAttributes:run', names)
                self.assertIsNot(self.module.DistributedNorm.forward, self.original)
            finally:
                for namespace, name, old in changed:
                    namespace[name] = old
        self.assertIs(self.module.DistributedNorm.forward, self.original)
        self.assertIs(model_batch.ModelBatch.run, run_before)

    def test_the_collective_installer_with_the_lever_off_binds_no_gather_shim(self):
        ccl = types.ModuleType('fake_ccl_for_wp5')
        ccl.tt_all_reduce = lambda *args, **kwargs: None
        sys.modules['fake_ccl_for_wp5'] = ccl
        sys.modules['fake_dn_for_wp5'] = self.module
        import model_batch
        run_before = model_batch.ModelBatch.run
        self.addCleanup(lambda: (sys.modules.pop('fake_ccl_for_wp5', None), sys.modules.pop('fake_dn_for_wp5', None),
                                 setattr(model_batch.ModelBatch, 'run', run_before)))
        with patch.dict(os.environ, FOUR), patch.object(gather_tp, 'NORM_MODULE', 'fake_dn_for_wp5'):
            os.environ.pop(options.OPTIONS_FLAG, None)
            changed = collective.install(ccl, scope=True)
            for namespace, name, old in changed:
                namespace[name] = old
        self.assertIs(self.module.DistributedNorm.forward, self.original)

    def test_the_pinned_digest_is_a_sha256(self):
        self.assertRegex(gather_tp.FORWARD_SHA256, r'^[0-9a-f]{64}$')


class TheSmokeRule(unittest.TestCase):
    ENV = dict(UNIT_MAJOR, **{options.OPTIONS_FLAG: 'rs-w1+ag-w1'})

    def engaged(self, name='ag-w1+rs-w1', rs=128, ag=129, fallbacks=0, audited=0):
        return '[PINDIAG] tp4 ccl options engaged set=%s rows=64 rs=%d ag=%d fallbacks=%d audited=%d' % (name, rs, ag, fallbacks, audited)

    def audit(self, op, owner='warm1', round_number=1, chips=4, elements=100):
        return ('[PINDIAG] tp4 ccl options audit shape=64x5120 owner=%s round=%d op=%s calls=32 chips=%d elements=%d exact=True'
                % (owner, round_number, op, chips, elements))

    def test_a_profile_without_the_flag_logs_nothing(self):
        self.assertEqual(ccl_options_smoke.problems({}, 'nothing'), [])
        self.assertEqual(ccl_options_smoke.problems(None, ''), [])
        self.assertTrue(ccl_options_smoke.problems({}, self.engaged()))
        self.assertTrue(ccl_options_smoke.problems({options.OPTIONS_FLAG: '0'}, self.audit('rs')))

    def test_a_clean_log(self):
        self.assertEqual(ccl_options_smoke.problems(self.ENV, '\n'.join([self.engaged(), self.engaged()])), [])
        self.assertEqual(ccl_options_smoke.problems(dict(self.ENV, **{options.OPTIONS_FLAG: 'ag-w1'}), self.engaged('ag-w1', rs=0)), [])
        self.assertEqual(ccl_options_smoke.problems(dict(self.ENV, **{options.OPTIONS_FLAG: 'served'}), self.engaged('served', rs=0)), [])

    def test_a_reduce_scatter_only_set_needs_no_gather_count_and_no_gather_audit(self):
        env = dict(UNIT_MAJOR, **{options.OPTIONS_FLAG: 'rs-c1'})
        self.assertEqual(ccl_options_smoke.problems(env, self.engaged('rs-c1', ag=0)), [])
        self.assertTrue(ccl_options_smoke.problems(env, self.engaged('rs-c1', rs=0, ag=0)))
        audited = dict(env, **{options.AUDIT_FLAG: '1'})
        self.assertEqual(ccl_options_smoke.problems(audited, '\n'.join([self.engaged('rs-c1', ag=0), self.audit('rs')])), [])
        self.assertTrue(ccl_options_smoke.problems(audited, self.engaged('rs-c1', ag=0)))

    def test_no_engaged_line_wrong_set_zero_counts(self):
        self.assertTrue(ccl_options_smoke.problems(self.ENV, 'no lever lines'))
        self.assertTrue(ccl_options_smoke.problems(self.ENV, self.engaged(name='ag-w1+rs-w2')))
        self.assertTrue(ccl_options_smoke.problems(self.ENV, self.engaged(rs=0)))
        self.assertTrue(ccl_options_smoke.problems(self.ENV, self.engaged(ag=0)))
        self.assertTrue(ccl_options_smoke.problems(self.ENV, '[PINDIAG] tp4 ccl options engaged set=x'))
        self.assertTrue(ccl_options_smoke.problems(dict(self.ENV, **{options.OPTIONS_FLAG: 'bogus'}), self.engaged()))

    def test_a_fall_back_or_a_mismatch_is_a_finding(self):
        fell = '[PINDIAG] tp4 ccl options fell back rows=64 op=ag reason=keywords differ'
        self.assertTrue(ccl_options_smoke.problems(self.ENV, self.engaged() + '\n' + fell))
        mismatch = '[PINDIAG] tp4 ccl options audit mismatch round=1 shape=64x5120 call=0 op=rs chip 1: 1 of 5 elements differ'
        self.assertTrue(ccl_options_smoke.problems(self.ENV, self.engaged() + '\n' + mismatch))
        self.assertTrue(ccl_options_smoke.problems({}, mismatch))

    def test_the_audit_needs_a_replay_line_of_four_chips_for_each_op_of_the_set(self):
        env = dict(self.ENV, **{options.AUDIT_FLAG: '1'})
        both = '\n'.join([self.engaged(), self.audit('rs'), self.audit('ag')])
        self.assertEqual(ccl_options_smoke.problems(env, both), [])
        self.assertTrue(ccl_options_smoke.problems(env, '\n'.join([self.engaged(), self.audit('ag')])))                  # no reduce-scatter line
        self.assertTrue(ccl_options_smoke.problems(env, '\n'.join([self.engaged(), self.audit('rs'), self.audit('ag', round_number=0)])))
        self.assertTrue(ccl_options_smoke.problems(env, '\n'.join([self.engaged(), self.audit('rs'), self.audit('ag', chips=2)])))
        self.assertTrue(ccl_options_smoke.problems(env, '\n'.join([self.engaged(), self.audit('rs'), self.audit('ag', elements=0)])))
        gather_only = dict(env, **{options.OPTIONS_FLAG: 'ag-w1'})
        self.assertEqual(ccl_options_smoke.problems(gather_only, '\n'.join([self.engaged('ag-w1', rs=0), self.audit('ag')])), [])

    def test_one_owner_per_served_block(self):
        env = dict(self.ENV, **{options.AUDIT_FLAG: '1', 'QWEN_FAST_M3_BLOCKS': '2'})
        one = '\n'.join([self.engaged(), self.audit('rs'), self.audit('ag')])
        self.assertTrue(ccl_options_smoke.problems(env, one))
        two = '\n'.join([one, self.audit('rs', owner='capture2'), self.audit('ag', owner='capture2')])
        self.assertEqual(ccl_options_smoke.problems(env, two), [])


if __name__ == '__main__':
    unittest.main()
