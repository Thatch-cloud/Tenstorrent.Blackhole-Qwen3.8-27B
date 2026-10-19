"""Exact resume P0 and the turn telemetry (docs/tp4-exact-resume.md 5.2, 5.3) on the CPU.

  switches   the three switches are strict, default off, and refused in every half configuration (registry, scheduler install,
             Lever N policy, smoke rule);
  off        with them unset the scheduler graft makes exactly the decisions of the base commit's graft (loaded from git) on a
             sticky DFlash scenario, and the registry snapshots the keys it always had;
  undrop     the graft lifts vLLM's EAGLE drop for the length of the hit lookup only, plans floor2048(P) (plus C0 without the carry),
             and a next turn resumes one chunk higher than today, restored exactly (the fake model's chain check);
  carry      a checkpoint carrying the window rows below it is granted above P - 2048 when they cover [P' - 2048, Q), never
             otherwise; the lifecycle admits such a hit only on a covering window and builds its capture with the carried rows;
  window     the capture splices the carried rows in front of its own: the window of a carried resume equals the cold window of the
             same prompt, row for row (the fake device below holds per-chip bytes), and the audit reads the transport back;
  oracle     the prefix judge's oracle predicts the undrop and the carry; the markers and the gate read the new lines;
  telemetry  TurnMemory classifies every divergence class, counts only; the API side counts tool-call edge whitespace.
"""

import os
import subprocess
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import c2_prefix_gate as gate  # noqa: E402
import dflash_prefill_window as window  # noqa: E402
import exact_resume_smoke  # noqa: E402
import levern_policy  # noqa: E402
import prefix_judge as judge  # noqa: E402
import prefix_markers as markers  # noqa: E402
import prefix_window_carry as carry_module  # noqa: E402
import qwen_prefix_metrics as metrics  # noqa: E402
import qwen_prefix_registry as prefix_registry  # noqa: E402
import qwen_prefix_scheduler_patch as graft  # noqa: E402
import serving_lifecycle  # noqa: E402

import test_qwen_prefix_scheduler_patch as graft_fakes  # noqa: E402
import test_sticky_sessions as sticky_tests  # noqa: E402

CHUNK, BLOCK = prefix_registry.CHUNK, prefix_registry.BLOCK
UNDROP, CARRY, AUDIT = prefix_registry.ENV_UNDROP, prefix_registry.ENV_WINDOW_CARRY, prefix_registry.ENV_WINDOW_CARRY_AUDIT
STICKY = 'QWEN_FAST_STICKY_SESSIONS'
# The commit the flag-off equivalence is held against: tp4/swift (the production tree this branch was merged onto in tp4/next-1). The branch itself grew from
# tp4/fusion-2 (a3d6583c), whose graft predates the prefix tiers' class counters, so the equivalence is held against the graft it was merged with.
BASE = '27af6902'


def base_module(relative):
    """scripts/ci/<relative> at BASE, or None without git history."""
    try:
        result = subprocess.run(['git', 'show', '%s:scripts/ci/%s' % (BASE, relative)], capture_output=True, cwd=str(HERE), timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    module = ModuleType(relative[:-3] + '_base')
    module.__file__ = str(HERE / relative)
    exec(compile(result.stdout.decode('utf-8'), '%s@%s' % (relative, BASE), 'exec'), module.__dict__)
    return module


def no_history(case):
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        case.fail('no git history for %s: the flag-off equivalence did not run' % BASE)
    case.skipTest('no git history for %s' % BASE)


@contextmanager
def environment(values):
    with mock.patch.dict(os.environ):
        for name in (UNDROP, CARRY, AUDIT, STICKY, 'QWEN_PREFIX_TURN_TELEMETRY', 'QWEN_PREFIX_TURN_MEMORY'):
            os.environ.pop(name, None)
        os.environ.update(values)
        yield


# ------------------------------------------------------------------------------------------------
# Switches
# ------------------------------------------------------------------------------------------------
class SwitchTests(unittest.TestCase):
    def test_the_switches_are_strict_and_default_off(self):
        self.assertEqual((prefix_registry.undrop_enabled({}), prefix_registry.window_carry_enabled({}),
                          prefix_registry.window_carry_audit_enabled({})), (False, False, False))
        self.assertTrue(prefix_registry.turn_telemetry_enabled({}), 'the counting is on unless 0')
        for name in (UNDROP, CARRY, AUDIT, 'QWEN_PREFIX_TURN_TELEMETRY'):
            for value in ('yes', 'true', '2', ' 1'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    prefix_registry.strict_switch({name: value}, name)
        with self.assertRaises(ValueError):
            levern_policy.window_carry_on({CARRY: 'on'})
        with self.assertRaises(ValueError):
            carry_module.enabled({CARRY: 'on'})

    def test_every_half_configuration_is_refused(self):
        self.assertEqual(prefix_registry.exact_resume_problems({}), [])
        self.assertEqual(prefix_registry.exact_resume_problems({UNDROP: '1'}), [])
        self.assertEqual(prefix_registry.exact_resume_problems({UNDROP: '1', CARRY: '1', AUDIT: '1'}), [])
        self.assertIn('needs %s=1' % UNDROP, prefix_registry.exact_resume_problems({CARRY: '1'})[0])
        self.assertIn('no carried window to audit', prefix_registry.exact_resume_problems({UNDROP: '1', AUDIT: '1'})[0])
        self.assertIn('only 0 or 1', prefix_registry.exact_resume_problems({UNDROP: 'x'})[0])
        self.assertEqual(prefix_registry.telemetry_problems({'QWEN_PREFIX_TURN_MEMORY': '-1'})[0][:24], 'QWEN_PREFIX_TURN_MEMORY=')

    def test_the_registry_keeps_its_keys_with_everything_off(self):
        plain = prefix_registry.PrefixRegistry(budget_bytes=1, environ={'QWEN_PREFIX_TURN_TELEMETRY': '0'})
        self.assertEqual(sorted(plain.stats), sorted(prefix_registry.STAT_NAMES))
        self.assertIsNone(plain.turns)
        carried = prefix_registry.PrefixRegistry(budget_bytes=1, environ={UNDROP: '1', CARRY: '1'})
        self.assertTrue(set(prefix_registry.WINDOW_STAT_NAMES) <= set(carried.stats))
        self.assertTrue(set(prefix_registry.TURN_STAT_NAMES) <= set(carried.stats))

    def test_levern_valid_step_is_unchanged_off_and_takes_a_carried_rest_on(self):
        with environment({}):
            self.assertFalse(levern_policy.valid_step(33834, 32768, 1066))
            self.assertTrue(levern_policy.valid_step(33834, 30720, 3114))
        with environment({CARRY: '1'}):
            self.assertTrue(levern_policy.valid_step(33834, 32768, 1066))
            self.assertFalse(levern_policy.valid_step(33834, 32768, 1000), 'a step that is neither the whole rest nor aligned')
            self.assertFalse(levern_policy.valid_step(33834, 32000, 1834), 'never off the 2048-token boundary')


# ------------------------------------------------------------------------------------------------
# The scheduler graft: vLLM's drop modelled where vLLM 0.25.1 applies it (the coordinator)
# ------------------------------------------------------------------------------------------------
class VllmEagleCoordinator(graft_fakes.Coordinator):
    """The unitary coordinator's find_longest_cache_hit: drop_eagle_block = 0 in eagle_group_ids (kv_cache_coordinator.py:480-496)."""

    def __init__(self, pool):
        super(VllmEagleCoordinator, self).__init__(pool)
        self.eagle_group_ids = {0}
        self.lookups = []

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        hits = []
        for index in range(max_cache_hit_length // BLOCK):
            block = self.pool.cached_block_hash_to_block.get_one_block((block_hashes[index], 0))
            if block is None:
                break
            hits.append(block)
        dropped = 0 in self.eagle_group_ids
        if dropped and hits:
            hits.pop()
        self.lookups.append(dropped)
        return (hits,), len(hits) * BLOCK


class VllmEagleManager(graft_fakes.Manager):
    def __init__(self, num_blocks):
        super(VllmEagleManager, self).__init__(num_blocks)
        self.coordinator = VllmEagleCoordinator(self.block_pool)

    def get_computed_blocks(self, request):
        blocks, hit = self.coordinator.find_longest_cache_hit(request.block_hashes, request.num_tokens - 1)
        return graft_fakes.Blocks(blocks), hit


def scheduler_for(num_blocks=900):
    scheduler = graft_fakes.FakeScheduler(num_blocks=num_blocks, max_model_len=65536)
    scheduler.kv_cache_manager = VllmEagleManager(num_blocks)
    scheduler.num_lookahead_tokens = 16
    scheduler.vllm_config.speculative_config = SimpleNamespace(method='dflash', num_speculative_tokens=15)
    return scheduler


def tokens(count, seed):
    return graft_fakes.tokens(count, seed)


class Harness(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, graft_fakes.fake_vllm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        environ = mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})
        environ.start()
        self.addCleanup(environ.stop)
        self.logs = []

    def logger(self, message, *values):
        self.logs.append(message % values if values else message)

    def make(self, environ, module=graft):
        scheduler = scheduler_for()
        registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 40, environ=environ)
        with graft_fakes.Quiet():
            registry.enable_mid_loop_capture()
        installed = module.install(scheduler, registry=registry, kill_switch_path=None, logger=self.logger,
                                   stats=prefix_registry.StatsExport(path='', logger=self.logger), environ=environ)
        self.model = graft_fakes.FakeGdnModel(registry)
        return scheduler, installed

    def serve(self, scheduler, request, window=None):
        """One admission, the fake model's prefill (exact restores and captures), and - when `window` - the fast path's export of
        the window rows below floor2048(P) (its payload: the positions themselves), then the request's finish."""
        scheduler.add(request)
        out = scheduler.schedule()
        registry = scheduler._qwen_prefix.registry
        (data,) = out.scheduled_new_reqs
        grant = registry.grant_for(data.req_id)
        with graft_fakes.Quiet():
            self.model.prefill(data.req_id, request.all_token_ids, data.num_computed_tokens)
        prompt = request.num_prompt_tokens
        boundary = prompt // CHUNK * CHUNK
        if window and boundary and registry.stored_at(data.req_id, boundary):
            start = max(0, prompt - CHUNK)
            registry.attach_window(data.req_id, boundary, start, list(range(start, boundary)), 10)
        request.status = SimpleNamespace(name='FINISHED_STOPPED')
        scheduler.finish(request)
        return data.num_computed_tokens, grant


def turn_chain(prompts, seed=1):
    """A conversation: each prompt the previous one extended (the previous answer and a new message), as token lists."""
    base = tokens(max(prompts), seed)
    return [base[:length] for length in prompts]


class GraftOffTests(Harness):
    def scenario(self, module, environ):
        scheduler, installed = self.make(environ, module)
        trace = []
        for index, prompt in enumerate(turn_chain([9000, 10100, 12500, 12900, 16000])):
            q, grant = self.serve(scheduler, graft_fakes.Request('t%d' % index, prompt))
            trace.append((q, grant.describe() if grant else None))
        trace.append(list(scheduler.kv_cache_manager.coordinator.lookups))
        stats = dict(installed.registry.stats)
        for name in ('capture_ms', 'restore_ms') + prefix_registry.TURN_STAT_NAMES:
            stats.pop(name, None)
        return trace, stats, [line for line in self.logs if not line.startswith('install ')]

    def test_the_switches_off_make_the_base_graft_s_decisions(self):
        base = base_module('qwen_prefix_scheduler_patch.py')
        if base is None:
            no_history(self)
        for environ in ({STICKY: '1'}, {STICKY: '1', UNDROP: '0', CARRY: '0'}):
            with self.subTest(environ=environ):
                self.logs = []
                mine = self.scenario(graft, environ)
                self.logs = []
                theirs = self.scenario(base, environ)
                self.assertEqual(mine, theirs)
                self.assertTrue(all(mine[0][-1]), 'every lookup dropped the last block, as vLLM does')


class UndropTests(Harness):
    def test_install_refuses_every_half_configuration(self):
        for environ, words in (({STICKY: '1', CARRY: '1'}, 'needs %s=1' % UNDROP), ({UNDROP: '1'}, 'without sticky sessions'),
                               ({STICKY: '1', UNDROP: '1', AUDIT: '1'}, 'no carried window to audit'),
                               ({STICKY: '1', UNDROP: 'y'}, 'only 0 or 1')):
            with self.subTest(environ=environ), self.assertRaisesRegex(graft.PrefixInstallError, words):
                self.make(environ)
        scheduler = scheduler_for()
        del scheduler.kv_cache_manager.coordinator.eagle_group_ids
        with self.assertRaisesRegex(graft.PrefixInstallError, 'no eagle_group_ids'):
            graft.install(scheduler, registry=prefix_registry.PrefixRegistry(budget_bytes=1), kill_switch_path=None,
                          logger=self.logger, stats=prefix_registry.StatsExport(path='', logger=self.logger),
                          environ={STICKY: '1', UNDROP: '1'})

    def test_the_lookup_keeps_its_last_block_and_the_set_is_restored(self):
        scheduler, installed = self.make({STICKY: '1', UNDROP: '1'})
        coordinator = scheduler.kv_cache_manager.coordinator
        self.assertTrue(installed.undrop and not installed.carry)
        self.assertIn('window carry engaged undrop=1 carry=0 audit=0 ceiling=floor2048(P-2048)', self.logs)
        self.assertIn('install sticky=1 lookahead=16 drop_last=True ceiling=floor2048(P-2048)', self.logs,
                      'the sticky install line is vLLM\'s own drop setting, unchanged')
        first = turn_chain([9000, 12300])
        q0, grant0 = self.serve(scheduler, graft_fakes.Request('a', first[0]))
        self.assertEqual((q0, grant0.capture_positions()), (0, [6144, 8192]), 'floor2048(P) and C0 planned')
        q1, grant1 = self.serve(scheduler, graft_fakes.Request('b', first[1]))
        self.assertEqual(q1, 8192, 'one chunk above today (6144), exact: the fake model checked the restore')
        self.assertEqual(coordinator.lookups, [False, False], 'no lookup dropped a block')
        self.assertEqual(coordinator.eagle_group_ids, {0}, 'the coordinator keeps its own setting between lookups')

    def test_without_the_carry_the_window_ceiling_still_holds(self):
        scheduler, installed = self.make({STICKY: '1', UNDROP: '1'})
        chain = turn_chain([9000, 10100])
        self.serve(scheduler, graft_fakes.Request('a', chain[0]))
        q, _ = self.serve(scheduler, graft_fakes.Request('b', chain[1]))
        self.assertEqual(q, 6144, 'P\' - 2048 = 8052 < 8192: C0 (planned beside floor2048(P)) is the resume')


class CarryTests(Harness):
    ON = {STICKY: '1', UNDROP: '1', CARRY: '1'}

    def test_a_carried_window_lets_the_next_turn_resume_at_the_published_boundary(self):
        scheduler, installed = self.make(self.ON)
        registry = installed.registry
        chain = turn_chain([9000, 10100, 10300])
        q0, grant0 = self.serve(scheduler, graft_fakes.Request('a', chain[0]), window=True)
        self.assertEqual(grant0.capture_positions(), [8192], 'the carry plans only floor2048(P)')
        entry = registry.get(graft_fakes.Request('probe', chain[0]).block_hashes[8192 // BLOCK - 1])
        self.assertEqual((entry.window.start, entry.window.end, entry.window.payload[0]), (9000 - CHUNK, 8192, 9000 - CHUNK))
        q1, grant1 = self.serve(scheduler, graft_fakes.Request('b', chain[1]), window=True)
        self.assertEqual(q1, 8192, 'above P\' - 2048 = 8052, on the carried window')
        self.assertTrue(prefix_registry.window_covers(grant1.checkpoint, 10100, 8192))
        self.assertEqual((registry.stats['window_grants'], registry.stats['window_grant_tokens']), (1, 2048))
        q2, _ = self.serve(scheduler, graft_fakes.Request('c', chain[2]), window=True)
        self.assertEqual(q2, 8192, 'the window carried by turn 0 still covers [P\'\' - 2048, 8192)')

    def test_no_window_or_a_short_one_keeps_the_ceiling(self):
        scheduler, installed = self.make(self.ON)
        chain = turn_chain([9000, 10100])
        self.serve(scheduler, graft_fakes.Request('a', chain[0]), window=False)
        q, _ = self.serve(scheduler, graft_fakes.Request('b', chain[1]), window=True)
        self.assertLess(q, 8052, 'no window on the checkpoint at 8192: never granted above P\' - 2048')
        registry = installed.registry
        entry = prefix_registry.Checkpoint('k', 8192, [0] * 8192)
        entry.window = prefix_registry.Window(8100, 8192, None, 1)
        self.assertFalse(prefix_registry.window_covers(entry, 10100, 8192), 'rows from 8100 cannot cover [8052, 8192)')
        self.assertTrue(prefix_registry.window_covers(entry, 10148, 8192))
        self.assertFalse(prefix_registry.window_covers(entry, 9048, 6144), 'a window is the one of its own boundary')
        self.assertIsNotNone(registry)

    def test_the_window_is_charged_to_the_budget_and_goes_with_its_checkpoint(self):
        registry = prefix_registry.PrefixRegistry(budget_bytes=1000, environ=self.ON)
        registry.begin_step()
        request = SimpleNamespace(all_token_ids=list(range(4200)), num_prompt_tokens=4200)
        registry.stage(prefix_registry.Grant('r', 0, 0, None, None, [(4096, 'k')], request, 4096))
        registry.commit({'r': 0})
        self.assertIsNotNone(registry.capture('r', 4096, rec='rec', carry='carry', nbytes=100, loop_pos=4096))
        self.assertTrue(registry.stored_at('r', 4096))
        self.assertIsNotNone(registry.attach_window('r', 4096, 2152, ['rows'], 300))
        self.assertEqual((registry.bytes, registry.get('k').nbytes), (400, 400))
        self.assertIsNone(registry.attach_window('r', 2048, 0, ['rows'], 1))
        self.assertEqual(registry.stats['window_missing'], 1)
        self.assertIsNone(registry.attach_window('r', 4096, 2152, ['rows'], 5000), 'over the budget: the checkpoint is evicted')
        self.assertEqual((registry.bytes, len(registry.entries)), (0, 0))
        registry.forget_request('r')
        self.assertFalse(registry.stored_at('r', 4096))


# ------------------------------------------------------------------------------------------------
# The window: a fake mesh that keeps every chip's bytes
# ------------------------------------------------------------------------------------------------
CHIPS, WIDTH, TAPS = 4, 8, 2


class Device(object):
    """A mesh tensor: data [chips, d0, d1, d2, d3]; shape is the per-chip shape, as ttnn reports a mesh tensor's."""

    def __init__(self, data):
        self.data = data
        self.dtype = torch.bfloat16

    @property
    def shape(self):
        return tuple(self.data.shape[1:])


class Ops(object):
    bfloat16 = torch.bfloat16
    TILE_LAYOUT = 'tile'
    DRAM_MEMORY_CONFIG = 'dram'

    def __init__(self):
        self.deallocated = []
        self.uploads = 0

    def ConcatMeshToTensor(self, mesh, dim):
        assert dim == 0
        return 'concat'

    def ShardTensorToMesh(self, mesh, dim):
        assert dim == 0
        return 'shard'

    def slice(self, value, begins, ends):
        index = (slice(None),) + tuple(slice(b, e) for b, e in zip(begins, ends))
        return Device(value.data[index].clone())

    def clone(self, value, memory_config=None):
        return Device(value.data.clone())

    def concat(self, values, dim, memory_config=None):
        return Device(torch.cat([value.data for value in values], dim=dim + 1))

    def deallocate(self, value):
        self.deallocated.append(value)

    def get_device_tensors(self, value):
        # gdn_multitoken_conv.addresses (the pair helper the capture imports) reads two shards' buffer addresses
        return [SimpleNamespace(buffer_address=lambda value=value: id(value)), SimpleNamespace(buffer_address=lambda value=value: id(value) + 1)]

    def to_torch(self, value, mesh_composer=None):
        assert mesh_composer == 'concat'
        return value.data.reshape((value.data.shape[0] * value.data.shape[1],) + tuple(value.data.shape[2:])).clone()

    def from_torch(self, host, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        assert mesh_mapper == 'shard' and dtype == torch.bfloat16
        self.uploads += 1
        return Device(host.reshape((CHIPS, host.shape[0] // CHIPS) + tuple(host.shape[1:])).clone())


def feature(tap, positions):
    """The target taps of these positions on every chip: a function of (tap, position, chip, column) only, as a model's are of the
    prompt up to the position (bfloat16-exact integers)."""
    rows = torch.tensor(list(positions), dtype=torch.float32)
    chips = torch.arange(CHIPS, dtype=torch.float32).view(CHIPS, 1, 1, 1, 1)
    columns = torch.arange(WIDTH, dtype=torch.float32).view(1, 1, 1, 1, WIDTH)
    value = (rows.view(1, 1, 1, -1, 1) % 251) + 7 * columns + 3 * chips + 11 * tap
    return value.to(torch.bfloat16)


class TapCapture(object):
    """target_features.LayerOutputCapture on the fake mesh: the chunk the model is running yields every tap's output over its bucket
    (padded rows hold garbage), and the snapshot keeps the window rows as the real one does."""

    def __init__(self, model, layer_ids, snapshot, release, storage_ids):
        self.model, self.snapshot, self.release = model, snapshot, release
        self.values = []

    @contextmanager
    def capture(self):
        yield self
        start, valid, bucket = self.model.last
        for tap in range(TAPS):
            value = feature(tap, range(start, start + bucket))
            value[:, :, :, valid:, :] = 99
            self.values.append(self.snapshot(Device(value)))

    def outputs(self):
        return tuple(self.values)

    def close(self):
        for value in self.values:
            self.release(value)
        self.values = []


class ChunkModel(sticky_tests.RouteModel):
    def __init__(self):
        super(ChunkModel, self).__init__()
        inner = self._forward_prefill_chunk_masked_tp
        self.mesh_device = 'mesh'

        def chunk(token_buf, valid_len, chunk_start, page_table, bucket, **kwargs):
            self.last = (chunk_start, valid_len, bucket)
            return inner(token_buf, valid_len, chunk_start, page_table, bucket, **kwargs)

        self._forward_prefill_chunk_masked_tp = chunk


class WindowTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(window.tp_shapes, 'active', lambda: SimpleNamespace(draft_taps=WIDTH))
        patcher.start()
        self.addCleanup(patcher.stop)
        tap_patch = mock.patch.object(window, 'LayerOutputCapture', TapCapture)
        tap_patch.start()
        self.addCleanup(tap_patch.stop)

    def prefill(self, ops, position, start=0, carry=None, audit=False):
        model = ChunkModel()
        kwargs = dict(start=start, prefix_route=True)
        if carry is not None:
            kwargs['carry'] = carry
        capture = window.PrefillWindowCapture(ops, model, position, tuple(range(TAPS)), **kwargs)
        capture.carry_audit = audit
        with capture.capture():
            model._qwen_prefix_prefill_slots('tokens', 'pages', [0], valid_lens=[position], starts=[start], req_ids=['r'])
        self.assertTrue(capture.complete)
        return capture

    def cold(self, position):
        lo = max(0, position - CHUNK)
        return [feature(tap, range(lo, position)) for tap in range(TAPS)]

    def test_a_carried_resume_has_the_cold_window_row_for_row(self):
        ops = Ops()
        first = self.prefill(ops, 9000)
        for got, want in zip(first.outputs(), self.cold(9000)):
            self.assertTrue(torch.equal(got.data, want))
        start, payload, nbytes, sha = carry_module.export(ops, 'mesh', first.outputs(), 9000, 8192)
        self.assertEqual((start, tuple(payload[0].shape)), (6952, (CHIPS, 1, 8192 - 6952, WIDTH)))
        stored = prefix_registry.Window(start, 8192, payload, nbytes, sha)
        for prompt in (10100, 10239):
            with self.subTest(prompt=prompt):
                carried = carry_module.Carried(stored, prompt, 8192)
                second = self.prefill(ops, prompt, start=8192, carry=carried, audit=True)
                self.assertEqual(carried.transport, 'none', 'the rows are uploaded when the window is first read')
                outputs = second.outputs()
                self.assertEqual(carried.transport, 'exact')
                for got, want in zip(outputs, self.cold(prompt)):
                    self.assertEqual(got.shape[2], CHUNK)
                    self.assertTrue(torch.equal(got.data, want), 'the spliced window is the cold window')
                self.assertEqual(carry_module.window_sha(ops, 'mesh', second.outputs()),
                                 carry_module.window_sha(ops, 'mesh', [Device(value) for value in self.cold(prompt)]))
                uploaded = list(carried.uploaded)
                second.close()
                self.assertTrue(all(any(tensor is gone for gone in ops.deallocated) for tensor in uploaded))
        first.close()

    def test_a_carried_resume_beside_the_aligned_window_has_the_cold_window_row_for_row(self):
        """QWEN_FAST_TAP_WINDOW_ALIGNED is ON in the ship profile: the capture that took the rows (the aligned join) and the capture that
        carries them (parts behind the carried rows) must both hand the drafter the cold window, and the aligned flag must not change one byte."""
        ops = Ops()
        with mock.patch.dict(os.environ, {window.ALIGNED_FLAG: '1'}):
            first = self.prefill(ops, 9000)
            self.assertTrue(first.aligned)
            for got, want in zip(first.outputs(), self.cold(9000)):
                self.assertTrue(torch.equal(got.data, want))
            start, payload, nbytes, sha = carry_module.export(ops, 'mesh', first.outputs(), 9000, 8192)
            stored = prefix_registry.Window(start, 8192, payload, nbytes, sha)
            for prompt in (10100, 10239):
                with self.subTest(prompt=prompt):
                    carried = carry_module.Carried(stored, prompt, 8192)
                    second = self.prefill(ops, prompt, start=8192, carry=carried, audit=True)
                    self.assertTrue(second.aligned)
                    outputs = second.outputs()
                    self.assertEqual(carried.transport, 'exact')
                    for got, want in zip(outputs, self.cold(prompt)):
                        self.assertEqual(got.shape[2], CHUNK)
                        self.assertTrue(torch.equal(got.data, want), 'the spliced window is the cold window with the aligned flag on')
                    uploaded = list(carried.uploaded)
                    second.close()
                    self.assertTrue(all(any(tensor is gone for gone in ops.deallocated) for tensor in uploaded))
            first.close()

    def test_a_prompt_on_the_boundary_exports_its_whole_window_without_releasing_it(self):
        ops = Ops()
        capture = self.prefill(ops, 8192)
        outputs = capture.outputs()
        start, payload, _, _ = carry_module.export(ops, 'mesh', outputs, 8192, 8192)
        self.assertEqual((start, payload[0].shape[2]), (6144, CHUNK))
        self.assertFalse(any(any(tensor is gone for gone in ops.deallocated) for tensor in outputs), 'the window outputs stay alive')
        capture.close()

    def test_the_ledger_counts_exactly_the_carried_rows(self):
        stored = prefix_registry.Window(6952, 8192, [torch.zeros(CHIPS, 1, 1240, WIDTH, dtype=torch.bfloat16)], 1)
        carried = carry_module.Carried(stored, 10100, 8192)
        self.assertEqual((carried.start, carried.end, carried.rows), (8052, 8192, 140))
        self.assertEqual(window.resume_start(10100, 8192, 140), 8192)
        for start, rows in ((8192, 139), (8192, 0), (8000, 140), (10240, 0)):
            with self.subTest(start=start, rows=rows), self.assertRaises(ValueError):
                window.resume_start(10100, start, rows)
        with self.assertRaisesRegex(ValueError, 'carries no window covering'):
            carry_module.Carried(stored, 8900, 8192)
        model = ChunkModel()
        with self.assertRaisesRegex(ValueError, 'must cover'):
            window.PrefillWindowCapture(Ops(), model, 10100, (0,), start=6144, prefix_route=True, carry=carried)

    def test_after_prefill_files_the_rows_and_logs_the_audit(self):
        ops = Ops()
        capture = self.prefill(ops, 9000)
        lines = []
        registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 40, environ={UNDROP: '1', CARRY: '1'})
        registry.begin_step()
        request = SimpleNamespace(all_token_ids=list(range(9000)), num_prompt_tokens=9000)
        registry.stage(prefix_registry.Grant('r', 0, 0, None, None, [(8192, 'k')], request, 8192))
        registry.commit({'r': 0})
        registry.capture('r', 8192, rec='rec', carry='carry', nbytes=10, loop_pos=8192)
        carry_module.after_prefill(capture, 'r', registry, ops, 'mesh', lambda template, *values: lines.append(template.format(*values)),
                                   audit=True)
        self.assertEqual(registry.get('k').window.end, 8192)
        self.assertTrue(lines[0].startswith('[PINDIAG] prefix: window carry stored req=r pos=8192 rows=1240 bytes='))
        found = markers.scan(lines)
        self.assertEqual(found['window_stored'][0]['rows'], 1240)
        self.assertEqual((found['window_audits'][0]['carried'], found['window_audits'][0]['exact']), (0, True))
        lines = []
        carry_module.after_prefill(capture, 'nobody', registry, ops, 'mesh', lambda template, *values: lines.append(template.format(*values)))
        self.assertEqual(lines, [], 'no checkpoint stored by this request: nothing read, nothing logged')
        capture.close()


# ------------------------------------------------------------------------------------------------
# The lifecycle
# ------------------------------------------------------------------------------------------------
class LifecycleCarryTests(sticky_tests.LifecycleFixture):
    def admit(self, prompt, computed, window=None, carry='1'):
        registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 30)
        grant = sticky_tests.committed(registry, 'request', computed, [1] * prompt)
        if window is not None:
            grant.checkpoint.window = window
        with sticky_tests.sticky_env('1'), mock.patch.dict(os.environ, {CARRY: carry}):
            case = self.fixture(prompt=prompt, computed=computed)
        with sticky_tests.registry_holder(sticky_tests.holder_with(registry)):
            try:
                case.result = case.worker.execute_model(case.scheduled)
                case.error = None
            except ValueError as error:
                case.error = error
        return case

    def test_a_covered_resume_above_the_window_ceiling_is_admitted_with_its_rows(self):
        stored = prefix_registry.Window(6952, 8192, [torch.zeros(CHIPS, 1, 1240, WIDTH, dtype=torch.bfloat16)], 1)
        case = self.admit(10100, 8192, stored)
        self.assertIsNone(case.error)
        self.assertTrue(case.lifecycle.window_carry)
        (call,) = case.capture_factory.call_args_list
        self.assertEqual(call.args, (10100,))
        self.assertEqual(call.kwargs['start'], 8192)
        self.assertEqual((call.kwargs['carry'].start, call.kwargs['carry'].rows), (8052, 140))

    def test_an_uncovered_resume_is_still_an_engine_fault(self):
        for stored, carry in ((None, '1'), (prefix_registry.Window(8100, 8192, [], 1), '1'),
                              (prefix_registry.Window(6952, 8192, [], 1), '0')):
            with self.subTest(carry=carry, stored=stored):
                case = self.admit(10100, 8192, stored, carry=carry)
                self.assertIn('R is above the prompt minus 2048', str(case.error))
                self.assertTrue(case.lifecycle.failed)

    def test_the_switch_off_imports_nothing_new(self):
        with mock.patch.dict(sys.modules):
            sys.modules.pop('prefix_window_carry', None)
            case = self.admit(12000, 8192, carry='0')
            self.assertNotIn('prefix_window_carry', sys.modules)
        self.assertFalse(case.lifecycle.window_carry)
        with self.assertRaisesRegex(ValueError, 'only 0 or 1'):
            self.admit(12000, 8192, carry='on')

    def test_an_ordinary_resume_builds_the_capture_it_always_built(self):
        case = self.admit(12000, 8192)
        self.assertIsNone(case.error)
        case.capture_factory.assert_called_once_with(12000, start=8192)


# ------------------------------------------------------------------------------------------------
# The oracle, the markers, the gate, the smoke
# ------------------------------------------------------------------------------------------------
class OracleTests(unittest.TestCase):
    def test_the_oracle_predicts_what_the_graft_grants(self):
        """The judge's oracle is an independent model of the engine: on the same conversation it must name the Q and the plan the
        scheduler graft grants, with the switches off, with the undrop, and with the carry (its windows filed as the fast path does)."""
        chain = turn_chain([9000, 10100, 12300, 12500, 16000])
        modes = (('sticky', {STICKY: '1'}, {}), ('undrop', {STICKY: '1', UNDROP: '1'}, dict(undrop=True)),
                 ('carry', {STICKY: '1', UNDROP: '1', CARRY: '1'}, dict(undrop=True, carry=True)))
        seen = {}
        for name, environ, kwargs in modes:
            with self.subTest(mode=name):
                oracle = judge.Oracle(capacity=100, sticky=True, **kwargs)
                predicted = [(entry['q'], entry['plan']) for entry in (oracle.admit(graft_fakes.SALT, turn) for turn in chain)]
                harness = Harness('run')
                harness.setUp()
                try:
                    scheduler, _ = harness.make(environ)
                    granted = []
                    for index, turn in enumerate(chain):
                        q, grant = harness.serve(scheduler, graft_fakes.Request('t%d' % index, turn), window='carry' in name)
                        granted.append((q, grant.capture_positions() if grant else []))
                finally:
                    harness.doCleanups()
                self.assertEqual(predicted, granted)
                seen[name] = [q for q, _ in granted]
        self.assertEqual(seen['sticky'], [0, 6144, 6144, 10240, 10240])
        self.assertTrue(all(b >= a for a, b in zip(seen['sticky'], seen['undrop'])))
        self.assertTrue(all(b >= a for a, b in zip(seen['undrop'], seen['carry'])))
        self.assertGreater(sum(seen['carry']), sum(seen['sticky']))
        self.assertEqual(judge.Oracle(capacity=100, sticky=True, carry=True).carry, False, 'the carry needs the undrop')

    def test_an_unsticky_oracle_is_unchanged(self):
        chain = turn_chain([9000, 10100])
        plain = judge.Oracle(capacity=100)
        self.assertEqual([plain.admit('s', turn)['q'] for turn in chain], [0, 8192])


class GateTests(unittest.TestCase):
    def arm(self, env):
        return dict(sticky=True, prefix=True, served_env=dict(env))

    def test_the_switches_reach_the_driver_only_on_a_sticky_arm(self):
        self.assertEqual(gate.exact_resume_switches(self.arm({UNDROP: '1', CARRY: '1'})), dict(undrop=True, carry=True))
        self.assertEqual(gate.exact_resume_switches(dict(sticky=False, served_env={UNDROP: '1'})), dict(undrop=False, carry=False))

    def test_findings(self):
        lines = ['[PINDIAG] prefix: window carry engaged undrop=1 carry=1 audit=1 ceiling=x',
                 '[PINDIAG] prefix: window carry carried req=chatcmpl-pfx-a-2-hit-1a2b3c4d Q=8192 P=10100 rows=140',
                 '[PINDIAG] prefix: window carry audit req=chatcmpl-pfx-a-2-hit-1a2b3c4d P=10100 Q=8192 carried=140 window_sha=%s '
                 'transport=exact exact=True' % ('a' * 64),
                 '[PINDIAG] prefix: window carry audit req=chatcmpl-pfx-a-2-cold-5e6f7a8b P=10100 Q=0 carried=0 window_sha=%s '
                 'transport=none exact=True' % ('a' * 64)]
        scanned = markers.scan(lines)
        pairs = [dict(cold='pfx-a-2-cold', hit='pfx-a-2-hit')]
        arm = self.arm({UNDROP: '1', CARRY: '1', AUDIT: '1'})
        problems, missing, report = gate.exact_resume_findings(arm, scanned, [], pairs)
        self.assertEqual((problems, missing), ([], []))
        self.assertIn('exact resume audit: 2 window lines, 1 pairs compared, 0 differ', report)
        scanned = markers.scan(lines[:-1] + [lines[-1].replace('a' * 64, 'b' * 64)])
        problems, _, _ = gate.exact_resume_findings(arm, scanned, [], pairs)
        self.assertIn('differs from its cold twin', problems[0])
        problems, _, _ = gate.exact_resume_findings(arm, markers.scan(lines[1:]), [], pairs)
        self.assertIn('no "[PINDIAG] prefix: window carry engaged" line', problems[0])
        self.assertEqual(gate.exact_resume_findings(self.arm({}), markers.scan(lines), [], pairs), ([], [], []))

    def test_the_smoke_rule(self):
        engaged = '[PINDIAG] prefix: window carry engaged undrop=1 carry=1 audit=0 ceiling=x\n'
        on = {UNDROP: '1', CARRY: '1', 'QWEN_PREFIX_REUSE': '1', STICKY: '1'}
        self.assertEqual(exact_resume_smoke.problems(on, engaged), [])
        self.assertEqual(exact_resume_smoke.problems({}, 'nothing'), [])
        self.assertIn('without', exact_resume_smoke.problems({}, engaged)[0])
        self.assertIn('no engaged line', exact_resume_smoke.problems(on, 'nothing')[-1])
        self.assertIn('the engaged line says', exact_resume_smoke.problems(dict(on, **{AUDIT: '1'}), engaged)[-1])
        self.assertIn('%s=1 without %s=1' % (CARRY, UNDROP), exact_resume_smoke.problems({CARRY: '1'}, engaged))
        self.assertIn('did not read back', exact_resume_smoke.problems(on, engaged + '[PINDIAG] prefix: window carry audit req=r P=1 '
                                                                       'Q=0 carried=0 window_sha=ab transport=mismatch exact=False')[0])


class ContractTests(unittest.TestCase):
    def profile(self, env, gate_only=True):
        base = dict(QWEN_PREFIX_REUSE='1', QWEN_FAST_STICKY_SESSIONS='1')
        base.update(env)
        return dict(name='p', gate_only=gate_only, env=base)

    def test_the_contract_refuses_every_misuse_and_drops_inherited_switches(self):
        import serving_c2_contract as contract

        self.assertEqual(contract.exact_resume_problems(dict(name='p', env={})), [])
        self.assertEqual(contract.exact_resume_problems(self.profile({UNDROP: '1', CARRY: '1', AUDIT: '1'})), [])
        self.assertIn('gate-only', contract.exact_resume_problems(self.profile({UNDROP: '1'}, gate_only=False))[0])
        self.assertIn('without %s=1' % UNDROP, contract.exact_resume_problems(self.profile({CARRY: '1'}))[0])
        self.assertIn('without %s=1' % CARRY, contract.exact_resume_problems(self.profile({UNDROP: '1', AUDIT: '1'}))[0])
        self.assertIn('only 0 or 1', contract.exact_resume_problems(self.profile({UNDROP: 'yes'}))[0])
        self.assertIn('need QWEN_PREFIX_REUSE=1', contract.exact_resume_problems(dict(name='p', gate_only=True, env={UNDROP: '1'}))[0])
        environ = {UNDROP: '1', CARRY: '1', 'OTHER': 'x'}
        contract.apply_environment(dict(name='p', env={}, mesh_graph_descriptor='d'), environ)
        self.assertNotIn(UNDROP, environ)
        self.assertNotIn(CARRY, environ)


class PlanTests(unittest.TestCase):
    def test_the_plans_are_known_to_the_job_reader_and_the_gate(self):
        import c2_serving_job

        for plan in ('exact-resume', 'exact-resume-timed'):
            self.assertIn(plan, c2_serving_job.PREFIX_PLANS)
            self.assertIn(plan, gate.PLANS)
        self.assertEqual(gate.PLAN_ARMS['exact-resume'][0][1:3], ('exact_resume', 'prefix'))
        self.assertTrue(gate.PLAN_ARMS['exact-resume'][0][5], 'the judged arm holds the oracle strictly')
        self.assertIn('exact_resume_timed', gate.TIMED_SCENARIOS)
        self.assertNotIn('exact_resume', gate.TIMED_SCENARIOS)

    def test_the_scenario_sends_a_pair_per_turn_and_fits_the_context(self):
        import prefix_replay as replay

        class Conv(object):
            def __init__(self, name):
                self.name, self.inputs = name, []

            def extend(self, size):
                self.inputs.append(size)

        class Driver(object):
            max_tokens, prompt_limit = 256, 100000

            def __init__(self):
                self.pairs, self.events, self.fitted = [], [], []

            def context(self):
                return 262144

            def pair(self, conv, case):
                self.pairs.append((case, list(conv.inputs)))
                return dict(ok=True)

            def answer(self, conv, hit):
                pass

            def event(self, name, **fields):
                self.events.append((name, fields))

        driver = Driver()
        with mock.patch.object(replay, 'fitted_conversation', lambda driver_, name, target: (driver.fitted.append(target) or Conv(name), 0)):
            replay.SCENARIOS['exact_resume_timed'](driver, agents=8, turns=3)
        self.assertEqual(driver.fitted[:2], [9000, 33000])
        self.assertLess(driver.fitted[2], 100000, 'the long first turn is held under the prompt limit with room for its turns')
        self.assertEqual(len(driver.pairs), 3 * (1 + len(replay.EXACT_RESUME_INPUTS)))
        self.assertEqual(driver.pairs[-1], ('er-120000', list(replay.EXACT_RESUME_INPUTS)))

    def test_the_comparator_pairs_hits_and_fails_on_a_different_answer(self):
        import exact_resume_compare as compare

        def hit(conv, turn, q, ttft, ids=(1, 2)):
            return dict(role='hit', continuation=True, ok=True, conv=conv, turn=turn, prompt_tokens=10100, token_ids=list(ids),
                        ttft_s=ttft, markers=dict(q=q))
        control = [hit('er-9000', 1, 6144, 2.5), hit('er-9000', 2, 8192, 2.4), dict(role='cold', conv='er-9000', turn=1)]
        lever = [hit('er-9000', 1, 8192, 2.0), hit('er-9000', 2, 8192, 2.1)]
        lines = []
        rows, problems = compare.compare(control, lever)
        self.assertEqual(problems, [])
        self.assertEqual([(row['tail_a'], row['tail_b'], round(row['delta'], 2)) for row in rows], [(3956, 1908, -0.5), (1908, 1908, -0.3)])
        rows, problems = compare.compare(control, [hit('er-9000', 1, 8192, 2.0, ids=(1, 3)), hit('er-9000', 2, 8192, 2.1)])
        self.assertEqual(len(problems), 1, 'one divergence, reported once per conversation')
        compare.render(rows, problems, 'control', 'lever', out=lines.append)
        self.assertTrue(lines[-1].startswith('EXACT_RESUME_COMPARE verdict=FAIL'))


# ------------------------------------------------------------------------------------------------
# Telemetry
# ------------------------------------------------------------------------------------------------
class TurnTelemetryTests(unittest.TestCase):
    def memory(self):
        registry = prefix_registry.PrefixRegistry(budget_bytes=1)
        return registry.turns, registry.stats

    def finished(self, request_id, prompt, answer, status='FINISHED_STOPPED', salt=graft_fakes.SALT):
        request = graft_fakes.Request(request_id, prompt + answer, salt=salt, prompt=len(prompt))
        request.status = SimpleNamespace(name=status)
        return request

    def classify(self, memory, stats, request_id, prompt):
        return memory.classify(graft_fakes.Request(request_id, prompt), stats)

    def test_every_class(self):
        memory, stats = self.memory()
        prompt = tokens(5000, 3)
        answer = tokens(300, 4)
        self.assertTrue(memory.record(self.finished('t0', prompt, answer), stats))
        new = tokens(700, 5)
        cases = (
            ('extends', prompt + answer + new),
            ('in_answer', prompt + answer[:100] + new),
            ('at_decode_start', prompt + new),
            ('at_p_minus_1', prompt[:-1] + new),
            ('at_p_minus_2', prompt[:-2] + new),
            ('before_prompt_end', prompt[:4994] + new),
            ('earlier', prompt[:3000] + new),
            ('first', tokens(6000, 9)),
        )
        for index, (name, text) in enumerate(cases):
            with self.subTest(name=name):
                self.assertEqual(self.classify(memory, stats, 'n%d' % index, text), name)
        self.assertEqual(self.classify(memory, stats, 'n0', cases[0][1]), None, 'once per request')
        self.assertEqual(stats['turn_new_lt1024'], 1)
        self.assertEqual(stats['turn_in_answer_lt256'], 1)
        self.assertEqual(stats['turn_out_lt1024'], 6, 'every continuation matched by its anchor counts the previous answer')

    def test_counts_only_and_bounded(self):
        memory, stats = self.memory()
        self.assertFalse(memory.record(self.finished('u', tokens(3000, 1), tokens(10, 2), salt=None), stats), 'unsalted')
        self.assertFalse(memory.record(self.finished('a', tokens(3000, 1), tokens(10, 2), status='FINISHED_ABORTED'), stats))
        memory.capacity = 2
        for index in range(4):
            memory.record(self.finished('r%d' % index, tokens(3000, 10 + index), tokens(10, 2)), stats)
        self.assertEqual((len(memory.anchors), stats['turn_forgotten']), (2, 2))
        self.assertTrue(all(isinstance(value[3], prefix_registry.array) for value in memory.anchors.values()))

    def test_the_graft_counts_salted_admissions_and_remembers_finished_turns(self):
        harness = Harness('run')
        harness.setUp()
        try:
            scheduler, installed = harness.make({STICKY: '1'})
            first, answer = tokens(9000, 1), tokens(300, 7)
            harness.serve(scheduler, graft_fakes.Request('a', first + answer, prompt=9000))
            harness.serve(scheduler, graft_fakes.Request('b', first + answer + tokens(800, 8)))
            stats = installed.registry.stats
            self.assertEqual((stats['turn_first'], stats['turn_extends'], stats['turn_records']), (1, 1, 1),
                             'b is remembered too, but it generated no answer: nothing to remember')
            unsalted = graft_fakes.Request('c', first + answer + tokens(900, 9), salt=None)
            harness.serve(scheduler, unsalted)
            self.assertEqual(stats['turn_first'] + stats['turn_extends'], 2, 'an unsalted request is never classified')
        finally:
            harness.doCleanups()

    def test_the_tool_argument_whitespace_is_counted_and_the_conversion_unchanged(self):
        module = types.ModuleType(metrics.PARSER_MODULE)
        calls = []
        module._qwen3_arg_converter = lambda raw, partial: calls.append((raw, partial)) or '{}'
        before = metrics.tool_arg_counts()
        self.assertTrue(metrics.install_tool_arg_counter(lambda name, callback: self.fail('already imported'),
                                                          modules={metrics.PARSER_MODULE: module}))
        converter = module._qwen3_arg_converter
        self.assertEqual(converter('<parameter=path>\nsrc/a.py\n</parameter>\n', True), '{}')
        self.assertEqual(converter('<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=content>\ndef f():\n    pass\n\n</parameter>\n',
                                   False), '{}')
        self.assertEqual(converter('<parameter=old>\n    return 1\n</parameter>\n', False), '{}')
        after = metrics.tool_arg_counts()
        delta = dict((key, after[key] - before[key]) for key in after)
        self.assertEqual(delta, dict(calls=2, calls_edge_whitespace=2, params=3, params_edge_whitespace=2))
        self.assertEqual(len(calls), 3, 'the original converter ran every time')
        self.assertFalse(metrics.wrap_arg_converter(module), 'wrapped once')


if __name__ == '__main__':
    unittest.main()
