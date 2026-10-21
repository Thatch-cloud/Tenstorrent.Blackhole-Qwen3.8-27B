"""The NVMe cold tier (qwen_kv_nvme) under the real scheduler graft, on the fake world of the host-tier tests, over a file-backed fake block device.

Pure python, stdlib only, no card, no drive: the device is a sparse file in a temp directory (qwen_nvme_store's alignment rules enforced on it), the tier IO is
test_qwen_prefix_scheduler_patch.FakeKvIO, and the model is its FakeGdnModel, which restores every checkpoint against the cold chain of the row's own tokens - so every
scenario below also proves that what came back from the NVMe is exactly what a cold prefill would have computed (the model raises otherwise), and check_kv() that every
block the pool serves, and every record of the host tier, holds the bytes of its hash.

Held:
  * the flags: strict parsing, the refusals (p2p not built, a name without the switch, the switch without the host tier), the registry refusing to start on a bad one;
  * FLAG-OFF IDENTITY: with the host tier on and the NVMe names unset, the registry, the tier and the device calls of a random traffic are what they were on tp4/next-1
    (the pinned digest), and no key, slot or thread of the NVMe tier exists;
  * spill on host-tier eviction (async: the scheduler never waits for the drive, a full queue drops the record as the host tier always did), restore on a miss in both
    host stores (blocks and checkpoint), a restart (a new registry on the same device) that still restores, TTL, LRU, a full device, fingerprints, digests;
  * a crash in the middle of a spill on a simulated disk leaves a device that opens, with only whole entries;
  * per-tenant deletion, the whole-tier wipe (reset_prefix_cache, the kill switch), the command directory;
  * a restore beside a busy writer; the sliced restore reading NVMe blocks a slice at a time; the direct-backend seam with a double.
"""

import array
import errno
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qwen_kv_nvme as kvnvme  # noqa: E402
import prefix_markers  # noqa: E402
import qwen_nvme_store as store_module  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import qwen_prefix_scheduler_patch as patch_module  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from qwen_prefix_registry import BLOCK, CHUNK, PrefixRegistry  # noqa: E402
from test_qwen_nvme_store import SimDisk  # noqa: E402
from test_qwen_prefix_scheduler_patch import FakeKvIO, Quiet, Request, kv_content  # noqa: E402
from test_qwen_prefix_tiers import SMALL_POOL, Rig, salt_of, span  # noqa: E402

PROFILE = 'p-test'
ALIGN = store_module.ALIGN


def nvme_environ(device, **extra):
    environ = {'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': device, 'QWEN_PREFIX_NVME_VERIFY': 'all', 'QWEN_PREFIX_NVME_QUEUE_BYTES': str(16 << 20),
               'QWEN_C2_PROFILE': PROFILE, 'QWEN_PREFIX_NVME_COMPAT': 'rev1', 'QWEN_PREFIX_NVME_THREADS': '2',
               'QWEN_PREFIX_NVME_RETENTION_ACK': '1',
               # (the rig is the evict mode with no stop hooks unless a test asks otherwise: the defaults - write-through, a 4 s drain that arms process-wide hooks - have their own tests)
               'QWEN_PREFIX_NVME_SPILL': 'evict', 'QWEN_PREFIX_NVME_DRAIN_S': '0'}
    environ.update(extra)
    return environ


class NvmeRig(Rig):
    """The host-tier rig with the NVMe tier on over a sparse file. `file_factory` swaps the device class (a simulated disk); `restart()` is a new process on the same device."""

    def __init__(self, device, nvme_env=None, environ=None, file_factory=None, snap_bytes=1 << 20, journal_bytes=1 << 20, **kwargs):
        merged = dict(environ or {})
        merged.update(nvme_environ(device, **(nvme_env or {})))
        self.device = device
        self.file_factory = file_factory
        self.early = []
        self.extra_patches = [mock.patch.object(registry_module, 'log', self.early_logger)]       # (the NVMe tier logs through the registry's log: keep its lines too)
        parse_plan = kvnvme.nvme_plan

        def geometry(environ, tier_config):
            plan = parse_plan(environ, tier_config)
            if plan is not None:
                plan.snap_bytes, plan.journal_bytes = snap_bytes, journal_bytes
            return plan

        self.extra_patches.append(mock.patch.object(kvnvme, 'nvme_plan', geometry))
        if file_factory is not None:
            original = store_module.NvmeStore.open.__func__

            def opener(cls, path, **options):
                options['file_factory'] = file_factory
                return original(cls, path, **options)

            self.extra_patches.append(mock.patch.object(kvnvme.NvmeStore, 'open', classmethod(opener)))
        for patcher in self.extra_patches:
            patcher.start()
        try:
            Rig.__init__(self, environ=merged, **kwargs)
        except BaseException:
            for patcher in reversed(self.extra_patches):
                patcher.stop()
            raise
        self.lines = self.early + self.lines

    def early_logger(self, message, *values):
        line = message % values if values else message
        if hasattr(self, 'lines'):
            self.lines.append(line)
        else:
            self.early.append(line)

    @property
    def nvme(self):
        return self.registry.nvme

    def close(self):
        if self.registry.nvme is not None:
            try:
                self.registry.nvme.close()
            except Exception:
                pass
        Rig.close(self)
        for patcher in reversed(self.extra_patches):
            patcher.stop()

    def settle(self):
        """The writer has everything it was given."""
        assert self.nvme.flush(30.0), 'the NVMe writer did not drain'

    def session_somewhere(self, request, count=64):
        """How many of the first `count` blocks of the session the host tier or the NVMe holds."""
        keys = request.block_hashes[:count]
        return sum(1 for key in keys if self.registry.tier.has(key) or self.nvme.store.lookup(store_module.KV, key) is not None)

    def nvme_has_session(self, request, count=None):
        keys = request.block_hashes[:count] if count else request.block_hashes
        return [self.nvme.store.lookup(store_module.KV, key) is not None for key in keys]


class NvmeCase(unittest.TestCase):
    def setUp(self):
        shm = '/dev/shm' if os.path.isdir('/dev/shm') and os.access('/dev/shm', os.W_OK) else None
        self.directory = tempfile.mkdtemp(prefix='qkvn-test-', dir=shm)
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.device = os.path.join(self.directory, 'nvme.img')
        with open(self.device, 'wb') as handle:
            handle.truncate(64 << 20)
        self.rigs = []

    TIER = 200       # blocks: bigger than any one flush of the scenarios (a flush larger than the tier drops its top blocks, as the host tier always did)

    def rig(self, tier=None, num_blocks=SMALL_POOL, budget=1 << 20, **kwargs):
        made = NvmeRig(self.device, tier=tier or self.TIER, num_blocks=num_blocks, budget=budget, **kwargs)
        self.addCleanup(made.close)
        self.rigs.append(made)
        return made

    def push_out(self, made, request, fillers=5, tenant='bravo'):
        """Serve `request`, then enough other tenants' traffic that its blocks leave the device pool AND the host tier; wait for the writer."""
        made.serve(request)
        self.round = getattr(self, 'round', 0) + 1
        for index in range(fillers):
            made.serve(Request('fill-%d-%s-%d' % (self.round, request.request_id, index), span(6000, 200 * self.round + index),
                               salt=salt_of('%s%d-%d' % (tenant, index, self.round))))
        made.settle()
        return request

    def push_all_the_way(self, made, request, limit=12):
        """push_out, then more traffic until every block of the session's first 64 is on the drive (what is in host RAM when the engine stops is lost)."""
        self.push_out(made, request)
        for more in range(limit):
            if made.nvme_has_session(request, 64).count(True) == 64:
                return request
            self.push_out(made, first_turn('x%d-%s' % (more, request.request_id), 'xt%d' % more, seed=300 + more), fillers=3)
        assert made.nvme_has_session(request, 64).count(True) == 64, 'the session did not reach the drive'
        return request

    def returning(self, request, extra=700, name='again'):
        return Request('%s-%s' % (name, request.request_id), request.all_token_ids + span(extra, 3), salt=request.cache_salt)


def first_turn(name='a1', tenant='alpha', tokens=4200, seed=1):
    return Request(name, span(tokens, seed), salt=salt_of(tenant))


ACKED = {'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': '/dev/nvme1n1', 'QWEN_PREFIX_NVME_RETENTION_ACK': '1'}


class FlagTests(unittest.TestCase):
    def test_nothing_set_is_off_and_not_a_problem(self):
        self.assertEqual(kvnvme.parse({}), (None, []))
        self.assertEqual(kvnvme.parse({'QWEN_PREFIX_NVME': '0'}), (None, []))
        self.assertEqual(kvnvme.flag_problems({}), [])

    def test_defaults(self):
        plan, problems = kvnvme.parse(ACKED)
        self.assertEqual(problems, [])
        self.assertEqual((plan.spill, plan.through_blocks, plan.through_idle_s, plan.allow_zero_retention), ('through', 32, 0, False))
        self.assertEqual((plan.device, plan.ttl_s, plan.max_age_s, plan.max_bytes, plan.verify, plan.restore, plan.free_floor_bytes),
                         ('/dev/nvme1n1', 86400, 0, 0, 'sample', 'host', None))
        self.assertEqual((plan.queue_bytes, plan.threads, plan.format, plan.discard, plan.audit, plan.cmd_dir), (2 << 30, 4, 'auto', False, False, None))

    def test_every_value_is_strict(self):
        base = {'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': '/dev/x', 'QWEN_PREFIX_NVME_RETENTION_ACK': '1'}
        bad = [
            ('QWEN_PREFIX_NVME_SPILL', 'both'), ('QWEN_PREFIX_NVME_RETENTION_ACK', '2'), ('QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION', 'yes'),
            ('QWEN_PREFIX_NVME', '2'), ('QWEN_PREFIX_NVME', 'true'), ('QWEN_PREFIX_NVME_DEVICE', 'nvme1n1'), ('QWEN_PREFIX_NVME_DEVICE', '/dev/x y'),
            ('QWEN_PREFIX_NVME_DEVICE', '-/dev/x'), ('QWEN_PREFIX_NVME_TTL_S', '0'), ('QWEN_PREFIX_NVME_TTL_S', '-5'), ('QWEN_PREFIX_NVME_TTL_S', '1e5'),
            ('QWEN_PREFIX_NVME_TTL_S', ' 5'), ('QWEN_PREFIX_NVME_TTL_S', '99999999999'), ('QWEN_PREFIX_NVME_MAX_AGE_S', 'x'), ('QWEN_PREFIX_NVME_MAX_BYTES', '12.5'),
            ('QWEN_PREFIX_NVME_MAX_BYTES', '4096'), ('QWEN_PREFIX_NVME_VERIFY', 'fast'), ('QWEN_PREFIX_NVME_RESTORE', 'dma'), ('QWEN_PREFIX_NVME_FREE_FLOOR_BYTES', '-1'),
            ('QWEN_PREFIX_NVME_QUEUE_BYTES', '1024'), ('QWEN_PREFIX_NVME_THREADS', '0'), ('QWEN_PREFIX_NVME_THREADS', '65'), ('QWEN_PREFIX_NVME_FORMAT', 'yes'),
            ('QWEN_PREFIX_NVME_DISCARD', '2'), ('QWEN_PREFIX_NVME_AUDIT', 'on'), ('QWEN_PREFIX_NVME_COMPAT', 'a b'), ('QWEN_PREFIX_NVME_COMPAT', 'x' * 65),
            ('QWEN_PREFIX_NVME_DRAIN_S', '9'), ('QWEN_PREFIX_NVME_DRAIN_S', '-1'), ('QWEN_PREFIX_NVME_DRAIN_S', '2.5'),
            ('QWEN_PREFIX_NVME_CMD_DIR', 'relative/dir')]
        for name, value in bad:
            plan, problems = kvnvme.parse(dict(base, **{name: value}))
            self.assertTrue(problems and plan is None, '%s=%r should be refused' % (name, value))
            self.assertTrue(any(name in problem for problem in problems), (name, value, problems))
        good = dict(base, QWEN_PREFIX_NVME_TTL_S='3600', QWEN_PREFIX_NVME_MAX_AGE_S='86400', QWEN_PREFIX_NVME_MAX_BYTES='1073741824', QWEN_PREFIX_NVME_VERIFY='all',
                    QWEN_PREFIX_NVME_FREE_FLOOR_BYTES='0', QWEN_PREFIX_NVME_QUEUE_BYTES=str(1 << 30), QWEN_PREFIX_NVME_THREADS='8', QWEN_PREFIX_NVME_FORMAT='1',
                    QWEN_PREFIX_NVME_DISCARD='1', QWEN_PREFIX_NVME_AUDIT='1', QWEN_PREFIX_NVME_COMPAT='abc-1.2_x', QWEN_PREFIX_NVME_CMD_DIR='/models/.qwen-c2/nvme', QWEN_PREFIX_NVME_SPILL='through', QWEN_PREFIX_NVME_THROUGH_BLOCKS='64',
                    QWEN_PREFIX_NVME_THROUGH_IDLE_S='0', QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION='1', QWEN_PREFIX_NVME_DRAIN_S='8')
        plan, problems = kvnvme.parse(good)
        self.assertEqual(problems, [])
        self.assertEqual((plan.ttl_s, plan.max_age_s, plan.max_bytes, plan.free_floor_bytes, plan.format, plan.discard, plan.audit),
                         (3600, 86400, 1 << 30, 0, 'force', True, True))

    def test_p2p_is_refused_until_it_is_built(self):
        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_RESTORE='p2p'))
        self.assertIsNone(plan)
        self.assertTrue(any('not built' in problem for problem in problems), problems)

    def test_a_name_without_the_switch_would_be_inert_and_is_refused(self):
        for env in ({'QWEN_PREFIX_NVME_DEVICE': '/dev/x'}, {'QWEN_PREFIX_NVME': '0', 'QWEN_PREFIX_NVME_TTL_S': '60'}, {'QWEN_PREFIX_NVME_AUDIT': '1'}):
            plan, problems = kvnvme.parse(env)
            self.assertIsNone(plan)
            self.assertTrue(any('inert' in problem for problem in problems), (env, problems))

    def test_the_switch_needs_a_device(self):
        plan, problems = kvnvme.parse({'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_RETENTION_ACK': '1'})
        self.assertTrue(any('QWEN_PREFIX_NVME_DEVICE' in problem for problem in problems), problems)

    def test_the_retention_acknowledgement_is_required_and_says_why(self):
        plan, problems = kvnvme.parse({'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': '/dev/x'})
        self.assertIsNone(plan)
        self.assertEqual(len(problems), 1)
        for word in ('QWEN_PREFIX_NVME_RETENTION_ACK=1', 'zero-retention', 'cannot tell what its node advertises', '2026-10-21'):
            self.assertIn(word, problems[0])
        with self.assertRaisesRegex(ValueError, 'QWEN_PREFIX_NVME_RETENTION_ACK'):
            PrefixRegistry(budget_bytes=1 << 20, environ={'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': '/dev/x', 'QWEN_PREFIX_HOST_TIER_GIB': '1'})

    def test_write_through_is_the_default_and_idle_zero_means_at_once(self):
        plan, problems = kvnvme.parse(ACKED)
        self.assertEqual((problems, plan.spill, plan.through, plan.through_blocks, plan.through_idle_s), ([], 'through', True, 32, 0))
        self.assertEqual(plan.drain_s, 4)
        self.assertEqual((kvnvme.DEFAULT_SPILL, kvnvme.DEFAULT_THROUGH_IDLE_S), ('through', 0))

    def test_the_through_knobs_are_valid_by_default_and_refused_beside_the_evict_override(self):
        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_THROUGH_BLOCKS='8'))
        self.assertEqual((problems, plan.through, plan.through_blocks), ([], True, 8))
        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_THROUGH_IDLE_S='30'))
        self.assertEqual((problems, plan.through_idle_s), ([], 30))
        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_SPILL='through', QWEN_PREFIX_NVME_THROUGH_BLOCKS='8'))
        self.assertEqual((problems, plan.through, plan.through_blocks), ([], True, 8))
        for knob in ('QWEN_PREFIX_NVME_THROUGH_BLOCKS', 'QWEN_PREFIX_NVME_THROUGH_IDLE_S'):
            plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_SPILL='evict', **{knob: '8'}))
            self.assertIsNone(plan)
            self.assertTrue(any(knob in problem and 'evict mode would ignore it' in problem for problem in problems), problems)
        plan, problems = kvnvme.parse(dict(ACKED, QWEN_PREFIX_NVME_SPILL='evict'))
        self.assertEqual((problems, plan.spill, plan.through), ([], 'evict', False))

    def test_the_registry_refuses_to_start_on_a_bad_value_or_without_the_host_tier(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_PREFIX_NVME_TTL_S'):
            PrefixRegistry(budget_bytes=1 << 20, environ=dict(ACKED, QWEN_PREFIX_NVME_TTL_S='0', QWEN_PREFIX_HOST_TIER_GIB='1'))
        with self.assertRaisesRegex(ValueError, 'host KV tier'):
            PrefixRegistry(budget_bytes=1 << 20, environ=dict(ACKED))
        with self.assertRaisesRegex(ValueError, 'inert'):
            PrefixRegistry(budget_bytes=1 << 20, environ={'QWEN_PREFIX_NVME_TTL_S': '600', 'QWEN_PREFIX_HOST_TIER_GIB': '1'})

    def test_the_names_the_registry_and_the_contract_know_are_the_modules(self):
        self.assertEqual(registry_module.ENV_NVME_NAMES, kvnvme.NAMES)
        self.assertTrue(set(kvnvme.GATE_ONLY) <= set(kvnvme.NAMES))

    def test_the_counter_table_has_no_duplicates_and_names_are_metric_safe(self):
        names = list(kvnvme.COUNTER_NAMES) + list(kvnvme.GAUGE_NAMES)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith('nvme_') for name in names))


class FingerprintTests(unittest.TestCase):
    def plan(self, **env):
        plan, problems = kvnvme.parse(dict(ACKED, **env))
        self.assertEqual(problems, [])
        return plan

    def fingerprint(self, io=None, environ=None, **plan_env):
        return kvnvme.layout_fingerprint(io or FakeKvIO(), self.plan(**plan_env), dict(environ or {}))

    def test_the_fingerprint_names_what_makes_a_block_mean_something_else(self):
        base, components = self.fingerprint()
        self.assertEqual(len(base), 8)
        self.assertEqual(list(components), ['layout', 'epoch', 'model', 'drafter', 'kv', 'block_bytes', 'page', 'chunk', 'mesh', 'gdn_bf16', 'ckpt_format', 'env', 'compat'])
        self.assertEqual(self.fingerprint()[0], base, 'deterministic')
        changed = {
            'model repo': self.fingerprint(environ={'QWEN_C2_MODEL_REPO': 'org/model', 'QWEN_C2_MODEL_REVISION': 'a' * 40})[0],
            'model revision': self.fingerprint(environ={'QWEN_C2_MODEL_REPO': 'org/model', 'QWEN_C2_MODEL_REVISION': 'b' * 40})[0],
            'compat token': self.fingerprint(QWEN_PREFIX_NVME_COMPAT='rev2')[0],
            'another compat token': self.fingerprint(QWEN_PREFIX_NVME_COMPAT='rev3')[0],
            'gdn dtype': self.fingerprint(environ={'QWEN35_GDN_STATE_BF16': '1'})[0],
            'a byte-defining switch': self.fingerprint(environ={'QWEN_SDPA_BF8': '1'})[0],
            'drafter': self.fingerprint(environ={'QWEN_FAST_DRAFTER_CHECKPOINT': 'some-other-drafter'})[0],
        }
        io = FakeKvIO()
        io.fingerprint = 'another-kv-layout'
        changed['kv layout'] = self.fingerprint(io=io)[0]
        io2 = FakeKvIO()
        io2.block_bytes = 64
        changed['block bytes'] = self.fingerprint(io=io2)[0]
        io3 = FakeKvIO()
        io3.chips = 4
        changed['mesh'] = self.fingerprint(io=io3)[0]
        with mock.patch.object(kvnvme, 'KV_EPOCH', kvnvme.KV_EPOCH + 1):
            changed['epoch'] = self.fingerprint()[0]
        self.assertEqual(len(set(changed.values()) | {base}), len(changed) + 1, changed)

    def test_a_deploy_does_not_change_it_neither_the_profile_name_nor_the_image_build(self):
        base = self.fingerprint(environ={'QWEN_C2_PROFILE': 'one-profile'})[0]
        self.assertEqual(self.fingerprint(environ={'QWEN_C2_PROFILE': 'another-profile-name'})[0], base)
        self.assertEqual(self.fingerprint(environ={})[0], base)
        # the image's source revision file is not read any more (no code-version component exists)
        with mock.patch('builtins.open', side_effect=AssertionError('the fingerprint must not read a file')):
            self.assertEqual(self.fingerprint(environ={'QWEN_C2_PROFILE': 'x'})[0], base)
        self.assertFalse(hasattr(kvnvme, 'code_version') or hasattr(kvnvme, 'CODE_VERSION_FILE'))

    def test_the_drafter_is_the_pinned_table_entry_and_the_default_is_the_unset_flag(self):
        table = json.loads((Path(__file__).resolve().parent / 'drafter_checkpoints.json').read_text(encoding='utf-8'))
        default = table['default']
        identity = kvnvme.drafter_identity({})
        self.assertEqual(identity, '%s@%s' % (default, table['checkpoints'][default]['revision'][:12]))
        self.assertEqual(kvnvme.drafter_identity({'QWEN_FAST_DRAFTER_CHECKPOINT': default}), identity, 'naming the default is not another drafter')
        self.assertEqual(kvnvme.drafter_identity({'QWEN_FAST_DRAFTER_CHECKPOINT': 'not-in-the-table'}), 'not-in-the-table')
        with mock.patch.object(kvnvme.importlib, 'import_module', side_effect=ImportError('no table')):
            self.assertEqual(kvnvme.drafter_identity({}), 'default')
            self.assertEqual(kvnvme.drafter_identity({'QWEN_FAST_DRAFTER_CHECKPOINT': 'x-y'}), 'x-y')

    def test_the_epoch_is_a_plain_integer_a_kernel_author_can_bump(self):
        self.assertIs(type(kvnvme.KV_EPOCH), int)
        self.assertGreaterEqual(kvnvme.KV_EPOCH, 1)
        source = (Path(__file__).resolve().parent / 'qwen_kv_nvme.py').read_text(encoding='utf-8')
        self.assertIn('BUMP IT', source)
        self.assertIn('prefill NUMERICS', source)


class FakeTorch(object):
    """Just enough of torch for TorchOps: tensors that are bytes with a dtype name and a shape."""

    uint8 = 'uint8'
    bfloat16 = 'bfloat16'
    float32 = 'float32'

    class Tensor(object):
        def __init__(self, data, dtype, shape):
            self.data, self.dtype, self.shape = bytes(data), 'torch.' + dtype if not str(dtype).startswith('torch.') else dtype, tuple(shape)

        def detach(self):
            return self

        def contiguous(self):
            return self

        def reshape(self, *shape):
            shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
            if shape == (-1,):
                return FakeTorch.Tensor(self.data, self.dtype, (len(self.data),))
            return FakeTorch.Tensor(self.data, self.dtype, shape)

        def view(self, dtype):
            return FakeTorch.Tensor(self.data, dtype, (len(self.data),))

        def numpy(self):
            return bytearray(self.data)

        def __eq__(self, other):
            return isinstance(other, FakeTorch.Tensor) and (self.data, self.dtype, self.shape) == (other.data, other.dtype, other.shape)

        def __hash__(self):
            return hash((self.data, self.dtype, self.shape))

    @staticmethod
    def frombuffer(buffer, dtype):
        return FakeTorch.Tensor(bytes(buffer), dtype, (len(buffer),))


def fake_checkpoint(rec, carry, pos=4096, window=None, nbytes=77):
    checkpoint = types.SimpleNamespace(key=b'k' * 32, pos=pos, token_ids=array.array('q', range(pos)), rec=rec, carry=carry, nbytes=nbytes, window=window, chain=b'c' * 32,
                                       tenant='alpha')
    return checkpoint


class CodecTests(unittest.TestCase):
    def codec(self, *ops):
        return kvnvme.CheckpointCodec(list(ops) or [kvnvme.BytesOps(), kvnvme.TorchOps(FakeTorch)])

    def roundtrip(self, codec, checkpoint):
        buffers = codec.pack(checkpoint)
        data = b''.join(bytes(buffer) for buffer in buffers)
        return codec.unpack(data), data

    def test_bytes_items_round_trip(self):
        image, data = self.roundtrip(self.codec(), fake_checkpoint(b'state-bytes', b'carry'))
        self.assertEqual((image.pos, image.rec, image.carry, image.nbytes, image.window), (4096, b'state-bytes', b'carry', 77, None))
        self.assertEqual(list(image.token_ids), list(range(4096)))
        self.assertEqual(len(data) % 8, 0)

    def test_lists_of_tensors_round_trip_with_dtype_and_shape(self):
        rec = [FakeTorch.Tensor(bytes(range(48)), 'bfloat16', (2, 3, 4)), FakeTorch.Tensor(bytes(range(7)), 'float32', (7,))]
        carry = [FakeTorch.Tensor(b'\x01\x02\x03\x04', 'bfloat16', (2, 2))]
        image, _ = self.roundtrip(self.codec(), fake_checkpoint(rec, carry))
        self.assertEqual(image.rec, rec)
        self.assertEqual(image.carry, carry)
        self.assertEqual([tensor.shape for tensor in image.rec], [(2, 3, 4), (7,)])

    def test_a_carried_window_comes_back_and_one_the_adapters_cannot_hold_is_left_off(self):
        window = types.SimpleNamespace(start=2048, end=4096, payload=[FakeTorch.Tensor(bytes(range(16)), 'bfloat16', (4, 2))], nbytes=16, digest='sha-of-window')
        image, _ = self.roundtrip(self.codec(), fake_checkpoint(b'r', b'c', window=window))
        self.assertEqual((image.window['start'], image.window['end'], image.window['nbytes'], image.window['digest']), (2048, 4096, 16, 'sha-of-window'))
        self.assertEqual(image.window['payload'], window.payload)
        odd = types.SimpleNamespace(start=0, end=2048, payload=[object()], nbytes=5, digest=None)
        image, _ = self.roundtrip(self.codec(), fake_checkpoint(b'r', b'c', pos=2048, window=odd))
        self.assertIsNone(image.window)
        self.assertEqual(image.rec, b'r')

    def test_an_item_no_adapter_describes_is_a_type_error(self):
        with self.assertRaises(TypeError):
            self.codec().pack(fake_checkpoint(object(), b'c'))

    def test_garbage_is_a_value_error(self):
        codec = self.codec()
        for data in (b'', b'x' * 5, b'QCKP0002' + bytes(100), b'QCKP0001' + b'\xff\xff\xff\x7f' + bytes(20)):
            with self.assertRaises((ValueError, UnicodeDecodeError, json.JSONDecodeError)):
                codec.unpack(data)
        buffers = codec.pack(fake_checkpoint(b'r', b'c'))
        data = b''.join(bytes(buffer) for buffer in buffers)
        with self.assertRaises(ValueError):
            codec.unpack(data[:-16])

    def test_preconverted_items_are_packed_as_their_torch_values_and_converted_again_on_the_way_back(self):
        calls = []

        class Converted(object):
            padded_nbytes = 64

            def __init__(self, host, shape, dtype):
                self.host, self.shape, self.dtype = host, tuple(shape), dtype

        module = types.ModuleType('fake_model_module')
        module.torch = FakeTorch
        module.ttnn = types.SimpleNamespace(ConcatMeshToTensor=lambda mesh, dim: ('composer', mesh, dim), to_torch=lambda host, mesh_composer: host)
        sys.modules['fake_model_module'] = module
        self.addCleanup(sys.modules.pop, 'fake_model_module', None)

        class Model(object):
            mesh_device = 'mesh'

            def _qwen_prefix_preconvert(self, rec, carry):
                calls.append((len(rec), len(carry)))
                return [Converted(tensor, tensor.shape, tensor.dtype) for tensor in rec], [Converted(tensor, tensor.shape, tensor.dtype) for tensor in carry], 4242

        Model.__module__ = 'fake_model_module'
        model = Model()
        codec = self.codec(kvnvme.BytesOps(), kvnvme.ConvertedOps(model), kvnvme.TorchOps(FakeTorch))
        rec = [Converted(FakeTorch.Tensor(bytes(range(32)), 'bfloat16', (4, 4)), (4, 4), 'torch.bfloat16')]
        carry = [Converted(FakeTorch.Tensor(bytes(range(8)), 'bfloat16', (2, 2)), (2, 2), 'torch.bfloat16')]
        window = types.SimpleNamespace(start=0, end=4096, payload=[FakeTorch.Tensor(b'wwww', 'bfloat16', (2, 1))], nbytes=100, digest=None)
        image, data = self.roundtrip(codec, fake_checkpoint(rec, carry, window=window))
        self.assertEqual(calls, [(1, 1)], 'the model\'s own conversion ran once, on the stored values')
        self.assertTrue(isinstance(image.rec[0], Converted) and image.rec[0].host == rec[0].host)
        self.assertEqual(image.nbytes, 4242 + 100, 'the registry\'s charge: the converted size and the carried window')
        self.assertEqual(image.window['payload'][0], window.payload[0])

    def test_the_default_codec_takes_the_ios_own_or_builds_one_from_its_model(self):
        own = object()
        io = types.SimpleNamespace(checkpoint_codec=own)
        self.assertIs(kvnvme.default_codec(io), own)
        self.assertEqual([type(ops).__name__ for ops in kvnvme.default_codec(FakeKvIO()).ops], ['BytesOps', 'TorchOps'])
        io = types.SimpleNamespace(model=types.SimpleNamespace(_qwen_prefix_preconvert=lambda *a: None))
        self.assertEqual([type(ops).__name__ for ops in kvnvme.default_codec(io).ops], ['BytesOps', 'ConvertedOps', 'TorchOps'])


try:
    import torch as real_torch
    import qwen_prefix_model_fixture as F
    import test_qwen_prefix_model_runtime as M
except ImportError:          # (the stdlib-only lane: the rest of this file needs none of it)
    real_torch = None


@unittest.skipUnless(real_torch is not None, 'needs torch (the CPU suite has it)')
class RealModelCodecTests(unittest.TestCase if real_torch is None else M.ExactTestBase):
    """The codec against the REAL staged model source (qwen_prefix_model_patch's adapter and methods, on the model fixture's recording ttnn): a checkpoint that went through
    pack and unpack restores the same state a cold prefill computes, byte for byte, whether the store holds torch tensors or the model's preconverted host tensors."""

    def checkpoint(self, environ):
        engine = M.Engine(traced=False, environ=environ)
        engine.warm()
        base = F.prompt(12000, seed=91)
        conv = {'id': 'codec'}
        engine.turn(conv, base[:5000], 0, [4096])
        entry = engine.registry.get(F.key_at(base[:5000].tolist(), 4096))
        self.assertIsNotNone(entry)
        return engine, base, conv, entry

    def codec_for(self, engine):
        return kvnvme.default_codec(types.SimpleNamespace(model=engine.model))

    def test_preconverted_checkpoints_survive_pack_and_unpack_and_restore_exactly(self):
        engine, base, conv, entry = self.checkpoint(M.PreconvertedCheckpoints.ENV)
        self.assertTrue(all(hasattr(item, 'host') for item in list(entry.rec) + list(entry.carry)), 'the store holds preconverted tensors')
        codec = self.codec_for(engine)
        self.assertEqual([type(ops).__name__ for ops in codec.ops], ['BytesOps', 'ConvertedOps', 'TorchOps'])
        data = b''.join(bytes(buffer) for buffer in codec.pack(entry))
        image = codec.unpack(data)
        self.assertEqual((image.pos, list(image.token_ids) == list(entry.token_ids), image.nbytes), (entry.pos, True, entry.nbytes))
        self.assertEqual(len(image.rec), len(entry.rec))
        for original, back in zip(list(entry.rec) + list(entry.carry), list(image.rec) + list(image.carry)):
            self.assertTrue(torch_equal(original.host.data, back.host.data), 'the host tensor that comes back is the one that was stored')
            self.assertEqual((back.shape, back.dtype, back.padded_nbytes), (original.shape, original.dtype, original.padded_nbytes))
        entry.rec, entry.carry = image.rec, image.carry              # the store now holds what came off the drive
        self.run_turn_and_cold(engine, conv, base[:6500], 4096, [6144])
        self.assertEqual(engine.registry.stats['restores'], 1)

    def test_torch_checkpoints_survive_pack_and_unpack_and_restore_exactly(self):
        engine, base, conv, entry = self.checkpoint(M.ON)
        self.assertTrue(all(isinstance(item, real_torch.Tensor) for item in list(entry.rec) + list(entry.carry)))
        codec = self.codec_for(engine)
        data = b''.join(bytes(buffer) for buffer in codec.pack(entry))
        image = codec.unpack(data)
        for original, back in zip(list(entry.rec) + list(entry.carry), list(image.rec) + list(image.carry)):
            self.assertEqual((back.dtype, tuple(back.shape)), (original.dtype, tuple(original.shape)))
            self.assertTrue(real_torch.equal(original, back))
        entry.rec, entry.carry = image.rec, image.carry
        self.run_turn_and_cold(engine, conv, base[:6500], 4096, [6144])


def torch_equal(a, b):
    return bool(real_torch.equal(a, b))


class BackendTests(unittest.TestCase):
    def test_host_is_built_and_p2p_raises_not_built_with_its_contract_documented(self):
        self.assertEqual((kvnvme.make_backend('host').name, kvnvme.make_backend('host').direct), ('host', False))
        with self.assertRaises(kvnvme.NotBuilt):
            kvnvme.make_backend('p2p')
        with self.assertRaises(kvnvme.NotBuilt):
            kvnvme.P2PBackend()
        with self.assertRaises(ValueError):
            kvnvme.make_backend('dma')
        doc = kvnvme.P2PBackend.__doc__
        for word in ('3 GiB', '1 GiB', 'ALIGNMENT', 'SCATTER', 'INTEGRITY', 'ENCRYPTION', '6.95 GB/s'):
            self.assertIn(word.lower(), doc.lower())
        self.assertEqual((kvnvme.P2PBackend.STAGING_WINDOW_MAX_BYTES, kvnvme.P2PBackend.STAGING_WINDOW_OFFSET), (3 << 30, 1 << 30))
        self.assertEqual(kvnvme.P2PBackend.window_blocks(2228224), 1445)
        self.assertTrue(kvnvme.P2PBackend.direct)

    def test_the_interface_is_two_methods(self):
        backend = kvnvme.RestoreBackend()
        with self.assertRaises(NotImplementedError):
            backend.read_to_host(None, [], [])
        with self.assertRaises(NotImplementedError):
            backend.restore_to_device(None, [], [], None)


class FlagOffTests(NvmeCase):
    """The NVMe names unset: nothing of the tier exists, and the host tier is what tp4/next-1 made it."""

    def traffic(self, seed, tier):
        rng = random.Random(seed)
        made = Rig(num_blocks=170, budget=1 << 20, tier=tier)
        try:
            systems = {name: span(2300, 500 + index) for index, name in enumerate(('alpha', 'bravo', 'charlie'))}
            histories = {}
            for step in range(36):
                name = rng.choice(['alpha', 'bravo', 'charlie'])
                convo = rng.randrange(2)
                key = (name, convo)
                base = histories.get(key, systems[name])
                tokens = list(base) + span(rng.randrange(1200, 3200), 4000 + 31 * step + convo)
                made.serve(Request('r%d' % step, tokens, salt=salt_of(name)))
                made.check_kv()
                histories[key] = None if len(tokens) > 7000 else tokens + span(rng.randrange(50, 400), 9000 + step)
                if histories[key] is None:
                    del histories[key]
            registry = made.registry
            return dict(
                rows=[None if row is None else [row[0], row[1].q if row[1] else 0, row[1].h if row[1] else 0, row[1].capture_positions() if row[1] else []]
                      for row in made.rows],
                entries=[[key.hex(), entry.pos, entry.nbytes] for key, entry in registry.entries.items()], bytes=registry.bytes,
                stats=sorted((key, value) for key, value in registry.stats.items() if not key.endswith('_ms')),
                tier=[[key.hex(), record.index, record.tenant, hashlib.sha256(bytes(record.payload)).hexdigest()] for key, record in registry.tier.records.items()],
                tier_stats=sorted(registry.tier.stats.items()), reads=made.kv.reads, writes=made.kv.writes,
                free=len(made.scheduler.kv_cache_manager.block_pool.free), snapshot=sorted(registry.snapshot()))
        finally:
            made.close()

    # The sha256 of everything observable of 12 runs (4 seeds x host tiers of 400, 60 and 1 blocks) as produced by tp4/next-1 (758264ac), before this tier existed.
    GOLDEN_SHA256 = 'c3317b64135c41015796f47075082b70e483baab2cec8c1f9ad8e4bb872b4f09'

    def test_the_host_tier_with_the_nvme_names_unset_is_the_host_tier_that_was(self):
        runs = {'%d-%d' % (seed, tier): self.traffic(seed, tier) for seed in range(4) for tier in (400, 60, 1)}
        text = json.dumps(runs, sort_keys=True, separators=(',', ':'))
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), self.GOLDEN_SHA256)
        totals = {name: sum(dict(run['stats'])[name] for run in runs.values()) for name in ('tier_spill_blocks', 'tier_restore_blocks', 'tier_evicted', 'grants')}
        self.assertTrue(all(totals.values()), 'the scenarios spill, evict, restore and grant: %s' % totals)

    def test_nothing_of_the_nvme_tier_exists_when_it_is_off(self):
        made = Rig(num_blocks=SMALL_POOL, budget=1 << 20, tier=64)
        self.addCleanup(made.close)
        registry = made.registry
        self.assertIsNone(registry.nvme_plan)
        self.assertIsNone(registry.nvme)
        self.assertIsNone(registry.tier.spill_sink)
        self.assertEqual((registry.tier._spare, registry.tier._cap_slots), (0, 64))
        self.assertEqual(len(registry.tier._free), 64, 'the slab is mapped for the cap and not a slot more')
        self.assertIsNone(made.graft.tier_hooks.nvme)
        self.assertFalse(any(name.startswith('nvme_') for name in registry.snapshot()))
        self.assertFalse(any(line.startswith('nvme tier') for line in made.lines))
        deadline = time.time() + 10          # (another test's writer may be a moment from exiting)
        while any(thread.name == 'qwen-nvme-writer' for thread in threading.enumerate()) and time.time() < deadline:
            time.sleep(0.01)
        self.assertFalse(any(thread.name == 'qwen-nvme-writer' for thread in threading.enumerate()))

    def test_without_the_host_tier_the_nvme_names_refuse_to_start(self):
        with self.assertRaisesRegex(ValueError, 'host KV tier'):
            PrefixRegistry(budget_bytes=1 << 20, environ=nvme_environ('/dev/x'))


class SpillAndRestoreTests(NvmeCase):
    def test_an_evicted_session_returns_from_the_nvme_exactly(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        self.assertIn(True, made.nvme_has_session(a), 'a session left the host tier for the NVMe')
        self.assertEqual(made.session_somewhere(a), 64, 'every block up to the boundary is in the host tier or on the NVMe')
        on_nvme = sum(1 for key in a.block_hashes[:64] if not made.registry.tier.has(key))
        self.assertGreater(on_nvme, 0)
        start, grant = made.serve(self.returning(a))
        self.assertEqual((start, grant.q), (4096, 4096), 'the whole conversation up to its last boundary is a hit again')
        stats = made.registry.snapshot()
        self.assertEqual((stats['nvme_restore_requests'], stats['nvme_restore_blocks'], stats['nvme_ckpt_restored']), (1, on_nvme, 1), 'the host tier\'s own records are not read twice')
        self.assertEqual((stats['nvme_digest_failures'], stats['nvme_restore_failures'], stats['nvme_miss_blocks']), (0, 0, 0))
        self.assertGreater(stats['nvme_digest_checks'], 0)
        self.assertEqual(stats['tier_restore_requests'], 1, 'and then the host tier restores it to the device, as it always did')
        self.assertTrue(made.check_kv())
        self.assertEqual(made.registry.tier.pinned, set(), 'a restore leaves no pin behind')
        self.assertTrue(any(line.startswith('nvme tier restore req=again-a1 blocks=') and 'backend=host verify=all' in line for line in made.lines), made.lines[-8:])
        self.assertTrue(any(line.startswith('nvme tier checkpoint restored pos=4096') for line in made.lines))
        self.assertTrue(any(line.startswith('nvme tier spill blocks=') for line in made.lines))
        self.assertTrue(any(line.startswith('nvme tier open device=') for line in made.lines))
        self.assertTrue(any(line.startswith('nvme tier on backend=host') for line in made.lines))

    def test_a_restore_the_pool_has_no_room_for_is_a_miss_that_reads_and_pins_nothing(self):
        made = self.rig(num_blocks=100)
        a = self.push_out(made, first_turn())
        start, grant = made.serve(Request('long', a.all_token_ids + span(2200, 3), salt=a.cache_salt))
        stats = made.registry.snapshot()
        self.assertEqual(start, 0)
        self.assertEqual((stats['tier_restore_refused_room'], stats['nvme_restore_blocks']), (1, 0))
        self.assertEqual(made.registry.tier.pinned, set())
        self.assertTrue(made.check_kv())

    def test_without_the_nvme_the_same_traffic_is_a_miss(self):
        plain = Rig(num_blocks=SMALL_POOL, budget=1 << 20, tier=200)
        self.addCleanup(plain.close)
        a = first_turn()
        plain.serve(a)
        for index in range(5):
            plain.serve(Request('fill-%d' % index, span(6000, 20 + index), salt=salt_of('bravo%d' % index)))
        start, grant = plain.serve(self.returning(a))
        self.assertEqual(start, 0)

    def test_the_slab_slots_the_writer_held_come_back_and_the_host_tier_keeps_its_cap(self):
        made = self.rig()
        tier = made.registry.tier
        self.assertEqual((tier._cap_slots, tier._spare, len(tier._free)), (self.TIER, 524288, self.TIER + 524288))
        self.push_out(made, first_turn())
        self.assertEqual(len(tier.records), self.TIER)
        self.assertEqual(tier._taken, self.TIER)
        self.assertEqual(len(tier._free), 524288, 'every retained slot was given back')
        self.assertEqual(made.nvme.queue_kv_bytes + made.nvme.queue_ckpt_bytes, 0)
        self.assertTrue(made.check_kv())

    def test_the_scheduler_never_waits_for_the_drive(self):
        made = self.rig()
        store = made.nvme.store
        started = time.time()
        with store._wlock:                         # the writer is stuck in a long device write
            a = self.push_out.__func__(self, made, first_turn(), fillers=0) if False else first_turn()
            made.serve(a)
            for index in range(5):
                made.serve(Request('fill-%d' % index, span(6000, 20 + index), salt=salt_of('bravo%d' % index)))
            queued = made.nvme.queue_kv_bytes
            self.assertGreater(queued, 0, 'records are waiting for the writer')
            self.assertLess(time.time() - started, 20)
            self.assertEqual(made.registry.snapshot()['nvme_spill_blocks'], 0)
            # the retained slots are held, and the host tier still serves and spills
            self.assertLess(len(made.registry.tier._free), 524288)
            self.assertTrue(made.check_kv())
        made.settle()
        self.assertGreater(made.registry.snapshot()['nvme_spill_blocks'], 0)

    def test_a_full_queue_drops_the_record_as_the_host_tier_always_did(self):
        made = self.rig()
        made.nvme.plan.queue_bytes = 1000          # (31 blocks; the shared budget also holds the checkpoints, which then do not fit either)
        with made.nvme.store._wlock:
            a = first_turn()
            made.serve(a)
            for index in range(5):
                made.serve(Request('fill-%d' % index, span(6000, 20 + index), salt=salt_of('bravo%d' % index)))
        made.settle()
        stats = made.registry.snapshot()
        self.assertGreater(stats['nvme_spill_dropped_queue'], 0)
        self.assertEqual(stats['nvme_spill_failures'], 0)
        start, grant = made.serve(self.returning(a))
        self.assertTrue(made.check_kv())
        self.assertIn(start, (0, 4096), 'a miss or an exact hit; never anything else')

    def test_blocks_no_checkpoint_reaches_are_not_written_to_the_drive(self):
        made = self.rig(tier_env={'QWEN_PREFIX_HOST_TIER_MIN_TOKENS': '8192'})
        self.push_out(made, first_turn())
        stats = made.registry.snapshot()
        self.assertEqual(stats['nvme_spill_blocks'], 0)
        self.assertEqual(stats['nvme_entries'], 0)

    def test_a_known_block_is_not_written_twice(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        written = made.registry.snapshot()['nvme_spill_blocks']
        made.serve(self.returning(a))                    # restored through the host tier, then evicted again by more traffic
        for index in range(5):
            made.serve(Request('more-%d' % index, span(6000, 40 + index), salt=salt_of('charlie%d' % index)))
        made.settle()
        stats = made.registry.snapshot()
        self.assertGreater(stats['nvme_spill_known'], 0, 'the restored blocks were on the drive already')
        self.assertGreaterEqual(stats['nvme_spill_blocks'], written)

    def test_a_restart_restores_what_the_last_run_spilled(self):
        made = self.rig()
        a = self.push_all_the_way(made, first_turn())
        entries = made.registry.snapshot()['nvme_entries']
        made.close()
        again = self.rig()
        stats = again.registry.snapshot()
        self.assertEqual(stats['nvme_entries'], entries, 'the index came back from the snapshot and the journal')
        self.assertEqual(stats['nvme_recovered_entries'], entries)
        self.assertEqual(len(again.registry.entries), 0, 'nothing in RAM: this is a new process')
        start, grant = again.serve(self.returning(a))
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertTrue(again.check_kv())

    def test_the_checkpoint_comes_back_with_its_token_ids_and_is_checked_like_any_other(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        self.assertEqual(len(made.registry.entries) , 1 + 0 if False else len(made.registry.entries))
        again = Request('again', a.all_token_ids[:3000] + span(1500, 77), salt=a.cache_salt)         # diverges inside the last boundary: the stored one must not be used
        start, grant = made.serve(again)
        self.assertLessEqual(start, 2048)
        self.assertTrue(made.check_kv())

    def test_a_checkpoint_the_store_dropped_for_its_budget_survives_on_the_nvme(self):
        made = self.rig(budget=3)
        for index in range(6):
            made.serve(Request('s%d' % index, span(4200, 100 + index), salt=salt_of('tenant%d' % index)))
        made.settle()
        stats = made.registry.snapshot()
        self.assertGreater(stats['evicted_lru'], 0, 'the 3-checkpoint budget evicted some')
        self.assertGreater(stats['nvme_ckpt_spilled'], 0, 'and the NVMe kept them')


class ExpiryTests(NvmeCase):
    def clocked(self, made, clock):
        made.nvme.clock = clock
        made.nvme.store.clock = clock

    def test_an_entry_past_its_ttl_is_a_miss_and_the_turn_is_served_cold_and_exactly(self):
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_TTL_S': '3600'})
        clock = types.SimpleNamespace(now=time.time())
        self.clocked(made, lambda: clock.now)
        a = self.push_out(made, first_turn())
        clock.now += 3000
        self.assertIn(True, made.nvme_has_session(a))
        clock.now += 700
        self.assertNotIn(True, made.nvme_has_session(a), 'expired')
        start, grant = made.serve(self.returning(a))
        self.assertEqual(start, 0)
        self.assertTrue(made.check_kv())

    def test_a_restore_restarts_the_clock_and_the_sweep_removes_the_expired(self):
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_TTL_S': '3600'})
        clock = types.SimpleNamespace(now=time.time())
        self.clocked(made, lambda: clock.now)
        a = self.push_out(made, first_turn())
        b = self.push_out(made, first_turn('b1', 'beta', seed=9))
        clock.now += 3000
        made.serve(self.returning(a))                            # a's blocks are used; b's are not
        clock.now += 1000
        removed = made.nvme.store.sweep()
        self.assertGreater(removed, 0)
        self.assertIsNone(made.nvme.store.lookup(store_module.KV, b.block_hashes[0]), 'b was not used for 4000 s: expired')
        self.assertIsNotNone(made.nvme.store.lookup(store_module.KV, a.block_hashes[0]), 'a was used 1000 s ago')
        made.nvme.poll()
        stats = made.registry.snapshot()
        self.assertGreater(stats['nvme_expired_ttl'], 0)

    def test_the_poll_sweeps_on_the_writer_thread(self):
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_TTL_S': '3600'})
        clock = types.SimpleNamespace(now=time.time())
        self.clocked(made, lambda: clock.now)
        made.nvme.last_sweep = clock.now
        a = self.push_out(made, first_turn())
        clock.now += 4000
        made.nvme.poll()
        made.settle()
        deadline = time.time() + 10
        while made.nvme.store.snapshot_stats()['entries'] and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(made.nvme.store.snapshot_stats()['entries'], 0)

    def test_the_age_cap_expires_a_session_that_is_used_every_day(self):
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_TTL_S': '100000', 'QWEN_PREFIX_NVME_MAX_AGE_S': '5000'})
        clock = types.SimpleNamespace(now=time.time())
        self.clocked(made, lambda: clock.now)
        a = self.push_out(made, first_turn())
        clock.now += 4000
        made.nvme.store.touch(store_module.KV, a.block_hashes[0])
        clock.now += 2000
        self.assertIsNone(made.nvme.store.lookup(store_module.KV, a.block_hashes[0]))


class CapacityTests(NvmeCase):
    def test_a_tiny_device_evicts_the_least_recently_used_session_and_serves_on(self):
        with open(self.device, 'wb') as handle:
            handle.truncate(2 << 20)
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_FREE_FLOOR_BYTES': '0'}, snap_bytes=16 * ALIGN, journal_bytes=16 * ALIGN)
        made.nvme.store.free_floor_bytes = 0
        sessions = []
        for index in range(6):
            sessions.append(self.push_out(made, first_turn('s%d' % index, 'tenant%d' % index, seed=50 + index), fillers=5))
        stats = made.registry.snapshot()
        self.assertGreater(stats['nvme_evicted_lru'], 0, 'the device filled and evicted')
        self.assertLessEqual(stats['nvme_bytes'], stats['nvme_capacity_bytes'])
        self.assertEqual(stats['nvme_spill_failures'], 0)
        self.assertGreater(stats['nvme_entries'], 0)
        start, grant = made.serve(self.returning(sessions[0]))                # the oldest is gone: cold, exact
        self.assertTrue(made.check_kv())
        self.assertIn(start, (0, 4096))

    def test_a_device_too_small_for_one_object_refuses_it_without_harm(self):
        with open(self.device, 'wb') as handle:
            handle.truncate(1 << 20)
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_MAX_BYTES': str(1 << 20)}, snap_bytes=16 * ALIGN, journal_bytes=16 * ALIGN)
        made.nvme.store.max_bytes = 2 * ALIGN
        self.push_out(made, first_turn(), fillers=5)
        stats = made.registry.snapshot()
        self.assertGreater(stats['nvme_spill_dropped_full'] + stats['nvme_evicted_lru'], 0)
        self.assertLessEqual(stats['nvme_bytes'], 2 * ALIGN)
        self.assertTrue(made.check_kv())


class IntegrityTests(NvmeCase):
    def corrupt(self, made, request, index):
        entry = made.nvme.store.lookup(store_module.KV, request.block_hashes[index])
        with open(self.device, 'r+b') as handle:
            handle.seek(entry.offset)
            data = bytearray(handle.read(ALIGN))
            data[0] ^= 0xFF
            handle.seek(entry.offset)
            handle.write(bytes(data))

    def test_a_block_that_changed_on_the_drive_is_refused_dropped_and_the_turn_is_cold(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        self.corrupt(made, a, 10)
        start, grant = made.serve(self.returning(a))
        stats = made.registry.snapshot()
        self.assertEqual(start, 0, 'a corrupt block is a miss, and the turn prefills cold (the model verifies the answer against the cold chain)')
        self.assertEqual((stats['nvme_digest_failures'], stats['nvme_restore_refused_digest']), (1, 1))
        self.assertIsNone(made.nvme.store.lookup(store_module.KV, a.block_hashes[10]), 'dropped')
        made.settle()
        made.nvme.flush()
        self.assertTrue(made.check_kv())
        # the drop is durable: a restart does not offer it again
        made.close()
        again = self.rig()
        self.assertIsNone(again.nvme.store.lookup(store_module.KV, a.block_hashes[10]))

    def test_sampling_checks_one_block_in_sixteen_and_all_checks_every_one(self):
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_VERIFY': 'sample'})
        a = self.push_out(made, first_turn())
        self.corrupt(made, a, 5)                        # not a sampled index (0, 16, 32, 48): a sampled restore does not see it
        start, grant = made.serve(self.returning(a))
        stats = made.registry.snapshot()
        self.assertEqual(stats['nvme_digest_checks'], 1 + 4 , '4 sampled blocks of 64 and the checkpoint')
        self.assertEqual(stats['nvme_digest_failures'], 0)
        self.assertEqual(start, 4096)

    def test_verify_off_checks_nothing(self):
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_VERIFY': 'off'})
        a = self.push_out(made, first_turn())
        made.serve(self.returning(a))
        self.assertEqual(made.registry.snapshot()['nvme_digest_checks'], 0)

    def test_a_checkpoint_that_changed_on_the_drive_is_dropped_and_the_turn_is_cold(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        entry = made.nvme.store.lookup(store_module.CKPT, a.block_hashes[63])
        with open(self.device, 'r+b') as handle:
            handle.seek(entry.offset + 40)
            handle.write(b'\x00\x01\x02')
        start, grant = made.serve(self.returning(a))
        stats = made.registry.snapshot()
        self.assertEqual(start, 0)
        self.assertEqual((stats['nvme_ckpt_restore_failures'], stats['nvme_ckpt_restored']), (1, 0))
        self.assertTrue(made.check_kv())

    def test_a_device_read_that_fails_is_a_miss_and_three_in_a_row_latch_the_tier(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        made.nvme.store.bf.fail = lambda what, offset, length: (_ for _ in ()).throw(IOError('dead')) if what == 'read' else None
        start, grant = made.serve(self.returning(a))
        stats = made.registry.snapshot()
        self.assertEqual(start, 0, 'the device would not answer: a cold turn, and the model checks it against the cold chain')
        self.assertGreater(stats['nvme_ckpt_restore_failures'], 0)
        self.assertTrue(made.check_kv())
        self.assertTrue(made.nvme.active, 'one failure is not a latch')
        entry = made.nvme.store.lookup(store_module.CKPT, a.block_hashes[63])
        for _ in range(3):
            made.nvme.load_checkpoint(entry)
        self.assertEqual(made.registry.snapshot()['nvme_latched'], 1)
        self.assertFalse(made.nvme.active)
        self.assertTrue(made.registry.tier_on, 'the host tier goes on without it')
        made.nvme.store.bf.fail = None
        start, grant = made.serve(Request('late', a.all_token_ids + span(701, 5), salt=a.cache_salt))
        self.assertEqual(made.registry.snapshot()['nvme_restore_requests'], 0, 'a latched tier reads nothing more')

    def test_a_block_read_that_fails_is_a_miss_for_the_restore(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        made.serve_checkpoint_first = None
        again = self.returning(a)
        made.nvme.store.bf.fail = None
        hooks = made.graft.tier_hooks
        self.assertEqual(hooks.effective_hit(again, 0), 4096)         # (adopts the checkpoint)
        made.nvme.store.bf.fail = lambda what, offset, length: (_ for _ in ()).throw(IOError('dead')) if what == 'read' else None
        start, grant = made.serve(again)
        stats = made.registry.snapshot()
        self.assertEqual(start, 0)
        self.assertGreater(stats['nvme_restore_failures'], 0)
        self.assertEqual(stats['nvme_restore_blocks'], 0)
        self.assertTrue(made.check_kv())

    def test_the_audit_reads_every_object_back_after_it_is_written(self):
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_AUDIT': '1'})
        self.push_out(made, first_turn())
        stats = made.registry.snapshot()
        self.assertGreater(stats['nvme_audit_reads'], 60)
        self.assertEqual(stats['nvme_audit_mismatches'], 0)


class FingerprintDropTests(NvmeCase):
    def test_entries_of_another_layout_are_dropped_at_open_and_never_restored(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        entries = made.registry.snapshot()['nvme_entries']
        self.assertGreater(entries, 0)
        made.close()
        other = self.rig(environ={'QWEN_SDPA_BF8': '1'})                # a switch that changes the stored bytes by design
        stats = other.registry.snapshot()
        self.assertEqual((stats['nvme_fp_drops'], stats['nvme_entries']), (entries, 0))
        self.assertTrue(any('dropped %d entries of another layout fingerprint' % entries in line for line in other.lines), other.lines[:4])
        start, grant = other.serve(self.returning(a))
        self.assertEqual(start, 0)
        self.assertEqual(other.registry.snapshot()['nvme_restore_requests'], 0)
        other.close()

    def test_a_deploy_keeps_the_tier_warm_a_new_profile_name_and_image_do_not_matter(self):
        made = self.rig()
        a = self.push_all_the_way(made, first_turn())
        entries = made.registry.snapshot()['nvme_entries']
        self.assertGreater(entries, 0)
        made.close()
        # another image build serving a renamed profile, and a switch that is bit-exact against its baseline flipped: the same tier
        after = self.rig(environ={'QWEN_C2_PROFILE': 'a-renamed-profile', 'QWEN_FAST_SDPA_PF_ONEQ': '1', 'QWEN_FAST_LEVER_N': '1'})
        stats = after.registry.snapshot()
        self.assertEqual((stats['nvme_fp_drops'], stats['nvme_entries']), (0, entries))
        start, grant = after.serve(self.returning(a))
        self.assertEqual((start, grant.q), (4096, 4096), 'the returning session restored from the drive after the deploy')
        self.assertGreater(after.registry.snapshot()['nvme_restore_blocks'], 0)
        self.assertTrue(after.check_kv())
        after.close()

    def test_a_new_epoch_is_a_new_fingerprint(self):
        made = self.rig()
        self.push_out(made, first_turn())
        made.close()
        with mock.patch.object(kvnvme, 'KV_EPOCH', kvnvme.KV_EPOCH + 1):
            newer = self.rig()
            self.assertEqual(newer.registry.snapshot()['nvme_entries'], 0)
            newer.close()

    def test_the_compat_token_forces_a_fresh_tier(self):
        made = self.rig()
        self.push_out(made, first_turn())
        made.close()
        newer = self.rig(nvme_env={'QWEN_PREFIX_NVME_COMPAT': 'rev2'})
        self.assertEqual(newer.registry.snapshot()['nvme_entries'], 0)
        newer.close()


class DeletionTests(NvmeCase):
    def test_deleting_a_salt_removes_that_tenants_entries_everywhere_on_the_drive(self):
        made = self.rig()
        a = self.push_out(made, first_turn('a1', 'alpha'))
        b = self.push_out(made, first_turn('b1', 'beta', seed=9))
        self.assertIn(True, made.nvme_has_session(a))
        self.assertIn(True, made.nvme_has_session(b))
        removed = made.nvme.delete_salt(a.cache_salt)
        self.assertGreater(removed, 60)
        self.assertNotIn(True, made.nvme_has_session(a))
        self.assertIsNone(made.nvme.store.lookup(store_module.CKPT, a.block_hashes[63]))
        self.assertIn(True, made.nvme_has_session(b), 'another tenant is untouched')
        start, grant = made.serve(self.returning(a))
        self.assertEqual(start, 0)
        made.settle()                    # (a record the host tier just evicted is in flight to the drive for a moment, and a request that asks in that moment is a miss)
        start, grant = made.serve(self.returning(b))
        self.assertEqual(start, 4096)
        made.close()
        again = self.rig()
        self.assertEqual(again.nvme.store.tenant_entries('alpha'), 0)
        self.assertGreater(again.nvme.store.tenant_entries('beta'), 0)

    def test_the_tenant_tag_is_the_registrys_not_the_salts(self):
        made = self.rig()
        a = self.push_out(made, first_turn('a1', 'alpha'))
        other_salt = 'qps1.alpha.' + 'e' * 64                      # another salt of the same tenant tag
        self.assertEqual(registry_module.tenant_of_salt(other_salt), 'alpha')
        self.assertGreater(made.nvme.delete_salt(other_salt), 0)

    def test_the_whole_tier_wipe_empties_the_drive_and_the_queue(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        self.assertGreater(made.nvme.wipe('test'), 0)
        self.assertEqual(made.nvme.store.snapshot_stats()['entries'], 0)
        self.assertEqual(made.nvme.store.allocator.free, made.nvme.store.geometry.data_bytes)
        start, grant = made.serve(self.returning(a))
        self.assertEqual(start, 0)
        self.assertTrue(made.check_kv())

    def test_reset_prefix_cache_wipes_the_drive_with_the_registry(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        self.assertGreater(made.registry.snapshot()['nvme_entries'], 0)
        made.scheduler.reset_prefix_cache()
        self.assertEqual(made.registry.snapshot()['nvme_entries'], 0)
        made.close()
        again = self.rig()
        self.assertEqual(again.registry.snapshot()['nvme_entries'], 0, 'and a restart does not bring it back')

    def test_the_prefix_kill_switch_wipes_the_drive(self):
        flag = os.path.join(self.directory, 'prefix-reuse.off')
        made = self.rig()
        a = self.push_out(made, first_turn())
        made.graft.kill_switch.path = flag
        open(flag, 'w').close()
        made.graft.kill_switch._last_poll = None
        made.graft.kill_switch_engaged()
        self.assertEqual(made.registry.snapshot()['nvme_entries'], 0)

    def test_the_tier_kill_switch_latches_the_nvme_and_wipes_it(self):
        flag = os.path.join(self.directory, 'kv-tier.off')
        made = self.rig(tier_env={'QWEN_PREFIX_HOST_TIER_OFF_PATH': flag})
        a = self.push_out(made, first_turn())
        self.assertGreater(made.registry.snapshot()['nvme_entries'], 0)
        open(flag, 'w').close()
        made.graft.tier_hooks.kill_switch._last_poll = None
        start, grant = made.serve(self.returning(a))
        self.assertEqual(start, 0)
        stats = made.registry.snapshot()
        self.assertEqual((stats['tier_latched'], stats['nvme_latched'], stats['nvme_entries']), (1, 1, 0))

    def test_a_command_directory_wipes_and_deletes_a_tenant(self):
        commands = os.path.join(self.directory, 'cmd')
        os.mkdir(commands)
        made = self.rig(nvme_env={'QWEN_PREFIX_NVME_CMD_DIR': commands})
        a = self.push_out(made, first_turn('a1', 'alpha'))
        b = self.push_out(made, first_turn('b1', 'beta', seed=9))
        open(os.path.join(commands, 'tenant.alpha'), 'w').close()
        open(os.path.join(commands, 'ignored.txt'), 'w').close()
        made.nvme.last_cmd = 0.0
        made.nvme.poll()
        made.settle()
        deadline = time.time() + 10
        while made.nvme.store.tenant_entries('alpha') and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(made.nvme.store.tenant_entries('alpha'), 0)
        self.assertGreater(made.nvme.store.tenant_entries('beta'), 0)
        self.assertEqual(sorted(os.listdir(commands)), ['ignored.txt'], 'a command file is consumed')
        open(os.path.join(commands, 'wipe'), 'w').close()
        made.nvme.last_cmd = 0.0
        made.nvme.poll()
        deadline = time.time() + 10
        while made.nvme.store.snapshot_stats()['entries'] and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(made.nvme.store.snapshot_stats()['entries'], 0)


class CrashTests(NvmeCase):
    def test_a_crash_in_the_middle_of_a_spill_leaves_a_device_that_opens_with_whole_entries_only(self):
        hook = threading.excepthook
        threading.excepthook = lambda args: None if issubclass(args.exc_type, SimDisk.Crash) else hook(args)
        self.addCleanup(setattr, threading, 'excepthook', hook)
        for crash_after in (1, 2, 3, 5, 8):
            with open(self.device, 'wb') as handle:
                handle.truncate(64 << 20)
            made = NvmeRig(self.device, tier=self.TIER, num_blocks=SMALL_POOL, budget=1 << 20, file_factory=SimDisk)
            try:
                bf = made.nvme.store.bf
                bf.pending = []
                bf.crash_at = bf.calls + crash_after
                a = first_turn()
                made.serve(a)
                for index in range(5):
                    made.serve(Request('fill-%d' % index, span(6000, 20 + index), salt=salt_of('bravo%d' % index)))
                deadline = time.time() + 10
                while bf.calls < bf.crash_at and time.time() < deadline:
                    time.sleep(0.005)
                time.sleep(0.05)
                bf.crash(random.Random(crash_after))
                os.close(bf.fd)
                made.nvme.closed = True
                with made.nvme.cond:
                    made.nvme.cond.notify_all()
            finally:
                made.registry.nvme = None
                made.close()
            again = NvmeRig(self.device, tier=self.TIER, num_blocks=SMALL_POOL, budget=1 << 20)
            try:
                store = again.nvme.store
                for entry in list(store.entries.values()):
                    out = bytearray(entry.nbytes)
                    store.bf.read_run(entry.offset, [entry.nbytes], [memoryview(out)])
                    self.assertEqual(hashlib.sha256(bytes(out)).digest(), entry.digest, 'crash after %d calls: an entry that is not whole' % crash_after)
                start, grant = again.serve(self.returning(a))
                self.assertIn(start, (0, 4096), 'a cold turn or an exact hit')
                self.assertTrue(again.check_kv())
            finally:
                again.close()


class ConcurrencyTests(NvmeCase):
    def test_a_restore_beside_a_busy_writer_is_exact_and_the_writer_yields(self):
        made = self.rig()
        a = self.push_all_the_way(made, first_turn('a1', 'alpha'))
        slow = threading.Event()
        original = made.nvme.store.bf.fail

        def pace(what, offset, length):
            if what == 'write':
                slow.set()
                time.sleep(0.01)

        made.nvme.store.bf.fail = pace
        for index in range(3):                                 # new traffic keeps the writer busy while A returns
            made.serve(Request('busy-%d' % index, span(6000, 70 + index), salt=salt_of('gamma%d' % index)))
        start, grant = made.serve(self.returning(a))
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertTrue(made.check_kv())
        made.nvme.store.bf.fail = original
        made.settle()
        self.assertEqual(made.registry.snapshot()['nvme_spill_failures'], 0)

    def test_two_returning_sessions_that_share_their_first_blocks_both_restore(self):
        made = self.rig()
        system = span(2300, 500)
        one = Request('one', system + span(1900, 5), salt=salt_of('shared'))
        made.serve(one)
        for index in range(4):
            made.serve(Request('fill-%d' % index, span(6000, 20 + index), salt=salt_of('bravo%d' % index)))
        made.settle()
        two = Request('two', one.all_token_ids + span(600, 6), salt=one.cache_salt)
        three = Request('three', one.all_token_ids + span(650, 7), salt=one.cache_salt)
        made.serve(two)
        made.serve(three)
        self.assertTrue(made.check_kv())


class SlicedRestoreTests(NvmeCase):
    def test_a_sliced_restore_reads_the_nvme_blocks_a_slice_at_a_time(self):
        made = self.rig(tier_env={'QWEN_PREFIX_TIER_RESTORE_SLICE': '16', 'QWEN_PREFIX_TIER_RESTORE_SLICE_MS': '60000'})
        hooks = made.graft.tier_hooks
        hooks.hold_ready = True                                 # (the admission wrapper that asks for the hold is test_qwen_prefix_tier_slice's)
        a = self.push_out(made, first_turn())
        on_nvme = sum(1 for key in a.block_hashes[:64] if not made.registry.tier.has(key))
        self.assertTrue(hooks.slicing)
        again = self.returning(a)
        held, steps = 0, 0
        while hooks.hold(again) and steps < 40:
            held += 1
            made.graft.step_no += 1
            steps += 1
        self.assertGreater(held, 1, 'the restore took more than one step')
        job_lines = [line for line in made.lines if line.startswith('host tier restore slice req=again-a1')]
        self.assertGreater(len(job_lines), 1)
        stats = made.registry.snapshot()
        self.assertEqual(stats['nvme_restore_blocks'], on_nvme)
        self.assertGreater(stats['nvme_restore_requests'], 1, 'one NVMe read per slice')
        start, grant = made.serve(again)
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertEqual(made.registry.tier.pinned, set(), 'the job let go of every pin')
        self.assertTrue(made.check_kv())

    def test_a_sliced_restore_whose_nvme_block_vanished_is_abandoned_and_the_turn_goes_cold(self):
        made = self.rig(tier_env={'QWEN_PREFIX_TIER_RESTORE_SLICE': '16', 'QWEN_PREFIX_TIER_RESTORE_SLICE_MS': '60000'})
        hooks = made.graft.tier_hooks
        hooks.hold_ready = True
        a = self.push_out(made, first_turn())
        again = self.returning(a)
        self.assertTrue(hooks.hold(again))
        made.nvme.store.delete([(store_module.KV, a.block_hashes[40])])
        made.graft.step_no += 1
        while hooks.hold(again):
            made.graft.step_no += 1
        self.assertTrue(any('slice aborted req=again-a1' in line and 'NVMe block' in line for line in made.lines), made.lines[-6:])
        start, grant = made.serve(again)
        self.assertEqual(start, 0)
        self.assertTrue(made.check_kv())


class DirectBackendTests(NvmeCase):
    class Double(kvnvme.RestoreBackend):
        """What a P2P backend does, in memory: the entries' bytes into the device blocks the pool allocated, with no host record."""

        name = 'double'
        direct = True

        def __init__(self):
            self.calls = []

        def restore_to_device(self, tier, entries, block_ids, io):
            self.calls.append((len(entries), list(block_ids)))
            for entry, block in zip(entries, block_ids):
                io.device[block] = bytes(tier.store.read_bytes(entry))

    def test_nvme_blocks_go_straight_to_the_device_blocks_and_the_turn_is_exact(self):
        made = self.rig()
        double = self.Double()
        made.nvme.backend = double
        a = self.push_out(made, first_turn())
        records_before = len(made.registry.tier.records)
        start, grant = made.serve(self.returning(a))
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertTrue(double.calls and double.calls[0][0] > 0)
        self.assertEqual(made.registry.snapshot()['nvme_restore_requests'], 1)
        self.assertTrue(made.check_kv())
        self.assertLessEqual(len(made.registry.tier.records), records_before, 'the NVMe blocks did not become host records')

    def test_a_direct_backend_fails_to_a_miss_when_a_block_is_gone(self):
        made = self.rig()
        made.nvme.backend = self.Double()
        a = self.push_out(made, first_turn())
        start, grant = made.serve(Request('again', a.all_token_ids + span(700, 3), salt=a.cache_salt))
        made.nvme.store.delete([(store_module.KV, a.block_hashes[30])])
        start, grant = made.serve(Request('again2', a.all_token_ids + span(701, 3), salt=a.cache_salt))
        self.assertTrue(made.check_kv())


class ZeroRetentionTests(NvmeCase):
    """Owner decision 2026-10-21: the tier refuses to start on a node that advertises zero-retention unless an explicit override says otherwise; the serving process cannot learn the
    node agent's attestation, so it reads the signals it can see and requires the profile's acknowledgement."""

    def test_each_signal_a_serving_process_can_see_is_named(self):
        self.assertEqual(kvnvme.zero_retention_signals({}, exists=lambda path: False), [])
        for name, value, word in (('THATCH_ATTESTATION_PRIVATE_KEY', 'ab' * 32, 'attestations'), ('THATCH_ZERO_RETENTION', '1', 'advertises zero-retention'),
                                  ('THATCH_ZERO_RETENTION', 'TRUE', 'advertises zero-retention'), ('THATCH_NO_LOG', '1', 'log control')):
            reasons = kvnvme.zero_retention_signals({name: value}, exists=lambda path: False)
            self.assertEqual(len(reasons), 1, (name, reasons))
            self.assertIn(word, reasons[0])
            self.assertNotIn('ab' * 32, reasons[0], 'the key itself is never printed')
        for name, value in (('THATCH_ZERO_RETENTION', '0'), ('THATCH_ZERO_RETENTION', ''), ('THATCH_NO_LOG', '0'), ('THATCH_ATTESTATION_PRIVATE_KEY', '')):
            self.assertEqual(kvnvme.zero_retention_signals({name: value}, exists=lambda path: False), [], (name, value))
        marker = kvnvme.zero_retention_signals({}, exists=lambda path: path == kvnvme.ZERO_RETENTION_MARKER)
        self.assertEqual(len(marker), 1)
        self.assertIn('marker', marker[0])

    def refused(self, **environ):
        made = NvmeRig(self.device, tier=self.TIER, num_blocks=SMALL_POOL, budget=1 << 20, environ=environ)
        self.addCleanup(made.close)
        return made, made.lines

    def test_a_zero_retention_node_does_not_start_the_tier_and_says_so(self):
        for environ in ({'THATCH_ZERO_RETENTION': '1'}, {'THATCH_NO_LOG': '1'}, {'THATCH_ATTESTATION_PRIVATE_KEY': 'f' * 64}):
            with open(self.device, 'wb') as handle:
                handle.truncate(64 << 20)
            made, written = self.refused(**environ)
            registry = made.registry
            self.assertIsNone(registry.nvme)
            self.assertIsNone(registry.nvme_plan)
            self.assertEqual(registry.nvme_refusal, 'zero_retention')
            self.assertIsNone(registry.tier.spill_sink)
            self.assertEqual(registry.tier._spare, 0, 'no slot is mapped for a tier that does not run')
            line = [text for text in written if 'REFUSED' in text]
            self.assertEqual(len(line), 1, written)
            self.assertIn('nvme tier REFUSED to start on a zero-retention node', line[0])
            self.assertIn('QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION', line[0])
            snapshot = registry.snapshot()
            self.assertEqual((snapshot['nvme_refused_zero_retention'], snapshot['nvme_on'], snapshot['nvme_unavailable']), (1, 0, 0))
            self.assertNotIn(b'QNVMSB01', Path(self.device).read_bytes()[:8192], 'the drive was not touched')
            self.assertEqual(sum(Path(self.device).read_bytes()[:1 << 20]), 0)
            start, grant = made.serve(first_turn())          # and the engine serves, host tier and all
            self.assertTrue(made.check_kv())
            made.close()

    def test_the_override_starts_it_anyway_and_logs_why(self):
        made, written = self.refused(THATCH_ZERO_RETENTION='1', QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION='1')
        self.assertIsNotNone(made.registry.nvme)
        self.assertIsNone(made.registry.nvme_refusal)
        self.assertTrue(any('starts on a node that looks zero-retention because QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION=1' in text for text in written), written[:3])
        self.assertEqual(made.registry.snapshot()['nvme_refused_zero_retention'], 0)

    def test_no_signal_no_refusal(self):
        made, written = self.refused()
        self.assertIsNotNone(made.registry.nvme)
        self.assertFalse(any('REFUSED' in text for text in written))


class UnavailableDriveTests(NvmeCase):
    """A drive that is not there, not ours or in use costs the engine its NVMe tier, not its life: loud line, a counter, the host tier serves."""

    def start(self, device, **kwargs):
        made = NvmeRig(device, tier=self.TIER, num_blocks=SMALL_POOL, budget=1 << 20, **kwargs)
        self.addCleanup(made.close)
        return made, made.lines

    def assert_unavailable(self, made, written):
        self.assertIsNone(made.registry.nvme)
        self.assertEqual(made.registry.nvme_refusal, 'unavailable')
        self.assertTrue(any('nvme tier UNAVAILABLE' in text for text in written), written[:3])
        snapshot = made.registry.snapshot()
        self.assertEqual((snapshot['nvme_unavailable'], snapshot['nvme_on']), (1, 0))
        self.assertIsNone(made.registry.tier.spill_sink)
        start, grant = made.serve(first_turn())
        self.assertTrue(made.check_kv())

    def test_a_missing_device(self):
        made, written = self.start(os.path.join(self.directory, 'nothing-here'))
        self.assert_unavailable(made, written)

    def test_a_device_that_is_someone_elses_is_never_written(self):
        with open(self.device, 'wb') as handle:
            handle.write(b'this is somebody\'s filesystem' * 100)
            handle.truncate(64 << 20)
        before = Path(self.device).read_bytes()
        made, written = self.start(self.device)
        self.assert_unavailable(made, written)
        self.assertEqual(Path(self.device).read_bytes(), before)

    def test_a_device_another_process_holds(self):
        holder = store_module.NvmeStore.open(self.device, size=64 << 20, snap_bytes=16 * ALIGN, journal_bytes=16 * ALIGN)
        self.addCleanup(holder.close)
        made, written = self.start(self.device)
        self.assert_unavailable(made, written)

    def test_format_force_takes_a_foreign_device_when_the_operator_says_so(self):
        with open(self.device, 'wb') as handle:
            handle.write(b'old data' * 1000)
            handle.truncate(64 << 20)
        made, written = self.start(self.device, nvme_env={'QWEN_PREFIX_NVME_FORMAT': '1'})
        self.assertIsNotNone(made.registry.nvme)


class WriteThroughTests(NvmeCase):
    """QWEN_PREFIX_NVME_SPILL=through: a warm copy reaches the drive soon after a session goes idle, before any eviction."""

    def through(self, **extra):
        env = {'QWEN_PREFIX_NVME_SPILL': 'through', 'QWEN_PREFIX_NVME_THROUGH_IDLE_S': '0'}
        env.update(extra)
        return self.rig(num_blocks=3000, nvme_env=env)

    def idle_steps(self, made, count=12):
        for index in range(count):
            made.serve(Request('idle-%d-%d' % (len(made.rows), index), span(300, 900 + index), salt=salt_of('idler')))      # (short: nothing of its own to copy)
        made.settle()

    def test_without_the_option_an_idle_session_that_was_never_evicted_is_not_on_the_drive(self):
        made = self.rig(num_blocks=3000)
        a = first_turn()
        made.serve(a)
        self.idle_steps(made)
        self.assertEqual(made.registry.snapshot()['nvme_entries'], 0)
        self.assertFalse(made.graft.tier_hooks.idle)
        self.assertIsNone(made.registry.tier.commit_sink)

    def test_an_idle_session_is_copied_in_bounded_steps_and_lands_on_the_drive_without_an_eviction(self):
        made = self.through(QWEN_PREFIX_NVME_THROUGH_BLOCKS='16')
        a = first_turn()
        made.serve(a)
        hooks = made.graft.tier_hooks
        self.assertIsNotNone(made.registry.tier.commit_sink)
        self.assertTrue(any('spill=through through_blocks=16 through_idle_s=0' in line for line in made.lines))
        self.assertGreater(made.nvme.backlog_blocks, 0, 'the finished request is waiting')
        self.idle_steps(made, 6)
        stats = made.registry.snapshot()
        self.assertEqual(stats['tier_spill_blocks'], 0, 'no block was evicted from the device: this is the idle copy, not the spill')
        self.assertEqual(stats['nvme_through_copied_blocks'], 64)
        self.assertEqual(stats['nvme_through_copy_steps'], 4, '64 blocks at 16 a step')
        self.assertEqual(stats['nvme_through_queued'], 64)
        self.assertEqual(made.session_somewhere(a), 64)
        self.assertEqual(made.nvme_has_session(a, 64).count(True), 64, 'every block is on the drive')
        self.assertIsNotNone(made.nvme.store.lookup(store_module.CKPT, a.block_hashes[63]), 'and so is the checkpoint')
        self.assertEqual(made.nvme.backlog_blocks, 0)
        self.assertEqual(stats['nvme_through_failures'], 0)
        self.assertEqual(len(made.registry.tier.held), 0, 'every hold was given back')
        self.assertTrue(made.check_kv())

    def test_a_restart_after_the_idle_copy_restores_a_session_that_was_never_evicted(self):
        made = self.through()
        a = first_turn()
        made.serve(a)
        self.idle_steps(made)
        made.close()
        again = self.through()
        self.assertEqual(len(again.registry.tier.records), 0, 'a new process: nothing in RAM')
        start, grant = again.serve(self.returning(a))
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertGreater(again.registry.snapshot()['nvme_restore_blocks'], 0)
        self.assertTrue(again.check_kv())

    def test_the_default_mode_loses_the_same_session_at_a_restart(self):
        made = self.rig(num_blocks=3000)
        a = first_turn()
        made.serve(a)
        made.settle()
        made.close()
        again = self.rig(num_blocks=3000)
        start, grant = again.serve(self.returning(a))
        self.assertEqual(start, 0)

    def test_the_copy_waits_for_the_session_to_be_idle(self):
        made = self.through(QWEN_PREFIX_NVME_THROUGH_IDLE_S='600')
        clock = types.SimpleNamespace(now=1000.0)
        made.graft.kill_switch.clock = lambda: clock.now
        a = first_turn()
        made.serve(a)
        self.idle_steps(made, 3)
        self.assertEqual(made.registry.snapshot()['nvme_through_copied_blocks'], 0, 'a minute later is not idle yet')
        clock.now += 601
        self.idle_steps(made, 4)
        self.assertEqual(made.registry.snapshot()['nvme_through_copied_blocks'], 64)

    def test_a_follow_up_turn_does_not_copy_what_the_drive_already_has(self):
        made = self.through()
        a = first_turn()
        made.serve(a)
        self.idle_steps(made)
        copied = made.registry.snapshot()['nvme_through_copied_blocks']
        made.serve(self.returning(a))                    # a device hit: the same 64 blocks, plus a tail that is not a whole published block
        self.idle_steps(made)
        stats = made.registry.snapshot()
        self.assertLessEqual(stats['nvme_through_copied_blocks'] - copied, 1, 'only a new block can be copied')

    def test_a_block_that_left_the_device_meanwhile_is_left_to_the_ordinary_spill(self):
        made = self.through(QWEN_PREFIX_NVME_THROUGH_IDLE_S='600')
        clock = types.SimpleNamespace(now=1000.0)
        made.graft.kill_switch.clock = lambda: clock.now
        a = first_turn()
        made.serve(a)
        pool = made.scheduler.kv_cache_manager.block_pool
        pool.cached_block_hash_to_block.map.clear()          # (the device lost them all)
        clock.now += 601
        self.idle_steps(made, 4)
        self.assertEqual(made.registry.snapshot()['nvme_through_copied_blocks'], 0)

    def test_a_failing_device_read_in_the_idle_copy_is_counted_and_three_latch_the_host_tier(self):
        made = self.through()
        made.kv.fail_read = True
        for index in range(4):
            made.serve(Request('s%d' % index, span(4200, 400 + index), salt=salt_of('t%d' % index)))
        for index in range(4):
            made.serve(Request('idle-%d' % index, span(300, 900 + index), salt=salt_of('idler')))
        stats = made.registry.snapshot()
        self.assertGreaterEqual(stats['nvme_through_failures'], 1)
        self.assertEqual(stats['tier_latched'], 1)
        made.kv.fail_read = False

    def test_held_records_are_never_victims_and_a_removed_one_gives_its_slot_back_when_released(self):
        made = self.rig(tier=4)
        tier = made.registry.tier
        for index in range(4):
            tier.put(b'h%d' % index, kv_content(b'h%d' % index), 't', b'c', index)
        tier.held.add(b'h0')
        for index in range(4, 8):
            tier.put(b'h%d' % index, kv_content(b'h%d' % index), 't', b'c', index)
        self.assertIn(b'h0', tier.records, 'the oldest record is held: the next oldest went instead')
        self.assertNotIn(b'h1', tier.records)
        free_before = len(tier._free)
        tier.remove(b'h0')
        self.assertEqual(len(tier._free), free_before, 'a held record\'s slot waits for its reader')
        self.assertIn(b'h0', tier.zombies)
        tier.release_held(b'h0')
        self.assertEqual(len(tier._free), free_before + 1)
        self.assertEqual((tier.held, tier.zombies), (set(), {}))

    def test_a_failed_batch_is_written_again_once(self):
        made = self.rig()
        failures = []

        def flaky(what, offset, length):
            if what == 'write' and not failures:
                failures.append(offset)
                raise IOError('a transient error')

        made.nvme.store.bf.fail = flaky
        a = self.push_out(made, first_turn())
        stats = made.registry.snapshot()
        self.assertEqual(failures and 1, 1)
        self.assertEqual(stats['nvme_spill_retries'], 1)
        self.assertEqual(stats['nvme_spill_failures'], 0)
        self.assertGreater(stats['nvme_spill_blocks'], 0)

    def test_random_traffic_with_the_idle_copy_is_exact(self):
        rng = random.Random(5)
        made = self.rig(num_blocks=170, tier=60, nvme_env={'QWEN_PREFIX_NVME_SPILL': 'through', 'QWEN_PREFIX_NVME_THROUGH_IDLE_S': '0', 'QWEN_PREFIX_NVME_THROUGH_BLOCKS': '24'})
        systems = {name: span(2300, 500 + index) for index, name in enumerate(('alpha', 'bravo', 'charlie'))}
        histories = {}
        for step in range(30):
            name = rng.choice(['alpha', 'bravo', 'charlie'])
            key = (name, rng.randrange(2))
            base = histories.get(key, systems[name])
            tokens = list(base) + span(rng.randrange(1200, 3200), 4000 + 31 * step)
            made.serve(Request('r%d' % step, tokens, salt=salt_of(name)))
            made.check_kv()
            histories[key] = None if len(tokens) > 7000 else tokens + span(rng.randrange(50, 400), 9000 + step)
            if histories[key] is None:
                del histories[key]
            if step % 5 == 4:
                made.settle()
        made.settle()
        stats = made.registry.snapshot()
        self.assertGreater(stats['nvme_through_copied_blocks'], 0)
        self.assertEqual((stats['nvme_digest_failures'], stats['nvme_spill_failures'], stats['nvme_through_failures']), (0, 0, 0))
        self.assertEqual(len(made.registry.tier.held), 0)


class DefaultWriteThroughTests(NvmeCase):
    """The defaults (owner, 2026-10-21): write-through, idle 0. A finished session is queued for the drive at the next step, with nothing evicted and the clock not advancing."""

    def test_idle_zero_is_immediate_not_off(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})               # (no override: whatever the defaults are)
        self.assertEqual((made.nvme.plan.spill, made.nvme.plan.through_idle_s, made.nvme.plan.through_blocks), ('through', 0, 32))
        self.assertIsNotNone(made.registry.tier.commit_sink)
        frozen = types.SimpleNamespace(now=5000.0)
        made.graft.kill_switch.clock = lambda: frozen.now                                       # (time never moves: 'idle 0' must not wait for it)
        a = first_turn()
        made.serve(a)
        self.assertGreater(made.nvme.backlog_blocks, 0, 'the finished request is queued for the copy')
        for index in range(8):
            made.serve(Request('idle-%d' % index, span(300, 900 + index), salt=salt_of('idler')))
        made.settle()
        self.assertEqual(made.nvme_has_session(a, 64).count(True), 64, 'every block reached the drive without an eviction and without the clock moving')
        self.assertEqual(made.registry.snapshot()['tier_spill_blocks'], 0)
        self.assertIsNotNone(made.nvme.store.lookup(store_module.CKPT, a.block_hashes[63]))
        self.assertEqual(made.nvme.backlog_blocks, 0)

    def test_the_evict_override_is_still_a_choice(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': 'evict'})
        self.assertIsNone(made.registry.tier.commit_sink)
        made.serve(first_turn())
        made.settle()
        self.assertEqual(made.registry.snapshot()['nvme_entries'], 0)

    def test_a_deploy_restart_keeps_a_session_that_was_never_evicted(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        a = first_turn()
        made.serve(a)
        for index in range(8):
            made.serve(Request('idle-%d' % index, span(300, 900 + index), salt=salt_of('idler')))
        made.close()                                                                            # the drain
        again = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''}, environ={'QWEN_C2_PROFILE': 'the-next-profile'})
        start, grant = again.serve(self.returning(a))
        self.assertEqual((start, grant.q), (4096, 4096))
        self.assertTrue(again.check_kv())


class DriveLossTests(NvmeCase):
    """The drive vanishing at run time (a PCI rescan after a card reset disabled its controller: the node stays, the namespace reads 0 bytes). The tier degrades to the host tier with
    ONE line and one counter, never crashes or wedges the scheduler, and comes back on a restart once the device is live."""

    def lost(self, made, why='the namespace nvme0n1 now reads 0 bytes'):
        state = types.SimpleNamespace(problem=None)
        made.nvme._probe = lambda: state.problem
        state.why = why
        return state

    def trouble(self, made):
        return [line for line in made.lines if 'DEVICE LOST' in line]

    def traffic(self, made, count=8):
        """Short requests: the scheduler steps that run the idle copy of whatever finished before them."""
        for index in range(count):
            made.serve(Request('idle-%d-%d' % (len(made.rows), index), span(300, 900 + index), salt=salt_of('idler')))

    def assert_degraded(self, made, a):
        stats = made.registry.snapshot()
        self.assertEqual((stats['nvme_device_lost'], stats['nvme_latched'], stats['nvme_on']), (1, 1, 0))
        self.assertEqual(len(self.trouble(made)), 1, [line for line in made.lines if 'nvme' in line][-6:])
        before = len(made.lines)
        # the engine carries on: new sessions, a returning one (a cold prefill or a host-tier hit), the idle copy and the spill sink all quiet
        made.serve(first_turn('b1', 'beta', seed=5))
        start, grant = made.serve(self.returning(a))
        self.assertTrue(made.check_kv())
        for index in range(6):
            made.serve(Request('more-%d' % index, span(6000, 70 + index), salt=salt_of('filler%d' % index)))
        self.assertEqual([line for line in made.lines[before:] if 'nvme tier' in line], [], 'one line, however long the drive stays gone')
        self.assertEqual(made.registry.snapshot()['nvme_device_lost'], 1)
        self.assertFalse(made.graft.tier_hooks.idle, 'nothing is waiting for a copy toward a drive that is gone')

    def test_the_health_look_notices_the_vanished_namespace_between_steps(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        state = self.lost(made)
        a = first_turn()
        made.serve(a)
        self.traffic(made)
        made.settle()
        self.assertEqual(made.registry.snapshot()['nvme_on'], 1)
        state.problem = state.why
        made.nvme.last_health = 0.0                                                             # (the periodic look is due)
        made.nvme.poll()
        self.assertIn('device lost', made.nvme.latched)
        self.assert_degraded(made, a)

    def test_a_write_the_device_refuses_with_a_gone_errno_latches_at_once_with_no_retry(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        a = first_turn()

        def fail(op, offset, length):
            if op in ('write', 'sync'):
                raise OSError(errno.ENODEV, 'No such device')

        made.nvme.store.bf.fail = fail
        made.serve(a)
        self.traffic(made)
        made.nvme.flush(10.0)
        stats = made.registry.snapshot()
        self.assertEqual((stats['nvme_spill_failures'], stats['nvme_spill_retries']), (1, 0), 'one failed batch, and no second try on a device that is gone')
        self.assert_degraded(made, a)

    def test_an_io_error_latches_at_once_when_the_probe_says_the_device_is_gone_else_after_three(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        state = self.lost(made)

        def fail(op, offset, length):
            if op in ('write', 'sync'):
                raise OSError(errno.EIO, 'Input/output error')

        made.nvme.store.bf.fail = fail
        a = first_turn()
        state.problem = None
        made.serve(a)
        self.traffic(made)
        made.nvme.flush(10.0)
        self.assertIsNone(made.nvme.latched, 'an EIO the probe cannot explain is a strike, not a verdict')
        self.assertTrue(0 < made.registry.snapshot()['nvme_spill_failures'] < 3)
        made.nvme.failures = 0                                                                  # (clean slate: the next failure alone must decide)
        state.problem = state.why
        made.serve(first_turn('b1', 'beta', seed=5))
        self.traffic(made)
        made.nvme.flush(10.0)
        self.assertIn('device lost', made.nvme.latched or '')
        self.assert_degraded(made, a)

    def test_reads_of_a_vanished_drive_make_the_returning_turn_cold_and_latch(self):
        made = self.rig()
        a = self.push_all_the_way(made, first_turn())
        self.assertGreater(made.registry.snapshot()['nvme_entries'], 60)
        made.close()
        again = self.rig()
        state = self.lost(again)
        state.problem = state.why

        def fail(op, offset, length):
            if op == 'read':
                raise OSError(errno.EIO, 'Input/output error')

        again.nvme.store.bf.fail = fail
        start, grant = again.serve(self.returning(a))
        self.assertEqual(start, 0, 'a cold prefill: the NVMe blocks could not be read')
        self.assertTrue(again.check_kv())
        self.assertIn('device lost', again.nvme.latched)
        self.assertEqual(again.registry.snapshot()['nvme_device_lost'], 1)
        self.assertEqual(len(self.trouble(again)), 1)

    def test_a_checkpoint_read_of_a_vanished_drive_latches_too(self):
        made = self.rig()
        a = self.push_all_the_way(made, first_turn())
        entry = made.nvme.store.lookup(store_module.CKPT, a.block_hashes[63])
        self.assertIsNotNone(entry)
        state = self.lost(made)
        state.problem = state.why

        def fail(op, offset, length):
            if op == 'read':
                raise OSError(errno.ENXIO, 'No such device or address')

        made.nvme.store.bf.fail = fail
        self.assertIsNone(made.nvme.load_checkpoint(entry))
        self.assertIn('device lost', made.nvme.latched)
        self.assertEqual(len(self.trouble(made)), 1)

    def test_a_wipe_or_a_tenant_delete_that_cannot_reach_the_drive_says_so_and_does_not_raise(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        self.assertGreater(made.registry.snapshot()['nvme_entries'], 0)
        state = self.lost(made)
        state.problem = state.why

        def fail(op, offset, length):
            if op in ('write', 'sync'):
                raise OSError(errno.EIO, 'Input/output error')

        made.nvme.store.bf.fail = fail
        self.assertEqual(made.nvme.wipe('test'), 0)
        self.assertTrue(any('wipe NOT APPLIED' in line for line in made.lines), made.lines[-4:])
        self.assertGreater(len(made.nvme.store.entries), 0, 'the entries are still the drive\'s')
        self.assertEqual(made.nvme.delete_salt(a.cache_salt), 0)
        self.assertTrue(any('tenant delete NOT APPLIED' in line for line in made.lines))
        self.assertEqual(len(self.trouble(made)), 1)
        # a failure the probe cannot explain is not swallowed
        state.problem = None
        made.nvme.latched = None
        with self.assertRaises(OSError):
            made.nvme.wipe('test')

    def test_the_tier_is_back_after_a_restart_once_the_device_is_live(self):
        made = self.rig()
        a = self.push_all_the_way(made, first_turn())
        entries = made.registry.snapshot()['nvme_entries']
        self.assertGreater(entries, 60)
        state = self.lost(made)
        state.problem = state.why
        made.nvme.last_health = 0.0
        made.nvme.poll()
        self.assertIsNotNone(made.nvme.latched)
        made.close()
        back = self.rig()
        self.assertEqual((back.registry.snapshot()['nvme_on'], back.registry.snapshot()['nvme_entries']), (1, entries))
        start, grant = back.serve(self.returning(a))
        self.assertEqual((start, grant.q), (4096, 4096))

    def test_a_namespace_that_reads_zero_bytes_at_start_is_unavailable_and_never_formatted(self):
        with open(self.device, 'wb') as handle:
            handle.truncate(0)
        for extra in ({}, {'QWEN_PREFIX_NVME_FORMAT': '1'}):
            made = NvmeRig(self.device, tier=self.TIER, num_blocks=SMALL_POOL, budget=1 << 20, nvme_env=extra)
            self.addCleanup(made.close)
            self.assertIsNone(made.registry.nvme)
            self.assertEqual(made.registry.nvme_refusal, 'unavailable')
            self.assertTrue(any('nvme tier UNAVAILABLE' in line and 'reads 0 bytes' in line for line in made.lines), made.lines[:3])
            self.assertEqual(os.path.getsize(self.device), 0, 'nothing was written to it')
        with self.assertRaises(store_module.DeviceGone):
            store_module.NvmeStore.open(self.device, format='force')
        self.assertEqual(os.path.getsize(self.device), 0)

    def test_the_writer_thread_never_dies_quietly(self):
        made = self.rig()
        original = made.nvme._write_batch

        def broken(batch):
            raise RuntimeError('boom')

        made.nvme._write_batch = broken
        self.push_out(made, first_turn())
        self.assertTrue(made.nvme.flush(10.0), 'the queue drained: the writer latched and gave the slots back')
        self.assertIn('writer thread failed', made.nvme.latched)
        self.assertEqual(made.nvme.queue_kv_bytes + made.nvme.queue_ckpt_bytes, 0)
        self.assertTrue(made.check_kv())
        made.nvme._write_batch = original


class DeviceProbeTests(unittest.TestCase):
    """device_problem: a stat and a few sysfs attribute reads, never device IO. The block device is found by major:minor (inside a container its node can have any name)."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='qkvn-sysfs-')
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.sysfs = os.path.join(self.directory, 'sys')
        self.node = os.path.join(self.directory, 'any-name-in-a-container')
        Path(self.node).write_bytes(b'x' * 4096)                                      # (a regular file stands in for the node: its major:minor is 0:0)
        patcher = mock.patch.object(store_module.stat, 'S_ISBLK', lambda mode: True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.controller = os.path.join(self.sysfs, 'devices', 'pci0000:00', 'nvme', 'nvme9')
        self.namespace = os.path.join(self.controller, 'nvme9n1')

    def attribute(self, path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(text + '\n')

    def live(self, sectors='1953525168', state='live', linked=True):
        self.attribute(os.path.join(self.namespace, 'size'), sectors)
        self.attribute(os.path.join(self.controller, 'state'), state)
        os.makedirs(os.path.join(self.sysfs, 'dev', 'block'), exist_ok=True)
        link = os.path.join(self.sysfs, 'dev', 'block', '0:0')
        if linked and not os.path.lexists(link):
            os.symlink(self.namespace, link)

    def problem(self, expect=0):
        return store_module.device_problem(self.node, expect, self.sysfs)

    def test_a_live_device_is_fine_whatever_its_node_is_called(self):
        self.live()
        self.assertIsNone(self.problem(expect=900 << 30))

    def test_a_dead_or_deleting_controller_is_reported(self):
        self.live(state='dead')
        self.assertIn('is dead', self.problem())
        self.attribute(os.path.join(self.controller, 'state'), 'deleting')
        self.assertIn('is deleting', self.problem())
        self.attribute(os.path.join(self.controller, 'state'), 'deleting (no IO)')
        self.assertIn('deleting (no IO)', self.problem())

    def test_a_namespace_that_reads_zero_bytes_or_less_than_it_was_formatted_for(self):
        self.live(sectors='0')
        self.assertIn('reads 0 bytes', self.problem())
        self.live(sectors='2048')
        self.assertIn('now reads 1048576 bytes, the store was formatted for 2097152', self.problem(expect=2 << 20))
        self.assertIsNone(self.problem(expect=1 << 20))

    def test_transient_states_are_not_a_verdict(self):
        for state in ('resetting', 'connecting', 'new'):
            self.live(state=state)
            self.attribute(os.path.join(self.controller, 'state'), state)
            self.assertIsNone(self.problem(), state)

    def test_a_device_number_with_no_sysfs_entry_is_gone_but_a_host_without_block_sysfs_cannot_be_asked(self):
        self.live(linked=False)
        self.assertIn('no sysfs entry', self.problem())
        self.assertIsNone(store_module.device_problem(self.node, 0, os.path.join(self.directory, 'no-sysfs-here')))

    def test_a_node_that_is_gone(self):
        os.remove(self.node)
        self.assertIn('is gone', self.problem())

    def test_a_regular_file_stands_in_for_a_device_in_the_tests(self):
        with mock.patch.object(store_module.stat, 'S_ISBLK', lambda mode: False):
            self.assertIsNone(self.problem(expect=4096))
            self.assertIn('formatted for 8192', self.problem(expect=8192))
            Path(self.node).write_bytes(b'')
            self.assertIn('reads 0 bytes', self.problem())


class StopTests(NvmeCase):
    """A graceful stop drains the writer's queue inside a bounded time, journals, snapshots when there is time, and leaves the device consistent."""

    def slow(self, made, seconds):
        def fail(op, offset, length):
            if op == 'write':
                time.sleep(seconds)

        made.nvme.store.bf.fail = fail

    def queued(self, made, count=8):
        a = first_turn()
        made.serve(a)
        for index in range(count):
            made.serve(Request('idle-%d' % index, span(300, 900 + index), salt=salt_of('idler')))
        return a

    def test_a_stop_flushes_what_is_queued_then_closes_and_a_restart_finds_it(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        self.slow(made, 0.02)
        a = self.queued(made)
        self.assertGreater(made.nvme.queue_kv_bytes, 0, 'the writer is behind')
        began = time.monotonic()
        self.assertTrue(made.nvme.shutdown(budget_s=8))
        self.assertLess(time.monotonic() - began, 6)
        self.assertTrue(made.nvme.closed)
        self.assertTrue(any('nvme tier stopped drained=1' in line for line in made.lines), made.lines[-3:])
        again = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        self.assertEqual(again.nvme_has_session(a, 64).count(True), 64)
        self.assertIsNotNone(again.nvme.store.lookup(store_module.CKPT, a.block_hashes[63]))

    def test_the_budget_is_kept_and_what_the_writer_did_not_reach_is_dropped_cleanly(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        self.slow(made, 0.15)
        self.queued(made)
        began = time.monotonic()
        drained = made.nvme.shutdown(budget_s=1.5)
        elapsed = time.monotonic() - began
        self.assertFalse(drained)
        self.assertLess(elapsed, 4.0, 'the stop returned by its budget plus the batch in hand')
        self.assertTrue(made.nvme.closed)
        made.nvme.writer.join(10)                                                                  # (the batch in hand finishes and gives its holds back)
        self.assertFalse(made.nvme.writer.is_alive())
        self.assertEqual(len(made.registry.tier.held), 0, 'every hold was given back')
        self.assertTrue(made.nvme.queue_kv_bytes + made.nvme.queue_ckpt_bytes == 0)
        made.nvme.store.close()                                  # (an abandoned writer leaves the store open: in a real stop the process is exiting; here the lock must go)
        again = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        self.assertIsNotNone(again.nvme, 'the device is consistent: it opens, with whatever prefix was written')

    def test_no_offers_after_the_stop_began_and_a_second_stop_is_a_no_op(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        self.assertTrue(made.nvme.active)
        made.nvme.begin_stop()
        self.assertFalse(made.nvme.active)
        self.assertGreater(made.nvme.stop_deadline, time.monotonic())
        self.assertTrue(made.nvme.shutdown(budget_s=3))
        began = time.monotonic()
        self.assertTrue(made.nvme.shutdown(budget_s=3))
        self.assertLess(time.monotonic() - began, 0.1)

    def test_the_index_is_snapshotted_when_time_remains_and_not_when_it_does_not(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        self.queued(made)
        before = made.nvme.store.stats['snapshots']
        made.nvme.shutdown(budget_s=6)
        self.assertEqual(made.nvme.store.stats['snapshots'], before + 1)
        quick = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        before = quick.nvme.store.stats['snapshots']
        quick.nvme.shutdown(budget_s=0.2)
        self.assertEqual(quick.nvme.store.stats['snapshots'], before, 'no time for one: the journal is enough')

    def test_a_lost_device_is_not_waited_for(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        made.nvme._probe = lambda: 'gone'
        made.nvme.latch('device lost: gone', lost=True)
        began = time.monotonic()
        made.nvme.shutdown(budget_s=8)
        self.assertLess(time.monotonic() - began, 1.0)
        self.assertTrue(made.nvme.closed)

    def test_a_forked_child_never_closes_or_writes_the_parents_drive(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        made.nvme._pid += 1                                                                    # (as if this were a child process holding a copy)
        self.assertTrue(made.nvme.shutdown(budget_s=1))
        self.assertFalse(made.nvme.closed)
        made.nvme._pid -= 1

    def test_the_hooks_chain_sigterm_and_register_the_exit_finalizers(self):
        fake = types.SimpleNamespace(SIGTERM=15, SIG_DFL=0, SIG_IGN=1, handlers={})
        fake.getsignal = lambda number: fake.handlers.get(number, fake.SIG_DFL)
        fake.signal = lambda number, handler: fake.handlers.__setitem__(number, handler)
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_DRAIN_S': '5', 'QWEN_PREFIX_NVME_SPILL': 'evict'})
        calls = []
        with mock.patch('multiprocessing.util.Finalize') as finalize, mock.patch.object(kvnvme.atexit, 'register') as registered, \
                mock.patch.object(kvnvme.os, 'kill', lambda pid, number: calls.append(('kill', number))):
            made.nvme._hooks_armed = False
            self.assertTrue(made.nvme.arm_stop_hooks(fake))
            self.assertFalse(made.nvme.arm_stop_hooks(fake), 'armed once')
            self.assertEqual(finalize.call_args[0][0], None)
            self.assertEqual(finalize.call_args[1]['exitpriority'], 100)
            registered.assert_called_once_with(made.nvme.shutdown)
        handler = fake.handlers[15]
        self.assertTrue(callable(handler))
        with mock.patch.object(kvnvme.os, 'kill', lambda pid, number: calls.append(('kill', number))):
            handler(15, None)                                        # the previous handler was the default one: drain, then die by it
        self.assertTrue(made.nvme.closed)
        self.assertEqual(calls, [('kill', 15)])
        self.assertEqual(fake.handlers[15], fake.SIG_DFL)

    def test_a_previous_handler_is_called_after_the_drain_clock_starts_and_an_ignored_signal_stays_ignored(self):
        seen = []
        fake = types.SimpleNamespace(SIGTERM=15, SIG_DFL=0, SIG_IGN=1, handlers={15: lambda number, frame: seen.append(('previous', self.tier.stopping))})
        fake.getsignal = lambda number: fake.handlers.get(number, fake.SIG_DFL)
        fake.signal = lambda number, handler: fake.handlers.__setitem__(number, handler)
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_DRAIN_S': '5', 'QWEN_PREFIX_NVME_SPILL': 'evict'})
        self.tier = made.nvme
        with mock.patch('multiprocessing.util.Finalize'), mock.patch.object(kvnvme.atexit, 'register'):
            made.nvme._hooks_armed = False
            made.nvme.arm_stop_hooks(fake)
            fake.handlers[15](15, None)
        self.assertEqual(seen, [('previous', True)])
        self.assertFalse(made.nvme.closed, 'the previous handler (the engine\'s own: it raises SystemExit) owns the exit; the finalizer completes the drain')
        ignored = types.SimpleNamespace(SIGTERM=15, SIG_DFL=0, SIG_IGN=1, handlers={15: 1})
        ignored.getsignal = lambda number: ignored.handlers[number]
        ignored.signal = lambda number, handler: self.fail('SIG_IGN must be left alone')
        made.close()
        other = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_DRAIN_S': '5', 'QWEN_PREFIX_NVME_SPILL': 'evict'})
        with mock.patch('multiprocessing.util.Finalize'), mock.patch.object(kvnvme.atexit, 'register'):
            other.nvme._hooks_armed = False
            other.nvme.arm_stop_hooks(ignored)

    def test_drain_zero_arms_nothing_and_a_plain_close_still_flushes(self):
        made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})              # (the rig's DRAIN_S is 0)
        self.assertEqual(made.nvme.plan.drain_s, 0)
        fake = types.SimpleNamespace(SIGTERM=15, SIG_DFL=0, SIG_IGN=1)
        fake.getsignal = lambda number: self.fail('no signal handler may be touched')
        self.assertFalse(made.nvme.arm_stop_hooks(fake))
        self.slow(made, 0.01)
        a = self.queued(made, count=4)
        self.assertTrue(made.nvme.close())
        again = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_SPILL': ''})
        self.assertEqual(again.nvme_has_session(a, 64).count(True), 64)

    def test_the_registry_arms_the_hooks_of_a_tier_with_a_drain_budget(self):
        with mock.patch.object(kvnvme.NvmeTier, 'arm_stop_hooks', autospec=True, return_value=True) as arm:
            made = self.rig(nvme_env={'QWEN_PREFIX_NVME_DRAIN_S': '4'})
            self.assertEqual(arm.call_count, 1)
            self.assertIs(arm.call_args[0][0], made.nvme)


@unittest.skipUnless(os.environ.get('QWEN_NVME_STOP_CHILD'), 'the body of a child process of StopProcessTests')
class StopChildProbe(NvmeCase):
    """Run only as a child of StopProcessTests: a rig with a slow writer and the real stop hooks. Mode 'default': this very process takes the SIGTERM (default disposition before the
    hooks). Mode 'engine': a multiprocessing child (fork) whose previous SIGTERM handler raises SystemExit, as vLLM's engine core does; its exit runs the finalizers."""

    def test_child(self):
        mode, report = os.environ['QWEN_NVME_STOP_CHILD'].split(':', 1)
        shutil.rmtree(self.directory, True)
        self.directory = os.path.dirname(report)
        self.device = os.path.join(self.directory, 'nvme.img')

        def run():
            if mode == 'engine':
                def previous(number, frame):
                    raise SystemExit()

                signal.signal(signal.SIGTERM, previous)
            made = self.rig(num_blocks=3000, nvme_env={'QWEN_PREFIX_NVME_DRAIN_S': '6', 'QWEN_PREFIX_NVME_SPILL': ''})

            def slow(op, offset, length):
                if op == 'write':
                    time.sleep(0.03)

            made.nvme.store.bf.fail = slow
            made.serve(first_turn())
            for index in range(8):
                made.serve(Request('idle-%d' % index, span(300, 900 + index), salt=salt_of('idler')))
            Path(report).write_text('ready %d %d\n' % (made.nvme.queue_kv_bytes, os.getpid()))
            time.sleep(120)

        if mode == 'engine':
            import multiprocessing

            process = multiprocessing.get_context('fork').Process(target=run)
            process.start()
            process.join(110)
            Path(report + '.exit').write_text('%s\n' % process.exitcode)
        else:
            run()


class StopProcessTests(NvmeCase):
    """A real SIGTERM to a real process: default disposition (the handler drains, then dies by the signal) and the engine-core pattern (a previous handler raising SystemExit in a
    multiprocessing child, whose exit runs the finalizers instead of atexit)."""

    def run_child(self, mode):
        report = os.path.join(self.directory, 'report.txt')
        with open(self.device, 'wb') as handle:
            handle.truncate(64 << 20)
        environ = dict(os.environ, QWEN_NVME_STOP_CHILD='%s:%s' % (mode, report), PYTHONDONTWRITEBYTECODE='1')
        child = subprocess.Popen([sys.executable, '-B', '-m', 'unittest', 'test_qwen_kv_nvme.StopChildProbe.test_child'], cwd=os.path.dirname(os.path.abspath(__file__)), env=environ,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        end = time.monotonic() + 60
        while time.monotonic() < end and not os.path.exists(report) and child.poll() is None:
            time.sleep(0.05)
        self.assertTrue(os.path.exists(report), 'the child did not get ready')
        fields = Path(report).read_text().split()
        self.assertGreater(int(fields[1]), 0, 'the writer is behind: there is something to drain')
        began = time.monotonic()
        os.kill(int(fields[2]), signal.SIGTERM)
        code = child.wait(timeout=60)
        return code, time.monotonic() - began, report

    def check_drained(self, expected_blocks=64):
        store = store_module.NvmeStore.open(self.device, format='never')
        self.addCleanup(store.close)
        stats = store.snapshot_stats()
        self.assertGreaterEqual(stats['kv_entries'], expected_blocks)
        self.assertGreaterEqual(stats['ckpt_entries'], 1)

    def test_sigterm_with_the_default_disposition_drains_then_dies_by_the_signal(self):
        code, elapsed, report = self.run_child('default')
        self.assertEqual(code, -signal.SIGTERM)
        self.assertLess(elapsed, 9)
        self.check_drained()

    def test_the_engine_core_pattern_systemexit_in_a_child_drains_in_the_finalizer(self):
        code, elapsed, report = self.run_child('engine')
        self.assertEqual(Path(report + '.exit').read_text().strip(), '0', 'SystemExit() leaves the child normally')
        self.assertLess(elapsed, 9)
        self.check_drained()


class ContractTests(unittest.TestCase):
    """serving_c2_contract: the names, the gate, the waiver."""

    ENV = {'QWEN_PREFIX_REUSE': '1', 'QWEN_PREFIX_HOST_TIER_GIB': '32', 'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': '/dev/nvme1n1', 'QWEN_PREFIX_NVME_RETENTION_ACK': '1'}

    def profile(self, name='x-gate', gate_only=False, waiver=None, **env):
        merged = dict(self.ENV, **env)
        merged = dict((key, value) for key, value in merged.items() if value is not None)
        profile = {'name': name, 'env': merged}
        if gate_only:
            profile['gate_only'] = True
        if waiver is not None:
            profile['owner_traffic_waiver'] = waiver
        return profile

    def waiver(self, levers=None):
        return {'id': 'nvme-test-2026-10-21', 'levers': levers or {'QWEN_PREFIX_HOST_TIER_GIB': '32', 'QWEN_PREFIX_NVME': '1'}, 'waives': 'the tier and the NVMe tier run unqualified',
                'evidence': ['CPU tests'], 'decision': 'PENDING the owner'}

    def nvme_problems(self, profile):
        return [problem for problem in contract.prefix_policy_problems(profile) if 'NVME' in problem or 'NVMe' in problem]

    def test_the_names_are_the_modules(self):
        self.assertEqual(contract.PREFIX_NVME_NAMES, kvnvme.NAMES)
        self.assertEqual(contract.PREFIX_NVME_GATE_ONLY, kvnvme.GATE_ONLY)
        self.assertEqual(contract.NVME_WAIVABLE_LEVERS, {'QWEN_PREFIX_NVME': '1'})

    def test_a_gate_profile_may_name_the_tier_and_a_clean_one_has_no_problem(self):
        self.assertEqual(self.nvme_problems(self.profile()), [])
        self.assertEqual(self.nvme_problems(self.profile(QWEN_PREFIX_NVME_AUDIT='1', QWEN_PREFIX_NVME_FORMAT='1', QWEN_PREFIX_NVME_ALLOW_ON_ZERO_RETENTION='1',
                                                           QWEN_PREFIX_NVME_SPILL='through')), [])

    def test_a_bad_value_the_engine_would_refuse_is_a_problem_of_the_profile(self):
        problems = self.nvme_problems(self.profile(QWEN_PREFIX_NVME_TTL_S='0'))
        self.assertTrue(any('QWEN_PREFIX_NVME_TTL_S' in problem for problem in problems), problems)
        problems = self.nvme_problems(self.profile(QWEN_PREFIX_NVME_RESTORE='p2p'))
        self.assertTrue(any('not built' in problem for problem in problems), problems)

    def test_the_acknowledgement_is_required_of_every_profile_that_names_the_tier(self):
        problems = self.nvme_problems(self.profile(QWEN_PREFIX_NVME_RETENTION_ACK=None))
        self.assertTrue(any('QWEN_PREFIX_NVME_RETENTION_ACK=1 is required' in problem for problem in problems), problems)

    def test_the_tier_needs_the_host_tier_and_the_registry(self):
        problems = self.nvme_problems(self.profile(QWEN_PREFIX_HOST_TIER_GIB=None))
        self.assertTrue(any('QWEN_PREFIX_HOST_TIER_GIB is not set' in problem for problem in problems), problems)
        profile = self.profile(QWEN_PREFIX_REUSE=None)
        self.assertTrue(any('set without QWEN_PREFIX_REUSE=1' in problem for problem in contract.prefix_policy_problems(profile)))

    def test_a_traffic_profile_names_the_tier_only_under_a_waiver_that_names_it(self):
        problems = self.nvme_problems(self.profile(name='x-traffic', waiver=self.waiver({'QWEN_PREFIX_HOST_TIER_GIB': '32'})))
        self.assertTrue(any('only when its owner_traffic_waiver names it' in problem for problem in problems), problems)
        profile = self.profile(name='x-traffic', waiver=self.waiver())
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        self.assertEqual(self.nvme_problems(profile), [])
        self.assertEqual(contract.waived_levers(profile), {'QWEN_PREFIX_HOST_TIER_GIB': '32', 'QWEN_PREFIX_NVME': '1'})

    def test_a_waiver_names_the_tier_at_its_value_and_a_stale_one_is_refused(self):
        profile = self.profile(name='x-traffic', waiver=self.waiver({'QWEN_PREFIX_HOST_TIER_GIB': '32', 'QWEN_PREFIX_NVME': '2'}))
        self.assertTrue(any('QWEN_PREFIX_NVME is waivable only at 1' in problem for problem in contract.traffic_waiver_problems(profile)))
        profile = self.profile(name='x-traffic', QWEN_PREFIX_NVME=None, QWEN_PREFIX_NVME_RETENTION_ACK=None, QWEN_PREFIX_NVME_DEVICE=None, waiver=self.waiver())
        self.assertTrue(any('names QWEN_PREFIX_NVME=1 and the profile env has None' in problem for problem in contract.traffic_waiver_problems(profile)))

    def test_the_names_that_are_a_gates_own_stay_a_gates_even_under_the_waiver(self):
        for name in contract.PREFIX_NVME_GATE_ONLY:
            value = '/models/.qwen-c2/cmd' if name.endswith('CMD_DIR') else '1'
            profile = self.profile(name='x-traffic', waiver=self.waiver(), **{name: value})
            problems = self.nvme_problems(profile)
            self.assertTrue(any(name in problem and "a gate's own" in problem for problem in problems), (name, problems))

    def test_an_inherited_value_is_dropped_under_a_profile_that_does_not_name_it(self):
        environ = dict((name, '1') for name in contract.PREFIX_NVME_NAMES)
        kept = contract.apply_environment({'env': {'QWEN_PREFIX_NVME_TTL_S': '60'}, 'engine': {}, 'mesh_graph_descriptor': '/x'}, dict(environ))
        self.assertEqual([name for name in contract.PREFIX_NVME_NAMES if name in kept], ['QWEN_PREFIX_NVME_TTL_S'])


class MarkerTests(NvmeCase):
    """The gate reads the tier's lines (prefix_markers.scan)."""

    def scan(self, made):
        return prefix_markers.scan(['[PINDIAG] prefix: ' + line for line in made.lines])

    def test_the_lines_of_a_spill_and_a_restore_are_read(self):
        made = self.rig()
        a = self.push_out(made, first_turn())
        made.serve(self.returning(a))
        scanned = self.scan(made)
        self.assertEqual(len(scanned['nvme_open']), 1)
        self.assertEqual(scanned['nvme_open'][0]['spill'], 'evict')
        self.assertEqual(len(scanned['nvme_on']), 1)
        self.assertEqual((scanned['nvme_on'][0]['backend'], scanned['nvme_on'][0]['verify']), ('host', 'all'))
        self.assertGreater(len(scanned['nvme_spills']), 0)
        self.assertEqual(sum(entry['blocks'] for entry in scanned['nvme_spills']), made.registry.snapshot()['nvme_spill_blocks'])
        self.assertEqual(len(scanned['nvme_restores']), 1)
        self.assertEqual(scanned['nvme_restores'][0]['from'], 0)
        self.assertEqual(scanned['nvme_restores'][0]['tag'], 'again-a1' if scanned['nvme_restores'][0]['tag'] else None)
        self.assertEqual(len(scanned['nvme_ckpt_restores']), 1)
        self.assertEqual(scanned['nvme_ckpt_restores'][0]['pos'], 4096)
        self.assertEqual(scanned['nvme_trouble'], [])

    def test_a_refusal_and_a_latch_are_trouble(self):
        made = NvmeRig(self.device, tier=self.TIER, num_blocks=SMALL_POOL, budget=1 << 20, environ={'THATCH_ZERO_RETENTION': '1'})
        self.addCleanup(made.close)
        scanned = self.scan(made)
        self.assertEqual((scanned['nvme_on'], len(scanned['nvme_trouble'])), ([], 1))
        self.assertIn('REFUSED', scanned['nvme_trouble'][0]['line'])
        other = self.rig()
        other.nvme.latch('test')
        self.assertIn('latched off', self.scan(other)['nvme_trouble'][0]['line'])

    def test_a_log_without_the_tier_has_empty_lists(self):
        scanned = prefix_markers.scan(['[PINDIAG] prefix: host tier on gib=1.00 kv_gib=0.50 block_bytes=32 verify=all audit=0 spill_max_blocks=512 min_tokens=0 slack=0 off_path=x'])
        for name in ('nvme_open', 'nvme_on', 'nvme_spills', 'nvme_restores', 'nvme_ckpt_restores', 'nvme_trouble'):
            self.assertEqual(scanned[name], [])
        self.assertEqual(len(scanned['tier_on']), 1)


class MetricsTests(NvmeCase):
    def test_the_snapshot_carries_every_counter_and_gauge_only_while_the_tier_is_on(self):
        made = self.rig()
        self.push_out(made, first_turn())
        snapshot = made.registry.snapshot()
        for name in kvnvme.COUNTER_NAMES + kvnvme.GAUGE_NAMES:
            self.assertIn(name, snapshot)
        self.assertEqual(snapshot['nvme_on'], 1)
        self.assertGreater(snapshot['nvme_spill_bytes'], 0)
        self.assertGreater(snapshot['nvme_capacity_bytes'], snapshot['nvme_bytes'])
        self.assertIn('nvme_spilled=', made.registry.telemetry_lines(host={})[0])

    def test_the_metrics_module_documents_every_nvme_counter(self):
        import qwen_prefix_metrics as metrics

        for name in kvnvme.COUNTER_NAMES:
            self.assertIn(name, metrics.COUNTERS, name)
        for name in kvnvme.GAUGE_NAMES:
            self.assertIn(name, metrics.GAUGES, name)

    def test_a_latched_tier_says_so(self):
        made = self.rig()
        made.nvme.latch('test')
        snapshot = made.registry.snapshot()
        self.assertEqual((snapshot['nvme_on'], snapshot['nvme_latched']), (0, 1))


class RandomTrafficTests(NvmeCase):
    def run_traffic(self, seed, nvme):
        rng = random.Random(seed)
        if nvme:
            made = self.rig(tier=60, num_blocks=170)
        else:
            made = Rig(num_blocks=170, budget=1 << 20, tier=60)
            self.addCleanup(made.close)
        systems = {name: span(2300, 500 + index) for index, name in enumerate(('alpha', 'bravo', 'charlie'))}
        histories, rows = {}, []
        for step in range(36):
            name = rng.choice(['alpha', 'bravo', 'charlie'])
            convo = rng.randrange(2)
            key = (name, convo)
            base = histories.get(key, systems[name])
            tokens = list(base) + span(rng.randrange(1200, 3200), 4000 + 31 * step + convo)
            start, grant = made.serve(Request('r%d' % step, tokens, salt=salt_of(name)))
            rows.append((start, grant.q if grant else 0))
            made.check_kv()
            if nvme and step % 6 == 5:
                made.settle()
            histories[key] = None if len(tokens) > 7000 else tokens + span(rng.randrange(50, 400), 9000 + step)
            if histories[key] is None:
                del histories[key]
        if nvme:
            made.settle()
        return rows, made.registry.snapshot()

    def test_random_traffic_is_exact_and_the_nvme_only_adds_hits(self):
        total_restored = 0
        for seed in range(4):
            with_rows, with_stats = self.run_traffic(seed, nvme=True)
            without_rows, without_stats = self.run_traffic(seed, nvme=False)
            self.assertEqual(len(with_rows), len(without_rows))
            self.assertEqual(with_stats['nvme_digest_failures'], 0)
            self.assertEqual(with_stats['nvme_spill_failures'], 0)
            self.assertGreaterEqual(with_stats['grant_tokens'], without_stats['grant_tokens'], 'seed %d: the NVMe never lowers the hits' % seed)
            total_restored += with_stats['nvme_restore_blocks']
            self.rigs[-1].close()
        self.assertGreater(total_restored, 0, 'the traffic returned sessions from the drive')


if __name__ == '__main__':
    unittest.main()
