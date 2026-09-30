"""The four-card publication warm and per-request proposal capture, run for real on a shape-checking fake ttnn.

The second review of the S2 TP4 port found two served modules that no test reached and that died at four cards on the
pair's literals: publication_warm (the extent block's attach-time warm: a (1, 4, 2048, 128) K/V bank and a 2,048-wide
query) and dflash_proposal_trace (every request's proposal capture: the drafter's cached K/V banks and the bank slices).
The attach simulation fakes both, and the torch fakes of test_publication_warm replace the K/V projection and the shard
mapper with width-blind stand-ins, so nothing on CPU could have failed. Here QWEN_FAST_TP=4 is set under the c2-packed-tp4
profile's real environment, the tp_addresses seam is installed as serving_startup.start installs it, and

  - publication_warm.warm runs its whole plan (64 packed and 7 sequential publications) through today's DFlashDevice
    publication code, the four-card DraftKVHistory, the feature projection's tap packing and gather, and the K/V
    projection twin, over tp4_shape_fake, which refuses any shape the device would refuse;
  - the four-card DraftKVHistory is built by its own constructor over pool-lent banks and query at the four-card shapes;
  - PreparedDFlashProposal is constructed by its own code (every context bucket's uploads, the warm-up pass, the capture)
    and updated (the bank slices into the bucket's cached K/V), around a proposal pass that holds the drafter's own cached
    K/V and GQA checks (draft_attention.validate_attention, through the seam).

What stays a stand-in: the drafter's forward (execute_proposal) - its arithmetic is a card's job (HW-B) - and the weights.
Nothing here touches a card."""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import dflash_device  # noqa: E402,F401
import dflash_packed_proposal  # noqa: E402,F401
import draft_attention  # noqa: E402
import draft_kv_history  # noqa: E402,F401
import draft_kv_history_tp  # noqa: E402
import draft_kv_slide_tp  # noqa: E402
import feature_projection_tp  # noqa: E402
import mesh_link_policy  # noqa: E402
from packed_shapes import m3_shape  # noqa: E402
import publication_warm  # noqa: E402
import tp_shapes  # noqa: E402
from dflash_device import DFlashDevice  # noqa: E402
from dflash_proposal_trace import PreparedDFlashProposal  # noqa: E402
from test_tp4_attach_profile import environment as profile_environment  # noqa: E402
from tp4_shape_fake import Collectives, ShapeError, ShapeOps, Tensor  # noqa: E402
from tp_test_support import four_cards  # noqa: E402

PROFILES = ('c2-packed-tp4', 'c2-packed-tp4-gate')
# The profiles that run the eager publication chain (QWEN_FAST_TP_KV_SLIDE=0): the fake ttnn runs it for real; the slide's launch is a
# card's job, so under the profiles that switch it on the transport is a shape-checking stand-in (test_draft_kv_slide_tp holds the
# launch itself and the slide's bytes against the eager chain).
EAGER_PROFILES = ('c2-packed-tp4-gate-noslide', 'c2-packed-tp4-speed-noslide')
SLIDES = []


def checking_transport(mesh, active, delta, spare, *, history_rows, prefix):
    """draft_kv_slide_tp.prepare's contract on the shape fake: refuse what the launch would refuse, record the call."""
    draft_kv_slide_tp.geometry(history_rows, prefix)
    for value, expected in ((active, draft_kv_slide_tp.bank_shape()), (delta, draft_kv_slide_tp.delta_shape()),
                            (spare, draft_kv_slide_tp.bank_shape())):
        if tuple(value.shape) != expected:
            raise ShapeError('slide operand %r, wanted %r' % (tuple(value.shape), expected))
    SLIDES.append((history_rows, prefix))
    return lambda: None
WIDTH = 2052
LAYERS = 5
RING_DESCRIPTOR = HERE / 'qwen_p150x4_ring_mesh_graph_descriptor.textproto'
MESH = SimpleNamespace(shape=[1, 4])


def served(name):
    """The launched process at QWEN_FAST_TP=4 under `name`'s real environment, with the seam installed."""
    from contextlib import ExitStack, contextmanager

    @contextmanager
    def enter():
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, profile_environment(name), clear=True))
            stack.enter_context(four_cards())
            stack.enter_context(patch.object(draft_kv_slide_tp, 'prepare', checking_transport))
            del SLIDES[:]
            # The ring descriptor lives in the image; the committed copy has the audited bytes (test_tp4_attach_profile).
            real_audit = mesh_link_policy.audit_descriptor
            stack.enter_context(patch.object(mesh_link_policy, 'audit_descriptor',
                lambda env, shape, read=None: real_audit(env, shape, read=lambda path: RING_DESCRIPTOR.read_bytes())))
            mesh_link_policy._ring_projection_links.cache_clear()
            yield
    return enter()


def weights(ops):
    """The drafter's shared weights at the four-card widths: the fc projection is the real projection_shards of a
    (5120, 5 x 5120) weight, one (6,400, 5,120) matrix per chip; each layer's K/V projections are 5,120 by the chip's two
    KV heads."""
    fc = torch.empty((tp_shapes.HIDDEN, 5 * tp_shapes.HIDDEN), device='meta')
    with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
        per_chip = feature_projection_tp.projection_shards(fc)
        kv = tp_shapes.active().draft_kv_heads * 128
    assert len(per_chip) == 4 and tuple(per_chip[0].shape) == (6400, 5120), tuple(per_chip[0].shape)
    projection = Tensor(ops, per_chip[0].shape, ops.bfloat16)
    norm = Tensor(ops, (1, 1, 1, 5120), ops.bfloat16)
    kernel = ops.WormholeComputeKernelConfig(math_fidelity='hifi4')
    layers = [(dict(operations=ops, native_head_layout=True, kernel=kernel,
                    projections=dict(k=Tensor(ops, (5120, kv), ops.bfloat16), v=Tensor(ops, (5120, kv), ops.bfloat16)),
                    head_norms=dict(k=Tensor(ops, (1, 1, 1, 128), ops.bfloat16))), 'mlp', 'weights', 'convolution')
              for layer in range(LAYERS)]
    return SimpleNamespace(closed=False, layers=layers, projection=projection, feature_norm=norm,
                           tensors=[projection, norm])


def block(users=4):
    shape = m3_shape(WIDTH)
    return SimpleNamespace(extent=True, shape=shape, users=shape.users, rows_per_user=shape.rows_per_user,
                           block_rows=shape.block_rows, taps=(), segment_slots=())


class WarmTests(unittest.TestCase):
    def run_warm(self, name):
        ops = ShapeOps(4)
        shared = weights(ops)
        held = set(ops.live)
        lines, allocated = [], []
        original = publication_warm.Scratch.zeros

        def zeros(self, shape, **keywords):
            value = original(self, shape, **keywords)
            allocated.append(value)
            return value

        with served(name), patch.object(publication_warm.Scratch, 'zeros', zeros):
            summary = publication_warm.warm(block(), operations=ops, mesh=MESH, pool=SimpleNamespace(bucket_rows=(1, 2, 4)),
                                            shared_weights=shared, collectives=Collectives(), log=lines.append)
        return ops, held, summary, lines, allocated

    def test_the_whole_plan_publishes_at_the_four_card_shapes_under_both_profiles(self):
        for name in PROFILES:
            with self.subTest(profile=name):
                ops, held, summary, lines, allocated = self.run_warm(name)
                self.assertIsNotNone(summary, lines)
                self.assertEqual(summary['shapes'], 71)
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith(publication_warm.MARKER), lines)
                # The scratch is the served width's: a 2-KV-head bank pair, a 1,024-wide query, 1,280-column taps.
                self.assertEqual([tuple(value.shape) for value in allocated[:5]],
                                 [(1, 1, 32, 1024), (1, 2, 2048, 128), (1, 2, 2048, 128), (1, 1, 2048, 5120),
                                  (1, 1, 2048, 5120)])
                self.assertEqual({tuple(value.shape) for value in allocated[5:]},
                                 {(1, 1, 64, 1280), (1, 1, 1, 1280), (1, 1, 2, 1280), (1, 1, 4, 1280)})
                # Every publication gathered its projection over the four chips, and nothing leaked or freed twice.
                gathers = [event for event in ops.events if event[0] == 'all_gather']
                self.assertGreaterEqual(len(gathers), 71)
                self.assertEqual(ops.live, held, 'the warm released exactly what it allocated')
                # Every publication slid each layer's k and v at the steady state: 71 publications x 5 layers x 2
                self.assertEqual(len(SLIDES), 71 * LAYERS * 2)
                self.assertEqual({rows for rows, _ in SLIDES}, {2048})

    def test_the_eager_chain_profiles_publish_the_same_plan_without_the_slide(self):
        for name in EAGER_PROFILES:
            with self.subTest(profile=name):
                ops, held, summary, lines, allocated = self.run_warm(name)
                self.assertIsNotNone(summary, lines)
                self.assertEqual(summary['shapes'], 71)
                self.assertEqual(SLIDES, [], 'QWEN_FAST_TP_KV_SLIDE=0 runs the eager six-op chain')
                self.assertEqual(ops.live, held)

    def test_the_fake_refuses_the_pairs_shapes_on_the_four_card_path(self):
        """Control: with the two literals the review found put back, the same run fails."""
        ops = ShapeOps(4)
        shared = weights(ops)
        pair_kv, pair_query = (1, 4, 2048, 128), (1, 1, 32, 2048)
        with served('c2-packed-tp4'), patch.object(publication_warm, 'kv_shape', lambda: pair_kv), \
                patch.object(publication_warm, 'query_shape', lambda: pair_query):
            with self.assertRaises((ShapeError, ValueError)):
                publication_warm.warm(block(), operations=ops, mesh=MESH, pool=SimpleNamespace(bucket_rows=(1,)),
                                      shared_weights=shared, collectives=Collectives(), log=lambda text: None)

    def test_the_pairs_class_would_not_have_served_the_four_card_warm(self):
        ops = ShapeOps(4)
        shared = weights(ops)
        with served('c2-packed-tp4'), patch.object(draft_kv_history_tp, 'DraftKVHistory', draft_kv_history.DraftKVHistory):
            with self.assertRaises((ShapeError, ValueError)):
                publication_warm.warm(block(), operations=ops, mesh=MESH, pool=SimpleNamespace(bucket_rows=(1,)),
                                      shared_weights=shared, collectives=Collectives(), log=lambda text: None)


def lent_banks(ops):
    kv = (1, tp_shapes.geometry(4).draft_kv_heads, 2048, 128)
    return [{side: {head: Tensor(ops, kv, ops.bfloat16) for head in ('k', 'v')} for side in ('active', 'spare')}
            for layer in range(LAYERS)]


def lent_query(ops):
    return Tensor(ops, (1, 1, 32, tp_shapes.geometry(4).draft_query), ops.bfloat16)


class DraftCacheTests(unittest.TestCase):
    def test_the_four_card_cache_is_built_by_its_own_constructor_over_lent_banks_and_query(self):
        ops = ShapeOps(4)
        shared = weights(ops)
        parameters = [layer[0] for layer in shared.layers]
        history = Tensor(ops, (1, 1, 2048, 5120), ops.bfloat16)
        with served('c2-packed-tp4'):
            cache = draft_kv_history_tp.DraftKVHistory(ops, MESH, parameters, history, position=4096, history_rows=2048,
                                                       storage=lent_banks(ops), query=lent_query(ops))
            self.assertEqual(len(cache.active), LAYERS)
            self.assertTrue(all(tuple(bank['k'].shape) == (1, 2, 2048, 128) for bank in cache.active))
            # A publication at each prefix the packed block takes, and the commit that swaps the banks.
            for prefix in (1, 7, 16, 32):
                features = Tensor(ops, (1, 1, 32, 5120), ops.bfloat16)
                publication = cache.prepare(features, prefix, position=cache.position)
                cache.discard(publication)
            # the slide published each layer's k and v at every prefix, over the four-card operand shapes
            self.assertEqual(SLIDES, [(2048, prefix) for prefix in (1, 7, 16, 32) for _ in range(LAYERS * 2)])

    def test_pool_lent_banks_at_the_pairs_shape_are_refused_at_four_cards(self):
        ops = ShapeOps(4)
        shared = weights(ops)
        parameters = [layer[0] for layer in shared.layers]
        pair_banks = [{side: {head: Tensor(ops, (1, 4, 2048, 128), ops.bfloat16) for head in ('k', 'v')}
                       for side in ('active', 'spare')} for layer in range(LAYERS)]
        with served('c2-packed-tp4'), self.assertRaises(ValueError):
            draft_kv_history_tp.DraftKVHistory(ops, MESH, parameters, Tensor(ops, (1, 1, 2048, 5120), ops.bfloat16),
                                               position=4096, history_rows=2048, storage=pair_banks,
                                               query=lent_query(ops))


class ProposalCaptureTests(unittest.TestCase):
    """PreparedDFlashProposal at four cards: its own constructor and update() over the four-card cache, around a proposal
    pass that keeps the drafter's cached-K/V and GQA checks."""

    def drafter(self, ops, position=4096, block_rows=16):
        shared = weights(ops)
        parameters = [layer[0] for layer in shared.layers]
        banks = lent_banks(ops)
        query = lent_query(ops)
        cache = draft_kv_history_tp.DraftKVHistory(ops, MESH, parameters, Tensor(ops, (1, 1, 2048, 5120), ops.bfloat16),
                                                   position=position, history_rows=2048, storage=banks, query=query)
        device = object.__new__(DFlashDevice)
        device.__dict__.update(
            operations=ops, mesh=MESH, collectives=Collectives(), name='proposal capture', layers=tuple(shared.layers),
            owned=[], borrowed=list(shared.tensors), closed=False, pending=None, position=position, history_rows=2048,
            block_rows=block_rows, history=Tensor(ops, (1, 1, 2048, 5120), ops.bfloat16),
            spare_history=Tensor(ops, (1, 1, 2048, 5120), ops.bfloat16), kv_history=cache, progress=None,
            pool_slot=None, shared_weights=None)
        return device

    def proposal_pass(self, calls):
        found = tp_shapes.geometry(4)

        def execute(device, identifiers, history, mask, rope, *, context, owned, retain, stage, audit=True,
                    cached_history=None, **keywords):
            ops = device.operations
            self.assertIsNotNone(cached_history)
            self.assertEqual(len(cached_history), LAYERS)
            query = Tensor(ops, (1, found.draft_heads, 32, 128), ops.bfloat16)
            live = Tensor(ops, (1, found.draft_kv_heads, 32, 128), ops.bfloat16)
            for cache in cached_history:
                self.assertEqual({name: tuple(cache[name].shape) for name in ('k', 'v')},
                                 {name: (1, found.draft_kv_heads, context, 128) for name in ('k', 'v')})
                key = retain(ops.concat([cache['k'], live], dim=2))
                value = retain(ops.concat([cache['v'], live], dim=2))
                draft_attention.validate_attention(ops, query, key, value, mask)
            calls.append(context)
            return SimpleNamespace(chunks=[], projected=None)
        return execute

    def test_the_capture_builds_and_replays_at_two_kv_heads_per_chip(self):
        for name in PROFILES:
            with self.subTest(profile=name):
                ops, calls = ShapeOps(4), []
                with served(name), patch.object(DFlashDevice, 'execute_proposal', self.proposal_pass(calls)):
                    device = self.drafter(ops)
                    capture = PreparedDFlashProposal(device, max_new_tokens=1024)
                    self.assertTrue(capture.buckets)
                    for bucket in capture.buckets.values():
                        self.assertEqual([tuple(value.shape) for layer in bucket.cached_history for value in layer.values()],
                                         [(1, 2, bucket.context, 128)] * (2 * LAYERS))
                    # The eager warm-up pass and the traced capture each ran the pass once per bucket, and a request's
                    # per-round update copies the four-card banks in.
                    self.assertEqual(sorted(calls), sorted(list(capture.buckets) * 2))
                    for bucket in capture.buckets.values():
                        capture.update(bucket, 3)
                        owned = capture.update(bucket, 5, defer_finish=True)
                        dflash_device_release(ops, owned)
                    capture.close()

    def test_the_pairs_four_kv_head_bank_would_have_failed_the_same_capture(self):
        """Control: the capture's own literals put back to the pair's are refused by the fake."""
        ops, calls = ShapeOps(4), []
        with served('c2-packed-tp4'), patch.object(DFlashDevice, 'execute_proposal', self.proposal_pass(calls)):
            device = self.drafter(ops)
            real = tp_shapes.geometry(4)._replace(draft_kv_heads=4)
            with patch('dflash_proposal_trace.tp_shapes.active', return_value=real):
                with self.assertRaises((ShapeError, AssertionError, ValueError)):
                    PreparedDFlashProposal(device, max_new_tokens=1024)


def dflash_device_release(ops, owned):
    from gdn_multitoken_conv import release_owned

    release_owned(ops, owned)


if __name__ == '__main__':
    unittest.main()
