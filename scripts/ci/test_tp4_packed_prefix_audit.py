"""tp4/packed-prefix: the sticky-session writer audit extended to the four-card eight-seat levers (docs/tp4-packed-prefix.md).

docs/sticky-sessions-writer-audit.md shows, for the pair's S2 path, that every device write of a resumed request lands in
the request's own blocks at or above R or outside vLLM's KV pool. The four-card eight-seat profiles add levers and shapes
the pair's audit never saw. This module holds what can run on the CPU:

  inventory  none of the TP4 levers (the fused commit in place on live banks, the GDN commit lanes, the draft K/V slide, the
             quad drafts at two blocks, the verify-glue levers, the attention fold) names a call that writes the target K/V
             pool: the ordered K/V writers are the pool's only writers, and a lever that grows one must be audited first;
  frontier   a verify round at a 262k position writes only pages at or above the frontier, through a 4,096-wide table, with
             the shared prefix [0, R) below it;
  guard      two same-tenant blocks of four seats share their first block: every seat of both blocks crossing a 64-token
             boundary together is not a K/V conflict under sticky sessions, a real conflict across a block's seats still is,
             and the sentinel pad keeps its meaning at width 4,096.

The hardware half is the exactness-shared arm on eight agents (E2 of references/tp4-packed-prefix-jobs): per-window KV digests
of every window two requests share must agree across the decode."""

import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import serving_packed_step  # noqa: E402
import serving_page_binding  # noqa: E402
import verify_trace_t2  # noqa: E402

STICKY = 'QWEN_FAST_STICKY_SESSIONS'
BLOCK = 64
# The four-card levers and the twins they run through: every module that exists for QWEN_FAST_TP=4 and is not a K/V pool writer.
LEVER_MODULES = (
    'fused_commit_tp.py',              # QWEN_FAST_FUSED_COMMIT (+_INPLACE, _LIVE_BANKS): the drafter's K/V banks
    'draft_kv_slide_tp.py',            # QWEN_FAST_TP_KV_SLIDE: the drafter's sliding history
    'draft_kv_history_tp.py',          # the drafter's history cache
    'quad_draft_tp.py',                # QWEN_FAST_QUAD_DRAFT (+_BLOCKS=2): the four-user drafts
    'gdn_commit_dma_tp.py',            # QWEN_FAST_TP4_COMMIT_LANES: the GDN state publication
    'gdn_rows_dma_tp.py',              # QWEN_FAST_TP4_GDN_GLUE
    'tp4_vglue.py',                    # the verify-glue levers' flags and markers
    'attention_block_fold_tp.py',      # QWEN_FAST_TP4_ATTN_FOLD: data movement on the query and the result
    'extent_attention_fold_tp.py',     # the attention reader with the fold
    'extent_attention_replay_tp.py',   # the extent reader (QWEN_FAST_SDPA_MODES tail,share): reads the pool through page tables
    'dflash_packed_proposal_coordinator.py',
)
# A call that writes the target K/V pool: the ordered writers and the stock cache updates.
POOL_WRITE = re.compile(r'paged_update_cache|paged_fill_cache|fill_cache\(|ordered_cache\.update|ordered_cache_tp\.update'
                        r'|packed_ordered_cache\.|SegmentedOrderedCacheWriter')


def blocks_of(start, count):
    return list(range(start, start + count))


def engine_with(blocks, width):
    pages = torch.full((1, width), blocks[0], dtype=torch.int32)
    pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return SimpleNamespace(pages=pages)


class FakeBlock(object):
    def __init__(self, engines, rows=16):
        self.engines = list(engines)
        self.shape = SimpleNamespace(users=len(self.engines), rows_per_user=rows)

    def segment_of(self, engine):
        for index, candidate in enumerate(self.engines):
            if candidate is engine:
                return index
        raise ValueError('engine not bound')


class sticky(object):
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.patch = mock.patch.dict(os.environ)
        self.patch.start()
        os.environ.pop(STICKY, None)
        if self.value is not None:
            os.environ[STICKY] = self.value

    def __exit__(self, *args):
        self.patch.stop()


class InventoryTests(unittest.TestCase):
    def test_no_tp4_lever_names_a_call_that_writes_the_target_kv_pool(self):
        for name in LEVER_MODULES:
            with self.subTest(module=name):
                path = HERE / name
                self.assertTrue(path.is_file(), name)
                found = POOL_WRITE.findall(path.read_text(encoding='utf-8'))
                self.assertEqual(found, [], '%s names a K/V pool write: audit it (docs/tp4-packed-prefix.md) before adding it '
                                           'to a sticky-session profile' % name)

    def test_the_pool_writers_are_the_ordered_cache_family_and_nothing_a_lever_adds(self):
        for name in ('ordered_cache_tp.py', 'packed_ordered_cache.py', 'packed_cache_writer.py'):
            self.assertTrue((HERE / name).is_file(), name)
        self.assertTrue(POOL_WRITE.search((HERE / 'packed_cache_writer.py').read_text(encoding='utf-8')),
                        'the pattern is live: the writer module matches it')


class FrontierTests(unittest.TestCase):
    """A verify round at a 262k position, through a 4,096-wide table: the writes are the rows [position, position + 16)."""

    R, P = 249856, 253920                 # R = a 2048 multiple (122 chunks); P the 262k profile's prompt limit
    POOL = 19968

    def binding(self, blocks):
        pages = torch.full((1, 4096), blocks[0], dtype=torch.int32)
        pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        binding = object.__new__(serving_page_binding.VerifierPageBinding)
        binding.engine = SimpleNamespace(phase='idle', pages=pages)
        binding.operations, binding.mesh = Mock(), object()
        binding.physical_pages, binding.failed, binding.capacity = self.POOL, False, 4096
        binding.blocks, binding.bindings = tuple(blocks), {}
        return binding

    def test_rows_at_the_frontier_never_touch_the_shared_prefix_or_the_pad_at_width_4096(self):
        shared = blocks_of(100, self.R // BLOCK)                       # vLLM's cached prefix [0, R): 3,904 blocks
        private = blocks_of(5000, (self.P - self.R + BLOCK - 1) // BLOCK)
        blocks = shared + private
        self.assertLess(len(blocks), 4096)
        binding = self.binding(blocks)
        for position in range(self.P, self.P + 256, 16):
            needed = (position + 16 + BLOCK - 1) // BLOCK
            if needed > len(blocks):
                with self.assertRaisesRegex(ValueError, 'must cover every scheduled verifier row'):
                    binding.refresh(blocks, position=position, rows=16)
                blocks = blocks + [9000 + len(blocks)]
            binding.refresh(blocks, position=position, rows=16)
            rows = verify_trace_t2.kv_tile_rows(range(position, position + 16), binding.engine.pages[0].tolist())
            pages = {page for page, _ in rows}
            self.assertFalse(pages & set(shared), position)
            self.assertNotIn(shared[0], pages, 'never the pad')
            self.assertNotIn(0, pages)

    def test_the_sentinel_keeps_its_meaning_at_width_4096(self):
        blocks = blocks_of(100, 3904) + blocks_of(5000, 63)
        table = engine_with(blocks, 4096).pages
        mapped = serving_packed_step.pad_sentinel_table(table, 5)
        self.assertEqual(len(mapped[0]), 4096)
        self.assertEqual(mapped[0][:len(blocks)], blocks)
        self.assertEqual(set(mapped[0][len(blocks):]), {-6})


class TwoBlockGuardTests(unittest.TestCase):
    """Eight seats on two packed 64-row blocks (QWEN_FAST_M3_BLOCKS=2): the guard runs once per block, over the block's own
    four segments (segment_of)."""

    def tenant_block(self, base, width=4096):
        # four same-tenant requests sharing their cached first block, each with its own private blocks
        first = base
        return [engine_with([first] + blocks_of(base + 10 + 100 * seat, 2), width) for seat in range(4)]

    def test_every_seat_of_both_blocks_crossing_a_boundary_together_is_not_a_conflict(self):
        for width in (68, 4096):
            with self.subTest(width=width):
                for base in (7, 3000):
                    engines = self.tenant_block(base, width)
                    block = FakeBlock(engines)
                    owners = [(engine, 184) for engine in engines]          # rows 184..199: the last eight past the bound
                    with sticky(None):
                        self.assertIn('kv tile rows shared', serving_packed_step.kv_guard(owners, block) or '')
                    with sticky('1'):
                        self.assertIsNone(serving_packed_step.kv_guard(owners, block))

    def test_a_real_conflict_inside_a_block_is_caught_at_width_4096(self):
        engines = self.tenant_block(7, 4096)
        engines[1].pages[0, 1] = engines[0].pages[0, 1]           # seat 1's table names seat 0's private block
        with sticky('1'):
            reason = serving_packed_step.kv_guard([(engine, 70) for engine in engines], FakeBlock(engines))
        self.assertIn('kv tile rows shared: users 0,1', reason or '')

    def test_the_two_blocks_are_guarded_separately_and_share_only_reads(self):
        first, second = self.tenant_block(7), self.tenant_block(7)         # the same tenant's first block in both blocks
        with sticky('1'):
            for engines in (first, second):
                self.assertIsNone(serving_packed_step.kv_guard([(engine, 184) for engine in engines], FakeBlock(engines)))
        # the sentinel is per segment, so equal segments of the two blocks map their pads alike and nothing crosses blocks
        self.assertEqual(serving_packed_step.pad_sentinel_table(first[2].pages, 2),
                         serving_packed_step.pad_sentinel_table(second[2].pages, 2))


if __name__ == '__main__':
    unittest.main()
