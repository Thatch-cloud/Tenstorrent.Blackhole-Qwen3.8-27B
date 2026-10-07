"""tp4/packed-prefix: the sticky-session writer audit extended to the four-card eight-seat levers (docs/tp4-packed-prefix.md).

docs/sticky-sessions-writer-audit.md shows, for the pair's S2 path, that every device write of a resumed request lands in
the request's own blocks at or above R or outside vLLM's KV pool. The four-card eight-seat profiles add levers and shapes
the pair's audit never saw. This module holds what can run on the CPU:

  inventory  none of the TP4 levers (the fused commit in place on live banks, the GDN commit lanes, the draft K/V slide, the
             quad drafts at two blocks, the verify-glue levers, the attention fold) names a call that writes the target K/V
             pool: the ordered K/V writers are the pool's only writers, and a lever that grows one must be audited first;
  frontier   a verify round at a 262k position writes only pages at or above the frontier, through a 4,096-wide table, with
             the shared prefix [0, R) below it;
  census     every device-write call site of those modules (a copy of any kind, a host-to-device copy, a generic_op DMA kernel), found
             by the syntax tree and not by a name pattern, is on an audited table of what it writes; a new site fails until audited;
             and no write-site module names the K/V pool at all;
  null       page 0, vLLM's null block, is what an idle segment of a padded round maps to and is refused anywhere in a live table;
  width      at attach a model whose chunk-input page table is not at the warmed width is refused under sticky sessions (a hit would
             compile at a new shape after the traces are parked), and nothing changes with the switches off;
  guard      two same-tenant blocks of four seats share their first block: every seat of both blocks crossing a 64-token
             boundary together is not a K/V conflict under sticky sessions, a real conflict across a block's seats still is,
             and the sentinel pad keeps its meaning at width 4,096.

The hardware half is the exactness-shared arm on eight agents (E2 of references/tp4-packed-prefix-jobs): per-window KV digests
of every window two requests share must agree across the decode."""

import ast
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

import packed_verifier  # noqa: E402
import serving_packed_step  # noqa: E402
import serving_runtime  # noqa: E402
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
    # tp4/w1 (ship/262k-prefix): the wave-1 device writers, none of which names the pool; their write sites are classified below
    'tp4_draft_conv.py',               # QWEN_FAST_TP4_DRAFT_CONV: the drafter's fused convolution into a fresh output (+ draft_conv_io_fast.cpp, draft_conv_out.cpp)
    'draft_convolution_fused_tp.py',   # the served fused convolution the lever is audited against
    'tp4_draft_heads.py',              # QWEN_FAST_TP4_DRAFT_HEADS: the drafter's head split and merge as whole-tile copies (+ draft_heads_copy.cpp)
    'draft_head_layout_tp.py',         # the head layout the tile copies follow
    'verify_prestage.py',              # QWEN_FAST_TP4_*PRESTAGE*, _ENTRY_DIET, _WINDOW_VALIDATE: host-side staging; its writes go through packed_verifier.write_packed
    'tile_collective_tp.py',           # QWEN_FAST_TP4_RS_UNIT_MAJOR and the ring topology: reduce-scatter over activations into fresh outputs
    'dflash_proposal_trace.py',        # the drafter's proposal trace: its masks, history and cache are the drafter's own
    'serving_page_binding.py',         # the verifier's page-table refresh: the table the ordered writers address through
    'packed_verifier.py',              # write_packed: the verifier's staged inputs (tokens, positions, page tables)
    # tp4/w2: the wave-2 attention and conv-gates levers (QWEN_FAST_TP4_SDPA=multi, QWEN_FAST_TP4_CONV_GATES_SPREAD); write sites classified below
    'sdpa_multi_tp.py',                # the one-launch block attention: its own stacked table, cur_pos and mask (+ the SDPA audit's counters)
    'sdpa_long_tp.py',                 # the SDPA mode selector and the grid configurations: no write site
    'gdn_conv_gates_spread.py',        # QWEN_FAST_TP4_CONV_GATES_SPREAD: the block conv-gates launch on spread gate cores
    'gdn_block_conv_tp.py',            # QWEN_FAST_TP4_GDN_BLOCK_CONV: the block conv stage that calls the spread launch (no write site)
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


WRITE_CALLS = ('copy', 'copy_host_to_device_tensor', 'generic_op')
# Every device-write call site of the lever modules, by (module, enclosing function, call): what it writes. A generic_op takes the
# tensors its caller built (the GDN state lanes, the drafter banks, the query and result staging), so its census row fixes the SITE;
# which tensors a caller passes is the hardware half (E2's window digests).
AUDITED_WRITE_SITES = {
    ('fused_commit_tp.py', 'project', 'copy'): 'the drafter K/V banks (the fused commit)',
    ('draft_kv_slide_tp.py', 'prepare', 'generic_op'): 'the drafter K/V history (the slide DMA)',
    ('draft_kv_history_tp.py', '__init__', 'copy'): 'the drafter history banks',
    ('draft_kv_history_tp.py', 'prepare', 'copy'): 'the drafter history spare banks',
    ('quad_draft_tp.py', '_update', 'copy_host_to_device_tensor'): 'the quad draft\'s staged host payload',
    ('quad_draft_tp.py', '_update.copy_cache', 'copy'): 'the quad draft\'s own cache',
    ('gdn_commit_dma_tp.py', 'prepare.execute', 'generic_op'): 'the GDN state commit lanes',
    ('gdn_rows_dma_tp.py', 'launch', 'generic_op'): 'the GDN rows glue',
    ('attention_block_fold_tp.py', '_launch', 'generic_op'): 'the attention fold\'s query and result staging',
    ('extent_attention_replay_tp.py', 'stage', 'copy_host_to_device_tensor'): 'the extent reader\'s staged positions and page tables',
    # tp4/w1 (ship/262k-prefix). Classified: none of these addresses the target K/V pool. A conv output is a fresh ttnn.empty that the
    # call refuses to alias with its inputs; the heads tile copy writes the drafter's own q/k/v tensors; the drafter caches are the
    # drafter's banks; the staged verifier inputs and page tables are refreshed at execute time (page_binding.refresh) and checked by
    # WINDOW_VALIDATE. The hazard left is a stale table naming the user's first block, which under prefix caching is shared: that is
    # the E2 exactness-shared arm's window digests, run on the ship-prefix-audit profile.
    ('tp4_draft_conv.py', 'convolution', 'generic_op'): 'a fresh convolution output of the drafter (the fast conv)',
    ('draft_convolution_fused_tp.py', 'served_fused_convolution', 'generic_op'): 'a fresh convolution output of the drafter (the served conv)',
    ('tp4_draft_heads.py', 'launch', 'generic_op'): 'the drafter\'s head split and merge outputs (whole-tile copies)',
    ('quad_draft_tp.py', 'served_quad_fused_convolution', 'generic_op'): 'a fresh convolution output of the quad draft',
    ('dflash_proposal_trace.py', 'borrow_pooled_mask', 'copy_host_to_device_tensor'): 'the drafter\'s pooled attention mask',
    ('dflash_proposal_trace.py', 'publish_outputs', 'copy'): 'the drafter\'s proposal outputs',
    ('dflash_proposal_trace.py', 'update', 'copy_host_to_device_tensor'): 'the drafter\'s staged host payload',
    ('dflash_proposal_trace.py', 'update.copy_history_and_cache', 'copy'): 'the drafter\'s history and cache',
    ('dflash_proposal_trace.py', '_update', 'copy_host_to_device_tensor'): 'the drafter\'s staged host payload',
    ('dflash_proposal_trace.py', '_update.copy_cache', 'copy'): 'the drafter\'s own cache',
    ('serving_page_binding.py', 'refresh', 'copy_host_to_device_tensor'): 'the verifier\'s page table (the address the ordered writers use)',
    ('packed_verifier.py', 'write_packed', 'copy_host_to_device_tensor'): 'the verifier\'s staged tokens, positions and page tables',
    # tp4/w2. Classified: multi gathers each user's lent table row into its own stacked table in-trace and reads the pool through it (the
    # pool is never a write destination: its programs are readers, exactness item 3 of sdpa_multi_tp); the two refresh launches write the
    # stacked table, cur_pos and the mask; the audit's counter pages are its own. F1's launch writes fresh conv, beta and g outputs and
    # advances the BLOCK windows in place exactly as the served op does (the block windows are copies of the users' windows made by
    # rows_dma), and the audit copies the same windows into shared scratch. Neither module names the pool.
    ('sdpa_multi_tp.py', 'refresh', 'generic_op'): 'multi\'s own stacked table, cur_pos and mask (the gather and mask launches)',
    ('sdpa_multi_tp.py', 'call', 'generic_op'): 'the SDPA audit\'s own counter pages',
    ('gdn_conv_gates_spread.py', 'launch', 'generic_op'): 'fresh conv, beta and g outputs and the block windows advanced in place as the served op does',
    ('gdn_conv_gates_spread.py', 'launch', 'copy'): 'the audit\'s shared scratch windows',
}
# Sites that make more than one write call (the drafter proposal trace publishes two outputs and copies history and cache together).
SITES_WITH_SEVERAL_CALLS = {('dflash_proposal_trace.py', 'publish_outputs', 'copy'): 2,
                            ('dflash_proposal_trace.py', 'update.copy_history_and_cache', 'copy'): 2,
                            ('sdpa_multi_tp.py', 'refresh', 'generic_op'): 2}
POOL_NAMES = re.compile(r'_paged_kv|kv_cache|k_cache|v_cache|paged_cache|kv_pool', re.IGNORECASE)


def write_sites(path):
    """[(enclosing function path, call name)] of every call to a write-capable ttnn entry in a module, from its syntax tree."""
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack, self.found = [], []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', None)
            if name in WRITE_CALLS:
                self.found.append(('.'.join(self.stack), name))
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(ast.parse(Path(path).read_text(encoding='utf-8')))
    return visitor.found


class WriterCensusTests(unittest.TestCase):
    def test_every_device_write_site_of_the_levers_is_audited_and_none_is_new(self):
        found = {}
        for name in LEVER_MODULES:
            for function, call in write_sites(HERE / name):
                found.setdefault((name, function, call), 0)
                found[(name, function, call)] += 1
        self.assertEqual(sorted(found), sorted(AUDITED_WRITE_SITES),
                         'a TP4 lever gained or lost a device write: audit it (docs/tp4-packed-prefix.md) before it joins a sticky '
                         'profile, then update AUDITED_WRITE_SITES')
        self.assertEqual({site: count for site, count in found.items() if count != 1}, SITES_WITH_SEVERAL_CALLS)

    def test_no_write_site_module_names_the_kv_pool(self):
        for name in sorted({site[0] for site in AUDITED_WRITE_SITES}):
            with self.subTest(module=name):
                text = (HERE / name).read_text(encoding='utf-8')
                self.assertEqual(POOL_NAMES.findall(text), [], '%s writes to the device and names the K/V pool' % name)

    def test_the_census_sees_a_generic_copy_the_name_pattern_misses(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            probe = Path(folder) / 'lever.py'
            probe.write_text('def commit(ttnn, source, cache):\n    ttnn.copy(source, cache)\n', encoding='utf-8')
            self.assertEqual(write_sites(probe), [('commit', 'copy')])
            self.assertEqual(POOL_WRITE.findall(probe.read_text(encoding='utf-8')), [], 'the old pattern is blind to it')


class NullBlockTests(unittest.TestCase):
    """vLLM's block 0 is never handed to a request (serving_kv_reservation.NULL_BLOCKS), so page 0 is the one page nothing owns:
    an idle segment of a padded round is mapped to it and a live table must never name it."""

    def engine(self, extent):
        return SimpleNamespace(users=4, rows_per_user=16, replay_capacity=4096, extent=extent, MAX_IDLE_SEGMENTS=2,
                               shape=SimpleNamespace(page_width=4096))

    def test_an_idle_segment_maps_to_page_zero_at_width_4096(self):
        for extent in (False, True):
            with self.subTest(extent=extent):
                idle = packed_verifier.PackedVerifierEngine.idle_inputs(self.engine(extent), (0, 1))
                self.assertEqual(sorted(idle), [2, 3])
                for tokens, start, pages in idle.values():
                    self.assertEqual(tuple(pages.shape), (1, 4096))
                    self.assertEqual(int(pages.abs().sum()), 0)
                    self.assertEqual(packed_verifier.page_zero_index(pages, start, 16), 0)

    def test_a_live_table_naming_page_zero_inside_its_used_range_is_found(self):
        table = engine_with(blocks_of(5000, 63), 4096).pages
        self.assertIsNone(packed_verifier.page_zero_index(table, 4000, 16))
        table[0, 10] = 0
        self.assertEqual(packed_verifier.page_zero_index(table, 4000, 16), 10)

    def test_the_reservation_never_hands_out_page_zero(self):
        import serving_kv_reservation as reservation

        self.assertEqual(reservation.NULL_BLOCKS, 1)
        self.assertEqual(reservation.pool_blocks(SimpleNamespace(kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(num_gpu_blocks=19968)))), 19967)


class PageWidthTests(unittest.TestCase):
    """The attach-time width check (serving_runtime.prefill_warm_before_traces): under sticky sessions a model whose chunk-input
    page table is not at the width the eager prefill was warmed at (runner.max_num_blocks_per_req) is refused."""

    FOUR = {'QWEN_FAST_TP': '4'}
    STICKY_ON = {'QWEN_FAST_TP': '4', 'QWEN_PREFIX_REUSE': '1', STICKY: '1'}

    def warm(self, environ, buffer_width=None, width=4096):
        import test_tp4_prefill_scratch as scratch

        model = scratch.Model()
        if buffer_width is not None:
            model._chunk_full_page_table_buf = SimpleNamespace(shape=(1, 1, 1, buffer_width))
        with mock.patch.object(serving_runtime, 'pindiag'):
            return serving_runtime.prefill_warm_before_traces(SimpleNamespace(max_num_blocks_per_req=width), model, 2, environ)

    def test_no_buffer_or_a_buffer_at_the_warmed_width_is_admitted(self):
        self.assertTrue(self.warm(self.STICKY_ON))
        self.assertTrue(self.warm(self.STICKY_ON, buffer_width=4096))

    def test_a_buffer_at_another_width_is_refused_at_attach(self):
        for buffer_width in (2048, 4128, 4097):
            with self.subTest(buffer_width=buffer_width):
                with self.assertRaisesRegex(ValueError, 'absent or the width 4096'):
                    self.warm(self.STICKY_ON, buffer_width=buffer_width)

    def test_with_either_switch_off_nothing_is_checked(self):
        for environ in (self.FOUR, dict(self.FOUR, QWEN_PREFIX_REUSE='1'), dict(self.FOUR, **{STICKY: '1'})):
            with self.subTest(environ=sorted(environ)):
                self.assertTrue(self.warm(environ, buffer_width=2048))


if __name__ == '__main__':
    unittest.main()
