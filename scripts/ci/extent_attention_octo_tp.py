"""The octo-T8 extent readers at four cards: ONE eight-row group per bundle (K64j 'G8B1', flags 0x21), beside the pinned twin.

THE GEOMETRY. extent_attention_replay_tp.ExtentSegmentReader is qualified at G8B2: a 16-row segment bundles as two eight-row groups
(one SDPA launch of two batch entries, KV share, flags 0x23). Octo-T8 (docs/tp4-octo.md) is eight users of EIGHT rows in the same 64-row
block: each segment is ONE eight-row group, a bundle of one entry. KV share needs a second entry to share with, so the flag set is
tail | extent = 0x21 (pooled_attention_replay.mode_flags drops the share bit at one entry; the q-slice is illegal at one KV head).

WHY A TWIN AND NOT AN EDIT. extent_attention_replay_tp.py is held unedited by the four-card evidence (packed_any_evidence_tp4.json pins
its sha256: CB2b qualified those bytes), and the pair's extent_attention_replay.py by the TP2 one. This module subclasses the pinned
segment reader and replaces its constructor's two G8B2 refusals (the bundle count and the flag set) with the G8B1 ones, everything else
line for line: the same word, the same cur_pos, the same narrow masks from the same pinned mask kernel (prepare_narrow at one batch), the same
staging and the same call (execute_extent, which already takes bundles of one to three groups). The K64j binary and its four kernels are
UNCHANGED: 0x21 is an existing K64j flag set (K1/K3 qualified it at G8B2 and G4B3, and the harness skip section ran a one-entry bundle); what no card has
run is a bundle of one eight-row group, which is the qualification job Q1 (optimisation/ttnn-op/k64j: k64j_card_b.py combo G8B1:0x21, extent_reader_card_b.py
--octo). Until Q1 passes the geometry is UNQUALIFIED and gate only (serving_octo logs it).

WHAT STAYS THE SAME PER ROW. The accumulation order of a row's online softmax is fixed by the 256-key chunk boundaries, the per-core chunk ranges and
the reduction tree (cores per entry, 16 for any bundle of one to three entries) and the per-op arithmetic. A bundle of one entry has the same
16 cores, the same chunks and the same tree as each entry of G8B2; the share handshake only changes who reads the KV from DRAM (the leader) and who
receives it (the twin). So an octo row is the M3 row at another launch geometry, not another sum. That is an argument, not a measurement: Q1's K2
section (the native one-row decode against the extent path, bit for bit) is the measurement.
"""

import os

import extent_attention_replay_tp as quad
from extent_attention_replay import ENGAGED_MARKER, EXTENT_GROUP_ROWS, K, LAYOUT, TREE_SCRATCH_ENV, check_start
import pooled_attention_replay as pooled
from pooled_attention_replay import apply_sdpa_modes, sdpa_modes, validate_segments

OCTO_ROWS = 8                                                    # rows per segment: ONE group
OCTO_ENTRIES = 1                                                 # groups per bundle (G8B1)
OCTO_FLAGS = pooled.QWEN_MASK_TAIL | pooled.QWEN_RUNTIME_EXTENT  # 0x21: tail | extent (no KV share at one entry)
OCTO_MARKER = '[OCTO] extent readers'


def is_octo_block(segments):
    """Whether these (first, last) packed segments are the octo block's: every segment exactly OCTO_ROWS rows. A 16-row (M3) or 32-row (M1) segment is not."""
    try:
        return len(segments) > 0 and all(last - first == OCTO_ROWS for first, last in segments)
    except (TypeError, ValueError):
        return False


def bundles_of(rows, max_group_rows):
    """The octo segment's bundles: the pinned layout of OCTO_ROWS rows at eight-row groups, which must be ONE bundle of ONE group of eight rows."""
    bundles = LAYOUT(rows, max_group_rows)
    if (len(bundles) != 1 or len(bundles[0]) != OCTO_ENTRIES or bundles[0][0]['rows'] != OCTO_ROWS or bundles[0][0]['offset'] != 0):
        raise ValueError('Octo extent replay is G8B1 only (one eight-row group per bundle, 0x21): a %d-row segment bundles as %r'
                         % (rows, [[group['rows'] for group in bundle] for bundle in bundles]))
    return bundles


class OctoSegmentReader(quad.ExtentSegmentReader):
    """One octo user's extent reader: the pinned four-card segment reader's contracts at ONE eight-row group, flags 0x21."""

    def __init__(self, operations, mesh, rows, page_width, pages_host, *, storage, max_group_rows, start):
        import torch

        # Every refusal before anything is allocated (the pinned constructor's order).
        if type(rows) is not int or rows != OCTO_ROWS:
            raise ValueError('The octo extent reader serves an explicit T8 segment, got %r' % (rows,))
        if type(max_group_rows) is not int or max_group_rows != EXTENT_GROUP_ROWS:
            raise ValueError('Octo extent replay is qualified at eight-row groups only (K64j G8B1, 0x21 at one KV head), got %r'
                             % (max_group_rows,))
        if os.environ.get(TREE_SCRATCH_ENV) != '1':
            raise ValueError('Eight-row replay requires process-fixed compact native scratch (%s=1), '
                             'the pinned reader\'s G8 precondition' % TREE_SCRATCH_ENV)
        modes = sdpa_modes()
        if 'tail' not in modes:
            raise ValueError('Extent replay needs QWEN_FAST_SDPA_MODES to include tail (K64j refuses 0x20 without 0x1)')
        if type(page_width) is not int or page_width < 4 or page_width % (K // 64):
            raise ValueError('Extent replay needs an integer page-table width of whole 256-key families, got %r' % (page_width,))
        if getattr(pages_host, 'ndim', None) != 2 or pages_host.shape[0] != 1 or pages_host.shape[1] < page_width:
            raise ValueError('One complete native cache page table required')
        bundles = bundles_of(rows, max_group_rows)
        self.operations, self.mesh = operations, mesh
        self.rows, self.page_width, self.capacity = rows, page_width, page_width * 64
        self.max_group_rows = max_group_rows
        self.closed = self.failed = False
        self.start = None
        self.validate(start)
        lent = quad.validate_extent_storage(operations, storage, bundles, page_width)
        self.borrowed = [tensor for pair in lent for tensor in pair]
        self.cur_pos = [positions for table, positions in lent]
        self.owned, self.metadata, self.programs = [], [], []
        self.calls, self.refresh_calls = 0, 0
        self.mask_scope = None
        grid = mesh.compute_with_storage_grid_size()
        try:
            self.positions = quad._upload(operations, mesh, torch.zeros(8, dtype=torch.int32), operations.int32)
            self.owned.append(self.positions)
            for bundle, (table, positions) in zip(bundles, lent, strict=True):
                count = bundle[0]['rows']
                mask = quad._upload(operations, mesh, torch.zeros(len(bundle), 1, count * quad.head_rows(), K, dtype=torch.bfloat16),
                                    operations.bfloat16)
                self.owned.append(mask)
                config = operations.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y),
                    exp_approx_mode=False, q_chunk_size=0, k_chunk_size=256)
                self.metadata.append((bundle, table, mask, config))
                self.programs.append(quad.prepare_narrow(mesh, self.positions, mask, rows=count, batches=len(bundle),
                                                         offset=bundle[0]['offset']))
            quad.independent(operations, [*self.borrowed, *self.owned], 'Extent reader buffers')
            # Design A6: the word, cur_pos and tables for `start`, before any forward reads them.
            self.stage(start, table=pages_host)
            apply_sdpa_modes(self, modes | {'extent'})
            flags = tuple(getattr(self, 'sdpa_modes_applied', None) or ())
            if len(flags) != len(self.metadata) or any(value != OCTO_FLAGS for value in flags):
                raise ValueError('Octo extent replay is qualified at 0x21 only (tail, extent at G8B1); '
                                 'QWEN_FAST_SDPA_MODES=%s gives %s' % (','.join(sorted(modes)), ['0x%x' % value for value in flags]))
        except BaseException:
            self.close()
            raise


def build_packed(reader, operations, mesh, segments, page_width, tables_host, *, storage, max_group_rows, starts):
    """Everything extent_attention_replay_tp.PackedExtentReplayReader.__init__ does, over OctoSegmentReader, onto `reader` (a PackedExtentReplayReader
    that has not run its constructor). Line for line the pinned one but for the segment class and the engaged line."""
    reader.segments = validate_segments(segments)
    tables_host, starts = list(tables_host), tuple(starts)
    lent = None if storage is None else [list(pairs) for pairs in storage]
    if (len(tables_host) != len(reader.segments) or len(starts) != len(reader.segments)
            or lent is None or len(lent) != len(reader.segments)):
        raise ValueError('One host page table, one start and one lent (table, cur_pos) set per packed segment required')
    if type(max_group_rows) is not int or type(page_width) is not int:
        raise ValueError('Integer group width and page-table width required')
    if not is_octo_block(reader.segments):
        raise ValueError('The octo extent reader serves segments of exactly %d rows, got %r' % (OCTO_ROWS, reader.segments))
    # Every start on the host before any reader is built (a bad start refuses the block with nothing staged).
    for (first, last), start in zip(reader.segments, starts, strict=True):
        check_start(start, last - first, page_width * 64)
    for (first, last), pairs in zip(reader.segments, lent, strict=True):
        quad.validate_extent_storage(operations, pairs, bundles_of(last - first, max_group_rows), page_width)
    quad.independent(operations, [tensor for pairs in lent for pair in pairs for tensor in pair], 'Extent storage')
    reader.operations, reader.mesh = operations, mesh
    reader.page_width, reader.capacity = page_width, page_width * 64
    reader.max_group_rows = max_group_rows
    reader.rows = reader.segments[-1][1]
    reader.readers = []
    reader.calls = 0
    reader.closed = False
    try:
        for (first, last), pages_host, pairs, start in zip(reader.segments, tables_host, lent, starts, strict=True):
            reader.readers.append(OctoSegmentReader(operations, mesh, last - first, page_width, pages_host,
                                                    storage=pairs, max_group_rows=max_group_rows, start=start))
    except BaseException:
        reader.close()
        raise
    quad._pindiag('%s segments=%d flags=%s mask=narrow capacity=%d geometry=G8B1' % (
        ENGAGED_MARKER, len(reader.readers),
        ','.join('0x%x' % value for member in reader.readers for value in member.sdpa_modes_applied), reader.capacity))
    return reader
