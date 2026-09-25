"""The ordered K/V write over a verify block wider than its kernel's 32-row tile.

WHY. ordered_cache.update is the fixture's audited BF8 read-modify-write of the paged
K/V cache: one worker core per query row, chained by a semaphore so the rows of one
page land in order, over a (1, rows, 32, 256) prepared tile with rows in (1, 2, 4, 8,
16, 32) (ordered_cache.validate_shapes) - the geometry the frozen 32K runtime was
audited at. The 64-row M3 verify block (packed_shapes.m3_shape, four T16 users) writes
64 rows per full-attention layer, and the per-row alternative,
attention_batch.SerialCacheWriter, is pinned frozen-recipe evidence that also stops at
32 rows. Neither is edited: the block's writer runs the audited kernel exactly as
qualified, once per 32-row tile of the block.

HOW. SegmentedOrderedCacheWriter takes the block's tiles - for each 32-row slice of the
block, that slice's own positions word (32,) and page-table rows (32, page_width),
device tensors the fixture uploads at construction (model_batch.prepare_inputs, so they
are pre-trace by the block's construction order) and the packed block restages before
every verify (packed_verifier.stage_packed) - and per call slices the prepared K/V on
its row axis (a whole-tile DRAM slice, the way the serial writer slices one row) and
runs ordered_cache.update over each tile with that tile's metadata. The packed
segments are 16 rows and every tile boundary is a segment boundary, so no user's rows
are split across launches, and two launches on one command queue keep the order one
launch had. Blocks within one tile keep the plain OrderedCacheWriter; this writer
refuses them. Unverified on hardware until the four-user gate runs.
"""

from types import SimpleNamespace

TILE_ROWS = 32
KV_HEADS, KV_WIDTH = 32, 256


def tile_rows(rows):
    """The 32-row tiles of a block wider than one kernel tile, in row order."""
    if type(rows) is not int or rows <= TILE_ROWS or rows % TILE_ROWS:
        raise ValueError('Segmented ordered cache writes serve blocks of whole %d-row tiles beyond one tile'
                         % TILE_ROWS)
    return tuple((first, first + TILE_ROWS) for first in range(0, rows, TILE_ROWS))


def tile(rows, positions, pages):
    """One tile record: its row span and its own staged positions word and page-table rows."""
    return SimpleNamespace(rows=tuple(rows), positions=positions, pages=pages)


def validate_tiles(operations, tiles):
    """The block's tiles from row 0 in order, each 32 rows with a row-major int32 (32,)
    positions word and (32, page_width) page-table rows of one width. Returns
    (tiles, block_rows, page_width)."""
    tiles = tuple(tiles)
    cursor, widths = 0, set()
    for entry in tiles:
        rows = tuple(getattr(entry, 'rows', ()))
        positions, pages = getattr(entry, 'positions', None), getattr(entry, 'pages', None)
        if rows != (cursor, cursor + TILE_ROWS) or positions is None or pages is None:
            raise ValueError('Cache tiles must cover the block from row 0 in %d-row tiles, each with its own '
                             'positions and page-table rows' % TILE_ROWS)
        if (tuple(positions.shape) != (TILE_ROWS,) or positions.dtype != operations.int32
                or positions.layout != operations.ROW_MAJOR_LAYOUT):
            raise ValueError('A cache tile stages a row-major int32 (%d,) positions word' % TILE_ROWS)
        if (len(pages.shape) != 2 or pages.shape[0] != TILE_ROWS or pages.dtype != operations.int32
                or pages.layout != operations.ROW_MAJOR_LAYOUT):
            raise ValueError('A cache tile stages row-major int32 (%d, page_width) page-table rows' % TILE_ROWS)
        widths.add(pages.shape[1])
        cursor += TILE_ROWS
    if tile_rows(cursor) != tuple(entry.rows for entry in tiles) or len(widths) != 1:
        raise ValueError('Cache tiles must number more than one, over one page-table width')
    return tiles, cursor, widths.pop()


class SegmentedOrderedCacheWriter:
    """attention_batch.OrderedCacheWriter's contract over a block of several 32-row tiles:
    the audited ordered update once per tile, each over that tile's staged metadata."""

    def __init__(self, mesh, operations, kernels, tiles):
        self.tiles, self.rows, self.page_width = validate_tiles(operations, tiles)
        self.mesh, self.operations, self.kernels = mesh, operations, kernels
        self.calls = 0

    def __call__(self, cache, packed, *, update_idxs_tensor, page_table):
        from ordered_cache import update

        operations = self.operations
        shape = tuple(packed.shape)
        if shape != (1, self.rows, KV_HEADS, KV_WIDTH):
            raise ValueError('Unexpected native prepared KV shape for the %d-row block: %r' % (self.rows, shape))
        # The model passes the block's whole positions word and page table; each tile
        # carries its own staged slice of exactly these, so only the geometry is checked.
        if tuple(update_idxs_tensor.shape) != (self.rows,) or tuple(page_table.shape) != (self.rows, self.page_width):
            raise ValueError('The block positions and page table must cover every tile row')
        converted = packed.memory_config() != operations.DRAM_MEMORY_CONFIG
        interleaved = operations.to_memory_config(packed, operations.DRAM_MEMORY_CONFIG) if converted else packed
        try:
            for entry in self.tiles:
                first, last = entry.rows
                piece = operations.slice(interleaved, (0, first, 0, 0), (1, last, KV_HEADS, KV_WIDTH),
                                         memory_config=operations.DRAM_MEMORY_CONFIG)
                try:
                    update(self.mesh, cache, piece, entry.positions, entry.pages, self.kernels)
                finally:
                    operations.deallocate(piece)
        finally:
            if converted:
                operations.deallocate(interleaved)
        self.calls += 1
