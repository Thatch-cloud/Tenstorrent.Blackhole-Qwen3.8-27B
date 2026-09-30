"""A pure-Python mirror of K1's head-slice rule and index math (factory F14/F15, reader R6-R8, writer W1-W3).

The factory, the kernels and the Python wiring compute the same things; this module states them once so the
CPU tests can check every shape the replay can produce against the legacy (unsliced) path:

  slice_rule        F14/F15: rows_per_kv G, each head's first row tile floor(h*G/32) and end ceil((h+1)*G/32),
                    the slice (the widest span), and the refusal the factory raises (None when it builds)
  q_tiles           R7: the DRAM Q tiles one core of (entry, head) reads, in L1 order
  mask_tiles        R8 with read_mask_chunk<PNHt>: the DRAM mask tiles it reads for one chunk, in L1 order
  writer_writes     W2/W3: every face-line write of the root core of (entry, head): (row, L1 slot, DRAM tile,
                    byte offset); legacy_writer_writes is the stock writer's (write_partial_tiles_to_memory)
  cb_bytes          the factory's CB total per core (the '[QWEN-SDPA] flags=... cb_bytes=' figure)
  output_rows       what a (possibly broken) sliced call leaves in DRAM, row by row, given per-row values: the
                    fake ttnn of the card-B dry run is built on it, so a broken writer or reader variant shows

It imports nothing but the standard library and runs anywhere (the card-B harness uses slice_rule, cb_bytes
and slice_applies on the device host too).
"""

TILE = 32
FACE = 16
HEAD_DIM_TILES = 8              # DHt = vDHt = 256 / 32
HEADS_PER_TOKEN = 12            # folded rows per draft token on one chip (6 per KV head)
KV_HEADS = 2
ELEMENT = 2                     # bf16 output
FACE_LINE_BYTES = FACE * ELEMENT
FACE_BYTES = FACE * FACE * ELEMENT


def ceil_div(a, b):
    return -(-a // b)


def pnht_full(query_rows):
    """Q's row tiles per entry (PNH / 32 with PNH padded to 32)."""
    return ceil_div(query_rows, TILE)


def head_span(head, rows_per_kv):
    """KV head `head`'s row tiles [first, end): the tiles holding its folded rows."""
    return (head * rows_per_kv) // TILE, ceil_div((head + 1) * rows_per_kv, TILE)


def slice_rule(num_q_heads, num_kv_heads=KV_HEADS, pnht=None):
    """F14/F15 for a call with num_q_heads folded rows per entry: dict(rows_per_kv, pnht_full, slice_tiles,
    starts, ends, refusal). refusal is None when the factory builds the 0x4 program, else the start of the
    TT_FATAL text it raises (the card-B harness matches refusals by these)."""
    pnht = pnht_full(num_q_heads) if pnht is None else pnht
    out = dict(num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, pnht_full=pnht, rows_per_kv=None,
               slice_tiles=None, starts=[], ends=[], refusal=None)
    if num_kv_heads <= 1 or num_q_heads % num_kv_heads:
        out['refusal'] = '[QWEN-SDPA] q-slice needs num_q_heads'
        return out
    rows_per_kv = num_q_heads // num_kv_heads
    spans = [head_span(head, rows_per_kv) for head in range(num_kv_heads)]
    slice_tiles = max(end - first for first, end in spans)
    out.update(rows_per_kv=rows_per_kv, slice_tiles=slice_tiles, starts=[first for first, _ in spans],
               ends=[end for _, end in spans])
    if slice_tiles >= pnht:
        out['refusal'] = '[QWEN-SDPA] q-slice saves no tile'
    elif any(first + slice_tiles > pnht or end > first + slice_tiles for first, end in spans):
        out['refusal'] = '[QWEN-SDPA] q-slice does not cover KV head'
    return out


def slice_applies(rows, heads_per_token=HEADS_PER_TOKEN, num_kv_heads=KV_HEADS):
    """The Python wiring's rule (pooled_attention_replay): a bundle of `rows`-row groups gets 0x4 exactly when
    the factory builds it, i.e. when the slice saves a tile: 6-8 rows (3 -> 2 tiles) and 16 (6 -> 3); 1-5 never."""
    return slice_rule(rows * heads_per_token, num_kv_heads)['refusal'] is None


def q_tile_start(head, rows_per_kv, sliced):
    """R6 / W1: the slice's first row tile ((h * G) >> 5), 0 without the slice."""
    return (head * rows_per_kv) >> 5 if sliced else 0


def program_pnht(num_q_heads, flags, num_kv_heads=KV_HEADS):
    """The PNHt the factory builds for these flags (the F4 line's PNHt): the slice under 0x4, else Q's."""
    rule = slice_rule(num_q_heads, num_kv_heads)
    if flags & 0x4:
        if rule['refusal']:
            raise ValueError('flags 0x%x on %d rows: %s' % (flags, num_q_heads, rule['refusal']))
        return rule['slice_tiles']
    return rule['pnht_full']


def q_tiles(entry, head, num_q_heads, sliced, dht=HEAD_DIM_TILES, bug=None):
    """R7 + read_q: the Q tile ids entry `entry`'s core of `head` reads, in L1 order. bug='r7' drops the
    q_tile_start offset (the slice reads the tiles from row tile 0)."""
    rule = slice_rule(num_q_heads)
    full = rule['pnht_full']
    pnht = rule['slice_tiles'] if sliced else full
    start = q_tile_start(head, rule['rows_per_kv'], sliced) if bug != 'r7' else 0
    base = entry * full * dht + start * dht
    return [base + index for index in range(pnht * dht)]


def mask_tiles(entry, head, num_q_heads, sliced, mask_width_t, sk_chunk_t=8, chunk=None, bmask=None, bug=None):
    """R8 + read_mask_chunk<PNHt>: the mask tile ids read for chunk `chunk` (None: the tail chunk, the mask's
    last sk_chunk_t columns), in L1 order (row tile by row tile). bug='r8_row' drops the q_tile_start row
    offset; bug='r8_stride' uses the slice's PNHt as the batch stride (the review's two R8 failure modes)."""
    rule = slice_rule(num_q_heads)
    full = rule['pnht_full']
    pnht = rule['slice_tiles'] if sliced else full
    start = q_tile_start(head, rule['rows_per_kv'], sliced)
    stride = pnht if bug == 'r8_stride' else full
    offset = 0 if bug == 'r8_row' else start * mask_width_t
    batch = entry % (bmask or (entry + 1))
    column = (mask_width_t - sk_chunk_t) if chunk is None else chunk * sk_chunk_t
    first = batch * stride * mask_width_t + offset + column
    return [first + row * mask_width_t + col for row in range(pnht) for col in range(sk_chunk_t)]


def row_face_offset(tile_row):
    """write_partial_tiles_to_memory's in-tile byte offset of a row's first face line (faces 0/1 for rows 0-15,
    2/3 for 16-31); the second face line is FACE_BYTES further."""
    return tile_row * FACE_LINE_BYTES if tile_row < FACE else (tile_row + FACE) * FACE_LINE_BYTES


def writer_writes(entry, head, num_q_heads, sliced, vdht=HEAD_DIM_TILES, bug=None):
    """W2/W3: [(row, L1 slot, DRAM tile, byte offset)] for every face-line write the root core of (entry,
    head) issues, in issue order (hidden tile outer, row inner, two face lines per row). The L1 slot is the
    cb_out tile the bytes come from; DRAM tiles are the full layout's. bug='w3' skips the q_tile_start rebase
    of the L1 slot; bug='w2' uses the slice's out_chunk_tiles as the entry stride (the stock writer's form)."""
    rule = slice_rule(num_q_heads)
    full = rule['pnht_full']
    pnht = rule['slice_tiles'] if sliced else full
    rows_per_kv = rule['rows_per_kv']
    start = q_tile_start(head, rows_per_kv, sliced)
    out_tile_id = entry * (pnht if bug == 'w2' else full) * vdht
    rebase = 0 if bug == 'w3' else start
    writes = []
    for hidden in range(vdht):
        for index in range(rows_per_kv):
            row = head * rows_per_kv + index
            head_tile, tile_row = divmod(row, TILE)
            slot = (head_tile - rebase) * vdht + hidden
            dram = out_tile_id + head_tile * vdht + hidden
            offset = row_face_offset(tile_row)
            writes.append((row, slot, dram, offset))
            writes.append((row, slot, dram, offset + FACE_BYTES))
    return writes


def legacy_writer_writes(entry, head, num_q_heads, vdht=HEAD_DIM_TILES):
    """The stock writer (write_partial_tiles_to_memory, PNHt = Q's row tiles), in the same form."""
    full = pnht_full(num_q_heads)
    rows_per_kv = num_q_heads // KV_HEADS
    out_tile_id = entry * full * vdht
    writes = []
    for hidden in range(vdht):
        for index in range(rows_per_kv):
            row = head * rows_per_kv + index
            head_tile, tile_row = divmod(row, TILE)
            tile_index = head_tile * vdht + hidden
            offset = row_face_offset(tile_row)
            writes.append((row, tile_index, out_tile_id + tile_index, offset))
            writes.append((row, tile_index, out_tile_id + tile_index, offset + FACE_BYTES))
    return writes


def slot_row_tile(slot, head, num_q_heads, sliced, vdht=HEAD_DIM_TILES):
    """The global row tile whose results the compute kernel left in cb_out slot `slot` (compute keeps the Q
    tiles' order: L1 row tile k is Q row tile q_tile_start + k), or None past the CB's pushed tiles."""
    rule = slice_rule(num_q_heads)
    pnht = rule['slice_tiles'] if sliced else rule['pnht_full']
    row_tile = slot // vdht
    if not 0 <= row_tile < pnht:
        return None
    return q_tile_start(head, rule['rows_per_kv'], sliced) + row_tile


# ---------------------------------------------------------------------------------------------
# CB bytes per core: the factory's CB list at k_chunk 256, DHt = vDHt = 8, bf16 Q / mask / out and bf8 K / V.
# ---------------------------------------------------------------------------------------------

BF16_TILE = 2048
BF8_TILE = 1088
DRAM_ALIGNMENT = 64


def page_table_bytes(capacity, page=64, alignment=DRAM_ALIGNMENT):
    """c_9: one int32 page-table row, aligned to the DRAM page alignment."""
    return ceil_div((capacity // page) * 4, alignment) * alignment


def cb_bytes(capacity, pnht, rounds=4, sk_chunk_t=8, dht=HEAD_DIM_TILES):
    """The factory's sum of CB sizes (F4's cb_bytes) with compact tree scratch (`rounds` slots):
    q, mask, q_rm, qk_im and the six PNHt*8-tile output-shaped CBs (c_25, c_26, c_23, c_16, c_20) at bf16;
    eleven PNHt-tile stats CBs (c_6, c_7, c_17, c_18, c_21, c_22, c_27-c_31); three 1-tile scalars (c_5, c_11,
    c_12); c_19 = (out + 2 stats) x rounds; K and V double-buffered at bf8; the page-table row."""
    q = pnht * dht
    qk = pnht * sk_chunk_t
    out = pnht * dht
    tiles = (q + qk + q + qk          # c_0 q, c_3 mask, c_10 q_rm, c_24 qk_im
             + 5 * out                # c_25, c_26, c_23, c_16, c_20
             + 11 * pnht              # stats
             + 3                      # c_5, c_11, c_12
             + (out + 2 * pnht) * rounds)
    kv = 2 * (sk_chunk_t * dht * 2) * BF8_TILE
    return tiles * BF16_TILE + kv + page_table_bytes(capacity)


# ---------------------------------------------------------------------------------------------
# What a sliced call leaves in DRAM (the fake ttnn of the card-B dry run).
# ---------------------------------------------------------------------------------------------

def output_rows(batches, num_q_heads, sliced, row_value, *, bugs=(), garbage=None):
    """{(entry, row): value} a call writes, from row_value(entry, head, q_row, mask_row) - the value the core of
    (entry, head) computes for the Q row it loaded into a slot position and the mask row loaded beside it.
    bugs: any of 'r7', 'r8_row', 'r8_stride', 'w2', 'w3' (the index slips each models). Rows a broken writer
    sends to another entry's tiles overwrite that entry's; an L1 slot past the CB's tiles yields `garbage`.
    Rows nobody writes are absent (the caller keeps DRAM's stale bytes there)."""
    rule = slice_rule(num_q_heads)
    full = rule['pnht_full']
    rows_per_kv = rule['rows_per_kv']
    pnht = rule['slice_tiles'] if sliced else full
    written = {}
    for entry in range(batches):
        for head in range(KV_HEADS):
            start = q_tile_start(head, rows_per_kv, sliced)
            q_first = (0 if 'r7' in bugs else start)
            mask_stride = pnht if 'r8_stride' in bugs else full
            mask_first = entry * mask_stride + (0 if 'r8_row' in bugs else start)   # in row tiles of the mask
            slots = {}
            for k in range(pnht):
                for i in range(TILE):
                    q_row = (q_first + k) * TILE + i
                    mask_entry, mask_row_tile = divmod(mask_first + k, full)
                    slots[(k, i)] = row_value(entry, head, q_row, (mask_entry, mask_row_tile * TILE + i))
            out_entry_stride = pnht if 'w2' in bugs else full
            rebase = 0 if 'w3' in bugs else start
            for index in range(rows_per_kv):
                row = head * rows_per_kv + index
                head_tile, tile_row = divmod(row, TILE)
                k = head_tile - rebase
                value = slots.get((k, tile_row), garbage) if 0 <= k < pnht else garbage
                target_tile = entry * out_entry_stride + head_tile
                target_entry, target_row_tile = divmod(target_tile, full)
                written[(target_entry, target_row_tile * TILE + tile_row)] = value
    return written
