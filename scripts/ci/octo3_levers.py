"""Every lever of the fused stack and what it does on an OCTO round (tp4/octo-3, docs/tp4-octo3.md section 2). Stdlib only, host side.

The octo block is a THIRD packed block of eight users x eight rows over the pool slots, beside the two M3 blocks of four users x sixteen rows. A lever of the combined arm (-fx-all: W2, engine reuse
and the op-fusion set) is either

  native       the same code runs at the octo shape: it is generic in users and rows (keyed on the 64 block rows, a per-block or per-pass state) - read from the code, proven by the audited attach A1
  octo-served  the lever's design is the M3 shape's (four users of sixteen rows): the octo block runs the SERVED composition, says so in a line of its own (never a 'fell back' line, which every
               fusion smoke rule fails on) and the judge demands that line on an octo profile
  m3-only      the lever belongs to the M3 blocks and is not built for the octo block at all (the octo block keeps the pre-lever path, logged where the lever has a marker)
  drafter      a drafter-side lever: the octo draft pass (OctoPass, a QuadPass at eight users) takes it like the quad does, or the fused quad drafters draft the octo round (quaddraft arm)
  host         a host-side lever (round host, host gap, engine start): shape-independent by construction
  prefill      acts in prefill or engine start only: no octo-specific behaviour

The test (test_octo3_levers) holds that every flag the combined arm sets over the production profile has a row here, so a lever added to the combined arm without an octo status fails the CPU suite.
"""

NATIVE, SERVED, M3_ONLY, DRAFTER, HOST, PREFILL = 'native', 'octo-served', 'm3-only', 'drafter', 'host', 'prefill'
STATUSES = (NATIVE, SERVED, M3_ONLY, DRAFTER, HOST, PREFILL)

# flag -> (status, one line: why, and the marker or judge rule that shows it)
LEVERS = {
    # --- W2 and engine reuse (the production stack) ---
    'QWEN_FAST_TP4_SDPA': (M3_ONLY, 'W2 multi SDPA is the M3 blocks\' one launch over sixteen-row users; the octo reader keeps eight single launches or the bundle (extent_attention_fold_tp skips sdpa_long_tp.apply for '
                                    'eight-row segments; w2_switch.applies already excluded the octo block); serving_octo admits it beside the octo shape'),
    'QWEN_FAST_TP4_CONV_GATES_SPREAD': (NATIVE, 'F1 runs in the M3 blocks\' conv stage; the octo block reaches it through glue8\'s block conv (gdn_block_conv8_tp.stage, one launch at 64 rows, '
                                                'the w2.off kill switch checked there too); without glue8 the octo block runs the served per-user conv gates'),
    'QWEN_FAST_PARKED_ENGINES': (HOST, 'the octo block is built first and the parked set after it; an octo round drafts with the book\'s quad traces cut to seven rows (or the octo pass, which the coordinator keeps outside the book)'),
    'QWEN_FAST_PARKED_DRAFTS': (DRAFTER, 'the draft book holds pairs and block quads; the octo pass\'s trace is not in it (octo_draft_tp.EXCLUDED_FLAGS no longer refuses the flag; the coordinator closes and rebuilds '
                                         'the octo trace after a rebound member): the one lifecycle no card has run'),
    'QWEN_FAST_LEVERN_BUILD_MS': (HOST, 'the governor charges admission per slot; no shape coupling'),
    # --- verify-pass levers ---
    'QWEN_FAST_TP4_SHARD_ARGMAX': (NATIVE, 'scans the 64-row logits shard (rows 1..64, the device grid); no users or rows_per_user in the plan'),
    'QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2': (NATIVE, 'the tree fold of the same scan'),
    'QWEN_FAST_KV_PAGE_WRITER': (SERVED, 'the design (design_signature, the card-M record) is four users of sixteen rows; the octo block\'s eight users of eight rows keep the chained writer and log '
                                         '"[PINDIAG] tp4 kv page writer octo 8-row served path" once (kv_page_writer_tp4_smoke demands it on an octo profile); an 8x8 design is its own card-M record'),
    'QWEN_FAST_CCL_OPTIONS': (NATIVE, 'keyed on the block rows (64) and the layer count; the engaged line is per rows, the audit needs two block owners'),
    'QWEN_FAST_MLP_CFG': (NATIVE, 'bound by block rows (64); needs QWEN_FAST_SINGLE_GATEUP=1, which the octo shape requires'),
    # --- drafter levers (the octo draft pass, or the fused quads cut to seven rows) ---
    'QWEN_FAST_DRAFT_PERMUTE': (DRAFTER, 'the octo pass hooks kv, fold and unfold at shape=octo (halves=2 users=4 block=8); octo_draft_tp._execute tells the audit which pass is eager'),
    'QWEN_FAST_DRAFT_QKV1': (DRAFTER, 'site=octo in draft_attention_branch; the fused weight is shared with the quad'),
    'QWEN_FAST_DRAFT_HEAD64': (DRAFTER, 'the octo pass is a QuadPass: its head is the quad\'s two 32-row halves as one 64-row matmul'),
    'QWEN_FAST_DRAFT_GATEUP1': (DRAFTER, 'row-based (per_core_M = 2 as the quad); the WP6 audit budget is per pass (draft_fusion_tp.set_shape)'),
    'QWEN_FAST_DRAFT_MM_GRID': (DRAFTER, 'row-based grids; audited per pass'),
    'QWEN_FAST_DRAFT_REDUCE': (DRAFTER, 'row-based; audited per pass'),
    'QWEN_FAST_DRAFT_TAIL': (DRAFTER, 'row-based; audited per pass'),
    # --- host levers ---
    'QWEN_FAST_TP4_ROUND_HOST_SELECT': (HOST, 'the batched selection is row-wise for any user count (the octo draft returns into the same path)'),
    'QWEN_FAST_TP4_ROUND_HOST_READ': (HOST, 'the octo readback uses round_host.merge_chunk_candidates at block_rows 32'),
    'QWEN_FAST_TP4_ROUND_HOST_KEYED': (HOST, 'keyed on the block\'s own shape and fixture; the keyed-share smoke rule counts one verify for an octo step'),
    'QWEN_FAST_TP4_ROUND_HOST_LEAN': (HOST, 'drops the per-phase and commit audit lines only (the lean ledger replaces them); no device work'),
    'QWEN_FAST_PRESTAGE_DIFF': (HOST, 'a per-block record; the pre-stage covers three blocks (the octo block is block C); the destinations overlap check is the open question (hostgap rule fails loudly)'),
    'QWEN_FAST_WRITE_PACKED_LEAN': (HOST, 'a lean address book and mapper per block; the engaged line is per block, no shape assumption'),
    'QWEN_FAST_BATCHED_READS': (HOST, 'the verify read-back works at 64 rows for any shape; the octo draft pass\'s own collect uses its unbatched reads (not applied there: a ~0.5-1 ms gap, no marker, no failure)'),
    'QWEN_FAST_TAP_WINDOW_ALIGNED': (HOST, 'the drafter window\'s tile-aligned join at a returning turn\'s rebind (tp4/rebind-fast); data movement before any block runs, shape-independent'),
    'QWEN_FAST_SDPA_PF_ONEQ': (PREFILL, 'the chunked prefill SDPA with one q chunk per core; acts in prefill, needs the K64j-OQ binary (the oneq arms)'),
    'QWEN_FAST_GRID13': (NATIVE, 'matmul grids for the 13x10 grid: the prefill all-gather matmuls, the verify block\'s output projections at 64 rows (the octo block is 64 rows) and the fused commit\'s feature projection'),
    'QWEN_FAST_DEVICE_ZEROS': (PREFILL, 'engine start: fills the pool\'s zero buffers (the (8, 8) extent storage is lent by the same filler)'),
    'QWEN_FAST_LAZY_SHARD_W': (PREFILL, 'engine start: the weight loader'),
}

# Flags of the combined arm that are not levers of the decode pass at all and need no row.
NOT_LEVERS = ('QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT',)


def status_of(flag):
    return LEVERS[flag][0] if flag in LEVERS else None


def table():
    """[(flag, status, note)] sorted by status then flag."""
    return sorted(((flag, status, note) for flag, (status, note) in LEVERS.items()), key=lambda row: (STATUSES.index(row[1]), row[0]))


def counts():
    found = {status: 0 for status in STATUSES}
    for status, _note in LEVERS.values():
        found[status] += 1
    return found
