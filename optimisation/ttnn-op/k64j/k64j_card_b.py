"""K64j K1-K4 (c2-serve-for-real-plan.md section 2.3, hardware iterations) on the qualification card: the runtime
extent (flag 0x20) of graft K64j equals the compile-time served call byte for byte, at every family, mixed, in a
trace whose extent is rewritten between replays, with skipped entries, and under KV share.

WHAT IT DECIDES. K64j (optimisation/ttnn-op/k64j: factory F19-F22, kernels R10 / C3 / W4) gives the served
non-causal replay call a cur_pos tensor of one int32 word per entry: E - 1, E = (start // 256 + 1) * 256 the entry's
256-key family, or -1 (UINT32_MAX) to skip the entry; under KV share (0x2) every entry takes slot 0's word (R10).
The call carries the tail flag (0x1) and the NARROW one-chunk mask (256 columns: the positions [E - 256, E), -inf
past each row's position, the bits the causal writer's generate_mask would write) and runs on the C-wide page table
(C = the served 131,328 keys), whose pages past each entry's E are POISONED (K = 0, V = +16384: any read moves every
row). The reference is K64i's served call with the same flags minus 0x20 at capacity E: the table truncated to E,
the same narrow mask. The P0 probe showed K64i's tail on these shapes equal to legacy (section E, runs 36218136852
and 36218529407), so an equal 0x20 call is the served arithmetic at E, and an equal output also proves the runtime
split read nothing past E.

Sections (--sections, default K1/K3's N, X, M, K, L, T; per seed in the order N, X, M, K, L, T, K2, X7, Z, timing;
N, L, T and timing on the first seed only; CB2a's K2, X7 and Z are below):
  N  K3's refusals, decisive: the loaded binary must refuse the unknown-flag control 0x11 ('[QWEN-SDPA] unknown
     flags'), 0x20 without the tail flag, 0x21 with a wide mask, 0x21 without a cur_pos tensor, 0x21 with B + 1
     words, 0x21 on a causal call and 0x01 with a cur_pos tensor, each with its factory literal (F19, F20). An
     accepted call is a decisive difference (FAIL); a refusal with another text is a warning.
  X  K1: every (shape, flags) of --combos (G4B3 0x21 / 0x23; G8B2 0x21 / 0x23 / 0x27 slice / 0x2F slice and
     read-ahead), every extent of --extents (K1's six, 2,304 ... 131,328), every start of --starts (+0, +7, +127,
     +240, +255: entry b sits at E - 256 + min(255, s + b * rows)), --seeds (K1 asks 0-4) and --variants: the 0x20
     call at words E - 1 equals the reference at capacity E, per entry.
  M  K1 mixed: a B = 3 call without share (0x21, G4B3), its three entries three users at three different families,
     each entry equal to its own family's reference; and for every share combo, a call whose cur_pos words differ
     per slot equals the same call with every slot set to slot 0's word (slot 0's family has poison past it, the
     other words point below and above it).
  K  K3 skips, eager, on 0x21 G4B3 (three users at three families): -1 on the entries of each pattern of
     probe.SKIP_PATTERNS, every live entry equal to the all-live call, the skipped rows recorded after the output was
     poisoned; under share (0x23) slot 0 = -1 skips the whole bundle (it must return: the watchdog) and slot 1 = -1
     with slot 0 live equals the all-live call (slot 0 wins); and a B = 1 call that skips its only user.
  L  K3 liveness: per combo and extent below C, words E (one poisoned chunk more) against E - 1 on a zero mask must
     move every entry; a dead control decides nothing.
  T  K1 trace: per --trace-combos (0x21 on G4B3, 0x27 on G8B2), one program captured once and replayed
     --trace-families times with the cur_pos words AND the narrow mask rewritten between replays
     (copy_host_to_device_tensor, as the extent readers will), spanning more than 50 families: each replay equals
     the eager call at the same words and mask; entry 0 equals the reference at its family on --trace-references
     evenly spaced replays; skips replayed in the same trace (non-share combos); then the FENCE trace, the same
     program captured on a table poisoned past each entry's own family, replayed at the offsets 0, 1, 127, 128, 254,
     255 of those families, each equal to the eager call on the CLEAN table, and one replay at words E that must
     differ (the poison is live inside the trace).
  timing (unless --no-timing; recorded): the 0x21 call at E - 1 on the C-wide table against the 0x1 call at capacity
     E, per extent; and 0 against B skipped entries.

CB2a (s2-design.md W10a and 6.2; opt-in: --sections K2,X7,Z, per seed in that order). These run the SERVED calls'
kwargs: page_table_tensor by keyword, the model's scale (HD ** -0.5) and output memory (L1, --output-memory):
  K2 DECISIVE, and it alone sets the exactness policy (design 6.1; its own verdict is k2_verdict=). The solo path's
     own call against the packed extent path, row by row. The reference is the native decode call as
     SerialAttentionReader issues it for one live user (scripts/ci/attention_batch.py:96-105): one query row of 12
     heads (a half-tile Q), B = 1, the row's own cur_pos p, the user's (1, C / 64) table, and the model's kwargs
     verbatim (NATIVE_DECODE: docker/qwen-c2-graft/graft/attention/tp.py:737-754 - the device grid, exp_approx_mode
     False, q_chunk_size 0, k_chunk_size 0, the scale, the output in L1; is_causal and compute_kernel_config not
     passed). What source cannot pin is listed in UNVERIFIED, recorded and printed at run time. The subject is the
     same rows inside one G8B2 0x27 call as the extent reader issues it: the 16-row ticket at start s folded into
     one B = 2 bundle (offsets 0 and 8, LAYOUT(16, 8)), cur_pos E - 1 in both slots, the narrow mask the pinned
     kernel writes at capacity 256 with the word s & 255 (served_mask), the C-wide table poisoned past E. Every row
     p < E of the ticket (the boundary cap: rows past E are wrong by design, design 2.4) must equal its native row
     bit for bit. Tickets: every start of --k2-sweep (128..300: families 256 and 512, every row slot and mask word
     there), then s = E - 256 + r for E in --cb2-extents and r in --cb2-starts; per variant (normal, peaky: the
     peaky rows aim at their own key and at the first key past it). RECORDED, not decisive: the tickets of
     --k2-floor (100..127), their rows below 128 (the admission floor's other side, expected to differ). Liveness
     (peaky only, on the reference path): the native call at cur_pos p + 1 must move the ticket's first row against
     the native call at p.
  X7 DECISIVE (design 2.3 item 5): 0x7 with the narrow mask == 0x7 with the wide mask carrying the real -inf tail
     (served_mask at capacity E with the word s), both at capacity E on the table truncated to E / 64; and the served
     0x27 at E - 1 on the poisoned C-wide table == that 0x7 wide call; E in --cb2-extents, s = E - 256 + r for r in
     --cb2-starts, per variant. Liveness, on the reference path: per family, the 0x7 wide call with a zero mask must
     move every entry whose tail masks a key.
  Z  DECISIVE (design B5): the stale-writer zone. G8B2 0x27 captured once on the clean C-wide table and replayed at
     every family of --z-families (256 ... 3,840: the 15 with E / 256 < 16 cores per head) and s = E - 256 + r, r in
     --z-starts (0 and 32, the idle segments' starts, then 7, 127, 240, 255), the words and the narrow mask
     rewritten between replays: each replay == the eager call == the compile-time 0x7 call at E (the table truncated
     to E / 64, the same narrow mask). A hang is the watchdog's (exit 3), never a verdict.
K4 (the mask kernel at capacity 256 against its host mirror) is W10b's R1 (extent_reader_card_b.py): the report lists
it as not_run.

The log: every 0x20 program built must log one F4 '[QWEN-SDPA] flags=' line with 0x20 set, followed by exactly one
F22 '[QWEN-SDPA] runtime-extent entries=' line carrying its B, kv_share and q_slice; a requested program without its
lines means the graft was mounted but not executed. The F4 cb_bytes of each 0x20 program at 131,328 keys is
recorded against the K64i table (card.CB_BYTES): K64j adds the two cur_pos sticks (c_8, c_15).

Verdict: one 'K64J_CARD verdict=...' line.
  PASS         every decisive comparison byte-equal, every liveness control moved, no failure, no decisive section
               cut by the deadline: K1 and K3 hold for 0x20 on this binary.
  FAIL         a decisive comparison differs (or a refusal was accepted) on an otherwise valid run.
  NO-DECISION  a failure (wrong binary or kernels, no compact scratch, a missing factory or F22 line, a section that
               raised, the watchdog), a dead liveness control, a decisive section cut by the deadline, SIGTERM, or
               no decisive comparison ran.
The line also carries k2_verdict=, K2's own verdict, which decides design 6.1's branch: PASS (every K2 row equal: the
strict policy stands), FAIL (a K2 row differs on a valid K2 run: S2 stops for the user's decision D-c), NO-DECISION
(a K2 failure, a global failure, a dead K2 control, K2 cut by the deadline, an error, or no K2 row compared; another
section's failure alone does not void it), not_run.

RUN with run_card_b.sh only (QUAL_CARD, default card B), in the C2 image with graft K64j mounted as the arm mounts
it, WATCHER=1 first. The helpers above the device section import no ttnn and are tested on CPU by
test_k64j_card_b.py and test_k64j_cb2a.py, which also run the whole flow on a fake ttnn whose broken variants the
sections must catch.
"""

import argparse
import json
import mmap
import os
from pathlib import Path
import re
import signal
import statistics
import sys

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / 'k64j_probe', HERE.parent / 'sdpa_decode_qwen'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import probe_k64j_card_b as probe  # noqa: E402 - the pool, the poison, the masks, the watchdog, the deadline
import split_model  # noqa: E402 - the families

k1, card = probe.k1, probe.card

CARD = 'K64J_CARD'
PLAN = ('c2-serve-for-real-plan.md section 2.3 (K1, K3); s2-design.md W10a / CB2a (K2, X7, Z); K4 is W10b R1, not '
        'in this harness')
CAPACITY = probe.CAPACITY
EXTENTS = probe.EXTENTS
STARTS = probe.STARTS
SEEDS = (0, 1, 2)
VARIANTS = ('normal',)
K_CHUNK = split_model.K_CHUNK
MAGIC, LEGACY = card.MAGIC, card.LEGACY
TAIL, SHARE, SLICE, READAHEAD = 0x1, 0x2, 0x4, 0x8
EXTENT = 0x20                           # apply_factory_k64j.FLAG_EXTENT
UNKNOWN_CONTROL = 0x10                  # apply_factory_k64j.UNKNOWN_FLAG_CONTROL: still refused
KNOWN_FLAGS = TAIL | SHARE | SLICE | READAHEAD | EXTENT
SHAPES = {'G4B3': (4, 3), 'G8B2': (8, 2), 'G4B1': (4, 1)}   # rows per fold group, entries (G4B1: K's idle call)
BUNDLE_SHAPES = ('G4B3', 'G8B2')
FLAG_SETS = (0x21, 0x23, 0x27, 0x2F)
TRACE_COMBOS = (('G4B3', 0x21), ('G8B2', 0x27))
TRACE_FAMILIES = 64
TRACE_REFERENCES = 8
SKIP_EXTENTS = probe.SKIP_EXTENTS
SKIP_PATTERNS = probe.SKIP_PATTERNS
FENCE_OFFSETS = probe.FENCE_OFFSETS
DEFAULT_SECTIONS = ('N', 'X', 'M', 'K', 'L', 'T')      # K1 / K3 (CB1)
CB2A_SECTIONS = ('K2', 'X7', 'Z')                       # s2-design.md W10a (CB2a): opt-in
SECTIONS = DEFAULT_SECTIONS + CB2A_SECTIONS
RUN_ORDER = ('N', 'X', 'M', 'K', 'L', 'T', 'K2', 'X7', 'Z', 'timing')
FIRST_SEED_ONLY = ('N', 'L', 'T', 'timing')
DECISIVE_SECTIONS = ('N', 'X', 'M', 'K', 'T', 'K2', 'X7', 'Z')
DECISIVE_KINDS = ('refusal', 'extent_vs_reference', 'mixed_vs_reference', 'share_slot0', 'skip_live',
                  'share_skip_slot0_wins', 'trace_vs_eager', 'trace_vs_reference', 'trace_skip_live',
                  'trace_fence_vs_clean', 'k2_native_vs_extent', 'x7_narrow_vs_wide', 'x7_extent_vs_wide',
                  'z_trace_vs_eager', 'z_trace_vs_reference')
DEADLINE_MARGIN_S = probe.DEADLINE_MARGIN_S
OPEN_EXTRA_S = probe.OPEN_EXTRA_S
ENV_RECORDED = probe.ENV_RECORDED
KERNEL_ROOT = probe.KERNEL_ROOT
NOT_RUN = {'K4': 'the mask kernel at capacity 256 against its host mirror is W10b R1 (extent_reader_card_b.py), '
                 'not a section of this harness'}

# CB2a (s2-design.md W10a, 2.3, 2.4, 6.1, 6.2).
TICKET_ROWS = 16                                        # one served ticket (a packed round's rows per user)
SERVED_ROWS, SERVED_BATCH = 8, 2                        # LAYOUT(16, 8) = parallel_groups(256, 16, max_group_rows=8)
SERVED_OFFSETS = (0, 8)                                 # its one bundle's group offsets: one G8B2 call per ticket
SERVED_FLAGS = TAIL | SHARE | SLICE | EXTENT            # 0x27: the extent reader's call at G8B2
COMPILE_FLAGS = SERVED_FLAGS & ~EXTENT                  # 0x7: v235's served call, at capacity E
MIN_LIVE_START = 128                                    # the admission floor (design 1.4 #9, W1 MIN_LIVE_START)
K2_SWEEP = (128, 300)                                   # every ticket start in it (families 256 and 512)
K2_FLOOR = (100, 127)                                   # recorded: the floor's other side
CB2_EXTENTS = (2304, 4352, 16640, 65792, 131328)        # K2's and X7's families
CB2_STARTS = (0, 7, 127, 240, 255)                      # s mod 256 of their tickets
Z_FAMILIES = tuple(range(K_CHUNK, 4096, K_CHUNK))       # the 15 families with E / 256 < 16 cores per head
IDLE_STARTS = (0, 32)                                   # an idle segment's start: page 0, tile row 0 or 1
Z_STARTS = IDLE_STARTS + (7, 127, 240, 255)
OUTPUT_MEMORY = ('l1', 'dram')

# K2's reference: the model's native decode call (docker/qwen-c2-graft/graft/attention/tp.py, the graft the serving
# image installs over the model tree, docker/qwen-c2-serving.Dockerfile:30-45). test_k64j_cb2a parses that source and
# keeps these equal to it; SerialAttentionReader forwards the kwargs unchanged, one B = 1 call per row.
NATIVE_PROGRAM_CONFIG = dict(exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0)   # + the device grid
NATIVE_CALL_KWARGS = ('page_table_tensor', 'cur_pos_tensor', 'scale', 'program_config', 'memory_config')
NATIVE_SCALE = 256 ** -0.5                              # tp.py:134 self.scale = self.HD ** -0.5, HD 256
NATIVE_DECODE = dict(
    source='docker/qwen-c2-graft/graft/attention/tp.py',
    sha256='749a9814453f71f419708f738f7ce2211cf5cc56f220f3052d43e4840a53d153',   # docker/qwen-c2-graft/graft.sha256
    lines=dict(program_config='737-742', call='743-754', scale='134', forward_decode_same='622-627, 645-656'),
    program_config=dict(NATIVE_PROGRAM_CONFIG, compute_with_storage_grid_size='the device grid (tp.py:736)'),
    call=list(NATIVE_CALL_KWARGS), scale=NATIVE_SCALE, memory_config='L1 (tp.py:753)',
    not_passed=['is_causal', 'attn_mask', 'compute_kernel_config'],
    solo_path='scripts/ci/attention_batch.py:82-113 SerialAttentionReader: per row one B = 1 call, the query row '
              'sliced to DRAM (:102-103), the row\'s own (1,) cur_pos and (1, W) table, **kwargs forwarded (:105)')
UNVERIFIED = (
    'is_causal: the native call does not pass it (tp.py:743-754) and neither does this harness, so both take the op '
    'binding\'s default; that it is True is inferred (a non-causal paged call with k_chunk_size 0 is refused), not '
    'read in this repo',
    'compute_kernel_config: passed by neither the native call nor the served replay (attention_parallel.py:17-19), so '
    'both take the op default; its value (HiFi2, fp32_dest_acc_en false per the k64j_probe README reading of '
    'sdpa_decode.cpp:77) is a tt-metal source outside this repo',
    'the query dtype bf16: tp.py:6 and no typecast in _decode_from_prep; attn_decode_prep\'s output dtype is the op\'s',
    'the query shape (1, 1, 12, 256), HD 256 and 12 local heads: from the attn_decode_prep log line '
    '(docs/gdn-conv-path-2026-09-19.md:165), not from source (model_config.py reads them from the checkpoint)',
    'the one-row engine: SerialAttentionReader passes attn_decode_prep\'s q unsliced (attention_batch.py:96-99), in '
    'the layout the op returns; this harness runs the multi-row engines\' shape, a DRAM-interleaved row slice',
    'the paged K / V: bf8 (QWEN_SDPA_BF8=1, docker/qwen-c2-serving.Dockerfile:89; tp.py:846-847) in 64-key blocks, '
    'as this harness\'s pool (probe.Pool) holds them',
)

# The K64j factory's literals (apply_factory_k64j.py; test_k64j_card_b keeps these equal to it).
EXTENT_LOG_MARKER = '[QWEN-SDPA] runtime-extent entries='
EXTENT_BINARY_MARKER = EXTENT_LOG_MARKER.encode()
EXTENT_MASK_REFUSAL = '[QWEN-SDPA] runtime extent (0x20) needs the tail flag (0x1) and a narrow'
EXTENT_CUR_POS_REFUSAL = '[QWEN-SDPA] runtime extent (0x20) needs an interleaved cur_pos tensor'
EXTENT_LAYOUT_REFUSAL = '[QWEN-SDPA] runtime extent (0x20) needs an int32 row-major cur_pos tensor of B='
CUR_POS_REFUSAL = '[QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor'
UNKNOWN_NEEDLE = probe.UNKNOWN_NEEDLE
SLICE_BINARY_MARKER = probe.SLICE_BINARY_MARKER
SCRATCH_BINARY_MARKER = card.SCRATCH_ENV.encode()
EXTENT_LINE = re.compile(r'\[QWEN-SDPA\] runtime-extent entries=([0-9]+) kv_share=([a-z]+) q_slice=([a-z]+) '
                         r'writer=(\S+) cur_pos_stick_bytes=([0-9]+)')
# The four K64j qwen kernels (make_k64j_kernels.OUTPUTS) and the five stock sources (probe.STOCK_KERNELS) under
# the mounted sdpa_decode/device/kernels.
K64J_KERNELS = {
    'dataflow/reader_decode_qwen.cpp': 'adb6091878ba3f0a0805846ff56f05352610d7fe779b5ae96320c437c095db49',
    'dataflow/reader_decode_qwen_slice.cpp': '518d8096e3cceb160eaef8ab4f0ae976ccbffd3904d31176b7f9d02828c37f8a',
    'compute/sdpa_flash_decode_qwen.cpp': '409a1aafc3ffaaca2c6afba0e999525b7d141491f0a52e5583efa70447ee5c0e',
    'dataflow/writer_decode_qwen_slice.cpp': '642c36f809be0f1ad1664deb405dc310dabaa32d5628710320a118eb37a6cc7a',
}
STOCK_KERNELS = probe.STOCK_KERNELS


# ---------------------------------------------------------------------------------------------
# Pure helpers (no ttnn).
# ---------------------------------------------------------------------------------------------

def valid_combo(shape, flags):
    """A (shape, flags) the K64j factory builds and this harness runs: 0x20 with the tail flag; share needs B > 1;
    the slice only where it saves a tile (the 8-row groups); read-ahead needs share."""
    if shape not in SHAPES or flags & ~KNOWN_FLAGS or not flags & EXTENT or not flags & TAIL:
        return False
    rows, batch = SHAPES[shape]
    if flags & SHARE and batch < 2:
        return False
    if flags & SLICE and not probe.q_slice_saves(rows):
        return False
    return not (flags & READAHEAD and not flags & SHARE)


def default_combos():
    return [(shape, flags) for shape in BUNDLE_SHAPES for flags in FLAG_SETS if valid_combo(shape, flags)]


def parse_combos(text):
    """'G4B3:0x21,G8B2:0x27' -> [(shape, flags)]; ValueError on an unknown or invalid combo."""
    out = []
    for token in (part.strip() for part in text.split(',')):
        if not token:
            continue
        shape, sep, value = token.partition(':')
        try:
            flags = int(value, 16) if sep else None
        except ValueError:
            flags = None
        if flags is None or not valid_combo(shape, flags):
            raise ValueError('not a runnable (shape, flags) combo: %r' % token)
        if (shape, flags) not in out:
            out.append((shape, flags))
    return out


def combo_name(shape, flags):
    return '%s:0x%x' % (shape, flags)


def reference_flags(flags):
    """The compile-time served call a 0x20 call is compared with: the same flags without 0x20."""
    return flags & ~EXTENT


def bundle_positions(extent, start, batch, rows):
    """One bundle's entry positions in family E: the replay's consecutive row groups of one user, entry b at
    E - 256 + min(255, s + b * rows) (a bundle never crosses its family: the boundary cap)."""
    return [extent - K_CHUNK + min(K_CHUNK - 1, start + slot * rows) for slot in range(batch)]


def share_positions(first, batch, rows):
    """The positions of a share bundle whose entry 0 sits at `first`: the rest follow in entry 0's family."""
    extent = split_model.extent(first)
    return bundle_positions(extent, first - (extent - K_CHUNK), batch, rows)


def words_for(positions, share):
    """The cur_pos words the extent readers write: E - 1 per entry (-1 kept: a skipped entry); under share every
    slot carries slot 0's word, which is the one the kernels read (R10)."""
    words = [-1 if position < 0 else split_model.extent(position) - 1 for position in positions]
    return [words[0]] * len(words) if share else words


def mixed_extents(extents, batch):
    """`batch` different families spread over the requested extents (cycled when fewer)."""
    ordered = sorted(set(extents))
    if len(ordered) >= batch:
        return [ordered[(slot * (len(ordered) - 1)) // max(1, batch - 1)] for slot in range(batch)]
    return [ordered[slot % len(ordered)] for slot in range(batch)]


def share_slot_extents(extents, capacity, batch):
    """(slot 0's family, the other slots' families) for the share slot-0 control: slot 0 a middle family below the
    capacity (poison past it), the others the remaining families, below and above it (cycled)."""
    below = sorted(extent for extent in set(extents) if extent < capacity) or [K_CHUNK]
    base = below[len(below) // 2]
    others = sorted(extent for extent in set(extents) if extent != base) or [min(capacity, base + K_CHUNK)]
    return base, [others[slot % len(others)] for slot in range(batch - 1)]


def reference_sample(count, wanted):
    """Which of `count` trace replays get a compile-time reference: `wanted` evenly spaced (the first and last)."""
    if wanted <= 0 or count <= 0:
        return set()
    if wanted >= count:
        return set(range(count))
    if wanted == 1:
        return {0}
    return {round(index * (count - 1) / (wanted - 1)) for index in range(wanted)}


def mask_row_positions(word, rows=SERVED_ROWS, batches=SERVED_BATCH, offset=0):
    """attention_mask_replay.cpp:24 (and attention_mask_replay.mask_position): folded row h of entry b sits at
    word + offset + b * rows + (h % (rows * 6)) / 6 - [[position per row] per entry]."""
    return [[word + offset + batch * rows + (head % (rows * 6)) // 6 for head in range(rows * 12)]
            for batch in range(batches)]


def served_mask(torch, word, capacity, width=None, rows=SERVED_ROWS, batches=SERVED_BATCH, offset=0):
    """The pinned mask kernel on the host (attention_mask_replay.cpp:18-33) run with the start word `word` at
    `capacity`, into a zero-initialised tensor (attention_replay.py:51): (batches, 1, rows * 12, width) bf16, -inf
    (0xff80) in the last 256 columns - the cache positions [capacity - 256, capacity) - wherever the column's position
    is past the row's (mask_row_positions), +0.0 elsewhere. At rows 8 there is no padding head (h < 96 = rows * 12,
    cpp :28). width (default capacity) keeps the tensor's LAST width columns: the wide mask is width = capacity."""
    width = capacity if width is None else width
    if (any(type(value) is not int for value in (word, capacity, width)) or word < 0 or capacity < K_CHUNK
            or capacity % K_CHUNK or not K_CHUNK <= width <= capacity):
        raise ValueError('served_mask needs a word >= 0, a 256-aligned capacity and a width of 256 .. capacity')
    positions = torch.tensor(mask_row_positions(word, rows, batches, offset), dtype=torch.int64)    # (B, rows*12)
    cache = torch.arange(capacity - K_CHUNK, capacity, dtype=torch.int64)                           # cpp :27
    tail = torch.where(cache[None, None, :] > positions[:, :, None], float('-inf'), 0.0)            # cpp :28
    mask = torch.zeros(batches, 1, rows * 12, width, dtype=torch.float32)
    mask[:, 0, :, width - K_CHUNK:] = tail
    return mask.to(torch.bfloat16)


def narrow_mask(torch, start):
    """The extent reader's (2, 1, 96, 256) mask for a ticket at `start`: the pinned kernel at capacity 256 with the
    relative word start & 255 (design 1.4 #2, 2.3 item 2)."""
    return served_mask(torch, start & (K_CHUNK - 1), K_CHUNK)


def wide_mask(torch, start, extent):
    """v235's (2, 1, 96, E) mask for the same ticket at capacity E: the pinned kernel with the absolute word."""
    return served_mask(torch, start, extent)


def accept_limit(start, rows=TICKET_ROWS):
    """design 2.4: a ticket at `start` commits at most min(rows, E - start) rows, those at positions below E."""
    return min(rows, split_model.extent(start) - start)


def ticket_positions(start, rows=TICKET_ROWS):
    return list(range(start, start + rows))


def valid_positions(start, rows=TICKET_ROWS):
    """The ticket's rows the boundary cap commits (positions < E): the rows K2 compares."""
    return list(range(start, start + accept_limit(start, rows)))


def native_cur_pos(position):
    """The solo path's cur_pos word for the row at `position`: the row's own position (SerialAttentionReader gives
    each row its singleton position tensor, attention_batch.py:101-105; the model's cur_pos_tt)."""
    return position


def k2_tickets(sweep, floor, extents, starts):
    """K2's tickets in run order, decisive first: every start of `sweep` (lo, hi inclusive), then E - 256 + r for E in
    `extents` and r in `starts`, then the recorded `floor` starts (None: none)."""
    tickets = [dict(kind='sweep', start=start, extent=split_model.extent(start))
               for start in range(sweep[0], sweep[1] + 1)]
    tickets += [dict(kind='family', start=extent - K_CHUNK + offset, extent=extent)
                for extent in extents for offset in starts]
    if floor:
        tickets += [dict(kind='floor', start=start, extent=split_model.extent(start))
                    for start in range(floor[0], floor[1] + 1)]
    return tickets


def k2_compared(ticket):
    """The rows of a K2 ticket that are compared: every committed row (decisive), or, for a floor ticket, its
    committed rows below the admission floor (recorded)."""
    rows = valid_positions(ticket['start'])
    return rows if ticket['kind'] != 'floor' else [position for position in rows if position < MIN_LIVE_START]


def z_plan(families, starts):
    """Z's replays in order: (E, s) for every family and start, s = E - 256 + r."""
    return [(extent, extent - K_CHUNK + offset) for extent in families for offset in starts]


def token_query(torch, seed, variant, position, keys=None, table=None):
    """One token's (12, 256) bf16 query at `position`: the same bytes for the solo row and inside the packed ticket.
    normal: N(0, 1). peaky: 0.1 x N(0, 1) plus, per head, 6 x the unit vector of its KV head's key at the token's own
    position, at the first position PAST it (masked on both paths: a boundary read one key too far moves the row),
    at two earlier positions in its final chunk (anywhere earlier when it opens the chunk) and three anywhere earlier
    (the host keys through `table`), so the own key and the one past it weigh alike."""
    generator = torch.Generator().manual_seed((seed * 1000003 + position) * 2 + (1 if variant == 'peaky' else 0))
    tokens = torch.randn(12, card.HEAD_DIM, generator=generator)
    if variant == 'peaky':
        if keys is None or table is None:
            raise ValueError('Peaky queries need the host keys and the page table')
        tokens *= 0.1
        last = len(table) * card.PAGE - 1
        chunk = split_model.extent(position) - K_CHUNK
        earlier = chunk if chunk < position else 0
        for head in range(12):
            kv = head // 6
            aims = [position, position + 1]
            if position > 0:
                aims += torch.randint(earlier, position, (2,), generator=generator).tolist()
                aims += torch.randint(0, position, (3,), generator=generator).tolist()
            for aim in aims:
                aim = min(aim, last)
                vector = keys[int(table[aim // card.PAGE]), kv, aim % card.PAGE].float()
                tokens[head] += 6 * vector / vector.norm().clamp_min(1e-3)
    elif variant != 'normal':
        raise ValueError('Unknown query variant %r' % (variant,))
    return tokens.to(torch.bfloat16)


def ticket_query(torch, tokens, start):
    """The ticket's (1, 2, 96, 256) G8B2 query: its 16 token queries (tokens: position -> (12, 256)) folded as the
    extent reader's device fold lays them (attention_head_fold.fold_query per group, groups at SERVED_OFFSETS)."""
    host = torch.stack([tokens(position) for position in ticket_positions(start)])[None]
    return card.fold_entries(torch, host, SERVED_OFFSETS, SERVED_ROWS)


def parse_range(text, name):
    """'lo:hi' (inclusive) -> (lo, hi); '' -> None."""
    if not text:
        return None
    lo, sep, hi = text.partition(':')
    if not sep:
        raise ValueError('%s must be lo:hi, got %r' % (name, text))
    lo, hi = int(lo), int(hi)
    if not 0 <= lo <= hi:
        raise ValueError('%s must have 0 <= lo <= hi, got %r' % (name, text))
    return lo, hi


def section_runs(args):
    runs = []
    for seed in args.seeds:
        for name in RUN_ORDER:
            if name == 'timing' and args.no_timing:
                continue
            if name != 'timing' and name not in args.sections:
                continue
            if name in FIRST_SEED_ONLY and seed != args.seeds[0]:
                continue
            runs.append((seed, name))
    return runs


def run_tag(seed, name):
    return '%s/seed%d' % (name, seed)


def extent_lines(text):
    """The factory's F4 and F22 lines in log order: [('F4', fields) | ('F22', fields)]."""
    events = []
    for match in card.FACTORY_LINE.finditer(text):
        events.append((match.start(), 'F4', card.factory_lines(match.group(0))[0]))
    for match in EXTENT_LINE.finditer(text):
        events.append((match.start(), 'F22', dict(entries=int(match.group(1)), kv_share=match.group(2),
                                                  q_slice=match.group(3), writer=match.group(4),
                                                  stick=int(match.group(5)))))
    return [(kind, fields) for _offset, kind, fields in sorted(events, key=lambda event: event[0])]


def extent_line_problems(events):
    """Every F4 line with 0x20 is followed by exactly one F22 line (before the next F4) with its B, kv_share and
    q_slice; no F22 line without such an F4 line; no F22 line on a program without 0x20."""
    problems = []
    for index, (kind, fields) in enumerate(events):
        if kind != 'F4':
            continue
        flags = int(fields['flags'], 16)
        following = []
        for later_kind, later in events[index + 1:]:
            if later_kind == 'F4':
                break
            following.append(later)
        if not flags & EXTENT:
            if following:
                problems.append('an F22 line after the F4 line of flags=%s B=%d (no 0x20)' % (fields['flags'], fields['B']))
            continue
        name = 'flags=%s B=%d St=%d' % (fields['flags'], fields['B'], fields['St'])
        if len(following) != 1:
            problems.append('%s: %d F22 lines, expected one' % (name, len(following)))
            continue
        line = following[0]
        expected = dict(entries=fields['B'], kv_share=fields['kv_share'], q_slice='true' if flags & SLICE else 'false',
                        writer='writer_decode_qwen_slice.cpp')
        wrong = {key: (line[key], value) for key, value in expected.items() if line[key] != value}
        if wrong:
            problems.append('%s: F22 (found, expected) %r' % (name, wrong))
    first_f4 = next((index for index, (kind, _fields) in enumerate(events) if kind == 'F4'), len(events))
    if any(kind == 'F22' for kind, _fields in events[:first_f4]):
        problems.append('an F22 line before any F4 line')
    return problems


def cb_extras(events, capacity=CAPACITY):
    """Each 0x20 program's F4 cb_bytes against K64i's table at the same capacity and PNHt (card.CB_BYTES): K64j
    adds the c_8 and c_15 cur_pos sticks. Only where the table has an entry."""
    out = []
    for kind, fields in events:
        if kind != 'F4' or not int(fields['flags'], 16) & EXTENT:
            continue
        key = (fields['St'] * split_model.TILE, fields['PNHt'])
        if key[0] != capacity or key not in card.CB_BYTES:
            continue
        out.append(dict(flags=fields['flags'], B=fields['B'], PNHt=fields['PNHt'], cb_bytes=fields['cb_bytes'],
                        k64i_cb_bytes=card.CB_BYTES[key], extra=fields['cb_bytes'] - card.CB_BYTES[key]))
    return out


def tally(comparisons):
    return probe.tally(comparisons)


def decide(report):
    """PASS / FAIL / NO-DECISION from the comparisons, liveness, failures, error and deadline."""
    comparisons = report.get('comparisons', [])
    decisive = [entry for entry in comparisons if entry['decisive']]
    differing = [entry for entry in decisive if entry['differing']]
    reasons = []
    if report.get('failures'):
        reasons.append('measurement invalid: %d failures' % len(report['failures']))
    if report.get('error'):
        reasons.append('error: %s' % report['error'])
    dead = [entry for entry in report.get('liveness', []) if not entry['live']]
    if dead:
        reasons.append('%d liveness controls did not move (%s): the poison or the off-by-one control is dead'
                       % (len(dead), ', '.join(entry['label'] for entry in dead[:4])))
    deadline = report.get('deadline') or {}
    cut = [tag for tag in deadline.get('skipped', []) if tag.split('/')[0] in DECISIVE_SECTIONS]
    if cut:
        reasons.append('the deadline (%ss) cut %d decisive section runs (%s)'
                       % (deadline.get('seconds'), len(cut), ', '.join(cut[:6])))
    if not decisive:
        reasons.append('no decisive comparison ran')
    if reasons:
        verdict = 'NO-DECISION'
    elif differing:
        verdict = 'FAIL'
    else:
        verdict = 'PASS'
    return dict(verdict=verdict, reasons=reasons, decisive=len(decisive), decisive_differing=len(differing),
                first_differing=[entry['label'] for entry in differing[:6]], k2=k2_verdict(report))


def k2_verdict(report):
    """K2's own verdict, the one design 6.1 branches on: not_run (K2 not requested); NO-DECISION on an error, a
    failure that is K2's or no section's (the binary, the kernels, the scratch, a pool, the factory log), a dead K2
    liveness control, a K2 run cut by the deadline, or no K2 row compared; otherwise FAIL if any K2 row differs, else
    PASS. Another section's failure ('X7/...', 'Z/...') does not void K2."""
    if 'K2' not in (report.get('sections') or ()):
        return 'not_run'
    others = set(SECTIONS) - {'K2'}
    if report.get('error') or any(failure.split('/')[0] not in others for failure in report.get('failures', [])):
        return 'NO-DECISION'
    if any(entry['section'] == 'K2' and not entry['live'] for entry in report.get('liveness', [])):
        return 'NO-DECISION'
    if any(tag.split('/')[0] == 'K2' for tag in (report.get('deadline') or {}).get('skipped', [])):
        return 'NO-DECISION'
    rows = [entry for entry in report.get('comparisons', []) if entry['kind'] == 'k2_native_vs_extent']
    if not rows:
        return 'NO-DECISION'
    return 'FAIL' if any(entry['differing'] for entry in rows) else 'PASS'


def verdict_line(report):
    decision = report['decision']
    counts = tally(report.get('comparisons', []))
    words = [CARD, 'verdict=%s' % decision['verdict']]

    def part(name, kinds):
        runs = sum(counts.get(kind, {}).get('runs', 0) for kind in kinds)
        equal = sum(counts.get(kind, {}).get('equal', 0) for kind in kinds)
        return '%s=%d/%d' % (name, equal, runs) if runs else '%s=none' % name

    words.append(part('extent', ('extent_vs_reference',)))
    words.append(part('mixed', ('mixed_vs_reference',)))
    words.append(part('share_slot0', ('share_slot0', 'share_skip_slot0_wins')))
    words.append(part('trace', ('trace_vs_eager', 'trace_vs_reference')))
    words.append(part('fence', ('trace_fence_vs_clean',)))
    words.append(part('skip', ('skip_live', 'trace_skip_live')))
    words.append(part('refusals', ('refusal',)))
    liveness = report.get('liveness', [])
    words.append('live=%d/%d' % (sum(1 for entry in liveness if entry['live']), len(liveness)))
    if report.get('skip_written'):
        words.append('skipped_rows=%s' % report['skip_written'])
    words.append('families=%d' % report.get('trace_families_distinct', 0))
    words.append(part('k2', ('k2_native_vs_extent',)))
    rows = report.get('k2_rows') or {}
    if rows.get('compared'):
        words.append('k2_rows=%d/%d' % (rows['equal'], rows['compared']))
    if rows.get('floor'):
        words.append('k2_floor_differing=%d/%d' % (rows['floor_differing'], rows['floor']))
    words.append('k2_verdict=%s' % decision.get('k2', 'not_run'))
    words.append(part('x7', ('x7_narrow_vs_wide', 'x7_extent_vs_wide')))
    words.append(part('z', ('z_trace_vs_eager', 'z_trace_vs_reference')))
    if report.get('z_families'):
        words.append('z_families=%d' % len(report['z_families']))
    words.append('k4=not_run')
    if report.get('sections_failed'):
        words.append('sections_failed=%s' % ','.join(report['sections_failed']))
    if (report.get('deadline') or {}).get('skipped'):
        words.append('deadline_skipped=%d' % len(report['deadline']['skipped']))
    if decision['first_differing']:
        words.append('first_differing=%s' % json.dumps(decision['first_differing']))
    if decision['reasons']:
        words.append('reasons=%s' % json.dumps(decision['reasons']))
    return ' '.join(words)


def binary_literals(path):
    """Which of the K64j, q-slice and compact-scratch literals the _ttnncpp.so at `path` carries."""
    with open(path, 'rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
        return dict(extent=view.find(EXTENT_BINARY_MARKER) >= 0, slice=view.find(SLICE_BINARY_MARKER) >= 0,
                    scratch=view.find(SCRATCH_BINARY_MARKER) >= 0)


# ---------------------------------------------------------------------------------------------
# Device harness.
# ---------------------------------------------------------------------------------------------

class ExtentPool(probe.Pool):
    """The probe's pool (the seed's bf8 K/V, the poison blocks, one page table per user) with the 0x20 calls."""

    def words(self, values):
        return self.positions(values)

    def mask(self, positions, extents, rows, zero=False):
        return self.upload(probe.position_mask(self.torch, positions, extents, rows, K_CHUNK, zero=zero))

    def extent_run(self, query, pages, words, mask, flags, label):
        return self.run(query, pages, causal=False, cur_pos=words, mask=mask, sentinel=MAGIC | flags, label=label)

    def reference_run(self, query, pages, mask, flags, label):
        return self.run(query, pages, causal=False, mask=mask, sentinel=MAGIC | reference_flags(flags), label=label)

    # CB2a: the served calls' own kwargs.
    output = 'l1'                                       # --output-memory: the model's memory_config (tp.py:753)

    def output_memory(self):
        return self.ttnn.L1_MEMORY_CONFIG if self.output == 'l1' else self.ttnn.DRAM_MEMORY_CONFIG

    def grid(self):
        grid = self.device.compute_with_storage_grid_size()
        return (grid.x, grid.y)

    def served_launch(self, query, pages, mask, flags, *, words=None, label):
        """The replay's call as attention_parallel.execute issues it (attention_parallel.py:17-19) with the extent
        reader's cur_pos_tensor (design W1 execute_extent) when `words` is given: page_table_tensor by keyword,
        is_causal False, the mask, the model's scale and memory_config, and the pooled config (the device grid,
        exp_approx_mode False, q_chunk_size the qwen sentinel, k_chunk_size 256). Recorded as a requested program."""
        ttnn = self.ttnn
        config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid(), exp_approx_mode=False,
                                        q_chunk_size=MAGIC | flags, k_chunk_size=K_CHUNK)
        options = dict(page_table_tensor=pages, is_causal=False, attn_mask=mask, scale=NATIVE_SCALE,
                       program_config=config, memory_config=self.output_memory())
        if words is not None:
            options['cur_pos_tensor'] = words
        self.report['_requested'].add(probe.program_key(int(pages.shape[1]) * card.PAGE, int(query.shape[1]),
                                                        int(mask.shape[3]), flags))
        with probe.WATCHDOG.op(label):
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(query, self.k, self.v, **options)

    def served_run(self, query, pages, mask, flags, *, words=None, label):
        return self.host(self.served_launch(query, pages, mask, flags, words=words, label=label), 'read back ' + label)

    def native_launch(self, query, pages, cur_pos, label):
        """K2's reference: the native decode call with exactly the kwargs tp.py:743-754 passes (NATIVE_DECODE) -
        page_table_tensor, cur_pos_tensor, scale, program_config (the device grid, exp_approx_mode False,
        q_chunk_size 0, k_chunk_size 0) and memory_config; is_causal, attn_mask and compute_kernel_config are not
        passed, as there."""
        ttnn = self.ttnn
        config = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid(), **NATIVE_PROGRAM_CONFIG)
        with probe.WATCHDOG.op(label):
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query, self.k, self.v, page_table_tensor=pages, cur_pos_tensor=cur_pos, scale=NATIVE_SCALE,
                program_config=config, memory_config=self.output_memory())

    def native_run(self, query, pages, cur_pos, label):
        return self.host(self.native_launch(query, pages, cur_pos, label), 'read back ' + label)


def query_for(torch, pool, batch, rows, seed, variant, extent, salt=0):
    """(1, batch, rows * 12, 256): card.build_query ('peaky' aims at keys of user 0's first E keys)."""
    table = pool.tables[0][:extent // card.PAGE] if variant == 'peaky' else None
    keys = pool.keys if variant == 'peaky' else None
    return card.build_query(torch, batch, seed + 31 * salt, variant, keys, table, rows=rows)


def record(report, entry, verbose=True):
    probe.record(report, entry, verbose)


def repeated(torch, host, index, batch):
    return host[:, index:index + 1].repeat(1, batch, 1, 1).contiguous()


def section_refusals(ttnn, torch, pool, args, report):
    """N: every call the K64j factory must refuse, refused with its literal."""
    rows, batch = SHAPES['G4B3']
    extent = min(args.extents)
    positions = bundle_positions(extent, 0, batch, rows)
    scope = probe.Scope(ttnn)
    results = {}
    try:
        query = scope.keep(pool.upload(query_for(torch, pool, batch, rows, 0, 'normal', extent, salt=40)))
        pages = scope.keep(pool.pages_reference(0, extent, batch))
        narrow = scope.keep(pool.mask(positions, [extent] * batch, rows))
        wide = scope.keep(pool.upload(probe.position_mask(torch, positions, [extent] * batch, rows, extent)))
        words = scope.keep(pool.words([extent - 1] * batch))
        extra = scope.keep(pool.words([extent - 1] * (batch + 1)))
        cases = (
            ('unknown flag 0x10 (0x11)', dict(mask=narrow, sentinel=MAGIC | UNKNOWN_CONTROL | TAIL), UNKNOWN_NEEDLE),
            ('0x20 without the tail flag', dict(mask=narrow, cur_pos=words, sentinel=MAGIC | EXTENT),
             EXTENT_MASK_REFUSAL),
            ('0x21 with a wide mask', dict(mask=wide, cur_pos=words, sentinel=MAGIC | EXTENT | TAIL), EXTENT_MASK_REFUSAL),
            ('0x21 without a cur_pos tensor', dict(mask=narrow, sentinel=MAGIC | EXTENT | TAIL), EXTENT_CUR_POS_REFUSAL),
            ('0x21 with B + 1 words', dict(mask=narrow, cur_pos=extra, sentinel=MAGIC | EXTENT | TAIL),
             EXTENT_LAYOUT_REFUSAL),
            ('0x21 on a causal call', dict(causal=True, cur_pos=words, sentinel=MAGIC | EXTENT | TAIL), CUR_POS_REFUSAL),
            ('0x01 with a cur_pos tensor', dict(mask=narrow, cur_pos=words, sentinel=MAGIC | TAIL), CUR_POS_REFUSAL),
        )
        for name, call, needle in cases:
            label = 'N/%s' % name
            probe.DEADLINE.check(label)
            try:
                out = pool.launch(query, pages, causal=call.get('causal', False), cur_pos=call.get('cur_pos'),
                                  mask=call.get('mask'), sentinel=call['sentinel'], label=label, expect_program=False)
            except Exception as error:  # noqa: BLE001 - a TT_FATAL surfaces as RuntimeError
                text = ' '.join(str(error).split())
                results[name] = dict(refused=True, matched=needle in text, needle=needle, message=text[:300])
            else:
                ttnn.deallocate(out)
                results[name] = dict(refused=False, matched=False, needle=needle, message='accepted')
            outcome = results[name]
            record(report, probe.comparison('N', 'refusal', label, 0 if outcome['refused'] else 1, True,
                                            matched=outcome['matched']))
            if outcome['refused'] and not outcome['matched']:
                report['warnings'].append('%s: refused, but not with %r: %s' % (label, needle, outcome['message']))
    finally:
        scope.close()
    report['refusals'] = results


def section_extent(ttnn, torch, pool, seed, args, report):
    """X: the 0x20 call at E - 1 on the poisoned C-wide table against the compile-time call at E, per entry."""
    for shape, flags in args.combos:
        rows, batch = SHAPES[shape]
        for extent in args.extents:
            base = 'X/%s/seed%d/E%d' % (combo_name(shape, flags), seed, extent)
            probe.DEADLINE.check(base)
            scope = probe.Scope(ttnn)
            try:
                pages = scope.keep(pool.pages_causal([0] * batch, [extent] * batch))
                reference_pages = scope.keep(pool.pages_reference(0, extent, batch))
                words = scope.keep(pool.words([extent - 1] * batch))
                for start in args.starts:
                    positions = bundle_positions(extent, start, batch, rows)
                    mask = scope.keep(pool.mask(positions, [extent] * batch, rows))
                    for variant in args.variants:
                        label = '%s+%d/%s' % (base, start, variant)
                        probe.DEADLINE.check(label)
                        query = scope.keep(pool.upload(query_for(torch, pool, batch, rows, seed, variant, extent)))
                        got = pool.extent_run(query, pages, words, mask, flags, label + ' extent')
                        reference = pool.reference_run(query, reference_pages, mask, flags, label + ' reference')
                        probe.finite_or_fail(torch, report, label, reference)
                        for index in range(batch):
                            record(report, probe.comparison(
                                'X', 'extent_vs_reference', '%s/entry%d' % (label, index),
                                card.differing(torch, probe.slot(got, index), probe.slot(reference, index)), True,
                                shape=shape, flags=flags, extent=extent, start=start, position=positions[index],
                                seed=seed, variant=variant))
            finally:
                scope.close()


def section_mixed(ttnn, torch, pool, seed, args, report):
    """M: three users at three families in one 0x21 call; and slot 0's word under share."""
    rows, batch = SHAPES['G4B3']
    if ('G4B3', 0x21) in args.combos and len(set(args.extents)) > 1:
        extents = mixed_extents(args.extents, batch)
        users = list(range(batch))
        for start in args.starts:
            label = 'M/mixed/seed%d/%s+%d' % (seed, ','.join(map(str, extents)), start)
            probe.DEADLINE.check(label)
            scope = probe.Scope(ttnn)
            try:
                positions = [extent - K_CHUNK + start for extent in extents]
                host_query = query_for(torch, pool, batch, rows, seed, 'normal', min(extents), salt=7)
                host_mask = probe.position_mask(torch, positions, extents, rows, K_CHUNK)
                query = scope.keep(pool.upload(host_query))
                pages = scope.keep(pool.pages_causal(users, extents))
                mask = scope.keep(pool.upload(host_mask))
                words = scope.keep(pool.words([extent - 1 for extent in extents]))
                got = pool.extent_run(query, pages, words, mask, 0x21, label + ' extent')
                for index, extent in enumerate(extents):
                    ref_query = scope.keep(pool.upload(repeated(torch, host_query, index, batch)))
                    ref_mask = scope.keep(pool.upload(host_mask[index:index + 1].repeat(batch, 1, 1, 1).contiguous()))
                    ref_pages = scope.keep(pool.pages_reference(users[index], extent, batch))
                    reference = pool.reference_run(ref_query, ref_pages, ref_mask, 0x21,
                                                   '%s/entry%d reference' % (label, index))
                    probe.finite_or_fail(torch, report, label, reference)
                    record(report, probe.comparison(
                        'M', 'mixed_vs_reference', '%s/entry%d/E%d' % (label, index, extent),
                        card.differing(torch, probe.slot(got, index), probe.slot(reference, 0)), True,
                        extent=extent, position=positions[index], seed=seed))
            finally:
                scope.close()
    elif ('G4B3', 0x21) in args.combos:
        report['warnings'].append('M/mixed: one extent requested, no mixed families to run')
    for shape, flags in args.combos:
        if not flags & SHARE:
            continue
        rows, batch = SHAPES[shape]
        base_extent, others = share_slot_extents(args.extents, pool.capacity, batch)
        label = 'M/share_slot0/%s/seed%d/E%d' % (combo_name(shape, flags), seed, base_extent)
        probe.DEADLINE.check(label)
        scope = probe.Scope(ttnn)
        try:
            positions = bundle_positions(base_extent, 7, batch, rows)
            query = scope.keep(pool.upload(query_for(torch, pool, batch, rows, seed, 'normal', base_extent, salt=9)))
            pages = scope.keep(pool.pages_causal([0] * batch, [base_extent] * batch))
            mask = scope.keep(pool.mask(positions, [base_extent] * batch, rows))
            mixed_words = [base_extent - 1] + [extent - 1 for extent in others]
            differ = scope.keep(pool.words(mixed_words))
            same = scope.keep(pool.words([base_extent - 1] * batch))
            got = pool.extent_run(query, pages, differ, mask, flags, label + ' mixed words')
            expected = pool.extent_run(query, pages, same, mask, flags, label + ' slot 0 everywhere')
            record(report, probe.comparison('M', 'share_slot0', '%s/words=%s' % (label, ','.join(map(str, mixed_words))),
                                            card.differing(torch, got, expected), True, shape=shape, flags=flags,
                                            words=mixed_words, seed=seed))
        finally:
            scope.close()


def section_skip(ttnn, torch, pool, seed, args, report):
    """K: -1 words, eager, the output poisoned first; under share slot 0 decides; and a B = 1 call that skips."""
    rows, batch = SHAPES['G4B3']
    extents = [min(SKIP_EXTENTS[index % len(SKIP_EXTENTS)], pool.capacity) for index in range(batch)]
    positions = [extent - 1 - 17 * index for index, extent in enumerate(extents)]
    users = list(range(batch))
    written = report.setdefault('skip_rows', [])
    shape = (1, batch, rows * 12, card.HEAD_DIM)
    scope = probe.Scope(ttnn)
    try:
        query = scope.keep(pool.upload(query_for(torch, pool, batch, rows, seed, 'normal', min(extents), salt=13)))
        pages = scope.keep(pool.pages_causal(users, extents))
        mask = scope.keep(pool.mask(positions, extents, rows))
        live_words = [extent - 1 for extent in extents]
        full = pool.extent_run(query, pages, scope.keep(pool.words(live_words)), mask, 0x21, 'K all live')

        def skipped_call(words, flags, the_pages, the_mask, label):
            address = pool.poison_output(shape)
            out = pool.launch(query, the_pages, causal=False, cur_pos=scope.keep(pool.words(words)), mask=the_mask,
                              sentinel=MAGIC | flags, label=label)
            reused = None if address is None else k1.buffer_address(out) == address
            return pool.host(out, 'read back ' + label), reused

        def rows_state(got, index, reused, label):
            nan = int(torch.isnan(probe.slot(got, index).float()).all(dim=-1).sum())
            state = 'unwritten' if nan == rows * 12 else ('written' if nan == 0 else 'partial')
            if reused is not True:
                state += '-unpoisoned'
            written.append(dict(label='%s/entry%d' % (label, index), nan_rows=nan, rows=rows * 12, state=state))

        for pattern in SKIP_PATTERNS:
            if max(pattern) >= batch:
                continue
            label = 'K/seed%d/skip%s' % (seed, ''.join(map(str, pattern)))
            probe.DEADLINE.check(label)
            got, reused = skipped_call(probe.skip_positions(live_words, pattern), 0x21, pages, mask, label)
            live = [index for index in range(batch) if index not in pattern]
            if live:
                differing = sum(card.differing(torch, probe.slot(got, index), probe.slot(full, index)) for index in live)
                record(report, probe.comparison('K', 'skip_live', label, differing, True, pattern=list(pattern),
                                                live=live, seed=seed, poison_address_reused=reused))
            else:
                report['skip_all_call'] = 'returned'             # nothing to compare: it returned, no hang
            for index in pattern:
                rows_state(got, index, reused, label)
        if ('G4B3', 0x23) in args.combos:
            share_extent = extents[1] if len(extents) > 1 else extents[0]
            share_positions_ = bundle_positions(share_extent, 7, batch, rows)
            share_pages = scope.keep(pool.pages_causal([0] * batch, [share_extent] * batch))
            share_mask = scope.keep(pool.mask(share_positions_, [share_extent] * batch, rows))
            share_words = [share_extent - 1] * batch
            share_full = pool.extent_run(query, share_pages, scope.keep(pool.words(share_words)), share_mask, 0x23,
                                         'K share all live')
            label = 'K/seed%d/share/slot0_skipped' % seed
            probe.DEADLINE.check(label)
            got, reused = skipped_call([-1] + share_words[1:], 0x23, share_pages, share_mask, label)
            report['share_skip_slot0'] = 'returned'
            for index in range(batch):
                rows_state(got, index, reused, label)
            label = 'K/seed%d/share/slot1_skipped' % seed
            probe.DEADLINE.check(label)
            got, _reused = skipped_call(share_words[:1] + [-1] + share_words[2:], 0x23, share_pages, share_mask, label)
            record(report, probe.comparison('K', 'share_skip_slot0_wins', label, card.differing(torch, got, share_full),
                                            True, seed=seed))
        # One call whose only user is skipped: every core returns at once.
        idle_rows, _idle_batch = SHAPES['G4B1']
        idle_query = scope.keep(pool.upload(query_for(torch, pool, 1, idle_rows, seed, 'normal', extents[0], salt=17)))
        idle_pages = scope.keep(pool.pages_causal([0], [extents[0]]))
        idle_mask = scope.keep(pool.mask([positions[0]], [extents[0]], idle_rows))
        out = pool.launch(idle_query, idle_pages, causal=False, cur_pos=scope.keep(pool.words([-1])), mask=idle_mask,
                          sentinel=MAGIC | 0x21, label='K/idle B=1')
        with probe.WATCHDOG.op('K idle synchronize'):
            ttnn.synchronize_device(pool.device)
        ttnn.deallocate(out)
        report['skip_idle_call'] = 'returned'
    finally:
        scope.close()
    states = sorted({entry['state'] for entry in written})
    report['skip_written'] = ','.join(states) if states else None


def section_liveness(ttnn, torch, pool, seed, args, report):
    """L: words E (one poisoned chunk more) against E - 1, zero mask, per combo and extent below the capacity."""
    for shape, flags in args.combos:
        rows, batch = SHAPES[shape]
        for extent in args.extents:
            if extent >= pool.capacity:
                continue
            label = 'L/%s/seed%d/E%d' % (combo_name(shape, flags), seed, extent)
            probe.DEADLINE.check(label)
            scope = probe.Scope(ttnn)
            try:
                query = scope.keep(pool.upload(query_for(torch, pool, batch, rows, seed, 'normal', extent, salt=21)))
                pages = scope.keep(pool.pages_causal([0] * batch, [extent] * batch))
                zero = scope.keep(pool.mask([extent - 1] * batch, [extent] * batch, rows, zero=True))
                before = pool.extent_run(query, pages, scope.keep(pool.words([extent - 1] * batch)), zero, flags,
                                         label + ' E-1')
                after = pool.extent_run(query, pages, scope.keep(pool.words([extent] * batch)), zero, flags, label + ' E')
                probe.liveness(torch, report, 'L', label, before, after, range(batch))
            finally:
                scope.close()


class TraceRig:
    """One 0x20 program on one page table, eager or captured: the cur_pos words and the narrow mask are device
    tensors rewritten in place between replays (copy_host_to_device_tensor), as the extent readers will."""

    def __init__(self, ttnn, torch, device, pool, scope, query, pages, flags, rows, batch):
        self.ttnn, self.torch, self.device, self.pool, self.scope = ttnn, torch, device, pool, scope
        self.query, self.pages, self.flags, self.rows, self.batch = query, pages, flags, rows, batch
        self.words = self.mask = self.trace = self.output = None

    def host_mask(self, positions, zero=False):
        extents = [split_model.extent(max(0, position)) for position in positions]
        if self.flags & SHARE:
            extents = [extents[0]] * self.batch
        live = [position if position >= 0 else extent - 1 for position, extent in zip(positions, extents)]
        return probe.position_mask(self.torch, live, extents, self.rows, K_CHUNK, zero=zero)

    def eager(self, words, host_mask, label):
        pool = self.pool
        word_tensor, mask_tensor = pool.words(words), pool.upload(host_mask)
        try:
            return pool.extent_run(self.query, self.pages, word_tensor, mask_tensor, self.flags, label + ' eager')
        finally:
            self.ttnn.deallocate(word_tensor)
            self.ttnn.deallocate(mask_tensor)

    def capture(self, words, host_mask, traces, label):
        ttnn, pool = self.ttnn, self.pool
        self.words = self.scope.keep(pool.words(words))
        self.mask = self.scope.keep(pool.upload(host_mask))
        with probe.WATCHDOG.op(label + ' capture'):
            self.trace, self.output = card.capture(ttnn, self.device, lambda: pool.launch(
                self.query, self.pages, causal=False, cur_pos=self.words, mask=self.mask, sentinel=MAGIC | self.flags,
                label=label + ' capture'))
        traces.append(self.trace)
        self.scope.keep(self.output)

    def replay(self, words, host_mask, what):
        ttnn, torch = self.ttnn, self.torch
        word_source = ttnn.from_torch(torch.tensor(list(words), dtype=torch.int32), dtype=ttnn.int32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT)
        mask_source = ttnn.from_torch(host_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        with probe.WATCHDOG.op('stage ' + what):
            ttnn.copy_host_to_device_tensor(word_source, self.words)
            ttnn.copy_host_to_device_tensor(mask_source, self.mask)
        with probe.WATCHDOG.op('replay ' + what):
            ttnn.execute_trace(self.device, self.trace, cq_id=0, blocking=True)
            ttnn.synchronize_device(self.device)
        with probe.WATCHDOG.op('trace read back ' + what):
            return ttnn.to_torch(self.output)


def trace_plan(capacity, families, batch, rows, seed, share):
    """The trace's positions per replay: probe.trace_positions (entries spread over every family); a share bundle
    keeps entry 0's position and puts the other entries in its family."""
    plan = probe.trace_positions(capacity, families, batch, seed)
    if share:
        plan = [tuple(share_positions(values[0], batch, rows)) for values in plan]
    return plan


def section_trace(ttnn, torch, device, pool, seed, args, report):
    """T: per trace combo, one program replayed with its words and mask rewritten; skips; the fence."""
    for shape, flags in args.trace_combos:
        rows, batch = SHAPES[shape]
        share = bool(flags & SHARE)
        users = [0] * batch if share else list(range(batch))
        plan = trace_plan(pool.capacity, args.trace_families, batch, rows, seed, share)
        report['trace_families_distinct'] = max(report.get('trace_families_distinct', 0),
                                                probe.distinct_families([values[:1] if share else values
                                                                         for values in plan]))
        name = combo_name(shape, flags)
        scope = probe.Scope(ttnn)
        traces = []
        try:
            host_query = query_for(torch, pool, batch, rows, seed, 'normal', pool.capacity, salt=50)
            query = scope.keep(pool.upload(host_query))
            clean = scope.keep(pool.pages_causal(users, [pool.capacity] * batch, poison=False))
            rig = TraceRig(ttnn, torch, device, pool, scope, query, clean, flags, rows, batch)
            eager = {}

            def eager_at(positions, label, zero=False, words=None):
                key = (tuple(positions), zero, None if words is None else tuple(words))
                if key not in eager:
                    eager[key] = rig.eager(words_for(positions, share) if words is None else words,
                                           rig.host_mask(positions, zero), label)
                return eager[key]

            eager_at(plan[0], 'T/%s warm' % name)
            rig.capture(words_for(plan[0], share), rig.host_mask(plan[0]), traces, 'T/%s' % name)
            sample = reference_sample(len(plan), args.trace_references)
            for index, positions in enumerate(plan):
                label = 'T/%s/seed%d/replay%d/%s' % (name, seed, index, ','.join(map(str, positions)))
                probe.DEADLINE.check(label)
                got = rig.replay(words_for(positions, share), rig.host_mask(positions), label)
                record(report, probe.comparison('T', 'trace_vs_eager', label,
                                                card.differing(torch, got, eager_at(positions, label)), True,
                                                combo=name, positions=list(positions), seed=seed))
                if index in sample:
                    extent = split_model.extent(positions[0])
                    refscope = probe.Scope(ttnn)
                    try:
                        ref_query = refscope.keep(pool.upload(repeated(torch, host_query, 0, batch)))
                        ref_pages = refscope.keep(pool.pages_reference(users[0], extent, batch))
                        mask0 = rig.host_mask(positions)[0:1].repeat(batch, 1, 1, 1).contiguous()
                        ref_mask = refscope.keep(pool.upload(mask0))
                        reference = pool.reference_run(ref_query, ref_pages, ref_mask, flags, label + ' reference')
                    finally:
                        refscope.close()
                    record(report, probe.comparison('T', 'trace_vs_reference', '%s/entry0/E%d' % (label, extent),
                                                    card.differing(torch, probe.slot(got, 0), probe.slot(reference, 0)),
                                                    True, combo=name, extent=extent, seed=seed), verbose=False)
            if not share:
                last = plan[-1]
                full = eager_at(last, 'T/%s skip base' % name)
                for pattern in SKIP_PATTERNS:
                    if max(pattern) >= batch:
                        continue
                    label = 'T/%s/seed%d/skip%s' % (name, seed, ''.join(map(str, pattern)))
                    probe.DEADLINE.check(label)
                    words = probe.skip_positions(words_for(last, share), pattern)
                    got = rig.replay(words, rig.host_mask(last), label)
                    live = [index for index in range(batch) if index not in pattern]
                    if not live:
                        report.setdefault('trace_skip_all', {})[name] = 'returned'
                        continue
                    differing = sum(card.differing(torch, probe.slot(got, index), probe.slot(full, index))
                                    for index in live)
                    record(report, probe.comparison('T', 'trace_skip_live', label, differing, True, combo=name,
                                                    pattern=list(pattern), live=live, seed=seed))
            probe.release_traces(ttnn, device, traces)
            fence_trace(ttnn, torch, device, pool, seed, args, report, scope, traces, query, users, shape, flags,
                        eager_at)
        finally:
            probe.release_traces(ttnn, device, traces)
            scope.close()


def fence_trace(ttnn, torch, device, pool, seed, args, report, scope, traces, query, users, shape, flags, eager_at):
    """T's fence: the same program captured on a table poisoned past each entry's own family, replayed inside those
    families (equal to eager on the CLEAN table) and at words E (must differ: the poison is live in the trace)."""
    rows, batch = SHAPES[shape]
    share = bool(flags & SHARE)
    extents = probe.fence_extents(args.extents, pool.capacity, batch)
    if share:
        extents = [extents[0]] * batch
    plan = probe.fence_plan(extents)
    if share:
        plan = [tuple(share_positions(values[0], batch, rows)) for values in plan]
    name = combo_name(shape, flags)
    report.setdefault('fence_extents', {})[name] = extents
    fenced = scope.keep(pool.pages_causal(users, extents))
    rig = TraceRig(ttnn, torch, device, pool, scope, query, fenced, flags, rows, batch)
    rig.eager(words_for(plan[0], share), rig.host_mask(plan[0]), 'T/%s fence warm' % name)   # builds the program
    rig.capture(words_for(plan[0], share), rig.host_mask(plan[0]), traces, 'T/%s fence' % name)
    for index, positions in enumerate(plan):
        label = 'T/%s/seed%d/fence%d/%s' % (name, seed, index, ','.join(map(str, positions)))
        probe.DEADLINE.check(label)
        got = rig.replay(words_for(positions, share), rig.host_mask(positions), label)
        record(report, probe.comparison('T', 'trace_fence_vs_clean', label,
                                        card.differing(torch, got, eager_at(positions, label)), True,
                                        combo=name, positions=list(positions), extents=extents, seed=seed))
    beyond = list(extents)
    label ='T/%s/seed%d/fence/E' % (name, seed)
    probe.DEADLINE.check(label)
    got = rig.replay(beyond, rig.host_mask(plan[0], zero=True), label)
    probe.liveness(torch, report, 'T', label, eager_at(plan[0], label, zero=True, words=beyond), got,
                   [index for index, extent in enumerate(extents) if extent < pool.capacity])


def section_timing(ttnn, torch, device, pool, args, report):
    """Recorded: the 0x21 call at E - 1 on the C-wide table against the 0x1 call at capacity E; 0 vs B skips."""
    rows, batch = SHAPES['G4B3']
    scope = probe.Scope(ttnn)
    rows_out = []
    try:
        query = scope.keep(pool.upload(query_for(torch, pool, batch, rows, 0, 'normal', pool.capacity, salt=60)))
        shapes = []
        for extent in args.extents:
            positions = bundle_positions(extent, 0, batch, rows)
            pages = scope.keep(pool.pages_causal([0] * batch, [extent] * batch))
            ref_pages = scope.keep(pool.pages_reference(0, extent, batch))
            mask = scope.keep(pool.mask(positions, [extent] * batch, rows))
            words = scope.keep(pool.words([extent - 1] * batch))
            shapes.append(('runtime E%d' % extent, extent, (lambda p=pages, m=mask, w=words: pool.launch(
                query, p, causal=False, cur_pos=w, mask=m, sentinel=MAGIC | 0x21, label='timing'))))
            shapes.append(('compile E%d' % extent, extent, (lambda p=ref_pages, m=mask: pool.launch(
                query, p, causal=False, mask=m, sentinel=MAGIC | TAIL, label='timing'))))
        full_pages = scope.keep(pool.pages_causal([0] * batch, [pool.capacity] * batch, poison=False))
        top_mask = scope.keep(pool.mask([pool.capacity - 1] * batch, [pool.capacity] * batch, rows, zero=True))
        for count in (0, batch):
            words = scope.keep(pool.words([-1] * count + [pool.capacity - 1] * (batch - count)))
            shapes.append(('skip%d' % count, pool.capacity, (lambda w=words: pool.launch(
                query, full_pages, causal=False, cur_pos=w, mask=top_mask, sentinel=MAGIC | 0x21, label='timing'))))
        medians = {name: [] for name, _extent, _once in shapes}
        samples = {name: [] for name, _extent, _once in shapes}
        for index in range(args.rounds):
            probe.DEADLINE.check('timing round %d' % index)
            turn = index % len(shapes)
            for name, _extent, once in shapes[turn:] + shapes[:turn]:
                got = probe.eager_median(ttnn, device, once, args, 'timing %s round %d' % (name, index))
                medians[name].append(statistics.median(got))
                samples[name].extend(got)
        for name, extent, _once in shapes:
            rows_out.append(dict(name=name, extent=extent, eager=k1.summary(samples[name]),
                                 round_medians=medians[name]))
            print('timing %-16s %.1f us' % (name, statistics.median(samples[name])), flush=True)
    finally:
        scope.close()
    by = {row['name']: row['eager']['median_us'] for row in rows_out}
    ratios = {}
    for extent in args.extents:
        runtime, compile_time = by.get('runtime E%d' % extent), by.get('compile E%d' % extent)
        if runtime and compile_time:
            ratios['E%d' % extent] = runtime / compile_time
    report['timing'] = dict(rows=rows_out, runtime_over_compile=ratios,
                            skip_us={name: by[name] for name in by if name.startswith('skip')})


# ---------------------------------------------------------------------------------------------
# CB2a: K2, X7, Z (s2-design.md W10a).
# ---------------------------------------------------------------------------------------------

class ServedRig(TraceRig):
    """TraceRig on the served call (ExtentPool.served_launch): Z's trace. replay() is TraceRig's: the words and the
    narrow mask rewritten in place (copy_host_to_device_tensor), the trace executed, the output read back."""

    def eager(self, words, host_mask, label):
        pool = self.pool
        word_tensor, mask_tensor = pool.words(words), pool.upload(host_mask)
        try:
            return pool.served_run(self.query, self.pages, mask_tensor, self.flags, words=word_tensor,
                                   label=label + ' eager')
        finally:
            self.ttnn.deallocate(word_tensor)
            self.ttnn.deallocate(mask_tensor)

    def capture(self, words, host_mask, traces, label):
        ttnn, pool = self.ttnn, self.pool
        self.words = self.scope.keep(pool.words(words))
        self.mask = self.scope.keep(pool.upload(host_mask))
        with probe.WATCHDOG.op(label + ' capture'):
            self.trace, self.output = card.capture(ttnn, self.device, lambda: pool.served_launch(
                self.query, self.pages, self.mask, self.flags, words=self.words, label=label + ' capture'))
        traces.append(self.trace)
        self.scope.keep(self.output)


def row_slice(rows, index):
    """(1, 1, 12, 256): one token row of an unfolded (1, T, 12, 256) output."""
    return rows[:, index:index + 1]


def section_k2(ttnn, torch, pool, seed, args, report):
    """K2: per ticket, the G8B2 0x27 extent call against the native B1 call of every committed row."""
    tickets = k2_tickets(args.k2_sweep, args.k2_floor, args.cb2_extents, args.cb2_starts)
    counts = report.setdefault('k2_rows', dict(compared=0, equal=0, floor=0, floor_differing=0, capped=0))
    table = pool.tables[0]
    scope = probe.Scope(ttnn)
    families = {}

    def family(extent):
        """The user's tables for family E (both poisoned past E): the native (1, C / 64) row, the subject's (2, C /
        64) bundle table, and the subject's words E - 1."""
        if extent not in families:
            families[extent] = dict(
                native=scope.keep(pool.pages_causal([0], [extent])),
                subject=scope.keep(pool.pages_causal([0] * SERVED_BATCH, [extent] * SERVED_BATCH)),
                words=scope.keep(pool.words([extent - 1] * SERVED_BATCH)))
        return families[extent]

    try:
        for variant in args.variants:
            tokens, natives = {}, {}

            def token(position, variant=variant, tokens=tokens):
                if position not in tokens:
                    tokens[position] = token_query(torch, seed, variant, position, pool.keys, table)
                return tokens[position]

            def native(position, cur_pos, label, token=token, natives=natives):
                """The solo row's output (cached): its token's query, B = 1, cur_pos, family(E(position))'s row."""
                key = (position, cur_pos)
                if key not in natives:
                    probe.DEADLINE.check(label)
                    query = pool.upload(token(position)[None, None].contiguous())
                    word = pool.positions([cur_pos])
                    try:
                        natives[key] = pool.native_run(query, family(split_model.extent(position))['native'], word,
                                                       label)
                    finally:
                        ttnn.deallocate(query)
                        ttnn.deallocate(word)
                    probe.finite_or_fail(torch, report, label, natives[key])
                return natives[key]

            for ticket in tickets:
                start, extent, kind = ticket['start'], ticket['extent'], ticket['kind']
                label = 'K2/seed%d/%s/E%d/s%d' % (seed, variant, extent, start)
                probe.DEADLINE.check(label)
                tables = family(extent)
                query = pool.upload(ticket_query(torch, token, start))
                mask = pool.upload(narrow_mask(torch, start))
                try:
                    got = pool.served_run(query, tables['subject'], mask, SERVED_FLAGS, words=tables['words'],
                                          label=label + ' extent')
                finally:
                    ttnn.deallocate(query)
                    ttnn.deallocate(mask)
                rows = card.unfold_entries(torch, got, SERVED_ROWS)             # (1, 16, 12, 256), ticket order
                compared = k2_compared(ticket)
                decisive = kind != 'floor'
                differing, moved = 0, []
                for position in compared:
                    reference = native(position, native_cur_pos(position), '%s/p%d native' % (label, position))
                    count = card.differing(torch, row_slice(rows, position - start), reference)
                    differing += count
                    if count:
                        moved.append(position)
                if decisive:
                    counts['compared'] += len(compared)
                    counts['equal'] += len(compared) - len(moved)
                    counts['capped'] += TICKET_ROWS - len(compared)
                else:
                    counts['floor'] += len(compared)
                    counts['floor_differing'] += len(moved)
                record(report, probe.comparison('K2', 'k2_native_vs_extent' if decisive else 'k2_floor', label,
                                                differing, decisive, ticket=kind, extent=extent, start=start,
                                                rows=len(compared), differing_rows=moved, seed=seed, variant=variant))
                # Liveness (peaky: every row aims at the key past it), on the reference path alone: the native call
                # one key further must move the ticket's first row - the comparison sees a boundary one key off.
                if decisive and variant == 'peaky' and compared and compared[0] + 1 < pool.capacity:
                    position = compared[0]
                    own = native(position, position, '%s/p%d native at p' % (label, position))
                    beyond = native(position, position + 1, '%s/p%d native at p + 1' % (label, position))
                    moved_rows = probe.moved_rows(torch, own, beyond)[0]
                    report['liveness'].append(dict(section='K2', label='%s/p%d+1' % (label, position),
                                                   moved_rows=moved_rows, rows=12, live=moved_rows > 0))
    finally:
        scope.close()


def section_x7(ttnn, torch, pool, seed, args, report):
    """X7: 0x7 narrow == 0x7 wide (the real -inf tail) at capacity E; 0x27 at E - 1 on the C-wide table == 0x7 wide."""
    table = pool.tables[0]
    for extent in args.cb2_extents:
        base = 'X7/seed%d/E%d' % (seed, extent)
        probe.DEADLINE.check(base)
        scope = probe.Scope(ttnn)
        live_done = False
        try:
            reference_pages = scope.keep(pool.pages_reference(0, extent, SERVED_BATCH))
            extent_pages = scope.keep(pool.pages_causal([0] * SERVED_BATCH, [extent] * SERVED_BATCH))
            words = scope.keep(pool.words([extent - 1] * SERVED_BATCH))
            for offset in args.cb2_starts:
                start = extent - K_CHUNK + offset
                inner = probe.Scope(ttnn)                                # the wide mask is E columns: free it soon
                try:
                    narrow_host = narrow_mask(torch, start)
                    narrow = inner.keep(pool.upload(narrow_host))
                    wide = inner.keep(pool.upload(wide_mask(torch, start, extent)))
                    for variant in args.variants:
                        label = '%s+%d/%s' % (base, offset, variant)
                        probe.DEADLINE.check(label)
                        query = inner.keep(pool.upload(ticket_query(
                            torch, lambda position, variant=variant: token_query(torch, seed, variant, position,
                                                                                 pool.keys, table), start)))
                        got_narrow = pool.served_run(query, reference_pages, narrow, COMPILE_FLAGS,
                                                     label=label + ' 0x7 narrow')
                        got_wide = pool.served_run(query, reference_pages, wide, COMPILE_FLAGS, label=label + ' 0x7 wide')
                        got_extent = pool.served_run(query, extent_pages, narrow, SERVED_FLAGS, words=words,
                                                     label=label + ' 0x27')
                        probe.finite_or_fail(torch, report, label, got_wide)
                        record(report, probe.comparison('X7', 'x7_narrow_vs_wide', label + '/narrow',
                                                        card.differing(torch, got_narrow, got_wide), True,
                                                        extent=extent, start=start, seed=seed, variant=variant))
                        record(report, probe.comparison('X7', 'x7_extent_vs_wide', label + '/0x27',
                                                        card.differing(torch, got_extent, got_wide), True,
                                                        extent=extent, start=start, seed=seed, variant=variant))
                        # Liveness, on the reference (wide) path alone: a zero wide mask must move every entry
                        # whose tail masks a key - the comparison sees the mask's content.
                        masked = [index for index in range(SERVED_BATCH)
                                  if bool(torch.isinf(narrow_host[index].float()).any())]
                        if not live_done and masked:
                            zero = inner.keep(pool.upload(torch.zeros(SERVED_BATCH, 1, SERVED_ROWS * 12, extent,
                                                                      dtype=torch.bfloat16)))
                            opened = pool.served_run(query, reference_pages, zero, COMPILE_FLAGS,
                                                     label=label + ' 0x7 zero wide mask')
                            moved = probe.moved_rows(torch, got_wide, opened)
                            for index in masked:
                                report['liveness'].append(dict(section='X7', label='%s/zero_mask/entry%d' % (label, index),
                                                               moved_rows=moved[index], rows=SERVED_ROWS * 12,
                                                               live=moved[index] > 0))
                            live_done = True
                finally:
                    inner.close()
        finally:
            scope.close()


def section_z(ttnn, torch, device, pool, seed, args, report):
    """Z: one G8B2 0x27 trace replayed over the stale-writer families at their idle and boundary starts."""
    plan = z_plan(args.z_families, args.z_starts)
    report['z_families'] = sorted(set(args.z_families))
    scope, traces, references = probe.Scope(ttnn), [], {}
    try:
        query = scope.keep(pool.upload(card.build_query(torch, SERVED_BATCH, seed, 'normal', rows=SERVED_ROWS)))
        pages = scope.keep(pool.pages_causal([0] * SERVED_BATCH, [pool.capacity] * SERVED_BATCH, poison=False))
        rig = ServedRig(ttnn, torch, device, pool, scope, query, pages, SERVED_FLAGS, SERVED_ROWS, SERVED_BATCH)

        def words(extent):
            return [extent - 1] * SERVED_BATCH

        first_extent, first_start = plan[0]
        rig.eager(words(first_extent), narrow_mask(torch, first_start), 'Z/seed%d warm' % seed)   # builds it
        rig.capture(words(first_extent), narrow_mask(torch, first_start), traces, 'Z/seed%d' % seed)
        for index, (extent, start) in enumerate(plan):
            label = 'Z/seed%d/replay%d/E%d+%d' % (seed, index, extent, start - (extent - K_CHUNK))
            probe.DEADLINE.check(label)
            host_mask = narrow_mask(torch, start)
            got = rig.replay(words(extent), host_mask, label)
            eager = rig.eager(words(extent), host_mask, label)
            if extent not in references:
                references[extent] = scope.keep(pool.pages_reference(0, extent, SERVED_BATCH))
            mask = pool.upload(host_mask)
            try:
                reference = pool.served_run(query, references[extent], mask, COMPILE_FLAGS,
                                            label=label + ' 0x7 reference')
            finally:
                ttnn.deallocate(mask)
            probe.finite_or_fail(torch, report, label, reference)
            record(report, probe.comparison('Z', 'z_trace_vs_eager', label, card.differing(torch, got, eager), True,
                                            extent=extent, start=start, seed=seed))
            record(report, probe.comparison('Z', 'z_trace_vs_reference', label, card.differing(torch, got, reference),
                                            True, extent=extent, start=start, seed=seed), verbose=False)
    finally:
        probe.release_traces(ttnn, device, traces)
        scope.close()


def native_decode_lines():
    """What the harness prints before K2 runs: the native kwargs it uses and what source cannot pin."""
    lines = ['K2 native decode (%s:%s, sha256 %s): program_config %s, call kwargs %s, scale %r, memory_config %s; '
             'not passed: %s' % (NATIVE_DECODE['source'], NATIVE_DECODE['lines']['call'], NATIVE_DECODE['sha256'][:16],
                                 NATIVE_DECODE['program_config'], ','.join(NATIVE_DECODE['call']), NATIVE_SCALE,
                                 NATIVE_DECODE['memory_config'], ','.join(NATIVE_DECODE['not_passed']))]
    lines += ['UNVERIFIED K2: %s' % item for item in UNVERIFIED]
    return lines


def check_binary(args, report):
    """The mapped _ttnncpp.so: the expected K64j binary (required), stage 4 plus the F22 literal, compact scratch."""
    path, markers = card.loaded_binary()
    literals = binary_literals(path)
    stage = probe.binary_stage(markers, literals['slice'])
    sha = k1.file_sha256(path)
    report['binary'] = dict(path=path, sha256=sha, markers=markers, literals=literals, stage=stage,
                            k64j=stage == 4 and literals['extent'], expected_sha256=args.expect_binary_sha256 or None)
    print('binary %s sha256 %s stage=%d literals=%s' % (path, sha[:16], stage, literals), flush=True)
    ok = True
    if not args.expect_binary_sha256:
        report['failures'].append('no --expect-binary-sha256: the K64j graft\'s _ttnncpp.so sha (build_k64j.sh prints '
                                  'K64J_TTNNCPP_SHA256) is required')
        ok = False
    elif sha != args.expect_binary_sha256:
        report['failures'].append('the loaded _ttnncpp.so is %s, not the expected %s (read the launched argv)'
                                  % (sha[:16], args.expect_binary_sha256[:16]))
        ok = False
    if not report['binary']['k64j'] or not literals['scratch']:
        report['failures'].append('the loaded _ttnncpp.so is not K64j: stage %d, literals %r (it needs the stage-4 '
                                  'markers, %r and %s)' % (stage, literals, EXTENT_LOG_MARKER, card.SCRATCH_ENV))
        ok = False
    return ok


def check_kernels(root, report):
    """The four K64j qwen kernels and the five stock sources under the mounted kernels directory."""
    found, ok = {}, True
    for kind, table in (('k64j', K64J_KERNELS), ('stock', STOCK_KERNELS)):
        for name, expected in table.items():
            path = Path(root) / name
            found[name] = k1.file_sha256(path) if path.is_file() else None
            if found[name] != expected:
                ok = False
                report['failures'].append('%s is %s, not the %s %s' % (path, (found[name] or 'missing')[:16], kind,
                                                                         expected[:16]))
    report['kernels'] = dict(root=str(root), found=found)
    return ok


def run(args, report, checkpoint=None):
    import torch
    import ttnn

    options = dict(device_id=args.device_id, l1_small_size=24576)
    if {'T', 'Z'} & set(args.sections):
        options['trace_region_size'] = args.trace_region_bytes
    with probe.WATCHDOG.op('open device', extra=OPEN_EXTRA_S):
        device = ttnn.open_device(**options)
    try:
        try:
            device.enable_program_cache()
            report['program_cache_enabled_call'] = True
        except Exception as error:  # noqa: BLE001 - default-on in newer runtimes
            report['program_cache_enabled_call'] = repr(error)[:200]
        grid = device.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        if not check_binary(args, report):
            return
        if args.kernel_root and not check_kernels(args.kernel_root, report):
            return
        if os.environ.get(card.SCRATCH_ENV) != '1':
            report['failures'].append('%s=1 is required: the arm sets it, and the G8 programs do not fit L1 without it'
                                      % card.SCRATCH_ENV)
            return
        run_sections(ttnn, torch, device, args, report, checkpoint)
    finally:
        with probe.WATCHDOG.op('close device'):
            ttnn.close_device(device)


def run_sections(ttnn, torch, device, args, report, checkpoint=None):
    """Every (seed, section), each isolated: one that raises is a failure and the run goes on; the deadline stops
    it and lists the rest; the report is checkpointed after every section."""
    handlers = {
        'N': lambda pool, seed: section_refusals(ttnn, torch, pool, args, report),
        'X': lambda pool, seed: section_extent(ttnn, torch, pool, seed, args, report),
        'M': lambda pool, seed: section_mixed(ttnn, torch, pool, seed, args, report),
        'K': lambda pool, seed: section_skip(ttnn, torch, pool, seed, args, report),
        'L': lambda pool, seed: section_liveness(ttnn, torch, pool, seed, args, report),
        'T': lambda pool, seed: section_trace(ttnn, torch, device, pool, seed, args, report),
        'K2': lambda pool, seed: section_k2(ttnn, torch, pool, seed, args, report),
        'X7': lambda pool, seed: section_x7(ttnn, torch, pool, seed, args, report),
        'Z': lambda pool, seed: section_z(ttnn, torch, device, pool, seed, args, report),
        'timing': lambda pool, seed: section_timing(ttnn, torch, device, pool, args, report),
    }
    runs = section_runs(args)
    report['plan_runs'] = [run_tag(seed, name) for seed, name in runs]
    done, failed = report.setdefault('sections_done', []), report.setdefault('sections_failed', [])
    pool, pool_seed, dead_seed = None, None, None
    try:
        for index, (seed, name) in enumerate(runs):
            tag = run_tag(seed, name)
            if seed == dead_seed:
                continue
            try:
                probe.DEADLINE.check(tag)
                if pool_seed != seed:
                    if pool is not None:
                        pool.close()
                    pool, pool_seed = None, seed
                    try:
                        pool = ExtentPool(ttnn, torch, device, args.capacity, seed, 3, report)
                        pool.output = args.output_memory
                    except probe.DeadlineReached:
                        raise
                    except Exception as error:  # noqa: BLE001 - that seed's sections cannot run
                        dead_seed = seed
                        report['failures'].append('pool/seed%d: %s (its sections did not run)'
                                                  % (seed, probe.one_line(error)))
                        print('SECTION FAILED pool/seed%d: %s' % (seed, probe.one_line(error)), flush=True)
                        continue
                handlers[name](pool, seed)
            except probe.DeadlineReached as reached:
                skipped = [run_tag(later_seed, later) for later_seed, later in runs[index:]]
                report['deadline'] = dict(seconds=probe.DEADLINE.seconds, reached_at=str(reached), skipped=skipped)
                report['warnings'].append('deadline %ss reached at %s: %d section runs not run (%s)'
                                          % (probe.DEADLINE.seconds, reached, len(skipped), ', '.join(skipped)))
                print('DEADLINE reached at %s: skipped %s' % (reached, ', '.join(skipped)), flush=True)
                break
            except Exception as error:  # noqa: BLE001 - isolate the section; a TT_FATAL surfaces as RuntimeError
                failed.append(tag)
                report['failures'].append('%s: %s' % (tag, probe.one_line(error)))
                print('SECTION FAILED %s: %s' % (tag, probe.one_line(error)), flush=True)
            else:
                done.append(tag)
            finally:
                if checkpoint is not None:
                    checkpoint(tag)
    finally:
        if pool is not None:
            pool.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--capacity', type=int, default=CAPACITY, help='the C-wide page table (keys)')
    parser.add_argument('--extents', default=','.join(map(str, EXTENTS)))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)), help='entry 0 at E - 256 + s, 0..255')
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='K1 asks 0-4')
    parser.add_argument('--variants', default=','.join(VARIANTS), help='normal and/or peaky')
    parser.add_argument('--combos', default=','.join(combo_name(*combo) for combo in default_combos()),
                        help='X, M and L: shape:flags pairs (G4B3 0x21/0x23, G8B2 0x21/0x23/0x27/0x2F)')
    parser.add_argument('--trace-combos', default=','.join(combo_name(*combo) for combo in TRACE_COMBOS))
    parser.add_argument('--sections', default=','.join(DEFAULT_SECTIONS),
                        help='any of %s (default K1/K3\'s %s; CB2a is K2,X7,Z)' % (', '.join(SECTIONS),
                                                                                ','.join(DEFAULT_SECTIONS)))
    parser.add_argument('--k2-sweep', default='%d:%d' % K2_SWEEP,
                        help='K2: every ticket start lo..hi (inclusive, lo >= %d)' % MIN_LIVE_START)
    parser.add_argument('--k2-floor', default='%d:%d' % K2_FLOOR,
                        help='K2, recorded: ticket starts lo..hi below %d, their rows below it ("" none)' % MIN_LIVE_START)
    parser.add_argument('--cb2-extents', default=','.join(map(str, CB2_EXTENTS)), help='K2\'s and X7\'s families')
    parser.add_argument('--cb2-starts', default=','.join(map(str, CB2_STARTS)), help='their tickets\' s mod 256')
    parser.add_argument('--z-families', default=','.join(map(str, Z_FAMILIES)), help='Z\'s families')
    parser.add_argument('--z-starts', default=','.join(map(str, Z_STARTS)), help='Z\'s s mod 256 per family')
    parser.add_argument('--output-memory', choices=OUTPUT_MEMORY, default=OUTPUT_MEMORY[0],
                        help='K2, X7 and Z: the calls\' output memory (the model\'s is L1, tp.py:753)')
    parser.add_argument('--trace-families', type=int, default=TRACE_FAMILIES)
    parser.add_argument('--trace-references', type=int, default=TRACE_REFERENCES,
                        help='replays whose entry 0 is compared with the compile-time call at its family (one JIT '
                             'build each)')
    parser.add_argument('--trace-region-bytes', type=int, default=16 << 20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--watchdog', type=float, default=0, help='seconds per device call before os._exit(3); 0 off')
    parser.add_argument('--deadline-s', type=float, default=0,
                        help='stop cleanly between device calls this long after start (the runner: its container '
                             'timeout less %d s); 0 off' % DEADLINE_MARGIN_S)
    parser.add_argument('--expect-binary-sha256', default='',
                        help='the mapped _ttnncpp.so must be this (build_k64j.sh\'s K64J_TTNNCPP_SHA256; required)')
    parser.add_argument('--kernel-root', default=KERNEL_ROOT, help='the mounted sdpa_decode kernels ("" skips)')
    args = parser.parse_args(argv)

    def ints(text):
        return [int(value) for value in text.split(',') if value.strip()]

    try:
        args.extents = ints(args.extents)
        args.starts = ints(args.starts)
        args.seeds = ints(args.seeds)
        args.combos = parse_combos(args.combos)
        args.trace_combos = parse_combos(args.trace_combos)
        args.k2_sweep = parse_range(args.k2_sweep, '--k2-sweep')
        args.k2_floor = parse_range(args.k2_floor, '--k2-floor')
        args.cb2_extents = ints(args.cb2_extents)
        args.cb2_starts = ints(args.cb2_starts)
        args.z_families = ints(args.z_families)
        args.z_starts = ints(args.z_starts)
    except ValueError as error:
        parser.error(str(error))
    if args.deadline_s < 0:
        parser.error('--deadline-s must be >= 0')
    args.variants = [value for value in args.variants.split(',') if value]
    args.sections = [value for value in args.sections.split(',') if value]
    try:
        probe.check_capacity(args.capacity)
        for extent in args.extents:
            probe.check_capacity(extent, 'extent')
        for extent in args.cb2_extents + args.z_families:
            probe.check_capacity(extent, 'CB2a family')
    except ValueError as error:
        parser.error(str(error))
    # CB2a's own arguments are checked against --capacity only when their sections run (the defaults are the served
    # geometry's; a K1 run on a smaller table never reads them).
    if (args.k2_sweep is None or args.k2_sweep[0] < MIN_LIVE_START
            or ('K2' in args.sections and split_model.extent(args.k2_sweep[1]) > args.capacity)):
        parser.error('--k2-sweep must be lo:hi with lo >= %d and its last family within --capacity' % MIN_LIVE_START)
    if args.k2_floor is not None and args.k2_floor[1] >= MIN_LIVE_START:
        parser.error('--k2-floor must lie below %d (or be "")' % MIN_LIVE_START)
    for name, extents, sections in (('--cb2-extents', args.cb2_extents, ('K2', 'X7')),
                                    ('--z-families', args.z_families, ('Z',))):
        if not extents or len(set(extents)) != len(extents) or (
                set(sections) & set(args.sections) and max(extents) > args.capacity):
            parser.error('%s must be non-empty, distinct and at most --capacity' % name)
    for name, starts in (('--cb2-starts', args.cb2_starts), ('--z-starts', args.z_starts)):
        if not starts or any(not 0 <= start < K_CHUNK for start in starts):
            parser.error('%s must be non-empty and 0..255' % name)
    if not args.extents or max(args.extents) > args.capacity or len(set(args.extents)) != len(args.extents):
        parser.error('--extents must be distinct and at most --capacity')
    if not args.starts or any(not 0 <= start < K_CHUNK for start in args.starts):
        parser.error('--starts must be 0..255')
    if not args.seeds or not args.variants or any(v not in ('normal', 'peaky') for v in args.variants):
        parser.error('--seeds and --variants (normal, peaky) must be non-empty')
    if not args.combos or any(name not in SECTIONS for name in args.sections):
        parser.error('--combos must be non-empty and --sections a subset of %s' % ','.join(SECTIONS))
    if args.trace_families < 1 or args.trace_references < 0 or min(args.iters, args.rounds) < 1 or args.warmup < 0:
        parser.error('--trace-families, --iters and --rounds must be >= 1, --trace-references >= 0')
    if args.expect_binary_sha256 and (len(args.expect_binary_sha256) != 64
                                      or any(c not in '0123456789abcdef' for c in args.expect_binary_sha256)):
        parser.error('--expect-binary-sha256 must be a full lowercase sha256')
    return args


def term_handler(report, write_report):
    """SIGTERM: record the error, write the partial report (NO-DECISION) at once and unwind (probe.term_handler's
    shape, with this harness's verdict)."""

    def on_term(signum, _frame):
        if report.get('terminated'):
            return
        report['terminated'] = signum
        report['error'] = ('terminated by signal %d during %r (the container timeout: --deadline-s did not stop the '
                           'run first)' % (signum, probe.WATCHDOG.label))
        try:
            decision = decide(report)
            write_report(dict(decision=decision, verdict_line=verdict_line(dict(report, decision=decision)),
                              in_progress='terminated'))
        except Exception:  # noqa: BLE001 - best effort; main writes again after the unwinding
            pass
        raise probe.Terminated(signum)

    return on_term


def check_log(report, text):
    """The factory's lines for every requested program: F4 (probe.missing_programs) and the F22 pairing."""
    lines = card.factory_lines(text)
    report['factory_lines'] = lines
    for key in probe.missing_programs(lines, report['_requested']):
        report['failures'].append('factory log: no [QWEN-SDPA] line for flags=0x%x B=%d St=%d mask_width_t=%d '
                                  '(graft mounted, not executed)' % key)
    events = extent_lines(text)
    report['extent_lines'] = [fields for kind, fields in events if kind == 'F22']
    for problem in extent_line_problems(events):
        report['failures'].append('extent log: %s' % problem)
    report['cb_bytes'] = cb_extras(events, report.get('capacity', CAPACITY))
    for entry in report['cb_bytes']:
        if not 0 < entry['extra'] <= 1024:
            report['warnings'].append('cb_bytes of the 0x20 program flags=%s B=%d PNHt=%d is %d, K64i\'s table %d: '
                                      'expected two cur_pos sticks more' % (entry['flags'], entry['B'], entry['PNHt'],
                                                                            entry['cb_bytes'], entry['k64i_cb_bytes']))


def main(argv=None):
    args = parse_args(argv)
    probe.DEADLINE = probe.Deadline(args.deadline_s)
    report = dict(card=CARD, plan=PLAN, passed=False, argv=list(sys.argv[1:] if argv is None else argv),
                  capacity=args.capacity, extents=args.extents, starts=args.starts, seeds=args.seeds,
                  variants=args.variants, combos=[combo_name(*combo) for combo in args.combos],
                  trace_combos=[combo_name(*combo) for combo in args.trace_combos], sections=args.sections,
                  deadline_s=args.deadline_s, run_order=list(RUN_ORDER), flag='0x%x' % EXTENT,
                  poison=dict(k=probe.POISON_K, v=probe.POISON_V, blocks=probe.POISON_BLOCKS),
                  predictions=probe.split_predictions(args.extents, args.capacity, [3, 2, 1]),
                  not_run=dict(NOT_RUN), env={name: os.environ.get(name) for name in ENV_RECORDED},
                  watchdog=args.watchdog, failures=[], warnings=[], comparisons=[], liveness=[])
    if set(CB2A_SECTIONS) & set(args.sections):
        tickets = k2_tickets(args.k2_sweep, args.k2_floor, args.cb2_extents, args.cb2_starts)
        report['cb2a'] = dict(
            k2_sweep=list(args.k2_sweep), k2_floor=list(args.k2_floor) if args.k2_floor else None,
            cb2_extents=args.cb2_extents, cb2_starts=args.cb2_starts, z_families=args.z_families,
            z_starts=args.z_starts, output_memory=args.output_memory, min_live_start=MIN_LIVE_START,
            served=dict(flags='0x%x' % SERVED_FLAGS, compile_flags='0x%x' % COMPILE_FLAGS, rows=SERVED_ROWS,
                        batch=SERVED_BATCH, offsets=list(SERVED_OFFSETS), k_chunk_size=K_CHUNK),
            k2_tickets={kind: sum(1 for ticket in tickets if ticket['kind'] == kind)
                        for kind in ('sweep', 'family', 'floor')})
    if 'K2' in args.sections:
        report['native_decode'] = dict(NATIVE_DECODE)
        report['unverified'] = list(UNVERIFIED)
        for line in native_decode_lines():
            print(line, flush=True)
    report['_requested'] = set()
    native = card.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        payload['requested_programs'] = sorted(list(key) for key in report.get('_requested', ()))
        payload['tally'] = tally(report.get('comparisons', []))
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def checkpoint(tag):
        try:
            write_report(dict(in_progress='after %s' % tag))
        except Exception as error:  # noqa: BLE001 - the final write still comes
            report['warnings'].append('checkpoint write after %s failed: %s' % (tag, probe.one_line(error)))

    def on_fire(label):
        try:
            write_report(dict(error='watchdog: %r exceeded its budget' % (label,), passed=False))
        except Exception:  # noqa: BLE001 - the WATCHDOG line stands
            pass

    watchdog = k1.Watchdog(args.watchdog, on_fire=on_fire).start()
    probe.WATCHDOG = card.WATCHDOG = k1.WATCHDOG = watchdog
    installed, previous = probe.install_signal(signal.SIGTERM, term_handler(report, write_report))
    try:
        try:
            with native:
                run(args, report, checkpoint)
            if report.get('binary', {}).get('stage', 0) >= 1:
                check_log(report, native.text())
        except probe.Terminated:
            pass                                                # report['error'] was set by the handler
        except Exception as error:  # noqa: BLE001
            report['error'] = '%s: %s' % (type(error).__name__, error)
        report['decision'] = decide(report)
        report['verdict_line'] = verdict_line(report)
        report['passed'] = report['decision']['verdict'] == 'PASS'
    finally:
        if installed:
            probe.install_signal(signal.SIGTERM, previous)
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    for warning in report['warnings']:
        print('WARN', warning)
    if report.get('error'):
        print('ERROR', report['error'])
    print(report.get('verdict_line', CARD + ' verdict=NO-DECISION'), flush=True)
    print('SDPA_K64J_CARD passed=%s comparisons=%d liveness=%d failures=%d warnings=%d report=%s native_log=%s' % (
        report['passed'], len(report['comparisons']), len(report['liveness']), len(report['failures']),
        len(report['warnings']), args.out, native.path), flush=True)
    if report.get('terminated'):
        return 128 + int(report['terminated'])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
