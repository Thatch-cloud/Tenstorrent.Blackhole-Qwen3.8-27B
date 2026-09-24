"""Verify-trace tuning T2 (QWEN_FAST_VERIFY_T2=1): cuts #1 and #2 of the verify-trace tuning
spec for the packed 64-row verify (four T16 users), both class B.

Off by default and read at each use, never at import. With the flag unset every path below
is the one that ran before it existed, and none of the modules the cuts need is imported.

  #1 'windows'   gdn_user_batch_conv.run_user_batched_projected: the four per-user
      gdn_conv_windows.build_windows launches of a GDN layer become ONE generic_op on every
      core of the grid (gdn_conv_windows_packed.py). It reads the same per-user pieces and
      histories the served launches read, at the same page and byte offsets, and moves the
      bytes with word copies (or local NOC copies) only: no compute kernel, no pack or unpack,
      so no value can round or be canonicalised. Faces 2-3 (tile rows 16-31) are zero in both,
      which is why the trimmed kernel needs rows == 16 (a host refusal and a static_assert).
      It reads the PIECES, not the 64-row block: the pieces of users 1 and 3 come through the
      served untilize / slice / tilize round trip, which this project measured to canonicalise
      -0 and denormals, and the windows are carried state - reading the block would be B*.
  #2 'kv_chains'  model_batch.cache_writer: the 64-row block's K/V write, today two 32-row
      launches of the audited ordered kernel per cache (each ONE semaphore chain over two
      users) plus a slice each, becomes one 64-row launch per cache with one chain PER USER
      (packed_ordered_cache.py). The kernels are ordered_cache.load_kernels' hash-pinned
      sources; only reader compile arg 7 (256 bytes of positions), CB16's page count (kept at
      the served 256) and the wait/signal runtime args change. Every cache tile row ends as
      the same read-modify-write chain in the same row order exactly when no two users write
      one (physical page, tile row) - the per-round host guard below. It runs three times:
      serving_packed_step.proposal_rows, BEFORE the round is drafted (a conflict drafts it at
      the engines' own widths, so the exact sequential step serves it); ineligible, at the
      step (the block's 16-row tickets have no capture beside the 64-row block, so a conflict
      first seen there refuses the round: every request fails, no byte is written); and,
      fail-closed, packed_verifier.stage_packed before any copy. vLLM allocations are
      disjoint (prefix caching is off for these models) and append-only, so the last two are
      backstops. The placeholders stay the served ones (every user on physical page 0, vLLM's
      null block); the one eager forward with them (the warm forward) runs ONE chain over all
      64 rows (model_batch kv_single_chain): the same kernels and compile args, the served row
      order, the served page-0 bytes. Only the captured forward bakes one chain per user.

Markers: packed_verifier logs MARKER once per captured verify trace with the counts of what
the captured forward engaged. FALLBACK, KV_SHARED and AUDIT_MISMATCH fail a gated arm.

QWEN_FAST_VERIFY_T2_SKIP (read only while the flag is on): a comma list of CUTS to leave out.
QWEN_FAST_VERIFY_T2_KV_ROWS: 64 (unset, the default) or 32 - two 32-row chained launches per
cache over the served tile metadata and slices, a no-rebuild fallback. Anything else raises.
QWEN_FAST_VERIFY_T2_AUDIT=1 (a correctness arm, never a timed one): the served windows are
built beside the packed ones inside the trace and compared, byte for byte, on both chips (G1
for #1): every GDN layer on the first round, then two layers per round in rotation (every
layer again within 24 rounds, however few rounds the arm runs). #2 has no in-trace audit -
both paths would write the same cache - so the audit logs each user's tile-row count per
round instead.
"""

import os

FLAG = 'QWEN_FAST_VERIFY_T2'
AUDIT_FLAG = 'QWEN_FAST_VERIFY_T2_AUDIT'
SKIP_FLAG = 'QWEN_FAST_VERIFY_T2_SKIP'
KV_ROWS_FLAG = 'QWEN_FAST_VERIFY_T2_KV_ROWS'
MARKER = '[PINDIAG] verify t2 engaged'
AUDIT_MARKER = '[PINDIAG] verify t2 audit'
AUDIT_MISMATCH = '[PINDIAG] verify t2 audit mismatch'
FALLBACK = '[PINDIAG] verify t2 fell back'
KV_SHARED = '[PINDIAG] verify t2 kv shared'
# #1, #2.
CUTS = ('windows', 'kv_chains')
KV_ROWS = (64, 32)
# The ordered kernel's addressing (reader_update_cache_interleaved_start_id.cpp): a 64-token
# paged block, 32-row tiles.
BLOCK_SIZE = 64
TILE_HEIGHT = 32
GDN_LAYERS = 48

# THE one table of what the cuts need at run time, by basename. Both image copy lists must
# name every entry (test_verify_trace_t2 checks it) and the card-M harness mounts exactly
# these (optimisation/ttnn-op/verify_t2/run_card_m.sh), so what card M tested is what ships.
RUNTIME_FILES = ('verify_trace_t2.py', 'gdn_conv_windows_packed.py', 'gdn_conv_windows_packed.cpp',
                 'packed_ordered_cache.py')

_COUNTS = {}
_LOGGED = set()
_AUDIT = dict(rounds=0)


def enabled():
    """QWEN_FAST_VERIFY_T2=1."""
    return os.environ.get('QWEN_FAST_VERIFY_T2') == '1'


def audit_enabled():
    """QWEN_FAST_VERIFY_T2_AUDIT=1 beside QWEN_FAST_VERIFY_T2=1."""
    return enabled() and os.environ.get('QWEN_FAST_VERIFY_T2_AUDIT') == '1'


def skipped():
    """The cuts QWEN_FAST_VERIFY_T2_SKIP names (empty when the flag is off). Raises on a name
    that is not a cut."""
    if not enabled():
        return frozenset()
    names = frozenset(name.strip() for name in os.environ.get('QWEN_FAST_VERIFY_T2_SKIP', '').split(',') if name.strip())
    unknown = sorted(names.difference(CUTS))
    if unknown:
        raise ValueError('%s names no cut: %s (cuts: %s)' % (SKIP_FLAG, ','.join(unknown), ','.join(CUTS)))
    return names


def cut(name):
    """QWEN_FAST_VERIFY_T2=1 and `name` not in QWEN_FAST_VERIFY_T2_SKIP."""
    if name not in CUTS:
        raise ValueError('Unknown verify t2 cut %r' % (name,))
    return enabled() and name not in skipped()


def kv_rows():
    """The chained K/V launch width: 64 unless QWEN_FAST_VERIFY_T2_KV_ROWS is exactly '32'
    ('64' is accepted as the default spelled out). Read once, at fixture construction."""
    value = os.environ.get('QWEN_FAST_VERIFY_T2_KV_ROWS')
    if value is None or value == '64':
        return 64
    if value == '32':
        return 32
    raise ValueError('%s must be 64 or 32; got %r' % (KV_ROWS_FLAG, value))


def note(cut, count=1):
    """Count one engagement of `cut` (reported, then reset, by the next take())."""
    _COUNTS[cut] = _COUNTS.get(cut, 0) + count


def take():
    """The engagements counted since the last take(), and a fresh count."""
    counts = dict(_COUNTS)
    _COUNTS.clear()
    return counts


def log_line(message):
    """One line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:
        pass


def log_once(message, key=None):
    """log_line, once per process per `key` (the message itself by default)."""
    key = message if key is None else key
    if key in _LOGGED:
        return False
    _LOGGED.add(key)
    log_line(message)
    return True


def fell_back(site, reason):
    """FALLBACK once per process per site: a gated arm fails on the first one."""
    return log_once('%s site=%s reason=%s' % (FALLBACK, site, reason), key=('fallback', site))


def engaged_line(site, **fields):
    """MARKER with the fields in the order given (the gate parses them into a dict)."""
    return '%s site=%s %s' % (MARKER, site, ' '.join('%s=%s' % (name, value) for name, value in fields.items()))


# --------------------------------------------------------------------------------------
# #2: the per-round host guard.
# --------------------------------------------------------------------------------------

def kv_tile_rows(positions, table):
    """Every (physical page, tile row) the ordered kernel's read-modify-write touches for
    these positions through this page-table row: block_size 64, TILE_HEIGHT 32, the reader's
    own `physical_block_id = pt[pos / 64]`, `block_row_tile = (pos % 64) / 32`. The key covers
    both heads and both caches, which share the geometry."""
    return {(int(table[int(position) // BLOCK_SIZE]), (int(position) % BLOCK_SIZE) // TILE_HEIGHT)
            for position in positions}


def kv_conflict(users):
    """None when no (page, tile row) is written by two users, else the first one found:
    dict(users=(first, second), page=..., tile_row=...). `users` is [(positions, table_row)]
    in SEGMENT order."""
    owner = {}
    for user, (positions, table) in enumerate(users):
        for key in sorted(kv_tile_rows(positions, table)):
            if owner.setdefault(key, user) != user:
                return dict(users=(owner[key], user), page=key[0], tile_row=key[1])
    return None


def kv_conflict_reason(conflict):
    return 'verify t2 kv tile rows shared: users %d,%d page %d tile row %d' % (
        conflict['users'][0], conflict['users'][1], conflict['page'], conflict['tile_row'])


def block_users(positions, pages, rows_per_user, users):
    """The guard's users from the block's host inputs (packed_host_inputs order: segment u is
    rows [u * rows_per_user, (u + 1) * rows_per_user), every row of a segment carrying that
    user's one table)."""
    return [([int(value) for value in positions[user * rows_per_user:(user + 1) * rows_per_user]],
             pages[user * rows_per_user])
            for user in range(users)]


# --------------------------------------------------------------------------------------
# #1: the in-model audit (G1).
# --------------------------------------------------------------------------------------

def audit_windows_of(result):
    """The served windows the audit built beside a GDN result's packed ones, every user's."""
    pieces = result.get('segment_results') or (result,)
    return [value for piece in pieces for value in (piece.get('audit_windows') or ())]


def audit_layers(round_number, layers=GDN_LAYERS):
    """The GDN layers round `round_number` (1-based) audits: all of them on round 1, then two
    per round in rotation - (2(r - 2)) % L and the next - so a 256-token arm (about 38 rounds
    at four users) compares every layer on its first replay and again within 24 rounds."""
    round_number = int(round_number)
    if round_number < 1:
        raise ValueError('rounds are 1-based')
    if round_number == 1:
        return tuple(range(layers))
    first = (2 * (round_number - 2)) % layers
    return (first, (first + 1) % layers)


def compare_windows(operations, record):
    """One retained GDN layer record's packed windows against the served ones built beside
    them, every user, every slot, every chip: int16 views of to_torch (logical rows; card M
    covers the padding). Returns (windows compared, [mismatch descriptions])."""
    import torch

    state, result, checkpoint = record
    pieces = result.get('segment_results') or (result,)
    compared, mismatches = 0, []
    for user, piece in enumerate(pieces):
        packed, served = list(piece.get('packed_conv_states') or ()), list(piece.get('audit_windows') or ())
        if len(packed) != 4 or len(served) != 4:
            mismatches.append('user %d holds %d packed and %d served windows' % (user, len(packed), len(served)))
            continue
        for slot, (mine, theirs) in enumerate(zip(packed, served)):
            chips = list(zip(operations.get_device_tensors(mine), operations.get_device_tensors(theirs)))
            for chip, (left, right) in enumerate(chips):
                a = operations.to_torch(left).contiguous().view(torch.int16)
                b = operations.to_torch(right).contiguous().view(torch.int16)
                if a.shape != b.shape or not torch.equal(a, b):
                    mismatches.append('user %d slot %d chip %d: %s' % (
                        user, slot, chip, int((a != b).sum()) if a.shape == b.shape else 'shape'))
            compared += 1
    return compared, mismatches


def layers_label(layers):
    """'0-47' for a contiguous run, else the comma list."""
    layers = list(layers)
    if len(layers) > 2 and layers == list(range(layers[0], layers[-1] + 1)):
        return '%d-%d' % (layers[0], layers[-1])
    return ','.join(str(layer) for layer in layers)


def audit_round(operations, records, round_number):
    """QWEN_FAST_VERIFY_T2_AUDIT: compare the layers audit_layers names for this round. Logs
    AUDIT_MARKER '<n> exact=True layers=<L> windows=<16 per layer>' or AUDIT_MISMATCH and
    raises."""
    layers = audit_layers(round_number, len(records) or GDN_LAYERS)
    compared, mismatches = 0, []
    for layer in layers:
        count, found = compare_windows(operations, records[layer])
        compared += count
        mismatches.extend('layer %d %s' % (layer, mismatch) for mismatch in found)
        if not count and not found:
            mismatches.append('layer %d nothing compared' % layer)
    if mismatches or not compared:
        message = '%s round=%d layers=%s %s' % (AUDIT_MISMATCH, round_number, layers_label(layers),
                                               '; '.join(mismatches[:4]) or 'nothing compared')
        log_line(message)
        raise AssertionError(message)
    _AUDIT['rounds'] += 1
    log_line('%s %d exact=True layers=%s windows=%d' % (AUDIT_MARKER, _AUDIT['rounds'], layers_label(layers), compared))
    return compared
