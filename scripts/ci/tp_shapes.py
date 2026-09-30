"""One geometry table for the fast path's tensor-parallel width, and the fail-closed choice of that width.

The S2 fast path (profile c2-packed) was built for the audited p150a pair, so its host code carries the pair's
numbers as literals: 12 folded query rows per token, 24 GDN value heads, 160 conv pages, 124,160 vocabulary
columns per shard. The four-card mesh (1, 4) halves every one of them. This module derives each per-chip width
from (tp, the model's config) in one place, so a port reads `geometry(tp).gdn_nv` where the pair's code says 24 and
a test can hold the two-card row to today's literals (test_tp_shapes) - the TP2 row must never move.

Choosing the width is separate and strict. QWEN_FAST_TP names it; unset means the pair, exactly as before. A
profile that names 4 (the four-card profiles do, beside mesh_device P150x4) is checked against the mesh the model
actually opened, and any disagreement stops the worker: no module infers the width from a chip count on its own
(memory: read-the-launched-argv - a value that never crossed into the process must not pass for the one that did).

Stdlib only, importable on py 3.7. Nothing here is sha256-pinned.
"""

from collections import namedtuple
import os

TP_SWITCH = 'QWEN_FAST_TP'
SUPPORTED = (2, 4)
PAIR = 2

TILE = 32
# Qwen3.8-27B (docs/decode-payload-bound.md; tp4 test_tp4_model_widths holds the same numbers against the graft).
HIDDEN = 5120
MLP = 17408
VOCAB = 248320
ATTENTION_HEADS = 24
ATTENTION_KV_HEADS = 4
ATTENTION_HEAD_DIM = 256
GDN_KEY_HEADS = 16
GDN_VALUE_HEADS = 48
GDN_HEAD_DIM = 128
GDN_CONV_TAPS = 4
# The a/b (beta, gate) columns of the fused qkvz projection are one per value head each.
GDN_AB_COLUMNS = 2
# The drafter (dflash_device.py, draft_attention_branch.py): its heads, MLP columns, embedding width and
# vocabulary are sharded over the chips, not replicated.
DRAFT_HEADS = 32
DRAFT_KV_HEADS = 8
DRAFT_HEAD_DIM = 128
# Rows of the quad's two-user pair fold sit above the head count and are not chips (quad_draft.py).
USERS_PER_PAIR = 2
# Pages: one fp32 128x128 recurrence state is 16 tiles per value head; the conv state is qkv / 32 tiles wide.
STATE_PAGES_PER_HEAD = 16
# The on-device sampler's largest per-device vocabulary (tt_sampling); vocab / tp must fit for it to be legal.
SAMPLER_LIMIT = 65536

Geometry = namedtuple('Geometry', (
    'tp', 'residual', 'vocab', 'mlp',
    'attn_heads', 'attn_kv_heads', 'attn_group', 'attn_fold_rows', 'attn_out',
    'gdn_nk', 'gdn_nv', 'gdn_qkv', 'gdn_z', 'gdn_qkvzab', 'gdn_qkvzab_padded', 'gdn_key', 'gdn_value',
    'gdn_state_pages', 'gdn_conv_pages', 'gdn_a_col', 'gdn_b_col',
    'draft_heads', 'draft_kv_heads', 'draft_query', 'draft_embedding', 'draft_taps'))


def ceil_tile(value):
    return -(-value // TILE) * TILE


def divide(total, tp, what):
    if total % tp:
        raise ValueError('%s (%d) does not split over %d chips' % (what, total, tp))
    return total // tp


def geometry(tp):
    """Every per-chip width at tensor-parallel width `tp` (2 or 4), or ValueError. Each is the model's total
    divided by the chip count; nothing here is measured."""
    if type(tp) is not int or tp not in SUPPORTED:
        raise ValueError('Tensor-parallel width must be one of %s, got %r' % (list(SUPPORTED), tp))
    heads = divide(ATTENTION_HEADS, tp, 'attention query heads')
    nk = divide(GDN_KEY_HEADS, tp, 'GDN key heads')
    nv = divide(GDN_VALUE_HEADS, tp, 'GDN value heads')
    qkv = 2 * nk * GDN_HEAD_DIM + nv * GDN_HEAD_DIM
    z = nv * GDN_HEAD_DIM
    qkvzab = qkv + z + GDN_AB_COLUMNS * nv
    kv_heads = divide(ATTENTION_KV_HEADS, tp, 'attention KV heads')
    return Geometry(
        tp=tp, residual=divide(HIDDEN, tp, 'hidden size'), vocab=divide(VOCAB, tp, 'vocabulary'),
        mlp=divide(MLP, tp, 'MLP width'),
        attn_heads=heads, attn_kv_heads=kv_heads, attn_group=heads // kv_heads,
        # The fold repacks a token's query heads as rows: one row per local query head.
        attn_fold_rows=heads, attn_out=heads * ATTENTION_HEAD_DIM,
        gdn_nk=nk, gdn_nv=nv, gdn_qkv=qkv, gdn_z=z, gdn_qkvzab=qkvzab, gdn_qkvzab_padded=ceil_tile(qkvzab),
        gdn_key=nk * GDN_HEAD_DIM, gdn_value=nv * GDN_HEAD_DIM,
        gdn_state_pages=STATE_PAGES_PER_HEAD * nv, gdn_conv_pages=qkv // TILE,
        # Columns of one projected row [qkv | z | a | b | pad]: a starts after qkv and z, b after a's one column
        # per value head (8192 / 8216 at the pair, 4096 / 4108 at four cards).
        gdn_a_col=qkv + z, gdn_b_col=qkv + z + nv,
        draft_heads=divide(DRAFT_HEADS, tp, 'drafter query heads'),
        draft_kv_heads=divide(DRAFT_KV_HEADS, tp, 'drafter KV heads'),
        draft_query=divide(DRAFT_HEADS, tp, 'drafter query heads') * DRAFT_HEAD_DIM,
        draft_embedding=divide(HIDDEN, tp, 'drafter embedding width'),
        draft_taps=divide(HIDDEN, tp, 'drafter feature taps'))


def number_word(count):
    """The word the fast path's refusal texts use for a small count (two, four, eight, sixteen)."""
    return {2: 'two', 4: 'four', 8: 'eight', 16: 'sixteen'}[count]


def active(environ=None):
    """The geometry of the width this process serves at (QWEN_FAST_TP, else the pair)."""
    return geometry(chip_count(environ))


def sampler_fits(tp):
    """Whether the on-device sampler accepts this width's vocabulary shard (62,080 at tp 4, not 124,160 at 2)."""
    return geometry(tp).vocab <= SAMPLER_LIMIT


def requested_tp(environ):
    """The width the launched process asked for: 2 when QWEN_FAST_TP is unset (the pair, unchanged), else the
    integer it names. A blank, non-decimal or unsupported value is refused rather than read as the pair."""
    if TP_SWITCH not in environ:
        return PAIR
    text = environ[TP_SWITCH]
    if not isinstance(text, str) or not text.isdigit() or int(text) not in SUPPORTED or str(int(text)) != text:
        raise ValueError('%s must be one of %s, got %r' % (TP_SWITCH, ', '.join(map(str, SUPPORTED)), text))
    return int(text)


def check_mesh(tp, num_devices, shape):
    """Refuse unless the opened mesh is exactly (1, tp) with tp devices; returns tp. The width came from the
    launched environment (requested_tp), the mesh from the model - both must say the same thing."""
    geometry(tp)
    if num_devices != tp or tuple(int(dim) for dim in shape) != (1, tp):
        raise ValueError('%s=%d needs a (1, %d) mesh of %d devices, but the model opened %s with %r devices'
                         % (TP_SWITCH, tp, tp, tp, list(shape), num_devices))
    return tp


def chip_count(environ=None):
    """How many per-chip shards a fast-path readback must return: the width this process serves at (QWEN_FAST_TP,
    else the pair). startup's select_for_model has already held it against the opened mesh, so a readback that
    finds another count has met a broken device set, not a different configuration."""
    return requested_tp(os.environ if environ is None else environ)


def all_chips(environ=None):
    """'Both' at the pair, 'All 4' at four cards: the subject of a refusal such as '<Both> chip-local outputs
    required', so the pair's messages stay the text they were."""
    count = chip_count(environ)
    return 'Both' if count == PAIR else 'All %d' % count


def count_word(environ=None):
    """'Two' at the pair, 'Four' at four cards: the leading word of a refusal such as '<Two> chip-local outputs
    required'."""
    return {2: 'Two', 4: 'Four'}[chip_count(environ)]


def vocab_shard(environ=None):
    """The vocabulary columns each chip holds of the LM head: 124,160 at the pair, 62,080 at four cards."""
    return geometry(chip_count(environ)).vocab


def select_for_model(environ, model):
    """The width `model` (the loaded target: num_devices, mesh_device.shape) must serve at, or ValueError.

    QWEN_FAST_TP set: the mesh must be exactly (1, tp). Unset: the pair's path, unchanged - a model that reports
    a mesh other than (1, 2) is refused (an unset switch must not serve a four-card mesh as if it were the pair).
    A stand-in model that reports no mesh at all is the caller's test double and is taken as the pair."""
    tp = requested_tp(environ)
    devices = getattr(model, 'num_devices', None)
    shape = getattr(getattr(model, 'mesh_device', None), 'shape', None)
    if TP_SWITCH in environ:
        if devices is None or shape is None:
            raise ValueError('%s=%d needs a model that reports its mesh' % (TP_SWITCH, tp))
        return check_mesh(tp, devices, shape)
    if (devices is not None and devices != PAIR) or (shape is not None and tuple(int(dim) for dim in shape) != (1, PAIR)):
        raise ValueError('%s is unset, which serves the (1, 2) pair, but the model opened %s with %r devices'
                         % (TP_SWITCH, None if shape is None else list(shape), devices))
    return PAIR


def mesh_width(mesh, environ=None):
    """The chip count of `mesh` when it is exactly the (1, tp) mesh this process serves at (QWEN_FAST_TP, else the
    pair), or None when it is not. Callers keep their own refusal text: `if width is None: raise ValueError(...)`.
    At the pair this is the `list(mesh.shape) == [1, 2]` test the fast path's collectives carried as a literal."""
    tp = requested_tp(os.environ if environ is None else environ)
    return tp if list(mesh.shape) == [1, tp] else None


def select(environ, num_devices, shape):
    """requested_tp and check_mesh in one call: the width this worker serves at, or ValueError."""
    return check_mesh(requested_tp(environ), num_devices, shape)


def k5_reader_arguments(tp):
    """The packed recurrence reader's compile-time geometry [Kt, Vt, H, RF, Ct, QOT, KOT, VOT, WTZ, ZOT] (without
    the source tag) at `tp`: gdn_seq_block.compile_args carries [4, 4, 24, 3, 160, 0, 32, 64, 96, 0] at the pair.
    The offsets are tile offsets in one row of [q | k | v]: k starts after the query columns, v after both."""
    found = geometry(tp)
    key_tiles = found.gdn_key // TILE
    return [GDN_HEAD_DIM // TILE, GDN_HEAD_DIM // TILE, found.gdn_nv, 3, found.gdn_conv_pages, 0,
            key_tiles, 2 * key_tiles, found.gdn_value // TILE, 0]
