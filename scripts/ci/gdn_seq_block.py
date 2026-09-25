"""K5-A: the packed verify's GDN recurrence with the work around the chain batched per block.

The served launch (`gdn_user_batch.execute`: `gdn_multitoken.load_kernels(root, True)` on 96
cores, one (user, head) per core) runs the whole T=1 decode step sixteen times per block: per
token it re-reads 18 DRAM pages, converts every input to fp32, normalises q and k, runs the
recurrence, then the fused norm/gate - 230 tile ops and 38 helper drains per token (K5 plan,
section 2). K5-A keeps the recurrence chain sequential and exact and moves everything that is
row-independent out of it:

  prologue  (once per block) the bf16 -> fp32 copies of q, k, v, z, g, beta and norm_w, exp(g)
            over the whole g tile, and the q/k L2 norms over all 16 token rows at once;
  chain     (per token) 124 tile ops: the served LLK calls on the served operand values, in the
            served srcA/srcB roles and K order, with the state add done in DST (DEST_TO_SRCB)
            and the state copy unpacked once and packed twice;
  epilogue  (once per block) the fused rms_norm * w * silu(z) over the 16 rows.
The reader stages each token's operands as bit copies of the prologue's packed fp32 words, so a
core reads 38 DRAM pages per block instead of 308. The 32 KiB bf16 snapshot of each token leaves on
both NoCs: K rows 0-1 from the writer (NoC 0), K rows 2-3 from the reader (NoC 1), each half in a
per-core rotated DRAM-bank order, into the served states layout. On one NoC the writes bound the
launch (card B: 383.8 us against 211.8 us with them compiled out); split, it runs at ~214.7 us.

Exactness class B* (K5 plan, 1.1 A1): every step is either the served call on the same values or
a property shown exact on this pinned tt-metal (the DEST_TO_SRCB add, docs/gdn-outer-add-
experiment.md; the double pack, docs/gdn-dual-state-copy.md). The rest - row independence of
the batched prologue/epilogue, SFPU exp position independence, TF32 idempotence of the staged
k~ column - is what the card-B probe's P0 byte compare (gdn_seq_block_device_test.py) decides.

Sources. The compute kernel is the served native compute prefix (decode_gated_delta_rule.cpp up
to `void kernel_main() {`, sha256-checked against gdn_multitoken.HASHES before slicing, the
gdn_shared_qk_compute.py pattern) followed by gdn_seq_block_compute.cpp: the served helpers are
reused verbatim. The reader and writer are new (gdn_seq_block_reader.cpp, _writer.cpp). Each
source carries a generated build header; SRC_TAG, the first 32 bits of each generated source's
sha256, is a compile arg so the JIT cache is keyed on content. No pinned file is changed.

Every CB has one producer RISC and one consumer RISC, with no exception: norm_w arrives in W
(reader -> compute) and the epilogue's bf16 round trips use RT, a compute-only ring - not the
norm_w ring, as the served kernel does, where the packer's local count never sees the reader's
push. So no CB changes hands and nothing needs the served tiles_received handoff. `execute` still
checks the served runtime pin (gdn_multitoken.validate_handoff_runtime, HANDOFF_HASHES) exactly as
gdn_user_batch.execute does, so turning the flag on never drops a runtime check the served launch
makes.

Flags (all read when the verify trace is built):
  QWEN_FAST_GDN_SEQ_BLOCK        '0' (default) or '1'; anything else raises. '1' requires
                                 QWEN_FAST_GDN_USER_BATCH=1.
  QWEN_FAST_GDN_SEQ_BLOCK_LEVEL  the A+ increments bitmask, default 0. A level whose generated
                                 sha256 triple is not in QUALIFIED is refused.
  QWEN_FAST_GDN_SEQ_BLOCK_AUDIT  audited arms only: layers (e.g. 0,23,47) whose served launch also
                                 runs on the same inputs (served_audit_launch, which keeps it out
                                 of the T1 gate's one-'coalesced'-per-layer count), both results
                                 compared after the replay (the audit, below).

QUALIFIED holds level 0 of variant A only: the triple card B's P0 passed at full plan coverage
(A 142/142 cases byte-identical to the served launch, N 0/14; probe run p0full, 2026-09-24).
The card-B probe builds anything else with the explicit `unqualified=True` builder argument,
never from the environment. The probe-only builds A0 (T5/T6 unfused, bisection) and N (the SFPU
state add, the negative control), and the timing diagnostics ('nosnap', 'passthrough'), are
builder arguments too, and can never be qualified.

The audit (QWEN_FAST_GDN_SEQ_BLOCK_AUDIT). gdn_user_batch_conv runs, for an audited layer, inside
the capture and right after that layer's own K5-A launch: a DRAM copy of each user's K5-A gated
output (ttnn.clone, which for an unchanged dtype is a reader -> writer page copy with no compute
kernel; attention_replay_audit snapshots its candidate the same way), then the served launch
(gdn_user_batch.execute, the served build) on the SAME input tensors with its output in DRAM,
not counted by the T1 gate. All of it is held outside `owned` (model_batch frees it with the
retained block). The model's own output is an L1 tensor the capture frees as scratch once the
layer has used it; the copy is taken while it is live, so no L1 is held across layers (holding
it once made a static CB/L1 clash, buffer start 736512 against CB end 742272,
docs/experiment-execution.md:1554-1558), and the served reference goes to DRAM for the same
reason. After every replay packed_verifier calls audit_round, which compares, per audited layer
and per user, on both chips:
  states   the model's own K5-A prefix states (the retained history the commit reads) against
           the served launch's - all 16 x 24 snapshots, exact bf16 bit patterns of the logical
           elements;
  output   the model's own K5-A gated output (its DRAM copy, written by the same replay) against
           the served launch's.
The compare reads through ttnn.to_torch, like verify_trace_t2's G1: on 9f9cd4f that readback
turns -0.0 and bf16 denormals into +0.0 and NaN into -Inf, so a difference in those alone is
invisible here (card B's P0 moved every byte raw and covered them). Padded tile rows are not
compared. One '[GDN-SEQ-BLOCK-AUDIT] layer=L user=U mismatches=N' line per (layer, user) per
round; any mismatch raises after the round's lines are logged.

Card B qualified the level-0 triple (P0 exact, P1 383.9 us against 526.2 us per launch); the
in-model evidence is the gate's (48 of 48 layers at level 0, every audit line mismatches=0).
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re

import gdn_multitoken as native
import gdn_user_batch as batch
import verify_trace_t1


DEFAULT_ROOT = batch.DEFAULT_ROOT
HERE = Path(__file__).resolve().parent
ROLES = ('reader', 'writer', 'compute')
SOURCES = dict(reader='gdn_seq_block_reader.cpp', writer='gdn_seq_block_writer.cpp',
               compute='gdn_seq_block_compute.cpp')
NATIVE_COMPUTE = 'compute/decode_gated_delta_rule.cpp'
MAIN = 'void kernel_main() {'
PREFIX_END = '}  // namespace\n\n'
BUILD_ANCHOR = '// @@GDN_SEQ_BLOCK_BUILD@@\n'
# The served helpers and names the generated compute calls; each must appear exactly once in the
# sliced prefix (the hash check already pins them - these name what the new code depends on).
NATIVE_ANCHORS = (
    'constexpr uint32_t cb_state = 5, cb_ones = 6;',
    'inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }',
    'inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }',
    'void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {',
    'void ew_off(uint32_t a, uint32_t ia0, uint32_t b, uint32_t ib0, uint32_t o, uint32_t n, int op) {',
    'void expc(uint32_t in, uint32_t o, uint32_t n) {',
    'void silu_tiles(uint32_t in, uint32_t o, uint32_t n) {',
    'void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {',
    'void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {',
    'void rowbcast_delta(uint32_t delta, uint32_t o, uint32_t Vt) {',
    'void rowsum_k(uint32_t in, uint32_t o, uint32_t Kt) {',
    'void inv_rms(uint32_t in, uint32_t o, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {',
    'void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {',
)

FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK'
LEVEL_FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK_LEVEL'
AUDIT_FLAG = 'QWEN_FAST_GDN_SEQ_BLOCK_AUDIT'
ROWS = 16
HEADS = batch.HEADS
LAYERS = 48
LEVEL_BITS = 4                 # the A+ increments (i)-(iv) of the plan
IMPLEMENTED_LEVELS = (0,)      # none of the increments exists yet
VARIANTS = ('A', 'A0', 'N')    # A: served candidate; A0: bisection only; N: negative control
DIAGNOSTICS = (None, 'nosnap', 'passthrough')  # P1 (iii) timing builds; never exact, never served

# level -> {'reader': sha256, 'writer': sha256, 'compute': sha256} of the generated variant-A
# sources, committed only after card-B P0 shows 0 differing bytes at full plan coverage (the
# fused_commit.QUALIFIED_KERNEL_SHA256 pattern). Level 0 is the split-NoC snapshot writer (W4):
# probe run r2 on image P5 (tt-metal 9f9cd4fd, native compute b59314e0), 'pass' at full plan
# coverage (142/142, R4 128/128, traced exact), may_commit_qualified true; 214.8 us per launch
# against the served 526.2 us. It replaces the single-NoC build (c9becdc5 / 277082a8 / 5275b2cc,
# run p0full, 383.9 us), which was never served.
QUALIFIED = {
    0: dict(reader='784deb4e398cfaa5a7c2b415c1a8be71925ccfd61fe18d801b11fb138dbc45c0',
            writer='215ef7da5d01d5c5fa720e8f6218481f27491c075151140376b457d760e1870d',
            compute='a9782f1434673d05fb7270b810836101ef38b51becfbd5c13d67880a937deffe'),
}

# model_batch's per-capture marker (K5 plan 3.1) and the gate's regex for it (3.9).
MARKER = '[PINDIAG] gdn seq_block calls this captured forward:'
MARKER_TEMPLATE = MARKER + ' {} of {} GDN layers level={}'   # model_batch's pindiag template
GATE_PATTERN = re.compile(r'gdn seq_block calls this captured forward: ([1-9][0-9]*) of ([0-9]+) GDN layers '
                          r'level=([0-9]+)')
AUDIT_MARKER = '[GDN-SEQ-BLOCK-AUDIT]'
AUDIT_PATTERN = re.compile(r'\[GDN-SEQ-BLOCK-AUDIT\] layer=([0-9]+) user=([0-9]+) mismatches=([0-9]+)')
# The per-user result key gdn_user_batch_conv sets when K5-A ran (and the level it ran at), and
# the one holding an audited layer's launches (outside `owned`).
RESULT_KEY, LEVEL_KEY, AUDIT_KEY = 'seq_block', 'seq_block_level', 'seq_block_audit'

# ---- the CB plan: index -> (name, pages, dtype, producer RISC, consumer RISC) ----
# The K5 plan's section 3.3 table, with two changes. (1) Its W (8 pages, reader -> compute for
# norm_w, then compute -> compute for the epilogue's round trips, the served full ring) is split into
# W (4 pages, reader -> compute) and RT (4 pages, compute -> compute). Same bytes, and no CB with two
# producer RISCs: in the served pattern the packer's local tiles_received never counts the reader's
# push, so the second round trip's wait passes before its pack lands (llk_io_pack.h:58,68;
# llk_io_unpack.h:30) and only pipeline depth kept it safe. (2) The snapshot ring is split so both
# data-movement RISCs write it, one NoC each: SOUT (K rows 0-1, compute -> writer) and SOUT2 (K rows
# 2-3, compute -> reader), 8 pages a token each; SOUT2 takes index 21, freed by running the k norm
# through FAC_Q after the q norm has popped it (FAC_K is gone). SOUT holds two half-tokens and SOUT2
# 14 pages, so the total is today's 630,784 B exactly; both are popped two pages (a column) at a time.
PAGE_BYTES = dict(bf16=2048, fp32=4096)
RISCS = ('reader', 'compute', 'writer')
ONES = 6
CB_PLAN = {
    0: ('IN_QK', 8, 'bf16', 'reader', 'compute'),
    1: ('IN_V', 4, 'bf16', 'reader', 'compute'),
    2: ('IN_Z', 4, 'bf16', 'reader', 'compute'),
    3: ('IN_GB', 2, 'bf16', 'reader', 'compute'),
    4: ('S0', 16, 'bf16', 'reader', 'compute'),
    5: ('FB', 16, 'bf16', 'compute', 'compute'),
    6: ('ONES', 1, 'fp32', 'reader', 'compute'),
    7: ('SOUT', 16, 'bf16', 'compute', 'writer'),    # snapshot K rows 0-1, NoC 0: 8 pages a token
    8: ('W', 4, 'bf16', 'reader', 'compute'),
    9: ('OUT', 4, 'bf16', 'compute', 'writer'),
    10: ('X', 8, 'fp32', 'compute', 'compute'),
    11: ('QKN', 8, 'fp32', 'compute', 'reader'),
    12: ('VF', 4, 'fp32', 'compute', 'reader'),
    13: ('ZF', 4, 'fp32', 'compute', 'compute'),
    14: ('GF', 1, 'fp32', 'compute', 'compute'),
    15: ('BF', 1, 'fp32', 'compute', 'reader'),
    16: ('GEXP', 1, 'fp32', 'compute', 'reader'),
    17: ('WF', 4, 'fp32', 'compute', 'compute'),
    18: ('NSQ', 4, 'fp32', 'compute', 'compute'),
    19: ('SUM', 1, 'fp32', 'compute', 'compute'),
    20: ('FAC_Q', 1, 'fp32', 'compute', 'compute'),
    21: ('SOUT2', 14, 'bf16', 'compute', 'reader'),  # snapshot K rows 2-3, NoC 1: 8 pages a token
    22: ('TOKA', 10, 'fp32', 'reader', 'compute'),
    23: ('TOKB', 8, 'fp32', 'reader', 'compute'),
    24: ('S', 16, 'fp32', 'compute', 'compute'),
    25: ('H', 16, 'fp32', 'compute', 'compute'),
    26: ('UD', 4, 'fp32', 'compute', 'compute'),
    27: ('DL', 8, 'fp32', 'compute', 'compute'),
    28: ('OT', 4, 'fp32', 'compute', 'writer'),
    29: ('O', 4, 'fp32', 'writer', 'compute'),
    30: ('RT', 4, 'bf16', 'compute', 'compute'),
}
# A0 (bisection only): T5 unfused, the outer product in its own 16-page ring (+64 KiB).
A0_EXTRA = {31: ('OUTER', 16, 'fp32', 'compute', 'compute')}
SERVED_CB_BYTES = sum(native.cb_plan(True)[0].values()) * 2048 + sum(native.cb_plan(True)[1].values()) * 4096


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_GDN_SEQ_BLOCK=1. Default OFF; anything but '0' or '1' is a configuration error, and
    so is '1' without QWEN_FAST_GDN_USER_BATCH=1 (the new launch replaces the batched one)."""
    environ = os.environ if environ is None else environ
    on = _flag(FLAG, environ)
    if on and not batch.enabled(environ):
        raise ValueError('%s=1 requires %s=1' % (FLAG, batch.FLAG))
    return on


def level(environ=None):
    """QWEN_FAST_GDN_SEQ_BLOCK_LEVEL: a decimal bitmask of the A+ increments, default 0."""
    value = (os.environ if environ is None else environ).get(LEVEL_FLAG, '0')
    if type(value) is not str or not value.isdigit() or value != str(int(value)):
        raise ValueError('%s must be a non-negative decimal integer' % LEVEL_FLAG)
    wanted = int(value)
    if wanted >= 1 << LEVEL_BITS:
        raise ValueError('%s holds %d increment bits' % (LEVEL_FLAG, LEVEL_BITS))
    return wanted


def audit_layers(environ=None):
    """QWEN_FAST_GDN_SEQ_BLOCK_AUDIT: comma-separated GDN layer indices, e.g. '0,23,47'. Unset or
    empty: no audit. Anything else - a blank item, a sign, a leading zero, a repeat, a layer
    outside 0..47 - is a configuration error."""
    value = (os.environ if environ is None else environ).get(AUDIT_FLAG, '')
    if value == '':
        return ()
    layers = []
    for item in value.split(','):
        if not item.isdigit() or item != str(int(item)) or int(item) >= LAYERS or int(item) in layers:
            raise ValueError('%s must list distinct GDN layers 0..%d, e.g. 0,23,47' % (AUDIT_FLAG, LAYERS - 1))
        layers.append(int(item))
    return tuple(layers)


def marker(calls, layers, built_level):
    """model_batch's line after a captured forward: how many GDN layers ran this launch."""
    return MARKER_TEMPLATE.format(int(calls), int(layers), int(built_level))


def audit_line(layer, user, mismatches, **fields):
    """The audit's per-(layer, user) line; `fields` follow in the order given (round, output, states)."""
    return '%s layer=%d user=%d mismatches=%d%s' % (AUDIT_MARKER, layer, user, mismatches,
                                                   ''.join(' %s=%s' % item for item in fields.items()))


def audit_active(environ=None):
    """QWEN_FAST_GDN_SEQ_BLOCK_AUDIT's layers when QWEN_FAST_GDN_SEQ_BLOCK=1, else (): what
    model_batch and packed_verifier read once, at construction."""
    return audit_layers(environ) if enabled(environ) else ()


_AUDIT_LAYER = [None]


@contextmanager
def audit_scope(layer):
    """model_batch names the GDN layer (0..47) whose decode runs inside; gdn_user_batch_conv asks
    audit_layer() whether QWEN_FAST_GDN_SEQ_BLOCK_AUDIT covers it. The previous layer comes back
    on exit, whatever the decode raised."""
    if type(layer) is not int or not 0 <= layer < LAYERS:
        raise ValueError('A GDN layer 0..%d required' % (LAYERS - 1))
    previous = _AUDIT_LAYER[0]
    _AUDIT_LAYER[0] = layer
    try:
        yield layer
    finally:
        _AUDIT_LAYER[0] = previous


def audit_layer(environ=None):
    """The GDN layer in audit_scope when QWEN_FAST_GDN_SEQ_BLOCK_AUDIT lists it, else None."""
    layer = _AUDIT_LAYER[0]
    return layer if layer is not None and layer in audit_layers(environ) else None


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


# ---- CB plan ----

def plan(variant='A'):
    if variant not in VARIANTS:
        raise ValueError('Unknown gdn_seq_block variant %r' % (variant,))
    entries = dict(CB_PLAN)
    if variant == 'A0':
        entries.update(A0_EXTRA)
    return entries


def cb_plan(variant='A'):
    """(bf16 pages by index, fp32 pages by index), the shape of gdn_multitoken.cb_plan."""
    entries = plan(variant)
    return ({index: entry[1] for index, entry in entries.items() if entry[2] == 'bf16'},
            {index: entry[1] for index, entry in entries.items() if entry[2] == 'fp32'})


def cb_bytes(variant='A'):
    return sum(entry[1] * PAGE_BYTES[entry[2]] for entry in plan(variant).values())


# ---- sources ----

def _replace_once(source, before, after, what):
    if source.count(before) != 1:
        raise ValueError('%s: expected one %r' % (what, before.strip()))
    return source.replace(before, after, 1)


def prefix(source):
    """The served native compute up to its entry point: includes, CB names, helpers."""
    if source.count(MAIN) != 1:
        raise ValueError('Expected one pinned compute entry point')
    head = source[:source.index(MAIN)]
    if not head.endswith(PREFIX_END):
        raise ValueError('Native prefix no longer ends with its helper namespace')
    for anchor in NATIVE_ANCHORS:
        if head.count(anchor) != 1:
            raise ValueError('Native helper anchor changed: %s' % anchor)
    return head


def native_prefix(root=DEFAULT_ROOT):
    """Read the served compute, check it against the pin BEFORE slicing, slice the prefix."""
    data = (Path(root) / native.KERNEL_ROOT / NATIVE_COMPUTE).read_bytes()
    if hashlib.sha256(data).hexdigest() != native.HASHES[NATIVE_COMPUTE]:
        raise ValueError('Native compute hash changed: %s' % NATIVE_COMPUTE)
    return prefix(data.decode())


def build_header(built_level, variant, diag):
    return ('// gdn_seq_block generated build: level=%d variant=%s diag=%s\n'
            '#define GDN_SEQ_BLOCK_LEVEL %d\n'
            '#define GDN_SEQ_BLOCK_VARIANT %d\n'
            '#define GDN_SEQ_BLOCK_DIAG %d\n') % (built_level, variant, diag or 'none', built_level,
                                                   VARIANTS.index(variant), DIAGNOSTICS.index(diag))


def _source(directory, role):
    data = (Path(directory) / SOURCES[role]).read_bytes().decode()
    if '\r' in data:
        raise ValueError('%s must be LF-only: its sha256 is its identity' % SOURCES[role])
    return data


def generate(root=DEFAULT_ROOT, built_level=0, *, variant='A', diag=None, sources=HERE):
    """role -> generated source text. Refuses an unimplemented level, variant or diagnostic."""
    if type(built_level) is not int or built_level not in IMPLEMENTED_LEVELS:
        raise ValueError('gdn_seq_block level %r is not implemented (levels %s)' % (built_level, IMPLEMENTED_LEVELS))
    if variant not in VARIANTS:
        raise ValueError('Unknown gdn_seq_block variant %r' % (variant,))
    if diag not in DIAGNOSTICS:
        raise ValueError('Unknown gdn_seq_block diagnostic %r' % (diag,))
    header = build_header(built_level, variant, diag)
    kernels = {}
    for role in ROLES:
        text = _replace_once(_source(sources, role), BUILD_ANCHOR, header, SOURCES[role])
        kernels[role] = native_prefix(root) + text if role == 'compute' else text
    return kernels


def sha256(kernels):
    return {role: hashlib.sha256(kernels[role].encode()).hexdigest() for role in ROLES}


def src_tag(source):
    """The first 32 bits of the source's sha256: a compile arg, so the JIT cache key covers content."""
    return int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)


class Build(dict):
    """role -> generated source, plus what it was generated as. Only load_kernels makes one."""

    def __init__(self, kernels, built_level, variant, diag, qualified):
        super().__init__(kernels)
        self.level, self.variant, self.diag, self.qualified = built_level, variant, diag, qualified

    def tag(self, role):
        return src_tag(self[role])


def load_kernels(root=DEFAULT_ROOT, built_level=0, *, variant='A', diag=None, unqualified=False, sources=HERE):
    """The generated sources, refused unless QUALIFIED holds their sha256 triple.

    `unqualified=True` is the card-B probe's builder argument and nothing else's: it is never read
    from the environment, and it is the only way to build A0, N or a diagnostic."""
    if type(unqualified) is not bool:
        raise ValueError('unqualified must be an explicit bool')
    kernels = generate(root, built_level, variant=variant, diag=diag, sources=sources)
    qualified = variant == 'A' and diag is None and QUALIFIED.get(built_level) == sha256(kernels)
    if not qualified and not unqualified:
        raise ValueError('gdn_seq_block level %d (variant %s, diag %s) is not qualified: QUALIFIED holds no '
                         'matching sha256 triple; only the card-B probe builds it unqualified'
                         % (built_level, variant, diag or 'none'))
    return Build(kernels, built_level, variant, diag, qualified)


_SERVED = {}


def served_kernels(root=None, environ=None):
    """The qualified build for QWEN_FAST_GDN_SEQ_BLOCK_LEVEL, generated once per (root, level)."""
    root = Path(root if root is not None else (os.environ if environ is None else environ).get(
        'TT_METAL_HOME', str(DEFAULT_ROOT)))
    key = (str(root), level(environ))
    if key not in _SERVED:
        _SERVED[key] = load_kernels(root, key[1])
    return _SERVED[key]


# ---- program ----

def compile_args(role, kernels):
    """Per role; every value but SRC_TAG and LEVEL is the served fused build's
    (gdn_user_batch.compile_args, which the unit tests hold these against)."""
    if role == 'reader':
        # Kt, Vt, H, RF, Ct, QOT, KOT, VOT, WTZ, ZOT, SRC_TAG
        return [4, 4, HEADS, 3, 160, 0, 32, 64, 96, 0, kernels.tag(role)]
    if role == 'writer':
        return [4, 4, HEADS, kernels.tag(role)]
    if role == 'compute':
        return [batch.bits(1e-6), batch.bits(128 ** -0.5), batch.bits(128e-6), batch.bits(128 ** 0.5),
                kernels.level, kernels.tag(role)]
    raise ValueError('Unknown kernel role')


# Indices into the per-user eight [qkv, beta, gate, initial, output, states, z, norm_w]. Both
# data-movement kernels write states: the writer K rows 0-1 of each token, the reader K rows 2-3.
ACCESSORS = dict(reader=(0, 1, 2, 3, 6, 7, 5), writer=(4, 5), compute=())


def runtime_args(role, head, addresses):
    """One core's runtime arguments from that user's eight buffer addresses. The compute reads
    none; it is still given the served [rows] so every core's argument shape is the served one."""
    if type(head) is not int or not 0 <= head < HEADS:
        raise ValueError('Head index within the 24-head TP2 shard required')
    if len(addresses) != 8:
        raise ValueError('Eight per-user buffer addresses required')
    if role == 'reader':
        return [head, addresses[0], addresses[1], addresses[2], addresses[3], addresses[6], addresses[7],
                addresses[5]]
    if role == 'writer':
        return [head, addresses[4], addresses[5]]
    if role == 'compute':
        return [ROWS]
    raise ValueError('Unknown kernel role')


def validate_users(shapes):
    """gdn_user_batch.validate_users, plus the one width this kernel is written for."""
    widths = batch.validate_users(shapes)
    if any(rows != ROWS for rows in widths):
        raise ValueError('gdn_seq_block runs 16-row segments only; other widths take the served launch')
    return widths


def build_program(operations, mesh, user_shards, kernels):
    """gdn_user_batch.build_program's geometry exactly - its core shares, its rectangle ranges and
    one descriptor per role under QWEN_FAST_VERIFY_T1 (#12), its configs - with these kernels,
    their arguments and this CB plan on the union of the cores.

    `user_shards[u]` is that user's eight per-chip tensor lists, in the order
    `[qkv, beta, gate, initial, output, states, z, norm_w]`.
    """
    if not isinstance(kernels, Build):
        raise ValueError('gdn_seq_block kernels must come from gdn_seq_block.load_kernels')
    users = len(user_shards)
    chips = batch.mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    shares = batch.core_shares(grid.x, grid.y, users)
    coalesce = verify_trace_t1.cut('coalesce')
    if coalesce:
        ranges = [verify_trace_t1.rectangle_set(operations, share) for share in shares]
        union = verify_trace_t1.rectangle_set(operations, [point for share in shares for point in share])
    else:
        ranges = [operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(*point),
                                                                operations.CoreCoord(*point))
                                           for point in share]) for share in shares]
        union = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(*point),
                                                              operations.CoreCoord(*point))
                                         for share in shares for point in share])
    buffers = []
    io, fp32 = cb_plan(kernels.variant)
    for counts, dtype, page in ((io, operations.bfloat16, 2048), (fp32, operations.float32, 4096)):
        for index, count in counts.items():
            buffers.append(operations.CBDescriptor(total_size=count * page, core_ranges=union,
                format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                    page_size=page, tile=operations.TileDescriptor(operations.Tile([32, 32])))]))
    configs = dict(
        reader=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_1,
                                                       noc=operations.NOC.RISCV_1_default),
        writer=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                                                       noc=operations.NOC.RISCV_0_default),
        compute=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4,
                                                   fp32_dest_acc_en=True, math_approx_mode=False))
    program = operations.MeshProgramDescriptor()
    coalesced = []
    for chip in range(chips):
        descriptors = []
        private, weights = [], []
        planned = []
        for shards, share, cores in zip(user_shards, shares, ranges, strict=True):
            local = [value[chip] for value in shards]
            addresses = [value.buffer_address() for value in local]
            if len(set(addresses)) != len(addresses):
                raise ValueError('One user inputs, initial state and prefix outputs must not alias')
            if any(address in private for address in addresses[:7]):
                raise ValueError('Packed users must not share any buffer but the norm weight')
            private.extend(addresses[:7])
            weights.append(addresses[7])
            for role in ROLES:
                args = list(compile_args(role, kernels))
                for index in ACCESSORS[role]:
                    args.extend(operations.TensorAccessorArgs(local[index]).get_compile_time_args())
                if coalesce:
                    planned.append((role, args, cores, [(point, runtime_args(role, head, addresses))
                                                        for head, point in enumerate(share)]))
                    continue
                runtime = operations.RuntimeArgs()
                for head, (horizontal, vertical) in enumerate(share):
                    runtime[horizontal][vertical] = runtime_args(role, head, addresses)
                descriptor = operations.KernelDescriptor(kernel_source=kernels[role],
                    source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=args, config=configs[role])
                descriptor.runtime_args = runtime
                descriptors.append(descriptor)
        if coalesce:
            descriptors, merged = batch.coalesced_descriptors(operations, kernels, configs, planned, union)
            coalesced.append(merged)
        if any(weight in private for weight in weights):
            raise ValueError('The shared norm weight must not alias any packed user buffer')
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=descriptors, cbs=buffers)
    if coalesce:
        # The T1 gate counts one 'coalesced' per GDN layer; this launch is that layer's launch.
        verify_trace_t1.note('coalesced' if all(coalesced) else 'coalesce_fallback')
    return program


def execute(mesh, users, operations=None, *, output_memory=None, kernels=None):
    """Every packed user's T=16 recurrence and fused norm/gate in ONE launch, K5-A kernels.

    The return is gdn_user_batch.execute's; the signature is the plan's (section 3.1), which is
    gdn_user_batch.execute's less its positional `kernels`: the build is the qualified one for
    QWEN_FAST_GDN_SEQ_BLOCK_LEVEL unless a build is passed BY KEYWORD (the card-B probe's own, or
    gdn_user_batch_conv's served_kernels(), taken once so it can report the level). So the served
    call `gdn_user_batch.execute(mesh, users, kernels, operations, output_memory=...)` becomes
    `execute(mesh, users, operations, output_memory=...)`; a build or a source dict in the
    `operations` slot (the drop-in mistake) is refused.
    `users` is a sequence of `(qkv, beta, gate, initial, z, norm_w)` tuples; returns
    `[(output, states), ...]` in user order, each the served shape, dtype, layout and placement -
    `(1, 16, 3072)` gated output in `output_memory` (L1 by default), `(16, 24, 128, 128)` bf16
    prefix states in DRAM.
    """
    if isinstance(operations, dict):
        raise ValueError('gdn_seq_block.execute takes operations third; pass a build as kernels=')
    if operations is None:
        import ttnn as operations

    if kernels is None:
        kernels = served_kernels()
    if not isinstance(kernels, Build):
        raise ValueError('gdn_seq_block kernels must come from gdn_seq_block.load_kernels')
    groups = [tuple(user) for user in users]
    widths = validate_users([[tuple(value.shape) for value in user] for user in groups])
    # The served launch's runtime pin (gdn_user_batch.py:306), kept so the flag never drops it.
    native.validate_handoff_runtime(Path(os.environ.get('TT_METAL_HOME', str(DEFAULT_ROOT))))
    if output_memory is None:
        output_memory = operations.L1_MEMORY_CONFIG
    if output_memory not in (operations.DRAM_MEMORY_CONFIG, operations.L1_MEMORY_CONFIG):
        raise ValueError('Output memory must be interleaved DRAM or L1')
    if any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT or
           value.memory_config() != operations.DRAM_MEMORY_CONFIG for user in groups for value in user):
        raise ValueError('Batched GDN requires interleaved DRAM BF16 TILE inputs')
    chips = batch.mesh_chips(mesh)
    grid = mesh.compute_with_storage_grid_size()
    batch.core_shares(grid.x, grid.y, len(groups))

    produced = []
    try:
        for rows in widths:
            output = operations.empty((1, rows, 3072), device=mesh, dtype=operations.bfloat16,
                                      layout=operations.TILE_LAYOUT, memory_config=output_memory)
            produced.append(output)
            states = operations.empty((rows, 24, 128, 128), device=mesh, dtype=operations.bfloat16,
                                      layout=operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG)
            produced.append(states)
        tensor_groups = [[*user[:4], produced[2 * index], produced[2 * index + 1], *user[4:]]
                         for index, user in enumerate(groups)]
        flat = []
        for group in tensor_groups:
            for value in group:
                if not any(value is kept for kept in flat):
                    flat.append(value)
        shards = {id(value): operations.get_device_tensors(value) for value in flat}
        if any(len(shards[id(value)]) != chips for value in flat):
            raise ValueError('Every chip of the mesh required for every batched GDN tensor')
        program = build_program(operations, mesh, [[shards[id(value)] for value in group]
                                                   for group in tensor_groups], kernels)
        operations.generic_op(flat, program)
        return [(produced[2 * index], produced[2 * index + 1]) for index in range(len(groups))]
    except BaseException:
        for value in produced:
            operations.deallocate(value)
        raise


def served_audit_launch(mesh, users, served, operations=None, *, output_memory=None):
    """QWEN_FAST_GDN_SEQ_BLOCK_AUDIT's second launch of an audited layer: the served batched launch
    (gdn_user_batch.execute with the served build) on the same inputs, returning its
    `[(output, states), ...]` for the after-replay compare.

    Its verify_trace_t1 note is NOT counted. Under QWEN_FAST_VERIFY_T1 the served build notes
    'coalesced' once per launch (gdn_user_batch.py:288), and the T1 gate requires exactly one per
    GDN layer per capture (lever_n_m3native_gate.py:1190-1194): counted, three audited layers would
    log 51 of 48 and fail an arm whose launches were all correct. So the counts are taken before
    the launch and restored after it, whatever it raises."""
    return _uncounted(lambda: batch.execute(mesh, users, served, operations, output_memory=output_memory))


def _uncounted(launch):
    """launch() with the verify_trace_t1 counts it notes dropped and the earlier ones kept."""
    kept = verify_trace_t1.take()
    try:
        return launch()
    finally:
        verify_trace_t1.take()
        for name, count in kept.items():
            verify_trace_t1.note(name, count)


def audit_launches(mesh, users, served, layer, outputs, operations=None):
    """QWEN_FAST_GDN_SEQ_BLOCK_AUDIT for one audited layer, inside the capture, right after that
    layer's own K5-A launch: `outputs`, that launch's gated outputs (one per user, still live),
    each copied to DRAM (ttnn.clone; an unchanged dtype is a page copy, no compute kernel), then
    the served launch (`served`, the served build) on the same `users` inputs with its outputs in
    DRAM, not counted by the T1 gate. Nothing here launches K5-A: the model's own output and its
    own retained states are what audit_round compares.

    Returns one dict per user, {'layer', 'output' (the copy of the model's own K5-A output),
    'served_output', 'served_states'}, held outside `owned`: the caller frees them (model_batch,
    with the retained block). On a raise everything allocated here is freed first."""
    if operations is None:
        import ttnn as operations
    outputs, groups = list(outputs), list(users)
    if len(outputs) != len(groups):
        raise ValueError('One K5-A output per audited user required (%d for %d)' % (len(outputs), len(groups)))
    dram = operations.DRAM_MEMORY_CONFIG
    copies, theirs = [], None
    try:
        for output in outputs:
            copies.append(operations.clone(output, memory_config=dram))
            if copies[-1] is output:
                copies.pop()
                raise AssertionError('The audit copy of a K5-A output must be a new tensor')
        theirs = served_audit_launch(mesh, groups, served, operations, output_memory=dram)
        if len(theirs) != len(copies):
            raise AssertionError('The served audit launch returned %d of %d users' % (len(theirs), len(copies)))
    except BaseException:
        for value in copies + [value for pair in theirs or () for value in pair]:
            operations.deallocate(value)
        raise
    return [dict(layer=layer, output=copy, served_output=served_output, served_states=served_states)
            for copy, (served_output, served_states) in zip(copies, theirs)]


def audit_held_of(result):
    """Every tensor the audit holds for a GDN layer result, every user's, in user order (the
    verify_trace_t2.audit_windows_of shape: a packed result lists its users in segment_results)."""
    pieces = result.get('segment_results') or (result,)
    return [value for piece in pieces if piece.get(AUDIT_KEY)
            for value in (piece[AUDIT_KEY]['output'], piece[AUDIT_KEY]['served_output'],
                          piece[AUDIT_KEY]['served_states'])]


def differing(operations, mine, theirs):
    """Differing bf16 elements of two device tensors, summed over chips: int16 views of to_torch
    (logical elements; see the module docstring for what that readback cannot see). A shape or
    chip-count difference counts every element; no chip, or an empty readback, counts as one
    (nothing compared is not a match)."""
    import torch

    left, right = operations.get_device_tensors(mine), operations.get_device_tensors(theirs)
    count = 0 if len(left) == len(right) and left else 1
    for one, two in zip(left, right):
        a = operations.to_torch(one).contiguous().view(torch.int16)
        b = operations.to_torch(two).contiguous().view(torch.int16)
        count += int((a != b).sum()) if a.shape == b.shape and a.numel() else max(a.numel(), b.numel(), 1)
    return count


def compare_record(operations, record, layer):
    """One retained GDN layer record, (state, result, checkpoint): per user, (user, output
    mismatches, states mismatches) - the model's own K5-A output (its DRAM copy) against the served
    one, and the model's own K5-A prefix states against the served ones. A user without the
    audit's launch for this layer, or a side compared with itself, raises: an audit that compared
    nothing must not read as a pass."""
    state, result, checkpoint = record
    pieces = result.get('segment_results') or (result,)
    found = []
    for user, piece in enumerate(pieces):
        held = piece.get(AUDIT_KEY)
        if not held or held.get('layer') != layer or not piece.get(RESULT_KEY):
            raise AssertionError('%s GDN layer %d user %d holds no K5-A audit launch' % (AUDIT_MARKER, layer, user))
        if held['output'] is held['served_output'] or piece['states'] is held['served_states'] or any(
                value is piece.get('output') for value in (held['output'], held['served_output'])):
            raise AssertionError('%s GDN layer %d user %d compares a tensor with itself or with the freed '
                                 'L1 output' % (AUDIT_MARKER, layer, user))
        found.append((user, differing(operations, held['output'], held['served_output']),
                      differing(operations, piece['states'], held['served_states'])))
    return found


def audit_round(operations, records, layers, round_number):
    """After a replay (packed_verifier): every audited layer's retained record through
    compare_record. One audit_line per (layer, user); if any element differed, raises after the
    whole round's lines are logged. Returns the number of (layer, user) pairs compared."""
    compared, total = 0, 0
    for layer in layers:
        if not 0 <= layer < len(records):
            message = '%s round=%d layer=%d: the block retains %d GDN layers' % (AUDIT_MARKER, round_number,
                                                                                  layer, len(records))
            log_line(message)
            raise AssertionError(message)
        try:
            found = compare_record(operations, records[layer], layer)
        except AssertionError as error:
            log_line('%s round=%d' % (error, round_number))
            raise
        for user, output, states in found:
            log_line(audit_line(layer, user, output + states, round=round_number, output=output, states=states))
            total += output + states
            compared += 1
    if total:
        raise AssertionError('%s round=%d: %d differing elements (K5-A against the served launch)'
                             % (AUDIT_MARKER, round_number, total))
    return compared


def audit(root=DEFAULT_ROOT, users=batch.MAX_USERS, built_level=0, variant='A', diag=None):
    """Host-only description of what one K5-A launch would build; never opens a device."""
    kernels = load_kernels(root, built_level, variant=variant, diag=diag, unqualified=True)
    io, fp32 = cb_plan(variant)
    return dict(status='host-source-and-placement audit only; no compilation or hardware certification',
                transforms='compute = the served native prefix (sha256-checked, sliced at kernel_main) + '
                           'gdn_seq_block_compute.cpp; reader and writer new',
                level=built_level, variant=variant, diag=diag, qualified=kernels.qualified,
                qualified_triple=QUALIFIED.get(built_level),
                native_sha256=native.HASHES[NATIVE_COMPUTE],
                generated_sha256=sha256(kernels),
                src_tag={role: kernels.tag(role) for role in ROLES},
                compile_args={role: compile_args(role, kernels) for role in ROLES},
                users=users, workers_per_user=HEADS, workers=users * HEADS,
                core_shares=batch.core_shares(11, 10, users), rows_per_user=ROWS,
                cb_indices=sorted(list(io) + list(fp32)),
                cb_bytes_per_worker=cb_bytes(variant), served_cb_bytes_per_worker=SERVED_CB_BYTES,
                cb_owners={index: dict(name=entry[0], pages=entry[1], dtype=entry[2], producer=entry[3],
                                       consumer=entry[4]) for index, entry in sorted(plan(variant).items())},
                dram_page_reads_per_worker=8 + 4 + 4 + 2 + 16 + 4,
                launches_per_layer=1)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Read-only host audit; never opens a device')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--users', type=int, default=batch.MAX_USERS)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--variant', choices=VARIANTS, default='A')
    parser.add_argument('--diag', choices=[value for value in DIAGNOSTICS if value], default=None)
    options = parser.parse_args()
    print(json.dumps(audit(options.root, options.users, options.level, options.variant, options.diag), indent=2))
