"""The TP4 verify-glue levers (branch tp4/vglue): one strict flag per lever, and the markers a gated arm reads.

Every lever is a byte-identical replacement of ops the packed 64-row verify (or the GDN commit) already runs, so
each one is behind its own flag and an A/B can isolate it. All flags are strict: unset or '0' is off, '1' is on,
anything else raises. They are read at capture or attach (like verify_trace_t1/t2), never at import, and any of
them on while the process serves the pair (QWEN_FAST_TP unset or 2) raises: these levers exist for four cards only
and the pair's launches stay exactly what they were.

  QWEN_FAST_TP4_COMMIT_LANES   C1a  GDN commit: the 8-lane pipelined DMA (gdn_commit_lanes_tp.cpp).
  QWEN_FAST_TP4_SHARD_VALUES   V4a  sampler: the per-shard max value by gather at the argmax id, not a 2-core Reduce.
  QWEN_FAST_TP4_GDN_GLUE       V2   GDN split/merge as one DMA launch each (gdn_rows_dma_tp).
  QWEN_FAST_TP4_GDN_BLOCK_CONV V1   one conv-gates launch per GDN layer for all four users (needs V2).
  QWEN_FAST_TP4_ATTN_FOLD      V3a  attention query fold-in / result fold-out as one DMA launch each.
  QWEN_FAST_TP4_VGLUE_AUDIT    a correctness arm only: each engaged lever is compared with the served path in-trace.

Markers: FALLBACK and AUDIT_MISMATCH fail a gated arm; ENGAGED is the proof a lever ran.

Stdlib only, py 3.7.
"""

import os

import tp_shapes

COMMIT_LANES = 'QWEN_FAST_TP4_COMMIT_LANES'
SHARD_VALUES = 'QWEN_FAST_TP4_SHARD_VALUES'
GDN_GLUE = 'QWEN_FAST_TP4_GDN_GLUE'
GDN_BLOCK_CONV = 'QWEN_FAST_TP4_GDN_BLOCK_CONV'
ATTN_FOLD = 'QWEN_FAST_TP4_ATTN_FOLD'
AUDIT = 'QWEN_FAST_TP4_VGLUE_AUDIT'

LEVERS = (COMMIT_LANES, SHARD_VALUES, GDN_GLUE, GDN_BLOCK_CONV, ATTN_FOLD)
ALL_FLAGS = LEVERS + (AUDIT,)

ENGAGED = '[PINDIAG] tp4 vglue engaged'
FALLBACK = '[PINDIAG] tp4 vglue fell back'
AUDIT_MARKER = '[PINDIAG] tp4 vglue audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 vglue audit mismatch'

# What each lever needs at run time, by basename. The image copy lists and the CPU allowlist must name every entry
# (test_tp4_vglue checks it), so what the CPU tests proved is what ships.
RUNTIME_FILES = ('tp4_vglue.py', 'gdn_commit_lanes_tp.cpp', 'gdn_rows_dma_tp.py', 'gdn_rows_dma_tp.cpp',
                 'gdn_device_loop_state_tp.py', 'gdn_block_conv_tp.py', 'attention_block_fold_tp.py',
                 'attention_block_fold_tp.cpp')


def _read(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def _check_width(environ):
    source = os.environ if environ is None else environ
    if tp_shapes.chip_count(source) != tp_shapes.PAIR:
        return
    on = [name for name in ALL_FLAGS if _read(name, source)]
    if on:
        raise ValueError('%s are TP4 levers: they need QWEN_FAST_TP=4, this process serves the pair' % ', '.join(on))


def enabled(name, environ=None):
    """Whether lever `name` (one of LEVERS) is on. Strict; raises on any lever flag at two cards, and when a lever
    needs another that is off (V1 needs V2)."""
    if name not in LEVERS:
        raise ValueError('Unknown vglue lever %r' % (name,))
    _check_width(environ)
    on = _read(name, environ)
    if on and name == GDN_BLOCK_CONV and not _read(GDN_GLUE, environ):
        raise ValueError('%s needs %s=1 (the block conv reads the split and merge DMA)' % (GDN_BLOCK_CONV, GDN_GLUE))
    return on


def audit_enabled(environ=None):
    """QWEN_FAST_TP4_VGLUE_AUDIT=1 with at least one lever on."""
    _check_width(environ)
    return _read(AUDIT, environ) and any(_read(name, environ) for name in LEVERS)


def engaged_levers(environ=None):
    """The levers that are on, in LEVERS order."""
    return tuple(name for name in LEVERS if enabled(name, environ))


def marker(site, **counts):
    """The ENGAGED line: `[PINDIAG] tp4 vglue engaged site=<site> k=v ...` with the counts in the order given."""
    return ' '.join([ENGAGED, 'site=%s' % site] + ['%s=%s' % pair for pair in counts.items()])


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
