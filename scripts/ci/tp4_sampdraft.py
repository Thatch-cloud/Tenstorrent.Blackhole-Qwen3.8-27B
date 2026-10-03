"""The TP4 sampler-tail and drafter op levers (branch tp4/samp-draft): one strict flag per lever and the markers a gated arm reads.

Every lever replaces ops the served path already runs with a faster launch that returns the same bytes (the sampler: the same
ids; the drafter conv and head ops: the same tiles), so each is behind its own flag and an A/B can isolate it. All flags are strict:
unset or '0' is off, '1' is on, anything else raises. They are read at capture time (like tp4_vglue and draft_wide_tp), never at
import, and any of them on while the process serves the pair (QWEN_FAST_TP unset or 2) raises: these levers exist for four cards only
and the pair's launches stay exactly what they were.

  QWEN_FAST_TP4_SHARD_ARGMAX       S1   sampler: one tile-native scan kernel on 110 cores plus a one-core fold replaces the
                                        untilize + ArgMax + (gather or 2-core max) of verify_trace_t1.sample_shards.
  QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT      a correctness arm only (needs S1): today's sample_shards runs beside the kernel in the same
                                        capture and every row of every round is compared (ids exactly, values as numbers).
  QWEN_FAST_TP4_DRAFT_CONV         D2a  drafter: the fused convolution's I/O kernel is rewritten (word copies, a writer on the
                                        second data-movement core, double-buffered input); the compute kernel is the served one.
  QWEN_FAST_TP4_DRAFT_CONV_AUDIT        a correctness arm only (needs D2a): the served I/O kernel runs beside the new one on the same
                                        operands and the two outputs are byte-compared (compare_pending, called per round).
  QWEN_FAST_TP4_DRAFT_HEADS        D2c  drafter: nlp_create_qkv_heads / nlp_concat_heads (one core each) become one multi-core
                                        tile-copy launch each (a tile permutation: head_dim 128 is four whole tiles).
  QWEN_FAST_TP4_DRAFT_HEADS_AUDIT       a correctness arm only (needs D2c): the served head split / merge runs beside the launch on the
                                        same operands and the outputs are byte-compared on every chip, as D2a's audit does.

Markers: every FALLBACK and AUDIT_MISMATCH line fails a gated arm (c2_smoke_check); ENGAGED is the proof a lever ran.

Stdlib only, py 3.7.
"""

import os
import sys

import tp_shapes

SHARD_ARGMAX = 'QWEN_FAST_TP4_SHARD_ARGMAX'
SHARD_ARGMAX_AUDIT = 'QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT'
DRAFT_CONV = 'QWEN_FAST_TP4_DRAFT_CONV'
DRAFT_CONV_AUDIT = 'QWEN_FAST_TP4_DRAFT_CONV_AUDIT'
DRAFT_HEADS = 'QWEN_FAST_TP4_DRAFT_HEADS'
DRAFT_HEADS_AUDIT = 'QWEN_FAST_TP4_DRAFT_HEADS_AUDIT'

LEVERS = (SHARD_ARGMAX, DRAFT_CONV, DRAFT_HEADS)
AUDITS = {SHARD_ARGMAX_AUDIT: SHARD_ARGMAX, DRAFT_CONV_AUDIT: DRAFT_CONV, DRAFT_HEADS_AUDIT: DRAFT_HEADS}
ALL_FLAGS = LEVERS + tuple(AUDITS)

# One (engaged, fell back, audit, audit mismatch) marker prefix per lever; c2_smoke_check and the tests read these.
SARG_ENGAGED = '[PINDIAG] tp4 shard argmax engaged'
SARG_FALLBACK = '[PINDIAG] tp4 shard argmax fell back'
SARG_AUDIT = '[PINDIAG] tp4 shard argmax audit'
SARG_MISMATCH = '[PINDIAG] tp4 shard argmax audit mismatch'
CONV_ENGAGED = '[PINDIAG] tp4 draft conv engaged'
CONV_FALLBACK = '[PINDIAG] tp4 draft conv fell back'
CONV_AUDIT = '[PINDIAG] tp4 draft conv audit'
CONV_MISMATCH = '[PINDIAG] tp4 draft conv audit mismatch'
HEADS_ENGAGED = '[PINDIAG] tp4 draft heads engaged'
HEADS_FALLBACK = '[PINDIAG] tp4 draft heads fell back'
HEADS_AUDIT = '[PINDIAG] tp4 draft heads audit'
HEADS_MISMATCH = '[PINDIAG] tp4 draft heads audit mismatch'

# What each lever needs at run time, by basename. The image copy lists and the CPU allowlist must name every entry
# (test_tp4_sampdraft checks it), so what the CPU tests proved is what ships.
RUNTIME_FILES = ('tp4_sampdraft.py', 'tp4_shard_argmax.py', 'tp4_shard_argmax_scan.cpp', 'tp4_shard_argmax_fold.cpp',
                 'tp4_draft_conv.py', 'draft_conv_io_fast.cpp', 'draft_conv_out.cpp',
                 'tp4_draft_heads.py', 'draft_heads_copy.cpp')


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
    """Whether lever `name` (one of LEVERS) is on. Strict; raises on any lever or audit flag at two cards."""
    if name not in LEVERS:
        raise ValueError('Unknown sampdraft lever %r' % (name,))
    _check_width(environ)
    return _read(name, environ)


def audit_enabled(audit, environ=None):
    """Whether audit flag `audit` (one of AUDITS) is on. An audit without its lever is a misconfigured arm and raises: it
    would pass having audited nothing."""
    if audit not in AUDITS:
        raise ValueError('Unknown sampdraft audit %r' % (audit,))
    _check_width(environ)
    if not _read(audit, environ):
        return False
    if not _read(AUDITS[audit], environ):
        raise ValueError('%s needs %s=1: the audit would compare nothing' % (audit, AUDITS[audit]))
    return True


def drafter_audit_on(environ=None):
    """Whether either drafter audit (D2a's conv audit, D2c's heads audit) is on: the capture sites open an audit scope for it."""
    return audit_enabled(DRAFT_CONV_AUDIT, environ) or audit_enabled(DRAFT_HEADS_AUDIT, environ)


def validate(environ=None):
    """Every flag strict and every audit paired with its lever, checked once at attach: a misconfigured audit arm must fail before
    the first capture, not at its first round's readback after the cards are loaded. Raises ValueError."""
    _check_width(environ)
    for name in ALL_FLAGS:
        _read(name, environ)
    for audit in AUDITS:
        audit_enabled(audit, environ)


def log_line(message):
    """One line into the server log: loguru where it exists, stderr otherwise (packed_verifier.diagnostic's way). Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, file=sys.stderr, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:
        pass


_LOGGED = set()


def note(kind, text):
    """One marker line per (kind, text), so a loop of 20 convs prints its few distinct lines."""
    key = (kind, text)
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    log_line('%s %s' % (kind, text))
