"""Sampled host audits: QWEN_FAST_AUDIT_FIRST and QWEN_FAST_AUDIT_EVERY (default off, gate profiles only; stdlib only, py 3.7).

WHY. The in-trace audits of the verify block (the GDN glue audit QWEN_FAST_TP4_VGLUE_AUDIT, the packed-window audit QWEN_FAST_VERIFY_T2_AUDIT, the F1 spread audit, the K5-A audit)
build the served path beside the lever INSIDE the captured trace, so the device pays for them on every replay whatever the host does; what the host adds is the READ-BACK after each
replay: `tp4_vglue.audit_round` copies every audited entry of two GDN layers off all four chips (an M3 layer holds 37 entries of a four-user block, an octo layer 73 of eight users:
8 pieces, the merged output and 8 x 8 block-conv entries, each read as one tensor per chip, twice). At eight users that is roughly a thousand small `to_torch` calls after EVERY round, and
round one reads all 48 layers. The octo-2 audited glue8 attach (profile ...-octo-glue8-audit) warmed for 412 s and decoded at 3.4 tok/s a seat until its users timed out: the audit was not
wrong, it was never going to finish.

WHAT. The host read-backs are SAMPLED. Round r (1-based, per block) is audited when r <= FIRST, or when (r - FIRST) is a multiple of EVERY. Unset, every round is audited exactly as
before (this module then changes nothing: `sampled` is True and `ordinal(r)` is r). Set, the audits keep their layer rotation by the ORDINAL of the audited round (the k-th audited round
audits what the k-th round did: every layer on the first, then two per audited round in rotation), so sampling thins the rounds, not the layers an audit covers over a long run.

  QWEN_FAST_AUDIT_FIRST   a decimal integer >= 1: the first N rounds of a block are all audited (default 1 when only EVERY is set).
  QWEN_FAST_AUDIT_EVERY   a decimal integer >= 1: after those, every K-th round is audited. Required: FIRST without EVERY is a configuration error (it would audit nothing after N rounds).

Anything malformed raises ValueError naming the flag (never a silent "all rounds"). What is NOT sampled: the counter audits that read a few words after each replay (the page writer, the CCL
options, the MLP, the shard argmax, the multi SDPA, the round-host and pre-stage audits) - they cost almost nothing - and the audits that guard a round's own state (the extent audit, the
singles audit). The audit line a sampled arm writes says `sample=first:N,every:K` so a log states what it checked.

A sampled audit is weaker than an every-round one by construction (it saw one round in K), which is the point of the profile that carries it: the audited attach proves the lever on the rounds
it audits, the text gates and the timed pairs prove the rest. The profile's description says so.
"""

import os

FIRST_FLAG = 'QWEN_FAST_AUDIT_FIRST'
EVERY_FLAG = 'QWEN_FAST_AUDIT_EVERY'
NAMES = (FIRST_FLAG, EVERY_FLAG)
DEFAULT_FIRST = 1


def _count(name, source):
    text = source.get(name)
    if text is None or text == '':
        return None
    if type(text) is not str or not text.isdigit() or text != str(int(text)) or int(text) < 1:
        raise ValueError('%s must be a decimal integer from 1, got %r' % (name, text))
    return int(text)


def settings(environ=None):
    """None while sampling is off (neither flag set), else (first, every)."""
    source = os.environ if environ is None else environ
    first, every = _count(FIRST_FLAG, source), _count(EVERY_FLAG, source)
    if first is None and every is None:
        return None
    if every is None:
        raise ValueError('%s=%d without %s: the audits would stop after the first %d rounds (set both)' % (FIRST_FLAG, first, EVERY_FLAG, first))
    return (DEFAULT_FIRST if first is None else first, every)


def enabled(environ=None):
    return settings(environ) is not None


def sampled(round_number, environ=None):
    """Whether the block's round `round_number` (1-based) is audited. True for every round while sampling is off."""
    plan = settings(environ)
    if plan is None:
        return True
    first, every = plan
    if round_number < 1:
        raise ValueError('rounds are 1-based')
    return round_number <= first or (round_number - first) % every == 0


def ordinal(round_number, environ=None):
    """The 1-based position of `round_number` among the audited rounds (what an audit's layer rotation is keyed on). `round_number` itself while sampling is off."""
    plan = settings(environ)
    if plan is None:
        return round_number
    first, every = plan
    return round_number if round_number <= first else first + (round_number - first) // every


def label(environ=None):
    """'first:N,every:K' while sampling is on, else ''. Appended to the audits' lines."""
    plan = settings(environ)
    return '' if plan is None else 'first:%d,every:%d' % plan


def audited_rounds(total, environ=None):
    """The rounds among 1..total that are audited (the planner's and the tests' view)."""
    return [round_number for round_number in range(1, total + 1) if sampled(round_number, environ)]


def problems(environ=None):
    """[problem] naming a malformed flag, [] when sampling is off or well formed. The attach and the profile contract call this."""
    try:
        settings(environ)
    except ValueError as failure:
        return [str(failure)]
    return []
