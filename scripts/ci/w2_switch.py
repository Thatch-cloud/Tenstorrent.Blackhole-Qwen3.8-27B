"""The W2 runtime kill switch: the file w2.off beside levern.off, prefix-reuse.off and parked.off (stdlib only, host only, py 3.7).

W2 is QWEN_FAST_TP4_SDPA=multi (one SDPA launch for every user of a packed block, sdpa_multi_tp) plus QWEN_FAST_TP4_CONV_GATES_SPREAD=1 (F1, the
conv-gates launch on spread gate cores, gdn_conv_gates_spread). Both are BAKED INTO THE PACKED BLOCK'S CAPTURED TRACES: the verify trace is recorded
once, at the attach, and a replay re-issues the device program and never comes back to Python, so no flag can switch it per round, and a second
trace without W2 would need a second capture (trace region and DRAM that production does not have free) that no card has ever run. The kill switch
therefore works at two points, both on the host and both between traces, and it never touches a trace:

  1. LIVE, at the next round decision (about one round, at most POLL_S plus the round in flight). The file appears; the next question a round asks
     (serving_packed_step.proposal_rows / block_rows_for before the drafts, ineligible at the step) latches the switch and answers "the packed block does
     not serve this round". From then on EVERY round on a block W2 runs on goes to the sequential step whole - the existing, qualified fallback for a round
     the block cannot serve (one pass per user over each user's own per-request engine): the SERVED SDPA and the UNSPREAD conv gates, tickets drafted at the
     engines' own widths. It is byte-identical in greedy output (a packed round and a sequential round are exact against each other; that is what the packed
     gates prove) and slower: a brown-out, not a mode to stay in. The round whose trace is already replaying finishes untouched. The latch holds for the
     life of the process (removing the file does not bring the packed block back; a restart does).
  2. AT THE NEXT ATTACH (an engine restart with the file still on the hub mount): W2 is not attached at all. sdpa_long_tp.apply leaves `multi` None and
     gdn_block_conv_tp.stage runs the served conv-gates call, so the packed blocks are captured WITHOUT W2 and serve at full packed speed on the pre-W2 paths.
     A restart with the file present is the way back to full speed without an image rollback; removing the file and restarting brings W2 back. Audit flags
     (QWEN_FAST_TP4_SDPA_AUDIT, QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT) are a gate's own and an audited lever cannot be skipped: with one set the attach
     ignores the file (one line says so) and only the live latch applies.

A process whose environment does not name W2 (neither lever) never reads the file: `configured` is False and every function below answers at once with
no system call, so a profile without W2 is byte for byte what it was. With W2 on and the file absent the only cost is one stat() at most every POLL_S
seconds; the output of every round is unchanged.

Gate-only drill (QWEN_FAST_W2_OFF_AFTER=n, serving_c2_contract.w2_problems refuses it outside a gate profile): after n packed rounds the SERVER writes the
flag file itself (at QWEN_FAST_W2_OFF_PATH, which the drill profile points at a path inside the container) and the ordinary poll finds it, so the card
job exercises the real file path with no operator in the loop (K1 of the engine-reuse pack does the same for parked.off, without a file).

Lines (c2_smoke_check.w2_kill_problems reads them): KILL_LINE once at the latch, ATTACH_LINE once when the attach skipped W2, ROUTED_LINE once per block
the first time a round was sent to the sequential step, DRILL_LINE once when the drill wrote the file."""

import os
import time

SDPA_FLAG = 'QWEN_FAST_TP4_SDPA'
SDPA_MULTI = 'multi'
SDPA_AUDIT_FLAG = 'QWEN_FAST_TP4_SDPA_AUDIT'
SPREAD_FLAG = 'QWEN_FAST_TP4_CONV_GATES_SPREAD'
SPREAD_AUDIT_FLAG = 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'

OFF_FILE = 'w2.off'
OFF_PATH = '/models/.qwen-c2/w2.off'
OFF_PATH_ENV = 'QWEN_FAST_W2_OFF_PATH'
OFF_AFTER_FLAG = 'QWEN_FAST_W2_OFF_AFTER'
PREFIX = 'QWEN_FAST_W2_'
NAMES = (OFF_PATH_ENV, OFF_AFTER_FLAG)
GATE_ONLY = (OFF_AFTER_FLAG,)
POLL_S = 0.25
# The blocks W2 runs on: sixteen-row users (the multi launch needs T16 users, F1 refuses any other segment).
W2_ROWS_PER_USER = 16

KILL_MARKER = '[PINDIAG] w2 kill switch'
KILL_LINE = KILL_MARKER + ' {} present: packed rounds on the W2 blocks take the exact sequential step until the engine restarts (the served SDPA, unspread conv gates)'
ATTACH_LINE = KILL_MARKER + ' {} present at attach: W2 is not attached (served SDPA, unspread conv gates in the packed blocks)'
ATTACH_IGNORED_LINE = KILL_MARKER + ' {} present at attach but ignored: an audit flag is set and an audited lever cannot be skipped'
ROUTED_LINE = '[PINDIAG] w2 kill switch routed the round to the sequential step: block rows={} users={}'
DRILL_LINE = '[PINDIAG] w2 kill switch drill (gate only): wrote {} after {} packed rounds'
REASON = 'w2 kill switch'


def _log(message):
    """One [PINDIAG] line into the server log: loguru where it exists, stderr otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            import sys

            print(message, file=sys.stderr, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:  # noqa: BLE001 - a log line never fails a serve
        pass


def _touch(path):
    with open(path, 'w') as handle:
        handle.write('qwen-c2-w2-drill\n')


def _on(environ, name, wanted='1'):
    value = environ.get(name)
    return value is not None and value.strip() == wanted


def multi_on(environ):
    return (environ.get(SDPA_FLAG) or '').strip() == SDPA_MULTI


def spread_on(environ):
    return _on(environ, SPREAD_FLAG)


def audit_on(environ):
    return (environ.get(SDPA_AUDIT_FLAG) or '0').strip() not in ('', '0') or (environ.get(SPREAD_AUDIT_FLAG) or '0').strip() not in ('', '0')


def off_after(environ):
    """QWEN_FAST_W2_OFF_AFTER: None when unset or empty, else a positive whole number of packed rounds; anything else raises (the contract refuses it first)."""
    value = environ.get(OFF_AFTER_FLAG)
    if value is None or value == '':
        return None
    if not (value.isascii() and value.isdigit() and value == str(int(value)) and int(value) > 0):
        raise ValueError('%s must be a positive whole number of packed rounds, got %r' % (OFF_AFTER_FLAG, value))
    return int(value)


class Switch(object):
    """One process's W2 kill switch. `configured`: the environment names W2 (either lever). `attach_off()` is decided once, at the first question; `killed()` is the
    live latch. Both answer False at once when W2 is not configured."""

    def __init__(self, environ=None, clock=time.monotonic, exists=os.path.exists, log=_log, touch=_touch):
        environ = os.environ if environ is None else environ
        self.path = environ.get(OFF_PATH_ENV, OFF_PATH) or None
        self.configured = multi_on(environ) or spread_on(environ)
        self.audited = audit_on(environ)
        self.after = off_after(environ)
        self.clock, self.exists, self.log, self.touch = clock, exists, log, touch
        self.attach = None         # undecided; True: W2 was not attached; False: it was
        self.latched = False
        self.rounds = 0
        self.drilled = False
        self.routed = set()
        self._polled = None

    # -- the attach -------------------------------------------------------------------------------------------------------------
    def present(self):
        if not self.path:
            return False
        try:
            return bool(self.exists(self.path))
        except Exception:  # noqa: BLE001 - an unreadable path is an absent file
            return False

    def attach_off(self):
        """Whether W2 must NOT be attached in this process: configured, the file present at the first question, and no audit flag. Decided once and kept
        (every block of the process agrees, whatever happens to the file later)."""
        if self.attach is None:
            self.attach = False
            if self.configured and self.present():
                if self.audited:
                    self.log(ATTACH_IGNORED_LINE.format(self.path))
                else:
                    self.attach = True
                    self.log(ATTACH_LINE.format(self.path))
        return self.attach

    # -- the live latch ---------------------------------------------------------------------------------------------------------
    def running(self):
        """W2 is configured and was attached (the attach did not skip it)."""
        return self.configured and not self.attach_off()

    def killed(self):
        """True once the file has been seen (latched for the life of the process). Polls at most every POLL_S; free when W2 is not running."""
        if self.latched:
            return True
        if not self.configured or self.attach is True:
            return False
        if self.attach is None and self.attach_off():
            return False
        now = self.clock()
        if self._polled is not None and now - self._polled < POLL_S:
            return False
        self._polled = now
        if self.present():
            self.latched = True
            self.log(KILL_LINE.format(self.path))
        return self.latched

    def note_round(self):
        """A packed round on a W2 block ran: the drill counts them and, at QWEN_FAST_W2_OFF_AFTER, writes the flag file (once). Free without the drill."""
        if self.after is None or self.drilled or not self.running():
            return
        self.rounds += 1
        if self.rounds >= self.after:
            self.drilled = True
            if not self.path:
                raise ValueError('%s needs %s to name a writable file' % (OFF_AFTER_FLAG, OFF_PATH_ENV))
            self.touch(self.path)
            self.log(DRILL_LINE.format(self.path, self.rounds))

    def note_routed(self, block):
        shape = getattr(block, 'shape', None)
        key = id(block)
        if key in self.routed:
            return
        self.routed.add(key)
        self.log(ROUTED_LINE.format(getattr(shape, 'rows_per_user', '?'), getattr(shape, 'users', '?')))


_switch = None


def switch():
    """The process's switch, built on first use from the environment."""
    global _switch
    if _switch is None:
        _switch = Switch()
    return _switch


def reset(new=None):
    """Drop (or replace) the process switch: tests, and nothing else."""
    global _switch
    _switch = new
    return _switch


def applies(block):
    """Whether W2 runs on this packed block: sixteen-row users (not the octo block's eight)."""
    shape = getattr(block, 'shape', None)
    return getattr(shape, 'rows_per_user', None) == W2_ROWS_PER_USER


def attach_off():
    """sdpa_long_tp.apply and gdn_block_conv_tp.stage ask this before they attach W2: True means leave the served path in place."""
    state = switch()
    return state.configured and state.attach_off()


def routes_sequential(block):
    """serving_packed_step asks this at every round decision for `block`: True when the kill switch has latched and W2 runs on the block, so the round
    goes to the sequential step whole. False (no system call) when W2 is not configured."""
    state = switch()
    if not state.configured:
        return False
    if not applies(block):
        return False
    if not state.killed():
        return False
    state.note_routed(block)
    return True


def note_round(block):
    """serving_packed_step calls this after a packed round on `block` ran (the drill's counter)."""
    state = switch()
    if state.after is not None and applies(block):
        state.note_round()
