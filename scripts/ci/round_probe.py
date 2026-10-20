"""tp4/round-breakdown (module round_probe): the device-side seams of a decode round that the host ledger cannot see (host only; nothing here changes a byte the device sees).

One flag, default off, 0 or 1 (anything else is a configuration error at the first round that reads it):

  QWEN_FAST_TP4_ROUND_PROBE=1        every EVERY-th round of the drafts' coordinator (any live count, any draft shape: the octo pass, the block quads, the single quad,
                                         the pairs and singles) is a PROBE ROUND. A probe round does two TIMED synchronizes and writes ONE line:

                                           [ROUND-PROBE] round=<n> live=<k> shape=<octo|quads2|quads1|quad|pairs|none> pairs=<p> singles=<q> every=<e>
                                                             drain_ms=<f|-> launch_ms=<f> sync_ms=<f|-> total_ms=<f>

                                         drain_ms   the wait for the work the step left in the queue (the fused commits of the publication; the deferred GDN commits go in later,
                                                    after the drafts' readback). The synchronize is the first thing the early draft does, so this is the commits' device time
                                                    still outstanding when the draft begins, a LOWER bound of their device time by the host turnaround (about 1 ms).
                                         launch_ms  the host time from the end of that wait to the end of the draft launches (preparation and enqueue; the draft is already running
                                                    on the device while the later launches are made).
                                         sync_ms    the wait for the launched draft after the launches returned, BEFORE the verify's pre-stage window. The draft's device time lies
                                                    in [sync_ms, launch_ms + sync_ms]; it is the whole of it when the launch is one enqueue (the octo pass, one quad).
                                       The ledger (round_host: QWEN_FAST_TP4_ROUND_HOST_LOG or _LEAN) writes this step's line right after the draft returns, so a reader pairs a probe
                                       line with the next `[PACKED-ROUND-HOST]` line (round_probe_report does).
  QWEN_FAST_TP4_ROUND_PROBE_EVERY=n  the period, an integer from 2 to 4096, default 16 (strict).

WHAT A PROBE ROUND COSTS. The pre-stage window (the host's write of the next verify's inputs, 7 to 12 ms) normally runs UNDER the draft; in a probe round it runs after it. A probe round
is therefore longer than a round by up to min(window, draft) plus the drain: its time is not a round time, and every reader drops it (round_probe_report; the other readers see
no new field, the ledger line is unchanged). The other rounds pay a counter and a modulo, and nothing else: no wrapper, no thread, no allocation. With the flag unset the coordinator
imports nothing from here (the hook is behind an environment test, as hostgap_instr's).

EXACTNESS. A synchronize waits for the queue; it enqueues nothing, reads nothing and reorders nothing on the device (one command queue, in order). The only change a probe round makes is the
host order: the window after the draft instead of under it, which is the order the round takes when no window is armed. Tokens are the target's own greedy decode either way.

A reader: round_probe_report.py (stdlib only). tests: test_round_probe.py.
"""

import os
import time

FLAG = 'QWEN_FAST_TP4_ROUND_PROBE'
EVERY_FLAG = 'QWEN_FAST_TP4_ROUND_PROBE_EVERY'
EVERY_DEFAULT = 16
EVERY_MIN = 2
EVERY_MAX = 4096
MARKER = '[ROUND-PROBE]'
ENGAGED_MARKER = '[PINDIAG] round probe engaged'
SHAPES = ('octo', 'quads2', 'quads1', 'quad', 'pairs', 'none')


def enabled(environ=None):
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % FLAG)
    return value == '1'


def every(environ=None):
    value = (os.environ if environ is None else environ).get(EVERY_FLAG)
    if value is None:
        return EVERY_DEFAULT
    if not value.isdigit() or not EVERY_MIN <= int(value) <= EVERY_MAX:
        raise ValueError('%s must be an integer from %d to %d' % (EVERY_FLAG, EVERY_MIN, EVERY_MAX))
    return int(value)


def log_line(text):
    """One line into the server log: loguru where it exists, stderr otherwise. Never raises."""
    import sys

    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


class Meter:
    """The process's counters: the rounds seen and the probes taken."""

    def __init__(self):
        self.rounds = 0
        self.probes = 0
        self.engaged = False


METER = Meter()


def reset():
    METER.rounds = 0
    METER.probes = 0
    METER.engaged = False


def shape_of(octo=None, block_quads=(), quad=None, groups=()):
    """The draft shape a round took, from the coordinator's own locals: the octo pass, the block quads (one or two), the single quad, pairs, or nothing launched."""
    if octo is not None:
        return 'octo'
    if block_quads:
        return 'quads%d' % min(len(block_quads), 2)
    if quad is not None:
        return 'quad'
    return 'pairs' if groups else 'none'


def begin(entries):
    """The start of the coordinator's prepare, once the live entries are known. Counts the round; on every EVERY-th one does the DRAIN synchronize and returns the probe
    (a dict) that `finish` completes, else None. Raises ValueError for a malformed flag (a bad value fails the first round, not the import)."""
    if not enabled():
        return None
    period = every()
    if not METER.engaged:
        METER.engaged = True
        log_line('%s every=%d' % (ENGAGED_MARKER, period))
    METER.rounds += 1
    if not entries or METER.rounds % period != 0:
        return None
    started = time.perf_counter()
    drain = None
    device = entries[0].get('device')
    operations, mesh = getattr(device, 'operations', None), getattr(device, 'mesh', None)
    if operations is not None:
        operations.synchronize_device(mesh)
        drain = (time.perf_counter() - started) * 1000
    return dict(live=len(entries), drain_ms=drain, launched=time.perf_counter())


def finish(probe, fence, round_number, shape, pairs=0, singles=0):
    """After the draft launches and before the pre-stage window: the SYNC synchronize on the draft's own fence (operations, mesh), then the line. `fence` None (nothing was launched)
    writes the line with sync_ms `-`. The synchronize's own exceptions propagate (the round's fence after it would have raised the same)."""
    launch_ms = (time.perf_counter() - probe['launched']) * 1000
    sync_ms = None
    if fence is not None:
        waited = time.perf_counter()
        fence[0].synchronize_device(fence[1])
        sync_ms = (time.perf_counter() - waited) * 1000
    METER.probes += 1
    drain_ms = probe['drain_ms']
    total = (0.0 if drain_ms is None else drain_ms) + launch_ms + (0.0 if sync_ms is None else sync_ms)
    log_line('%s round=%d live=%d shape=%s pairs=%d singles=%d every=%d drain_ms=%s launch_ms=%.2f sync_ms=%s total_ms=%.2f' % (
        MARKER, round_number, probe['live'], shape, pairs, singles, every(),
        '-' if drain_ms is None else '%.2f' % drain_ms, launch_ms, '-' if sync_ms is None else '%.2f' % sync_ms, total))
    return total
