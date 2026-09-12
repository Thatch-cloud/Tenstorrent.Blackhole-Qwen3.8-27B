"""Fenced trace latency for recurrence-only serial versus device token loops."""

import math
import statistics
import time


def measure(replay, validate, synchronize, *, repeats=10, blocks=3, clock=time.perf_counter_ns):
    if repeats < 1 or blocks < 1:
        raise ValueError('Positive repeat and block counts required')
    for arm in ('serial', 'multitoken'):
        replay(arm)
        synchronize()
        validate(arm)
    samples = []
    paired = []
    for block in range(blocks):
        current = []
        for arm in ('serial', 'multitoken', 'multitoken', 'serial'):
            synchronize()
            start = clock()
            for repeat in range(repeats):
                replay(arm)
            synchronize()
            milliseconds = (clock() - start) / 1e6 / repeats
            if not math.isfinite(milliseconds) or milliseconds <= 0:
                raise ValueError('Expected finite positive latency')
            validate(arm)
            current.append(milliseconds)
            samples.append(dict(block=block, arm=arm, milliseconds=milliseconds, exact=True))
        serial = (current[0] + current[3]) / 2
        candidate = (current[1] + current[2]) / 2
        paired.append(dict(serial_ms=serial, multitoken_ms=candidate, speedup=serial / candidate))
    return dict(samples=samples, paired_blocks=paired, repeats_per_sample=repeats,
                timed_replays=blocks * 4 * repeats, exact=True,
                serial_median_ms=statistics.median(sample['milliseconds'] for sample in samples if sample['arm'] == 'serial'),
                multitoken_median_ms=statistics.median(sample['milliseconds'] for sample in samples if sample['arm'] == 'multitoken'),
                median_paired_speedup=statistics.median(block['speedup'] for block in paired),
                scope='Recurrence-only fenced blocking trace latency; not model or committed-token throughput',
                checkpoint_policy='Both arms export every output and prefix state; immutable initial state',
                excluded='Compilation, capture, input packing/upload, host validation and continuation checks')
