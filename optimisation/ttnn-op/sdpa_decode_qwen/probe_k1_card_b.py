"""S0.K1 probe: time the SERVED decode SDPA (K64g) on the qualification card (K1 design sections 3, 7, 8).

WHAT IT DECIDES. K1 (flag 0x4, not built yet) makes every core of a KV head compute only the 2 Q row tiles
that hold that head's 48 folded rows, instead of all 3. The K/V partition, the tree and the K/V bytes do not
change, so K1 cuts compute, not bytes. What it saves depends on one number nobody has measured: whether the
sliced call stays compute-bound (scenario A, -12.5 to -14.0 ms per replay) or the share leader's serial
read-then-multicast loop binds once compute shrinks (B: -3 to -11 ms; C: ~0 without K1b's read-ahead).

The G4 batch-2 tail+share call (Q (1,2,48,256), mask (2,1,48,cap), flags 0x3) has exactly the per-core work
of the head-sliced G8 call: 2 row tiles x the same chunks, 16 cores per head, 32 leaders reading the same
142.9 MB, one twin each. So its time IS K1a's per-call time, to within the writer's row count. It is
measured here on the binary that serves today, with no build: exactness is not the point (no new kernel).

Shapes (all on the mounted graft; non-causal, k_chunk 256, bf8 paged pool, one page table repeated):
  G8B2_tail_share  the served call and the control that re-anchors card B against card M (flags 0x3,
                   PNHt 3, B 2: 815.8 us at 131,328 keys on card M)
  G4B2_tail_share  the K1a proxy (0x3, PNHt 2, B 2)
  G4B2_tail        the DRAM reference (0x1, PNHt 2, B 2: 64 DRAM readers)
  G4B1_tail        the compute reference (0x1, PNHt 2, B 1: 32 cores; compute-bound on card M)
  G8B2_tail        G8's DRAM-bound reference (0x1, PNHt 3, B 2; in the default set, not in the decision)
  G4B3_tail_share  the share-protocol-bound B=3 shape (only with --shapes)
Capacities 2,304 / 33,024 / 66,048 / 131,328 keys: 1 / 9 / 17 / 33 chunks on each head's busiest core.

Per capacity and shape:
  checksums  one poisoned call, then one legacy call (q_chunk_size 0). Poisoned (design N-S0): a NaN
             tensor of the output's shape is allocated and freed first, so the call's output usually
             takes that address and a row the kernels leave unwritten shows as NaN rather than as a stale
             correct row from an earlier call. Recorded: both sha256s (int16 view), the NaN count, whether
             the address was reused, equality with legacy. A shape card M qualified on K64f
             (G8B2_tail_share, G8B2_tail, G4B1_tail, G4B3_tail_share) that differs from legacy is a
             FAILURE (the mount is not the served build); the never-qualified G4 B2 shapes are WARNINGS,
             and G4B2_tail_share == legacy is the design's spot check, printed in the verdict line (a
             proxy that differs or leaves rows unwritten makes the verdict NO-DECISION: a share protocol
             that is wrong can also be fast, so its time is not K1a's);
  timing     eager: the median of --iters synchronised calls per round, over --rounds rounds that
             interleave the shapes (the rig's CI load drifts); trace: one --trace-calls call trace (one
             user's 16 layers) replayed --replays times, per call. A replay's last output is compared
             with the eager checksum.
Slopes, per shape and basis, in us per busiest-core chunk: two-point 1->9, 9->17, 17->33 and 9->33 (with
its intercept, the design's figures), and a least-squares fit over the multi-chunk points (>= 9 chunks)
with its intercept and worst relative residual. `rise` = fit slope / (1->9 slope) - 1 is the design's
linearity reading: every shape card M timed is flat to +-2%, except G8 share, which rises 11%.

Decision (section 3's table; thresholds relative to card B's OWN controls, at 131,328 keys):
  A  proxy <= 1.08 x G4B1_tail and its fit slope within 5% of G4B1_tail's -> GO K1a: enable 0x4
  C  proxy >= 0.93 x G8B2_tail_share -> GO, K1b-led: enable 0x4 and 0x8, and the card timing also runs 0xB
  B  between -> GO K1a + K1b: enable 0x4 and 0x8
  NO-DECISION: a failure (below), no timing, 131,328 or a control missing, controls that do not separate A
  from C (1.08 x G4B1_tail >= 0.93 x served) or inverted (G4B1_tail not faster than the served call), or a
  proxy that left output rows unwritten (NaN from the poison: it did not do K1a's work) or whose bytes
  differ from legacy (the spot check).
  K1b is built dormant in every case (section 8); the decision is which flags the arm enables. Also
  reported: K1a ALONE against section 7's candidate rules applied to the proxy (<= 0.78 x served meets
  the acceptance; > 0.90 x served is the stop rule), the saving, (served - proxy) x 64 calls per replay
  (eager: on the card basis and scaled by 761/815.8 to the trace basis, as section 3 does, plus the
  measured trace figure; trace: the measured figure, never scaled again), and section 3's 4 x 32k
  cross-check at 33,024 keys (recorded, not decided on).
The decision is taken on --basis (eager by default, the basis of every card-M figure the design quotes);
the other basis's scenario is printed beside it.

Failures (passed=False, verdict NO-DECISION): the loaded _ttnncpp.so is not a stage-3 (KV share) build, or
not --expect-binary-sha256; the mounted qwen kernels are not the served 280a847f / 8776fcc7 (--kernel-root);
the compact tree scratch is off (QWEN_SDPA_TREE_SCRATCH_ROUNDS=1, as the arm sets it); a requested qwen
program has no '[QWEN-SDPA] flags=' factory line, or a wrong kv_share / PNHt / cb_bytes (graft mounted is not
graft executed); a non-finite legacy output; a qualified shape that differs from legacy; the watchdog.

--watchdog S arms a per-device-call deadline (timed loops run under one deadline for the whole loop): a call
that has not returned prints WATCHDOG, writes the partial report and os._exit(3)s; a faulthandler backstop
(a C thread) dumps the stacks ('Timeout (') and exits 1 when a blocking call holds the GIL.

RUN with run_probe_k1.sh only (QUAL_CARD, default card B; the serving pair is refused without
ALLOW_SERVING_CARD=1), in the gate's image with the graft mounted exactly as the arm mounts it and a fresh
kernel cache. The helpers above Watchdog import no ttnn and are tested on CPU by test_probe_k1.py.
"""

import argparse
from contextlib import contextmanager
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_sdpa_decode_qwen_card_m as card  # noqa: E402 - the ttnn-free helpers: masks, fold, Case, digest

PROBE = 'S0.K1'
CAPACITIES = (2304, 33024, 66048, 131328)
DECISION_CAPACITY = 131328
CROSS_CHECK_CAPACITY = 33024  # section 3's 4 x 32k cross-check (recorded, not decided on)
CORES_PER_HEAD = 16          # max_cores_per_head_batch; every probe shape (B <= 3, 2 KV heads, 110 cores) gets 16
CALLS_PER_REPLAY = 64        # section 3: 64 calls x 761 us = the roofline's 48.7 ms per replay
TRACE_BASIS = 761.0 / 815.8  # section 3: the trace-inferred per-call time over card M's served call
# name -> (rows per fold group, batch B, sentinel, group offsets (the mask formula's), role)
SHAPES = {
    'G8B2_tail_share': (8, 2, card.TAIL_SHARE, (0, 8), 'served'),
    'G4B2_tail_share': (4, 2, card.TAIL_SHARE, (0, 4), 'proxy'),
    'G4B2_tail': (4, 2, card.TAIL, (0, 4), 'dram'),
    'G4B1_tail': (4, 1, card.TAIL, (12,), 'compute'),
    'G8B2_tail': (8, 2, card.TAIL, (0, 8), 'reference'),
    'G4B3_tail_share': (4, 3, card.TAIL_SHARE, (0, 4, 8), 'reference'),
}
SERVED, PROXY, DRAM, COMPUTE = 'G8B2_tail_share', 'G4B2_tail_share', 'G4B2_tail', 'G4B1_tail'
REQUIRED = (SERVED, PROXY, DRAM, COMPUTE)
DEFAULT_SHAPES = REQUIRED + ('G8B2_tail',)
QUALIFIED = ('G8B2_tail_share', 'G8B2_tail', 'G4B1_tail', 'G4B3_tail_share')   # card M, K64f, == legacy
A_TIME, A_SLOPE, C_TIME = 1.08, 0.05, 0.93   # section 3's table
ACCEPT, STOP = 0.78, 0.90                    # section 7's candidate acceptance and stop rule
# Card M, K64f binary (the decode side equals K64g's), eager median of 20: section 1's tables. Reference only.
CARD_M_US = {
    'G8B2_tail_share': {2304: 151.2, 33024: 304.2, 131328: 815.8},
    'G8B2_tail': {2304: 127.1, 33024: 315.7, 131328: 876.3},
    'G4B1_tail': {2304: 89.3, 33024: 209.7, 131328: 579.0},
    'G4B3_tail_share': {2304: 121.2, 33024: 364.1, 131328: 1098.8},
}
K64G_TTNNCPP_SHA256 = '134bc8347d6533b3c0a5d2a1efaf14261e7f554f90579d179b1f1b88e20944f4'
SERVED_KERNELS = {
    'dataflow/reader_decode_qwen.cpp': '280a847fae833891dffff1057d67b999288a183386cd058e3e1614755ce3499b',
    'compute/sdpa_flash_decode_qwen.cpp': '8776fcc7420c6f27a9c7ae06c54c391225a00ce78322c5397970d74a5063ca8a',
}
KERNEL_ROOT = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels'
OPEN_EXTRA_S = 600.0         # the first open JITs the firmware into the fresh kernel cache
ENV_RECORDED = (card.SCRATCH_ENV, 'TT_METAL_WATCHER', 'TT_METAL_CACHE', 'TT_METAL_HOME')

clock = time.perf_counter    # module level so the CPU dry run can drive a model clock


# ---------------------------------------------------------------------------------------------
# Pure helpers (no ttnn).
# ---------------------------------------------------------------------------------------------

def busiest_chunks(capacity, cores_per_head=None):
    """K/V chunks (k_chunk 256) on the busiest core of a head: 1, 9, 17 and 33 at the four capacities
    (get_workload_for_core: the chunks split over 16 cores, the remainder one each)."""
    card.num_blocks(capacity)
    return -(-(capacity // card.K_CHUNK) // (CORES_PER_HEAD if cores_per_head is None else cores_per_head))


def shape_geometry(name):
    """What the factory sees for a shape: B, folded rows, PNHt and the active cores."""
    rows, batches, sentinel, offsets, role = SHAPES[name]
    folded = rows * 12
    return dict(rows=rows, B=batches, folded_rows=folded, PNHt=-(-folded // card.TILE),
                flags=sentinel & 0xFF, cores=batches * card.KV_HEADS * CORES_PER_HEAD, role=role,
                offsets=list(offsets))


def summary(samples):
    if not samples:
        return None
    ordered = sorted(samples)
    return dict(median_us=statistics.median(ordered), min_us=ordered[0], mean_us=statistics.mean(ordered),
                p90_us=ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))],
                stdev_us=statistics.stdev(ordered) if len(ordered) > 1 else 0.0, n=len(ordered))


def slopes(times):
    """times: {capacity: us per call}. The per-chunk slopes, the 9->33 intercept, a least-squares fit over
    the multi-chunk points and the linearity readings (section 1's tables are the two-point forms)."""
    points = sorted((busiest_chunks(capacity), float(us)) for capacity, us in times.items() if us is not None)
    by = dict(points)
    out = dict(points=[[chunks, us] for chunks, us in points])
    for low, high in ((1, 9), (9, 17), (17, 33), (9, 33)):
        if low in by and high in by:
            out['slope_%d_%d' % (low, high)] = (by[high] - by[low]) / (high - low)
    if 'slope_9_33' in out:
        out['intercept_9_33'] = by[9] - 9 * out['slope_9_33']
    multi = [(chunks, us) for chunks, us in points if chunks >= 9]
    if len(multi) >= 2:
        mean_x = sum(x for x, _ in multi) / len(multi)
        mean_y = sum(y for _, y in multi) / len(multi)
        sxx = sum((x - mean_x) ** 2 for x, _ in multi)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in multi) / sxx
        intercept = mean_y - slope * mean_x
        out.update(fit_slope=slope, fit_intercept=intercept, fit_points=len(multi),
                   fit_max_residual=max(abs(y - intercept - slope * x) / y for x, y in multi))
    if 'fit_slope' in out and out.get('slope_1_9'):
        out['rise'] = out['fit_slope'] / out['slope_1_9'] - 1
    if out.get('slope_9_17') and 'slope_17_33' in out:
        out['curvature'] = out['slope_17_33'] / out['slope_9_17'] - 1
    return out


def k1a_alone(proxy, served):
    """Section 7's candidate rules (0x7 <= 0.78 x 0x3 passes; > 0.90 x 0x3 stops K1) on the proxy."""
    ratio = proxy / served
    if ratio <= ACCEPT:
        return 'meets-acceptance'
    if ratio > STOP:
        return 'stop-rule'
    return 'between'


PLANS = {
    'A': ('K1a', '0x4', 'K1b stays dormant (K64i carries it regardless, section 8); central -13 ms per replay'),
    'B': ('K1a+K1b', '0x4,0x8', 'the share leader binds once compute shrinks: K1b\'s read-ahead is needed'),
    'C': ('K1b-led', '0x4,0x8', 'K1a alone is ~0: the card timing also runs 0xB (read-ahead without slice), '
                                'which then pays at 3 tiles on its own'),
}


def decide(times, fits, invalid=(), basis='eager'):
    """Section 3's table on one basis. times: {shape: {capacity: us}}; fits: {shape: slopes(...)}.

    The saving per replay is (served - proxy) x 64 calls. On the eager basis that is the design's card
    basis, and its trace figure is the design's scaling by 761/815.8; on the trace basis the per-call times
    are already trace-measured, so the figure is the measured one, never scaled again."""
    decision = dict(scenario=None, verdict='NO-DECISION', plan=None, enable=None, note=None, reasons=list(invalid))
    capacity = DECISION_CAPACITY
    missing = [name for name in (PROXY, SERVED, COMPUTE) if capacity not in times.get(name, {})]
    if missing:
        decision['reasons'].append('not measured at %d keys: %s' % (capacity, ', '.join(missing)))
        return decision
    proxy, served, compute = times[PROXY][capacity], times[SERVED][capacity], times[COMPUTE][capacity]
    proxy_slope = fits.get(PROXY, {}).get('fit_slope')
    compute_slope = fits.get(COMPUTE, {}).get('fit_slope')
    per_replay = (served - proxy) * CALLS_PER_REPLAY / 1000.0
    numbers = dict(capacity=capacity, proxy_us=proxy, served_us=served, compute_us=compute,
                   proxy_over_compute=proxy / compute, proxy_over_served=proxy / served,
                   a_limit_us=A_TIME * compute, c_limit_us=C_TIME * served,
                   proxy_slope=proxy_slope, compute_slope=compute_slope,
                   proxy_rise=fits.get(PROXY, {}).get('rise'), served_rise=fits.get(SERVED, {}).get('rise'),
                   saving_us_per_call=served - proxy,
                   saving_ms_per_replay_card=per_replay if basis == 'eager' else None,
                   saving_ms_per_replay_trace=per_replay * TRACE_BASIS if basis == 'eager' else per_replay,
                   saving_trace_measured=basis == 'trace',
                   k1a_alone=k1a_alone(proxy, served))
    if DRAM in times and capacity in times[DRAM]:
        numbers['proxy_over_dram'] = proxy / times[DRAM][capacity]
    # Section 3: "at 4 x 32k: 304.2 -> 228-251 us per call, -3.4 to -4.9 ms: the cross-check for the audited
    # 32k arm". Recorded, not decided on.
    small = CROSS_CHECK_CAPACITY
    if small in times.get(PROXY, {}) and small in times.get(SERVED, {}):
        low_proxy, low_served = times[PROXY][small], times[SERVED][small]
        numbers['cross_check'] = dict(capacity=small, proxy_us=low_proxy, served_us=low_served,
                                      proxy_over_served=low_proxy / low_served,
                                      saving_ms_per_replay=(low_served - low_proxy) * CALLS_PER_REPLAY / 1000.0)
    if proxy_slope is not None and compute_slope:
        numbers['slope_delta'] = proxy_slope / compute_slope - 1
    decision.update(numbers)
    if not compute < served:
        decision['reasons'].append('controls inverted: the compute reference %s (%.1f us) is not faster than the '
                                   'served call %s (%.1f us)' % (COMPUTE, compute, SERVED, served))
    elif A_TIME * compute >= C_TIME * served:
        decision['reasons'].append('controls do not separate A from C: %.2f x %s = %.1f us >= %.2f x %s = %.1f us'
                                   % (A_TIME, COMPUTE, A_TIME * compute, C_TIME, SERVED, C_TIME * served))
    if proxy_slope is None or not compute_slope:
        decision['reasons'].append('no fit slope for %s and %s (two capacities of >= 9 chunks are needed)'
                                   % (PROXY, COMPUTE))
    if decision['reasons']:
        return decision
    if proxy <= A_TIME * compute and abs(proxy_slope / compute_slope - 1) <= A_SLOPE:
        scenario = 'A'
    elif proxy >= C_TIME * served:
        scenario = 'C'
    else:
        scenario = 'B'
    plan, enable, note = PLANS[scenario]
    decision.update(scenario=scenario, verdict='GO', plan=plan, enable=enable, note=note)
    return decision


def spot_check(checksums, capacity=None):
    """The design's spot check on the proxy's checksum entries: 'equal', 'equal-unpoisoned(k/n)' (equal, but k
    of the n outputs did not take the poisoned address, so an unwritten row could hide behind a stale
    correct one), 'DIFFERS(n)', 'UNWRITTEN(n)' or 'none'."""
    entries = [entry for entry in checksums if entry['shape'] == PROXY and (capacity is None or entry['capacity'] == capacity)]
    if not entries:
        return 'none'
    unwritten = sum(entry.get('nan_elements', 0) for entry in entries)
    if unwritten:
        return 'UNWRITTEN(%d)' % unwritten
    differing = sum(entry.get('differing_from_legacy', 0) for entry in entries)
    if differing:
        return 'DIFFERS(%d)' % differing
    unpoisoned = sum(1 for entry in entries if entry.get('poison_address_reused') is not True)
    return 'equal-unpoisoned(%d/%d)' % (unpoisoned, len(entries)) if unpoisoned else 'equal'


def verdict_line(decision, basis, other=None, spot='none'):
    """The one go/no-go line: the scenario, the flags to enable, the numbers behind it."""
    words = ['K1_PROBE', 'verdict=%s' % decision['verdict']]
    if decision['scenario']:
        words += ['scenario=%s' % decision['scenario'], 'plan=%s' % decision['plan'], 'enable=%s' % decision['enable']]
    words.append('basis=%s' % basis)
    if 'proxy_us' in decision:
        words += ['cap=%d' % decision['capacity'], 'proxy=%.1fus' % decision['proxy_us'],
                  'compute=%.1fus' % decision['compute_us'], 'served=%.1fus' % decision['served_us'],
                  'proxy/compute=%.3f(A<=%.2f)' % (decision['proxy_over_compute'], A_TIME),
                  'proxy/served=%.3f(C>=%.2f)' % (decision['proxy_over_served'], C_TIME)]
        if decision.get('slope_delta') is not None:
            words.append('slope=%.2f/%.2fus/chunk(%+.1f%%,A<=%d%%)' % (decision['proxy_slope'], decision['compute_slope'],
                                                                      100 * decision['slope_delta'], 100 * A_SLOPE))
        if decision.get('saving_trace_measured'):
            replay = 'replay=%.1fms(trace)' % -decision['saving_ms_per_replay_trace']
        else:
            replay = 'replay=%.1fms(card)/%.1fms(trace-scaled)' % (-decision['saving_ms_per_replay_card'],
                                                                   -decision['saving_ms_per_replay_trace'])
            if other is not None and other.get('saving_trace_measured') and 'saving_ms_per_replay_trace' in other:
                replay += '/%.1fms(trace)' % -other['saving_ms_per_replay_trace']
        words += ['saving=%.1fus/call' % decision['saving_us_per_call'], replay,
                  'k1a_alone=%s' % decision['k1a_alone']]
    words.append('spot_check=%s' % spot)
    if other is not None:
        words.append('other_basis=%s' % (other['scenario'] or other['verdict']))
    if decision['reasons']:
        words.append('reasons=%s' % json.dumps(decision['reasons']))
    return ' '.join(words)


def analyse(report, basis, invalid):
    """Slopes and decisions on both bases from report['timing']; the verdict line on `basis`. A proxy that
    left rows unwritten (the poison shows NaN) did not do K1a's work, and one whose bytes differ from legacy
    ran the share protocol wrongly (the same code path is bit-exact on every qualified shape; a twin that
    did not wait for its leader is both wrong and fast), so in either case its time decides nothing. The
    numbers are still reported."""
    report['spot_check'] = spot_check(report['checksums'])
    if report['spot_check'].startswith('UNWRITTEN'):
        invalid = list(invalid) + ['the proxy %s left output rows unwritten (%s): its time is not K1a\'s'
                                   % (PROXY, report['spot_check'])]
    elif report['spot_check'].startswith('DIFFERS'):
        invalid = list(invalid) + ['the proxy %s differs from legacy (%s): B=2 share on G4 is not exact, so its '
                                   'time is not trusted as K1a\'s' % (PROXY, report['spot_check'])]
    per_basis = {}
    for name in ('eager', 'trace'):
        times = {}
        for row in report['timing']:
            stats = row.get(name)
            if stats and stats.get('median_us') is not None:
                times.setdefault(row['shape'], {})[row['capacity']] = stats['median_us']
        fits = {shape: slopes(values) for shape, values in times.items()}
        reasons = list(invalid) or ([] if times else ['no %s timing' % name])
        per_basis[name] = dict(times=times, slopes=fits, decision=decide(times, fits, reasons, basis=name))
    report['slopes'] = {shape: {name: per_basis[name]['slopes'].get(shape) for name in per_basis}
                        for shape in sorted({shape for name in per_basis for shape in per_basis[name]['slopes']})}
    report['decision'] = {name: per_basis[name]['decision'] for name in per_basis}
    other = 'trace' if basis == 'eager' else 'eager'
    chosen, alternative = report['decision'][basis], report['decision'][other]
    if chosen['scenario'] and alternative['scenario'] and chosen['scenario'] != alternative['scenario']:
        report['warnings'].append('the bases disagree: %s says %s, %s says %s' % (basis, chosen['scenario'], other,
                                                                               alternative['scenario']))
    report['card_b_over_card_m'] = {
        shape: {capacity: per_basis['eager']['times'][shape][capacity] / reference
                for capacity, reference in CARD_M_US[shape].items() if capacity in per_basis['eager']['times'].get(shape, {})}
        for shape in CARD_M_US if shape in per_basis['eager']['times']}
    report['verdict_line'] = verdict_line(chosen, basis, alternative if alternative['scenario'] or 'proxy_us' in alternative
                                          else None, report['spot_check'])
    return report['verdict_line']


def slope_lines(report):
    """One log line per shape and basis: the fit, its intercept, the rise and the card-B / card-M ratio."""
    out = []
    for shape, bases in sorted(report.get('slopes', {}).items()):
        for basis, fit in sorted(bases.items()):
            if not fit or 'fit_slope' not in fit:
                continue
            words = ['slope', shape, basis, 'fit=%.2fus/chunk' % fit['fit_slope'], 'intercept=%.1fus' % fit['fit_intercept'],
                     'residual=%.1f%%' % (100 * fit['fit_max_residual'])]
            if 'slope_1_9' in fit:
                words.append('1->9=%.2f' % fit['slope_1_9'])
            if 'rise' in fit:
                words.append('rise=%+.1f%%' % (100 * fit['rise']))
            ratios = report.get('card_b_over_card_m', {}).get(shape) if basis == 'eager' else None
            if ratios:
                words.append('cardB/cardM=%s' % ','.join('%d:%.3f' % item for item in sorted(ratios.items())))
            out.append(' '.join(words))
    return out


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_kernels(root, report):
    """The served qwen kernels are the ones the JIT will build from the mounted op directory."""
    found = {}
    for name, expected in SERVED_KERNELS.items():
        path = Path(root) / name
        found[name] = file_sha256(path) if path.is_file() else None
        if found[name] != expected:
            report['failures'].append('%s is %s, not the served %s (the op directory is not K64g\'s)'
                                      % (path, (found[name] or 'missing')[:16], expected[:16]))
    report['kernels'] = dict(root=str(root), sha256=found)
    return not any(found[name] != expected for name, expected in SERVED_KERNELS.items())


# ---------------------------------------------------------------------------------------------
# Device harness.
# ---------------------------------------------------------------------------------------------

class Watchdog:
    """A per-device-call deadline: a hung NoC handshake cannot be interrupted from Python, so the poller
    prints WATCHDOG, writes the partial report and os._exit(3)s.

    The poll thread needs the GIL, and a blocking ttnn call (a read or a synchronize on a hung program) may
    hold it. So each op also arms a faulthandler backstop, a C thread: at the op's budget + `grace` it dumps
    every thread's stack ('Timeout (h:mm:ss)!') and exits 1, which run_probe_k1.sh reads as a hang. Leaving
    a nested op re-arms the backstop for the outer op's remaining time; leaving the outermost cancels it.

    Arming restarts faulthandler's thread (~75 us an arm and cancel), which would be charged to every launch
    of a timed loop, so a timed loop runs inside one span: one deadline for the whole loop, and the per-call
    ops inside it (Case.call's, Case.host's) cost nothing. card.Case uses this object through card.WATCHDOG."""

    def __init__(self, seconds, on_fire=None, grace=60.0, backstop=True, exit=os._exit):
        self.seconds, self.on_fire, self.grace, self.exit = seconds, on_fire, grace, exit
        self.backstop = bool(backstop and seconds)
        self.label, self.deadline = None, None
        self.coarse = False
        self.lock = threading.Lock()

    def start(self):
        if self.seconds:
            threading.Thread(target=self.poll, name='k1-probe-watchdog', daemon=True).start()
        return self

    def arm(self, seconds):
        if self.backstop:
            try:
                faulthandler.dump_traceback_later(seconds + self.grace, exit=True, file=sys.stdout)
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    def cancel(self):
        if self.backstop:
            try:
                faulthandler.cancel_dump_traceback_later()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    @contextmanager
    def span(self, label, seconds):
        """One deadline of `seconds` (at least the per-call one) over a timed loop."""
        if not self.seconds or self.coarse:
            yield
            return
        with self.op(label, extra=max(0.0, seconds - self.seconds)):
            self.coarse = True
            try:
                yield
            finally:
                self.coarse = False

    @contextmanager
    def op(self, label, extra=0.0):
        if not self.seconds or self.coarse:
            yield
            return
        budget = self.seconds + extra
        with self.lock:
            outer = (self.label, self.deadline)
            self.label, self.deadline = label, time.monotonic() + budget
        self.arm(budget)
        try:
            yield
        finally:
            with self.lock:
                self.label, self.deadline = outer
                remaining = None if outer[1] is None else max(outer[1] - time.monotonic(), 0.0)
            if remaining is None:
                self.cancel()
            else:
                self.arm(remaining)

    def check(self):
        """One poll: fire if the armed call is past its deadline. Returns whether it fired."""
        with self.lock:
            label, deadline = self.label, self.deadline
        if label is None or time.monotonic() < deadline:
            return False
        sys.stdout.write('WATCHDOG: %r did not return within its budget (%ss per call); exiting 3 (docker rm -f, '
                         'then reset this card only, by the runner\'s printed reset command)\n' % (label, self.seconds))
        sys.stdout.flush()
        try:
            if self.on_fire is not None:
                self.on_fire(label)
        finally:
            self.exit(3)
        return True

    def poll(self):
        while not self.check():
            time.sleep(1.0)


WATCHDOG = Watchdog(0)


def buffer_address(tensor):
    try:
        return int(tensor.buffer_address())
    except Exception:  # noqa: BLE001 - a tensor without one: the poison is then unverified, and says so
        return None


def poison(ttnn, torch, case, shape):
    """N-S0: upload a NaN (0x7FC0) tensor of the call's output shape, note its address and free it."""
    tensor = case.upload(torch.full(shape, float('nan'), dtype=torch.bfloat16), keep=False)
    try:
        return buffer_address(tensor)
    finally:
        ttnn.deallocate(tensor)


def checksum(ttnn, torch, case, capacity, name, inputs, report, eager_shas):
    query, mask, sentinel, batches, rows = inputs[name]
    entry = dict(capacity=capacity, shape=name, sentinel='0x%x' % sentinel, chunks=busiest_chunks(capacity))
    address = poison(ttnn, torch, case, (1, batches, rows * 12, card.HEAD_DIM))
    out = case.call(query, mask, sentinel)
    entry['poison_address_reused'] = None if address is None else buffer_address(out) == address
    host = case.host(out)
    legacy = case.host(case.call(query, mask, card.LEGACY))
    values, legacy_values = host.float(), legacy.float()
    entry.update(sha256=card.digest(torch, host), legacy_sha256=card.digest(torch, legacy),
                 nan_elements=int(torch.isnan(values).sum()), nonfinite_elements=int((~torch.isfinite(values)).sum()),
                 differing_from_legacy=card.differing(torch, host, legacy))
    entry['equal_legacy'] = entry['differing_from_legacy'] == 0
    eager_shas[(capacity, name)] = entry['sha256']
    report['checksums'].append(entry)
    label = 'cap%d/%s' % (capacity, name)
    if not bool(torch.isfinite(legacy_values).all()):
        report['failures'].append('%s: the legacy output is not finite (the inputs are broken)' % label)
    if entry['nan_elements']:
        report['warnings'].append('%s: %d output elements are NaN - rows the kernels left unwritten (the poison, '
                                  'N-S0)' % (label, entry['nan_elements']))
    if not entry['equal_legacy']:
        text = '%s: %s differs from legacy in %d elements' % (label, entry['sentinel'], entry['differing_from_legacy'])
        if name in QUALIFIED:
            report['failures'].append(text + ' (card M qualified this shape on K64f: the mount is not the served build)')
        else:
            report['warnings'].append(text + (' (the design\'s spot check; B=2 share on G4 was never qualified: '
                                              'the verdict is NO-DECISION)' if name == PROXY else ' (never qualified)'))
    if entry['poison_address_reused'] is False:
        report['warnings'].append('%s: the output did not take the poisoned address; its equality with legacy may be '
                                  'a stale earlier output' % label)
    print('checksum cap=%d %-16s %s sha %s legacy %s equal=%s nan=%d poisoned=%s' % (
        capacity, name, entry['sentinel'], entry['sha256'][:12], entry['legacy_sha256'][:12], entry['equal_legacy'],
        entry['nan_elements'], entry['poison_address_reused']), flush=True)


def eager_samples(ttnn, device, once, args, label):
    samples = []
    with WATCHDOG.span(label, args.watchdog):
        for _ in range(args.warmup):
            ttnn.deallocate(once())
        ttnn.synchronize_device(device)
        for _ in range(args.iters):
            started = clock()
            out = once()
            ttnn.synchronize_device(device)
            samples.append((clock() - started) * 1e6)
            ttnn.deallocate(out)
    return samples


def traced(ttnn, torch, device, case, query, mask, sentinel, args, label):
    """One trace of --trace-calls identical calls (one user's 16 layers), replayed; per call. Also whether
    the last replay's output is the eager bytes."""
    with WATCHDOG.op('capture %s' % label):
        trace, outputs = card.capture(ttnn, device, lambda: [case.call(query, mask, sentinel, record=False)
                                                             for _ in range(args.trace_calls)])
    samples = []
    try:
        with WATCHDOG.span('replay %s' % label, args.watchdog):
            for index in range(args.trace_warmup + args.replays):
                started = clock()
                ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                if index >= args.trace_warmup:
                    samples.append((clock() - started) * 1e6 / args.trace_calls)
        with WATCHDOG.op('trace read back %s' % label):
            last = ttnn.to_torch(outputs[-1])
    finally:
        with WATCHDOG.op('release trace %s' % label):
            ttnn.release_trace(device, trace)
        for output in outputs:
            ttnn.deallocate(output)
    return summary(samples), card.digest(torch, last)


def timing_capacity(ttnn, torch, device, case, capacity, inputs, args, report, eager_shas):
    shapes = list(args.shapes)
    rounds = {name: [] for name in shapes}
    samples = {name: [] for name in shapes}
    for index in range(args.rounds):
        turn = index % len(shapes)
        for name in shapes[turn:] + shapes[:turn]:          # no shape always runs first
            query, mask, sentinel, _batches, _rows = inputs[name]
            got = eager_samples(ttnn, device, lambda: case.call(query, mask, sentinel), args,
                                'timing cap=%d %s round %d' % (capacity, name, index))
            rounds[name].append(statistics.median(got))
            samples[name].extend(got)
    for name in shapes:
        query, mask, sentinel, _batches, _rows = inputs[name]
        row = dict(capacity=capacity, shape=name, chunks=busiest_chunks(capacity), sentinel='0x%x' % sentinel,
                   eager=summary(samples[name]), eager_round_medians=rounds[name], trace=None)
        spread = (max(rounds[name]) - min(rounds[name])) / min(rounds[name]) if rounds[name] else 0.0
        row['eager_round_spread'] = spread
        if len(rounds[name]) > 1 and spread > 0.05:
            report['warnings'].append('cap%d/%s: the eager round medians spread %.1f%% (%s): the host was loaded'
                                      % (capacity, name, 100 * spread, ', '.join('%.1f' % v for v in rounds[name])))
        if args.replays:
            row['trace'], sha = traced(ttnn, torch, device, case, query, mask, sentinel, args,
                                       'cap=%d %s' % (capacity, name))
            row['trace_output_sha256'] = sha
            row['trace_output_matches_eager'] = eager_shas.get((capacity, name)) in (None, sha)
            if not row['trace_output_matches_eager']:
                report['warnings'].append('cap%d/%s: the trace replay output differs from the eager call' % (capacity, name))
        reference = CARD_M_US.get(name, {}).get(capacity)
        row['card_m_us'] = reference
        report['timing'].append(row)
        print('timing cap=%d %-16s eager %.1f us%s%s' % (
            capacity, name, row['eager']['median_us'],
            '' if row['trace'] is None else ' trace %.1f us/call' % row['trace']['median_us'],
            '' if reference is None else ' (card M %.1f)' % reference), flush=True)


def check_binary(args, report):
    path, markers = card.loaded_binary()
    stage = card.binary_stage(markers)
    sha = file_sha256(path)
    report['binary'] = dict(path=path, sha256=sha, markers=markers, stage=stage,
                            expected_sha256=args.expect_binary_sha256 or None)
    print('binary %s sha256 %s stage=%d markers=%s' % (path, sha[:16], stage, markers), flush=True)
    if stage != 3:
        report['failures'].append('the loaded _ttnncpp.so is stage %d, not a stage-3 (KV share) build: the graft is not '
                                  'mounted, or is not K64f-or-later' % stage)
        return False
    if args.expect_binary_sha256 and sha != args.expect_binary_sha256:
        report['failures'].append('the loaded _ttnncpp.so is %s, not the expected %s (read the launched argv)'
                                  % (sha[:16], args.expect_binary_sha256[:16]))
        return False
    return True


def run(args, report):
    import torch
    import ttnn

    failures = report['failures']
    options = dict(device_id=args.device_id, l1_small_size=24576)
    if not args.no_timing and args.replays:
        options['trace_region_size'] = args.trace_region_bytes
    with WATCHDOG.op('open device', extra=OPEN_EXTRA_S):
        device = ttnn.open_device(**options)
    try:
        try:
            device.enable_program_cache()
            report['program_cache_enabled_call'] = True
        except Exception as error:  # noqa: BLE001 - default-on in newer runtimes
            report['program_cache_enabled_call'] = repr(error)[:200]
        if not check_binary(args, report):
            return
        if args.kernel_root and not check_kernels(args.kernel_root, report):
            return
        if os.environ.get(card.SCRATCH_ENV) != '1':
            failures.append('%s=1 is required: the arm sets it, and the G8 legacy calls do not fit L1 without it'
                            % card.SCRATCH_ENV)
            return
        requested = set()
        report['_requested'] = requested
        eager_shas = {}
        for capacity in args.capacities:
            case = card.Case(ttnn, torch, device, capacity, args.seed, requested=requested)
            try:
                begin = capacity - 256 + args.start
                inputs = {}
                for name in args.shapes:
                    rows, batches, sentinel, offsets, _role = SHAPES[name]
                    query = case.upload(card.build_query(torch, batches, args.seed, 'normal', rows=rows))
                    mask = case.upload(card.build_mask(torch, capacity, begin, offsets, rows=rows))
                    inputs[name] = (query, mask, sentinel, batches, rows)
                for name in args.shapes:
                    checksum(ttnn, torch, case, capacity, name, inputs, report, eager_shas)
                if not args.no_timing:
                    timing_capacity(ttnn, torch, device, case, capacity, inputs, args, report, eager_shas)
            finally:
                case.close()
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--capacities', default=','.join(map(str, CAPACITIES)))
    parser.add_argument('--shapes', default=','.join(DEFAULT_SHAPES), help='from %s' % ', '.join(SHAPES))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start', type=int, default=0, help='block start = capacity - 256 + this')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20, help='synchronised eager calls per shape per round')
    parser.add_argument('--rounds', type=int, default=2, help='eager rounds, interleaving the shapes')
    parser.add_argument('--trace-calls', type=int, default=16, help='calls per trace (one user\'s 16 layers)')
    parser.add_argument('--trace-warmup', type=int, default=2)
    parser.add_argument('--replays', type=int, default=10, help='timed trace replays per shape (0: no trace)')
    parser.add_argument('--trace-region-bytes', type=int, default=16 << 20)
    parser.add_argument('--basis', choices=('eager', 'trace'), default='eager', help='the decision\'s basis')
    parser.add_argument('--no-timing', action='store_true', help='checksums only (the watcher pass)')
    parser.add_argument('--watchdog', type=float, default=0, help='seconds per device call before os._exit(3); 0 off')
    parser.add_argument('--expect-binary-sha256', default='', help='the mapped _ttnncpp.so must be this (K64g: %s)'
                        % K64G_TTNNCPP_SHA256[:16])
    parser.add_argument('--kernel-root', default=KERNEL_ROOT, help='the mounted sdpa_decode kernels ("" skips)')
    args = parser.parse_args(argv)
    args.capacities = [int(value) for value in args.capacities.split(',') if value]
    args.shapes = [value for value in args.shapes.split(',') if value]
    unknown = [name for name in args.shapes if name not in SHAPES]
    if unknown or not args.shapes or len(set(args.shapes)) != len(args.shapes):
        parser.error('unknown or repeated shapes %r (known: %s)' % (unknown or args.shapes, ', '.join(SHAPES)))
    try:
        chunks = [busiest_chunks(capacity) for capacity in args.capacities]
    except ValueError as error:
        parser.error(str(error))
    if not args.capacities or len(set(chunks)) != len(chunks):
        parser.error('the capacities need distinct busiest-core chunk counts, got %r' % dict(zip(args.capacities, chunks)))
    if not 0 <= args.start <= 240:
        parser.error('--start must be 0..240')
    if min(args.iters, args.rounds, args.trace_calls) < 1 or min(args.warmup, args.trace_warmup, args.replays) < 0:
        parser.error('--iters, --rounds and --trace-calls must be >= 1; warmups and --replays >= 0')
    if args.expect_binary_sha256 and (len(args.expect_binary_sha256) != 64
                                      or any(c not in '0123456789abcdef' for c in args.expect_binary_sha256)):
        parser.error('--expect-binary-sha256 must be a full lowercase sha256')
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(probe=PROBE, design='k1-sdpa-head-slice-design.md sections 3, 7, 8', passed=False,
                  argv=list(sys.argv[1:] if argv is None else argv), capacities=args.capacities, shapes=args.shapes,
                  geometry={name: shape_geometry(name) for name in args.shapes},
                  chunks={capacity: busiest_chunks(capacity) for capacity in args.capacities}, basis=args.basis,
                  thresholds=dict(a_time=A_TIME, a_slope=A_SLOPE, c_time=C_TIME, accept=ACCEPT, stop=STOP,
                                  calls_per_replay=CALLS_PER_REPLAY, trace_basis=TRACE_BASIS),
                  env={name: os.environ.get(name) for name in ENV_RECORDED}, watchdog=args.watchdog,
                  card_m_reference_us=CARD_M_US, failures=[], warnings=[], checksums=[], timing=[])
    native = card.NativeLog(args.out.with_name(args.out.name + '.native.log'))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write_report(extra=None):
        payload = {key: value for key, value in report.items() if not key.startswith('_')}
        payload['requested_programs'] = sorted(card.describe(key) for key in report.get('_requested', ()))
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    def on_fire(label):
        try:
            write_report(dict(error='watchdog: %r exceeded its budget' % (label,), passed=False))
        except Exception:  # noqa: BLE001 - the main thread may be mid-update; the WATCHDOG line stands
            pass

    WATCHDOG = Watchdog(args.watchdog, on_fire=on_fire).start()
    card.WATCHDOG = WATCHDOG                     # card.Case's uploads, calls and read-backs
    try:
        try:
            with native:
                run(args, report)
            if report.get('binary', {}).get('stage') == 3:
                report['factory_lines'] = card.factory_lines(native.text())
                for problem in card.check_program_lines(report['factory_lines'], report.get('_requested', set())):
                    report['failures'].append('factory log: ' + problem)
        except Exception as error:  # noqa: BLE001
            report['error'] = '%s: %s' % (type(error).__name__, error)
        invalid = []
        if report['failures'] or report.get('error'):
            invalid.append('measurement invalid: %d failures%s' % (len(report['failures']),
                                                                   ', error' if report.get('error') else ''))
        if args.no_timing:
            invalid.append('no timing (--no-timing: the watcher pass)')
        try:
            analyse(report, args.basis, invalid)
        except Exception as error:  # noqa: BLE001 - the measurements are still written
            report['error'] = report.get('error') or 'analysis: %s: %s' % (type(error).__name__, error)
        report['passed'] = not report['failures'] and not report.get('error') and bool(report['checksums'])
    finally:
        write_report()
    for failure in report['failures']:
        print('FAIL', failure)
    for warning in report['warnings']:
        print('WARN', warning)
    if report.get('error'):
        print('ERROR', report['error'])
    for line in slope_lines(report):
        print(line)
    print(report.get('verdict_line', 'K1_PROBE verdict=NO-DECISION'), flush=True)
    print('SDPA_K1_PROBE passed=%s checksums=%d timing_rows=%d failures=%d warnings=%d report=%s native_log=%s' % (
        report['passed'], len(report['checksums']), len(report['timing']), len(report['failures']),
        len(report['warnings']), args.out, native.path), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
