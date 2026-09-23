#!/usr/bin/env python3
"""Card-M unit test and microbenches for gdn_prefill_conv_exact (GDN prefill conv lever #2).

WHAT IT PROVES. On card M, in the serving image, the op returns the SAME BYTES as the path it
replaces in gdn/tp.py forward_prefill - the image's own

    conv, new_state = _causal_conv1d_fir(qkv_L1, None, None, 4, device, memory_config=L1,
                                         conv_state=carry, weight_taps=taps, bias_dev=None,
                                         valid_len=valid_len)
    q, k, v = ttnn.slice(conv, ...) x 3           (the flat q/k/v split, kd = 1024)

compared with torch.equal on int16 views of q, k, v and new_state (so -0, denormals and NaN
payloads count). qkv is [1, T, 5120] bf16 TILE in L1 (as served), taps 4 x [1, 1, 5120].

  matrix     T in {32, 64, 2048}; valid_len in {1, 2, 3, 31, 32, 33, 34, 264, 520, 1288, 2047,
             2048} (those <= T) plus None; carry none / zeros / randn / special (randn with
             -0 and denormal entries); x randn at T=2048 over every carry, then every data kind
             (randn x1e-3, x1, x30, dyadic RNE ties, exact +-0, denormals, near bf16 max) at
             valid_len None / 1288 / 2047 and several seeds
  inputs     qkv, carry and taps are byte-unchanged by the op
  mirror     MIRROR_PACK (the served pack/unpack replayed literally) gives the same bytes
  denormal   on the -0 / denormal carry and data cases, every (PCX_FLUSH_X_DENORM 0/1/2,
             state canonicalisation) setting: which ones match the FIR is RECORDED per state path
             (valid_len None = the static slice, else the one-hot matmul). The flush models a
             round trip that flushes x itself; the canonicalisation (-0 -> +0, denormals too under
             CANON_DENORM) models the state: always on the one-hot path, under STATIC_CANON on the
             static one. A module default (FLUSH_X_DENORM_DEFAULT, CANON_DENORM_DEFAULT,
             STATIC_CANON_DEFAULT: what the op runs by default) that does not match every case of
             a path fails the run and names the setting that does; a case no setting matches
             fails on its own
  negative   NEG_FP32, NEG_SHIFT, NEG_STALE, NEG_TAPSWAP must each DIFFER on randn data
  chain      three T=2048 chunks then a valid_len=1288 tail, each new_state fed back as the next
             carry, against the FIR chain byte for byte (full chunks at valid_len 2048 and None)
  cache      a second call with a different valid_len and fresh buffers is exact and adds no
             program-cache entry
  real taps  --taps-from <HF snapshot dir>: layers 0 and 47's conv1d taps through the model's own
             prepare_conv_taps, both TP2 halves (skipped, and said so, when absent)

Microbenches (section 7; wall clock over synchronised batches - device-profiler numbers need
PROFILE=1 in run_card_m.sh, which also calls ttnn.ReadDeviceProfiler):
  shift      PCX_MB_SHIFT_ONLY (no compute kernel; c_0 = x shifted by 3 written out and checked
             against a torch shift), NOC-loopback copies vs PCX_SHIFT_WORDS. Gate <= 0.3 ms
  full       the op vs the composed FIR + 3 slices at T=2048. Gate <= 0.4 ms (served 2.28 ms)
  host       the wrapper's Python time per call on a program-cache hit, median of 200. Gate <= 50 us

RUN on card M only, inside the serving image, with a fresh kernel cache: run_card_m.sh does this.
The helpers above `Device part` import no ttnn and are unit-tested on CPU
(test_gdn_prefill_conv_card_m.py).
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

C, KD, K = 5120, 1024, 4
TS = (32, 64, 2048)
VALID_LENS = (1, 2, 3, 31, 32, 33, 34, 264, 520, 1288, 2047, 2048)
CARRIES = ('none', 'zeros', 'randn', 'special')
DATA = ('randn', 'small', 'large', 'ties', 'zeros', 'denormal', 'nearmax')
NEGATIVES = ('fp32', 'shift', 'stale', 'tapswap')
SEEDS = (0, 1, 2)
# The serving image's (0648ca9a) ttnn_gated_deltanet.py. It differs from the TT-Sim 9f9cd4fd copy
# (e2fe112e) only in _causal_conv1d_fir's tap slices, ttnn.slice(..., memory_config=_dram): memory
# placement, not arithmetic (card M pcx-20260923T063828 compared against this FIR and was exact on
# q/k/v). The comparison always runs against the image's own FIR; the pin says which one.
FIR_SHA256 = 'fac29122cc7c3c01b93b40822b16605b26c7c400e602f0ac221707a5fa7203ab'
# Spec section 1: the served FIR and the LLK headers whose arithmetic the op reproduces.
SHA_FILES = (
    'models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py',
    'tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h',
    'tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h',
    'tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h',
    'tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_silu.h',
)
GATES_MS = dict(shift=0.3, full=0.4)
GATE_HOST_US = 50.0
# The carry / data kinds that hold -0 or denormals: they run the setting search.
DENORMAL_CASES = ('special', 'denormal', 'zeros')


# ---------------------------------------------------------------------------------------------
# Pure helpers (CPU-tested).
# ---------------------------------------------------------------------------------------------

def valid_lens_for(T):
    return [vl for vl in VALID_LENS if vl <= T] + [None]


def matrix(ts=TS, seeds=SEEDS, quick=False):
    """The equality cases, as dicts (T, valid_len, carry, data, seed)."""
    cases = []
    for T in ts:
        for vl in valid_lens_for(T):
            for carry in CARRIES:
                if quick and T == 2048 and vl not in (1, 1288, 2048, None):
                    continue
                cases.append(dict(T=T, valid_len=vl, carry=carry, data='randn', seed=seeds[0]))
    if 2048 in ts:
        for data in DATA:
            for vl in (None, 1288, 2047):
                for carry in ('randn', 'special'):
                    for seed in (seeds if not quick else seeds[:1]):
                        case = dict(T=2048, valid_len=vl, carry=carry, data=data, seed=seed)
                        if case not in cases:
                            cases.append(case)
    return cases


def case_name(case):
    return 'T%(T)d_vl%(valid_len)s_%(carry)s_%(data)s_s%(seed)d' % case


def make_x(torch, kind, T, cols, seed):
    """[T, cols] bf16 of one data kind."""
    generator = torch.Generator().manual_seed(1000 + seed)
    base = torch.randn(T, cols, generator=generator)
    if kind == 'randn':
        return base.to(torch.bfloat16)
    if kind == 'small':
        return (base * 1e-3).to(torch.bfloat16)
    if kind == 'large':
        return (base * 30).to(torch.bfloat16)
    if kind == 'ties':
        # Dyadic values with 8-9 significant bits: products and sums land on bf16 halfway points.
        mantissa = torch.randint(128, 512, (T, cols), generator=generator).float()
        exponent = torch.randint(-8, 2, (T, cols), generator=generator).float()
        sign = torch.where(torch.rand(T, cols, generator=generator) < 0.5, -1.0, 1.0)
        return (sign * mantissa * torch.pow(2.0, exponent - 8)).to(torch.bfloat16)
    if kind == 'zeros':
        out = base.to(torch.bfloat16)
        mask = torch.rand(T, cols, generator=generator)
        out[mask < 0.25] = 0.0
        out[(mask >= 0.25) & (mask < 0.5)] = -0.0
        return out
    if kind == 'denormal':
        out = base.to(torch.bfloat16)
        mask = torch.rand(T, cols, generator=generator) < 0.3
        tiny = (base * 1e-39).to(torch.bfloat16)   # bf16 subnormals (|x| < 1.18e-38)
        out[mask] = tiny[mask]
        return out
    if kind == 'nearmax':
        # Near bf16 max, where x*w (|w| < ~4) overflows to inf: the FIR's handling is the reference.
        return (base.sign() * 3.0e38 * (0.5 + base.abs().clamp(max=1) / 2)).to(torch.bfloat16)
    raise ValueError(kind)


def make_carry(torch, kind, cols, seed):
    """None or [3, cols] bf16."""
    if kind == 'none':
        return None
    if kind == 'zeros':
        return torch.zeros(3, cols, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(2000 + seed)
    carry = torch.randn(3, cols, generator=generator).to(torch.bfloat16)
    if kind == 'special':
        carry[0, ::7] = -0.0
        carry[1, 3::11] = (torch.randn(carry[1, 3::11].shape, generator=generator) * 1e-39).to(torch.bfloat16)
        carry[2, 5::13] = -0.0
    elif kind != 'randn':
        raise ValueError(kind)
    return carry


def make_taps(torch, cols, seed):
    generator = torch.Generator().manual_seed(3000 + seed)
    return (torch.randn(K, cols, generator=generator) * 0.5).to(torch.bfloat16)


def int16(torch, tensor):
    return tensor.to(torch.bfloat16).contiguous().view(torch.int16)


def compare(torch, candidate, reference):
    """Byte comparison of two bf16 tensors: exact, how many differ, the largest ulp distance."""
    a, b = int16(torch, candidate), int16(torch, reference)
    if a.shape != b.shape:
        return dict(exact=False, shape=[list(a.shape), list(b.shape)])
    differ = a != b
    count = int(differ.sum())
    result = dict(exact=count == 0, differing=count)
    if count:
        ordered_a = torch.where(a < 0, -32768 - a.int(), a.int())
        ordered_b = torch.where(b < 0, -32768 - b.int(), b.int())
        result['max_ulp'] = int((ordered_a - ordered_b).abs().max())
        index = int(differ.reshape(-1).nonzero()[0])
        result['first'] = dict(index=index, candidate=int(a.reshape(-1)[index]), reference=int(b.reshape(-1)[index]))
    return result


def shifted_reference(torch, x, carry, s=3):
    """x shifted down by s rows over concat(carry or zeros, x): row t = x[t - s]."""
    T = x.shape[0]
    pad = carry if carry is not None else torch.zeros(3, x.shape[1], dtype=x.dtype)
    padded = torch.cat([pad, x], dim=0)
    return padded[3 - s:3 - s + T]


def file_shas(root):
    out = {}
    for relative in SHA_FILES:
        path = Path(root) / relative
        out[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return out


def summarise_timing(samples_ms):
    return dict(median_ms=statistics.median(samples_ms), min_ms=min(samples_ms), samples=len(samples_ms))


STATIC, ONE_HOT = 'static', 'one_hot'


def state_path(valid_len):
    """valid_len None: the FIR's static slice of x_padded (the reader's canon 0); else its one-hot matmul."""
    return STATIC if valid_len is None else ONE_HOT


def denormal_settings(pcx, valid_len):
    """The (flush_x_denorm, canon_denorm) settings a denormal case runs under. canon None is a raw
    state (no canonicalisation): only the static path can be raw (STATIC_CANON off); the one-hot
    path always canonicalises, so there canon is False (-0 only) or True (denormals too)."""
    canons = (None, False, True) if valid_len is None else (False, True)
    return [(flush, canon) for flush in pcx.FLUSH_X_DENORM_MODES for canon in canons]


def module_setting(pcx, valid_len):
    """The setting the op runs by default on this state path."""
    raw = valid_len is None and not pcx.STATIC_CANON_DEFAULT
    return (pcx.FLUSH_X_DENORM_DEFAULT, None if raw else pcx.CANON_DENORM_DEFAULT)


def setting_variant(setting):
    """gdn_prefill_conv_exact keyword arguments for one setting (static_canon is inert on the
    one-hot path, which always canonicalises)."""
    flush, canon = setting
    if canon is None:
        return dict(flush_x_denorm=flush, static_canon=False)
    return dict(flush_x_denorm=flush, canon_denorm=canon, static_canon=True)


def settings_choice(matched_per_case, default):
    """What one state path's denormal cases say. matched_per_case: per case, the settings whose
    output matched the FIR byte for byte. 'untested' (no case), 'inconsistent' (no single setting
    matches every case), 'keep' (the module default matches every case) or 'change' (only other
    settings do); 'consistent' lists the settings that match every case."""
    if not matched_per_case:
        return dict(verdict='untested', consistent=[])
    consistent = set(tuple(s) for s in matched_per_case[0])
    for matched in matched_per_case[1:]:
        consistent &= set(tuple(s) for s in matched)
    consistent = sorted(consistent, key=repr)
    if not consistent:
        outcome = 'inconsistent'
    elif tuple(default) in consistent:
        outcome = 'keep'
    else:
        outcome = 'change'
    return dict(verdict=outcome, consistent=[list(s) for s in consistent])


def recommend(static_choice, one_hot_choice, modes=(0, 1, 2)):
    """The [FLUSH_X_DENORM_DEFAULT, CANON_DENORM_DEFAULT, STATIC_CANON_DEFAULT] triples both paths
    accept (flush and CANON_DENORM are one module constant each for both paths; a path with no
    denormal case accepts every setting)."""
    def accepts(choice, setting):
        return choice['verdict'] == 'untested' or list(setting) in choice['consistent']

    out = []
    for flush in modes:
        for canon in (False, True):
            for static in (False, True):
                if accepts(one_hot_choice, (flush, canon)) and \
                        accepts(static_choice, (flush, canon if static else None)):
                    out.append([flush, canon, static])
    return out


def verdict(report):
    """Pass iff no failure was recorded and every section that ran is complete."""
    return not report['failures'] and report.get('cases_run', 0) > 0


class Watchdog:
    """A per-device-call deadline: a hung NoC handshake cannot be interrupted from Python, so the
    poller prints WATCHDOG, writes the partial report and os._exit(3)s."""

    def __init__(self, seconds, on_fire=None):
        self.seconds, self.on_fire = seconds, on_fire
        self.label, self.deadline = None, None
        self.lock = threading.Lock()

    def start(self):
        if self.seconds:
            threading.Thread(target=self.poll, name='pcx-watchdog', daemon=True).start()
        return self

    @contextmanager
    def op(self, label):
        if not self.seconds:
            yield
            return
        with self.lock:
            outer = (self.label, self.deadline)
            self.label, self.deadline = label, time.monotonic() + self.seconds
        try:
            yield
        finally:
            with self.lock:
                self.label, self.deadline = outer

    def poll(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                label, deadline = self.label, self.deadline
            if label is not None and time.monotonic() >= deadline:
                sys.stdout.write('WATCHDOG: %r did not return within %ss; exiting 3 (docker rm -f, then '
                                 'tt-smi -r card M only)\n' % (label, self.seconds))
                sys.stdout.flush()
                try:
                    if self.on_fire is not None:
                        self.on_fire(label)
                finally:
                    os._exit(3)


WATCHDOG = Watchdog(0)


# ---------------------------------------------------------------------------------------------
# Device part: card M only.
# ---------------------------------------------------------------------------------------------

class Bench:
    def __init__(self, ttnn, torch, pcx, fir, device):
        self.ttnn, self.torch, self.pcx, self.fir, self.device = ttnn, torch, pcx, fir, device

    def upload(self, host, memory_config=None):
        ttnn = self.ttnn
        return ttnn.from_torch(host.unsqueeze(0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=self.device, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)

    def read(self, tensor):
        return self.ttnn.to_torch(self.ttnn.get_device_tensors(tensor)[0]).to(self.torch.bfloat16)

    def inputs(self, x, carry, taps):
        ttnn = self.ttnn
        qkv = self.upload(x, ttnn.L1_MEMORY_CONFIG)
        state = self.upload(carry) if carry is not None else None
        tap_tensors = [self.upload(taps[j:j + 1].reshape(1, -1)) for j in range(K)]
        return qkv, state, tap_tensors

    def release(self, *tensors):
        for tensor in tensors:
            if tensor is None:
                continue
            if isinstance(tensor, (list, tuple)):
                self.release(*tensor)
            else:
                self.ttnn.deallocate(tensor)

    def reference(self, qkv, carry, taps, valid_len):
        ttnn = self.ttnn
        T = qkv.shape[1]
        with WATCHDOG.op('FIR T=%d vl=%s' % (T, valid_len)):
            conv, state = self.fir(qkv, None, None, K, self.device, memory_config=ttnn.L1_MEMORY_CONFIG,
                                   conv_state=carry, weight_taps=taps, bias_dev=None, valid_len=valid_len)
            q = ttnn.slice(conv, (0, 0, 0), (1, T, KD))
            k = ttnn.slice(conv, (0, 0, KD), (1, T, 2 * KD))
            v = ttnn.slice(conv, (0, 0, 2 * KD), (1, T, C))
        ttnn.deallocate(conv)
        return q, k, v, state

    def candidate(self, qkv, carry, taps, valid_len, **variant):
        with WATCHDOG.op('op T=%d vl=%s %s' % (qkv.shape[1], valid_len, sorted(variant))):
            return self.pcx.gdn_prefill_conv_exact(self.device, qkv, carry, taps, valid_len=valid_len,
                                                   key_dim_tp=KD, **variant)

    def compare_outputs(self, mine, theirs):
        with WATCHDOG.op('read back'):
            results = {name: compare(self.torch, self.read(a), self.read(b))
                       for name, a, b in zip(('q', 'k', 'v', 'new_state'), mine, theirs)}
        return results


def exact(results):
    return all(result['exact'] for result in results.values())


def equality_cases(bench, args, report):
    torch = bench.torch
    failures = report['failures']
    settings = report.setdefault('denormal_settings', {})
    settings.setdefault(STATIC, [])
    settings.setdefault(ONE_HOT, [])
    for case in matrix(args.ts, args.seeds, quick=args.quick):
        name = case_name(case)
        x = make_x(torch, case['data'], case['T'], C, case['seed'])
        carry = make_carry(torch, case['carry'], C, case['seed'])
        taps = make_taps(torch, C, case['seed'])
        qkv, state, tap_tensors = bench.inputs(x, carry, taps)
        try:
            reference = bench.reference(qkv, state, tap_tensors, case['valid_len'])
            watch_inputs = case['T'] <= 64 or (case['data'] == 'randn' and case['valid_len'] is None)
            before = [bench.read(t) for t in [qkv] + ([state] if state is not None else []) + tap_tensors] \
                if watch_inputs else None
            output = bench.candidate(qkv, state, tap_tensors, case['valid_len'])
            results = bench.compare_outputs(output, reference)
            entry = dict(case, name=name, results=results, exact=exact(results))
            if before is not None:
                after = [bench.read(t) for t in [qkv] + ([state] if state is not None else []) + tap_tensors]
                entry['inputs_unchanged'] = all(torch.equal(int16(torch, a), int16(torch, b)) for a, b in zip(before, after))
                if not entry['inputs_unchanged']:
                    failures.append('%s: the op changed a borrowed input' % name)
            denormal = case['carry'] in DENORMAL_CASES or case['data'] in DENORMAL_CASES
            if denormal:
                # Both state paths: the x round-trip flush reaches q/k/v and new_state either way.
                default = module_setting(bench.pcx, case['valid_len'])
                matched = [list(default)] if entry['exact'] else []
                for setting in denormal_settings(bench.pcx, case['valid_len']):
                    if setting == default:
                        continue
                    alternative = bench.candidate(qkv, state, tap_tensors, case['valid_len'], **setting_variant(setting))
                    if exact(bench.compare_outputs(alternative, reference)):
                        matched.append(list(setting))
                    bench.release(alternative)
                entry['matched_settings'] = matched
                settings[state_path(case['valid_len'])].append(matched)
                if not matched:
                    failures.append('%s: no flush / canonicalisation setting matches the FIR %s'
                                    % (name, {n: r for n, r in results.items() if not r['exact']}))
            elif not entry['exact']:
                failures.append('%s: %s' % (name, {n: r for n, r in results.items() if not r['exact']}))
            if args.mirror and case['seed'] == args.seeds[0] and case['T'] == 2048 and case['carry'] == 'randn' \
                    and case['valid_len'] in (None, 1288):
                mirrored = bench.candidate(qkv, state, tap_tensors, case['valid_len'], mirror_pack=True)
                entry['mirror_pack'] = exact(bench.compare_outputs(mirrored, output))
                bench.release(mirrored)
                if not entry['mirror_pack']:
                    failures.append('%s: MIRROR_PACK changed the bytes' % name)
            report['cases'].append(entry)
            report['cases_run'] = report.get('cases_run', 0) + 1
            print('%-44s %s%s' % (name, 'exact' if entry['exact'] else 'DIFF %s' % {n: r for n, r in results.items() if not r['exact']},
                                  '' if 'matched_settings' not in entry else ' matched=%s' % entry['matched_settings']),
                  flush=True)
            bench.release(output, reference)
        finally:
            bench.release(qkv, state, tap_tensors)
    choices = {}
    for path, valid_len in ((STATIC, None), (ONE_HOT, 1)):
        default = module_setting(bench.pcx, valid_len)
        choices[path] = dict(settings_choice(settings[path], default), module_default=list(default),
                             cases=len(settings[path]))
        print('denormal settings, %s path: %s' % (path, choices[path]), flush=True)
        if choices[path]['verdict'] == 'inconsistent':
            failures.append('%s path: no single (flush_x_denorm, canon_denorm) setting matches every denormal case'
                            % path)
        elif choices[path]['verdict'] == 'change':
            failures.append('%s path: the module default %s does not match the FIR; only %s do - set '
                            'gdn_prefill_conv_exact.FLUSH_X_DENORM_DEFAULT / CANON_DENORM_DEFAULT / '
                            'STATIC_CANON_DEFAULT' % (path, list(default), choices[path]['consistent']))
    choices['recommend'] = recommend(choices[STATIC], choices[ONE_HOT], bench.pcx.FLUSH_X_DENORM_MODES)
    report['denormal_choice'] = choices
    tested = choices[STATIC]['verdict'] != 'untested' and choices[ONE_HOT]['verdict'] != 'untested'
    if tested and not choices['recommend']:
        failures.append('the two state paths need different x flushes or CANON_DENORM: %s vs %s'
                        % (choices[STATIC]['consistent'], choices[ONE_HOT]['consistent']))


def negative_controls(bench, args, report):
    torch = bench.torch
    x = make_x(torch, 'randn', 2048, C, 0)
    carry = make_carry(torch, 'randn', C, 0)
    taps = make_taps(torch, C, 0)
    qkv, state, tap_tensors = bench.inputs(x, carry, taps)
    try:
        reference = bench.reference(qkv, state, tap_tensors, 1288)
        for negative in NEGATIVES:
            output = bench.candidate(qkv, state, tap_tensors, 1288, negative=negative)
            results = bench.compare_outputs(output, reference)
            report['negative'][negative] = dict(differs=not exact(results),
                                                differing={n: r.get('differing') for n, r in results.items()})
            print('negative %-8s %s' % (negative, 'differs (good)' if not exact(results) else 'EXACT (bad)'), flush=True)
            if exact(results):
                failures = report['failures']
                failures.append('negative control %s did not change the output: the test cannot see that fault' % negative)
            bench.release(output)
        bench.release(reference)
    finally:
        bench.release(qkv, state, tap_tensors)


def chain(bench, args, report):
    """Three full chunks then a tail, each new_state feeding the next carry, op chain vs FIR chain."""
    torch = bench.torch
    ttnn = bench.ttnn
    for full_vl in (2048, None):
        for start in ('none', 'zeros'):
            label = 'chain full_vl=%s start=%s' % (full_vl, start)
            taps = make_taps(torch, C, 5)
            tap_tensors = [bench.upload(taps[j:j + 1].reshape(1, -1)) for j in range(K)]
            carry_host = make_carry(torch, start, C, 5)
            mine = bench.upload(carry_host) if carry_host is not None else None
            theirs = bench.upload(carry_host) if carry_host is not None else None
            steps = []
            try:
                for step, vl in enumerate((full_vl, full_vl, full_vl, 1288)):
                    x = make_x(torch, 'randn', 2048, C, 10 + step)
                    qkv = bench.upload(x, ttnn.L1_MEMORY_CONFIG)
                    reference = bench.reference(qkv, theirs, tap_tensors, vl)
                    output = bench.candidate(qkv, mine, tap_tensors, vl)
                    results = bench.compare_outputs(output, reference)
                    steps.append(dict(step=step, valid_len=vl, exact=exact(results),
                                      differing={n: r.get('differing') for n, r in results.items() if not r['exact']}))
                    bench.release(qkv, output[:3], reference[:3], mine, theirs)
                    mine, theirs = output[3], reference[3]
                    if not exact(results):
                        report['failures'].append('%s step %d: %s' % (label, step, steps[-1]['differing']))
            finally:
                bench.release(mine, theirs, tap_tensors)
            report['chain'].append(dict(label=label, steps=steps))
            print('%s %s' % (label, [s['exact'] for s in steps]), flush=True)


def program_cache(bench, args, report):
    torch = bench.torch
    device = bench.device
    counts = []
    for index, vl in enumerate((2048, 1288, 7)):
        x = make_x(torch, 'randn', 2048, C, 20 + index)
        carry = make_carry(torch, 'randn', C, 20 + index)
        taps = make_taps(torch, C, 20 + index)
        qkv, state, tap_tensors = bench.inputs(x, carry, taps)
        reference = bench.reference(qkv, state, tap_tensors, vl)
        entries = device.num_program_cache_entries()
        output = bench.candidate(qkv, state, tap_tensors, vl)
        counts.append(device.num_program_cache_entries() - entries)
        results = bench.compare_outputs(output, reference)
        if not exact(results):
            report['failures'].append('program cache call %d (vl=%d): %s' % (index, vl, results))
        bench.release(output, reference, qkv, state, tap_tensors)
    report['program_cache'] = dict(new_entries_per_call=counts, descriptor_cache=bench.pcx.cache_size())
    print('program cache new entries per call %s' % counts, flush=True)
    if any(count != 0 for count in counts[1:]):
        report['failures'].append('a cached call added program-cache entries: %s' % counts)


def real_taps(bench, args, report):
    if not args.taps_from:
        report['real_taps'] = 'skipped: no --taps-from'
        return
    torch = bench.torch
    try:
        from safetensors import safe_open
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        root = Path(args.taps_from)
        index = json.loads(next(root.rglob('model.safetensors.index.json')).read_text())['weight_map']
        found = {}
        for layer in (0, 47):
            key = [name for name in index if name.endswith('layers.%d.linear_attn.conv1d.weight' % layer)]
            if len(key) != 1:
                raise KeyError('layer %d conv1d.weight: %r' % (layer, key))
            shard = next(root.rglob(index[key[0]]))
            with safe_open(str(shard), framework='pt') as handle:
                found[layer] = handle.get_tensor(key[0])
    except Exception as error:  # noqa: BLE001 - optional section; the reason is the result
        report['real_taps'] = 'skipped: %r' % (error,)
        print('real taps skipped: %r' % (error,), flush=True)
        return
    results = {}
    for layer, weight in found.items():
        try:
            # key_dim 2048 (16 x 128), value heads 48 x 128, TP2: qkv_dim 10240 -> 5120 per chip.
            flat = [tap.reshape(-1) for tap in tpc.prepare_conv_taps(weight, 2048, 16, 128, 48, 128, K, 2)]
            if any(tap.numel() != 2 * C for tap in flat):
                raise ValueError('prepare_conv_taps gave %s, expected 4 x %d' % ([tap.numel() for tap in flat], 2 * C))
        except Exception as error:  # noqa: BLE001 - the model helper's signature is the image's; say so
            results['layer%d' % layer] = 'skipped: %r' % (error,)
            continue
        for chip in (0, 1):
            host = torch.stack([tap[chip * C:(chip + 1) * C] for tap in flat]).to(torch.bfloat16)
            x = make_x(torch, 'randn', 2048, C, layer)
            carry = make_carry(torch, 'randn', C, layer)
            qkv, state, tap_tensors = bench.inputs(x, carry, host)
            reference = bench.reference(qkv, state, tap_tensors, 1288)
            output = bench.candidate(qkv, state, tap_tensors, 1288)
            outcome = bench.compare_outputs(output, reference)
            results['layer%d_chip%d' % (layer, chip)] = exact(outcome)
            if not exact(outcome):
                report['failures'].append('real taps layer %d chip %d: %s' % (layer, chip, outcome))
            bench.release(output, reference, qkv, state, tap_tensors)
    report['real_taps'] = results
    print('real taps %s' % results, flush=True)


def timed(bench, call, iters, warmup):
    ttnn = bench.ttnn
    for _ in range(warmup):
        bench.release(call())
    with WATCHDOG.op('synchronize'):
        ttnn.synchronize_device(bench.device)
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iters):
            bench.release(call())
        with WATCHDOG.op('synchronize'):
            ttnn.synchronize_device(bench.device)
        samples.append((time.perf_counter() - start) * 1000.0 / iters)
    return summarise_timing(samples)


def microbenches(bench, args, report):
    torch = bench.torch
    x = make_x(torch, 'randn', 2048, C, 0)
    carry = make_carry(torch, 'randn', C, 0)
    taps = make_taps(torch, C, 0)
    qkv, state, tap_tensors = bench.inputs(x, carry, taps)
    timing = report['timing']
    try:
        expected = shifted_reference(torch, x, carry)
        for words in (False, True):
            label = 'shift_%s' % ('words' if words else 'noc')
            out = bench.candidate(qkv, state, tap_tensors, None, shift_only=True, shift_words=words)
            check = compare(torch, bench.read(out)[0], expected)
            bench.release(out)
            timing[label] = dict(timed(bench, lambda: bench.candidate(qkv, state, tap_tensors, None, shift_only=True,
                                                                      shift_words=words), args.iters, args.warmup),
                                 exact=check['exact'])
            print('%s %s' % (label, timing[label]), flush=True)
            if not check['exact']:
                report['failures'].append('%s: the shifted tile differs from a torch shift %s' % (label, check))
        best = min(timing['shift_noc']['median_ms'], timing['shift_words']['median_ms'])
        timing['shift_gate_ms'] = GATES_MS['shift']
        timing['shift_gate_met'] = best <= GATES_MS['shift']
        timing['op'] = timed(bench, lambda: bench.candidate(qkv, state, tap_tensors, 1288), args.iters, args.warmup)
        timing['composed'] = timed(bench, lambda: bench.reference(qkv, state, tap_tensors, 1288), args.iters, args.warmup)
        timing['full_gate_ms'] = GATES_MS['full']
        timing['full_gate_met'] = timing['op']['median_ms'] <= GATES_MS['full']
        timing['speedup'] = timing['composed']['median_ms'] / timing['op']['median_ms']
        print('op %s composed %s' % (timing['op'], timing['composed']), flush=True)
        host = []
        for _ in range(args.host_calls):
            start = time.perf_counter()
            out = bench.candidate(qkv, state, tap_tensors, 1288)
            host.append((time.perf_counter() - start) * 1e6)
            bench.release(out)
        with WATCHDOG.op('synchronize'):
            bench.ttnn.synchronize_device(bench.device)
        timing['host_us'] = dict(median=statistics.median(host), p90=sorted(host)[int(0.9 * len(host))], calls=len(host))
        timing['host_gate_us'] = GATE_HOST_US
        timing['host_gate_met'] = timing['host_us']['median'] <= GATE_HOST_US
        print('host per call %s' % timing['host_us'], flush=True)
        if os.environ.get('TT_METAL_DEVICE_PROFILER') == '1':
            bench.ttnn.ReadDeviceProfiler(bench.device)
            timing['device_profiler'] = 'read: generated/profiler (the run script mounts it into the results)'
    finally:
        bench.release(qkv, state, tap_tensors)


def run(args, report):
    import torch
    import ttnn

    sys.path.insert(0, str(args.op_dir))
    sys.path.insert(0, os.environ.get('TT_METAL_HOME', '/opt/tt-metal'))
    import gdn_prefill_conv_exact as pcx
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir

    report['image_shas'] = file_shas(os.environ.get('TT_METAL_HOME', '/opt/tt-metal'))
    report['fir_sha_ok'] = report['image_shas'][SHA_FILES[0]] == FIR_SHA256
    report['op_files'] = {name: hashlib.sha256((Path(args.op_dir) / name).read_bytes()).hexdigest()
                          for name in pcx.RUNTIME_FILES}
    report['op_source_sha'] = pcx.source_sha(args.op_dir)
    print('FIR sha ok=%s op kernels %s' % (report['fir_sha_ok'], report['op_source_sha']), flush=True)
    if not report['fir_sha_ok']:
        report['failures'].append('the image FIR is not %s: exactness is against a different source'
                                  % FIR_SHA256[:8])
    device = ttnn.open_device(device_id=args.device_id)
    try:
        report['mesh_shape'] = list(device.shape) if hasattr(device, 'shape') else None
        grid = device.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        bench = Bench(ttnn, torch, pcx, _causal_conv1d_fir, device)
        sections = [s for s in args.sections]
        if 'equality' in sections:
            equality_cases(bench, args, report)
        if 'negative' in sections:
            negative_controls(bench, args, report)
        if 'chain' in sections:
            chain(bench, args, report)
        if 'cache' in sections:
            program_cache(bench, args, report)
        if 'real' in sections:
            real_taps(bench, args, report)
        if 'timing' in sections and not args.no_timing:
            microbenches(bench, args, report)
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--op-dir', type=Path, default=Path('/bench/pcx'))
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--ts', default=','.join(map(str, TS)))
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--sections', default='equality,negative,chain,cache,real,timing')
    parser.add_argument('--quick', action='store_true', help='a thinner T=2048 matrix (the watcher pass)')
    parser.add_argument('--no-mirror', dest='mirror', action='store_false')
    parser.add_argument('--taps-from', help='HF snapshot directory (model.safetensors.index.json) for real taps')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--host-calls', type=int, default=200)
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--watchdog', type=float, default=0)
    args = parser.parse_args(argv)
    args.ts = [int(value) for value in args.ts.split(',')]
    args.seeds = [int(value) for value in args.seeds.split(',')]
    args.sections = [value for value in args.sections.split(',') if value]
    unknown = set(args.sections) - {'equality', 'negative', 'chain', 'cache', 'real', 'timing'}
    if unknown or any(T % 32 or T <= 0 for T in args.ts):
        parser.error('unknown section %s or a T that is not a positive multiple of 32' % sorted(unknown))
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(passed=False, failures=[], cases=[], negative={}, chain=[], timing={},
                  denormal_settings={}, args={k: str(v) for k, v in vars(args).items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write(extra=None):
        payload = dict(report)
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    WATCHDOG = Watchdog(args.watchdog, on_fire=lambda label: write(dict(error='watchdog: %r' % label))).start()
    try:
        run(args, report)
        report['passed'] = verdict(report)
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised for the exit status
        report['error'] = repr(error)
        write()
        raise
    write()
    print('PASSED' if report['passed'] else 'FAILED: %d failures' % len(report['failures']), flush=True)
    for failure in report['failures'][:40]:
        print('  ' + failure, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
