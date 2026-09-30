#!/usr/bin/env python3
"""K5-A card-B probe: the served batched GDN launch against the K5-A launch, bit for bit, then timed.

Arms (K5 plan, section 4):
  C   the control: gdn_user_batch.execute with gdn_user_batch.load_kernels, the launch the arms
      serve. Its generated sources must hash to SERVED_SHA256, and QWEN_FAST_VERIFY_T1=1 must make
      it the coalesced build (verify_trace_t1 #12), or the probe refuses to run.
  A   K5-A at level 0 (gdn_seq_block.load_kernels(..., unqualified=True)).
  A0  bisection only: T5 and T6 unfused (outer product through its own ring, two copy passes).
  N   negative control: A with the state add on the SFPU, the known-inexact variant
      (docs/gdn-outer-add-experiment.md:6-17). N must differ somewhere, or the compare is blind.

P0, bitwise. Every case compares A, A0 and N against C: the gated output (logical values and the
padded tile bytes), all 16 x 24 x 16 snapshot tiles of every user, and that the carried state and
every input are unchanged. A mismatch reports the first differing (user, head, token, tile,
element), a ulp histogram and whether it is in the output, the snapshots or both. For
information only, C's error against a sequential fp64 reference by token position and decay class
(variant B's future Q0 baseline). Regimes:
  R1   the gdn_user_batch_device_test.py distributions, seeds 17, 23, 29, 31, 37
  R2   wide range: g in -U[0, 20], S0 = 4 randn, qkv = 8 randn
  R3   edge cases: S0 with +-0, the smallest normal and bf16 denormals; an all-zero k row (and q
       row) - the eps path; beta in {0, 1 - 2^-8}; g in {0, -88} (exp underflow)
  R3b  padding poison: rows 16-31 of the qkv, z, beta and g tiles (and beta/g columns 24-31) NaN
       or +-Inf, written through a raw page copy and read back to prove it landed
  R4   a 2,048-token chain: 128 launches, each arm feeding its own states[15] back as the next
       initial; outputs (logical and padded) and every snapshot compared at every launch, and every
       input re-read at the first, every 16th and the last launch
P0 passes when A and A0 show 0 differing bytes in every case, N shows at least one, no input moved,
and every output page of every arm was written. The result is 'pass' only when the run also covers
the whole plan (every arm and regime, the five R1 seeds, 4 users, all 128 R4 launches); the same
outcome on less is 'partial-pass', and only a 'pass' may commit a QUALIFIED triple.

Every P0 byte crosses to and from the device as a host-tilized uint32 page image through a raw page
copy (RAW_COPY): ttnn's own bf16 upload/readback flush -0.0 and denormals to +0.0 and turn NaN into
-Inf, which would hide exactly the differences a byte compare is for. And every tensor a P0 launch
allocates is first filled with a sentinel page (SentinelOperations): ttnn.empty does not clear
memory and a freed buffer is usually handed to the next arm, so a page an arm never wrote could
otherwise hold the previous arm's exact bytes and compare equal. A page still holding the sentinel
after the launch fails the case, C's included.

Traced (with timing): C, A and A0 each captured as one launch in a trace whose outputs stay
allocated, replayed, read back through the raw path and compared against C's replay - serving runs
K5-A only inside a trace.

P1 (ii), timing: per arm, 48 back-to-back launches over 48 distinct input sets captured in one
trace, replayed in serpentine order (C, A, A0 then A0, A, C, ...) so neither drift nor a fixed
predecessor favours an arm; per-launch medians and IQR. The 297/388 us lines are kernel times set
against v138's 525.8 us, and a trace time carries dispatch too, so p1_verdict applies them only
when C reproduces 525.8 us within +-3% (else 'uncalibrated'), and then to A's absolute median, to
A - C and to A / C alike. P1 (iii): `--diag nosnap,passthrough` adds A with the snapshot DRAM
writes compiled out (compute-bound time) and A with the chain replaced by pass-through (bytes-bound
time) to the timing rotation; they are never compared. P1 (i), the device-profiler duration, is
not done here (TODO: a profiler-enabled run at op-support 20000, RFP:108); until it is, an
uncalibrated run cannot return 'image-build'.

The last stdout line is one JSON object, {"kind": "gdn-seq-block-probe", ...}: the summary.

  python3 gdn_seq_block_device_test.py --out results.json
  python3 gdn_seq_block_device_test.py --out r.json --users 1 --regimes R1 --seeds 17 --skip-timing
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import gdn_multitoken as native
import gdn_seq_block as seq
import gdn_user_batch as batch
import verify_trace_t1


SERVED_SHA256 = dict(compute='9512188a20fd63f2853f1ac427f1b7dfaf96bab18f9bfb32e73c2d647283a9e0',
                     reader='6c56547a34384f9727c72fff1905566459b96233e703d5415d3f88328df0d072',
                     writer='2d62a883af12d4adc2834d1eb50920dce25f910320c5432a456705bb3daef68a')
ARMS = ('C', 'A', 'A0', 'N')
EXACT_ARMS = ('A', 'A0')
REGIMES = ('R1', 'R2', 'R3', 'R3b', 'R4')
# What the plan's P0 requires before a QUALIFIED triple is committed (K5 plan, section 4).
PLAN_R1_SEEDS = (17, 23, 29, 31, 37)
PLAN_R4_LAUNCHES = 128
PLAN_USERS = 4
P1_CENTRAL_US, P1_PASS_US = 297.0, 388.0
SERVED_LAUNCH_US = 525.8  # v138, run 35852642074 (the plan's C reference for P1 (i))
CALIBRATION = 0.03        # C must reproduce SERVED_LAUNCH_US within +-3% (K5 plan, P1 (i))
P1_CENTRAL_DELTA_US, P1_PASS_DELTA_US = P1_CENTRAL_US - SERVED_LAUNCH_US, P1_PASS_US - SERVED_LAUNCH_US
P1_CENTRAL_RATIO, P1_PASS_RATIO = P1_CENTRAL_US / SERVED_LAUNCH_US, P1_PASS_US / SERVED_LAUNCH_US
ROWS = seq.ROWS
MODULES = ('gdn_seq_block.py', 'gdn_seq_block_compute.cpp', 'gdn_seq_block_reader.cpp', 'gdn_seq_block_writer.cpp',
           'gdn_seq_block_device_test.py', 'gdn_user_batch.py', 'verify_trace_t1.py')
ULP_BUCKETS = ((1, '1'), (2, '2'), (4, '3-4'), (16, '5-16'), (256, '17-256'), (None, '>256'))
# The sentinel page: every bf16 halfword 0x7FC1, a quiet NaN whose payload no kernel here produces
# (the SFPU's NaN is 0x7FC0 / 0xFFC0). MAX_PAGES: a (16, 24, 128, 128) states tensor, the largest
# tensor a launch allocates.
SENTINEL_HALF = 0x7FC1
SENTINEL_WORD = (SENTINEL_HALF << 16) | SENTINEL_HALF
MAX_PAGES = ROWS * 24 * 16

# EXACT BYTES. ttnn's bf16 TILE upload and readback are not bit-exact: on 9f9cd4f (measured in ttsim,
# k5a/upload_probe.py) they turn -0.0 into +0.0, bf16 denormals into +0.0 and a NaN into -Inf. A
# compare built on them could not see a sign-of-zero or denormal difference, and R3/R3b would never
# put their specials on the device. So every P0 byte goes through a uint32 ROW_MAJOR tensor holding
# the host-tilized page image (exact) and this page-for-page raw copy between two tensors of one
# page size (2048 B): one bf16 tile == one 512-word row. Padding rows travel too.
RAW_WORKERS = 8
RAW_COPY = '''#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto source_args = TensorAccessorArgs<0>();
    constexpr auto destination_args = TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
    const auto source = TensorAccessor(source_args, get_arg_val<uint32_t>(0), 2048);
    const auto destination = TensorAccessor(destination_args, get_arg_val<uint32_t>(1), 2048);
    const uint32_t pages = get_arg_val<uint32_t>(2);
    const uint32_t worker = get_arg_val<uint32_t>(3);
    const uint32_t workers = get_arg_val<uint32_t>(4);
    cb_reserve_back(0, 1);
    const uint32_t scratch = get_write_ptr(0);
    for (uint32_t page = worker; page < pages; page += workers) {
        noc_async_read_tile(page, source, scratch);
        noc_async_read_barrier();
        noc_async_write_tile(page, destination, scratch);
        noc_async_write_barrier();
    }
}
'''


def padded_shape(shape):
    """The physical shape of a TILE tensor: the last two dims rounded up to 32."""
    shape = tuple(shape)
    return shape[:-2] + tuple(((value + 31) // 32) * 32 for value in shape[-2:])


def page_count(shape):
    """The 32x32 tile pages of a TILE tensor of logical `shape`."""
    total = 1
    for size in padded_shape(shape):
        total *= size
    return total // 1024


def pad_image(torch, value, fill=0.0):
    """`value` (bf16, bits kept) inside a padded_shape tensor whose padding holds `fill`."""
    image = torch.full(padded_shape(value.shape), fill, dtype=torch.float32).bfloat16()
    image[tuple(slice(0, size) for size in value.shape)] = value
    return image


def tile_image(torch, image):
    """A padded bf16 tensor as TILE pages, int32 [pages, 512]: page order (leading dims, tile row,
    tile column); in a page, face (r//16)*2 + c//16 then element (r%16)*16 + c%16 - the layout
    the kernels index (reader.cpp:7-10)."""
    rows, columns = image.shape[-2:]
    lead = image.numel() // (rows * columns)
    bits = image.contiguous().view(torch.int16).reshape(lead, rows // 32, 2, 16, columns // 32, 2, 16)
    return bits.permute(0, 1, 4, 2, 5, 3, 6).reshape(-1, 1024).contiguous().view(torch.int32)


def untile_image(torch, words, shape):
    """tile_image's inverse: int32 [pages, 512] -> the bf16 tensor of padded `shape`, bits kept."""
    rows, columns = shape[-2:]
    lead = 1
    for size in shape[:-2]:
        lead *= size
    bits = words.contiguous().view(torch.int16).reshape(lead, rows // 32, columns // 32, 2, 2, 16, 16)
    return bits.permute(0, 1, 3, 5, 2, 4, 6).reshape(shape).contiguous().view(torch.bfloat16)


def words_of(torch, value):
    """A uint32 readback as int32 bit patterns, whatever integer dtype to_torch returns."""
    wide = value.to(torch.int64) & 0xFFFFFFFF
    return torch.where(wide >= 2 ** 31, wide - 2 ** 32, wide).to(torch.int32)


def unwritten_pages(torch, image):
    """How many pages of a padded bf16 TILE image still hold the sentinel in every halfword: pages
    the launch never wrote."""
    return int((tile_image(torch, image) == SENTINEL_WORD).all(dim=1).sum())


class SentinelOperations:
    """`operations` (ttnn) for one launch, except that every tensor the launch allocates is first
    filled with the sentinel page by `fill` - so a page the launch never writes reads back as the
    sentinel instead of whatever a freed buffer held (the previous arm's output, or the readback
    sink's exact copy of C's). Both execute()s take it in their `operations` slot."""

    def __init__(self, operations, fill):
        self._operations, self._fill = operations, fill
        self.filled = 0

    def __getattr__(self, name):
        return getattr(self._operations, name)

    def empty(self, *args, **kwargs):
        value = self._operations.empty(*args, **kwargs)
        try:
            self._fill(value)
        except BaseException:
            self._operations.deallocate(value)
            raise
        self.filled += 1
        return value


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chips', type=int, default=1, choices=(1, 2))
    parser.add_argument('--users', type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--regimes', default=','.join(REGIMES))
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in PLAN_R1_SEEDS), help='R1 seeds')
    parser.add_argument('--other-seeds', default='17', help='R2, R3 and R3b seeds')
    parser.add_argument('--r4-launches', type=int, default=PLAN_R4_LAUNCHES)
    parser.add_argument('--r4-seed', type=int, default=41)
    parser.add_argument('--skip-timing', action='store_true')
    parser.add_argument('--timing-launches', type=int, default=48)
    parser.add_argument('--timing-rounds', type=int, default=25)
    parser.add_argument('--timing-arms', default='C,A,A0')
    parser.add_argument('--diag', default='', help='timing-only builds: nosnap,passthrough')
    parser.add_argument('--output-memory', choices=('l1', 'dram'), default='l1')
    parser.add_argument('--trace-region', type=int, default=134217728, help='bytes; 0 with --skip-timing')
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal')))
    arguments = parser.parse_args(argv)
    arguments.arms = [arm for arm in arguments.arms.split(',') if arm]
    arguments.regimes = [regime for regime in arguments.regimes.split(',') if regime]
    arguments.seeds = [int(seed) for seed in arguments.seeds.split(',') if seed]
    arguments.other_seeds = [int(seed) for seed in arguments.other_seeds.split(',') if seed]
    arguments.timing_arms = [arm for arm in arguments.timing_arms.split(',') if arm]
    arguments.diag = [name for name in arguments.diag.split(',') if name]
    if 'C' not in arguments.arms or any(arm not in ARMS for arm in arguments.arms):
        parser.error('--arms must include C and name only %s' % (ARMS,))
    if any(regime not in REGIMES for regime in arguments.regimes):
        parser.error('--regimes names only %s' % (REGIMES,))
    if any(name not in ('nosnap', 'passthrough') for name in arguments.diag):
        parser.error('--diag names only nosnap, passthrough')
    if not arguments.skip_timing and any(arm not in arguments.arms for arm in arguments.timing_arms):
        parser.error('--timing-arms must be a subset of --arms')
    return arguments


# ---- host-side comparison ----

def bits16(torch, value):
    return value.contiguous().view(torch.int16)


def ordered(torch, bits):
    """bf16 bit patterns (int16) -> integers that are monotone in the value (sign-magnitude)."""
    wide = bits.to(torch.int32)
    return torch.where(wide < 0, -(wide & 0x7FFF), wide)


def locate_output(index, width=3072):
    row, column = divmod(index, width)
    return dict(token=row if row < ROWS else None, padding_row=row >= ROWS, head=column // 128,
                tile=(column % 128) // 32, element=[row % 32, column % 32])


def locate_states(index):
    token, rest = divmod(index, 24 * 128 * 128)
    head, rest = divmod(rest, 128 * 128)
    row, column = divmod(rest, 128)
    return dict(token=token, head=head, tile=(row // 32) * 4 + column // 32, element=[row % 32, column % 32])


def compare(torch, expected, actual, locate):
    """Bitwise. On a mismatch: counts, the first differing element located, a ulp histogram. A shape
    or dtype mismatch is not measured at all (differing_bytes None), which record() counts as a
    failure."""
    if tuple(expected.shape) != tuple(actual.shape) or expected.dtype != actual.dtype:
        return dict(exact=False, differing=None, differing_bytes=None,
                    shape=[list(expected.shape), list(actual.shape)],
                    dtype=[str(expected.dtype), str(actual.dtype)])
    left, right = bits16(torch, expected).reshape(-1), bits16(torch, actual).reshape(-1)
    differs = left != right
    count = int(differs.sum())
    byte_count = int((expected.contiguous().view(torch.uint8) != actual.contiguous().view(torch.uint8)).sum())
    if count == 0:
        return dict(exact=True, differing=0, differing_bytes=0, of=int(left.numel()))
    first = int(differs.nonzero()[0])
    one, two = expected.reshape(-1).float(), actual.reshape(-1).float()
    nan_mismatch = differs & (torch.isnan(one) != torch.isnan(two))
    finite = differs & torch.isfinite(one) & torch.isfinite(two)
    ulps = (ordered(torch, left) - ordered(torch, right)).abs()[finite]
    histogram, lower = {}, 0
    for upper, label in ULP_BUCKETS:
        chosen = ulps > lower if upper is None else (ulps > lower) & (ulps <= upper)
        histogram[label] = int(chosen.sum())
        lower = upper if upper is not None else lower
    return dict(exact=False, differing=count, differing_bytes=byte_count, of=int(left.numel()),
                first=dict(locate(first), expected=float(one[first]), actual=float(two[first]),
                           expected_bits='0x%04x' % (int(left[first]) & 0xFFFF),
                           actual_bits='0x%04x' % (int(right[first]) & 0xFFFF)),
                ulp_histogram=histogram, nan_or_inf_mismatches=int(nan_mismatch.sum()),
                max_abs=float((one - two).abs()[finite].max()) if bool(finite.any()) else None)


# ---- the verdicts (pure: held by test_gdn_seq_block.ProbeVerdictTests) ----

def new_tally():
    return dict(cases=0, exact_cases=0, differing=0, differing_bytes=0, unmeasured=0, first_failure=None)


def record(tally, case, result):
    """Add one case's compare parts (name -> compare() dict) to an arm's tally; returns whether the
    case was exact. A part without a byte count (a shape or dtype mismatch), or a case with no parts
    at all, is unmeasured, and unmeasured is a failure: it must never add 0 differing bytes and pass."""
    parts = [part for part in result.values() if isinstance(part, dict)]
    unmeasured = sum(1 for part in parts if part.get('differing_bytes') is None) + (0 if parts else 1)
    exact = not unmeasured and all(part.get('exact') is True for part in parts)
    tally['cases'] += 1
    tally['exact_cases'] += int(exact)
    tally['unmeasured'] += unmeasured
    tally['differing'] += sum(part.get('differing') or 0 for part in parts)
    tally['differing_bytes'] += sum(part.get('differing_bytes') or 0 for part in parts)
    if not exact and tally['first_failure'] is None:
        tally['first_failure'] = dict(case=case, result=result)
    return exact


def arm_exact(tally):
    """Every case of the arm ran, was measured, and matched C in every byte."""
    return (tally is not None and tally['cases'] > 0 and tally['exact_cases'] == tally['cases']
            and tally['differing_bytes'] == 0 and tally['unmeasured'] == 0)


def detects(tally):
    """The negative control saw a real difference: some differing byte, every part measured."""
    return tally is not None and tally['cases'] > 0 and tally['differing_bytes'] > 0 and tally['unmeasured'] == 0


def plan_coverage(arms, regimes, seeds, users, r4_launches, r4_completed):
    """What is missing from the plan's P0 (section 4): every arm, every regime, the five R1 seeds, 4
    users and all 128 R4 launches, completed. Only a run missing nothing may pass."""
    missing = ['arm %s' % arm for arm in ARMS if arm not in arms]
    missing += ['regime %s' % regime for regime in REGIMES if regime not in regimes]
    if 'R1' in regimes:
        missing += ['R1 seed %d' % seed for seed in PLAN_R1_SEEDS if seed not in seeds]
    if users != PLAN_USERS:
        missing.append('users %d (plan: %d)' % (users, PLAN_USERS))
    if 'R4' in regimes and min(r4_launches, r4_completed) < PLAN_R4_LAUNCHES:
        missing.append('R4 %d of %d launches completed (plan: %d)' % (r4_completed, r4_launches, PLAN_R4_LAUNCHES))
    return dict(complete=not missing, missing=missing)


def p0_verdict(tallies, *, errors, inputs_unchanged, unwritten, coverage):
    """'pass' only when every exact arm that ran (A, A0) matched C in every byte of every case, N
    detected, no input moved, every output page of every arm was written, and the run covered the
    whole plan. The same outcome on partial coverage is 'partial-pass': informative, never a
    licence to commit a QUALIFIED triple. No exact arm at all, or any miss, is 'fail'; a case that
    raised is 'error'."""
    exact = {arm: arm_exact(tallies.get(arm)) for arm in EXACT_ARMS if arm in tallies}
    n_detects = detects(tallies['N']) if 'N' in tallies else None
    missed = not exact or not all(exact.values()) or n_detects is False or not inputs_unchanged or bool(unwritten)
    if errors:
        result = 'error'
    elif missed:
        result = 'fail'
    elif coverage['complete']:
        result = 'pass'
    else:
        result = 'partial-pass'
    return dict(result=result, passed=result == 'pass', may_commit_qualified=result == 'pass', exact=exact,
                n_detects=n_detects, inputs_unchanged=inputs_unchanged, unwritten=list(unwritten),
                errors=list(errors), coverage=coverage)


def traced_status(traced):
    """'not-run', 'error', 'exact' (every traced arm matched C's replay, every page written) or 'differs'."""
    if traced is None:
        return 'not-run'
    if traced.get('error'):
        return 'error'
    if traced.get('exact') and all(traced['exact'].values()) and not traced.get('unwritten'):
        return 'exact'
    return 'differs'


def overall(p0_result, traced):
    """The run's result: P0's, but a traced replay that differs fails it and one that raised makes it
    an error - serving runs K5-A only inside a trace."""
    status = traced_status(traced)
    if p0_result in ('pass', 'partial-pass') and status == 'differs':
        return 'fail'
    if p0_result in ('pass', 'partial-pass') and status == 'error':
        return 'error'
    return p0_result


def p1_verdict(a_us, c_us):
    """P1 (ii) from A's and C's per-launch trace medians. The 297/388 us lines are kernel times set
    against v138's 525.8 us per launch; a trace time also carries dispatch, and each arm runs 48
    launches back to back rather than interleaved as in the model, so its offset from kernel time is
    unknown in either direction (est.). So the lines apply only when C reproduces 525.8 us within
    +-3% ('uncalibrated' otherwise: P1 (i), the profiler check, would settle it), and then A must
    clear each line in all three forms: absolute, A - C (-137.8 us pass, -228.8 us central) and
    A / C (0.738 pass, 0.565 central)."""
    if a_us is None or c_us is None:
        return dict(label='not-run', verdict='not-run', calibrated=None)
    c_ratio = c_us / SERVED_LAUNCH_US
    calibrated = abs(c_ratio - 1) <= CALIBRATION
    delta, ratio = a_us - c_us, a_us / c_us
    central = a_us <= P1_CENTRAL_US and delta <= P1_CENTRAL_DELTA_US and ratio <= P1_CENTRAL_RATIO
    passes = a_us <= P1_PASS_US and delta <= P1_PASS_DELTA_US and ratio <= P1_PASS_RATIO
    if not calibrated:
        label, verdict = 'uncalibrated', ('uncalibrated: C runs %.1f us per launch here, not 525.8 us +-3%%; neither '
                                          'line applies until P1 (i) confirms the kernel time' % c_us)
    elif central:
        label, verdict = 'proceed', 'proceed (on central: A <= 297 us, A - C <= -228.8 us, A / C <= 0.565)'
    elif passes:
        label, verdict = 'image-build', 'image-build (A <= 388 us, A - C <= -137.8 us, A / C <= 0.738)'
    else:
        label, verdict = 'kill', 'kill (over the 388 us pass line in some form): zone-profile A before any in-model arm'
    return dict(label=label, verdict=verdict, calibrated=calibrated, c_over_v138=c_ratio, a_median_us=a_us,
                c_median_us=c_us, a_minus_c_us=delta, a_over_c=ratio)


def serpentine(arms, rounds):
    """The timing replay order: forward on even rounds, reversed on odd ones."""
    return [list(arms) if index % 2 == 0 else list(arms)[::-1] for index in range(rounds)]


# ---- the fp64 reference (information only) ----

def reference(torch, qkv, beta, gate, initial, z, norm_w):
    """One user's sequential recurrence and norm/gate in fp64 from the bf16 inputs, unrounded."""
    d = torch.float64
    x = qkv[0].to(d)
    q = x[:, 0:1024].reshape(ROWS, 8, 128).repeat_interleave(3, dim=1)
    k = x[:, 1024:2048].reshape(ROWS, 8, 128).repeat_interleave(3, dim=1)
    v = x[:, 2048:5120].reshape(ROWS, 24, 128)
    qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    decay = torch.exp(gate[0].to(d))
    strength = beta[0].to(d)
    weight = norm_w[0, 0].to(d)
    gates = z[0].to(d).reshape(ROWS, 24, 128)
    state = initial[0].to(d).clone()
    outputs, states = [], []
    for token in range(ROWS):
        state = state * decay[token][:, None, None]
        read = torch.einsum('hk,hkv->hv', kn[token], state)
        delta = (v[token] - read) * strength[token][:, None]
        state = state + kn[token][:, :, None] * delta[:, None, :]
        o = torch.einsum('hk,hkv->hv', qn[token], state)
        xn = o * torch.rsqrt((o * o).sum(-1, keepdim=True) + 128e-6) * 128 ** 0.5
        outputs.append((xn * weight * torch.nn.functional.silu(gates[token])).reshape(3072))
        states.append(state.clone())
    return torch.stack(outputs)[None], torch.stack(states), decay


def reference_error(torch, output, states, expected_output, expected_states, decay):
    by_token = []
    for token in range(ROWS):
        out_ref, st_ref = expected_output[0, token], expected_states[token]
        out_gap = (output[0, token].double() - out_ref).abs()
        st_gap = (states[token].double() - st_ref).abs()
        by_token.append(dict(token=token, output_max_abs=float(out_gap.max()),
                             output_rel=float(out_gap.max() / out_ref.abs().max().clamp_min(1e-30)),
                             state_max_abs=float(st_gap.max()),
                             state_rel=float(st_gap.max() / st_ref.abs().max().clamp_min(1e-30))))
    classes = {}
    per_head = (states.double() - expected_states).abs().amax(dim=(2, 3))  # (token, head)
    for name, low, high in (('fast a<0.5', 0.0, 0.5), ('mid 0.5<=a<0.9', 0.5, 0.9), ('slow a>=0.9', 0.9, 2.0)):
        chosen = (decay >= low) & (decay < high)
        if bool(chosen.any()):
            classes[name] = dict(count=int(chosen.sum()), state_max_abs=float(per_head[chosen].max()),
                                 state_mean_of_max=float(per_head[chosen].mean()))
    return dict(by_token=by_token, by_decay_class=classes)


# ---- inputs ----

def regime_inputs(torch, regime, seed, users):
    """Host bf16 inputs: (norm_w, [dict(qkv, beta, gate, initial, z) per user], poison per user)."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    norm_w = (1 + randn(1, 1, 128) * 0.1).bfloat16()
    groups, poison = [], []
    for user in range(users):
        if regime in ('R1', 'R3b', 'R4'):
            initial = randn(1, 24, 128, 128) * 0.05
            qkv, beta, gate, z = randn(1, ROWS, 5120), rand(1, ROWS, 24), -rand(1, ROWS, 24), randn(1, ROWS, 3072)
        elif regime == 'R2':
            initial = randn(1, 24, 128, 128) * 4
            qkv, beta, gate, z = randn(1, ROWS, 5120) * 8, rand(1, ROWS, 24), -rand(1, ROWS, 24) * 20, randn(1, ROWS, 3072)
        elif regime == 'R3':
            initial = randn(1, 24, 128, 128) * 0.05
            flat = initial.view(-1)
            specials = (0.0, -0.0, 2.0 ** -126, -2.0 ** -126, 2.0 ** -130, -2.0 ** -133)
            chosen = torch.randperm(flat.numel(), generator=generator)[:len(specials) * 2048]
            for index, value in enumerate(specials):
                flat[chosen[index * 2048:(index + 1) * 2048]] = value
            qkv, z = randn(1, ROWS, 5120), randn(1, ROWS, 3072)
            qkv[0, 3, 1024:2048] = 0.0   # an all-zero k row: the k-norm eps path
            qkv[0, 7, 0:1024] = 0.0      # and an all-zero q row: the q-norm eps path
            beta = torch.where(rand(1, ROWS, 24) < 0.5, torch.tensor(0.0), torch.tensor(1 - 2.0 ** -8))
            gate = torch.where(rand(1, ROWS, 24) < 0.5, torch.tensor(0.0), torch.tensor(-88.0))
        else:
            raise ValueError('Unknown regime %r' % (regime,))
        groups.append(dict(qkv=qkv.bfloat16(), beta=beta.bfloat16(), gate=gate.bfloat16(),
                           initial=initial.bfloat16(), z=z.bfloat16()))
        poison.append((float('nan'), float('nan'), float('inf'), float('-inf'))[user % 4] if regime == 'R3b' else None)
    return norm_w, groups, poison


def summarize_timing(samples):
    ordered_samples = sorted(samples)
    quartiles = statistics.quantiles(ordered_samples, n=4) if len(ordered_samples) >= 2 else [ordered_samples[0]] * 3
    return dict(median_us=statistics.median(ordered_samples), p25_us=quartiles[0], p75_us=quartiles[2],
                iqr_us=quartiles[2] - quartiles[0], min_us=ordered_samples[0], max_us=ordered_samples[-1],
                replays=len(ordered_samples))


def file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return None


def tt_metal_commit(root):
    """The tt-metal checkout's HEAD commit, read from its .git without a git binary, or None."""
    try:
        git = Path(root) / '.git'
        if git.is_file():  # 'gitdir: <path>'
            git = Path(git.read_text().split(':', 1)[1].strip())
        head = (git / 'HEAD').read_text().strip()
        if not head.startswith('ref: '):
            return head
        ref = head[len('ref: '):]
        if (git / ref).is_file():
            return (git / ref).read_text().strip()
        for line in (git / 'packed-refs').read_text().splitlines():
            if line.endswith(' ' + ref):
                return line.split()[0]
    except (OSError, IndexError):
        return None
    return None


def main():
    arguments = parse()
    here = Path(__file__).resolve().parent
    report = dict(scope='Uncertified card-B probe of K5-A (gdn_seq_block) against the served batched GDN launch '
                        '(gdn_user_batch); no model, no projection, no collective',
                  users=arguments.users, rows=ROWS, chips=arguments.chips, arms=arguments.arms,
                  regimes=arguments.regimes, seeds=arguments.seeds, other_seeds=arguments.other_seeds,
                  r4_launches=arguments.r4_launches,
                  native_sha256=native.HASHES,
                  imported_from={module.__name__: getattr(module, '__file__', None)
                                 for module in (native, seq, batch, verify_trace_t1)},
                  native_module_sha256=file_sha256(getattr(native, '__file__', None)),
                  module_sha256={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                 for name in MODULES if (here / name).exists()},
                  tt_metal_root=str(arguments.root), tt_metal_commit=tt_metal_commit(arguments.root),
                  tt_metal_home=os.environ.get('TT_METAL_HOME'),
                  tt_metal_home_commit=tt_metal_commit(os.environ.get('TT_METAL_HOME') or arguments.root),
                  verify_t1=verify_trace_t1.enabled(), coalesce=verify_trace_t1.cut('coalesce'),
                  result='did-not-run', stages=[], cases=[], r4=None, traced=None, timings=None, unwritten=[])
    summary = dict(kind='gdn-seq-block-probe', result='error')

    def stage(name, **details):
        report['stages'].append(dict(stage=name, **details))
        arguments.out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report['stages'][-1]), flush=True)

    mesh = None
    try:
        stage('import-runtime')
        import torch
        import ttnn
        report['ttnn_path'] = ttnn.__file__
        report['ttnn_version'] = getattr(ttnn, '__version__', None)
        if not report['coalesce']:
            raise AssertionError('QWEN_FAST_VERIFY_T1=1 with the coalesce cut is required: control C must be the '
                                 'coalesced served build the arms run (verify_trace_t1 #12)')

        stage('load-kernels', root=str(arguments.root))
        control = batch.load_kernels(arguments.root)
        control_sha = {role: hashlib.sha256(source.encode()).hexdigest() for role, source in control.items()}
        report['control_sha256'] = control_sha
        report['control_is_served'] = control_sha == SERVED_SHA256
        if not report['control_is_served']:
            raise AssertionError('Control C is not the served build: %s' % control_sha)
        builds = {}
        for arm in arguments.arms:
            if arm != 'C':
                builds[arm] = seq.load_kernels(arguments.root, 0, variant=arm, unqualified=True)
        for name in arguments.diag:
            builds['A-' + name] = seq.load_kernels(arguments.root, 0, variant='A', diag=name, unqualified=True)
        report['generated_sha256'] = {arm: seq.sha256(kernels) for arm, kernels in builds.items()}
        report['cb_bytes_per_core'] = dict({arm: seq.cb_bytes(kernels.variant) for arm, kernels in builds.items()},
                                           C=seq.SERVED_CB_BYTES)

        stage('mesh-open')
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, arguments.chips), l1_small_size=24576,
                                     trace_region_size=0 if arguments.skip_timing else arguments.trace_region)
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        report['core_shares'] = batch.core_shares(grid.x, grid.y, arguments.users)
        output_memory = ttnn.L1_MEMORY_CONFIG if arguments.output_memory == 'l1' else ttnn.DRAM_MEMORY_CONFIG

        def upload(value, memory=None):
            return ttnn.from_torch(value, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   memory_config=memory or ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        def raw_copy(source, destination, pages):
            workers = min(RAW_WORKERS, pages)
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(workers - 1, 0))])
            scratch = ttnn.CBDescriptor(total_size=2048, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048,
                                        tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
            program = ttnn.MeshProgramDescriptor()
            pairs = zip(ttnn.get_device_tensors(source), ttnn.get_device_tensors(destination), strict=True)
            for chip, (left, right) in enumerate(pairs):
                args = []
                for value in (left, right):
                    args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
                runtime = ttnn.RuntimeArgs()
                for worker in range(workers):
                    runtime[worker][0] = [left.buffer_address(), right.buffer_address(), pages, worker, workers]
                kernel = ttnn.KernelDescriptor(kernel_source=RAW_COPY,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                    compile_time_args=args, config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default))
                kernel.runtime_args = runtime
                coordinate = ttnn.MeshCoordinate(0, chip)
                program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
                    kernels=[kernel], cbs=[scratch])
            ttnn.generic_op([source, destination], program)

        def words_tensor(words):
            return ttnn.from_torch(words.reshape(1, 1, -1, 512), device=mesh, dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        # The sentinel page image, MAX_PAGES pages: the source every sentinel fill and scrub copies from.
        sentinel = words_tensor(torch.full((MAX_PAGES, 512), SENTINEL_WORD, dtype=torch.int32))

        def fill_sentinel(value):
            pages = page_count(value.shape)
            if pages > MAX_PAGES:
                raise ValueError('A launch allocated %d pages; the sentinel holds %d' % (pages, MAX_PAGES))
            raw_copy(sentinel, value, pages)

        def upload_exact(logical, image=None):
            """A BF16 TILE DRAM tensor of `logical`'s shape holding exactly `image`'s bytes (default:
            `logical` with zero padding), padding included."""
            image = pad_image(torch, logical) if image is None else image
            if tuple(image.shape) != padded_shape(logical.shape):
                raise ValueError('Image is not the padded shape of the logical tensor')
            target = ttnn.empty(tuple(logical.shape), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            words = tile_image(torch, image)
            source = words_tensor(words)
            try:
                raw_copy(source, target, words.shape[0])
                ttnn.synchronize_device(mesh)
            finally:
                ttnn.deallocate(source)
            return target

        def raw_host(value):
            """Every physical byte of a BF16 TILE tensor, per chip, as a bf16 tensor of its padded shape.
            The sink is overwritten with the sentinel before it is freed: it holds an exact copy of
            what it read, and the next launch may be handed its memory."""
            shape = padded_shape(value.shape)
            pages = page_count(value.shape)
            sink = words_tensor(torch.zeros(pages, 512, dtype=torch.int32))
            try:
                raw_copy(value, sink, pages)
                ttnn.synchronize_device(mesh)
                shards = ttnn.get_device_tensors(sink)
                if len(shards) != arguments.chips:
                    raise AssertionError('Expected %d chips, got %d' % (arguments.chips, len(shards)))
                images = [untile_image(torch, words_of(torch, ttnn.to_torch(shard)).reshape(pages, 512), shape)
                          for shard in shards]
                raw_copy(sentinel, sink, pages)
                return images
            finally:
                ttnn.deallocate(sink)

        def logical(images, shape):
            return [image[tuple(slice(0, size) for size in shape)].contiguous() for image in images]

        def release(produced):
            for output, states in produced or ():
                ttnn.deallocate(output)
                ttnn.deallocate(states)

        def run(arm, groups, operations=None):
            operations = ttnn if operations is None else operations
            if arm == 'C':
                return batch.execute(mesh, groups, control, operations, output_memory=output_memory)
            return seq.execute(mesh, groups, operations, output_memory=output_memory, kernels=builds[arm])

        def coalesced_once(arm, where):
            """The arm's launch was the coalesced build (one descriptor per role, verify_trace_t1 #12)."""
            counts = verify_trace_t1.take()
            where.setdefault('verify_t1_counts', {})[arm] = counts
            if counts != {'coalesced': 1}:
                raise AssertionError('Arm %s did not run as one coalesced launch: verify_trace_t1 counts %s'
                                     % (arm, counts))

        def read_back(arm, case, produced, unwritten):
            """(logical output, padded output, states) host images per user, per chip; every page
            still holding the sentinel is appended to `unwritten`."""
            images = []
            for user, (output, states) in enumerate(produced):
                padded, snapshots = raw_host(output), raw_host(states)
                for chip in range(arguments.chips):
                    for name, image in (('output', padded[chip]), ('states', snapshots[chip])):
                        pages = unwritten_pages(torch, image)
                        if pages:
                            unwritten.append(dict(arm=arm, case=case, user=user, chip=chip, tensor=name, pages=pages))
                images.append((logical(padded, output.shape), padded, snapshots))
            return images

        def moved_inputs(checks):
            """[(label, device tensor, host image)] -> the labels whose device bytes differ from the image."""
            moved = []
            for label, tensor, image in checks:
                for chip, shard in enumerate(raw_host(tensor)):
                    if not torch.equal(bits16(torch, shard), bits16(torch, image)):
                        moved.append(dict(label, chip=chip))
            return moved

        tallies = {arm: new_tally() for arm in arguments.arms if arm != 'C'}
        inputs_unchanged = True
        sentinel_operations = SentinelOperations(ttnn, fill_sentinel)
        verify_trace_t1.take()

        def compare_arm(expected, actual, users):
            """(output, padded output, states) host triples per user -> a per-user compare dict."""
            per_user = []
            for user in range(users):
                parts = {}
                for chip in range(arguments.chips):
                    suffix = '' if arguments.chips == 1 else '_chip%d' % chip
                    parts['output' + suffix] = compare(torch, expected[user][0][chip], actual[user][0][chip],
                                                       locate_output)
                    parts['output_padded' + suffix] = compare(torch, expected[user][1][chip], actual[user][1][chip],
                                                              locate_output)
                    parts['states' + suffix] = compare(torch, expected[user][2][chip], actual[user][2][chip],
                                                       locate_states)
                out_bad = any(not value['exact'] for name, value in parts.items() if name.startswith('output'))
                st_bad = any(not value['exact'] for name, value in parts.items() if name.startswith('states'))
                parts['where'] = 'both' if out_bad and st_bad else 'output-only' if out_bad else \
                    'states-only' if st_bad else 'none'
                per_user.append(parts)
            return per_user

        def merged_parts(per_user):
            return {'user%d_%s' % (user, name): value for user, parts in enumerate(per_user)
                    for name, value in parts.items() if name != 'where'}

        # ---------------- P0: regimes R1, R2, R3, R3b ----------------
        cases = [(regime, seed) for regime in arguments.regimes if regime != 'R4'
                 for seed in (arguments.seeds if regime == 'R1' else arguments.other_seeds)]
        for regime, seed in cases:
            case = '%s/seed%d' % (regime, seed)
            stage('p0-case', case=case)
            entry = dict(case=case, arms={}, error=None)
            report['cases'].append(entry)
            device_inputs = []
            try:
                norm_w, users_host, poison = regime_inputs(torch, regime, seed, arguments.users)
                norm_w_device = upload_exact(norm_w)
                device_inputs.append(norm_w_device)
                checks = [(dict(user=None, tensor='norm_w'), norm_w_device, pad_image(torch, norm_w))]
                groups = []
                for user, values in enumerate(users_host):
                    tensors = []
                    for name in ('qkv', 'beta', 'gate', 'initial', 'z'):
                        fill = 0.0 if poison[user] is None or name == 'initial' else poison[user]
                        image = pad_image(torch, values[name], fill)
                        tensors.append(upload_exact(values[name], image))
                        device_inputs.append(tensors[-1])
                        checks.append((dict(user=user, tensor=name), tensors[-1], image))
                    groups.append(tuple(tensors) + (norm_w_device,))
                # Every input byte on the device is the host image, padding (and R3b's poison) included.
                landed = moved_inputs(checks)
                if landed:
                    raise AssertionError('Inputs did not land byte for byte: %s' % landed)
                results = {}
                for arm in arguments.arms:
                    produced = run(arm, groups, sentinel_operations)
                    try:
                        ttnn.synchronize_device(mesh)
                        coalesced_once(arm, entry)
                        results[arm] = read_back(arm, case, produced, report['unwritten'])
                    finally:
                        release(produced)
                for arm in arguments.arms:
                    if arm == 'C':
                        continue
                    per_user = compare_arm(results['C'], results[arm], arguments.users)
                    merged = merged_parts(per_user)
                    exact = record(tallies[arm], case, merged)
                    entry['arms'][arm] = dict(exact=exact, where=[parts['where'] for parts in per_user],
                                              failures={name: value for name, value in merged.items()
                                                        if not value['exact']})
                # The carried state and every input are unchanged: every byte, padding included.
                moved = moved_inputs(checks)
                entry['inputs_unchanged'] = not moved
                entry['inputs_moved'] = moved
                inputs_unchanged = inputs_unchanged and not moved
                # Information only: C against a sequential fp64 reference.
                entry['reference_error_C'] = [
                    reference_error(torch, results['C'][user][0][0], results['C'][user][2][0],
                                    *reference(torch, values['qkv'], values['beta'], values['gate'],
                                               values['initial'], values['z'], norm_w))
                    for user, values in enumerate(users_host)]
            except BaseException as error:
                entry['error'] = repr(error)
                entry['traceback'] = traceback.format_exc()
                print(entry['traceback'], flush=True)
            finally:
                for value in device_inputs:
                    ttnn.deallocate(value)
            stage('p0-case-done', case=case, error=entry['error'],
                  exact={arm: value['exact'] for arm, value in entry['arms'].items()})

        # ---------------- P0: R4, the 2,048-token chain ----------------
        if 'R4' in arguments.regimes:
            r4_arms = [arm for arm in arguments.arms if arm != 'N']
            r4 = dict(launches=arguments.r4_launches, seed=arguments.r4_seed, arms=r4_arms, completed=0,
                      exact={arm: True for arm in r4_arms if arm != 'C'}, first_failure={}, error=None,
                      inputs_checked_at=[], inputs_moved=[])
            report['r4'] = r4
            stage('r4', launches=arguments.r4_launches)
            device_inputs = []
            try:
                generator = torch.Generator().manual_seed(arguments.r4_seed)
                norm_w = (1 + torch.randn(1, 1, 128, generator=generator) * 0.1).bfloat16()
                norm_w_image = pad_image(torch, norm_w)
                norm_w_device = upload_exact(norm_w, norm_w_image)
                device_inputs.append(norm_w_device)
                start = [(torch.randn(1, 24, 128, 128, generator=generator) * 0.05).bfloat16()
                         for unused in range(arguments.users)]
                carries = {arm: list(start) for arm in r4_arms}
                for launch in range(arguments.r4_launches):
                    case = 'R4/launch%d' % launch
                    check = launch == 0 or launch % 16 == 15 or launch == arguments.r4_launches - 1
                    step = []
                    for unused in range(arguments.users):
                        step.append(dict(qkv=torch.randn(1, ROWS, 5120, generator=generator).bfloat16(),
                                         beta=torch.rand(1, ROWS, 24, generator=generator).bfloat16(),
                                         gate=(-torch.rand(1, ROWS, 24, generator=generator)).bfloat16(),
                                         z=torch.randn(1, ROWS, 3072, generator=generator).bfloat16()))
                    names = ('qkv', 'beta', 'gate', 'z')
                    images = [{name: pad_image(torch, values[name]) for name in names} for values in step]
                    shared = []
                    results, moved = {}, []
                    try:
                        for values, image in zip(step, images, strict=True):
                            shared.append(tuple(upload_exact(values[name], image[name]) for name in names))
                        for arm in r4_arms:
                            initial_images = [pad_image(torch, carries[arm][user]) for user in range(arguments.users)]
                            initials = []
                            try:
                                for user in range(arguments.users):
                                    initials.append(upload_exact(carries[arm][user], initial_images[user]))
                                groups = [(qkv, beta, gate, initials[user], z, norm_w_device)
                                          for user, (qkv, beta, gate, z) in enumerate(shared)]
                                produced = run(arm, groups, sentinel_operations)
                                try:
                                    ttnn.synchronize_device(mesh)
                                    coalesced_once(arm, r4)
                                    results[arm] = read_back(arm, case, produced, report['unwritten'])
                                finally:
                                    release(produced)
                                if check:
                                    moved.extend(moved_inputs([(dict(launch=launch, arm=arm, user=user, tensor='initial'),
                                                                initials[user], initial_images[user])
                                                               for user in range(arguments.users)]))
                            finally:
                                for value in initials:
                                    ttnn.deallocate(value)
                            carries[arm] = [results[arm][user][2][0][ROWS - 1:ROWS].clone()
                                            for user in range(arguments.users)]
                        if check:
                            checks = [(dict(launch=launch, user=None, tensor='norm_w'), norm_w_device, norm_w_image)]
                            checks += [(dict(launch=launch, user=user, tensor=name), tensors[index], images[user][name])
                                       for user, tensors in enumerate(shared) for index, name in enumerate(names)]
                            moved.extend(moved_inputs(checks))
                            r4['inputs_checked_at'].append(launch)
                    finally:
                        for values in shared:
                            for value in values:
                                ttnn.deallocate(value)
                    if moved:
                        r4['inputs_moved'].extend(moved)
                        inputs_unchanged = False
                    for arm in r4_arms:
                        if arm == 'C':
                            continue
                        per_user = compare_arm(results['C'], results[arm], arguments.users)
                        merged = merged_parts(per_user)
                        if not record(tallies[arm], case, merged) and r4['exact'][arm]:
                            r4['exact'][arm] = False
                            r4['first_failure'][arm] = dict(launch=launch, where=[p['where'] for p in per_user],
                                                            failures={name: value for name, value in merged.items()
                                                                      if not value['exact']})
                    r4['completed'] = launch + 1
                    if launch % 16 == 15:
                        stage('r4-progress', completed=launch + 1, exact=r4['exact'])
            except BaseException as error:
                r4['error'] = repr(error)
                r4['traceback'] = traceback.format_exc()
                print(r4['traceback'], flush=True)
            finally:
                for value in device_inputs:
                    ttnn.deallocate(value)

        # ---------------- traced: one launch per arm, replayed, compared ----------------
        traced_arms = [arm for arm in ('C',) + EXACT_ARMS if arm in arguments.arms]
        if not arguments.skip_timing and len(traced_arms) > 1:
            traced = dict(arms=traced_arms, exact={}, failures={}, unwritten=[], error=None)
            report['traced'] = traced
            stage('traced-compare', arms=traced_arms)
            owned = []
            try:
                norm_w, users_host, unused = regime_inputs(torch, 'R1', 999, arguments.users)
                norm_w_device = upload_exact(norm_w)
                owned.append(norm_w_device)
                groups = []
                for values in users_host:
                    tensors = tuple(upload_exact(values[name]) for name in ('qkv', 'beta', 'gate', 'initial', 'z'))
                    owned.extend(tensors)
                    groups.append(tensors + (norm_w_device,))
                results = {}
                for arm in traced_arms:
                    release(run(arm, groups, sentinel_operations))  # compiled outside the capture
                    ttnn.synchronize_device(mesh)
                    verify_trace_t1.take()
                    produced = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        try:
                            # The sentinel fills are captured too: the replay refills before it runs.
                            produced = run(arm, groups, sentinel_operations)
                        finally:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        coalesced_once(arm, traced)
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        results[arm] = read_back(arm, 'traced', produced, traced['unwritten'])
                    finally:
                        ttnn.release_trace(mesh, trace)
                        release(produced)
                for arm in traced_arms[1:]:
                    merged = merged_parts(compare_arm(results['C'], results[arm], arguments.users))
                    traced['exact'][arm] = record(new_tally(), 'traced', merged)
                    traced['failures'][arm] = {name: value for name, value in merged.items() if not value['exact']}
            except BaseException as error:
                traced['error'] = repr(error)
                traced['traceback'] = traceback.format_exc()
                print(traced['traceback'], flush=True)
            finally:
                for value in owned:
                    ttnn.deallocate(value)
            stage('traced-compare-done', status=traced_status(traced))

        # ---------------- P1 (ii)/(iii): trace timing ----------------
        timing_arms = list(arguments.timing_arms) + ['A-' + name for name in arguments.diag]
        if not arguments.skip_timing and timing_arms:
            stage('timing-setup', arms=timing_arms, launches=arguments.timing_launches,
                  rounds=arguments.timing_rounds)
            orders = serpentine(timing_arms, arguments.timing_rounds)
            timings = dict(arms=timing_arms, launches=arguments.timing_launches, rounds=arguments.timing_rounds,
                           order='serpentine: %s, then reversed, alternating' % ','.join(timing_arms),
                           orders=[','.join(order) for order in orders], per_launch={}, verify_t1_counts={},
                           error=None)
            report['timings'] = timings
            sets, traces, owned = [], {}, []
            try:
                for index in range(arguments.timing_launches):
                    norm_w, users_host, unused = regime_inputs(torch, 'R1', 1000 + index, arguments.users)
                    norm_w_device = upload(norm_w)
                    owned.append(norm_w_device)
                    groups = []
                    for values in users_host:
                        tensors = tuple(upload(values[name]) for name in ('qkv', 'beta', 'gate', 'initial', 'z'))
                        owned.extend(tensors)
                        groups.append(tensors + (norm_w_device,))
                    sets.append(groups)
                for arm in timing_arms:
                    stage('timing-capture', arm=arm)
                    release(run(arm, sets[0]))
                    ttnn.synchronize_device(mesh)
                    verify_trace_t1.take()
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        for groups in sets:
                            # Freed inside the capture: the next launch reuses the same holes, so
                            # the trace's footprint is one launch's outputs whatever its length.
                            release(run(arm, groups))
                    finally:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    traces[arm] = trace
                    counts = verify_trace_t1.take()
                    timings['verify_t1_counts'][arm] = counts
                    if counts != {'coalesced': arguments.timing_launches}:
                        raise AssertionError('Timing arm %s: expected %d coalesced launches, verify_trace_t1 '
                                             'counts %s' % (arm, arguments.timing_launches, counts))
                for arm in timing_arms:
                    for unused in range(2):
                        ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                samples = {arm: [] for arm in timing_arms}
                stage('timing-replay')
                for order in orders:
                    for arm in order:
                        begin = time.perf_counter()
                        ttnn.execute_trace(mesh, traces[arm], cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples[arm].append((time.perf_counter() - begin) * 1e6 / arguments.timing_launches)
                for arm in timing_arms:
                    timings['per_launch'][arm] = summarize_timing(samples[arm])
                timings['samples_us'] = samples
                control_median = timings['per_launch'].get('C', {}).get('median_us')
                if control_median:
                    timings['c_vs_v138_ratio'] = control_median / SERVED_LAUNCH_US
                    for arm in timing_arms:
                        if arm != 'C':
                            delta = timings['per_launch'][arm]['median_us'] - control_median
                            timings['per_launch'][arm]['minus_c_us'] = delta
                            timings['per_launch'][arm]['verify_ms_est'] = 48 * delta / 1000
            except BaseException as error:
                timings['error'] = repr(error)
                timings['traceback'] = traceback.format_exc()
                print(timings['traceback'], flush=True)
            finally:
                for trace in traces.values():
                    ttnn.release_trace(mesh, trace)
                for value in owned:
                    ttnn.deallocate(value)
                ttnn.synchronize_device(mesh)

        # ---------------- verdict ----------------
        errors = [entry['case'] for entry in report['cases'] if entry['error']]
        if report['r4'] and report['r4']['error']:
            errors.append('R4')
        r4_completed = report['r4']['completed'] if report['r4'] else 0
        coverage = plan_coverage(arguments.arms, arguments.regimes, arguments.seeds, arguments.users,
                                 arguments.r4_launches, r4_completed)
        p0 = p0_verdict(tallies, errors=errors, inputs_unchanged=inputs_unchanged, unwritten=report['unwritten'],
                        coverage=coverage)
        report['p0'] = dict(p0, arms=tallies)
        report['result'] = overall(p0['result'], report['traced'])
        per_launch = (report['timings'] or {}).get('per_launch') or {}
        report['p1'] = p1_verdict((per_launch.get('A') or {}).get('median_us'), (per_launch.get('C') or {}).get('median_us'))
        stage('done', result=report['result'], p0=p0['result'], traced=traced_status(report['traced']),
              p1=report['p1']['label'])
        summary.update(result=report['result'],
                       p0=dict(result=p0['result'], passed=p0['passed'], may_commit_qualified=p0['may_commit_qualified'],
                               missing=coverage['missing'], errors=errors, exact=p0['exact'],
                               n_detects=p0['n_detects'], inputs_unchanged=inputs_unchanged,
                               unwritten_pages=sum(item['pages'] for item in report['unwritten']),
                               differing_bytes={arm: value['differing_bytes'] for arm, value in tallies.items()},
                               cases={arm: [value['exact_cases'], value['cases']] for arm, value in tallies.items()}),
                       traced=traced_status(report['traced']),
                       p1=dict(report['p1'], per_launch={arm: dict(median_us=value['median_us'], iqr_us=value['iqr_us'],
                                                                     minus_c_us=value.get('minus_c_us'))
                                                         for arm, value in per_launch.items()},
                               error=(report['timings'] or {}).get('error')),
                       users=arguments.users, chips=arguments.chips, grid=report['grid'],
                       cb_bytes_per_core=report['cb_bytes_per_core'], generated_sha256=report['generated_sha256'],
                       control_is_served=report['control_is_served'], verify_t1=report['verify_t1'],
                       report=str(arguments.out))
        return 0 if report['result'] == 'pass' else 2 if report['result'] == 'error' else 1
    except BaseException as error:
        report['result'] = 'error'
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        summary.update(result='error', error=repr(error)[:400])
        print(report['traceback'], flush=True)
        return 2
    finally:
        arguments.out.write_text(json.dumps(report, indent=2))
        if mesh is not None:
            import ttnn
            ttnn.close_mesh_device(mesh)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
