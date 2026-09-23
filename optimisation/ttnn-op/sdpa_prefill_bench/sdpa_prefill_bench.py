"""M1 of the prefill ranking: is the chunked-SDPA quadratic term bytes-bound or compute-bound?

The served C1 prefill spends ~22.7 s of each 131,072-token prompt in the chunked SDPA quadratic
term (16 full-attention layers, 64 chunks). Lever #1 (K/V prefix sharing in the chunked SDPA
reader) is worth 7-14 s only if that term is limited by the bytes each core reads; if it is
limited by compute, #1 is worth ~0 and the compute knobs (#1b) are the lever instead. This bench
answers that on ONE card with no model: it times the served call alone, then moves one knob at a
time and fits the slope of the time against the prefix length (chunk_start).

THE SERVED CALL (grafted attention/tp.py, forward_prefill_paged, run 35816715775):
    ttnn.transformer.chunked_scaled_dot_product_attention(
        input_tensor_q=q8, input_tensor_k=k_paged, input_tensor_v=v_paged,
        page_table_tensor=sdpa_page_table, chunk_start_idx_tensor=chunk_start_idx_tensor,
        compute_kernel_config=tpc.COMPUTE_HIFI2, program_config=SDPAProgramConfig(
            compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),   # 11 x 10
            exp_approx_mode=False, q_chunk_size=128, k_chunk_size=128))
  - Q (1, 12, 2048, 256): 12 local heads at TP2, typecast to bf8 under QWEN_SDPA_BF8=1 (the arm
    hard-sets it), so the served Q and paged K/V are all bf8.
  - K/V pools (blocks, 2, 64, 256): 2 local KV heads, 64-token pages (GQA 6:1).
  - The page table is padded to a multiple of 32 blocks (extra blocks masked by causality).
  - COMPUTE_HIFI2 = HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True.
  - The FLEXIBLE path: chunk_start comes from a device int32 tensor, so one program serves every
    chunk_start (q/k chunk fixed at 128).

ARMS (one knob each against the baseline) and how to read them (ranking, section 3, M1):
  baseline     q128/k128, bf8 Q and KV                        the served call
  bf16_kv      bf16 K/V (Q bf8)                               bytes x 1.88, compute unchanged
  bf16_qkv     bf16 Q and K/V (the non-bf8 serving mode)      fallback if the op wants one dtype
  exp_approx   exp_approx_mode=True                           compute only
  fp32_off     fp32_dest_acc_en=False                         compute only
  q256_2048    q256/k128 at 2048 rows   pre-registered: no gain or a regression (48 busy cores:
                                        causal chunked SDPA hands out Q chunks in PAIRS)
  q256_4096    q256/k128 at 4096 rows   96 busy cores again, and half the K/V bytes per token
  - bytes-bound:   the bf16/bf8 slope ratio follows 1.88 and q256_4096 roughly halves the
                   per-token slope -> build lever #1.
  - compute-bound: exp_approx (or fp32_off) moves the slope and the bytes barely matter ->
                   #1 is worth ~0; take #1b through a token gate.
The slope is ms per 1k keys of prefix, per call; for the 4096-row arm it is also given per 2048
query rows (per token), which is the figure the reading rules compare.

Timing: every (arm, chunk_start) pair is warmed up (program compile), then the pairs are timed
in interleaved rounds (arm-major order rotated per round) so card load drifts spread over every
arm alike; each sample is one call bracketed by ttnn.synchronize_device. The median per pair is
reported. Single device, no CCL.

Every ttnn/torch import is local to the device functions: the helpers import and unit-test under
plain CPython (test_sdpa_prefill_bench.py).
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

NH = 12            # local Q heads at TP2
NKV = 2            # local KV heads at TP2
HD = 256
BLOCK = 64         # page size
ROWS = 2048        # prefill chunk
GRID = (11, 10)    # Blackhole p150 compute_with_storage_grid_size
STARTS = (0, 32768, 65536, 129024)
BF8_BYTES_PER_ELEMENT = 1088 / 1024   # bfloat8_b tile: 1024 mantissa bytes + 64 exponent bytes
BF16_BYTES_PER_ELEMENT = 2.0
EXPECTED_BF16_RATIO = BF16_BYTES_PER_ELEMENT / BF8_BYTES_PER_ELEMENT   # 1.882

# Reading-rule thresholds (the ranking gives the directions; these are the cut points).
BYTES_BF16_RATIO_MIN = 1.5      # bf16/bf8 slope ratio at or above this: bytes move the slope
BYTES_Q256_4096_MAX = 0.65      # per-token slope ratio at or below this: halving bytes/token pays
COMPUTE_BF16_RATIO_MAX = 1.15   # bf16/bf8 ratio at or below this: bytes barely matter
COMPUTE_KNOB_MAX = 0.90         # exp_approx or fp32_off at or below this: compute moves the slope

ARMS = (
    dict(name='baseline', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=False, fp32_dest=True),
    dict(name='bf16_kv', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8', kv_dtype='bf16',
         exp_approx=False, fp32_dest=True),
    dict(name='bf16_qkv', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf16', kv_dtype='bf16',
         exp_approx=False, fp32_dest=True),
    dict(name='exp_approx', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=True, fp32_dest=True),
    dict(name='fp32_off', rows=2048, q_chunk=128, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=False, fp32_dest=False),
    dict(name='q256_2048', rows=2048, q_chunk=256, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=False, fp32_dest=True),
    dict(name='q256_4096', rows=4096, q_chunk=256, k_chunk=128, q_dtype='bf8', kv_dtype='bf8',
         exp_approx=False, fp32_dest=True),
)
ARM_NAMES = tuple(arm['name'] for arm in ARMS)


def arm_by_name(name):
    for arm in ARMS:
        if arm['name'] == name:
            return dict(arm)
    raise ValueError('unknown arm %r (known: %s)' % (name, ', '.join(ARM_NAMES)))


def validate(arm, starts, block=BLOCK):
    """The factory's preconditions this bench relies on: chunk_start a multiple of q_chunk (the
    flexible path's contract) and of the page size, rows a whole number of q chunks."""
    if arm['rows'] % arm['q_chunk'] or arm['rows'] % 32:
        raise ValueError('%s: rows %d not a multiple of q_chunk %d' % (arm['name'], arm['rows'], arm['q_chunk']))
    for start in starts:
        if type(start) is not int or start < 0 or start % arm['q_chunk'] or start % block:
            raise ValueError('%s: chunk_start %r must be a non-negative multiple of %d and %d'
                             % (arm['name'], start, arm['q_chunk'], block))


def blocks_for(tokens, block=BLOCK):
    """Pages covering `tokens`, padded to a multiple of 32 blocks as forward_prefill_paged pads."""
    needed = -(-tokens // block)
    return -(-needed // 32) * 32


def pool_blocks(arms, starts, block=BLOCK):
    """One K/V pool and one page table serve every arm: sized for the largest start + rows."""
    return blocks_for(max(starts) + max(arm['rows'] for arm in arms), block)


def q_shape(arm):
    return (1, NH, arm['rows'], HD)


def kv_shape(blocks):
    return (blocks, NKV, BLOCK, HD)


def busy_cores(arm, cores=GRID[0] * GRID[1], heads=NH):
    """Busy cores for causal chunked SDPA: Q chunks are handed out in pairs when a head has an
    even number of them (sdpa_program_factory.cpp:391-405 in the TT-Sim tree), so the work units
    are heads x q_chunks / 2. q128@2048 -> 96, q256@2048 -> 48, q256@4096 -> 96."""
    q_chunks = arm['rows'] // arm['q_chunk']
    units = heads * q_chunks // 2 if q_chunks % 2 == 0 else heads * q_chunks
    return min(units, cores)


def kv_bytes(arm, start):
    """K+V bytes the causal chunked SDPA reads for one call if nothing is shared: every Q head's
    every Q chunk reads the keys up to its own last row (prefix + causal part)."""
    per_element = BF16_BYTES_PER_ELEMENT if arm['kv_dtype'] == 'bf16' else BF8_BYTES_PER_ELEMENT
    q_chunks = arm['rows'] // arm['q_chunk']
    keys = sum(start + (index + 1) * arm['q_chunk'] for index in range(q_chunks))
    return NH * keys * HD * 2 * per_element


def fit(points):
    """Least-squares line through (chunk_start, ms): slope in ms per 1k keys, intercept in ms."""
    points = [(x, y) for x, y in points if y is not None and math.isfinite(y)]
    if len(points) < 2 or len({x for x, _ in points}) < 2:
        return None
    xs = [x / 1000.0 for x, _ in points]
    ys = [y for _, y in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    return dict(slope_ms_per_1k_keys=slope, intercept_ms=mean_y - slope * mean_x, points=len(points))


def slopes(results):
    """{arm: fit + per-token slope} from {arm: {start: median_ms or None}}."""
    table = {}
    for name, by_start in results.items():
        line = fit(sorted((int(start), ms) for start, ms in by_start.items()))
        if line is None:
            table[name] = None
            continue
        rows = arm_by_name(name)['rows']
        line['slope_ms_per_1k_keys_per_2048_rows'] = line['slope_ms_per_1k_keys'] * ROWS / rows
        table[name] = line
    return table


def _ratio(table, name, base='baseline', per_token=False):
    key = 'slope_ms_per_1k_keys_per_2048_rows' if per_token else 'slope_ms_per_1k_keys'
    if not table.get(name) or not table.get(base) or table[base][key] <= 0:
        return None
    return table[name][key] / table[base][key]


def verdict(table):
    """Apply the ranking's M1 reading rules to the fitted slopes."""
    bf16 = _ratio(table, 'bf16_kv')
    bf16_source = 'bf16_kv'
    if bf16 is None:
        bf16, bf16_source = _ratio(table, 'bf16_qkv'), 'bf16_qkv'
    ratios = dict(bf16=bf16, bf16_source=bf16_source if bf16 is not None else None,
                  exp_approx=_ratio(table, 'exp_approx'), fp32_off=_ratio(table, 'fp32_off'),
                  q256_2048=_ratio(table, 'q256_2048'),
                  q256_4096_per_token=_ratio(table, 'q256_4096', per_token=True),
                  expected_bf16_if_bytes_bound=EXPECTED_BF16_RATIO)
    reasons = []
    if table.get('baseline') is None or bf16 is None:
        return dict(verdict='incomplete', ratios=ratios,
                    reasons=['the baseline or both bf16 arms have no slope; nothing to compare'])
    knobs = [value for value in (ratios['exp_approx'], ratios['fp32_off']) if value is not None]
    compute_moves = bool(knobs) and min(knobs) <= COMPUTE_KNOB_MAX
    q4096 = ratios['q256_4096_per_token']
    if bf16 >= BYTES_BF16_RATIO_MIN and (q4096 is None or q4096 <= BYTES_Q256_4096_MAX):
        name = 'bytes-bound'
        reasons.append('bf16/bf8 slope ratio %.2f >= %.2f (1.88 if purely bytes)' % (bf16, BYTES_BF16_RATIO_MIN))
        if q4096 is None:
            reasons.append('q256_4096 has no slope; the halving check could not run')
        else:
            reasons.append('q256_4096 per-token slope ratio %.2f <= %.2f' % (q4096, BYTES_Q256_4096_MAX))
        action = 'build lever #1 (chunked SDPA K/V prefix sharing)'
    elif bf16 <= COMPUTE_BF16_RATIO_MAX and compute_moves:
        name = 'compute-bound'
        reasons.append('bf16/bf8 slope ratio %.2f <= %.2f: bytes barely matter' % (bf16, COMPUTE_BF16_RATIO_MAX))
        reasons.append('a compute knob moves the slope to %.2f <= %.2f' % (min(knobs), COMPUTE_KNOB_MAX))
        action = 'lever #1 is worth ~0; take #1b (exp_approx / fp32_dest off) through a token gate'
    else:
        name = 'mixed'
        reasons.append('bf16/bf8 slope ratio %.2f, compute knobs %s, q256_4096 per-token %s: '
                       'neither reading rule holds cleanly' % (
                           bf16, ', '.join('%.2f' % value for value in knobs) or 'n/a',
                           'n/a' if q4096 is None else '%.2f' % q4096))
        action = 'no clean call: size #1 from the bf16 ratio (bytes share ~ (ratio - 1) / 0.88)'
    q2048 = ratios['q256_2048']
    prereg = None if q2048 is None else ('confirmed' if q2048 >= 0.95 else 'refuted')
    if prereg is not None:
        reasons.append('pre-registered q256@2048 no-gain: %s (ratio %.2f)' % (prereg, q2048))
    return dict(verdict=name, action=action, ratios=ratios, reasons=reasons, q256_2048_preregistration=prereg)


def verdict_line(result):
    ratios = result['ratios']
    show = lambda value: 'n/a' if value is None else '%.3f' % value
    return ('M1 VERDICT: %s | bf16/bf8=%s (%s) exp_approx=%s fp32_off=%s q256@2048=%s q256@4096/token=%s | %s'
            % (result['verdict'], show(ratios['bf16']), ratios.get('bf16_source') or '-', show(ratios['exp_approx']),
               show(ratios['fp32_off']), show(ratios['q256_2048']), show(ratios['q256_4096_per_token']),
               result.get('action', '')))


def schedule(arms, starts, rounds):
    """Interleaved timing order: every round visits every (arm, start), the arm order rotated."""
    order = []
    for index in range(rounds):
        shift = index % len(arms)
        rotated = arms[shift:] + arms[:shift]
        for arm in rotated:
            for start in starts:
                order.append((arm['name'], start))
    return order


def format_table(results, table, starts):
    lines = ['%-11s %6s %5s' % ('arm', 'rows', 'cores') + ''.join(' %10s' % ('@%dk' % (s // 1024)) for s in starts)
             + ' %12s %14s' % ('ms/1k keys', 'per 2048 rows')]
    for arm in ARMS:
        if arm['name'] not in results:
            continue
        cells = ''.join(' %10s' % ('ERR' if results[arm['name']].get(str(s)) is None
                                   else '%.3f' % results[arm['name']][str(s)]) for s in starts)
        line = table.get(arm['name'])
        tail = (' %12s %14s' % ('n/a', 'n/a') if line is None else
                ' %12.4f %14.4f' % (line['slope_ms_per_1k_keys'], line['slope_ms_per_1k_keys_per_2048_rows']))
        lines.append('%-11s %6d %5d' % (arm['name'], arm['rows'], busy_cores(arm)) + cells + tail)
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------
# Device side. ttnn/torch are imported inside these functions only.
# ---------------------------------------------------------------------------------------------

def _dtype(ttnn, name):
    return ttnn.bfloat8_b if name == 'bf8' else ttnn.bfloat16


def build_inputs(ttnn, torch, device, arms, starts, seed=0):
    """One K/V pool per KV dtype, one scattered page table, one Q per (rows, dtype)."""
    blocks = pool_blocks(arms, starts)
    generator = torch.Generator().manual_seed(seed)
    inputs = dict(blocks=blocks, pools={}, queries={}, starts={})
    for kv_dtype in sorted({arm['kv_dtype'] for arm in arms}):
        pool = []
        for _ in range(2):
            host = torch.randn(kv_shape(blocks), generator=generator, dtype=torch.float32).to(torch.bfloat16)
            pool.append(ttnn.from_torch(host, dtype=_dtype(ttnn, kv_dtype), layout=ttnn.TILE_LAYOUT, device=device,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG))
        inputs['pools'][kv_dtype] = tuple(pool)
    permutation = torch.randperm(blocks, generator=generator).to(torch.int32).reshape(1, blocks)
    inputs['page_table'] = ttnn.from_torch(permutation, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for arm in arms:
        key = (arm['rows'], arm['q_dtype'])
        if key not in inputs['queries']:
            host = torch.randn(q_shape(arm), generator=generator, dtype=torch.float32).to(torch.bfloat16)
            inputs['queries'][key] = ttnn.from_torch(host, dtype=_dtype(ttnn, arm['q_dtype']), layout=ttnn.TILE_LAYOUT,
                                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for start in starts:
        inputs['starts'][start] = ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32,
                                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return inputs


def make_call(ttnn, device, inputs, arm, start, grid, start_mode):
    """A zero-argument closure making the served call for one (arm, chunk_start)."""
    k_pool, v_pool = inputs['pools'][arm['kv_dtype']]
    query = inputs['queries'][(arm['rows'], arm['q_dtype'])]
    compute = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                               fp32_dest_acc_en=arm['fp32_dest'], packer_l1_acc=True)
    program = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=arm['exp_approx'],
                                     q_chunk_size=arm['q_chunk'], k_chunk_size=arm['k_chunk'])
    common = dict(input_tensor_q=query, input_tensor_k=k_pool, input_tensor_v=v_pool,
                  page_table_tensor=inputs['page_table'], compute_kernel_config=compute, program_config=program)
    if start_mode == 'tensor':
        return lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
            chunk_start_idx_tensor=inputs['starts'][start], **common)
    return lambda: ttnn.transformer.chunked_scaled_dot_product_attention(chunk_start_idx=start, **common)


def run(options):
    import torch
    import ttnn

    arms = [arm_by_name(name) for name in options.arms]
    for arm in arms:
        validate(arm, options.starts)
    device = ttnn.open_device(device_id=options.device_id, l1_small_size=24576)
    report = dict(passed=False, arms=arms, starts=list(options.starts), start_mode=options.start_mode,
                  warmup=options.warmup, rounds=options.rounds)
    try:
        grid_size = device.compute_with_storage_grid_size()
        grid = (grid_size.x, grid_size.y)
        report['grid'] = list(grid)
        if grid != GRID:
            report['grid_note'] = 'grid %r differs from the served 11 x 10; busy-core figures assume 11 x 10' % (grid,)
        inputs = build_inputs(ttnn, torch, device, arms, options.starts, seed=options.seed)
        report['pool_blocks'] = inputs['blocks']
        calls, errors, paths = {}, {}, {}
        for arm in arms:
            for start in options.starts:
                key = (arm['name'], start)
                mode = options.start_mode
                call = make_call(ttnn, device, inputs, arm, start, grid, mode)
                try:
                    for _ in range(options.warmup):
                        ttnn.deallocate(call())
                    ttnn.synchronize_device(device)
                except Exception as error:  # noqa: BLE001 - one bad arm must not end the sweep
                    if mode == 'tensor' and options.fallback_scalar:
                        mode = 'scalar'
                        call = make_call(ttnn, device, inputs, arm, start, grid, mode)
                        try:
                            for _ in range(options.warmup):
                                ttnn.deallocate(call())
                            ttnn.synchronize_device(device)
                            errors[key] = 'tensor path failed (%s: %s); timed on the scalar path' % (
                                type(error).__name__, error)
                        except Exception as second:  # noqa: BLE001
                            errors[key] = '%s: %s' % (type(second).__name__, second)
                            continue
                    else:
                        errors[key] = '%s: %s' % (type(error).__name__, error)
                        continue
                calls[key], paths[key] = call, mode
        samples = {key: [] for key in calls}
        for name, start in schedule(arms, list(options.starts), options.rounds):
            key = (name, start)
            if key not in calls:
                continue
            ttnn.synchronize_device(device)
            began = time.perf_counter()
            out = calls[key]()
            ttnn.synchronize_device(device)
            samples[key].append((time.perf_counter() - began) * 1e3)
            ttnn.deallocate(out)
        finite = {}
        for arm in arms:
            key = (arm['name'], options.starts[0])
            if key in calls:
                out = calls[key]()
                finite[arm['name']] = bool(torch.isfinite(ttnn.to_torch(out).float()).all())
                ttnn.deallocate(out)
        results = {arm['name']: {str(start): (statistics.median(samples[(arm['name'], start)])
                                              if samples.get((arm['name'], start)) else None)
                                 for start in options.starts} for arm in arms}
        table = slopes(results)
        outcome = verdict(table)
        report.update(
            passed=bool(calls) and all(value is not None for value in results.get('baseline', {}).values()),
            median_ms=results, slopes=table, verdict=outcome, verdict_line=verdict_line(outcome),
            samples_ms={'%s@%d' % key: values for key, values in samples.items()},
            paths={'%s@%d' % key: value for key, value in paths.items()},
            errors={'%s@%d' % key: value for key, value in errors.items()}, finite_output=finite,
            busy_cores={arm['name']: busy_cores(arm) for arm in arms},
            kv_gbytes_per_call={arm['name']: {str(s): kv_bytes(arm, s) / 1e9 for s in options.starts} for arm in arms},
            table=format_table(results, table, list(options.starts)))
    finally:
        ttnn.close_device(device)
    return report


def parse_list(text, cast=str):
    return [cast(value) for value in text.split(',') if value.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True, help='JSON report path')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--arms', default=','.join(ARM_NAMES))
    parser.add_argument('--starts', default=','.join(map(str, STARTS)))
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=9, help='interleaved timing rounds (samples per pair)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--start-mode', choices=('tensor', 'scalar'), default='tensor',
                        help='tensor = the served flexible path (device chunk_start_idx_tensor)')
    parser.add_argument('--no-fallback-scalar', dest='fallback_scalar', action='store_false',
                        help='do not retry a failing tensor-path pair on the scalar path')
    options = parser.parse_args(argv)
    options.arms = parse_list(options.arms)
    options.starts = parse_list(options.starts, int)
    for name in options.arms:
        arm_by_name(name)
    if options.rounds < 1 or options.warmup < 1:
        parser.error('--rounds and --warmup must be at least 1')
    report = dict(passed=False)
    try:
        report = run(options)
    except Exception as error:  # noqa: BLE001
        report['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        options.out.parent.mkdir(parents=True, exist_ok=True)
        options.out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8', newline='\n')
    if report.get('table'):
        print(report['table'])
    for key, value in sorted((report.get('errors') or {}).items()):
        print('ERROR %s: %s' % (key, value))
    print(report.get('verdict_line') or 'M1 VERDICT: incomplete | %s' % report.get('error', 'no result'))
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
