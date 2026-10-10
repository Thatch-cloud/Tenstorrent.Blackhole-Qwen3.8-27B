"""Estimated vs measured table of the oneq lever, from the card-M report (oneq_card_m.py) and the planner's model. No ttnn.

The card job times ONE attention layer's chunked SDPA call per (arm, chunk_start) with the very program the model builds. A
prompt of N chunks makes N x 16 such calls (16 full-attention layers), and the per-call time is linear in the context
(R^2 = 1.000 in the prefill profile), so the card-M numbers give the solo-TTFT effect by the same sum the estimate uses:

    saving_s = WALL_PER_DEVICE x ATTENTION_LAYERS x sum over chunk starts C of (t_served(C) - t_oneq(C))

with t_arm(C) the least-squares line through that arm's measured medians (one line per Q memory). WALL_PER_DEVICE (1.1286) is the
profile's unprofiled wall per device second; it is an estimate, not a measurement of this lever: the model-gate A/B (three
alternating TTFT runs per side at 128k) is the measurement of the wall figure.

    python3 oneq_report.py <card report .json> [--qmem dram] [--markdown]
"""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oneq_planner as op  # noqa: E402

PROMPTS = (32768, 131072, 253952)       # 32k, 128k, 254k (124 chunks)


def fit(points):
    """Least-squares line through [(x, y)]: (intercept, slope). Needs two distinct x."""
    xs = [float(x) for x, _y in points]
    ys = [float(y) for _x, y in points]
    n = len(xs)
    if n < 2 or max(xs) == min(xs):
        raise ValueError('need two distinct contexts to fit a line, got %r' % (points,))
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    return mean_y - slope * mean_x, slope


def line(coefficients, x):
    return coefficients[0] + coefficients[1] * x


def medians(timing, rows, qmem, arm):
    """{start: median ms} of one arm from a report's 'timing' block ({rows: {qmem: {arm: {start: {median_ms}}}}})."""
    block = ((timing.get(str(rows)) or {}).get(qmem) or {}).get(arm) or {}
    return {int(start): value['median_ms'] for start, value in block.items() if value.get('median_ms') is not None}


def per_start_table(timing, rows, qmem, nqh=6, nkh=1, grid=(13, 10), arms=('served', 'oneq')):
    """[dict] one per timed start: context, modelled and measured per-layer us for the paired and the oneq arm, and the saving."""
    paired, oneq = op.plan(nqh, nkh, rows, 128, grid), op.plan(nqh, nkh, rows, 128, grid, oneq=True)
    served, one = medians(timing, rows, qmem, arms[0]), medians(timing, rows, qmem, arms[1])
    out = []
    for start in sorted(set(served) & set(one)):
        est_s, est_o = op.layer_us(paired, start), op.layer_us(oneq, start)
        out.append(dict(start=start, context=start + rows, est_served_us=est_s, est_oneq_us=est_o,
                        est_saving_us=est_s - est_o, served_us=served[start] * 1e3, oneq_us=one[start] * 1e3,
                        saving_us=(served[start] - one[start]) * 1e3, ratio=one[start] / served[start]))
    return out


def measured_saving_s(table, prompt_tokens, wall=op.WALL_PER_DEVICE, layers=op.ATTENTION_LAYERS):
    """Solo-TTFT saving (s) from the per-start table: lines through both arms, summed over the prompt's chunk starts."""
    served = fit([(row['start'], row['served_us']) for row in table])
    oneq = fit([(row['start'], row['oneq_us']) for row in table])
    saved_us = sum(line(served, start) - line(oneq, start) for start in op.chunk_starts(prompt_tokens))
    return saved_us * layers * wall / 1e6


def summary(timing, rows=2048, qmem='dram', nqh=6, nkh=1, grid=(13, 10), prompts=PROMPTS):
    """-> dict(per_start, prompts [dict]) for one (rows, Q memory); prompts rows carry the estimate and the measurement."""
    table = per_start_table(timing, rows, qmem, nqh, nkh, grid)
    estimates = {tokens: (floor, pessimistic) for tokens, _chunks, floor, pessimistic in op.estimate_table(prompts)}
    out = []
    for tokens in prompts:
        entry = dict(prompt_tokens=tokens, chunks=tokens // op.CHUNK_TOKENS, est_floor_s=estimates[tokens][0],
                     est_pessimistic_s=estimates[tokens][1], measured_s=None, measured_over_est=None)
        if len(table) >= 2:
            entry['measured_s'] = measured_saving_s(table, tokens)
            entry['measured_over_est'] = entry['measured_s'] / entry['est_floor_s']
        out.append(entry)
    return dict(per_start=table, prompts=out)


def render(result, markdown=False):
    pipe = '|' if markdown else ' '
    lines = []
    head = ('start', 'context', 'est served us', 'meas served us', 'est oneq us', 'meas oneq us', 'est saving us', 'meas saving us',
            'meas oneq/served')
    if markdown:
        lines += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    else:
        lines.append(' '.join('%14s' % name for name in head))
    for row in result['per_start']:
        cells = (row['start'], row['context'], '%.1f' % row['est_served_us'], '%.1f' % row['served_us'], '%.1f' % row['est_oneq_us'],
                 '%.1f' % row['oneq_us'], '%.1f' % row['est_saving_us'], '%.1f' % row['saving_us'], '%.3f' % row['ratio'])
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |' if markdown else ' '.join('%14s' % c for c in cells))
    lines.append('')
    head = ('prompt tokens', 'chunks', 'est saving s (13.51 us/step)', 'est saving s (16.5 us/step)', 'measured saving s', 'measured / est')
    if markdown:
        lines += ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    else:
        lines.append(' '.join('%16s' % name for name in head))
    for entry in result['prompts']:
        measured = '%.2f' % entry['measured_s'] if entry['measured_s'] is not None else 'pending'
        ratio = '%.2f' % entry['measured_over_est'] if entry['measured_over_est'] is not None else '-'
        cells = (entry['prompt_tokens'], entry['chunks'], '%.2f' % entry['est_floor_s'], '%.2f' % entry['est_pessimistic_s'], measured, ratio)
        lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |' if markdown else ' '.join('%16s' % c for c in cells))
    return NL.join(lines) + NL


NL = chr(10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    parser.add_argument('report', type=Path)
    parser.add_argument('--rows', type=int, default=2048)
    parser.add_argument('--qmem', default='dram')
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding='utf-8'))
    grid = tuple(report.get('grid') or (13, 10))
    geometry = report.get('geometry') or {}
    result = summary(report.get('timing') or {}, args.rows, args.qmem, geometry.get('nqh', 6), geometry.get('nkh', 1), grid)
    sys.stdout.write(render(result, args.markdown))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
