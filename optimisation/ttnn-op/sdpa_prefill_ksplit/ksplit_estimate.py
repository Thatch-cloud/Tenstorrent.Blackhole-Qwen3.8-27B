"""THE ESTIMATE of what the K-split is worth (docs/tp4-prefill3-ksplit.md section 3). Every number it prints is an ESTIMATE (E): arithmetic on the prefill cost model
(scripts/ci/prefill_cost_model.py: the PF1 device profile, the oneq card-M ratios, the production route fits) and on the K-split planner's block counts. Nothing here has run on a card.

    python3 -B ksplit_estimate.py            # the tables of the doc, as markdown
    python3 -B ksplit_estimate.py --check    # the numbers the doc quotes, asserted (exit 1 on drift)
    python3 -B ksplit_estimate.py --doc ../../../docs/tp4-prefill3-ksplit.md [--write]    # compare / rewrite the doc's marked block with the tables

THE MODEL. A masked tail pass of R rows costs `a + b x context` (the production fit, M: 329 / 381 ms and 1.274 / 1.251 ms per 1k tokens for the 512 / 1,024 buckets): the slope b is the
attention, the only part of a tail pass that grows with context (the K stream of the 16 attention layers). The K-split shares the K stream of a q chunk among S slices, so the slope becomes
b x r(context), r = ksplit_planner.layer_ratio (the planner's block counts: the busiest slice, the layer overhead and the relay hops of HOP_US each, over the oneq call at the same rows).
Three levels, because the ratio r is a model of one isolated call and a split pass keeps 96 cores busy (S x the 24 or 48 of its oneq tail): the same busy-core count as a 2,048-row chunk, which
runs at the 300 W board limit (loaded clock ~1,210 MHz against 1,345 in the profile). The tail passes of production are measured at the clock of 24 or 48 busy cores; their slope is 1.25-1.27 ms
per 1k tokens against the chunk's 2.108 (prefill_cost_model.PRODUCTION_CHUNK), so the split pass may lose up to that clock difference. The oneq A/B on the 2,048-row chunk measured a smaller
version of the same effect (an effective SDPA ratio 0.63 against the 0.505 modelled: x1.25). So r is multiplied by a clock penalty P (never below the isolated model, never above no gain):
  LOW      P = chunk slope / tail slope (1.65 at 512 rows, 1.68 at 1,024: the split pass runs at the chunk's per-block time) and a 40 us hop
  CENTRAL  P = 1.25 (the measured oneq in-situ factor) and an 18 us hop
  HIGH     P = 1.0 (the isolated model) and an 18 us hop
It takes the 512 and 1,024 buckets ONLY: the 128 and 256 buckets are not chain-qualified (the graft's _QWEN_PF_ROWS) and the 2,048 bucket has one slice.

THE PRODUCTION POPULATION (docs/tp4-prefill-2.md section 3.4, M): tail buckets over the window's 1,022 bucketed prompts: 128: 61, 256: 66, 512: 126, 1,024: 243, 2,048: 526. Their start contexts
are not published per prompt, only that the mean prompt is ~59k tokens (the programme's brief), so the default evaluates every tail pass at one MEAN context and prints the sensitivity: the
saving is close to linear in the context (r is flat above 32k), so the mean is what matters. `population_saving_s(level, passes=[(bucket, context), ...])` takes the real list instead.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / 'scripts' / 'ci'))

import ksplit_planner as planner  # noqa: E402
import prefill_cost_model as model  # noqa: E402

ENGAGED_BUCKETS = (512, 1024)
IN_SITU_FACTOR = 0.63 / 0.505                                       # M the oneq A/B: effective SDPA ratio of the lever in the served stack over card M's isolated ratio (prefill_cost_model.py)
LEVELS = (('low', None, planner.HOP_US_PESSIMISTIC), ('central', IN_SITU_FACTOR, planner.HOP_US), ('high', 1.0, planner.HOP_US))      # (name, clock penalty P or None = chunk slope / tail slope, hop us)
CONTEXTS = (4096, 32768, 65536, 120832, 163840, 253952)
POPULATION = {128: 61, 256: 66, 512: 126, 1024: 243, 2048: 526}      # M docs/tp4-prefill-2.md section 3.4 (they sum to 1,022 of the window's 1,024 finished prompts)
MEAN_PROMPT_TOKENS = 59000                                          # M the programme's brief
TAIL_ALLOWANCE = 2000                                               # E tokens of the prompt after the tail pass's start (the tail rows and the part of a chunk before them)
MEAN_TAIL_CONTEXT = MEAN_PROMPT_TOKENS - TAIL_ALLOWANCE             # E the context of the mean tail pass
SENSITIVITY_CONTEXTS = (32768, 57000, 80000, 120832, 160000)
ROUTE_S_TODAY = 1336.0                                              # M the window's route time under the fitted model (1,437 s measured), docs/tp4-prefill-2.md section 4
EXACT_RESUME_SHARE = 0.39                                           # E the share of that route time exact resume P0 removes (same section)
WALL_S = 6668.0                                                     # M the window's wall time, docs/tp4-prefill-2.md section 3.5


def level_of(name):
    return [row for row in LEVELS if row[0] == name][0]


def ratio(rows, context, hop_us):
    """r(context): ksplit over oneq, per layer (the planner's block counts and the relay hops); 1.0 when the lever does not take the rows."""
    return planner.layer_ratio(rows, context - context % 128, hop_us=hop_us)


def tail_attention_ms(bucket, context):
    """M the context-dependent part of a tail pass of this bucket: the production fit's slope times the context in k tokens."""
    return model.PRODUCTION_TAIL[bucket][1] * context / 1000.0


def tail_ms(bucket, context):
    a, b = model.PRODUCTION_TAIL[bucket]
    return a + b * context / 1000.0


def clock_penalty(bucket, level):
    """E P of a level: the factor the isolated ratio r is multiplied by in the served stack (see the module docstring)."""
    penalty = level_of(level)[1]
    return model.PRODUCTION_CHUNK[1] / model.PRODUCTION_TAIL[bucket][1] if penalty is None else penalty


def ratio_in_situ(bucket, context, level):
    """E the ratio of a split pass's attention to its oneq pass's at this level: the planner's r times the clock penalty, capped at 1 (no gain)."""
    return min(1.0, ratio(bucket, context, level_of(level)[2]) * clock_penalty(bucket, level))


def saving_ms(bucket, context, level):
    """E what the K-split saves on ONE tail pass of this bucket at this context, by level; 0 for a bucket it does not take."""
    if bucket not in ENGAGED_BUCKETS:
        return 0.0
    return tail_attention_ms(bucket, context) * (1.0 - ratio_in_situ(bucket, context, level))


def cost_model_saving_ms(bucket, context):
    """E the older row (prefill_cost_model.split_k_saving_ms, docs/tp4-prefill-2.md section 4) for one pass: the SDPA of a 2,048-row chunk at this context, wall-scaled, times a flat 0.8
    efficiency of the ideal split, for every bucket below 2,048 rows (S = 4 at 512 rows and below, 2 at 1,024)."""
    slices = model.split_k_slices(bucket)
    effective = 1.0 + (slices - 1) * model.SPLIT_K_EFFICIENCY
    return model.sdpa_kernel_ms(context) * model.WALL_PER_KERNEL * (1.0 - 1.0 / effective)


def population_passes(context=MEAN_TAIL_CONTEXT):
    """[(bucket, context)] of the window's tail passes with every pass at one context."""
    return [(bucket, context) for bucket, count in sorted(POPULATION.items()) for _ in range(count)]


def population_saving_s(level, passes=None, buckets=ENGAGED_BUCKETS):
    """E seconds the K-split saves over the window's tail passes (all at the mean context unless `passes` lists them)."""
    passes = population_passes() if passes is None else passes
    return sum(saving_ms(bucket, context, level) for bucket, context in passes if bucket in buckets) / 1000.0


def population_cost_model_s(context=MEAN_TAIL_CONTEXT):
    """E the same population priced the older way (every bucket below 2,048 rows)."""
    return sum(cost_model_saving_ms(bucket, ctx) for bucket, ctx in population_passes(context) if bucket < 2048) / 1000.0


def route_s(after_exact_resume):
    return ROUTE_S_TODAY * (1.0 - EXACT_RESUME_SHARE) if after_exact_resume else ROUTE_S_TODAY


def prompts_taken():
    return sum(POPULATION[bucket] for bucket in ENGAGED_BUCKETS)


def rows_for_tables():
    """[(bucket, context, low, central, high, tail ms, share of the tail pass, cost-model row)] for the engaged buckets at the doc's contexts."""
    out = []
    for bucket in ENGAGED_BUCKETS:
        for context in CONTEXTS:
            low, central, high = (saving_ms(bucket, context, level) for level, _s, _h in LEVELS)
            out.append((bucket, context, low, central, high, tail_ms(bucket, context), central / tail_ms(bucket, context), cost_model_saving_ms(bucket, context)))
    return out


def format_tables():
    lines = ['Per tail pass (ms; E; the pass is the production fit a + b x context; "isolated r" is the planner\'s per-layer ratio, "in-situ r" multiplies it by the central clock penalty; "older row" = prefill_cost_model.split_k_saving_ms for the same pass):', '',
             '| bucket | context | the pass today | S | isolated r | in-situ r (central) | saved low | saved central | saved high | central share of the pass | older row |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for bucket, context, low, central, high, total, share, older in rows_for_tables():
        lines.append('| %d | %s | %.0f | %d | %.2f | %.2f | %.0f | %.0f | %.0f | %.1f%% | %.0f |' % (
            bucket, format(context, ','), total, planner.split_k_slices(bucket), ratio(bucket, context, planner.HOP_US), ratio_in_situ(bucket, context, 'central'), low, central, high, 100 * share, older))
    lines += ['', 'Over the production window (%d of its 1,022 bucketed prompts are in a bucket the lever takes: 512 x %d, 1,024 x %d; every tail pass at the mean context %s tokens: the %dk mean prompt less %dk):' % (
        prompts_taken(), POPULATION[512], POPULATION[1024], format(MEAN_TAIL_CONTEXT, ','), MEAN_PROMPT_TOKENS // 1000, TAIL_ALLOWANCE // 1000), '',
        '| level | seconds saved in the window | share of today\'s route time (%.0f s) | share after exact resume (%.0f s) | share of the window\'s wall time (%.0f s) |' % (
            ROUTE_S_TODAY, route_s(True), WALL_S), '|---|---:|---:|---:|---:|']
    for level, _s, _h in LEVELS:
        seconds = population_saving_s(level)
        lines.append('| %s | %.1f | %.2f%% | %.2f%% | %.2f%% |' % (level, seconds, 100 * seconds / route_s(False), 100 * seconds / route_s(True), 100 * seconds / WALL_S))
    lines += ['', 'Sensitivity to the mean tail context (central level; the older row prices every bucket below 2,048 rows, 496 prompts):', '',
              '| mean tail context | saved, this lever (s) | share of today\'s route time | share after exact resume | the older row, same context (s) | share of today\'s route time |', '|---:|---:|---:|---:|---:|---:|']
    for context in SENSITIVITY_CONTEXTS:
        seconds = population_saving_s('central', population_passes(context))
        older = population_cost_model_s(context)
        lines.append('| %s | %.1f | %.2f%% | %.2f%% | %.1f | %.2f%% |' % (format(context, ','), seconds, 100 * seconds / route_s(False), 100 * seconds / route_s(True), older, 100 * older / route_s(False)))
    return '\n'.join(lines)


def check():
    """The numbers the doc quotes (assertions; exit 1 on drift)."""
    problems = []

    def expect(label, value, low, high):
        if not low <= value <= high:
            problems.append('%s = %.3f outside [%s, %s]' % (label, value, low, high))
    expect('1024-row bucket saving at 120k, central', saving_ms(1024, 120832, 'central'), 55, 75)
    expect('512-row bucket saving at 120k, central', saving_ms(512, 120832, 'central'), 90, 120)
    expect('1024-row ratio r at 120k', ratio(1024, 120832, planner.HOP_US), 0.49, 0.53)
    expect('512-row ratio r at 120k', ratio(512, 120832, planner.HOP_US), 0.25, 0.28)
    expect('512-row saving at 4k, central (the stream is short)', saving_ms(512, 4096, 'central'), 0.5, 4)
    expect('prompts the lever takes', prompts_taken(), 369, 369)
    expect('central seconds saved in the window at the mean context', population_saving_s('central'), 10, 16)
    expect('central share of today\'s route time', 100 * population_saving_s('central') / route_s(False), 0.75, 1.25)
    expect('low below central below high', float(population_saving_s('low') < population_saving_s('central') < population_saving_s('high')), 1, 1)
    expect('this lever against the older row at 120k, 512 rows', saving_ms(512, 120832, 'central') / cost_model_saving_ms(512, 120832), 0.4, 0.65)
    expect('this lever against the older row at 120k, 1024 rows', saving_ms(1024, 120832, 'central') / cost_model_saving_ms(1024, 120832), 0.4, 0.65)
    expect('the older row at 120k, whole population share (the prefill-2 order of magnitude)', 100 * population_cost_model_s(120832) / route_s(False), 4.0, 6.5)
    for bucket in (128, 256, 2048):
        expect('saving of the untaken bucket %d' % bucket, saving_ms(bucket, 120832, 'high'), 0.0, 0.0)
    for problem in problems:
        print('DRIFT', problem)
    return 1 if problems else 0


BEGIN = '<!-- ksplit-estimate:begin -->'
END = '<!-- ksplit-estimate:end -->'


def embedded(text):
    """The tables between the doc's markers (None when the markers are missing or out of order)."""
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        return None
    return text[text.index(BEGIN) + len(BEGIN):text.index(END)].strip(chr(10))


def with_tables(text):
    """`text` with the block between the markers replaced by the current tables."""
    if embedded(text) is None:
        raise ValueError('the markers %s ... %s must occur once, in order' % (BEGIN, END))
    return text[:text.index(BEGIN) + len(BEGIN)] + chr(10) + format_tables() + chr(10) + text[text.index(END):]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--doc', help='the document whose marked block holds the tables')
    parser.add_argument('--write', action='store_true', help='with --doc: rewrite the block (otherwise compare and exit 1 on drift)')
    options = parser.parse_args(argv)
    if options.doc:
        path = Path(options.doc)
        text = path.read_text(encoding='utf-8')
        if options.write:
            path.write_text(with_tables(text), encoding='utf-8', newline=chr(10))
            return 0
        if embedded(text) != format_tables():
            print('DRIFT the tables in %s are not the script\'s output (run with --doc --write)' % path)
            return 1
        return 0
    if options.check:
        return check()
    print(format_tables())
    return 0


if __name__ == '__main__':
    sys.exit(main())
