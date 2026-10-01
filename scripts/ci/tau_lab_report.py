"""The tau lab's report (c2_tau_lab.py's results directory in, an aggregates-only summary out).

    python3 scripts/ci/tau_lab_report.py --results ~/kwork64/taulab/results/<run> --out <public dir> \
        [--counters production-counters.json] [--data <data dir>]

tau = tokens committed per user per verify round. Per request, from the container's own log (which the lab's two log flags
fill) joined to the lab's turns by the request id the stream carried:

  [PACKED] request= segment= position= prefix= emitted= [cap=]   one per user per packed 4-user round (16 rows)
  [SEQUENTIAL] / [SEQ-PUBLISH] request= ...                      the 4-row sequential steps (never counted in tau)
  {"stage": "fast_serving_phases", ...}                          one record per request at close (rows, committed per round)
  the stream's continuous usage stats                            completion tokens per chunk

THE STATISTIC (design 1.3, tau-lab.md section 7): pooled agent T16 tau = sum of `emitted` / number of rounds over the
[PACKED] rounds (rows = 16, all four seats live), excluding each request's terminal round (cut by EOS or the budget), the
prefill seed token and the 4-row steps. A round with a cap= (a commit the remaining budget or an extent block cut) counts in
`tau` as served and is left out of `tau_uncapped`. Thinking-ON arms A1 (swe), A2 (own) and A4 (chained) are pooled with EQUAL
WEIGHT per set, the 95% interval a cluster bootstrap over conversations (10,000 resamples, resampled within each set).
A3 (calibration) and A5 (thinking OFF) are reported beside it and never pooled into it.

WHAT IT REPORTS (every number an aggregate; the public summary holds no id, text or token):
  arms             per arm: turns, rounds, tau, per-turn p10 / p50, worst seat (the lowest per-segment tau)
  k_inputs         the gate-G inputs: pooled tau and its CI, per-turn min / p10 / p50 per set, the share of turns under 4.2, tau
                   by context bucket (the >80k bucket separately), by output region (reasoning / content / tool), per segment,
                   worst-of-8 and worst-of-9 (P(min tau >= bar) over the strict and the 8 x 75 bars), and the K1 / K2 SCREENS:
                   the observed values times the design's fine-tune (+15-33%) and lookup (+3-5%) multipliers against the bars.
                   The screens are arithmetic, not the simulation: the class-0 sweep and drafter_sens.py rerun on tapes.jsonl
                   (private) decide K1 at gate G.
  thinking         A1 against A5 over the SAME turns: paired mean tau on and off
  checks           the design's 'before the number counts': the three sources agree; A3 within +/-7% of v162/v179; the image is
                   the production image; the derived profile's arithmetic diff is empty; positions 1-3 within +/-10% of
                   production's spec-decode counters
Private files written beside the turns (never uploaded): tapes.jsonl (per turn: its rounds), report.private.json.

Python 3.7 syntax, stdlib only: it runs on the rig host and in CI.
"""
import argparse
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import acceptance_report as acc  # noqa: E402

PRIMARY_ARMS = ('A1', 'A2', 'A4')
SET_OF_ARM = dict(A1='swe', A2='own', A4='chained')
SET_ORDER = ('swe', 'own', 'chained')
BUCKETS = ((4096, '4k'), (8192, '8k'), (16384, '16k'), (32768, '32k'), (65536, '64k'), (81920, '80k'), (None, '80k+'))
LONG_BUCKET = '80k+'
MIN_ROUNDS = 8                    # a turn with fewer counted rounds is not a per-turn tau
STRICT_BARS = (10.35, 13.5)       # design 1.3 / tau-lab 7: the strict bars (r = 1.0), the worst-of-9 statistic
K2_BARS = (5.95, 7.6, 8.2)        # 8 x 75 worst-seat bars at r = 1.0 / mid / 0.8 (design section 0.3)
LOW_TAU = 4.2                     # the plan: below it the frame falls under the DRAM floor
FINE_TUNE_CENTRAL = (1.15, 1.33)  # sim, design 1.3
LOOKUP = (1.03, 1.05)             # sim, design 1.3
K1_BAR = 10.35
SANITY_FLOOR = 5.0
A3_TOLERANCE = 0.07
POSITION_TOLERANCE = 0.10
RESAMPLES = 10000
DRAWS = 20000
ROUND_FIELDS = ('segment', 'position', 'prefix', 'emitted', 'rows', 'cap')

# Everything a public string may be: the report is aggregates, so a word outside this set (or the image tag) is a leak.
PUBLIC_WORDS = frozenset((
    'A1', 'A2', 'A3', 'A4', 'A5', 'swe', 'own', 'chained', 'calib', '4k', '8k', '16k', '32k', '64k', '80k', '80k+',
    'reasoning', 'content', 'tool', 'PASS', 'FAIL', 'MAYBE', 'NOT_ESTABLISHED', 'NOT_RUN', 'exact', 'coalesced', 'disagree',
    'none', 'unique', 'ambiguous', 'on', 'off', 'baked', 'production', 'non-production', 'phases', 'spec_lines'))
PUBLIC_KEY = re.compile(r'[A-Za-z0-9_.>+-]{1,48}')
IMAGE_TAG = re.compile(r'[0-9a-z.-]{3,40}')
MAX_LIST = 64


class PrivacyError(ValueError):
    """A value that is not an aggregate; the public summary is refused rather than written."""


def assert_public(value, path='summary'):
    """Refuse anything that is not an aggregate: strings outside PUBLIC_WORDS (the `image_tag` key aside), lists past MAX_LIST,
    nested lists of lists, non-finite numbers. The Qwen repo is public: its Actions logs and artifacts are too."""
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not PUBLIC_KEY.fullmatch(key):
                raise PrivacyError('%s: a key that is not a plain name' % path)
            if key == 'image_tag' and isinstance(item, str):
                if not IMAGE_TAG.fullmatch(item):
                    raise PrivacyError('%s.image_tag is not a plain tag' % path)
                continue
            assert_public(item, '%s.%s' % (path, key))
    elif isinstance(value, (list, tuple)):
        if len(value) > MAX_LIST:
            raise PrivacyError('%s: a list of %d (the most an aggregate needs is %d)' % (path, len(value), MAX_LIST))
        for item in value:
            if isinstance(item, (list, tuple)):
                raise PrivacyError('%s: a list inside a list' % path)
            assert_public(item, path + '[]')
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int):
        return
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise PrivacyError('%s: a non-finite number' % path)
    elif isinstance(value, str):
        if value not in PUBLIC_WORDS:
            raise PrivacyError('%s: a string that is not one of the report\'s words' % path)
    else:
        raise PrivacyError('%s: a %s' % (path, type(value).__name__))


# -- the log ---------------------------------------------------------------------------------------------------------

def parse_rounds(log_text):
    """{engine request id: [round, ...]} in log order: kind P (a packed round) or S (a sequential step), and the fields each
    line carries (acceptance_report.PATH_FIELD reads them by name)."""
    rounds = {}
    for match in acc.PATH_LINE.finditer(log_text):
        kind, request_id, rest = match.groups()
        if kind == 'SEQ-PUBLISH' and not rest.startswith('rows='):
            continue
        fields = {}
        for name, value in acc.PATH_FIELD.findall(rest):
            fields.setdefault(name, None if value == 'n/a' else int(value))
        entry = dict(kind=acc.PATH_KINDS[kind])
        for name in ROUND_FIELDS:
            entry[name] = fields.get(name)
        rounds.setdefault(request_id, []).append(entry)
    return rounds


def owner_of(request_id, index):
    """The key of `index` ({stream request id: ...}) the engine request id extends: cmpl-<hex> -> cmpl-<hex>-0-<sfx>, the
    id cut at 48 characters by the audit line included."""
    candidate = request_id
    while candidate:
        if candidate in index:
            return candidate
        cut = candidate.rfind('-')
        if cut <= 0:
            return None
        candidate = candidate[:cut]
    return None


def bucket_of(tokens):
    if tokens is None:
        return None
    for edge, label in BUCKETS:
        if edge is None or tokens <= edge:
            return label
    return None


# -- one turn --------------------------------------------------------------------------------------------------------

def counted_rounds(rounds):
    """The rounds the statistic counts: the packed ones before the request's terminal round, as (round, start offset in the
    completion's tokens). The offset starts at 1 (the prefill seed) and is tracked while every round's emitted count is
    known: a step that logs none (a sequential one) leaves later offsets unknown (None), never guessed."""
    out, offset = [], 1
    last = len(rounds) - 1
    for position, entry in enumerate(rounds):
        if entry['kind'] == 'P' and position != last and entry.get('emitted') is not None:
            out.append((entry, offset))
        if entry.get('emitted') is None:
            offset = None
        elif offset is not None:
            offset += entry['emitted']
    return out


def region_of(turn, start):
    if start is None:
        return None
    tool_at = turn.get('tool_at')
    if tool_at is not None and start >= tool_at:
        return 'tool'
    think = turn.get('think_tokens') or 0
    return 'reasoning' if start < think else 'content'


def stream_check(turn, rounds):
    """'exact' / 'coalesced' / 'disagree' for a request whose rounds are all packed (the stream's chunks against its [PACKED]
    emitted counts), None otherwise or without per-chunk counts."""
    if not rounds or any(entry['kind'] != 'P' or entry.get('emitted') is None for entry in rounds):
        return None
    blocks = [dict(committed=entry['emitted']) for entry in rounds]
    return acc.stream_agreement(dict(chunk_tokens=turn.get('chunk_tokens')), blocks)


def analyze_turn(turn, rounds):
    """The turn's statistics from its rounds: the counted rounds (segment, emitted, cap, region), their sums and tau."""
    counted = counted_rounds(rounds or [])
    emitted = [entry['emitted'] for entry, _ in counted]
    uncapped = [entry['emitted'] for entry, _ in counted if entry.get('cap') is None]
    tokens = turn.get('prompt_tokens') or turn.get('declared_tokens')
    return dict(rounds=len(emitted), emitted=sum(emitted), uncapped_rounds=len(uncapped), uncapped_emitted=sum(uncapped),
                tau=(sum(emitted) / float(len(emitted))) if emitted else None, bucket=bucket_of(tokens),
                segments=[entry.get('segment') for entry, _ in counted],
                regions=[region_of(turn, start) for _, start in counted],
                values=emitted, caps=[entry.get('cap') for entry, _ in counted])


# -- statistics ------------------------------------------------------------------------------------------------------

def quantile(values, q):
    """numpy's default (linear) quantile of a list, or None."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean(values):
    values = list(values)
    return sum(values) / float(len(values)) if values else None


def rounded(value, digits=3):
    return None if value is None else round(value, digits)


def distribution(taus):
    return dict(turns=len(taus), min=rounded(min(taus)) if taus else None, p10=rounded(quantile(taus, 0.10)),
                p50=rounded(quantile(taus, 0.50)), mean=rounded(mean(taus)))


def tally(stats):
    return (sum(item['emitted'] for item in stats), sum(item['rounds'] for item in stats))


def ratio(pair):
    return pair[0] / float(pair[1]) if pair[1] else None


def pooled_equal_weight(tallies):
    """Mean over the sets present of each set's sum / rounds. `tallies`: {set: (emitted, rounds)}."""
    values = [ratio(tallies[name]) for name in SET_ORDER if name in tallies and tallies[name][1]]
    return mean(values)


def cluster_tallies(items):
    """{set: [(emitted, rounds) per cluster]} from [(set, cluster, emitted, rounds)]."""
    clusters = {}
    for name, cluster, emitted, rounds in items:
        entry = clusters.setdefault(name, {}).setdefault(cluster, [0, 0])
        entry[0] += emitted
        entry[1] += rounds
    return dict((name, [tuple(pair) for pair in sorted(by.values())]) for name, by in clusters.items())


def bootstrap_ci(clusters, resamples=RESAMPLES, seed=0, level=0.95):
    """The percentile interval of the equal-weight pooled tau under a cluster bootstrap within each set, or None."""
    rng = random.Random(seed)
    names = [name for name in SET_ORDER if clusters.get(name)]
    if not names:
        return None
    draws = []
    for _ in range(resamples):
        values = []
        for name in names:
            pool = clusters[name]
            emitted = rounds = 0
            for _ in range(len(pool)):
                pair = pool[rng.randrange(len(pool))]
                emitted += pair[0]
                rounds += pair[1]
            if rounds:
                values.append(emitted / float(rounds))
        if values:
            draws.append(sum(values) / len(values))
    low = (1.0 - level) / 2.0
    return dict(low=rounded(quantile(draws, low)), high=rounded(quantile(draws, 1.0 - low)), resamples=len(draws))


def worst_of_k(weighted, k, bars, draws=DRAWS, seed=0):
    """P(min tau of k concurrent turns >= bar) for each bar: k turns drawn at random (duration-weighted: a turn counts as its
    rounds) from `weighted` = {set: [(tau, rounds)]}, the set chosen uniformly first (equal weight per set)."""
    rng = random.Random(seed)
    names = [name for name in SET_ORDER if weighted.get(name)]
    if not names:
        return None
    cumulative = {}
    for name in names:
        total, running = float(sum(rounds for _, rounds in weighted[name])), 0.0
        cumulative[name] = []
        for tau, rounds in weighted[name]:
            running += rounds / total
            cumulative[name].append((running, tau))
    hits = dict((bar, 0) for bar in bars)
    for _ in range(draws):
        lowest = None
        for _ in range(k):
            name = names[rng.randrange(len(names))]
            point, table = rng.random(), cumulative[name]
            low, high = 0, len(table) - 1
            while low < high:
                mid = (low + high) // 2
                if table[mid][0] >= point:
                    high = mid
                else:
                    low = mid + 1
            tau = table[low][1]
            lowest = tau if lowest is None or tau < lowest else lowest
        for bar in bars:
            if lowest >= bar:
                hits[bar] += 1
    return dict(('bar_%s' % ('%g' % bar).replace('.', '_'), round(hits[bar] / float(draws), 4)) for bar in bars)


def verdict(low, high, bar):
    """PASS when even the low end clears the bar, FAIL when even the high end misses it, else MAYBE."""
    if low is None or high is None:
        return 'NOT_ESTABLISHED'
    if low >= bar:
        return 'PASS'
    return 'FAIL' if high < bar else 'MAYBE'


def screen(value, bars):
    """The design's central fine-tune (+15-33%) times lookup (+3-5%) on `value`, against each bar."""
    if value is None:
        return None
    low = value * FINE_TUNE_CENTRAL[0] * LOOKUP[0]
    high = value * FINE_TUNE_CENTRAL[1] * LOOKUP[1]
    entry = dict(observed=rounded(value), projected_low=rounded(low), projected_high=rounded(high))
    for bar in bars:
        entry['bar_%s' % ('%g' % bar).replace('.', '_')] = verdict(low, high, bar)
    return entry


# -- the report ------------------------------------------------------------------------------------------------------

def read_jsonl(path):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def last_ok_turns(turns):
    """The last ok record per (arm, id): a resumed run appends a turn again only when it failed before."""
    chosen = {}
    for turn in turns:
        key = (turn.get('arm'), turn.get('id'))
        if turn.get('status') == 'ok' or key not in chosen:
            chosen[key] = turn
    return chosen


def position_comparison(lab, production):
    """Positions 1-3 of the lab's acceptance (vLLM's convention: over every draft) against production's, within +/-10% relative.
    `lab`/`production`: lists of per-position rates. -> dict or None."""
    if not lab or not production:
        return None
    entries, passed = {}, True
    for index in range(3):
        if index >= len(lab) or index >= len(production) or not production[index]:
            return None
        relative = (lab[index] - production[index]) / float(production[index])
        entries['position_%d' % (index + 1)] = dict(lab=rounded(lab[index], 4), production=rounded(production[index], 4),
                                                    relative=rounded(relative, 4))
        passed = passed and abs(relative) <= POSITION_TOLERANCE
    entries['verdict'] = 'PASS' if passed else 'FAIL'
    return entries


def production_positions(counters):
    """Per-position acceptance (vLLM's convention) from production's aggregates file: `per_position`, or `accepted_per_pos`
    over `num_drafts`. None when it holds neither."""
    if not isinstance(counters, dict):
        return None
    if isinstance(counters.get('per_position'), list):
        return [float(value) for value in counters['per_position']]
    accepted, drafts = counters.get('accepted_per_pos'), counters.get('num_drafts')
    if isinstance(accepted, list) and drafts:
        return [float(value) / float(drafts) for value in accepted]
    return None


METRIC_POSITION = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{([^}]*)\}\s+([0-9.eE+-]+)\s*$')
METRIC_DRAFTS = re.compile(r'^vllm:spec_decode_num_drafts(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$')
POSITION_LABEL = re.compile(r'position="([0-9]+)"')


def counters_from_metrics(text):
    """Production's spec-decode counters, aggregates only, from a scrape of its Prometheus /metrics: {accepted_per_pos: [n at
    position 1, 2, ...], num_drafts: n} (vLLM numbers the positions from 0), summed over engines and label sets. None when the
    scrape holds neither counter. Nothing but counters is read: no label, no text, no request."""
    per_position, drafts = {}, 0.0
    for line in text.splitlines():
        match = METRIC_POSITION.match(line.strip())
        if match:
            label = POSITION_LABEL.search(match.group(1))
            if label:
                at = int(label.group(1))
                per_position[at] = per_position.get(at, 0.0) + float(match.group(2))
            continue
        match = METRIC_DRAFTS.match(line.strip())
        if match:
            drafts += float(match.group(1))
    if not per_position or not drafts:
        return None
    return dict(accepted_per_pos=[per_position.get(at, 0.0) for at in range(max(per_position) + 1)], num_drafts=drafts)


def lab_positions(log_text):
    """(per-position rates, where they came from): vLLM's own SpecDecoding lines pooled over the run, else the phases records'."""
    totals = acc.spec_totals(acc.spec_decoding(log_text))
    if totals and totals.get('per_position'):
        return totals['per_position'], 'spec_lines'
    records, _ = acc.phase_records(log_text)
    blocks = []
    for record in records:
        blocks.extend(block for block in acc.acceptance_blocks(record) if not block['terminal'])
    return (acc.per_position(blocks, convention='vllm') if blocks else None), 'phases'


def phases_agreement(turn_rounds, log_text):
    """{turn key: 'unique' | 'ambiguous' | 'none'}: whether the fast_serving_phases records hold the turn's packed emitted
    sequence once, more than once, or not at all (the record carries no request id: the sequence is the key)."""
    records, _ = acc.phase_records(log_text)
    holders = {}
    for record in records:
        blocks = [block for block in record['blocks'] if block['committed'] > 0]
        packed = [block['committed'] for block in blocks if block.get('packed')] or [block['committed'] for block in blocks]
        holders[tuple(packed)] = holders.get(tuple(packed), 0) + 1
    users = {}
    for key, rounds in turn_rounds.items():
        sequence = tuple(entry['emitted'] for entry in rounds if entry['kind'] == 'P' and entry.get('emitted') is not None)
        if sequence:
            users.setdefault(sequence, []).append(key)
    verdicts = {}
    for sequence, keys in users.items():
        held = holders.get(sequence, 0)
        for key in keys:
            verdicts[key] = 'none' if not held else ('unique' if held == 1 and len(keys) == 1 else 'ambiguous')
    return verdicts, len(records)


def build(results_dir, manifest=None, counters=None, info=None, seed=0, log_text=None, turns=None, outputs=None):
    """-> (public summary, private report, tapes). Pure over the results directory's files (or the arguments)."""
    manifest = manifest or {}
    info = info or {}
    if turns is None:
        turns = read_jsonl(os.path.join(results_dir, 'turns.jsonl'))
    if log_text is None:
        path = os.path.join(results_dir, 'server.log')
        log_text = ''
        if os.path.isfile(path):
            with open(path, encoding='utf-8', errors='replace') as handle:
                log_text = handle.read()
    chosen = last_ok_turns(turns)
    by_stream_id = dict((turn['request_id'], key) for key, turn in chosen.items()
                        if turn.get('request_id') and turn.get('status') == 'ok')
    engine_rounds = parse_rounds(log_text)
    turn_rounds, unattributed = {}, 0
    for request_id, rounds in engine_rounds.items():
        stream_id = owner_of(request_id, by_stream_id)
        if stream_id is None:
            unattributed += 1
            continue
        turn_rounds.setdefault(by_stream_id[stream_id], []).extend(rounds)
    phases, phase_records_seen = phases_agreement(turn_rounds, log_text)

    stats, tapes = {}, []
    source = dict(stream_exact=0, stream_coalesced=0, stream_disagree=0, stream_none=0, phases_unique=0, phases_ambiguous=0,
                  phases_none=0, turns_with_rounds=0, turns_without_rounds=0)
    for key, turn in sorted(chosen.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        if turn.get('status') != 'ok':
            continue
        rounds = turn_rounds.get(key) or []
        stats[key] = analyze_turn(turn, rounds)
        stats[key]['thinking'] = turn.get('thinking')
        if rounds:
            source['turns_with_rounds'] += 1
            check = stream_check(turn, rounds)
            source['stream_' + (check if check in ('exact', 'coalesced', 'disagree') else 'none')] += 1
            source['phases_' + phases.get(key, 'none')] += 1
        else:
            source['turns_without_rounds'] += 1
        tapes.append(dict(arm=key[0], id=key[1], set=turn.get('set'), cluster=turn.get('cluster'), turn=turn.get('turn'),
                          prompt_tokens=turn.get('prompt_tokens'), completion_tokens=turn.get('completion_tokens'),
                          thinking=turn.get('thinking'), finish=turn.get('finish'),
                          rounds=[[entry['kind'], entry.get('segment'), entry.get('emitted'), entry.get('rows'),
                                   entry.get('cap')] for entry in rounds]))

    arms = {}
    for arm in sorted(set(key[0] for key in stats)):
        items = [(key, stat) for key, stat in stats.items() if key[0] == arm]
        usable = [stat for _, stat in items if stat['rounds'] >= MIN_ROUNDS]
        taus = [stat['tau'] for stat in usable]
        by_seat = {}
        for _, stat in items:
            for segment, value in zip(stat['segments'], stat['values']):
                pair = by_seat.setdefault(segment, [0, 0])
                pair[0] += value
                pair[1] += 1
        seats = dict(('seat_%s' % segment, rounded(pair[0] / float(pair[1]))) for segment, pair in sorted(
            by_seat.items(), key=lambda item: (item[0] is None, item[0])) if pair[1] and segment is not None)
        emitted, rounds_ = tally([stat for _, stat in items])
        arms[arm] = dict(turns=len(items), turns_counted=len(usable), rounds=rounds_, tau=rounded(ratio((emitted, rounds_))),
                         tau_per_turn=distribution(taus), seats=seats,
                         worst_seat=min(seats.values()) if seats else None,
                         turns_under_rounds=len(items) - len(usable))
        uncapped = (sum(stat['uncapped_emitted'] for _, stat in items), sum(stat['uncapped_rounds'] for _, stat in items))
        arms[arm]['tau_uncapped'] = rounded(ratio(uncapped))

    # -- the gate-G inputs: the thinking-ON arms A1, A2, A4 -----------------------------------------------------------
    primary = [(key, stat) for key, stat in stats.items() if key[0] in PRIMARY_ARMS and stat['rounds']]
    items = []
    for key, stat in primary:
        turn = chosen[key]
        items.append((SET_OF_ARM[key[0]], turn.get('cluster') or key[1], stat['emitted'], stat['rounds']))
    clusters = cluster_tallies(items)
    set_tallies = dict((name, (sum(pair[0] for pair in pool), sum(pair[1] for pair in pool))) for name, pool in clusters.items())
    pooled = pooled_equal_weight(set_tallies)
    ci = bootstrap_ci(clusters, seed=seed)
    per_set = {}
    weighted = {}
    for name in SET_ORDER:
        sets = [(key, stat) for key, stat in primary if SET_OF_ARM[key[0]] == name]
        taus = [stat['tau'] for _, stat in sets if stat['rounds'] >= MIN_ROUNDS]
        per_set[name] = dict(distribution(taus), rounds=sum(stat['rounds'] for _, stat in sets),
                             tau=rounded(ratio(tally([stat for _, stat in sets]))) if sets else None)
        weighted[name] = [(stat['tau'], stat['rounds']) for _, stat in sets if stat['rounds'] >= MIN_ROUNDS]
    all_taus = [stat['tau'] for _, stat in primary if stat['rounds'] >= MIN_ROUNDS]
    buckets = {}
    for label in [label for _, label in BUCKETS]:
        sets = [stat for _, stat in primary if stat['bucket'] == label]
        taus = [stat['tau'] for stat in sets if stat['rounds'] >= MIN_ROUNDS]
        if sets:
            buckets[label] = dict(distribution(taus), rounds=sum(stat['rounds'] for stat in sets),
                                  tau=rounded(ratio(tally(sets))))
    regions = {}
    for _, stat in primary:
        for region, value in zip(stat['regions'], stat['values']):
            if region:
                pair = regions.setdefault(region, [0, 0])
                pair[0] += value
                pair[1] += 1
    seats = {}
    for _, stat in primary:
        for segment, value in zip(stat['segments'], stat['values']):
            if segment is not None:
                pair = seats.setdefault(segment, [0, 0])
                pair[0] += value
                pair[1] += 1
    seat_tau = dict(('seat_%d' % segment, rounded(pair[0] / float(pair[1]))) for segment, pair in sorted(seats.items()))
    worst8 = worst_of_k(weighted, 8, STRICT_BARS + K2_BARS, seed=seed)
    worst9 = worst_of_k(weighted, 9, STRICT_BARS, seed=seed + 1)
    p10 = quantile(all_taus, 0.10)
    long_bucket = buckets.get(LONG_BUCKET) or {}
    low_seat = min((pair[0] / float(pair[1]) for pair in seats.values() if pair[1]), default=None)
    k_inputs = dict(
        pooled_tau=rounded(pooled), ci95=ci, sets=per_set, per_turn=distribution(all_taus),
        share_turns_under_4_2=rounded(sum(1 for tau in all_taus if tau < LOW_TAU) / float(len(all_taus))) if all_taus else None,
        buckets=buckets, regions=dict((name, rounded(pair[0] / float(pair[1]))) for name, pair in regions.items()),
        seats=seat_tau, worst_of_8=worst8, worst_of_9=worst9,
        k1=dict(screen=screen(pooled, (K1_BAR,)), bar=K1_BAR,
                sanity_floor_ci_upper_below_5=bool(ci and ci['high'] is not None and ci['high'] < SANITY_FLOOR)),
        k2=dict(p10=screen(p10, K2_BARS), long_bucket_tau=screen(long_bucket.get('tau'), K2_BARS),
                long_bucket_p10=screen(long_bucket.get('p10'), K2_BARS), worst_seat=screen(low_seat, K2_BARS)))

    # -- thinking on against off (A1 against A5 over the same turns) -----------------------------------------------------
    pairs = []
    for (arm, ident), stat in stats.items():
        if arm == 'A5' and stat['rounds'] >= MIN_ROUNDS:
            other = stats.get(('A1', ident))
            if other and other['rounds'] >= MIN_ROUNDS:
                pairs.append((other['tau'], stat['tau']))
    thinking = dict(pairs=len(pairs), tau_on=rounded(mean(on for on, _ in pairs)), tau_off=rounded(mean(off for _, off in pairs)),
                    mean_difference=rounded(mean(off - on for on, off in pairs)),
                    ratio_off_over_on=rounded(mean(off for _, off in pairs) / mean(on for on, _ in pairs)) if pairs else None)

    # -- the calibration -------------------------------------------------------------------------------------------------
    a3 = [(key, stat) for key, stat in stats.items() if key[0] == 'A3' and stat['rounds']]
    reference = (manifest.get('a3_reference') or {}).get('pooled_tau')
    refs = [chosen[key].get('ref_tau') for key, _ in a3 if chosen[key].get('ref_tau')]
    if reference is None and refs and len(refs) == len(a3):
        reference = mean(refs)
    lab_a3 = ratio(tally([stat for _, stat in a3])) if a3 else None
    relative = (lab_a3 - reference) / float(reference) if lab_a3 is not None and reference else None
    calibration = dict(turns=len(a3), lab_tau=rounded(lab_a3), reference_tau=rounded(reference), relative=rounded(relative, 4),
                       verdict=('NOT_RUN' if not a3 else 'NOT_ESTABLISHED' if relative is None
                                else 'PASS' if abs(relative) <= A3_TOLERANCE else 'FAIL'))
    per_prompt = [(chosen[key]['ref_tau'], stat['tau']) for key, stat in a3 if chosen[key].get('ref_tau')]
    if per_prompt:
        calibration['per_prompt_relative_max'] = rounded(max(abs(lab - ref) / ref for ref, lab in per_prompt), 4)

    # -- the checks -------------------------------------------------------------------------------------------------------
    with_rounds = max(source['turns_with_rounds'], 1)
    agree = (source['stream_disagree'] == 0 and source['turns_with_rounds'] > 0
             and (source['phases_unique'] + source['phases_ambiguous']) / float(with_rounds) >= 0.95)
    lab_pos, pos_source = lab_positions(log_text)
    comparison = position_comparison(lab_pos, production_positions(counters))
    checks = dict(
        sources=dict(verdict='PASS' if agree else 'FAIL', **source),
        calibration=calibration['verdict'],
        image=('production' if info.get('production') else 'non-production'),
        arithmetic_diff_empty=not info.get('arithmetic_extra'),
        positions_1_3=comparison if comparison else dict(verdict='NOT_ESTABLISHED'),
        unattributed_requests=unattributed)
    counts = info.get('counts') or {}
    incomplete = [arm for arm, entry in counts.items() if entry.get('skipped') or entry.get('error')]
    complete = bool(stats) and not incomplete and all(arm in set(key[0] for key in stats) for arm in (info.get('arms') or ()))
    public = dict(
        image_tag=info.get('image_tag') if info.get('image_tag') and IMAGE_TAG.fullmatch(info.get('image_tag') or '') else None,
        label='production' if info.get('production') else 'non-production', complete=complete,
        run=dict((arm, dict(('turns_' + key, value) for key, value in sorted(entry.items()))) for arm, entry in counts.items()),
        arms=arms, k_inputs=k_inputs, thinking=thinking, calibration=calibration, checks=checks,
        positions=dict(source=pos_source, phases_records=phase_records_seen,
                       lab=[rounded(value, 4) for value in (lab_pos or [])][:15]))
    assert_public(public)
    private = dict(public=public, turns=[dict(arm=key[0], id=key[1], set=chosen[key].get('set'),
                                              cluster=chosen[key].get('cluster'), bucket=stat['bucket'], rounds=stat['rounds'],
                                              tau=rounded(stat['tau'])) for key, stat in sorted(stats.items())])
    return public, private, tapes


def write_json(path, value):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=1, sort_keys=True)
        handle.write('\n')


def summary_lines(public):
    """The counts and aggregates a log may carry (never an id, a text or a token)."""
    k = public['k_inputs']
    ci = k.get('ci95') or {}
    lines = ['[TAULAB-REPORT] label=%s complete=%s pooled_tau=%s ci95=%s..%s p10=%s' % (
        public['label'], public['complete'], k.get('pooled_tau'), ci.get('low'), ci.get('high'),
        (k.get('per_turn') or {}).get('p10'))]
    for arm, entry in sorted(public['arms'].items()):
        lines.append('[TAULAB-REPORT] arm %s: turns=%d rounds=%d tau=%s p10=%s p50=%s worst_seat=%s' % (
            arm, entry['turns'], entry['rounds'], entry['tau'], entry['tau_per_turn']['p10'], entry['tau_per_turn']['p50'],
            entry['worst_seat']))
    lines.append('[TAULAB-REPORT] calibration=%s lab=%s reference=%s sources=%s positions_1_3=%s' % (
        public['calibration']['verdict'], public['calibration']['lab_tau'], public['calibration']['reference_tau'],
        public['checks']['sources']['verdict'], public['checks']['positions_1_3'].get('verdict')))
    return lines


def build_and_write(results_dir, public_dir, manifest=None, counters=None, production=None, image_tag=None, seed=0, say=print,
                    info=None):
    """The report over a results directory: private files beside the turns, the public summary under `public_dir` (when given).
    -> the public summary."""
    info = dict(info or {})
    info.setdefault('production', production)
    info.setdefault('image_tag', image_tag)
    public, private, tapes = build(results_dir, manifest=manifest, counters=counters, info=info, seed=seed)
    write_json(os.path.join(results_dir, 'report.private.json'), private)
    with open(os.path.join(results_dir, 'tapes.jsonl'), 'w', encoding='utf-8') as handle:
        for tape in tapes:
            handle.write(json.dumps(tape, sort_keys=True) + '\n')
    if public_dir:
        write_json(os.path.join(public_dir, 'tau-lab-summary.json'), public)
    for line in summary_lines(public):
        say(line)
    return public


def main(argv=None, say=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--metrics', default=None, help='a scrape of the production /metrics: write its spec-decode counters '
                                                       '(--counters-out) and stop')
    parser.add_argument('--counters-out', default=None)
    parser.add_argument('--data', default=None, help='the data directory (its manifest carries the A3 reference)')
    parser.add_argument('--counters', default=None)
    parser.add_argument('--seed', type=int, default=0)
    options = parser.parse_args(argv)
    if options.metrics:
        if not options.counters_out:
            parser.error('--metrics needs --counters-out')
        with open(options.metrics, encoding='utf-8', errors='replace') as handle:
            counters = counters_from_metrics(handle.read())
        if counters is None:
            say('no spec-decode counters in the scrape')
            return 2
        write_json(options.counters_out, counters)
        say('[TAULAB-REPORT] counters: positions=%d drafts=%d' % (len(counters['accepted_per_pos']), counters['num_drafts']))
        return 0
    if not options.results:
        parser.error('--results is required')
    manifest = {}
    if options.data and os.path.isfile(os.path.join(options.data, 'manifest.json')):
        with open(os.path.join(options.data, 'manifest.json'), encoding='utf-8') as handle:
            manifest = json.load(handle)
    counters = None
    if options.counters:
        with open(options.counters, encoding='utf-8') as handle:
            counters = json.load(handle)
    info = {}
    run_path = os.path.join(options.results, 'run.json')
    if os.path.isfile(run_path):
        with open(run_path, encoding='utf-8') as handle:
            info = json.load(handle)
    public = build_and_write(options.results, options.out, manifest=manifest, counters=counters, seed=options.seed, say=say,
                             info=info)
    return 0 if public['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
