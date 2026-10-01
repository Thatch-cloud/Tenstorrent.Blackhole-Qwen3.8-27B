"""The lanes window's gate plans: D0 alone, the round-time ladder, and one fast lane beside one to three standard lanes (host only).

The target for coding is ONE fast lane at >= 150 tok/s for a single user, running at the same time as standard lanes at >= 75 tok/s
each, on the four cards at TP4. Three questions decide whether the four-card stack gets there, and each has a plan:

  lanes-exact   EXACTNESS. Three arms on the same four real-text prompts (4,096 / 16,384 / 32,768 / 60,000 tokens, 1,024 out), every
                audit on: `ref-solo` (c2-packed-tp4-gate: each user alone on the per-request engines), `d0-solo` (c2-packed-tp4-solo-gate:
                each user alone on the one-user 16-row block, D0) and `lanes-mixed` (c2-packed-tp4-lanes-gate: user 0 marked fast beside
                three standard users, the ratio SWEPT on the one boot by QWEN_FAST_LANE_SCHEDULE so every frame shape runs). Every user's
                text under the lanes, and under D0, must equal its text alone on the per-request engines (the strict policy, a first
                divergence re-runs the pair once). The lane runtime must have engaged, granted the fast lane, and alternated: a run
                where no solo round ran proved nothing about the lanes.

  round-timing  ROUND TIMES, audits off (the timing profiles). `padded-4k` and `padded-32k`: four users whose budgets end one after
                another (600 / 1200 / 1800 / 2400 tokens, every stream to its budget), so the rounds run at four, three, two and one
                live users: the padded and full packed-round times P(4), P(3), P(2) and the lone user on today's per-request engines,
                in ONE boot per context. `base-lone-4k` and `d0-lone-4k` / `d0-lone-32k`: four real-text users one at a time on the
                per-request engines and on D0. What D0 gives a lone coding user: tokens per round and tokens per second, at the same
                prompts, against the engines it replaces.

  lanes-timing  BOTH BARS, audits off (c2-packed-tp4-lanes-time-gate). One fast user beside 3, 2 and 1 standard users at 4,096
                tokens and beside 3 at 32,768, real text to its natural end: the ratio swept on the boot (k = 0, 0.5, 1, 1.5, 2, then
                the controller), each stretch read by lanes_report - per-user steady rate over the frame, P, F, sigma, tau per lane,
                beside the design's frame arithmetic. `lanes-timing-more` adds 2 and 1 standard users at 32,768 and 3 at 120,000.

A missed bar is a RESULT, not a failure: the timing plans PASS when every arm was healthy and its cells were read, print every
stretch and the best cell per standard-user count, and say MET / MISSED against 150 / 75. NOT_EXERCISED says what could not be read
(a stretch with too few rounds, a lane that never alternated). FAIL is a broken arm, a wrong text, a desync.

Nothing here runs a rig: plan_arms builds the arms (harness arguments, profile, environment) and run_plan judges what came back;
c2_serving_gate wires both in for the plans in c2_serving_job.LANES_GATE_PLANS.
"""

import collections
import os

import lanes_report

EXACT, ROUND, LANES, LANES_MORE = 'lanes-exact', 'round-timing', 'lanes-timing', 'lanes-timing-more'
PLANS = (EXACT, ROUND, LANES, LANES_MORE)

REFERENCE = 'c2-packed-tp4-gate'                 # audited, no lane, no D0: the per-request engines
D0_EXACT = 'c2-packed-tp4-solo-gate'             # audited, D0
LANES_EXACT = 'c2-packed-tp4-lanes-gate'         # audited, D0 and the lanes
TIMING = 'c2-packed-tp4-time-gate'               # audits off, no lane, no D0
D0_TIMING = 'c2-packed-tp4-solo-time-gate'       # audits off, D0
LANES_TIMING = 'c2-packed-tp4-lanes-time-gate'   # audits off, D0 and the lanes

# What each profile must (not) carry, checked before any container starts: a wrong image or a profile edited away from its plan.
EXPECT = {
    REFERENCE: dict(solo=False, lanes=False, audits=True),
    D0_EXACT: dict(solo=True, lanes=False, audits=True),
    LANES_EXACT: dict(solo=True, lanes=True, audits=True),
    TIMING: dict(solo=False, lanes=False, audits=False),
    D0_TIMING: dict(solo=True, lanes=False, audits=False),
    LANES_TIMING: dict(solo=True, lanes=True, audits=False),
}
SCHEDULE_ENV, RATIO_ENV = 'QWEN_FAST_LANE_SCHEDULE', 'QWEN_FAST_LANE_RATIO'
GATE_KNOBS = (SCHEDULE_ENV, RATIO_ENV)           # what an arm may add with -e beyond c2_serving_gate.ARM_ENV_NAMES

EXACT_LENGTHS = (4096, 16384, 32768, 60000)
EXACT_MAX_TOKENS = 1024
# Descending k: the fast user's rounds are what runs out first (a stretch at k costs (1 + k) of them per frame, and text ends where it
# ends), so the ratios where 150 could be met are read first and the cheap packed-only end is the one left unread.
EXACT_SCHEDULE = '2@25,1@25,0.5@25,0@25,auto'    # 100 frames with both lanes live, then the controller
SWEEP_SCHEDULE = '2@25,1.5@25,1@25,0.5@25,0@25,auto'
LADDER_BUDGETS = '3:600,2:1200,1:1800'           # user 3 ends first (live 4 -> 3), then user 2, then user 1; user 0 runs to LADDER_MAX
LADDER_MAX_TOKENS = 2400
LONE_MAX_TOKENS = 1024
LONE_USERS = 4
TIMING_MAX_TOKENS = 4096
TIMING_MAX_TOKENS_131K = 2048
CONTEXTS = dict(k4=4096, k32=32768, k120=120000)

# docker limits per arm (an engine build up to 30 minutes, then the streams) and the streams' inactivity limit
ARM_SECONDS = {EXACT: 3300, ROUND: 3600, LANES: 3600, LANES_MORE: 3600}
STREAM_SECONDS = {EXACT: 1800, ROUND: 1800, LANES: 1800, LANES_MORE: 2400}
MIN_ALTERNATION_ROUNDS = 20    # solo rounds AND packed rounds an exactness arm must have run under the lanes
MIN_LIVE_ROUNDS = 24           # a ladder phase's rounds before its median is read


def profile_flags(profiles, name):
    entry = (profiles or {}).get('profiles', {}).get(name)
    env = (entry or {}).get('env') or {}
    return entry, env


def need_profile(g, profiles, name, plan):
    """The image's profile `name` exists and is the profile this plan expects (D0 flag, lane flag, audits): PlanError before any
    container starts."""
    entry, env = profile_flags(profiles, name)
    if entry is None:
        raise g.PlanError('%s serves profile %s, which the image\'s profiles do not carry (the lanes window\'s image builds them)'
                          % (plan, name))
    expect = EXPECT[name]
    solo, lanes = env.get('QWEN_FAST_SOLO_LANE') == '1', env.get('QWEN_FAST_LANE') == '1'
    audits = env.get('QWEN_FAST_VERIFY_T1_AUDIT') == '1' and env.get('QWEN_FAST_VERIFY_T2_AUDIT') == '1'
    wrong = []
    if solo != expect['solo']:
        wrong.append('QWEN_FAST_SOLO_LANE is %s' % ('on' if solo else 'off'))
    if lanes != expect['lanes']:
        wrong.append('QWEN_FAST_LANE is %s' % ('on' if lanes else 'off'))
    if audits != expect['audits']:
        wrong.append('the verify audits are %s' % ('on' if audits else 'off'))
    if wrong:
        raise g.PlanError('%s: profile %s is not the one this plan measures (%s)' % (plan, name, '; '.join(wrong)))
    if entry.get('gate_only') is not True:
        raise g.PlanError('%s: profile %s is not gate only: the lanes window never serves traffic' % (plan, name))


def sized(g, profiles, name, plan, lengths, budget):
    """(common arguments on `name`, the lengths text) as the matrix builds them."""
    common, text, _context = g.sized_arm_args(name, profiles, plan, list(lengths), budget)
    return common, text


def timing_env(env=()):
    """A timing arm adds no audit (the profile's own audits are off); only the lane knobs."""
    return tuple(env)


def exact_arms(g, profiles, lengths, notes):
    seconds = ARM_SECONDS[EXACT]
    lengths = tuple(lengths or EXACT_LENGTHS)
    if len(lengths) < 2 or len(lengths) > 4:
        raise g.PlanError('%s: needs two to four users (one fast, the rest standard), got %d lengths' % (EXACT, len(lengths)))
    if lengths != EXACT_LENGTHS:
        notes.append('%s: lengths %s replace the default %s' % (EXACT, list(lengths), list(EXACT_LENGTHS)))
    for name in (REFERENCE, D0_EXACT, LANES_EXACT):
        need_profile(g, profiles, name, EXACT)
    users = str(len(lengths))
    arms = []
    for arm, name, role in (('ref-solo', REFERENCE, 'solo'), ('d0-solo', D0_EXACT, 'd0')):
        common, text = sized(g, profiles, name, EXACT, lengths, EXACT_MAX_TOKENS)
        arms.append(g.Arm(arm, common + ['--prompt-lengths', text, '--max-tokens', str(EXACT_MAX_TOKENS), '--users', '1',
                                         '--sequential-users', users], seconds, profile=name, env=g.s2_env(profiles, name),
                          rerun=True, judged=False, role=role))
    common, text = sized(g, profiles, LANES_EXACT, EXACT, lengths, EXACT_MAX_TOKENS)
    arms.append(g.Arm('lanes-mixed', common + ['--prompt-lengths', text, '--max-tokens', str(EXACT_MAX_TOKENS), '--users', users,
                                                 '--stagger', str(g.STAGGER), '--user-lane', '0:fast'], seconds,
                      profile=LANES_EXACT, env=g.s2_env(profiles, LANES_EXACT) + ((SCHEDULE_ENV, EXACT_SCHEDULE),), rerun=True,
                      judged=False, role='lanes'))
    return arms


def round_arms(g, profiles, lengths, notes):
    seconds = ARM_SECONDS[ROUND]
    if lengths:
        notes.append('%s: --lengths is not used (the contexts are the design\'s 4,096 and 32,768)' % ROUND)
    for name in (TIMING, D0_TIMING):
        need_profile(g, profiles, name, ROUND)
    arms = []
    for label, context in (('4k', CONTEXTS['k4']), ('32k', CONTEXTS['k32'])):
        common, text = sized(g, profiles, TIMING, ROUND, (context,) * 4, LADDER_MAX_TOKENS)
        arms.append(g.Arm('padded-' + label, common + ['--prompt-lengths', text, '--max-tokens', str(LADDER_MAX_TOKENS), '--users',
                                                       '4', '--stagger', str(g.STAGGER), '--user-ignore-eos', '0,1,2,3',
                                                       '--user-max-tokens', LADDER_BUDGETS], seconds, profile=TIMING,
                          env=timing_env(), rerun=False, judged=False, role='ladder'))
    for arm, name, context, role in (('base-lone-4k', TIMING, CONTEXTS['k4'], 'base'), ('d0-lone-4k', D0_TIMING, CONTEXTS['k4'], 'd0'),
                                     ('d0-lone-32k', D0_TIMING, CONTEXTS['k32'], 'd0')):
        common, text = sized(g, profiles, name, ROUND, (context,) * LONE_USERS, LONE_MAX_TOKENS)
        arms.append(g.Arm(arm, common + ['--prompt-lengths', text, '--max-tokens', str(LONE_MAX_TOKENS), '--users', '1',
                                         '--sequential-users', str(LONE_USERS)], seconds, profile=name, env=timing_env(),
                          rerun=False, judged=False, role=role))
    return arms


def lanes_arm(g, profiles, plan, label, standard, context, budget):
    """One timing arm: one fast user (user 0) and `standard` standard users, every prompt `context` tokens, real text to its natural
    end, the ratio swept on the boot."""
    users = 1 + standard
    common, text = sized(g, profiles, LANES_TIMING, plan, (context,) * users, budget)
    return g.Arm('lanes-n%d-%s' % (standard, label),
                 common + ['--prompt-lengths', text, '--max-tokens', str(budget), '--users', str(users), '--stagger',
                           str(g.STAGGER), '--user-lane', '0:fast'], ARM_SECONDS[plan], profile=LANES_TIMING,
                 env=timing_env(((SCHEDULE_ENV, SWEEP_SCHEDULE),)), rerun=False, judged=False, role='timing')


def lanes_arms(g, profiles, plan, lengths, notes):
    if lengths:
        notes.append('%s: --lengths is not used (the contexts are the design\'s)' % plan)
    need_profile(g, profiles, LANES_TIMING, plan)
    if plan == LANES:
        return [lanes_arm(g, profiles, plan, '4k', 3, CONTEXTS['k4'], TIMING_MAX_TOKENS),
                lanes_arm(g, profiles, plan, '4k', 2, CONTEXTS['k4'], TIMING_MAX_TOKENS),
                lanes_arm(g, profiles, plan, '4k', 1, CONTEXTS['k4'], TIMING_MAX_TOKENS),
                lanes_arm(g, profiles, plan, '32k', 3, CONTEXTS['k32'], TIMING_MAX_TOKENS)]
    return [lanes_arm(g, profiles, plan, '32k', 2, CONTEXTS['k32'], TIMING_MAX_TOKENS),
            lanes_arm(g, profiles, plan, '32k', 1, CONTEXTS['k32'], TIMING_MAX_TOKENS),
            lanes_arm(g, profiles, plan, '120k', 3, CONTEXTS['k120'], TIMING_MAX_TOKENS_131K)]


def plan_arms(g, plan, profiles, lengths, notes):
    """The arms of one lanes plan, in order (c2_serving_gate.Arm objects)."""
    notes = notes if notes is not None else []
    if plan == EXACT:
        return exact_arms(g, profiles, lengths, notes)
    if plan == ROUND:
        return round_arms(g, profiles, lengths, notes)
    if plan in (LANES, LANES_MORE):
        return lanes_arms(g, profiles, plan, lengths, notes)
    raise g.PlanError('%r is not a lanes plan (%s)' % (plan, ', '.join(PLANS)))


# --- judging ------------------------------------------------------------------------------------------------------------

def lane_lines_in(log_text, marker):
    return [line for line in (log_text or '').splitlines() if marker in line]


def runner_log(runner, arm):
    try:
        with open(os.path.join(runner.results, arm, 'server.log'), errors='replace') as handle:
            return handle.read()
    except OSError:
        return None


def lanes_of(report):
    return (report or {}).get('lanes')


def lane_problems(label, report):
    """What fails an arm that ran the lane runtime: the report's own problems, or no lanes record at all."""
    lanes = lanes_of(report)
    if lanes is None:
        return ['%s: no lanes record (report[\'lanes\']): the harness mounted is not the lanes window\'s, or the runtime never engaged'
                % label]
    return ['%s: lanes: %s' % (label, problem) for problem in lanes.get('problems') or []]


def leak_problems(label, log_text, spec_lanes, solo_lane):
    """A profile without the lane (or D0) leaves the lane's (D0's) lines out of its log: an arm that shows them ran another path."""
    problems = []
    if not spec_lanes:
        leaked = [line for line in (log_text or '').splitlines() if '[LANE-' in line or '[LANE] engaged' in line]
        if leaked:
            problems.append('%s: the profile leaves QWEN_FAST_LANE off, but the server log carries lane lines (%s): this is not the '
                            'profile\'s path' % (label, leaked[0].strip()[:120]))
    if not solo_lane:
        leaked = lane_lines_in(log_text, '[SOLO-LANE]')
        if leaked:
            problems.append('%s: the profile leaves QWEN_FAST_SOLO_LANE off, but the server log carries solo-lane lines (%s)'
                            % (label, leaked[0].strip()[:120]))
    return problems


def run_exact(g, plan, runner, arms):
    """lanes-exact: the lanes' and D0's texts against the per-request engines', under the strict policy, a first divergence re-running
    the pair once; and that the runtimes engaged and alternated (else the texts prove nothing about them)."""
    by_role = {spec.role: spec for spec in arms}
    ref_spec, d0_spec, lanes_spec = by_role['solo'], by_role['d0'], by_role['lanes']
    wants = {name: g.asked(spec[1]) for name, spec in (('ref', ref_spec), ('d0', d0_spec), ('lanes', lanes_spec))}
    relaxation = runner.relaxation(g.spec_profile(runner, lanes_spec))
    ref, d0, lanes = g.run_arm(runner, plan, ref_spec), g.run_arm(runner, plan, d0_spec), g.run_arm(runner, plan, lanes_spec)

    pairs = dict(d0=(d0_spec, d0), lanes=(lanes_spec, lanes))
    results, ref_again = {}, []
    for name, (spec, report) in pairs.items():
        results[name] = g.matrix_verdict(report, ref, None, wants={'concurrent': wants[name], 'solo': wants['ref']},
                                         relaxation=relaxation)
        if results[name]['verdict'] == 'RERUN':
            runner.log('[C2-GATE] %s: %s diverged from the per-request engines - re-running it and its reference once (the '
                       'exactness policy)' % (plan, name))
            if not ref_again:
                ref_again.append(g.run_arm(runner, plan, ref_spec, '-rerun'))     # the reference runs again once, whoever asks
            again = (g.run_arm(runner, plan, spec, '-rerun'), ref_again[0])
            results[name] = g.matrix_verdict(report, ref, again if None not in again else None,
                                             wants={'concurrent': wants[name], 'solo': wants['ref']}, relaxation=relaxation)
            if None in again:
                results[name].update(verdict='FAIL', reason='a re-run arm left no gate report')
    problems, shortfalls, facts = [], [], {}
    for label, report in (('ref-solo', ref), ('d0-solo', d0), ('lanes-mixed', lanes)):
        if report is not None:
            problems += g.s2_g4_problems(label, report)
    problems += lane_problems('lanes-mixed', lanes) if lanes is not None else []
    logs = {name: runner_log(runner, name) for name in ('ref-solo', 'd0-solo', 'lanes-mixed')}
    problems += leak_problems('ref-solo', logs['ref-solo'], False, False)
    problems += leak_problems('d0-solo', logs['d0-solo'], False, True)
    if d0 is not None and not lane_lines_in(logs['d0-solo'], '[SOLO-LANE] round'):
        shortfalls.append('d0-solo: no "[SOLO-LANE] round" line: the one-user block never served a lone round, so its text is not D0\'s')
    if lanes is not None:
        record = lanes_of(lanes) or {}
        kinds = record.get('round_kinds') or {}
        facts['lanes_rounds'] = kinds
        if (kinds.get('solo') or 0) < MIN_ALTERNATION_ROUNDS or (kinds.get('packed') or 0) < MIN_ALTERNATION_ROUNDS:
            shortfalls.append('lanes-mixed: %s solo and %s packed rounds ran, fewer than %d of each: the lanes barely alternated' % (
                kinds.get('solo'), kinds.get('packed'), MIN_ALTERNATION_ROUNDS))
        if not [admit for admit in record.get('admits') or [] if admit.get('granted') == 'fast']:
            problems.append('lanes-mixed: no request was granted the fast lane')
        if 'schedule:' not in (record.get('engaged_line') or ''):
            problems.append('lanes-mixed: the engaged line does not show the ratio schedule: %s never reached the worker'
                            % SCHEDULE_ENV)
        stretches = [cell for cell in record.get('stretches') or [] if cell.get('read')]
        facts['ratios_read'] = sorted(set(cell['ratio'] for cell in stretches))
        missing = [ratio for ratio in ('0', '0.5', '1', '2') if ratio not in facts['ratios_read']]
        if missing:
            shortfalls.append('lanes-mixed: ratios %s never held 24 frames with every lane live (read: %s): the sweep did not reach '
                              'them' % (', '.join(missing), ', '.join(facts['ratios_read']) or 'none'))
    verdicts = [results['d0']['verdict'], results['lanes']['verdict']]
    lines = []
    for name in ('d0', 'lanes'):
        lines += ['%s vs per-request engines: %s' % (name, line) for line in results[name].get('lines') or []]
    result = g.with_checks(dict(verdict=g.worst(verdicts), arms=results), problems, shortfalls, facts)
    result['lines'] = lines + list(result.get('lines') or [])
    if lanes is not None and lanes_of(lanes):
        result['lines'] += lanes_report.verdict_lines(lanes_of(lanes))
    return result


def ladder_facts(report):
    """{live: {rounds, median_round_ms, tokens_per_user_per_round}} of a ladder arm's packed rounds, from the acceptance report's split
    by live count (round_split), keyed 'live/packed'."""
    split = ((report or {}).get('acceptance') or {}).get('rounds_by_live') or {}
    out = collections.OrderedDict()
    for group in split.get('groups') or []:
        key = '%d%s' % (group.get('live'), {True: 'p', False: 's', None: '?'}[group.get('packed')])
        out[key] = dict(rounds=group.get('rounds'), median_round_ms=group.get('median_round_ms'),
                        tokens_per_user_per_round=group.get('tokens_per_user_per_round'))
    return out


def lone_facts(report):
    """Each sequential stream's own rate and tokens per round (lanes_report.stream_rates) and their means."""
    rates = lanes_report.stream_rates((report or {}).get('streams'))
    good = [entry for entry in rates if entry.get('rate') is not None]
    return dict(users=rates, mean_rate=round(sum(e['rate'] for e in good) / len(good), 2) if good else None,
                mean_tau=round(sum(e['tau'] for e in good) / len(good), 3) if good else None)


def run_timing(g, plan, runner, arms):
    """round-timing, lanes-timing and lanes-timing-more: every arm healthy, every cell read, every number printed. Never FAIL on a
    missed bar."""
    problems, shortfalls, lines, facts = [], [], [], collections.OrderedDict()
    reports = {}
    for spec in arms:
        name = spec[0]
        report = g.run_arm(runner, plan, spec)
        reports[name] = report
        if report is None:
            problems.append('%s: the arm left no gate report' % name)
            continue
        problems += g.arm_problems(name, report, g.asked(spec[1]))
        role = spec.role
        log_text = runner_log(runner, name)
        solo_lane = spec.profile in (D0_TIMING, LANES_TIMING)
        problems += leak_problems(name, log_text, spec.profile == LANES_TIMING, solo_lane)
        if role == 'ladder':
            ladder = ladder_facts(report)
            facts[name] = dict(ladder=ladder)
            for key, entry in ladder.items():
                lines.append('%s: %s rounds %s median %s ms, %s tokens per user per round' % (
                    name, key, entry['rounds'], entry['median_round_ms'], entry['tokens_per_user_per_round']))
            seen = {key for key, entry in ladder.items() if (entry['rounds'] or 0) >= MIN_LIVE_ROUNDS}
            missing = [live for live in ('4p', '3p', '2p') if live not in seen]
            if missing:
                shortfalls.append('%s: no %d-round phase read at %s live (packed): P(n) unread there' % (
                    name, MIN_LIVE_ROUNDS, ', '.join(missing)))
        elif role in ('base', 'd0'):
            facts[name] = lone_facts(report)
            lines.append('%s: lone user mean %s tok/s at %s tokens per round (%s)' % (
                name, facts[name]['mean_rate'], facts[name]['mean_tau'], ', '.join(
                    '%s:%s' % (entry['user'], entry['rate']) for entry in facts[name]['users'])))
            if facts[name]['mean_rate'] is None:
                shortfalls.append('%s: no stream carried a chunk timeline: the lone rate is unread' % name)
            if role == 'd0' and not lane_lines_in(log_text, '[SOLO-LANE] round'):
                shortfalls.append('%s: no "[SOLO-LANE] round" line: D0 never served a lone round' % name)
        elif role == 'timing':
            problems += lane_problems(name, report)
            record = lanes_of(report) or {}
            facts[name] = dict(best=[dict(users=users, ratio=cell['ratio'], fast=cell.get('fast'),
                                          standard_min=cell.get('standard_min'), label=label)
                                     for users, cell, label in lanes_report.best_cells(record)],
                               unread=record.get('unread'), round_kinds=record.get('round_kinds'))
            lines += ['%s: %s' % (name, line) for line in lanes_report.verdict_lines(record)]
            for entry in facts[name]['best']:
                verdict = {'both': 'MET', 'standard-only': 'MISSED (fast)', 'neither': 'MISSED'}[entry['label']]
                lines.append('%s TARGET n=%d: fast %s / standard >= %s at ratio %s: %s' % (
                    name, entry['users'], entry['fast'], entry['standard_min'], entry['ratio'], verdict))
            if not facts[name]['best']:
                shortfalls.append('%s: no stretch was read (a stretch needs %d packed rounds with every lane live)' % (
                    name, lanes_report.MIN_STRETCH_ROUNDS))
            for cell in record.get('unread') or []:
                shortfalls.append('%s: ratio %s at %d live: %d packed rounds, unread' % (
                    name, cell['ratio'], cell['live'], cell['packed_rounds']))
            if 'schedule:' not in (record.get('engaged_line') or ''):
                problems.append('%s: the engaged line does not show the ratio schedule: %s never reached the worker' % (
                    name, SCHEDULE_ENV))
    result = dict(verdict='PASS', facts=facts)
    if plan == ROUND:
        lines += compare_d0(facts)
    result = g.with_checks(result, problems, shortfalls, facts)
    result['lines'] = lines + list(result.get('lines') or [])
    return result


def compare_d0(facts):
    """D0 against the per-request engines at 4,096 (the same four prompts): the lone user's rate and tokens per round, side by side."""
    base, d0 = facts.get('base-lone-4k'), facts.get('d0-lone-4k')
    if not base or not d0 or not base.get('mean_rate') or not d0.get('mean_rate'):
        return []
    return ['D0 vs per-request engines at 4096: %s vs %s tok/s (x%.2f), %s vs %s tokens per round' % (
        d0['mean_rate'], base['mean_rate'], d0['mean_rate'] / base['mean_rate'], d0['mean_tau'], base['mean_tau'])]


def run_plan(g, plan, runner, arms):
    if plan == EXACT:
        return run_exact(g, plan, runner, arms)
    return run_timing(g, plan, runner, arms)
