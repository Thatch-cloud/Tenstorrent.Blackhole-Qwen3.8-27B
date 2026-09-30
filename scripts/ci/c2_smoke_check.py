"""c2_smoke_check: the smoke's stop conditions, enforced (the four-card speed window's K1 and K2).

c2_serving_smoke.py records every test's exception as an `error` entry and exits 0, and the quad smoke step fails only
when the container exits or never becomes ready, so a bad smoke used to let a window chain go on. This reads the smoke's
own SMOKE_JSON line and the container log and exits non-zero on:
  - an `error` entry in a core test (warmup, warm_lifecycle, coding, concurrent4, long_real_text), a warmup status other
    than 200, or a stream (coding, a concurrent user, a warm_lifecycle row) that produced no tokens or no finish;
  - garbage text: a coding or concurrent stream whose kept text is empty, mostly non-printable, or a repetition of a few
    characters;
  - an audit mismatch line in the container log ('audit mismatch': the verify t1 audit, round b1, the extent audit);
  - a ramp commit above --max-ramp-kv-ms (default 50) when the served profile has the drafter's K/V slide on
    (QWEN_FAST_TP_KV_SLIDE=1): the MEDIAN, over the [PACKED-PUBLISH] rounds with a commit, of each round's largest
    prepare_history entry. The median, so the attach's few compile-bearing rounds do not fail it; the eager chain's ~310 ms
    per ramp user in v140 would;
  - the batched draft, when the smoke ran the `concurrent4_steady` test (four users past the 2,048-row draft window from the
    first round, the only mix a pair or the quad can serve): what the profile asked for must have run. QWEN_FAST_QUAD_DRAFT=1:
    the quad's marker exactly once, at least one round the quad served, no fallback and no disable line. Off: no quad line, and
    at least one round two packed pairs served (a comparison arm that never batched compares nothing). A
    [DRAFT-SINGLES-AUDIT] line with equal=0 or a [QUAD-AUDIT] line with equal=0 fails; a profile with
    QWEN_FAST_DRAFT_SINGLES_AUDIT set and no audit line fails;
  - the fused commit (QWEN_FAST_FUSED_COMMIT=1, the round-fence plan's H1b; at four cards fused_commit_tp), by the gate's own H1b
    rules (lever_n_m3native_gate.h1b_report: a refused build, an engaged line that disagrees with the flags, no fused publication,
    a refusal reason but `ramp` and `parity`, a discard after an in-place slide, an audit count that is not the fused count, an audit
    mismatch) and, here, that the engaged line is there once with the trace count the flags imply (users x (1 + 16 prefixes) in
    place, users otherwise), every audit line checked the items the width implies (10 banks x chips, twice in place: 80 at four
    cards), and - in a smoke that ran `concurrent4_steady` - that a round of four fused publications happened and, under
    QWEN_FAST_FUSED_COMMIT_LIVE_BANKS, that the live-bank marker was logged. A profile with the flag off must log no fused line.

  python c2_smoke_check.py --smoke-log smoke.log --container-log container.log --profile P [--profiles qwen_c2_profiles.json]
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

CORE = ('warmup', 'warm_lifecycle', 'coding', 'concurrent4', 'concurrent4_steady', 'long_real_text')
PUBLISH = re.compile(r'\[PACKED-PUBLISH\] round=\d+ stages=\{.*?prepare_history: \[([0-9.,\s]*)\]')
MISMATCH = re.compile(r'audit mismatch', re.IGNORECASE)
SLIDE_FLAG = 'QWEN_FAST_TP_KV_SLIDE'
QUAD_FLAG = 'QWEN_FAST_QUAD_DRAFT'
SINGLES_AUDIT_FLAG = 'QWEN_FAST_DRAFT_SINGLES_AUDIT'
FUSED_FLAG = 'QWEN_FAST_FUSED_COMMIT'
FUSED_INPLACE_FLAG = 'QWEN_FAST_FUSED_COMMIT_INPLACE'
FUSED_LIVE_BANKS_FLAG = 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'
FUSED_AUDIT_FLAG = 'QWEN_FAST_FUSED_COMMIT_AUDIT'
FUSED_PREFIXES = 16          # one slide trace per (segment, accepted prefix 1..rows_per_user) in place
FUSED_LINES = ('[PINDIAG] fused commit engaged', '[PINDIAG] fused commit refused', '[PACKED-FUSED] round=',
               '[PACKED-FUSED-AUDIT] round=')
STEADY_TEST = 'concurrent4_steady'
# One line per packed round the coordinator selected (dflash_packed_proposal_coordinator.SELECT_LINE, QWEN_FAST_PACKED_AUDIT):
# the quad's one group of four slots, or two packed pairs.
QUAD_ROUND = re.compile(r'\[PACKED-SELECT\] round=\d+ pairs=\[\[0, 1, 2, 3\]\] users=4 ')
PAIR_ROUND = re.compile(r'\[PACKED-SELECT\] round=\d+ pairs=\[\[0, 1\], \[2, 3\]\] users=4 ')
QUAD_MARKER = '[PINDIAG] quad draft engaged'
QUAD_DISABLED = '[PINDIAG] quad draft disabled'
QUAD_FALLBACK = '[QUAD-DRAFT] fallback'
QUAD_LINE = re.compile(r'\[QUAD-DRAFT\] round=\d+ built=')
QUAD_AUDIT = re.compile(r'\[QUAD-AUDIT\] round=\S+ equal=([01]) ')
SINGLES_AUDIT_LINE = re.compile(r'\[DRAFT-SINGLES-AUDIT\] round=\S+ group=\[[0-9, ]*\] equal=([01]) stage=(\S+) ')
DEFAULT_PROFILES = Path(__file__).resolve().parent / 'qwen_c2_profiles.json'


def smoke_results(text):
    """The dict of the last SMOKE_JSON line of a smoke log, or None."""
    found = None
    for line in text.splitlines():
        marker = line.find('SMOKE_JSON ')
        if marker >= 0:
            try:
                found = json.loads(line[marker + len('SMOKE_JSON '):])
            except ValueError:
                pass
    return found


def garbage(text):
    """A reason the kept text of a stream is not an answer, or None."""
    if not text or not text.strip():
        return 'empty text'
    sample = text[:300]
    printable = sum(1 for char in sample if char.isprintable() or char in '\n\t')
    if printable < 0.9 * len(sample):
        return 'mostly non-printable'
    if len(sample) >= 100 and len(set(sample)) <= 4:
        return 'a repetition of %d characters' % len(set(sample))
    return None


def stream_problems(name, stream, text_needed=True):
    problems = []
    if not isinstance(stream, dict):
        return ['%s: no result' % name]
    if 'error' in stream:
        return ['%s: %s' % (name, stream['error'])]
    if not stream.get('tokens'):
        problems.append('%s: no tokens' % name)
    if stream.get('finish') is None and 'text' in stream:
        problems.append('%s: no finish reason' % name)
    if text_needed and 'text' in stream:
        reason = garbage(stream['text'])
        if reason:
            problems.append('%s: garbage text (%s)' % (name, reason))
    return problems


def smoke_problems(results):
    problems = []
    if results is None:
        return ['no SMOKE_JSON line in the smoke log']
    for name in CORE:
        entry = results.get(name)
        if entry is None:
            continue
        if 'error' in entry:
            problems.append('%s: %s' % (name, entry['error']))
    if 'warmup' in results and 'error' not in results['warmup'] and results['warmup'].get('value') != 200:
        problems.append('warmup: status %s' % results['warmup'].get('value'))
    if 'coding' in results and 'error' not in results['coding']:
        problems += stream_problems('coding', results['coding'])
    for name in ('concurrent4', STEADY_TEST):
        if name in results and 'error' not in results[name]:
            for index, user in enumerate(results[name].get('users') or []):
                problems += stream_problems('%s user %d' % (name, index), user)
    if 'warm_lifecycle' in results and 'error' not in results['warm_lifecycle']:
        for label, rows in results['warm_lifecycle'].items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not row.get('tokens'):
                    problems.append('warm_lifecycle %s at %s characters: no tokens' % (label, row.get('chars')))
    return problems


def ramp_kv_median(log_text):
    """(median of each round's largest prepare_history entry, rounds counted), or (None, 0)."""
    largest = []
    for match in PUBLISH.finditer(log_text):
        values = [float(part) for part in match.group(1).replace(' ', '').split(',') if part]
        if any(values):
            largest.append(max(values))
    return (statistics.median(largest), len(largest)) if largest else (None, 0)


def slide_on(profile, profiles_path):
    document = json.loads(Path(profiles_path).read_text(encoding='utf-8'))
    entry = (document.get('profiles') or {}).get(profile)
    if entry is None:
        raise ValueError('profile %s is not in %s' % (profile, profiles_path))
    return (entry.get('env') or {}).get(SLIDE_FLAG) == '1'


def profile_env(profile, profiles_path):
    document = json.loads(Path(profiles_path).read_text(encoding='utf-8'))
    entry = (document.get('profiles') or {}).get(profile)
    if entry is None:
        raise ValueError('profile %s is not in %s' % (profile, profiles_path))
    return entry.get('env') or {}


def draft_facts(container_text):
    """What the container log says about the batched draft: rounds served by the quad and by two packed pairs, the quad's marker,
    fallback and disable lines, and the two audits' lines."""
    singles = SINGLES_AUDIT_LINE.findall(container_text)
    quad_audit = QUAD_AUDIT.findall(container_text)
    return dict(quad_rounds=len(QUAD_ROUND.findall(container_text)), pair_rounds=len(PAIR_ROUND.findall(container_text)),
                quad_markers=container_text.count(QUAD_MARKER), quad_disabled=container_text.count(QUAD_DISABLED),
                quad_fallbacks=container_text.count(QUAD_FALLBACK), quad_lines=len(QUAD_LINE.findall(container_text)),
                quad_audits=len(quad_audit), quad_audits_unequal=sum(1 for equal in quad_audit if equal != '1'),
                singles_audits=len(singles), singles_audits_unequal=sum(1 for equal, _ in singles if equal != '1'),
                singles_audit_stages=sorted({stage for equal, stage in singles if equal != '1'})[:4])


def draft_problems(facts, env, steady):
    """The problems the profile's batched-draft settings leave: see the module docstring. Only a smoke that ran the steady
    four-user mix can be held to it; the audits' unequal lines fail whatever ran."""
    problems = []
    if facts['quad_audits_unequal']:
        problems.append('%d [QUAD-AUDIT] lines with equal=0: the quad differs from the pair traces' % facts['quad_audits_unequal'])
    if facts['singles_audits_unequal']:
        problems.append('%d [DRAFT-SINGLES-AUDIT] lines with equal=0 (%s): a batched draft differs from the single-user draft'
                        % (facts['singles_audits_unequal'], ', '.join(facts['singles_audit_stages'])))
    if not steady:
        return problems
    if env.get(QUAD_FLAG) == '1':
        if facts['quad_markers'] != 1:
            problems.append('the quad marker (%s) appears %d times, not once: the quad never engaged or engaged twice'
                            % (QUAD_MARKER, facts['quad_markers']))
        if not facts['quad_rounds']:
            problems.append('no round was served by the quad (no [PACKED-SELECT] pairs=[[0, 1, 2, 3]] line): the four '
                            'steady users drafted some other way')
        if facts['quad_fallbacks'] or facts['quad_disabled']:
            problems.append('the quad fell back %d times and was disabled %d times' % (facts['quad_fallbacks'],
                                                                                        facts['quad_disabled']))
    else:
        if facts['quad_markers'] or facts['quad_lines'] or facts['quad_rounds']:
            problems.append('the quad ran (marker %d, round lines %d, rounds %d) on a profile with %s off'
                            % (facts['quad_markers'], facts['quad_lines'], facts['quad_rounds'], QUAD_FLAG))
        if not facts['pair_rounds']:
            problems.append('no round was served by two packed pairs: the comparison arm never batched a draft')
    if env.get(SINGLES_AUDIT_FLAG) not in (None, '', '0') and not facts['singles_audits']:
        problems.append('%s is set and no [DRAFT-SINGLES-AUDIT] line was logged' % SINGLES_AUDIT_FLAG)
    return problems


def fused_facts(env, container_text):
    """lever_n_m3native_gate.h1b_summary of the container log (the gate's own reading of the H1b lines), or None when the profile has
    no fused-commit flag on and the log holds none of its lines."""
    import lever_n_m3native_gate as gate

    if env.get(FUSED_FLAG) != '1' and not any(line in container_text for line in FUSED_LINES):
        return None
    return gate.h1b_summary(container_text)


def fused_problems(env, container_text, steady, facts=None):
    """The problems the profile's fused-commit settings leave: see the module docstring."""
    import lever_n_m3native_gate as gate

    problems = []
    if env.get(FUSED_FLAG) != '1':
        if any(line in container_text for line in FUSED_LINES):
            problems.append('%s is not set and the log holds fused-commit lines: the fused commit ran on a profile without it'
                            % FUSED_FLAG)
        return problems
    report = gate.h1b_report(env, container_text)
    problems += report['problems']
    facts = report if facts is None else facts
    engaged_lines = container_text.count(gate.FUSED_ENGAGED_MARKER)
    if engaged_lines != 1:
        problems.append('the fused commit\'s engaged line (%s) appears %d times, not once' % (gate.FUSED_ENGAGED_MARKER,
                                                                                            engaged_lines))
    engaged = facts.get('engaged')
    inplace = env.get(FUSED_INPLACE_FLAG) == '1'
    if engaged is not None:
        wanted = engaged['users'] * (1 + FUSED_PREFIXES) if inplace else engaged['users']
        if engaged['traces'] != wanted:
            problems.append('the fused commit captured %d traces, not %d (%d users, %s)' % (
                engaged['traces'], wanted, engaged['users'], 'a T_proj and %d slides each' % FUSED_PREFIXES if inplace
                else 'a T_proj each'))
    if env.get(FUSED_AUDIT_FLAG) == '1':
        chips = int(env.get('QWEN_FAST_TP') or 2)
        expected = 10 * chips * (2 if inplace else 1)    # ten deltas a chip, and in place ten banks a chip
        wrong = sorted({int(match.group(5)) for match in gate.FUSED_AUDIT_LINE.finditer(container_text)
                        if int(match.group(5)) != expected})
        if wrong:
            problems.append('audit lines checked %s items, not %d (10 K/V pieces x %d chips%s)' % (
                wrong, expected, chips, ', deltas and banks' if inplace else ''))
        if not facts.get('audits'):
            problems.append('%s is set and no [PACKED-FUSED-AUDIT] line was logged' % FUSED_AUDIT_FLAG)
    if steady:
        if not facts.get('four_fused_rounds'):
            problems.append('no round had all four users on the fused path (%d fused publications, %d today; reasons %s)' % (
                facts.get('fused', 0), facts.get('today', 0), facts.get('today_reasons')))
        if env.get(FUSED_LIVE_BANKS_FLAG) == '1' and not facts.get('live_banks'):
            problems.append('%s is set and the live-bank marker (%s) was never logged' % (FUSED_LIVE_BANKS_FLAG,
                                                                                            gate.FUSED_LIVE_BANKS_MARKER))
    return problems


def check(smoke_text, container_text, slide, max_ramp_ms=50.0, env=None):
    """(problems, facts) for a smoke log and a container log. `env` (the served profile's) adds the batched-draft
    stop conditions."""
    smoke = smoke_results(smoke_text)
    problems = smoke_problems(smoke)
    mismatches = [line.strip()[:200] for line in container_text.splitlines() if MISMATCH.search(line)]
    problems += ['audit mismatch in the container log: %s' % line for line in mismatches[:4]]
    median, rounds = ramp_kv_median(container_text)
    facts = dict(audit_mismatches=len(mismatches), publish_rounds=rounds, largest_prepare_history_median_ms=median)
    if env is not None:
        drafts = draft_facts(container_text)
        facts['draft'] = drafts
        steady = STEADY_TEST in (smoke or {}) and 'error' not in smoke[STEADY_TEST]
        problems += draft_problems(drafts, env, steady)
        fused = fused_facts(env, container_text)
        if fused is not None:
            facts['fused'] = fused
        problems += fused_problems(env, container_text, steady, fused)
    if slide:
        if median is None:
            problems.append('no [PACKED-PUBLISH] round with a commit: the ramp commit time is unread (QWEN_FAST_PACKED_AUDIT?)')
        elif median > max_ramp_ms:
            problems.append('ramp commit: median of the largest prepare_history per round is %.1f ms, above %.0f ms '
                            '(the slide is not taking effect)' % (median, max_ramp_ms))
    return problems, facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--smoke-log', type=Path, required=True)
    parser.add_argument('--container-log', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    parser.add_argument('--max-ramp-kv-ms', type=float, default=50.0)
    options = parser.parse_args(argv)
    try:
        smoke = options.smoke_log.read_text(encoding='utf-8', errors='replace')
        container = options.container_log.read_text(encoding='utf-8', errors='replace')
        slide = slide_on(options.profile, options.profiles)
        env = profile_env(options.profile, options.profiles)
    except (OSError, ValueError) as error:
        print('SMOKE_CHECK unreadable: %s' % error, file=sys.stderr)
        return 2
    problems, facts = check(smoke, container, slide, options.max_ramp_kv_ms, env)
    print('SMOKE_CHECK profile=%s slide=%s %s' % (options.profile, 'on' if slide else 'off', json.dumps(facts)))
    for problem in problems:
        print('SMOKE_CHECK FAILED: %s' % problem)
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
