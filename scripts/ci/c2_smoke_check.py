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

  - a code-prompt answer that is not text (coding, concurrent4_steady, steady_resend): a stream that finished with `stop` at
    its first token (an instant EOS: v172's users 0 and 1) or before MIN_ANSWER_TOKENS, or whose sample is mostly non-Latin
    script (v172's users 2 and 3: mixed-script symbols), and, at four cards (QWEN_FAST_TP not 2), a first prefill whose [MEMLEDGER] item=model_after_prefill
    line reports buffers: the model allocated device state at request time, after the packed traces were captured
    (serving_runtime.prefill_warm_before_traces), which is the hazard v172 fell into; also at four cards the eager prefill warm
    line must precede Metal's 'Allocating device buffers is unsafe due to the existence of an active trace' warning, and no
    prefill may compile a program beyond its window snapshot's ('[PINDIAG] four-card prefill programs=A->B window=W': B-A-W
    above zero is the #48536 sequence). Every prefill is counted, the ones that end at their first token (the max_tokens=1
    warmup, the first prefill after the attach) included; the engine builds' own lines ('[PINDIAG] four-card engine
    programs=A->B') are facts, not rules: they compile after the capture by design;
  - steady_resend (opt-in, after concurrent4_steady): the first steady prompt again, alone, judged as a code answer (above).
    Its prefill reuses, after packed and pair replays, the window-snapshot programs the steady test compiled after the
    capture - the one prefill program set no attach warm covers;

  python c2_smoke_check.py --smoke-log smoke.log --container-log container.log --profile P [--profiles qwen_c2_profiles.json]
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

CORE = ('warmup', 'warm_lifecycle', 'coding', 'concurrent4', 'concurrent4_steady', 'long_real_text', 'steady_resend')
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
RESEND_TEST = 'steady_resend'
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
# A code prompt asked to be explained and rewritten (800 or 1500 tokens out) does not end by itself in a few tokens.
MIN_ANSWER_TOKENS = 16
TEXT_TESTS = ('coding', STEADY_TEST, RESEND_TEST)
FOREIGN_SHARE = 0.3
FIRST_PREFILL_ITEM = re.compile(r'\[MEMLEDGER\] phase=prefill point=after \S+ item=model_after_prefill chip0=\S+ .*?buffers=(\d+) ')
# Four cards: the eager prefill is warmed before the packed traces (serving_runtime.prefill_warm_before_traces) and every prefill
# reports the programs it compiled (serving_runtime.prefill_tripwire). Metal prints its warning once per process, at the first
# allocation made with a trace live, so it is an order marker and not a count.
WARM_LINE = '[PINDIAG] four-card eager prefill warmed before the packed traces'
UNSAFE_ALLOCATION = 'Allocating device buffers is unsafe due to the existence of an active trace'
PREFILL_PROGRAMS = re.compile(r'\[PINDIAG\] four-card prefill programs=(\d+|None)->(\d+|None) window=(\d+) prompt=(\d+)')
ENGINE_PROGRAMS = re.compile(r'\[PINDIAG\] four-card engine programs=(\d+|None)->(\d+|None) ')
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


def foreign_script(text):
    """The share of a sample's letters outside Latin script (code and English are Latin), or 0.0 when it has none."""
    sample = [char for char in text[:300] if char.isalpha()]
    if not sample:
        return 0.0
    return sum(1 for char in sample if ord(char) > 0x2FF) / len(sample)


def answer_problems(name, stream):
    """Why a code-prompt stream is not an answer: an instant or early stop, or mostly foreign-script text."""
    if not isinstance(stream, dict) or 'error' in stream:
        return []
    problems = []
    tokens = stream.get('tokens')
    if stream.get('finish') == 'stop' and isinstance(tokens, int) and tokens < MIN_ANSWER_TOKENS:
        problems.append('%s: finished with stop after %d token(s), under %d (an instant EOS is not an answer to a code prompt)'
                        % (name, tokens, MIN_ANSWER_TOKENS))
    share = foreign_script(stream['text']) if isinstance(stream.get('text'), str) else 0.0
    if share > FOREIGN_SHARE:
        problems.append('%s: %.0f%% of the letters are outside Latin script (garbage, not an answer)' % (name, 100 * share))
    return problems


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
    for name in ('coding', RESEND_TEST):
        if name in results and 'error' not in results[name]:
            problems += stream_problems(name, results[name])
            problems += answer_problems(name, results[name])
    for name in ('concurrent4', STEADY_TEST):
        if name in results and 'error' not in results[name]:
            for index, user in enumerate(results[name].get('users') or []):
                problems += stream_problems('%s user %d' % (name, index), user)
                if name in TEXT_TESTS:
                    problems += answer_problems('%s user %d' % (name, index), user)
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


def first_prefill_buffers(container_text):
    """Buffers the model held after the FIRST prefill of the container log ([MEMLEDGER] item=model_after_prefill), or None."""
    match = FIRST_PREFILL_ITEM.search(container_text)
    return int(match.group(1)) if match else None


def late_program_problems(container_text):
    """(problems, facts) for the four-card prefill tripwire: the warm line must exist and precede Metal's 'unsafe allocation'
    warning (the packed traces' first allocation), and no prefill may compile a program beyond its window snapshot's
    (B - A - W above zero is the #48536 sequence: a program compiled after the traces were captured)."""
    problems, lines = [], container_text.splitlines()
    warm = next((i for i, line in enumerate(lines) if WARM_LINE in line), None)
    unsafe = next((i for i, line in enumerate(lines) if UNSAFE_ALLOCATION in line), None)
    if warm is None:
        problems.append('the four-card eager prefill warm never ran before the packed traces (no "%s" line): every prefill '
                        'would compile after the capture' % WARM_LINE)
    elif unsafe is not None and unsafe < warm:
        problems.append('the eager prefill warm (log line %d) came after the first allocation made with a trace live '
                        '(line %d, Metal "%s")' % (warm + 1, unsafe + 1, UNSAFE_ALLOCATION))
    prefills = [m.groups() for m in map(PREFILL_PROGRAMS.search, lines) if m]
    late = []
    for before, after, window, prompt in prefills:
        if before == 'None' or after == 'None':
            continue
        extra = int(after) - int(before) - int(window)
        if extra > 0:
            late.append((int(prompt), extra))
    problems += ['the prefill of %d tokens compiled %d program(s) beyond its window snapshot after the packed traces were '
                 'captured ([PINDIAG] four-card prefill programs=): the warm did not cover its shape' % item for item in late[:4]]
    if warm is not None and not prefills:
        problems.append('no "[PINDIAG] four-card prefill programs=" line: the prefill tripwire did not run')
    engines = [(before, after) for before, after in (m.groups() for m in map(ENGINE_PROGRAMS.search, lines) if m)
               if before != 'None' and after != 'None']
    return problems, dict(prefill_warm_line=None if warm is None else warm + 1, unsafe_allocation_line=None if unsafe is None else unsafe + 1,
                          prefills_counted=len(prefills), prefills_with_late_programs=len(late),
                          prefill_window_programs=sum(int(window) for _, _, window, _ in prefills),
                          engines_counted=len(engines), engine_programs=sum(int(after) - int(before) for before, after in engines))


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
    if env is not None and env.get('QWEN_FAST_TP', '2') != '2':
        late = first_prefill_buffers(container_text)
        facts['first_prefill_model_buffers'] = late
        programs, program_facts = late_program_problems(container_text)
        problems += programs
        facts.update(program_facts)
        if late:
            problems.append('the first prefill left %d model buffers allocated after the packed traces were captured '
                            '([MEMLEDGER] item=model_after_prefill): the prefill warm must run before the traces' % late)
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
