"""Compare a concurrent real-text M3native arm against a sequential one on the same prompts, offline.

    py -3.11 scripts/ci/real_text_compare.py CONCURRENT SEQUENTIAL [--json OUT]

Each argument is the gate's report: the artifact's m3native-gate.json, the gate's stdout
(m3native-gate-stdout.log), or a whole job log - the JSON between the gate's BEGIN/END markers
is extracted, with a per-line prefix such as a GitHub log's timestamp stripped.

Per user it checks that the two arms served the SAME prompt (prompt sha256), then exactness, the
sequential (single-stream) text being the reference: greedy decoding makes the packed output
byte-identical to the single-stream output. With the same --max-tokens in both arms (v157/v158,
v159/v160: 256) that means identical text, identical completion token count and the same
finish_reason - a stream that is a strict prefix of the other is a divergence (both spent the
same budget on different tokens), and one cut by EOS in one arm only is 'eos-mismatch'. Only
when the budgets differ may the shorter stream be a prefix of the longer (the gate's
compare_prefix semantics), and then only if it ended on its own budget ('length', completion
tokens = its --max-tokens).

It prints the first divergent character, both arms' acceptance summaries side by side, each
user's steady tok/s (acceptance_report.steady_rate) in both arms with the ratio concurrent /
single, and the QWEN_* flags the arms' configurations differ by: v157/v159 run the v155 flag set
and v158/v160 the v149/v150 one (the bf8 draft, T1, round Build 1, eight-row replay groups, SDPA
share and the GDN prefill conv only in the concurrent arms), so a side-by-side difference mixes
packing with those flags and is labelled so.

Stdlib only; exits 0 when every user is exact, 1 when any is not, 2 when a report is unreadable.

THE EXACTNESS POLICY (--policy; c2-serve-for-real-plan 2.2 item 5, fixed before G4). Near-tie tokens
flip within 0-758 characters when the arithmetic changes (docs/real-text-2026-09-24.md:222-227), so:
  - the reference is a solo (sequential) run on the SAME image with the same arithmetic flags;
  - full answers are compared, never prefixes (a 'prefix' verdict means the arms ran different
    budgets, which is NOT_COMPARABLE);
  - a first divergence triggers ONE re-run of both arms (--rerun CONCURRENT2 SEQUENTIAL2). A
    divergence that reproduces (the same first differing character, or the same text) FAILS - unless
    the arms differ in an arithmetic flag (any configuration flag outside ARITHMETIC_NEUTRAL, an unset
    flag read as its EFFECTIVE_DEFAULTS value), which makes it NOT_COMPARABLE: it needs a reference
    under the same arithmetic. A divergence that does not reproduce, or reappears somewhere else, is
    UNSTABLE (nondeterminism, itself a finding).
Verdicts: PASS (exit 0), FAIL (1), NOT_COMPARABLE (3), UNSTABLE (4), RERUN (5: a first divergence and
no --rerun given - the caller's cue to run both arms again). exactness_policy is the pure decision.

BRING-UP (--against-reference FILE): one served arm against a tracked reference's texts (the C2
serving image at 4 x 131072 against v235, scripts/ci/references/c2-serving/): IDENTICAL or DIVERGED per
user, NOT_COMPARABLE where the prompts differ (another corpus or tokenizer) or a divergence meets an
arithmetic difference, ERROR where the served stream failed (reference_verdicts).
"""

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lever_n_m3native_gate import BEGIN, END, compare_prefix  # noqa: E402

GITHUB_PREFIX = re.compile(r'^(?:[^\t]*\t){0,2}\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ')


def extract_report(text):
    """The gate report from a file's text: plain JSON, or the last BEGIN..END block of a log."""
    stripped = text.strip()
    if stripped.startswith('{'):
        return json.loads(stripped)
    lines = text.replace('\r\n', '\n').split('\n')
    begins = [i for i, line in enumerate(lines) if line.rstrip().endswith(BEGIN)]
    for begin in reversed(begins):
        prefix = lines[begin][:len(lines[begin].rstrip()) - len(BEGIN)]
        body = []
        for line in lines[begin + 1:]:
            if prefix and GITHUB_PREFIX.match(prefix):
                line = GITHUB_PREFIX.sub('', line, count=1)
            elif prefix and line.startswith(prefix):
                line = line[len(prefix):]
            if line.rstrip() == END:
                return json.loads('\n'.join(body))
            body.append(line)
    raise ValueError('no %s ... %s block found' % (BEGIN, END))


def load_report(path):
    return extract_report(Path(path).read_text(encoding='utf-8', errors='replace'))


def prompt_shas(report):
    """Each user's prompt sha256: the real-text provenance, else the comparisons."""
    users = (report.get('real_text') or {}).get('users') or []
    if users:
        return [u.get('prompt_sha256') for u in users]
    return [c.get('prompt_sha256') for c in report.get('comparisons') or []]


def first_divergence(a, b):
    """The first character index where a and b differ, or None when equal."""
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_user(concurrent, single, budgets=(None, None)):
    """One user's verdict: exact, finish-mismatch, token-mismatch, eos-mismatch, diverged, error,
    or prefix - the last only when the arms' budgets (--max-tokens) are both known and differ and
    the shorter stream ended on its own budget."""
    concurrent, single = concurrent or {}, single or {}
    actual, reference = concurrent.get('text') or '', single.get('text') or ''
    finish = (concurrent.get('finish_reason'), single.get('finish_reason'))
    completion = (concurrent.get('completion_tokens'), single.get('completion_tokens'))
    result = dict(concurrent_len=len(actual), single_len=len(reference), finish_concurrent=finish[0],
                  finish_single=finish[1], completion_concurrent=completion[0], completion_single=completion[1],
                  max_tokens_concurrent=budgets[0], max_tokens_single=budgets[1],
                  first_divergence=first_divergence(actual, reference))
    errors = [side for side, entry in (('concurrent', concurrent), ('single', single)) if not entry or entry.get('error')]
    if errors:
        result.update(verdict='error', errors=errors)
        return result
    identical_prefix, partial = compare_prefix(actual, reference)
    result.update(identical_prefix=identical_prefix, partial=partial)
    budgets_differ = None not in budgets and budgets[0] != budgets[1]
    if actual == reference:
        if finish[0] != finish[1]:
            verdict = 'finish-mismatch'
        elif completion[0] != completion[1]:
            verdict = 'token-mismatch'
        else:
            verdict = 'exact'
    elif not identical_prefix:
        verdict = 'diverged'
    else:
        shorter = 0 if len(actual) < len(reference) else 1
        if finish[shorter] == 'stop':
            verdict = 'eos-mismatch'
        elif budgets_differ and finish[shorter] == 'length' and completion[shorter] == budgets[shorter]:
            verdict = 'prefix'
        else:
            # The same budget spent on different tokens: a prefix here is a divergence near the end.
            verdict = 'diverged'
    result['verdict'] = verdict
    return result


def stream_budgets(report):
    """Each user's --max-tokens: the report's (detail-mode gate reports record it) or the
    comparisons'; None when neither says."""
    comparisons = report.get('comparisons') or []
    count = max(len(report.get('streams') or []), len(comparisons))
    per_user = [(comparisons[user] if user < len(comparisons) else {}).get('max_tokens') for user in range(count)]
    return [value if value is not None else report.get('max_tokens') for value in per_user]


# What a report's qwen_configuration covers (lever_n_m3native_gate.qwen_configuration). A report
# with no configuration_scope recorded QWEN_* only (v157-v160, v235); 'qwen-tt' reports also carry
# QWEN<n>_* (QWEN35_GDN_*, QWEN36_*), TT_*, MESH_DEVICE and OMP_NUM_THREADS. Two reports are compared
# only on the names both scopes cover, so an older report never reads as missing a flag.
LEGACY_SCOPE = 'qwen'
EXTENDED_SCOPE = 'qwen-tt'
EXTENDED_PREFIX = re.compile(r'(?:QWEN[0-9]*_|TT_)')
EXTENDED_NAMES = frozenset(('MESH_DEVICE', 'OMP_NUM_THREADS'))


def configuration_scope(report):
    return report.get('configuration_scope') or LEGACY_SCOPE


def in_scope(name, scope):
    if scope == LEGACY_SCOPE:
        return name.startswith('QWEN_')
    return bool(EXTENDED_PREFIX.match(name)) or name in EXTENDED_NAMES


def configuration_diff(concurrent, single):
    """{flag: [concurrent value, single value]} for every configuration flag the arms differ by,
    over the names both reports' scopes cover, or None when either report carries no configuration
    (a gate report from before it recorded one)."""
    first, second = concurrent.get('qwen_configuration'), single.get('qwen_configuration')
    if first is None or second is None:
        return None
    scopes = (configuration_scope(concurrent), configuration_scope(single))
    return {name: [first.get(name), second.get(name)] for name in sorted(set(first) | set(second))
            if all(in_scope(name, scope) for scope in scopes) and first.get(name) != second.get(name)}


def user_acceptance(report, user):
    for entry in (report.get('acceptance') or {}).get('users') or []:
        if entry.get('user') == user:
            full = entry.get('full_draft') or {}
            every = entry.get('all') or {}
            return dict(rounds=full.get('rounds'), mean=full.get('mean_emitted'), p_gt_8=full.get('p_emitted_gt_8'),
                        p_gt_11=full.get('p_emitted_gt_11'), p_eq_16=full.get('p_emitted_eq_16'),
                        max=full.get('max_emitted'), all_mean=every.get('mean_emitted'))
    return {}


def user_rate(report, user):
    for entry in (report.get('decode_rate') or {}).get('users') or []:
        if entry.get('user') == user:
            return dict(steady=entry.get('steady_tok_s'), mean_over_median=entry.get('mean_over_median_tok_s'),
                        all_active=entry.get('all_active_tok_s'), median_gap=entry.get('median_gap_tok_s'))
    return {}


def ratio(a, b):
    return round(a / b, 3) if a and b else None


def compare(concurrent, single):
    """The whole comparison as a dict (what --json writes)."""
    streams_c, streams_s = concurrent.get('streams') or [], single.get('streams') or []
    shas_c, shas_s = prompt_shas(concurrent), prompt_shas(single)
    budgets_c, budgets_s = stream_budgets(concurrent), stream_budgets(single)
    users = []
    for user in range(max(len(streams_c), len(streams_s))):
        sha_c = shas_c[user] if user < len(shas_c) else None
        sha_s = shas_s[user] if user < len(shas_s) else None
        entry = dict(user=user, prompt_sha256=sha_c, prompt_match=bool(sha_c) and sha_c == sha_s)
        if not entry['prompt_match']:
            entry.update(verdict='prompt-mismatch', prompt_sha256_single=sha_s)
        else:
            entry.update(compare_user(streams_c[user] if user < len(streams_c) else None,
                                      streams_s[user] if user < len(streams_s) else None,
                                      (budgets_c[user] if user < len(budgets_c) else None,
                                       budgets_s[user] if user < len(budgets_s) else None)))
        rate_c, rate_s = user_rate(concurrent, user), user_rate(single, user)
        entry.update(acceptance_concurrent=user_acceptance(concurrent, user),
                     acceptance_single=user_acceptance(single, user), rate_concurrent=rate_c, rate_single=rate_s,
                     rate_ratio_steady=ratio(rate_c.get('steady'), rate_s.get('steady')),
                     rate_ratio_all_active=ratio(rate_c.get('all_active'), rate_s.get('all_active')),
                     rate_ratio_median_gap=ratio(rate_c.get('median_gap'), rate_s.get('median_gap')))
        users.append(entry)
    exact = all(u.get('verdict') in ('exact', 'prefix') for u in users) and bool(users)
    differ = configuration_diff(concurrent, single)
    return dict(users=users, exact=exact, configuration_diff=differ,
                same_configuration=None if differ is None else not differ,
                concurrent=dict(users=concurrent.get('users'), sequential_users=concurrent.get('sequential_users'),
                                context=concurrent.get('context'), gate_passed=concurrent.get('gate_passed'),
                                accept=(concurrent.get('acceptance') or {}).get('summary_line')),
                single=dict(users=single.get('users'), sequential_users=single.get('sequential_users'),
                            context=single.get('context'), gate_passed=single.get('gate_passed'),
                            accept=(single.get('acceptance') or {}).get('summary_line')))


# QWEN_* flags that only log, audit, check or route: arms differing in them still run the same
# arithmetic. Everything else counts as arithmetic (conservative: an unknown flag makes a divergence
# NOT_COMPARABLE rather than a FAIL). An *_AUDIT flag re-computes a cut beside the served one and
# raises on a mismatch; the committed tokens come from the served path either way.
ARITHMETIC_NEUTRAL = frozenset((
    'QWEN_C2_SERVING', 'QWEN_C2_PROFILE', 'QWEN_C2_PROFILES', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS',
    'QWEN_FABRIC_LINK_PROBE', 'QWEN_FAST_FAULTHANDLER', 'QWEN_FAST_CARRY_LOG', 'QWEN_FAST_PHASE_LOG',
    'QWEN_FAST_PHASE_TIMING', 'QWEN_FAST_SEQ_PUBLISH_LOG', 'QWEN_FAST_MEMORY_LEDGER', 'QWEN_FAST_SHARD_CHECK',
    # Paths (the 'qwen-tt' scope): where the runtime, the weight cache and the kernel cache live.
    'TT_METAL_HOME', 'TT_CACHE_PATH', 'TT_METAL_CACHE',
))
ARITHMETIC_NEUTRAL_SUFFIXES = ('_AUDIT',)
# The value a flag takes when the environment does not set it, where the code defines one: an
# absent flag and its default are the same arithmetic. QWEN_FAST_OUTPUT_BUDGET: serving_fast_policy.
# OUTPUT_BUDGET (256 unset); the C2 contract exports its profile's value into every process of the
# image (serving_c2_contract.apply_environment), so a served arm carries 256 where v235 carried none.
EFFECTIVE_DEFAULTS = {'QWEN_FAST_OUTPUT_BUDGET': '256'}
# Reviewed successions (S2 design W9 and M2): {flag: (older value, newer value)}. Two arms that differ by
# exactly that pair of that flag are held to the same arithmetic, so a divergence between them FAILS
# instead of reading NOT_COMPARABLE: the K64j image serves exact through K64i's programs (it adds only the
# 0x20 branch and its kernels), and M2 - exact on the K64j image against v235's K64i reference - is the
# run that must fail if that is not so. c2_image_provenance.ENVIRONMENT_SUCCESSIONS is the same review
# for G1 (test_c2_image_overlay holds the two equal). Any other value of the flag is still arithmetic.
REVIEWED_SUCCESSIONS = {
    'QWEN_FAST_RUNTIME_BINARY_SHA256': ('cf54d716669be6b71f1d627e74892c90f562495dc9500589408a72b4ddccf4a4',
                                        '152951c1c0de5c9dfad2d62c295393a43b2ecf353965c55c709da7e539b975b7'),
}
POLICY_EXIT = dict(PASS=0, FAIL=1, NOT_COMPARABLE=3, UNSTABLE=4, RERUN=5)


def arithmetic_neutral(name):
    return name in ARITHMETIC_NEUTRAL or name.endswith(ARITHMETIC_NEUTRAL_SUFFIXES)


def reviewed_succession(name, values):
    """Whether two values of a flag are one of REVIEWED_SUCCESSIONS' pairs, in either order."""
    pair = REVIEWED_SUCCESSIONS.get(name)
    return pair is not None and sorted(values, key=str) == sorted(pair)


def arithmetic_diff(first, second):
    """{flag: [first, second]} for every arithmetic flag the two reports differ by, a flag either
    side leaves unset read as its EFFECTIVE_DEFAULTS value and a REVIEWED_SUCCESSIONS pair read as
    equal; None when either carries no configuration."""
    differ = configuration_diff(first, second)
    if differ is None:
        return None
    def effective(name, value):
        return EFFECTIVE_DEFAULTS.get(name) if value is None else value

    return {name: values for name, values in differ.items() if not arithmetic_neutral(name)
            and effective(name, values[0]) != effective(name, values[1])
            and not reviewed_succession(name, values)}


def policy_user(verdict):
    """compare_user's verdict under the policy: IDENTICAL, DIVERGED, NOT_COMPARABLE or ERROR."""
    if verdict == 'exact':
        return 'IDENTICAL'
    if verdict == 'error':
        return 'ERROR'
    if verdict in ('prompt-mismatch', 'prefix'):
        return 'NOT_COMPARABLE'
    return 'DIVERGED'   # diverged, finish-, token- or eos-mismatch: the full answers differ


def policy_pass(concurrent, single):
    """One pass of the policy: compare() per user, mapped to IDENTICAL / DIVERGED / NOT_COMPARABLE /
    ERROR, with a divergence under an arithmetic difference made NOT_COMPARABLE."""
    result = compare(concurrent, single)
    arithmetic = arithmetic_diff(concurrent, single)
    users = []
    for user in result['users']:
        verdict = policy_user(user.get('verdict'))
        entry = dict(user=user['user'], verdict=verdict, detail=user.get('verdict'),
                     first_divergence=user.get('first_divergence'), prompt_sha256=user.get('prompt_sha256'),
                     concurrent_len=user.get('concurrent_len'), single_len=user.get('single_len'))
        if verdict == 'DIVERGED' and arithmetic:
            entry.update(verdict='NOT_COMPARABLE', reason='arithmetic flags differ: %s' % ', '.join(sorted(arithmetic)))
        elif user.get('verdict') == 'prompt-mismatch':
            entry['reason'] = 'the arms served different prompts'
        elif user.get('verdict') == 'prefix':
            entry['reason'] = 'the arms ran different budgets'
        users.append(entry)
    return dict(users=users, arithmetic_diff=arithmetic, configuration_recorded=arithmetic is not None)


def texts(report):
    return [(stream or {}).get('text') for stream in report.get('streams') or []]


def lifecycle_user(concurrent, single, budgets=(None, None), ignore_eos=False):
    """One user of a lifecycle arm (the gate's --drops / --user-max-tokens / --user-ignore-eos)
    against its solo run on the same prompt at the arm's full budget, EOS on. What each event leaves
    must agree with the solo text as far as it goes: a dropped stream must be a PREFIX of it. A stream
    cut by its own smaller budget must be a prefix that spent exactly that budget ('length',
    completion_tokens = its max_tokens), or, where the solo run reached EOS inside that budget, the
    solo answer itself. An ignore_eos stream, when the solo run stopped at EOS, must run PAST it:
    strictly longer, the solo text its prefix, and the whole budget spent ('length', completion_tokens
    = its max_tokens) - an engine that ignores ignore_eos stops where the solo run did, and that is a
    divergence here, not a match. A user with no event is compared in full (compare_user).
    -> (policy verdict, detail)."""
    concurrent, single = concurrent or {}, single or {}
    if not single or single.get('error') or concurrent.get('error'):
        return 'ERROR', 'error'
    actual, reference = concurrent.get('text') or '', single.get('text') or ''
    finish, count = concurrent.get('finish_reason'), concurrent.get('completion_tokens')
    if concurrent.get('dropped'):
        return ('IDENTICAL' if reference.startswith(actual) else 'DIVERGED'), 'dropped %s' % concurrent['dropped']
    if ignore_eos and single.get('finish_reason') == 'stop':
        if finish == 'stop' and actual == reference:
            return 'DIVERGED', 'ignore_eos ignored: stopped at the solo EOS'
        past = (len(actual) > len(reference) and actual.startswith(reference) and finish == 'length'
                and budgets[0] is not None and count == budgets[0])
        return ('IDENTICAL' if past else 'DIVERGED'), 'ignore_eos past the solo EOS'
    if None not in budgets and budgets[0] < budgets[1]:
        if finish == 'length':
            consistent = reference.startswith(actual) and count == budgets[0]
        elif finish == 'stop':
            consistent = actual == reference and single.get('finish_reason') == 'stop'
        else:
            consistent = False
        return ('IDENTICAL' if consistent else 'DIVERGED'), 'its own budget %d' % budgets[0]
    detail = compare_user(concurrent, single, budgets)['verdict']
    return policy_user(detail), detail


def lifecycle_pass(concurrent, single):
    """policy_pass for a lifecycle arm: each user through lifecycle_user, the same shape out."""
    streams_c, streams_s = concurrent.get('streams') or [], single.get('streams') or []
    shas_c, shas_s = prompt_shas(concurrent), prompt_shas(single)
    budgets_c, budgets_s = stream_budgets(concurrent), stream_budgets(single)
    comparisons = concurrent.get('comparisons') or []
    arithmetic = arithmetic_diff(concurrent, single)
    users = []
    for user in range(max(len(streams_c), len(streams_s))):
        sha_c = shas_c[user] if user < len(shas_c) else None
        sha_s = shas_s[user] if user < len(shas_s) else None
        mine = streams_c[user] if user < len(streams_c) else None
        theirs = streams_s[user] if user < len(streams_s) else None
        entry = dict(user=user, prompt_sha256=sha_c, first_divergence=first_divergence(
            (mine or {}).get('text') or '', (theirs or {}).get('text') or ''))
        if not sha_c or sha_c != sha_s:
            entry.update(verdict='NOT_COMPARABLE', detail='prompt-mismatch', reason='the arms served different prompts')
        else:
            ignore = bool((comparisons[user] if user < len(comparisons) else {}).get('ignore_eos'))
            verdict, detail = lifecycle_user(mine, theirs, (budgets_c[user] if user < len(budgets_c) else None,
                                                            budgets_s[user] if user < len(budgets_s) else None), ignore)
            entry.update(verdict=verdict, detail=detail)
            if verdict == 'DIVERGED' and arithmetic:
                entry.update(verdict='NOT_COMPARABLE', reason='arithmetic flags differ: %s' % ', '.join(sorted(arithmetic)))
        users.append(entry)
    return dict(users=users, arithmetic_diff=arithmetic, configuration_recorded=arithmetic is not None)


def exactness_policy(concurrent, single, rerun=None, pass_function=None):
    """The policy's verdict for a concurrent arm against its solo reference, given the first pass
    and, after a first divergence, the re-run of both arms (rerun = (concurrent2, single2)).

    Per user: IDENTICAL; DIVERGED (in both passes, the same divergence: at the same character, or
    the re-run's text byte for byte the first run's - reproduced); UNSTABLE (in one pass only, or at
    different places in the two); NOT_COMPARABLE; ERROR; RERUN (diverged in the first pass, no re-run
    yet). Overall: FAIL on any DIVERGED or ERROR, else RERUN, else NOT_COMPARABLE, else UNSTABLE,
    else PASS. 'reproducible' says, per arm, whether the re-run repeated the first run's texts byte
    for byte. `pass_function` (default policy_pass; lifecycle_pass for the gate's lifecycle arms)
    makes one pass's verdicts."""
    pass_function = pass_function or policy_pass
    first = pass_function(concurrent, single)
    second = pass_function(*rerun) if rerun is not None else None
    first_texts = texts(concurrent)
    again_texts = texts(rerun[0]) if rerun is not None else []
    users = []
    for index, entry in enumerate(first['users']):
        again = second['users'][index] if second and index < len(second['users']) else None
        verdict = entry['verdict']
        reason = entry.get('reason') or (again or {}).get('reason')
        if verdict in ('ERROR', 'NOT_COMPARABLE'):
            final = verdict
        elif again is None:
            final = 'RERUN' if verdict == 'DIVERGED' else 'IDENTICAL'
        elif again['verdict'] in ('ERROR', 'NOT_COMPARABLE'):
            final = again['verdict']
        elif verdict == 'DIVERGED' and again['verdict'] == 'DIVERGED':
            same_text = (index < len(first_texts) and index < len(again_texts)
                         and first_texts[index] == again_texts[index])
            if entry.get('first_divergence') == again.get('first_divergence') or same_text:
                final = 'DIVERGED'
            else:
                final = 'UNSTABLE'
                reason = reason or 'diverged in both runs, at different characters (%s, %s)' % (
                    entry.get('first_divergence'), again.get('first_divergence'))
        elif verdict == again['verdict'] == 'IDENTICAL':
            final = 'IDENTICAL'
        else:
            final = 'UNSTABLE'
        users.append(dict(user=entry['user'], verdict=final, first=verdict,
                          rerun=again['verdict'] if again else None,
                          first_divergence=entry.get('first_divergence'),
                          rerun_divergence=again.get('first_divergence') if again else None,
                          reason=reason))
    finals = [u['verdict'] for u in users]
    if not users:
        overall = 'FAIL'
    elif 'DIVERGED' in finals or 'ERROR' in finals:
        overall = 'FAIL'
    elif 'RERUN' in finals:
        overall = 'RERUN'
    elif 'NOT_COMPARABLE' in finals:
        overall = 'NOT_COMPARABLE'
    elif 'UNSTABLE' in finals:
        overall = 'UNSTABLE'
    else:
        overall = 'PASS'
    reproducible = None
    if rerun is not None:
        reproducible = dict(concurrent=texts(concurrent) == texts(rerun[0]), single=texts(single) == texts(rerun[1]))
    return dict(verdict=overall, users=users, first=first, rerun=second, reproducible=reproducible)


def reference_verdicts(served, reference):
    """Bring-up: each served stream against the tracked reference's (the same prompts, the same
    budget): IDENTICAL, DIVERGED (with the first differing character), NOT_COMPARABLE (the prompts
    differ, or a divergence meets an arithmetic difference) or ERROR (the served stream failed)."""
    references = reference.get('streams') or []
    streams = served.get('streams') or []
    shas = prompt_shas(served)
    budgets = stream_budgets(served)
    arithmetic = arithmetic_diff(served, reference)
    users = []
    for index in range(max(len(references), len(streams))):
        mine = streams[index] if index < len(streams) else None
        theirs = references[index] if index < len(references) else None
        sha = shas[index] if index < len(shas) else None
        entry = dict(user=index, prompt_sha256=sha, reference_prompt_sha256=(theirs or {}).get('prompt_sha256'))
        if theirs is None or not sha or sha != theirs.get('prompt_sha256'):
            entry.update(verdict='NOT_COMPARABLE', reason='the served prompt is not the reference\'s')
        else:
            detail = compare_user(mine, theirs, (budgets[index] if index < len(budgets) else None,
                                                 theirs.get('max_tokens')))
            verdict = policy_user(detail['verdict'])
            entry.update(verdict=verdict, detail=detail['verdict'], first_divergence=detail.get('first_divergence'),
                         served_len=detail['concurrent_len'], reference_len=detail['single_len'],
                         finish=[detail['finish_concurrent'], detail['finish_single']],
                         completion=[detail['completion_concurrent'], detail['completion_single']])
            if verdict == 'DIVERGED' and arithmetic:
                entry.update(verdict='NOT_COMPARABLE',
                             reason='DIVERGED under different arithmetic flags: %s' % ', '.join(sorted(arithmetic)))
        users.append(entry)
    finals = [u['verdict'] for u in users]
    overall = ('IDENTICAL' if users and all(v == 'IDENTICAL' for v in finals) else
               'FAIL' if 'DIVERGED' in finals or 'ERROR' in finals or not users else 'NOT_COMPARABLE')
    return dict(verdict=overall, users=users, arithmetic_diff=arithmetic,
                configuration_diff=configuration_diff(served, reference))


# The launched argv against a reference's (read-the-launched-argv): flags the platform owns (its
# served name, host and port) and the chat endpoint's parsers are not the engine's; the reference
# mounted the hub at /models/hub where the agent mounts it at /models.
PLATFORM_OWN_FLAGS = ('--served-model-name', '--host', '--port')
PARSER_FLAGS = ('--reasoning-parser', '--tool-call-parser', '--enable-auto-tool-choice')
REFERENCE_HUB, AGENT_HUB = '/models/hub/', '/models/'


def engine_flags(argv, drop=()):
    """{flag: value, or True for a flag without one} of an argv's '--flag [value]' pairs, the `drop`
    flags left out (an absent flag then reads None, never equal to a present one)."""
    pairs, index = {}, 0
    argv = list(argv or ())
    while index < len(argv):
        flag = str(argv[index])
        value = argv[index + 1] if index + 1 < len(argv) and not str(argv[index + 1]).startswith('--') else None
        index += 2 if value is not None else 1
        if flag not in drop:
            pairs[flag] = True if value is None else value
    return pairs


def _engine_value(value):
    if not isinstance(value, str):
        return value
    value = value.replace(REFERENCE_HUB, AGENT_HUB)
    if value[:1] in '{[':
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def served_engine_diff(served_argv, reference_command):
    """{flag: [served, reference]} for every engine flag the contract's launched argv (the
    '[QWEN-C2] profile <name>: vLLM argv' list, sys.argv[1:]) and a reference's recorded command
    ([python, -m, module, ...]) differ in; JSON values compared as JSON. {} when they serve the same
    engine; None when either is not an argv list."""
    if not isinstance(served_argv, list) or not isinstance(reference_command, list) or len(reference_command) < 3:
        return None
    drop = PLATFORM_OWN_FLAGS + PARSER_FLAGS
    mine = {flag: _engine_value(value) for flag, value in engine_flags(served_argv, drop).items()}
    theirs = {flag: _engine_value(value) for flag, value in engine_flags(reference_command[3:], drop).items()}
    return {flag: [mine.get(flag), theirs.get(flag)] for flag in sorted(set(mine) | set(theirs))
            if mine.get(flag) != theirs.get(flag)}


def render_reference(result):
    lines = []
    for user in result['users']:
        if user['verdict'] == 'DIVERGED':
            lines.append('user %d: DIVERGED at character %s (%s; served %s / reference %s characters)' % (
                user['user'], user.get('first_divergence'), user.get('detail'), user.get('served_len'),
                user.get('reference_len')))
        else:
            lines.append('user %d: %s%s' % (user['user'], user['verdict'],
                                             ' (%s)' % user['reason'] if user.get('reason') else ''))
    differ = result.get('configuration_diff')
    if differ:
        lines.append('configuration differs from the reference (served | reference): %s' % ', '.join(
            '%s=%s|%s' % (name, values[0], values[1]) for name, values in sorted(differ.items())))
    lines.append('BRING-UP %s' % result['verdict'])
    return '\n'.join(lines)


def render_policy(result):
    lines = ['user verdict          first      rerun      diverge  rerun-diverge  reason']
    for user in result['users']:
        lines.append('%-4s %-15s %-10s %-10s %7s  %13s  %s' % (
            user['user'], user['verdict'], user['first'], user['rerun'] or '-', _f(user.get('first_divergence'), '%d'),
            _f(user.get('rerun_divergence'), '%d'), user.get('reason') or ''))
    arithmetic = result['first'].get('arithmetic_diff')
    if arithmetic:
        lines.append('arithmetic flags differ (c | s): %s' % ', '.join(
            '%s=%s|%s' % (name, values[0], values[1]) for name, values in sorted(arithmetic.items())))
    elif not result['first'].get('configuration_recorded'):
        lines.append('configurations not recorded in both reports: arithmetic equality is unverified')
    if result.get('reproducible') is not None:
        lines.append('re-run reproduced its first run byte for byte: concurrent=%s single=%s' % (
            result['reproducible']['concurrent'], result['reproducible']['single']))
    lines.append('POLICY %s' % result['verdict'])
    return '\n'.join(lines)


def _f(value, pattern='%.2f'):
    return '-' if value is None else pattern % value


def render(result):
    lines = ['concurrent: users=%s context=%s gate_passed=%s' % (
                 result['concurrent']['users'], result['concurrent']['context'], result['concurrent']['gate_passed']),
             'single:     users=%s sequential=%s context=%s gate_passed=%s' % (
                 result['single']['users'], result['single']['sequential_users'], result['single']['context'],
                 result['single']['gate_passed'])]
    differ = result.get('configuration_diff')
    if differ is None:
        lines.append('configurations: not recorded in both reports; the acceptance columns may compare different flag sets')
    elif differ:
        lines.append('configurations DIFFER (c | s), so acceptance and rate differences mix packing with these flags:')
        lines.extend('  %s=%s | %s' % (name, values[0], values[1]) for name, values in differ.items())
    else:
        lines.append('configurations: identical QWEN_* flags')
    lines.append('user verdict          len c/s        finish c/s     diverge  | mean emitted c/s  P>8 c/s      '
                 'P>11 c/s     P16 c/s      | steady tok/s c/s  ratio | all-active ratio  median-gap ratio')
    for u in result['users']:
        a, b = u.get('acceptance_concurrent') or {}, u.get('acceptance_single') or {}
        rc, rs = u.get('rate_concurrent') or {}, u.get('rate_single') or {}
        lines.append('%-4s %-15s %6s/%-6s %8s/%-8s %7s  | %5s/%-5s   %5s/%-5s  %5s/%-5s  %5s/%-5s  | %6s/%-6s  %6s | %6s  %6s' % (
            u['user'], u.get('verdict'), u.get('concurrent_len', '-'), u.get('single_len', '-'),
            u.get('finish_concurrent'), u.get('finish_single'), _f(u.get('first_divergence'), '%d'),
            _f(a.get('mean')), _f(b.get('mean')), _f(a.get('p_gt_8'), '%.3f'), _f(b.get('p_gt_8'), '%.3f'),
            _f(a.get('p_gt_11'), '%.3f'), _f(b.get('p_gt_11'), '%.3f'), _f(a.get('p_eq_16'), '%.3f'),
            _f(b.get('p_eq_16'), '%.3f'), _f(rc.get('steady'), '%.1f'), _f(rs.get('steady'), '%.1f'),
            _f(u.get('rate_ratio_steady'), '%.3f'), _f(u.get('rate_ratio_all_active'), '%.3f'),
            _f(u.get('rate_ratio_median_gap'), '%.3f')))
    label = '' if differ == {} else ' (configurations not recorded)' if differ is None else ' (different configurations)'
    lines.append('concurrent%s %s' % (label, result['concurrent']['accept']))
    lines.append('single%s     %s' % (label, result['single']['accept']))
    lines.append('EXACT' if result['exact'] else 'NOT EXACT')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('concurrent', type=Path, help='the concurrent arm (e.g. v157 or v159)')
    parser.add_argument('sequential', type=Path, nargs='?',
                        help='the sequential arm on the same prompts (e.g. v158 or v160)')
    parser.add_argument('--json', type=Path, help='also write the comparison here')
    parser.add_argument('--policy', action='store_true',
                        help='apply the exactness policy (verdict PASS/FAIL/NOT_COMPARABLE/UNSTABLE/RERUN)')
    parser.add_argument('--rerun', type=Path, nargs=2, metavar=('CONCURRENT2', 'SEQUENTIAL2'),
                        help='with --policy: the re-run of both arms after a first divergence')
    parser.add_argument('--against-reference', type=Path, metavar='FILE',
                        help='bring-up: compare CONCURRENT (a served arm) with a tracked reference file instead')
    options = parser.parse_args(argv)
    if options.against_reference is None and options.sequential is None:
        parser.error('the sequential arm is required (or --against-reference)')
    if options.rerun and not options.policy:
        parser.error('--rerun needs --policy')
    try:
        concurrent = load_report(options.concurrent)
        if options.against_reference is not None:
            reference = json.loads(options.against_reference.read_text(encoding='utf-8'))
        else:
            single = load_report(options.sequential)
        rerun = tuple(load_report(path) for path in options.rerun) if options.rerun else None
    except (OSError, ValueError) as error:
        print('unreadable report: %s' % error, file=sys.stderr)
        return 2
    if options.against_reference is not None:
        result = reference_verdicts(concurrent, reference)
        print(render_reference(result))
        code = 0 if result['verdict'] == 'IDENTICAL' else 1
    elif options.policy:
        result = exactness_policy(concurrent, single, rerun)
        print(render(compare(concurrent, single)))
        print(render_policy(result))
        code = POLICY_EXIT[result['verdict']]
    else:
        result = compare(concurrent, single)
        print(render(result))
        code = 0 if result['exact'] else 1
    if options.json:
        options.json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return code


if __name__ == '__main__':
    sys.exit(main())
