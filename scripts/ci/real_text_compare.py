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


def configuration_diff(concurrent, single):
    """{flag: [concurrent value, single value]} for every QWEN_* flag the arms differ by, or None
    when either report carries no configuration (a gate report from before it recorded one)."""
    first, second = concurrent.get('qwen_configuration'), single.get('qwen_configuration')
    if first is None or second is None:
        return None
    return {name: [first.get(name), second.get(name)] for name in sorted(set(first) | set(second))
            if first.get(name) != second.get(name)}


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
    parser.add_argument('sequential', type=Path, help='the sequential arm on the same prompts (e.g. v158 or v160)')
    parser.add_argument('--json', type=Path, help='also write the comparison here')
    options = parser.parse_args(argv)
    try:
        concurrent, single = load_report(options.concurrent), load_report(options.sequential)
    except (OSError, ValueError) as error:
        print('unreadable report: %s' % error, file=sys.stderr)
        return 2
    result = compare(concurrent, single)
    print(render(result))
    if options.json:
        options.json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0 if result['exact'] else 1


if __name__ == '__main__':
    sys.exit(main())
