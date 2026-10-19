"""Score the Swift evaluation window's coding set on a CPU host (stdlib only, Python 3.7 syntax).

    python3 scripts/ci/swift_coding_report.py --run base=coding-set-<profile>.json --run swift=coding-set-<profile>.json [--json-out report.json]

The smoke test coding_set (c2_serving_smoke.py) sends the twenty problems of references/swift/coding-problems.json to the served model, greedy, and saves every
answer (the reasoning and the content apart, the token counts, the finish reason) as coding-set-<profile>.json in the job's results. This script runs each answer's code against
the problem's own tests, in a child interpreter per problem (python -I, a scratch directory, CPU time and memory limits, no network when `unshare -rn` is available), and prints per
run how many problems pass, which, and the tokens the model spent thinking and answering; with two or more runs it also prints the pairwise table (solved by one only, tokens
per problem, tokens per solved problem).

RUN THIS ON A DISPOSABLE CPU HOST OR CONTAINER, never on a machine that holds anything of value: it executes code a language model wrote. It never runs on a card host.

Statuses: pass; fail (a test assertion); error (any other exception, a syntax error, a missing name); timeout; no_code (the answer has no code block); no_answer (the answer is empty:
the budget ended inside the reasoning, finish=length). The token counts are the server's (/tokenize on the reasoning and on the content, token_count_source "server") or, when
the smoke could not ask the server, the completion tokens split by the characters of each part ("estimated"); the report says which.
"""
import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(HERE, 'references', 'swift', 'coding-problems.json')
BLOCK = re.compile(r'```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```', re.S)
DEFAULT_TIMEOUT = 30
MEMORY_BYTES = 2 * 1024 ** 3


def load_problems(path=PROBLEMS):
    with open(path, encoding='utf-8') as handle:
        document = json.load(handle)
    return document, dict((item['id'], item) for item in document['problems'])


def extract_code(content, entry_point):
    """The code of the answer: the last fenced block that defines the entry point, else the last python block, else the last block; None when there is none."""
    blocks = [(language.lower(), body) for language, body in BLOCK.findall(content or '')]
    if not blocks:
        return None
    defining = [body for _, body in blocks if re.search(r'^\s*(?:async\s+)?def\s+%s\b' % re.escape(entry_point), body, re.M)]
    if defining:
        return defining[-1]
    python = [body for language, body in blocks if language in ('python', 'py', 'python3')]
    return (python or [body for _, body in blocks])[-1]


def sandbox_prefix():
    """['unshare', '-rn'] when an unprivileged network namespace works here, else []."""
    try:
        result = subprocess.run(['unshare', '-rn', 'true'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        return ['unshare', '-rn'] if result.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


def _limits(seconds):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (seconds + 2, seconds + 2))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 ** 2, 16 * 1024 ** 2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        os.setsid()
    return apply


def run_code(code, tests, timeout=DEFAULT_TIMEOUT, prefix=()):
    """(status, detail) of `code` followed by `tests` run as one script in a scratch directory: pass | fail | error | timeout."""
    with tempfile.TemporaryDirectory(prefix='swift-coding-') as scratch:
        path = os.path.join(scratch, 'check.py')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(code + '\n\n' + tests)
        environment = {'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1', 'HOME': scratch, 'LANG': 'C.UTF-8'}
        try:
            result = subprocess.run(list(prefix) + [sys.executable, '-I', path], cwd=scratch, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    timeout=timeout, preexec_fn=_limits(timeout))
        except subprocess.TimeoutExpired:
            return 'timeout', 'over %d s' % timeout
    if result.returncode == 0:
        return 'pass', ''
    error = result.stderr.decode('utf-8', 'replace').strip().splitlines()
    last = error[-1][:200] if error else 'exit %d' % result.returncode
    return ('fail' if last.startswith('AssertionError') else 'error'), last


def score_result(entry, problem, timeout=DEFAULT_TIMEOUT, prefix=()):
    """One saved answer -> dict(id, status, detail, finish, tokens...)."""
    content = entry.get('content') or ''
    row = dict(id=entry['id'], finish=entry.get('finish'))
    if not content.strip():
        row.update(status='no_answer', detail='empty content (finish=%s)' % entry.get('finish'))
    else:
        code = extract_code(content, problem['entry_point'])
        if code is None:
            row.update(status='no_code', detail='no fenced code block')
        else:
            status, detail = run_code(code, problem['tests'], timeout, prefix)
            row.update(status=status, detail=detail)
    row.update(token_counts(entry))
    return row


def token_counts(entry):
    """dict(reasoning_tokens, answer_tokens, total_tokens, token_source): the server's counts, or the completion tokens split by characters."""
    total = entry.get('completion_tokens')
    reasoning, answer = entry.get('reasoning_tokens'), entry.get('content_tokens')
    if isinstance(reasoning, int) and isinstance(answer, int):
        return dict(reasoning_tokens=reasoning, answer_tokens=answer, total_tokens=total if isinstance(total, int) else reasoning + answer, token_source='server')
    if not isinstance(total, int):
        return dict(reasoning_tokens=None, answer_tokens=None, total_tokens=None, token_source='none')
    r_chars, c_chars = len(entry.get('reasoning') or ''), len(entry.get('content') or '')
    share = r_chars / float(r_chars + c_chars) if r_chars + c_chars else 0.0
    return dict(reasoning_tokens=int(round(total * share)), answer_tokens=total - int(round(total * share)), total_tokens=total, token_source='estimated')


def score_run(path, problems, timeout=DEFAULT_TIMEOUT, prefix=()):
    with open(path, encoding='utf-8') as handle:
        document = json.load(handle)
    rows = []
    seen = set()
    for entry in document['results']:
        if entry['id'] not in problems:
            raise ValueError('%s: an answer for %r, which is not a problem of the set' % (path, entry['id']))
        seen.add(entry['id'])
        rows.append(score_result(entry, problems[entry['id']], timeout, prefix))
    for missing in sorted(set(problems) - seen):
        rows.append(dict(id=missing, status='no_answer', detail='not in the file', finish=None, reasoning_tokens=None, answer_tokens=None, total_tokens=None, token_source='none'))
    rows.sort(key=lambda row: row['id'])
    return dict(label=document.get('label'), model=document.get('model'), max_tokens=document.get('max_tokens'), rows=rows)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / float(len(values)) if values else None


def summarize(run):
    rows = run['rows']
    passed = [row for row in rows if row['status'] == 'pass']
    sources = sorted(set(row['token_source'] for row in rows))
    return dict(problems=len(rows), passed=len(passed), by_status=dict((status, sum(1 for row in rows if row['status'] == status)) for status in sorted(set(row['status'] for row in rows))),
                finish_length=sum(1 for row in rows if row['finish'] == 'length'),
                mean_reasoning_tokens=mean(row['reasoning_tokens'] for row in rows), mean_answer_tokens=mean(row['answer_tokens'] for row in rows),
                mean_total_tokens=mean(row['total_tokens'] for row in rows), mean_total_tokens_solved=mean(row['total_tokens'] for row in passed),
                token_source=sources[0] if len(sources) == 1 else 'mixed:' + ','.join(sources))


def compare(base, other):
    """Pairwise: the problems only one run solves, and the token ratio over the problems both solve."""
    a = dict((row['id'], row) for row in base['rows'])
    b = dict((row['id'], row) for row in other['rows'])
    both = sorted(i for i in a if a[i]['status'] == 'pass' and b.get(i, {}).get('status') == 'pass')
    ratio = None
    paired = [(a[i]['total_tokens'], b[i]['total_tokens']) for i in both if a[i]['total_tokens'] and b[i]['total_tokens']]
    if paired:
        ratio = sum(y for _, y in paired) / float(sum(x for x, _ in paired))
    return dict(only_first=sorted(i for i in a if a[i]['status'] == 'pass' and b.get(i, {}).get('status') != 'pass'),
                only_second=sorted(i for i in b if b[i]['status'] == 'pass' and a.get(i, {}).get('status') != 'pass'),
                solved_by_both=len(both), token_ratio_second_over_first_on_both=ratio)


def fmt(value, digits=0):
    return 'n/a' if value is None else ('%.*f' % (digits, value))


def lines(labels, runs):
    out = []
    for label in labels:
        run = runs[label]
        s = summarize(run)
        out.append('SWIFT_CODING %s: passed %d/%d %s finish_length=%d mean tokens reasoning=%s answer=%s total=%s solved-only total=%s (token counts: %s)' % (
            label, s['passed'], s['problems'], json.dumps(s['by_status'], sort_keys=True), s['finish_length'], fmt(s['mean_reasoning_tokens']), fmt(s['mean_answer_tokens']),
            fmt(s['mean_total_tokens']), fmt(s['mean_total_tokens_solved']), s['token_source']))
        out.append('SWIFT_CODING %s: pass %s | not passed %s' % (label, ' '.join(r['id'][:3] for r in run['rows'] if r['status'] == 'pass') or '-',
                                                              ' '.join('%s:%s' % (r['id'][:3], r['status']) for r in run['rows'] if r['status'] != 'pass') or '-'))
    for i, first in enumerate(labels):
        for second in labels[i + 1:]:
            c = compare(runs[first], runs[second])
            out.append('SWIFT_CODING %s vs %s: solved by both %d, only %s: %s, only %s: %s, tokens %s/%s on the problems both solve = %s' % (
                first, second, c['solved_by_both'], first, ' '.join(c['only_first']) or '-', second, ' '.join(c['only_second']) or '-', second, first,
                fmt(c['token_ratio_second_over_first_on_both'], 3)))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=FILE', help='a coding-set-<profile>.json from the job\'s results (at least one; two or more are compared)')
    parser.add_argument('--problems', default=PROBLEMS)
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help='seconds per problem')
    parser.add_argument('--json-out')
    parser.add_argument('--allow-network', action='store_true', help='do not require the network namespace (default: use it when it works, run without it otherwise, and say so)')
    options = parser.parse_args(argv)
    _, problems = load_problems(options.problems)
    prefix = [] if options.allow_network else sandbox_prefix()
    print('SWIFT_CODING sandbox: python -I in a scratch directory with cpu/memory limits, network %s' % ('isolated (unshare -rn)' if prefix else 'NOT isolated (run on a disposable host)'))
    labels, runs = [], {}
    for item in options.run:
        label, separator, path = item.partition('=')
        if not separator or not label or label in runs:
            parser.error('--run needs a distinct LABEL=FILE, got %r' % item)
        runs[label] = score_run(path, problems, options.timeout, prefix)
        labels.append(label)
    for line in lines(labels, runs):
        print(line)
    if options.json_out:
        with open(options.json_out, 'w', encoding='utf-8') as handle:
            json.dump(dict(runs=runs, summary=dict((label, summarize(runs[label])) for label in labels)), handle, indent=1, sort_keys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
