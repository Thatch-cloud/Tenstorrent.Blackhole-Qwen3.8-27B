"""Token agreement between two tensor-parallel configurations of one model, over the OpenAI API (stdlib only; runs on
the rig host beside c2_serving_smoke.py, Python 3.7 syntax).

    python3 tp_agreement.py collect http://127.0.0.1:8010 Qwen/Qwen3.8-27B out-tp4.json [--label tp4] [--root .]
    python3 tp_agreement.py compare out-tp2.json out-tp4.json [--report agreement.json]

WHY: TP4 sums each layer's partial products in a different order from TP2, so its greedy tokens are not bit-equal to
the pair's: near-ties flip and the text drifts after that. That is expected and is not a fault. What a TP4 bring-up
must show instead is (1) the text is coherent (not empty, not a repeated loop), (2) the two configurations agree on
a long common prefix before the first flip, and (3) the model is no less sure of its own greedy text (the mean
log-probability of the tokens each configuration chose, the perplexity a run assigns to what it wrote, is close).
`collect` writes one JSON file per configuration: a fixed set of real-text prompts (this repository's own documents
and sources, so a run needs no corpus) answered greedily, with each generated token's id and log-probability when the
server returns them. `compare` reads a reference (the pair) and a candidate (TP4) and reports, per prompt and overall,
the common-prefix length in tokens, the text similarity, the perplexity of each side and their ratio, and a verdict.

The verdict's thresholds are a starting point recorded in the report (median common prefix >= MIN_PREFIX tokens,
perplexity ratio <= MAX_PPL_RATIO, every completion coherent); the first TP4 run revises them from data, they are
reported and never used to fail a hardware job by themselves.
"""
import argparse
import difflib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

# (name, path under the checkout, characters of it, the instruction): real prose and real code, short to long.
PROMPTS = (
    ('doc-short', 'docs/decode-payload-bound.md', 3000, 'Summarise the key argument of this document in five bullet points.'),
    ('doc-medium', 'docs/gotchas.md', 12000, 'List the three most important pitfalls described here and why each matters.'),
    ('code-medium', 'scripts/ci/tp4_mesh.py', 12000, 'Explain what this module checks and why, section by section.'),
    ('doc-long', 'docs/bringup-2026-08-24.md', 48000, 'Write a concise timeline of what this bring-up log records.'),
    ('code-long', 'scripts/ci/c2_serving_gate.py', 90000, 'Describe how this driver launches and judges one arm of a gate plan.'),
)
MAX_TOKENS = 256
MIN_PREFIX = 24
MAX_PPL_RATIO = 1.10
MIN_TOKENS = 32
MIN_UNIQUE_RATIO = 0.25


def read_prompts(root):
    """[(name, user message)] for the PROMPTS whose file exists under `root`."""
    built = []
    for name, path, characters, instruction in PROMPTS:
        full = os.path.join(root, path)
        if not os.path.isfile(full):
            continue
        with open(full, encoding='utf-8', errors='replace') as handle:
            text = handle.read()[:characters]
        built.append((name, '%s\n\n---\n%s\n---' % (instruction, text)))
    return built


def post(base, path, body, timeout=1800):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(), method='POST',
                                     headers={'content-type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors='replace')[:400]


def ask(base, model, message, max_tokens=MAX_TOKENS, send=post):
    """One greedy completion: {text, tokens, logprobs, finish, logprobs_available, error}. Log-probabilities are asked
    for first; a server that refuses them (the on-device sampler cannot return them) is asked again without."""
    body = dict(model=model, messages=[{'role': 'user', 'content': message}], max_tokens=max_tokens, temperature=0,
                chat_template_kwargs={'enable_thinking': False})
    for with_logprobs in (True, False):
        request = dict(body)
        if with_logprobs:
            request.update(logprobs=True, top_logprobs=1, return_tokens_as_token_ids=True)
        status, answer = send(base, '/v1/chat/completions', request)
        if status == 200:
            choice = answer['choices'][0]
            content = (choice.get('logprobs') or {}).get('content') or []
            message_ = choice.get('message') or {}
            text = (message_.get('content') or '') + (message_.get('reasoning_content') or message_.get('reasoning') or '')
            usable = with_logprobs and len(content) > 0
            return dict(text=text, finish=choice.get('finish_reason'), logprobs_available=usable,
                        tokens=[entry.get('token') for entry in content] if usable else [],
                        logprobs=[entry.get('logprob') for entry in content] if usable else [],
                        completion_tokens=(answer.get('usage') or {}).get('completion_tokens'))
        error = 'HTTP %s: %s' % (status, str(answer)[:200])
    return dict(text='', finish=None, logprobs_available=False, tokens=[], logprobs=[], error=error)


def collect(base, model, root, label, max_tokens=MAX_TOKENS, send=post, log=print):
    prompts = read_prompts(root)
    if not prompts:
        raise SystemExit('no prompt file found under %s' % root)
    results = []
    for name, message in prompts:
        started = time.time()
        result = ask(base, model, message, max_tokens, send)
        result.update(name=name, prompt_characters=len(message), wall_s=round(time.time() - started, 1))
        results.append(result)
        log('AGREEMENT collect %s: %d tokens, logprobs=%s, %s, %.1f s%s' % (
            name, len(result['tokens']) or (result.get('completion_tokens') or 0), result['logprobs_available'],
            result.get('finish'), result['wall_s'], (' ERROR ' + result['error']) if result.get('error') else ''))
    return dict(label=label, model=model, max_tokens=max_tokens, prompts=results)


def perplexity(logprobs):
    """exp(-mean log-probability), or None for no tokens."""
    values = [value for value in logprobs if value is not None]
    if not values:
        return None
    return math.exp(-sum(values) / len(values))


def common_prefix(a, b):
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def coherent(result):
    """(ok, why): non-empty, long enough unless the model stopped, and not a repeated loop."""
    tokens = result.get('tokens') or result.get('text', '').split()
    if result.get('error') or not result.get('text', '').strip():
        return False, 'empty or errored'
    if len(tokens) < MIN_TOKENS and result.get('finish') != 'stop':
        return False, 'only %d tokens and no stop' % len(tokens)
    if len(tokens) >= MIN_TOKENS and len(set(tokens)) / float(len(tokens)) < MIN_UNIQUE_RATIO:
        return False, 'repeated loop (%d distinct of %d tokens)' % (len(set(tokens)), len(tokens))
    return True, ''


def compare(reference, candidate, min_prefix=MIN_PREFIX, max_ratio=MAX_PPL_RATIO):
    """The per-prompt and overall agreement of `candidate` with `reference` (both collect() outputs)."""
    by_name = dict((entry['name'], entry) for entry in candidate['prompts'])
    rows = []
    for ref in reference['prompts']:
        cand = by_name.get(ref['name'])
        if cand is None:
            rows.append(dict(name=ref['name'], missing=True))
            continue
        exact_tokens = ref.get('logprobs_available') and cand.get('logprobs_available')
        left = ref['tokens'] if exact_tokens else ref['text']
        right = cand['tokens'] if exact_tokens else cand['text']
        prefix = common_prefix(left, right)
        ok, why = coherent(cand)
        ref_ppl, cand_ppl = perplexity(ref['logprobs']), perplexity(cand['logprobs'])
        rows.append(dict(name=ref['name'], unit='tokens' if exact_tokens else 'characters', common_prefix=prefix,
                         reference_length=len(left), candidate_length=len(right),
                         identical=left == right,
                         text_similarity=round(difflib.SequenceMatcher(None, ref['text'], cand['text']).ratio(), 4),
                         reference_perplexity=ref_ppl, candidate_perplexity=cand_ppl,
                         perplexity_ratio=(cand_ppl / ref_ppl) if ref_ppl and cand_ppl else None,
                         coherent=ok, incoherent_because=why))
    present = [row for row in rows if not row.get('missing')]
    prefixes = sorted(row['common_prefix'] for row in present if row['unit'] == 'tokens')
    ratios = [row['perplexity_ratio'] for row in present if row['perplexity_ratio'] is not None]
    median = prefixes[len(prefixes) // 2] if prefixes else None
    reasons = []
    if len(present) != len(reference['prompts']):
        reasons.append('%d prompts missing from the candidate' % (len(reference['prompts']) - len(present)))
    for row in present:
        if not row['coherent']:
            reasons.append('%s incoherent: %s' % (row['name'], row['incoherent_because']))
    if median is None:
        reasons.append('no token-level comparison (a side returned no log-probabilities): text similarity only')
    elif median < min_prefix:
        reasons.append('median common prefix %d tokens < %d' % (median, min_prefix))
    if ratios and max(ratios) > max_ratio:
        reasons.append('perplexity ratio %.3f > %.2f on %s' % (max(ratios), max_ratio, [
            row['name'] for row in present if (row['perplexity_ratio'] or 0) > max_ratio]))
    return dict(reference=reference.get('label'), candidate=candidate.get('label'), prompts=rows,
                median_common_prefix=median, max_perplexity_ratio=max(ratios) if ratios else None,
                mean_perplexity_ratio=(sum(ratios) / len(ratios)) if ratios else None,
                thresholds=dict(min_prefix=min_prefix, max_perplexity_ratio=max_ratio),
                verdict='AGREE' if not reasons else 'REVIEW', reasons=reasons)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('collect')
    one.add_argument('base')
    one.add_argument('model')
    one.add_argument('output')
    one.add_argument('--label', default='run')
    one.add_argument('--root', default='.')
    one.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    two = sub.add_parser('compare')
    two.add_argument('reference')
    two.add_argument('candidate')
    two.add_argument('--report', default=None)
    two.add_argument('--min-prefix', type=int, default=MIN_PREFIX)
    two.add_argument('--max-ratio', type=float, default=MAX_PPL_RATIO)
    return parser


def main(argv=None, send=post, log=print):
    options = build_parser().parse_args(argv)
    if options.command == 'collect':
        result = collect(options.base, options.model, options.root, options.label, options.max_tokens, send, log)
        with open(options.output, 'w') as handle:
            json.dump(result, handle, indent=1)
        return 0 if all(not entry.get('error') for entry in result['prompts']) else 1
    with open(options.reference) as handle:
        reference = json.load(handle)
    with open(options.candidate) as handle:
        candidate = json.load(handle)
    report = compare(reference, candidate, options.min_prefix, options.max_ratio)
    if options.report:
        with open(options.report, 'w') as handle:
            json.dump(report, handle, indent=1)
    for row in report['prompts']:
        log('AGREEMENT %s: %s' % (row['name'], json.dumps(row)))
    log('AGREEMENT verdict=%s median_common_prefix=%s max_perplexity_ratio=%s%s' % (
        report['verdict'], report['median_common_prefix'], report['max_perplexity_ratio'],
        (' reasons: ' + '; '.join(report['reasons'])) if report['reasons'] else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
