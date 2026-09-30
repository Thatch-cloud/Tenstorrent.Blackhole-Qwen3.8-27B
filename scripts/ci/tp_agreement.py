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

Each prompt is answered TWICE, because the server samples two ways. A request with logprobs is sampled on the host
(at four devices the TT plugin's check_perform_device_sampling refuses device sampling for any batch that has a
logprobs request), and that is the only way to read per-token log-probabilities; a request without them takes the
on-device sampler the TP4 profiles enable (sample_on_device_mode decode_only), the path real traffic uses. `collect`
therefore records the host answer (top-TOP_K log-probabilities per token) and the device answer, and checks the device
text against the host text: identical, or first differing where the host's top two candidates are within
TIE_MARGIN (a documented bfloat16 near-tie). A device sampler that is wrong shows there.

`compare` judges, on top of the prefix and perplexity figures: (a) both files answered the same prompts (each prompt's
sha256 is stored, and a mismatch is NOT_COMPARABLE, not a verdict); (b) at the first token where the two
configurations part, each one's chosen token must be inside the other's top-TOP_K and within DIVERGENCE_MARGIN
nats of the other's choice (a real rounding flip is a near-tie; a wrong model's flip is not); (c) the perplexity
ratio must lie in [1 / MAX_PPL_RATIO, MAX_PPL_RATIO], since a confidently wrong model scores itself LOWER.
`self_check` is the part a job can fail on (c2_serving_smoke.py exits non-zero when it is not ok): every completion
coherent, no request errored, the device sampler agreeing with the host, and a sane perplexity of its own text.

The verdict's other thresholds are a starting point recorded in the report; the first TP4 run revises them from data.
"""
import argparse
import difflib
import hashlib
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
TOP_K = 5
# bfloat16 logits of magnitude 16-32 are 0.125 apart, and a different summation order moves a logit by a few of them:
# two candidates within half a nat at the step the configurations part are a rounding tie, not a different model.
DIVERGENCE_MARGIN = 0.5
# The host's top-1 to top-2 gap (nats) under which a device-sampled token that differs from the host's is a tie.
TIE_MARGIN = 0.5
# A model's own greedy text has a perplexity near 1-2; a distribution this flat is garbage.
MAX_SELF_PERPLEXITY = 6.0
SCHEMA = 2
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


def ask(base, model, message, max_tokens=MAX_TOKENS, send=post, logprobs=True):
    """One greedy completion: {text, tokens, logprobs, tops, piece_bytes, finish, logprobs_available, error}.
    logprobs=True asks for the top-TOP_K log-probabilities per token (the server samples this request on the host),
    and a server that refuses them is asked again without; logprobs=False is the on-device sampler's request."""
    body = dict(model=model, messages=[{'role': 'user', 'content': message}], max_tokens=max_tokens, temperature=0,
                chat_template_kwargs={'enable_thinking': False})
    for with_logprobs in ((True, False) if logprobs else (False,)):
        request = dict(body)
        if with_logprobs:
            request.update(logprobs=True, top_logprobs=TOP_K, return_tokens_as_token_ids=True)
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
                        tops=[[[top.get('token'), top.get('logprob')] for top in (entry.get('top_logprobs') or ())]
                              for entry in content] if usable else [],
                        piece_bytes=[len(entry['bytes']) if isinstance(entry.get('bytes'), list) else None
                                     for entry in content] if usable else [],
                        completion_tokens=(answer.get('usage') or {}).get('completion_tokens'))
        error = 'HTTP %s: %s' % (status, str(answer)[:200])
    return dict(text='', finish=None, logprobs_available=False, tokens=[], logprobs=[], tops=[], piece_bytes=[],
                error=error)


def prompt_hash(message):
    return hashlib.sha256(message.encode('utf-8')).hexdigest()


def collect(base, model, root, label, max_tokens=MAX_TOKENS, send=post, log=print):
    prompts = read_prompts(root)
    if not prompts:
        raise SystemExit('no prompt file found under %s' % root)
    results = []
    for name, message in prompts:
        started = time.time()
        result = ask(base, model, message, max_tokens, send)
        # the same prompt again without logprobs: the on-device sampler's path (the host answered the one above)
        device = ask(base, model, message, max_tokens, send, logprobs=False)
        result['device'] = dict(text=device['text'], finish=device.get('finish'), error=device.get('error'),
                                completion_tokens=device.get('completion_tokens'))
        result.update(name=name, prompt_characters=len(message), prompt_sha256=prompt_hash(message),
                      wall_s=round(time.time() - started, 1))
        results.append(result)
        match = device_check(result)
        log('AGREEMENT collect %s: %d tokens, logprobs=%s, %s, %.1f s, device_vs_host=%s%s' % (
            name, len(result['tokens']) or (result.get('completion_tokens') or 0), result['logprobs_available'],
            result.get('finish'), result['wall_s'], 'ok' if match['ok'] else 'DIFFERS (%s)' % match['why'],
            (' ERROR ' + result['error']) if result.get('error') else ''))
    return dict(label=label, model=model, max_tokens=max_tokens, schema=SCHEMA, prompts=results)


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


def device_check(entry):
    """{ok, identical, why, gap}: the on-device sampler's text against the host sampler's for one prompt. Identical is
    ok; a first difference is ok only where the host's own top two candidates were within TIE_MARGIN (the device
    took the other one of a bfloat16 near-tie). Without the device answer, or with a difference the host's
    log-probabilities cannot place, it is not ok."""
    device = entry.get('device')
    if not device:
        return dict(ok=False, identical=False, why='no device-sampled answer recorded', gap=None)
    if device.get('error') or entry.get('error'):
        return dict(ok=False, identical=False, why='request errored: %s' % (device.get('error') or entry.get('error')),
                    gap=None)
    host_text, device_text = entry.get('text', ''), device.get('text', '')
    if host_text == device_text:
        return dict(ok=True, identical=True, why='', gap=None)
    at = common_prefix(host_text, device_text)
    gap, why = tie_gap_at(entry, at)
    if gap is None:
        return dict(ok=False, identical=False, why='texts differ at character %d and %s' % (at, why), gap=None)
    if gap <= TIE_MARGIN:
        return dict(ok=True, identical=False, why='near-tie at character %d (host top-2 gap %.3f nats)' % (at, gap),
                    gap=gap)
    return dict(ok=False, identical=False, why='texts differ at character %d where the host was sure (top-2 gap '
                '%.3f nats > %.2f)' % (at, gap, TIE_MARGIN), gap=gap)


def tie_gap_at(entry, char_index):
    """(top-1 minus top-2 host log-probability at the token holding `char_index` of the text, '') or (None, why)."""
    tops, sizes = entry.get('tops') or [], entry.get('piece_bytes') or []
    if not entry.get('logprobs_available') or not tops or len(sizes) != len(tops) or any(size is None for size in sizes):
        return None, 'the host answer has no per-token candidates to place it'
    text = entry.get('text', '')
    offset = len(text[:char_index].encode('utf-8'))
    total = 0
    for index, size in enumerate(sizes):
        if total <= offset < total + size:
            candidates = sorted((top[1] for top in tops[index] if top[1] is not None), reverse=True)
            if len(candidates) < 2:
                return None, 'the host reported fewer than two candidates at that token'
            return candidates[0] - candidates[1], ''
        total += size
    return None, 'the difference lies past the host answer\'s tokens'


def first_divergence(ref, cand):
    """The first place two token-level answers part: {index, ok, reason, margin_ref, margin_cand}; None when they do
    not (one is a prefix of the other, or they are equal). ok: each side's token is in the other's top-TOP_K and
    within DIVERGENCE_MARGIN nats of the other side's own choice."""
    index = common_prefix(ref['tokens'], cand['tokens'])
    if index >= len(ref['tokens']) or index >= len(cand['tokens']):
        if len(ref['tokens']) == len(cand['tokens']):
            return None
        return dict(index=index, ok=False, reason='one answer ends at token %d where the other continues' % index,
                    margin_ref=None, margin_cand=None)
    chose_ref, chose_cand = ref['tokens'][index], cand['tokens'][index]
    ref_tops = dict((token, logprob) for token, logprob in (ref.get('tops') or [[]] * (index + 1))[index])
    cand_tops = dict((token, logprob) for token, logprob in (cand.get('tops') or [[]] * (index + 1))[index])
    if chose_cand not in ref_tops or chose_ref not in cand_tops:
        return dict(index=index, ok=False, margin_ref=None, margin_cand=None,
                    reason='at token %d each side\'s choice is outside the other\'s top-%d' % (index, TOP_K))
    # how far below its own top choice each side scores the other's token
    margin_ref = ref_tops[chose_ref] - ref_tops[chose_cand]
    margin_cand = cand_tops[chose_cand] - cand_tops[chose_ref]
    ok = margin_ref <= DIVERGENCE_MARGIN and margin_cand <= DIVERGENCE_MARGIN
    return dict(index=index, ok=ok, margin_ref=margin_ref, margin_cand=margin_cand,
                reason='' if ok else 'at token %d the sides are %.2f and %.2f nats apart (limit %.2f)' % (
                    index, margin_ref, margin_cand, DIVERGENCE_MARGIN))


def self_check(result, max_perplexity=MAX_SELF_PERPLEXITY):
    """(ok, reasons) for one configuration's collect() output alone: what the smoke can fail on. Every prompt answered
    without error, coherent, its device-sampled answer agreeing with its host-sampled one, and the model no more
    puzzled by its own greedy text than a working model is. It cannot say the model is RIGHT (a confidently wrong
    model passes); compare against the reference does."""
    reasons = []
    for entry in result.get('prompts') or ():
        name = entry.get('name')
        if entry.get('error'):
            reasons.append('%s errored: %s' % (name, entry['error']))
            continue
        ok, why = coherent(entry)
        if not ok:
            reasons.append('%s incoherent: %s' % (name, why))
        check = device_check(entry)
        if not check['ok']:
            reasons.append('%s device sampler vs host: %s' % (name, check['why']))
        own = perplexity(entry.get('logprobs') or [])
        if own is not None and own > max_perplexity:
            reasons.append('%s: perplexity %.2f of its own greedy text > %.1f' % (name, own, max_perplexity))
    if not result.get('prompts'):
        reasons.append('no prompt was answered')
    return not reasons, reasons


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
    unlike = [ref['name'] for ref in reference['prompts'] if ref['name'] in by_name and (
        ref.get('prompt_sha256') is None or by_name[ref['name']].get('prompt_sha256') != ref.get('prompt_sha256'))]
    if unlike:
        # Prompts are read from each job's own checkout: a changed file is another prompt, and two answers to two
        # prompts agree or differ by nothing.
        return dict(reference=reference.get('label'), candidate=candidate.get('label'), prompts=[],
                    median_common_prefix=None, max_perplexity_ratio=None, mean_perplexity_ratio=None,
                    thresholds={}, verdict='NOT_COMPARABLE',
                    reasons=['the two runs did not ask the same prompts (sha256 differs or is absent for %s): '
                             'collect both from the same checkout' % ', '.join(unlike)])
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
        parting = first_divergence(ref, cand) if exact_tokens else None
        rows.append(dict(divergence=parting, candidate_device_vs_host=device_check(cand),
                         reference_device_vs_host=device_check(ref), name=ref['name'], unit='tokens' if exact_tokens else 'characters', common_prefix=prefix,
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
    for row in present:
        if row.get('divergence') and not row['divergence']['ok']:
            reasons.append('%s: %s' % (row['name'], row['divergence']['reason']))
        if not row['candidate_device_vs_host']['ok']:
            reasons.append('%s candidate device sampler vs host: %s' % (row['name'], row['candidate_device_vs_host']['why']))
    # two-sided: a confidently wrong model assigns its own text a LOWER perplexity
    off = [row['name'] for row in present if row['perplexity_ratio'] is not None and not (
        1.0 / max_ratio <= row['perplexity_ratio'] <= max_ratio)]
    if off:
        reasons.append('perplexity ratio outside [%.3f, %.2f] on %s (max %.3f, min %.3f)' % (
            1.0 / max_ratio, max_ratio, off, max(ratios), min(ratios)))
    return dict(reference=reference.get('label'), candidate=candidate.get('label'), prompts=rows,
                median_common_prefix=median, max_perplexity_ratio=max(ratios) if ratios else None,
                mean_perplexity_ratio=(sum(ratios) / len(ratios)) if ratios else None,
                thresholds=dict(min_prefix=min_prefix, max_perplexity_ratio=max_ratio,
                                min_perplexity_ratio=1.0 / max_ratio, divergence_margin_nats=DIVERGENCE_MARGIN,
                                tie_margin_nats=TIE_MARGIN, top_k=TOP_K),
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
    return 2 if report['verdict'] == 'NOT_COMPARABLE' else 0


if __name__ == '__main__':
    sys.exit(main())
