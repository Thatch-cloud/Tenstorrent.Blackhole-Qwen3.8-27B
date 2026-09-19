"""Device profile of prefill at several prompt lengths. Attribution only, no PP claim.

Prefill measures 4,572 tok/s at 45k and about 1,180 tok/s at 262k, against roughly
28,600 tok/s if it were compute-bound at two cards of fp8 peak. So it runs at 4-16% of
compute and degrades with context, which points at attention rather than the
projections - but nothing in the profiles collected so far covers prefill, so that is
inference, not evidence.

Runs in-process through vLLM's offline LLM rather than the API server: the device
profiler can only see the process that opens the card, and a server subprocess is
invisible to it. Wrapped by dspark-combined-device-profile.sh, which sets
TT_METAL_DEVICE_PROFILER and the mid-run dump that keeps memory inside the container
limit.

One token of output per prompt, so the measurement is prefill plus a single decode
step, and the decode step is already attributed elsewhere.
"""

import argparse
import json
import os
import time

BEGIN = '<<<PREFILL_PROFILE_JSON_BEGIN>>>'
END = '<<<PREFILL_PROFILE_JSON_END>>>'
UNIT = 'compute the answer carefully. '


def build(tokens):
    """Roughly `tokens` tokens; undershoots deliberately, usage reports the truth."""
    return 'def solve(n):\n    # ' + UNIT * max(1, tokens // 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--context', type=int, default=65536)
    parser.add_argument('--lengths', default='4096,16384,32768')
    parser.add_argument('--signpost', action='store_true',
                        help='emit tracy signposts around each prefill so the device '
                             'rows can be attributed to a prompt length')
    options = parser.parse_args()

    report = dict(scope=__doc__, context=options.context, results=[])
    from vllm import LLM, SamplingParams

    signpost = None
    if options.signpost:
        try:
            from tracy import signpost as signpost
        except ImportError:
            report['signpost'] = 'tracy signpost unavailable'

    # num_gpu_blocks_override is not optional here: every working runner on this stack
    # passes it, and without it vLLM runs its own memory profiling pass to size the
    # cache, which is not something to discover inside a profiled run.
    blocks = -(-options.context // 64)
    llm = LLM(model=options.model, dtype='bfloat16', max_model_len=options.context,
              max_num_seqs=1, block_size=64, enforce_eager=False,
              num_gpu_blocks_override=blocks, enable_prefix_caching=False,
              additional_config=dict(tt=dict(trace_mode='decode_only',
                                             trace_region_size=1073741824,
                                             l1_small_size=24576)))
    report['blocks'] = blocks
    sampling = SamplingParams(max_tokens=1, temperature=0.0)
    for target in [int(v) for v in options.lengths.split(',')]:
        prompt = build(target)
        entry = dict(target_tokens=target)
        try:
            if signpost is not None:
                signpost('qwen_prefill_%d_begin' % target)
            started = time.perf_counter()
            out = llm.generate([prompt], sampling)
            elapsed = time.perf_counter() - started
            if signpost is not None:
                signpost('qwen_prefill_%d_end' % target)
            got = len(out[0].prompt_token_ids)
            entry.update(prompt_tokens=got, seconds=round(elapsed, 3),
                         tokens_per_s=round(got / elapsed, 1))
        except BaseException as error:
            entry['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
        report['results'].append(entry)
        print('prefill %7s tokens -> %s' % (entry.get('prompt_tokens'), entry.get('tokens_per_s')),
              flush=True)

    ok = [r for r in report['results'] if r.get('tokens_per_s')]
    if len(ok) > 1:
        first, last = ok[0], ok[-1]
        report['degradation'] = dict(
            from_tokens=first['prompt_tokens'], from_tok_s=first['tokens_per_s'],
            to_tokens=last['prompt_tokens'], to_tok_s=last['tokens_per_s'],
            ratio=round(first['tokens_per_s'] / last['tokens_per_s'], 2))
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
