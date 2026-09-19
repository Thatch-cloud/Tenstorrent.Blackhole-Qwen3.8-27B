"""Gate the GDN conv dispatch fix: identical tokens, and prove the path changed.

Two arms run this, one with the stock tp.py and one with the dispatch patched so
a full chunk takes the native ttnn.conv1d. The fix is only worth keeping if the
two produce the SAME tokens, because the argument for it is an equivalence
claimed in a source comment rather than a measured one.

Greedy sampling with a fixed prompt makes the generated ids a deterministic
function of the whole prefill, so comparing them tests the conv path's numerics
end to end rather than at a tolerance someone chose.

The timing is secondary and reported without ceremony. A speedup with diverging
tokens is not a result, it is a bug.
"""

import argparse
import io
import json
import os
import sys
import time

BEGIN = '<<<GDN_CONV_GATE_JSON_BEGIN>>>'
END = '<<<GDN_CONV_GATE_JSON_END>>>'


def build(tokens):
    """A deterministic prompt of roughly the requested token count."""
    word = 'the quick brown fox jumps over the lazy dog '
    return (word * (tokens // 9 + 2))[:tokens * 5]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--context', type=int, default=65536)
    parser.add_argument('--lengths', default='2048,6144')
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--arm', default='baseline')
    parser.add_argument('--json')
    options = parser.parse_args()

    report = {'arm': options.arm, 'max_tokens': options.max_tokens, 'results': []}
    from vllm import LLM, SamplingParams

    blocks = -(-options.context // 64)
    llm = LLM(model=options.model, dtype='bfloat16', max_model_len=options.context,
              max_num_seqs=1, block_size=64, enforce_eager=False,
              num_gpu_blocks_override=blocks, enable_prefix_caching=False,
              additional_config=dict(tt=dict(trace_mode='decode_only',
                                             trace_region_size=1073741824,
                                             l1_small_size=24576)))
    report['blocks'] = blocks
    # Greedy, and ignore_eos so a short answer cannot shorten one arm's id list
    # and make a divergence look like a length difference.
    sampling = SamplingParams(max_tokens=options.max_tokens, temperature=0.0,
                              ignore_eos=True)

    for target in [int(v) for v in options.lengths.split(',')]:
        prompt = build(target)
        entry = {'target_tokens': target}
        try:
            started = time.perf_counter()
            out = llm.generate([prompt], sampling)
            elapsed = time.perf_counter() - started
            completion = out[0].outputs[0]
            entry.update(prompt_tokens=len(out[0].prompt_token_ids),
                         seconds=round(elapsed, 3),
                         token_ids=list(completion.token_ids),
                         text=completion.text[:120])
            entry['tokens_per_s'] = round(entry['prompt_tokens'] / elapsed, 1)
        except BaseException as error:
            entry['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
        report['results'].append(entry)
        print('prefill %7s tokens -> %s tok/s, ids %s'
              % (entry.get('prompt_tokens'), entry.get('tokens_per_s'),
                 entry.get('token_ids')), flush=True)

    ok = [r for r in report['results'] if r.get('token_ids')]
    report['arms_complete'] = len(ok) == len(report['results'])
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
