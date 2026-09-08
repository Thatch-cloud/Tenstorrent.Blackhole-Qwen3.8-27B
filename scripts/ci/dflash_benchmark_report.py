"""Recompute PP / CTX / TG from complete DFlash hardware artifacts, without pooling contexts or arms."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from full_dflash_request import (summarize_dflash_requests, summarize_dflash_commit_requests,
    summarize_dflash_convolution_requests)


def report_rows(report):
    if report.get('passed') is not True:
        raise ValueError('A passed complete hardware request artifact is required')
    requests = report['request_checks']
    if len(requests) == 3:
        summaries = [('candidate', summarize_dflash_requests(requests), report['request_summary'])]
    elif len(requests) == 6:
        convolution = any(entry.get('fused_convolution') is True for entry in requests)
        summarize = summarize_dflash_convolution_requests if convolution else summarize_dflash_commit_requests
        combined = summarize(requests)
        summaries = [(arm, combined[arm], report['request_summary'][arm]) for arm in ('control', 'candidate')]
    else:
        raise ValueError('One audited arm or a complete audited ABBA experiment is required')
    output = []
    for arm, summary, recorded in summaries:
        if (report.get('context_lengths') != [summary['context']]
                or recorded.get('context') != summary['context']
                or recorded.get('streams') != 1
                or not math.isclose(recorded['committed_tokens_per_second'],
                    summary['committed_tokens_per_second'], rel_tol=1e-12)):
            raise ValueError('Recorded context, stream count and TG must agree with raw measurements')
        path = ['DFlash2', 'captured' if summary['proposal_capture'] else 'eager']
        if summary['commit_only_gdn']:
            path.append('commit-only GDN')
        if summary['fused_convolution']:
            path.append('fused convolution')
        output.append(dict(arm=arm, path=' + '.join(path), **summary['benchmark']))
    return output


def markdown(rows):
    output = ['| Run | Arm / path | PP tok/s | CTX tokens | TG tok/s | Streams (B) | Verify rows (T) | Decode tokens/request | Prefill + setup + decode |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in rows:
        counts = ', '.join(str(count) for count in row['committed_decode_tokens_per_request'])
        output.append(f"| [{row['run_id']}]({row['run_url']}) | {row['arm']}: {row['path']} "
            f"| {row['pp_tokens_per_second']:.2f} | {row['ctx_tokens']} | {row['tg_tokens_per_second']:.2f} "
            f"| {row['streams']} | Up to {row['verify_rows']} | {counts} "
            f"| {row['mean_prefill_setup_decode_ms'] / 1000:.2f} s |")
    output.extend(['', 'Offline complete coding requests, not a serving or held-out quality benchmark.',
        'CTX includes the chat template. PP includes target feature capture and first-token selection, not draft setup.',
        'TG counts committed decode tokens after the prefill seed; drafting, verification/readback and publication are included.',
        'Rates use total tokens / total measured time. Audits are excluded. Setup is not amortized; totals exclude model loading.'])
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', nargs=2, action='append', required=True, metavar=('RUN_ID', 'REPORT'))
    parser.add_argument('--format', choices=('json', 'markdown'), default='markdown')
    options = parser.parse_args()
    rows = []
    for run_id, filename in options.artifact:
        if not run_id.isascii() or not run_id.isdecimal() or int(run_id) <= 0:
            raise ValueError('A positive numeric GitHub Actions run ID is required')
        payload = Path(filename).read_bytes()
        for row in report_rows(json.loads(payload)):
            row.update(run_id=run_id,
                run_url=f'https://github.com/Thatch-cloud/Tenstorrent.Blackhole-Qwen3.8-27B/actions/runs/{run_id}',
                artifact_sha256=hashlib.sha256(payload).hexdigest())
            rows.append(row)
    print(json.dumps(rows, indent=2) if options.format == 'json' else markdown(rows))


if __name__ == '__main__':
    main()
