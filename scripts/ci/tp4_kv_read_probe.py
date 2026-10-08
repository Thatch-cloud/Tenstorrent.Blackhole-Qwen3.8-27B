"""The region-read qualification (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=kvread, C2_CARDS=quad): does ttnn.qwen_read_blocks, the device-side read the
prefix audit's cost depends on (docs/prefix-audit-cost.md), return the whole-cache read's bytes on the served (1, 4) mesh, and does it compile nothing?

    python3 -B /c2/scripts/ci/tp4_kv_read_probe.py --fabric FABRIC_1D --output /probe-results/kv-read-probe.json

The check itself is optimisation/ttnn-op/kv_region_read/kv_region_read_card.py (byte equality with the whole read for a block, a run, scattered ids, a shuffled
order and two runs; no program-cache growth; the cost per block and the split of the whole read into device read and host unpack). This wrapper only opens the
fabric the way the other four-card probes do, runs it over the pool-sized cache (19,968 blocks, one KV head per chip) and turns its JSON into one greppable line:

    KV_READ_PROBE verdict=PASS|FAIL|NOT-MEASURED ...

then the card check's JSON. Exit 0 on PASS, 1 on FAIL (a byte mismatch, a lost chip shard, program-cache growth, or no qwen_read_blocks in the image), 2 when the
check could not run at all. The job's reader is the run conclusion, so a FAIL here is a failed job: the window's P1ab-LN, E1-LN and P1a-CTL2 NEED it and are
skipped, never run on a read nobody checked.
"""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'optimisation' / 'ttnn-op' / 'kv_region_read'))

POOL_BLOCKS = 19968      # 8 seats x 262k at 64 tokens a block (docs/tp4-combined-window.md section 2)
FABRICS = ('FABRIC_1D', 'FABRIC_1D_RING')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=FABRICS, default=FABRICS[0])
    parser.add_argument('--output', required=True)
    parser.add_argument('--blocks', type=int, default=POOL_BLOCKS)
    parser.add_argument('--devices', type=int, default=4)
    return parser


def last_json(text):
    """The last top-level JSON object printed in `text` (the card check prints one, indented), or None."""
    start = text.rfind('\n{')
    candidates = [start + 1] if start >= 0 else []
    if text.startswith('{'):
        candidates.append(0)
    for begin in candidates:
        try:
            return json.loads(text[begin:])
        except ValueError:
            continue
    return None


def run(options, card=None, ttnn=None, log=print):
    """-> (report dict, exit code). `card` is the kv_region_read_card module and `ttnn` the ttnn module (fakes in the tests)."""
    report = dict(kind='kv-read-probe-quad', fabric=options.fabric, blocks=options.blocks, devices=options.devices)
    try:
        if card is None:
            import kv_region_read_card as card
        if ttnn is None:
            import ttnn
        if hasattr(ttnn, 'set_fabric_config'):
            ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = card.main(['--devices', str(options.devices), '--blocks', str(options.blocks), '--heads-per-chip', '1'])
        check = last_json(buffer.getvalue())
        report['check'] = check
        report['check_exit'] = code
        if check is None:
            report.update(verdict='NOT-MEASURED', error='the card check printed no JSON: ' + buffer.getvalue()[-400:])
            log('KV_READ_PROBE verdict=NOT-MEASURED error=%s' % report['error'])
            return report, 2
        ok = bool(check.get('ok')) and code == 0
        report['verdict'] = 'PASS' if ok else 'FAIL'
        sets = ('one', 'run', 'scattered', 'shuffled_order', 'two_runs')
        log('KV_READ_PROBE verdict=%s whole_read_s=%s whole_unpack_s=%s program_cache_growth=%s %s problems=%d' % (
            report['verdict'], check.get('whole_read_s'), check.get('whole_unpack_s'), check.get('program_cache_growth'),
            ' '.join('%s_ms=%s' % (name, (check.get(name) or {}).get('read_ms')) for name in sets), len(check.get('problems') or [])))
        return report, 0 if ok else 1
    except Exception as error:  # noqa: BLE001
        report.update(verdict='NOT-MEASURED', error='%s: %s' % (type(error).__name__, str(error)[:500]), traceback=traceback.format_exc()[-3000:])
        log('KV_READ_PROBE verdict=NOT-MEASURED error=%s' % report['error'])
        return report, 2


def main(argv=None, card=None, ttnn=None, log=print):
    options = build_parser().parse_args(argv)
    report, code = run(options, card=card, ttnn=ttnn, log=log)
    with open(options.output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    log(json.dumps(report, sort_keys=True))
    return code


if __name__ == '__main__':
    sys.exit(main())
