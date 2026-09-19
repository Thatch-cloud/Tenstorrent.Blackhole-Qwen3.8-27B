"""Group the prefill device profile by what an op actually does.

Written to correct a reading error rather than to add a measurement. The first
pass grouped by op name and put AllGatherMinimalMatmulAsyncOp in a "collectives"
bucket, which made collectives 37.2% of prefill and arithmetic 17%. That op is a
fused all-gather and matmul, so the bucket charged its matmul to communication
and the conclusion inverted: it argued for a sharding change that would have cost
decode 24.6 ms per step to buy back time prefill was not actually spending.

The fused op cannot be split from the profile, so it is split by traffic instead:
the communication it must do is bounded by the bytes TP2 forces across the link,
and priced at the separately measured fabric rate. Both ends of that bound are
reported, because a single number here would hide the estimate.
"""

import argparse
import collections
import csv
import io
import json

# Only these move bytes between cards and nothing else. AllGatherAsync is the
# layernorm-statistic gather (898 calls against LayerNormPreAllGather's 897),
# which is why it is tiny.
PURE_COMM = ('ReduceScatterMinimalAsyncDeviceOperation',
             'AllGatherAsyncDeviceOperation')
FUSED = 'AllGatherMinimalMatmulAsyncOp'
ARITH = ('MatmulDeviceOperation', 'SDPAOperation', 'SdpaDecodeDeviceOperation')
LAYOUT = ('SliceDeviceOperation', 'TilizeDeviceOperation',
          'UntilizeWithUnpaddingDeviceOperation', 'TilizeWithValPaddingDeviceOperation',
          'ReshapeViewDeviceOperation', 'ConcatDeviceOperation',
          'ShardedToInterleavedDeviceOperation', 'InterleavedToShardedDeviceOperation',
          'TransposeDeviceOperation', 'CopyDeviceOperation', 'TypecastDeviceOperation')
GDN = ('ChunkGdnScanOperation', 'ChunkGdnPrepOperation', 'GdnConvGatesDeviceOperation',
       'DecodeGatedDeltaRuleDeviceOperation')


def read(path, device='0', column='DEVICE KERNEL DURATION [ns]'):
    total = collections.Counter()
    calls = collections.Counter()
    handle = io.open(path, encoding='utf-8', errors='replace')
    for row in csv.DictReader(handle):
        if (row.get('DEVICE ID') or '').strip() != device:
            continue
        name = (row.get('OP NAME') or '').strip()
        try:
            value = float((row.get(column) or '').strip())
        except ValueError:
            continue
        total[name] += value
        calls[name] += 1
    return total, calls


def link_bytes(tokens, layers, hidden):
    """Bytes crossing the link per card for one combine per layer.

    TP2 shards the hidden dimension, so each card holds half and must receive the
    other half once per layer per token. This is a lower bound on a full TP2
    round-trip, which needs a reduce-scatter and an all-gather.
    """
    return tokens * layers * (hidden // 2) * 2


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--tokens', type=int, default=2568 + 10248 + 20488)
    parser.add_argument('--layers', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--fabric-gb-s', type=float, default=83.74,
                        help='measured all-gather ceiling, run 35424379930')
    parser.add_argument('--json')
    options = parser.parse_args()

    total, calls = read(options.csv)
    ms = dict((name, value / 1e6) for name, value in total.items())
    wall = sum(ms.values())

    def group(names):
        return sum(ms.get(name, 0.0) for name in names)

    traffic = link_bytes(options.tokens, options.layers, options.hidden)
    fused_ms = ms.get(FUSED, 0.0)
    pure_ms = group(PURE_COMM)
    scatter_ms = ms.get('ReduceScatterMinimalAsyncDeviceOperation', 0.0)

    # The reduce-scatter carries one combine per layer, so its observed rate is
    # what this fabric delivers for this traffic under real conditions. Pricing
    # the fused gather at the clean-probe ceiling is the optimistic end; pricing
    # it at the reduce-scatter own rate is the pessimistic end.
    ceiling_ms = 1e3 * traffic / (options.fabric_gb_s * 1e9)
    observed_gb_s = traffic / (scatter_ms / 1e3) / 1e9 if scatter_ms else 0.0
    observed_ms = 1e3 * traffic / (observed_gb_s * 1e9) if observed_gb_s else 0.0

    bounds = []
    for label, gather_ms in (('optimistic', ceiling_ms), ('pessimistic', observed_ms)):
        gather_ms = min(gather_ms, fused_ms)
        bounds.append(dict(
            label=label,
            fused_communication_ms=round(gather_ms, 1),
            fused_matmul_ms=round(fused_ms - gather_ms, 1),
            communication_ms=round(pure_ms + gather_ms, 1),
            communication_pct=round(100 * (pure_ms + gather_ms) / wall, 1),
            arithmetic_ms=round(group(ARITH) + fused_ms - gather_ms, 1),
            arithmetic_pct=round(100 * (group(ARITH) + fused_ms - gather_ms) / wall, 1)))

    report = dict(
        wall_ms=round(wall, 1),
        op_types=len(ms),
        calls=sum(calls.values()),
        fused_ms=round(fused_ms, 1),
        fused_pct=round(100 * fused_ms / wall, 2),
        pure_communication_ms=round(pure_ms, 1),
        pure_communication_pct=round(100 * pure_ms / wall, 2),
        layout_ms=round(group(LAYOUT), 1),
        layout_pct=round(100 * group(LAYOUT) / wall, 2),
        gdn_ms=round(group(GDN), 1),
        gdn_pct=round(100 * group(GDN) / wall, 2),
        link_traffic_gb=round(traffic / 1e9, 2),
        fabric_ceiling_gb_s=options.fabric_gb_s,
        reduce_scatter_gb_s=round(observed_gb_s, 1),
        reduce_scatter_pct_of_ceiling=round(100 * observed_gb_s / options.fabric_gb_s, 0),
        bounds=bounds)

    print('prefill device time %.1f ms, %d calls over %d op types'
          % (wall, report['calls'], report['op_types']))
    print('  fused all-gather+matmul %7.1f ms  %5.2f%%  (NOT pure communication)'
          % (fused_ms, report['fused_pct']))
    print('  pure communication      %7.1f ms  %5.2f%%'
          % (pure_ms, report['pure_communication_pct']))
    print('  layout / data movement  %7.1f ms  %5.2f%%' % (group(LAYOUT), report['layout_pct']))
    print('  GDN family              %7.1f ms  %5.2f%%' % (group(GDN), report['gdn_pct']))
    print('  link traffic %.2f GB; reduce-scatter achieves %.1f GB/s, %d%% of the %.1f probe ceiling'
          % (report['link_traffic_gb'], observed_gb_s,
             report['reduce_scatter_pct_of_ceiling'], options.fabric_gb_s))
    for bound in bounds:
        print('  [%-11s] communication %6.1f ms %5.1f%%   arithmetic %6.1f ms %5.1f%%'
              % (bound['label'], bound['communication_ms'], bound['communication_pct'],
                 bound['arithmetic_ms'], bound['arithmetic_pct']))

    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
