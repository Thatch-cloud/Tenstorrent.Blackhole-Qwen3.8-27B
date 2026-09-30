"""Ceiling for batched speculative decode, at the rescoped 131k context.

The question this answers: batching the fast path means adding a batch axis
through the draft device, verifier, masks, bridge and hook - weeks of
architectural work. Before committing that, is the result worth having?

Everything here is measured on this rig except where marked:

  405 GB/s      achieved DRAM bandwidth per card, run 35416870419 (measured;
                the spec is 512 and the earlier planning used 85% of it)
  9.96 GB       weight bytes per card per verifier pass under TP2
  32 KB/token   KV at bf8: 2 (K,V) x 1024 kv-width x 16 full-attention layers.
                The 48 GDN layers hold ~40 MB/user of recurrent state, flat in
                context, so they are excluded
  24.67 ms      draft, measured
  5.34 ms       selection and commit, measured
  12.1 / 16     tokens committed per T16 cycle, measured at 4096 context

The CEILING is the rate at the bandwidth floor: every non-DRAM overhead set to
zero. It is not achievable - today's verifier carries about 33 ms of non-weight
work - but it bounds what the architectural work could ever buy, which is the
decision at hand.
"""

import argparse

DRAM_GB_S = 405.0
WEIGHT_GB_PER_CARD = 9.96
KV_BYTES_PER_TOKEN = {'bf8': 32768.0, 'bf4': 16384.0}
DRAFT_MS = 24.67
SELECT_MS = 5.34
CARDS = 2


def cycle_floor_ms(context, users, kv, committed):
    """Irreducible DRAM time for one speculative cycle, plus measured draft and select."""
    weights_ms = 1e3 * WEIGHT_GB_PER_CARD * 1e9 / (DRAM_GB_S * 1e9)
    kv_bytes = context * KV_BYTES_PER_TOKEN[kv] * users / CARDS
    kv_ms = 1e3 * kv_bytes / (DRAM_GB_S * 1e9)
    return weights_ms + kv_ms + DRAFT_MS + SELECT_MS, weights_ms, kv_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', type=float, default=150.0)
    options = parser.parse_args()

    print('ceiling = tokens committed / cycle floor, with ALL overhead set to zero')
    print('target  = %.0f tok/s per user\n' % options.target)
    print('%8s %6s %5s %9s %9s %9s %9s %9s' %
          ('context', 'users', 'kv', 'weights', 'kv ms', 'floor ms', 'ceiling', 'vs target'))
    rows = []
    for context in (131072, 163840):
        for users in (4, 8):
            for kv in ('bf8', 'bf4'):
                floor, w_ms, kv_ms = cycle_floor_ms(context, users, kv, 12.1)
                ceiling = 12.1 / (floor / 1e3)
                rows.append((context, users, kv, ceiling, floor))
                print('%8d %6d %5s %9.1f %9.1f %9.1f %9.1f %+9.1f' %
                      (context, users, kv, w_ms, kv_ms, floor, ceiling,
                       ceiling - options.target))
    print()

    # the decision case, and how much overhead it leaves
    floor, _, _ = cycle_floor_ms(131072, 4, 'bf8', 12.1)
    budget = 12.1 / options.target * 1e3
    print('DECISION CASE: 131k, 4 users, bf8, acceptance 12.1')
    print('  cycle floor            %6.1f ms   (irreducible)' % floor)
    print('  budget at %3.0f tok/s    %6.1f ms' % (options.target, budget))
    print('  overhead allowed       %6.1f ms' % (budget - floor))
    print('  overhead today         %6.1f ms   (non-weight verifier work)' % 33.0)
    print('  required reduction     %5.0f%%' % (100 * (1 - (budget - floor) / 33.0)))
    print()
    print('sensitivity of the ceiling to acceptance, which was measured at 4096 and')
    print('has never been measured at long context:')
    for committed in (10.0, 11.0, 12.1, 13.0, 14.0):
        print('  %4.1f tokens/cycle -> %6.1f tok/s' % (committed, committed / (floor / 1e3)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
