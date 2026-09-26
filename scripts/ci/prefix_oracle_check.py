"""The prefix gates' oracle (prefix_judge.Oracle) against the REAL scheduler graft, request by request.

The hardware gates fail a sequential arm when a request's Q differs from the oracle's, so the oracle
must be the scheduler graft's rules and nothing else. This drives both with the same requests:

  - the graft: vLLM 0.25.1's real TTScheduler and KVCacheManager with prefix_scheduler_graft
    installed, built from the general-prefix engine config the way the P0a probe builds it
    (prefix_p0a_probe.SchedEnv / Drive: ttnn and the TT model stubbed, no device);
  - the oracle: prefix_judge.Oracle fed the same prompts and salts in the same order.

SCENARIOS covers the design's worked cases: a chain whose turns extend the previous prompt, a changed
suffix, an early divergence falling back to an older checkpoint, a fresh salt, an unsalted request,
previous prompts of 2047/2048/2049/4096 tokens with re-sends (the num_tokens - 1 cap), three
conversations of one tenant sharing a 4,300-token block (the gap capture), and a three-entry store
(LRU eviction). Prompts are fixed token lists, so decode output never enters them.

    python3 scripts/ci/prefix_oracle_check.py          # inside the serving image, as the probe runs

prints one line per request and 'ORACLE_VS_GRAFT <n> requests, <m> mismatches'; exits 1 on any
mismatch. GOLDEN is what the real graft committed on 2026-09-26 (vLLM 0.25.1 source, plugin
bf77cd63, scripts/ci at prefix/g1-harness), which test_prefix_oracle_check holds the oracle to on CPU.
"""

import random
import sys

import prefix_judge as judge

CHUNK = judge.CHUNK


def tokens(count, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 200000) for _ in range(count)]


def scenarios():
    """[(session, capacity or None, [(request id, prompt tokens, salt)])], deterministic."""
    out = []
    p1 = tokens(2600, 'p1')
    a1, a2, a3 = tokens(700, 'a1'), tokens(700, 'a2'), tokens(300, 'a3')
    p2 = p1 + a1 + tokens(1500, 'i2')
    p3 = p2 + a2 + tokens(5000, 'i3')
    p4 = p3 + a3 + tokens(900, 'i4')
    chain = [('chain-1', p1, 'c'), ('chain-2', p2, 'c'), ('chain-3', p3, 'c'), ('chain-4', p4, 'c'),
             ('chain-suffix', p3 + a3 + tokens(900, 'other'), 'c'),
             ('chain-early', p1 + a1 + tokens(1500, 'diverged') + a2 + tokens(5000, 'i3'), 'c'),
             ('chain-fresh-salt', p4, 'fresh-salt'), ('chain-unsalted', p4 + tokens(10, 'x'), None)]
    out.append(('chain', None, chain))
    for length in (2047, 2048, 2049, 4096):
        first = tokens(length, 'b%d' % length)
        steps = [('b%d-1' % length, first, 'b'), ('b%d-2' % length, first + tokens(200, 'ba') + tokens(120, 'f'), 'b'),
                 ('b%d-3' % length, first, 'b'), ('b%d-4' % length, first, 'b')]
        out.append(('b%d' % length, None, steps))
    block = tokens(4300, 'system')
    out.append(('shared', None, [('shared-%d' % index, block + tokens(2500, 'task%d' % index), 'tenant')
                                 for index in range(3)]))
    main = tokens(3000, 'm')
    main2 = main + tokens(50, 'ma') + tokens(3000, 'm2')
    lru = [('lru-main-1', main, 'm'), ('lru-main-2', main2, 'm')]
    lru += [('lru-other-%d' % index, tokens(3000, 'o%d' % index), 'o%d' % index) for index in range(4)]
    lru.append(('lru-main-3', main2 + tokens(50, 'mb') + tokens(500, 'm3'), 'm'))
    out.append(('lru', 3, lru))
    return out


# (request id, Q, h) the real graft committed (h None: no grant line, i.e. Q = 0 and nothing planned).
GOLDEN = (
    ('chain-1', 0, 0), ('chain-2', 2048, 2048), ('chain-3', 4096, 4096), ('chain-4', 10240, 10240),
    ('chain-suffix', 10240, 10240), ('chain-early', 2048, 3264), ('chain-fresh-salt', 0, 0), ('chain-unsalted', 0, None),
    ('b2047-1', 0, None), ('b2047-2', 0, 0), ('b2047-3', 0, None), ('b2047-4', 0, None),
    ('b2048-1', 0, 0), ('b2048-2', 2048, 2048), ('b2048-3', 0, 1984), ('b2048-4', 0, 1984),
    ('b2049-1', 0, 0), ('b2049-2', 2048, 2048), ('b2049-3', 2048, 2048), ('b2049-4', 2048, 2048),
    ('b4096-1', 0, 0), ('b4096-2', 4096, 4096), ('b4096-3', 0, 4032), ('b4096-4', 2048, 4032),
    ('shared-0', 0, 0), ('shared-1', 0, 4288), ('shared-2', 4096, 4288),
    ('lru-main-1', 0, 0), ('lru-main-2', 2048, 2048), ('lru-other-0', 0, 0), ('lru-other-1', 0, 0),
    ('lru-other-2', 0, 0), ('lru-other-3', 0, 0), ('lru-main-3', 0, 4096),
)


def run_oracle():
    """-> [(request id, Q, h, plan)] from the oracle alone (no vLLM)."""
    out = []
    for _, capacity, steps in scenarios():
        oracle = judge.Oracle(capacity)
        for rid, prompt, salt in steps:
            want = oracle.admit(salt, prompt)
            out.append((rid, want['q'], want['h'], want['plan']))
    return out


def agrees(oracle_q, oracle_h, graft_q, graft_h):
    """The graft prints no grant (h unknown) when a request gets Q = 0 and plans nothing."""
    return graft_q == oracle_q and (graft_h is None or graft_h == oracle_h)


def run_graft(say=print):  # pragma: no cover - needs vLLM 0.25.1 and the TT plugin (the serving image)
    import prefix_p0a_probe as probe
    import prefix_scheduler_graft as graft

    vllm_config, evidence = probe.build_config('general-prefix')
    if vllm_config is None:
        raise RuntimeError('the general-prefix config did not build: %s' % evidence.get('error'))
    env = probe.SchedEnv(vllm_config, [])
    out = []
    for name, capacity, steps in scenarios():
        registry = graft.PrefixRegistry(budget_bytes=None if capacity is None else capacity * graft.CHECKPOINT_NBYTES)
        scheduler, state = env.make(registry=registry)
        drive = probe.Drive(env, scheduler, state, name)
        for rid, prompt, salt in steps:
            drive.add(rid, prompt, 1, salt)
            drive.run()
            row = drive.row(rid)
            out.append((rid, row.q if row.q is not None else 0, row.h))
    return out


def main(say=print):  # pragma: no cover - see run_graft
    graft_rows = run_graft(say)
    oracle_rows = dict((rid, (q, h, plan)) for rid, q, h, plan in run_oracle())
    mismatches = 0
    for rid, q, h in graft_rows:
        want_q, want_h, plan = oracle_rows[rid]
        ok = agrees(want_q, want_h, q, h)
        mismatches += 0 if ok else 1
        say('%s graft Q=%s h=%s | oracle Q=%d h=%d plan=%s %s' % (rid, q, h, want_q, want_h, plan,
                                                                  'ok' if ok else 'MISMATCH'))
    golden = [(rid, q, h) for rid, q, h in graft_rows] == list(GOLDEN)
    say('ORACLE_VS_GRAFT %d requests, %d mismatches; GOLDEN %s' % (len(graft_rows), mismatches,
                                                                     'unchanged' if golden else 'CHANGED'))
    return 1 if mismatches else 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
