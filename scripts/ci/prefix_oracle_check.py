"""The prefix gates' oracle (prefix_judge.Oracle) against the REAL scheduler graft, request by request.

The hardware gates fail a sequential arm when a request's Q differs from the oracle's, so the oracle
must be the scheduler graft's rules and nothing else. This drives both with the same requests:

  - the graft: vLLM 0.25.1's real TTScheduler and KVCacheManager with G1's scheduler graft
    (qwen_prefix_scheduler_patch over qwen_prefix_registry) installed, built from the general-prefix
    engine config the way the P0a probe builds it (prefix_p0a_probe.SchedEnv / Drive: ttnn and the TT
    model stubbed, no device). Each registry has mid-loop captures enabled, as the served model graft
    declares at warmup (qwen_prefix_model_patch), so the gap boundary is planned;
  - the oracle: prefix_judge.Oracle fed the same prompts and salts in the same order.

SCENARIOS covers the design's worked cases: a chain whose turns extend the previous prompt, a changed
suffix, an early divergence falling back to an older checkpoint, a fresh salt, an unsalted request,
previous prompts of 2047/2048/2049/4096 tokens with re-sends (the num_tokens - 1 cap), three
conversations of one tenant sharing a 4,300-token block (the gap capture), and a three-entry store
(LRU eviction). Prompts are fixed token lists, so decode output never enters them.

    python3 scripts/ci/prefix_oracle_check.py          # inside the serving image, as the probe runs

prints one line per request and 'ORACLE_VS_GRAFT <n> requests, <m> mismatches'; exits 1 on any
mismatch, and on a GOLDEN change: GOLDEN is what the real graft committed (first recorded on
2026-09-26 with the P0a prototype at prefix/g1-harness; re-run on the integrated G1 graft, branch
prefix/g1, vLLM 0.25.1 source, plugin bf77cd63: unchanged), which test_prefix_oracle_check holds the
oracle to on CPU - a graft that now commits something else has to be re-recorded, not waved through.

WHICH graft: the step runs this from the mounted checkout (/c2/scripts/ci), so by default it drives
the checkout's GRAFT_FILES. Once the image carries its own copies (IMAGE_GRAFT: the plugin package the
patched TTScheduler imports them from, where docker/qwen-c2-overlay.txt lays them), those are the ones
driven, and each must be byte-identical to the checkout's: otherwise the oracle is held to a graft the
engine does not run.
"""

import hashlib
import importlib.util
import os
import random
import sys

import prefix_judge as judge

CHUNK = judge.CHUNK
HERE = os.path.dirname(os.path.abspath(__file__))
# The package the patched TTScheduler.__init__ imports the graft from (a relative import), and the two
# files it needs, the registry first: the graft imports it by its plain name when loaded outside a package.
IMAGE_GRAFT = '/opt/qwen-fast-plugin/src/vllm_tt_plugin'
GRAFT_FILES = ('qwen_prefix_registry.py', 'qwen_prefix_scheduler_patch.py')


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


def file_sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def graft_file(directory, name):
    return directory.rstrip('/\\') + '/' + name


def choose_graft(checkout=None, image=IMAGE_GRAFT, exists=os.path.exists, sha=file_sha):
    """-> (directory whose GRAFT_FILES to drive, problem or None). The image's copies when it has them
    (each must equal the checkout's byte for byte), else the checkout's, said so."""
    checkout = checkout or HERE
    present = [name for name in GRAFT_FILES if exists(graft_file(image, name))]
    if not present:
        return checkout, None
    if len(present) != len(GRAFT_FILES):
        return image, ('the image carries %s but not %s: the patched scheduler could not install the graft' % (
            ', '.join(graft_file(image, name) for name in present),
            ', '.join(name for name in GRAFT_FILES if name not in present)))
    for name in GRAFT_FILES:
        mine, theirs = sha(graft_file(checkout, name)), sha(graft_file(image, name))
        if mine != theirs:
            return image, ('the image serves %s (sha256 %s) but the checkout has %s %s: the oracle would be held '
                           'to a graft the engine does not run' % (graft_file(image, name), theirs, name, mine))
    return image, None


def load_graft(directory):
    """The registry and the scheduler graft from `directory`, installed as sys.modules[<their names>];
    returns the graft (qwen_prefix_scheduler_patch)."""
    module = None
    for name in GRAFT_FILES:
        stem = name[:-len('.py')]
        spec = importlib.util.spec_from_file_location(stem, graft_file(directory, name))
        module = importlib.util.module_from_spec(spec)
        sys.modules[stem] = module
        spec.loader.exec_module(module)
    return module


def run_graft(say=print, graft_path=None):  # pragma: no cover - needs vLLM 0.25.1 and the TT plugin (the serving image)
    import prefix_p0a_probe as probe

    graft = load_graft(graft_path) if graft_path else probe.graft
    probe.graft = graft
    vllm_config, evidence = probe.build_config('general-prefix')
    if vllm_config is None:
        raise RuntimeError('the general-prefix config did not build: %s' % evidence.get('error'))
    env = probe.SchedEnv(vllm_config, [])
    out = []
    for name, capacity, steps in scenarios():
        registry = graft.PrefixRegistry(budget_bytes=None if capacity is None
                                        else capacity * graft.prefix_registry.CHECKPOINT_NBYTES)
        # The served model graft declares mid-loop captures at warmup, before the scheduler exists.
        registry.enable_mid_loop_capture()
        scheduler, state = env.make(registry=registry)
        drive = probe.Drive(env, scheduler, state, name)
        for rid, prompt, salt in steps:
            drive.add(rid, prompt, 1, salt)
            drive.run()
            row = drive.row(rid)
            out.append((rid, row.q if row.q is not None else 0, row.h))
    return out


def verdict(graft_rows, say=print):
    """Graft rows against the oracle and GOLDEN. -> exit code (1 on a mismatch or a GOLDEN change)."""
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
    if not golden:
        say('FAIL: the graft no longer commits GOLDEN: re-record it (and test_prefix_oracle_check) deliberately')
    return 1 if mismatches or not golden else 0


def main(say=print):  # pragma: no cover - see run_graft
    path, problem = choose_graft()
    for name in GRAFT_FILES:
        if os.path.exists(graft_file(path, name)):
            say('graft driven: %s (sha256 %s)' % (graft_file(path, name), file_sha(graft_file(path, name))))
    if problem:
        say('FAIL: ' + problem)
        return 1
    if path != IMAGE_GRAFT:
        say('NOTE: the image carries no graft in %s: the checkout copy of the graft was driven' % IMAGE_GRAFT)
    return verdict(run_graft(say, path), say)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
