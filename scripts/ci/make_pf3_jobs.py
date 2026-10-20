"""The tp4/prefill-3 window pack (scripts/ci/references/fusion-jobs/PF3), generated (stdlib only).

    python3 scripts/ci/make_pf3_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_pf3_jobs.py --check   exit 1 if the checked-in pack is not what this generates

The masked tail pass of a prompt as a trace replay (QWEN_FAST_PREFILL_TAIL_TRACE, docs/tp4-prefill3.md) on the production checkpoint: ONE image (tp4-pf3-1, a placeholder: B0 builds it from the pushed tp4/prefill-3
head WITH the K64j-OQ graft), ONE window, ONE serving profile (the approved ship-4 profile is what B0 bakes; the jobs run the gate-only twins of the ship-5 candidate, make_pf3_profiles.py), the checkpoint
production serves (ukisai/Swift-1.5-Qwen3.8-27b, selected by the job keys C2_MODEL_REPO and C2_MODEL_REVISION: qwen_c2_checkpoint.py) in every smoke job. The pack answers, in value order: do the captures fit
(H0: the hole census, the trace region, the DRAM the arena holds), is the traced tail byte-exact against the stock pass on the audited twins, on cold prompts and on returning turns (A1, A2, AP1), how small can the
arena be (L1, L2), and how many milliseconds does a tail pass save (T1-T4 on the smoke shapes, R1-R2 on the returning turns), read from the '[PF3] tail' line both arms print. Nothing here changes the approved
profile, its image or its cache; every result is UNQUALIFIED.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_pf3_profiles as twins
import make_ship4_profiles as ship4

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'PF3'
IMAGE = 'tp4-pf3-1'
REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
SERVED = 'ukisai/Swift-1.5-Qwen3.8-27b'
SHIP = ship4.SHIP                      # the approved ship-4 profile: what production serves and what B0 bakes as the window image's default
CONTROL = twins.CONTROL                # the ship-5 candidate with the gate marker and QWEN_FAST_PREFILL_TAIL_TRACE=0: the control of every timed pair
ARM = twins.ARM                        # ... with =1 and the arena at its default: the timed arm
RUNG160, RUNG80 = (twins.PREFIX + 's5-' + rung for rung, _mb in twins.LADDER)
AUDIT = twins.AUDIT                    # the ship-5 audited twin without W3 WC, buckets 128 and 1024, with the lever and its audit
AUDIT_MID = twins.AUDIT_MID            # ... buckets 256 and 512
AGENT_TURNS_TEST = 'agent_turns'
ATTACH_TESTS = 'warmup,coding,concurrent8_steady'            # (the eight-seat profiles' smoke rule requires concurrent8_steady)
AUDIT_TESTS = 'warmup,coding,parked_turns,prefill_ladder_solo,concurrent8_steady,agent_turns'
AUDIT_MID_TESTS = 'warmup,coding,parked_turns,concurrent8_steady,agent_turns'
TIMED_TESTS = 'warmup,coding,parked_turns,prefill_ladder_solo,prefill_few_decoders,concurrent8_steady,agent_turns'

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/prefill-3 window (the masked tail pass of a prompt as a trace replay; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/prefill-3 and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a NEW tag in every template (a rebuild under an existing tag keeps the old image). This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a smoke job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory; the keys are the same
# in every smoke job of this pack, so each pair differs by the profile alone. The prefix steps (AP1, R1, R2) do not forward the selection into their container and serve the base checkpoint: a tail pass
# replays the same traces on either."""

CHECKPOINT_READ = ('CHECKPOINT READ (the workflow passes c2_smoke_check.py --expect-checkpoint): the container log holds "[QWEN-C2] checkpoint %s@%s key=... weight_cache=.../weights-ckpt/<key> base_cache=... '
                   '(the base cache is not read or written)" once, a "weight cache ...: found" line (the cache the Swift window built; this pack never builds it) and the worker\'s "[QWEN-C2] checkpoint bound ..." line; '
                   'the models answer names %s.' % (REPO, REVISION[:12], SERVED))
TELEMETRY = ('TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py): the read-only ARC sidecar samples every chip once a second through the smoke step; the first [TELEMETRY] line must say decode check ok. '
             'A rate difference that comes with a different loaded AICLK between the arms is the clocks, not the lever.')
ARC = 'TIMED: the ARC runner sets are at ZERO for this job (ORDER.txt, ARC-ZERO); no build, no CI push, no other job beside it.'
MEMORY = ('MEMORY READ (the ledger is on in these profiles): "[MEMLEDGER] phase=P7 ... trace_used=.." per chip after the attach is the trace region the lever\'s traces took (against the control\'s P7 line: '
          'python3 -B scripts/ci/pf3_report.py --control <control log> --arm <this log>, or --headroom <log> for the fit of N traces), and the P7 DRAM line is what the arena holds (the arena is %s MB a chip at its default; the '
          'admission need of a prompt of 2,048 tokens or more is unchanged, so a "dram admission hold" that the control did not draw is the arena\'s)' % twins.ARENA_MB)
TAIL_LINES = ('the lines in the container log: "[PINDIAG] lever N route installed: ... tailtrace=%s" once; on the arms "[PINDIAG] prefill tail trace engaged buckets=.. traces=<n> sink=<id> taps=5,19,33,47,61 arena_mb=<mb> '
              'arena_address=.. freed=<k> uncovered=0 capture_ms=.." once and one "[PF3] tail mode=traced bucket=.. valid=.. start=.. ms=.. stage_ms=.. replay_ms=.." per traced tail pass; on the control one "[PF3] tail mode=eager '
              'bucket=.. valid=.. start=.. ms=.." per tail pass below the 2,048-row bucket and no "engaged" line; no "fell back" line, no "[PF3] tail skip" line that the control would not also take (a skip is a call that '
              'ran the stock pass: vision, an untraced bucket, another page-table width, another tap set) and no program growth after the warm anywhere')
SHAPES = ('warmup; coding (one user); parked_turns (eight seats, forty closed-loop turns of 4k to 24k tokens: forty tail passes of every bucket, beside seven decoding seats); prefill_ladder_solo (one request at a time '
          'at about 4k, 32k, 128k and 254k tokens: four tail passes); prefill_few_decoders (one decoding seat beside a 120k-token arrival and two beside a 254k one); concurrent8_steady (eight users, then the octo rounds, '
          'which the lever never touches); agent_turns (six sessions of 64k to 200k tokens of coding context, twelve returning turns each: about seventy returning-turn tail passes of every bucket at the production context sizes)')


def rows():
    out = []
    out.append(dict(name='B0-build-window', mode='stop', minutes=40, actions='build', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False, prefix=None,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/prefill-3 head (= tp4/next-1, the ship-5 stack, plus prefill_tail_trace.py and its hooks in the Lever N route): the approved ship-4 profile %s '
                         'baked as the serving default with a session cap of 8, plus C2_BAKE_GRAFT=K64j-OQ. The flags are off in the baked profile (the route does not even import the module), so the image serves exactly what '
                         'production serves until a job names a twin. READ: "baking the serving default %s with a session cap of 8", the provenance report with no problem line, its K64j-OQ graft line, the image id. A build, '
                         'not a card job; no privileged builder container may be up and no card job may run during it. Estimate from the last builds: 35-40 minutes.' % (IMAGE, SHIP, SHIP))))
    out.append(dict(name='X0-status-rescan-reset', mode='stop', minutes=20, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, needs=('X0 <- B0',), model=False, prefix=None, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host: "weight cache ...: found" is the read of every smoke job '
                         'below. Stop rule: anything else. The owner stops the node agent and unserves production BEFORE it and restores production after Z with the cutover pack\'s own steps.')))
    out.append(dict(name='H0-pf3-footprint-attach', mode='stop', minutes=40, actions='reset smoke', profile=ARM, tests=ATTACH_TESTS, telemetry=True, box=75, needs=('H0 <- X0',), model=True, prefix=None, extra=[],
                    why=('H0 (FIT, STOP): %s, the ship-5 candidate with the gate marker and QWEN_FAST_PREFILL_TAIL_TRACE=1, the arena at its default (%s MB a chip), on Swift, three short tests. This is the first card to capture a '
                         'prefill trace in the served stack, so everything that can fail before a number exists fails here, cheaply: the compile pass (the conv op engaged on every GDN layer, the gate hook on every projection), '
                         'the four captures (128, 256, 512, 1,024 rows) inside the arena, the HOLE CENSUS (every tensor the captures freed lies inside the arena: "uncovered=0" in the engaged line; a nonzero count means the allocator '
                         'had free gaps below the arena and the attach refuses by name), and the arena coming back at its address. READ: c2_smoke_check clean; %s; the memory ledger: the trace region after the attach (P7) against the '
                         'control\'s (T1 reads the control, so RECORD this one now: the traces\' bytes are the arm minus the ship-5 figure of 69.9 MB used after the attach, the estimate being 82-218 MB for four traces) and the DRAM '
                         'free after the attach (the arena\'s %s MB comes out of it). STOP RULES: an attach error naming the arena, the hole census, the conv op or the hooks; "uncovered" above 0; a trace region that does not hold '
                         'the four traces; a stall. %s RUN WITH THE CI PAUSED.' % (ARM, twins.ARENA_MB, TAIL_LINES % 'traced', twins.ARENA_MB, CHECKPOINT_READ))))
    out.append(dict(name='A1-pf3-audited-attach', mode='stop', minutes=100, actions='reset smoke', profile=AUDIT, tests=AUDIT_TESTS, telemetry=True, box=165, needs=('A1 <- H0',), model=True, prefix=None, extra=[],
                    why=('A1 (EXACTNESS of the traced tail, buckets 128 and 1,024, STOP): %s, the ship-5 audited twin without W3 WC (every audit of the candidate) with QWEN_FAST_PREFILL_TAIL_TRACE=1, the buckets 128,1024 and '
                         'QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT=1, on Swift. The first 8 traced tails, every 24th after and the first two of each bucket run the STOCK pass beside the replay (the scratch restored between them) and require '
                         'every GDN recurrent state, conv carry and conv window, every real row of the output and every drafter tap row byte equal; the stock result is what the request is served. Tests: %s. READ: c2_smoke_check clean; '
                         '"[PINDIAG] prefill tail trace audit n=<k> exact=True bucket=.. valid=.. layers=48 taps=5" for every audited tail (a bucket-128 and a bucket-1,024 line at least twice each), NO "audit mismatch" (it raises and '
                         'names the first state: "layer <i> recurrent state" is claim C1 false, "layer <i> conv window" is C2 false, "tap layer <i> rows" is the tap hand-off, "the output rows" is the attention or K/V path), NO '
                         '"fell back"; the engaged line says uncovered=0. Two buckets and not four because the audited stack leaves the trace region 114.8 MB free after the first packed round. ANY failure is NO-GO for the lever and halts '
                         'the window. %s %s %s RUN WITH THE CI PAUSED.' % (AUDIT, SHAPES, TAIL_LINES % 'traced audit', MEMORY, CHECKPOINT_READ))))
    out.append(dict(name='A2-pf3-audited-attach-mid', mode='stop', minutes=80, actions='reset smoke', profile=AUDIT_MID, tests=AUDIT_MID_TESTS, telemetry=True, box=135, needs=('A2 <- A1',), model=True, prefix=None, extra=[],
                    why=('A2 (EXACTNESS of the traced tail, buckets 256 and 512, STOP): %s, the same twin for the other two programs (bucket 256 is the stock SDPA at two q chunks a head; 512 is the one-q-per-core call). Tests: %s. '
                         'READ as A1: every audit line exact=True, a bucket-256 and a bucket-512 line at least twice each, no mismatch, no fall-back. ANY failure is NO-GO for that bucket (the BUCKETS knob drops it) and halts the '
                         'window. %s %s %s RUN WITH THE CI PAUSED.' % (AUDIT_MID, AUDIT_MID_TESTS, TAIL_LINES % 'traced audit', MEMORY, CHECKPOINT_READ))))
    out.append(dict(name='AP1-pf3-audited-returning', mode='stop', minutes=90, actions='reset prefix', profile=None, tests=None, telemetry=False, box=150, needs=('AP1 <- A2',), model=False,
                    prefix=dict(plan='tier-timed', profile=AUDIT), extra=[],
                    why=('AP1 (EXACTNESS on RETURNING turns, STOP): c2_prefix_gate.py --plan tier-timed on %s: the cold prefill, the device-resident hit and the host-tier restore at about 32k, 128k and 254k tokens, with the '
                         'tail pass of each audited (the first eight are, and the first two of each bucket it touches). The returning turns are the production shape: a hit resumes on a 2,048-token boundary (exact resume P0), '
                         'its first step is a CHECKPOINT restore and its last a masked tail at 125k and 245k of context. READ: the plan verdict PASS (the returning answers equal the cold twins\') and "[PINDIAG] prefill tail trace '
                         'audit n=<k> exact=True" for every audited tail, NO "audit mismatch", NO "fell back". ANY failure is NO-GO for the lever. This step does not forward the Swift selection: it serves the base checkpoint. '
                         'RUN WITH THE CI PAUSED.' % AUDIT)))
    for name, profile, megabytes in (('L1-pf3-arena-160', RUNG160, '160'), ('L2-pf3-arena-80', RUNG80, '80')):
        out.append(dict(name=name, mode='soft', minutes=30, actions='reset smoke', profile=profile, tests=ATTACH_TESTS, telemetry=True, box=60, needs=('%s <- H0' % name.split('-')[0],) if name.startswith('L1') else ('L2 <- L1',),
                        model=True, prefix=None, extra=[],
                        why=('%s (ARENA LADDER, %s MB; information): %s, the timed arm with the arena at %s MB. The attach is the measurement: it passes when the engaged line says uncovered=0 and the arena came back at its address, '
                             'and it fails BY NAME (the hole census) when the arena is smaller than what the captures used. The smallest rung that passes is the arena a ship profile would carry; each megabyte of it is DRAM the '
                             'eager chunk loop does not have. A failed rung is the answer, not a halt. %s %s'
                             % (name.split('-')[0], megabytes, profile, megabytes, TAIL_LINES % 'traced', CHECKPOINT_READ))))
    pair = (('T1', 'T2'), ('T3', 'T4'))
    for index, (control, arm) in enumerate(pair, 1):
        letter = 'a' if index == 1 else 'b'
        out.append(dict(name='%s-ctl-%s' % (control, letter), mode='soft', minutes=70, actions='reset smoke', profile=CONTROL, tests=TIMED_TESTS, telemetry=True, box=120,
                        needs=('T1 <- A1 A2 AP1',) if index == 1 else ('T3 <- T2',), model=True, prefix=None, extra=[],
                        why=('%s (TIMED, the CONTROL of the %s smoke pair): %s, the ship-5 candidate with the gate marker for its waiver id and QWEN_FAST_PREFILL_TAIL_TRACE=0: the eager tail, byte for byte the candidate\'s, '
                             'which also prints the "[PF3] tail mode=eager" timing line, on Swift. Tests: %s. Every boot a fresh reset. %s %s %s'
                             % (control, 'first' if index == 1 else 'second', CONTROL, SHAPES, ARC, CHECKPOINT_READ, TELEMETRY))))
        out.append(dict(name='%s-pf3-%s' % (arm, letter), mode='soft', minutes=70, actions='reset smoke', profile=ARM, tests=TIMED_TESTS, telemetry=True, box=120,
                        needs=('T2 <- T1',) if index == 1 else ('T4 <- T3',), model=True, prefix=None, extra=[],
                        why=('%s (TIMED, the ARM of the %s smoke pair): %s, the control with QWEN_FAST_PREFILL_TAIL_TRACE=1 (and the arena at its default) and nothing else, the same tests as %s, on Swift. %s READ: c2_smoke_check '
                             'clean; %s. Then, after the four boots of the pair (T1 T2 T3 T4): python3 -B scripts/ci/pf3_report.py --control <T1 container log> --control <T3 container log> --arm <T2 container log> '
                             '--arm <T4 container log>: the tail milliseconds per bucket of the eager pass against the replay (count, mean, median, p90), the saving per bucket, the route milliseconds of the final steps by context band '
                             'and the trace region the lever took; exit 0 needs the arm\'s engaged line, no fell-back or mismatch line, a traced tail and an eager-only control. The cold ladder\'s TTFTs: python3 -B '
                             'scripts/ci/prefill_ladder_report.py <T1 results> --second <T2 results> (and the same for T3 T4): a ratio below 1 is the arm faster; the answer text of every rung must be EQUAL between the arms (greedy, '
                             'same checkpoint), and so must agent_turns\' returning answers. The control-to-control floor is T1 against T3. %s %s %s'
                             % (arm, 'first' if index == 1 else 'second', ARM, control, ARC, TAIL_LINES % 'traced', MEMORY, CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='R1-ctl-returning', mode='soft', minutes=90, actions='reset prefix', profile=None, tests=None, telemetry=False, box=150, needs=('R1 <- AP1',), model=False,
                    prefix=dict(plan='tier-timed', profile=CONTROL),
                    why=('R1 (TIMED, the CONTROL of the returning-turn pair): c2_prefix_gate.py --plan tier-timed on %s: the eager tail at about 32k, 128k and 254k of context (a cold prefill, a device-resident hit, a host-tier '
                         'restore), with the "[PF3] tail mode=eager" timing line on every tail pass. %s NO CI load while it runs. This step serves the base checkpoint.' % (CONTROL, ARC)), extra=[]))
    out.append(dict(name='R2-pf3-returning', mode='soft', minutes=90, actions='reset prefix', profile=None, tests=None, telemetry=False, box=150, needs=('R2 <- R1',), model=False,
                    prefix=dict(plan='tier-timed', profile=ARM),
                    why=('R2 (TIMED, the ARM of the returning-turn pair): the same plan on %s: the traced tail at the same sizes. READ: the plan verdict PASS (the returning answers equal the cold twins\') and, with R1: '
                         'python3 -B scripts/ci/pf3_report.py --control <R1 container log> --arm <R2 container log>: the saving per tail pass at 125k and 245k of context, where a returning turn\'s step is about 1.2 s and 2.7 s long. '
                         'There is one hit of each kind per size, so the figure is read for its size (a tail pass costs a flat 320-410 ms before any attention), not for its spread. %s' % (ARM, ARC)), extra=[]))
    out.append(dict(name='Z-reset', mode='soft', minutes=10, actions='status reset', profile=None, tests=None, telemetry=False, box=None, needs=(), model=False, prefix=None, extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, '
                         'link measurement, topology republish, agent restart, deploy and the post-checks. Nothing in this window touched production\'s image, profile, alias or cache.')))
    return out


def render_env(row):
    lines = [HEADER % dict(image=IMAGE)]
    why = row['why'] if ('Duration' in row['why'] or 'Estimate' in row['why']) else row['why'] + ' Duration (estimate): %d minutes.' % row['minutes']
    lines += ['# ' + piece for piece in jobs.wrap(why)]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['profile']:
        lines.append('C2_PROFILE=%s' % row['profile'])
    if row['model']:
        lines += ['C2_MODEL_REPO=%s' % REPO, 'C2_MODEL_REVISION=%s' % REVISION]
    if row['tests']:
        lines.append('C2_SMOKE_TESTS=%s' % row['tests'])
    if row.get('prefix'):
        lines += ['C2_PREFIX_PLAN=%s' % row['prefix']['plan'], 'C2_PREFIX_PROFILE=%s' % row['prefix']['profile'], 'C2_PREFIX_BASELINE=none']
    lines += row['extra']
    if row['telemetry']:
        lines.append('C2_TELEMETRY=1')
    if row.get('box') and not any(item.startswith('C2_BOX_MINUTES=') for item in row['extra']):
        lines.append('C2_BOX_MINUTES=%d' % row['box'])
    return '\n'.join(lines) + '\n'


def render_order(table):
    total = sum(row['minutes'] for row in table)
    names = [row['name'] for row in table]
    cards = total - next(row['minutes'] for row in table if row['name'].startswith('B0'))

    def minutes(tokens):
        return sum(row['minutes'] for row in table if row['name'].split('-')[0] in tokens.split())

    blocks = (('GATES', 'B0 X0 H0 A1 A2 AP1'), ('ARENA LADDER', 'L1 L2'), ('SMOKE ABAB', 'T1 T2 T3 T4'), ('RETURNING TURNS', 'R1 R2'), ('END', 'Z'))
    lines = [
        "# The order of the tp4/prefill-3 window (docs/tp4-prefill3.md; scripts/ci/make_pf3_jobs.py): the masked tail pass of a prompt as a trace replay (QWEN_FAST_PREFILL_TAIL_TRACE) on the ship-5 candidate and its "
        "gate-only twins (make_pf3_profiles.py), the checkpoint production serves (%s, revision %s) in every smoke job, ONE image (%s, a placeholder: B0 builds it from the pushed tp4/prefill-3 head WITH the K64j-OQ graft) and ONE window."
        % (REPO, REVISION[:12], IMAGE),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window "
        "(the NEEDS lines skip what depends on it).",
        "# WHAT THIS DECIDES (the owner's, never the window's): whether the tail pass of every prompt should be a replayed trace. Today it costs a flat 320 to 410 ms below the 2,048-row bucket (~3,155 eagerly dispatched ops "
        "against 15 to 150 ms of device work: docs/tp4-prefill-2.md section 3.2), 6 percent of a returning turn at 120k and about twice that once exact resume has taken the first chunk. The expected saving is 120 to 215 ms a tail "
        "pass (ESTIMATE, docs/tp4-prefill3.md section 4); the expectation is arithmetic on the PROD fits, not a measurement. H0 is the fit gate, A1 and A2 and AP1 the exactness gates; L1 and L2 size the arena; T1-T4 and R1-R2 read "
        "the saving. Every result is UNQUALIFIED.",
        "# THE NEW DEVICE FACTS: (1) a PREFILL trace in the served stack (the first since the chunk trace was dropped: four traces of ~3,400 ops over persistent buffers, the K/V pad blocks written to NP1's scratch page, "
        "the tap outputs retained); (2) the pad rows of the GDN layers made inert through the gates (claim C1: the chunk op with valid_len = bucket leaves the same recurrent state, byte for byte, as with valid_len = the real "
        "count) and the conv state selected by a one-hot matmul pair (claim C2); (3) an arena of ballast DRAM freed for the captures and allocated again after them. The audits of A1 and A2 settle C1 and C2; H0 settles the fit.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may "
        "be alive and no privileged builder container may be up during a card job.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for "
        "its run to finish; queued tags are cancelled. Never cancel a running card job.",
        "# P0. SWIFT IS ON THE CARD HOST AND ITS CONVERTED CACHE IS BUILT: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0, and the Swift window's W0 (or the production "
        "first boot) has built the keyed weight cache, so every smoke job here logs \"weight cache ...: found\". This pack builds nothing of it." % (REPO, REVISION),
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
        "# ARC-ZERO H0 R2",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE H0 and restores them AFTER R2, "
        "i.e. around every attach (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end; the GATES block is %d h %d min and answers whether the lever is exact and fits): %s. B0 is the WINDOW build."
        % (minutes(blocks[0][1]) // 60, minutes(blocks[0][1]) % 60, '; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks)),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean where it is a smoke job, and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   H0   (STOP) the engaged line once with uncovered=0, the traces' trace-region bytes and the arena's DRAM recorded, no 'fell back', no stall. A refusal naming the arena, the hole census, the conv op or the hooks is NO-GO.",
        "#   A1 A2   (STOP) the engaged line, every audit line exact=True (the first eight tails, every 24th and the first two of each bucket), no 'audit mismatch', no 'fell back'. ANY failure is NO-GO and halts the window.",
        "#   AP1  (STOP) the tier-timed plan PASS on %s and every audited tail exact=True at 125k and 245k of context." % AUDIT,
        "#   L1 L2   the attach passes (uncovered=0) or fails by name: the smallest passing arena is the answer.",
        "#   T1 T2 T3 T4   pf3_report.py (the tail milliseconds of both arms, both pairs, and the floor T1 against T3), prefill_ladder_report.py (the rungs' TTFT ratio and equal answers). GO: a SAVING in the tail line and no LOSS in a rung.",
        "#   R1 R2   pf3_report.py on the returning-turn pair: the saving per tail pass at 125k and 245k of context; the plan verdict PASS on both.",
        "# NO-GO for the traced tail: any stall, any audit mismatch, any fallback line, a text difference against the control, a program compiled after the warm, an engaged line missing on an arm, a nonzero hole census, a LOSS in the tail line. "
        "Adopting is the owner's decision: the flag is a gate instrument today and a ship profile needs an owner traffic waiver naming it (docs/tp4-prefill3.md).",
        "# Time (estimate, minutes): %s: %d minutes (%d h %d min) in all, %d minutes of cards (X0 to Z)." % (', '.join('%s %d' % (name.split('-')[0], row['minutes']) for name, row in zip(names, table)), total, total // 60, total % 60, cards),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in table]
    return '\n'.join(lines) + '\n'


def generate():
    table = rows()
    out = dict(('%s.env' % row['name'], render_env(row)) for row in table)
    out['ORDER.txt'] = render_order(table)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args(argv)
    if not (arguments.write or arguments.check):
        parser.error('--write or --check')
    wanted = generate()
    found = dict((path.name, path.read_text(encoding='utf-8')) for path in FOLDER.glob('*') if path.is_file()) if FOLDER.is_dir() else {}
    stale = sorted(name for name in set(wanted) | set(found) if wanted.get(name) != found.get(name))
    if arguments.check:
        for name in stale:
            sys.stderr.write('%s/%s is not what make_pf3_jobs.py generates: run it with --write\n' % (FOLDER, name))
        return 1 if stale else 0
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name in stale:
        if name in wanted:
            with open(str(FOLDER / name), 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(wanted[name])
        else:
            (FOLDER / name).unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main())
