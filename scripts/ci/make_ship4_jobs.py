"""The tp4/ship-4 window pack (scripts/ci/references/fusion-jobs/ship4), generated (stdlib only).

    python3 scripts/ci/make_ship4_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_ship4_jobs.py --check   exit 1 if the checked-in pack is not what this generates

ONE image (tp4-ship-4, a placeholder: B0 builds it from the pushed tp4/ship-4 head WITH the K64j-OQ graft), ONE window, ONE candidate: the combined ship profile `-fx-traffic-s4` (the approved fusion
ship profile + the prefill SDPA with one q chunk per core + the 13x10 matmul grids + octo-T8 with its three levers + the tile-aligned drafter window + the host KV tier with the preconverted
checkpoints, scripts/ci/fusion-wp/S4.json) under ONE owner traffic waiver whose decision is PENDING. The pack is that candidate's gates (the audited attach, the exactness pair, the eight-seat 262k gate,
the prefix hit, the platform replay), the decode ABAB against the approved fusion ship profile, the host tier beside the octo block (the audited and the timed concurrent-restore plans, the quota gate) and
the switchable control of the one line that may still be a loss (the 13x10 grids). The gate-only twins are make_ship4_profiles.py's; the candidate's own gate rows (G1 GX0 GX1 G2 SR and the ship build)
are make_fusion_jobs.ship_rows, so they cannot drift from the single-candidate packs.
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_fusion_profiles as fusion
import make_octo3_jobs as octo3_jobs
import make_octo3_profiles as octo3
import make_ship4_profiles as twins

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'ship4'
IMAGE = 'tp4-ship-4'
SPEC_NAME = twins.SHIP_NAME

SHIP = twins.SHIP
AUDIT = twins.AUDIT
NOGRID13 = twins.NOGRID13
CONTROL = fusion.NAMESPACE + 'traffic'                        # the approved fusion ship profile: the control of every pair, and the exactness control
PRODUCTION = fusion.PARENT

DEEP = octo3_jobs.DEEP
TIMED_TESTS = octo3_jobs.TIMED_TESTS
AUDIT_TESTS = 'warmup,coding,parked_turns,prefill_ladder_solo,concurrent8_steady'
GRID_TESTS = 'warmup,concurrent8_steady,concurrent8_code_32k_long'

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/ship-4 window (the combined ship candidate; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/ship-4 and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the
# order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent."""

ARC = 'TIMED: the ARC runner sets are at ZERO for this job (ORDER.txt, ARC-ZERO); no build, no CI push, no other job beside it.'
TELEMETRY = ('TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py): the read-only ARC sidecar samples every chip once a second through the smoke step; read telemetry-smoke/telemetry-summary.txt beside '
             'the result, and the first [TELEMETRY] line must say decode check ok. A gain that comes with a different loaded AICLK between the arms is the clocks, not the lever.')
OCTO_READ = octo3_jobs.OCTO_READ
TIER_LINES = ("the tier's attach lines in the container log ('host tier IO attached', 'host tier on gib=32.00' with verify=sample audit=0 on the candidate and verify=all audit=1 on the audited twin) and no "
              "'latched off' line")
PENDING = ('The candidate is a TRAFFIC profile under an owner traffic waiver whose decision is PENDING: it boots in a gate or a smoke and logs the word (c2_serving_job.read_bake refuses a PENDING profile '
           'only at the BUILD).')
SIX = ('ONE-LINE SUMMARY OF WHAT IS IN THE CANDIDATE: the approved fusion ship profile (W2, engine reuse, the op-fusion set, Lever N, prefix reuse) + QWEN_FAST_SDPA_PF_ONEQ=1 + QWEN_FAST_GRID13=1 + octo-T8 '
       '(QWEN_FAST_OCTO=live, the octo draft, the attention bundle, the glue levers) + QWEN_FAST_TAP_WINDOW_ALIGNED=1 + the host KV tier (32 GiB, verify=sample) with the preconverted checkpoints + the third '
       'block\'s memory plan (KV pool 16,500 blocks, trace region 640 MiB).')


def plan():
    return fusion.normalise(fusion.read_manifests())


def spec(plan_):
    return next(item for item in plan_['ship'] if item['name'] == SPEC_NAME)


def gate_rows(plan_):
    """The candidate's gate rows from the one definition (make_fusion_jobs.ship_rows): {name: row}."""
    rows = {row['name']: row for row in jobs.ship_rows(plan_, jobs.shipped_spec(plan_, spec(plan_)))}
    return rows


def rows(plan_):
    gates = gate_rows(plan_)
    deep_note = ('The two _long tests are the DEEP tests: eight real-code prompts of about 32k / 120k tokens with budgets (3,000 / 4,000, ignore_eos) that outlast the prefill queue, so that the last '
                 'user arrives while the first is still decoding and eight seats decode together at depth; the 800-token concurrent8_code_32k / _128k (run beside them for continuity) never have six '
                 'seats live at once, so no octo round runs in them.')
    out = []
    out.append(dict(name='B0-build-window', mode='stop', minutes=40, actions='build', profile='c2-packed-tp4', tests=None, telemetry=False, box=None, needs=(),
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % PRODUCTION, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/ship-4 head: the production profile %s baked as the serving default with a session cap of 8, plus C2_BAKE_GRAFT=K64j-OQ '
                         '(the prefill SDPA with one q chunk per core: the image\'s _ttnncpp.so is K64j plus the prefill factory\'s oneq edits, inert unless QWEN_FAST_SDPA_PF_ONEQ=1), the octo modules, '
                         'audit_sampling, every fusion package\'s files (the overlay manifest) and the host tier (version 2 of the qwen_kv_read extension: the raw block read and write, built on the '
                         'prefix-tiers branch and pinned in the Dockerfile). THE IMAGE IS A GATE-WINDOW IMAGE ONLY: the candidate\'s decision is PENDING, and a PENDING profile is never baked '
                         '(B1 builds the ship image from the commit that records the approval). READ: "baking the serving default %s with a session cap of 8", the provenance report with no problem line, '
                         'its K64j-OQ graft line and its "[KVREAD] qwen_kv_read 2" line, the closure checks, the image id. A build, not a card job; no privileged builder container may be up and no '
                         'card job may run during it. Estimate from the last builds: 35-40 minutes.' % (IMAGE, PRODUCTION, PRODUCTION))))
    out.append(dict(name='X0-status-rescan-reset', mode='stop', minutes=20, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, needs=('X0 <- B0',), extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. Stop rule: anything else. The owner stops the node agent and unserves production BEFORE it and '
                         'restores production after Z with the cutover pack\'s own steps.')))
    out.append(dict(name='A1-s4-audited-attach', mode='stop', minutes=110, actions='reset smoke', profile=AUDIT, tests=AUDIT_TESTS, telemetry=True, box=150, needs=('A1 <- X0',), extra=[],
                    why=('A1 (EXACTNESS of the whole candidate, STOP): %s = the candidate with EVERY audit on: the combined fusion arm with each lever\'s audit, the prefill SDPA with one q chunk per core and the 13x10 grids '
                         'with theirs, octo-T8 with the bundle, glue, vglue, packed-window and F1 audits (the host read-back audits SAMPLED: QWEN_FAST_AUDIT_FIRST=1, QWEN_FAST_AUDIT_EVERY=16), the tile-aligned '
                         'drafter window with its audit, the third block\'s memory plan, and the host KV tier with every instrument on (the multi launch\'s own SDPA audit beside W2). warmup, coding, parked_turns '
                         '(the 40 agent turns: returning sessions rebind onto parked engines), the solo prefill ladder (the prefill levers in every chunk) and concurrent8_steady (the octo rounds). READ: '
                         'c2_smoke_check clean under every rule at once, i.e. %s %s; the audits: "tp4 vglue audit <n> exact=True ... sample=first:1,every:16" lines with entries above 0, "[OCTO-ATTN-BUNDLE] '
                         'audit <n> exact=True", the T2 window and F1 spread audit lines, the oneq and grid13 audit lines (grid13_smoke), the K/V page writer\'s "octo 8-row served path" line once, the memory '
                         'ledger: octo_draft_built and the free DRAM after the attach per chip (RECORD the figure: the third block, the draft capture AND the 32 GiB host tier beside the engines are the plan\'s '
                         'named risks). NO "audit mismatch", no "fell back", no stall. ANY failure is NO-GO for the candidate and halts the window. The restore\'s own audits are TC1\'s (the smoke forces no eviction). %s'
                         % (AUDIT, OCTO_READ % AUDIT, TIER_LINES, TELEMETRY))))
    out.append(dict(gates['GX0-exact-control'], needs=('GX0 <- A1',)))
    out.append(dict(gates['GX1-exact-ship'], needs=('GX1 <- GX0',)))
    out.append(dict(gates['G1-ship-gate'], needs=('G1 <- GX1',)))
    out.append(dict(name='TC1-tier-concurrent-audited', mode='stop', minutes=90, actions='reset prefix', profile=None, tests=None, telemetry=False, box=150, needs=('TC1 <- G1',),
                    extra=['C2_PREFIX_PLAN=tier-concurrent', 'C2_PREFIX_PROFILE=%s' % AUDIT, 'C2_PREFIX_BASELINE=none'],
                    why=('TC1 (STOP): the AUDITED CONCURRENT TIER arm (c2_prefix_gate.py --plan tier-concurrent, scenario tier_concurrent) on the candidate\'s audited twin %s: a session built to the largest size the '
                         'profile admits, fresh-salt floods the size of the KV pool (16,500 blocks here, not the 19,968 the tier pack ran on) so that it leaves the device pool for host RAM, SEVEN SEATS decoding '
                         '(about 4k-token coding prompts, ignore_eos), a 10 s settle and a 60 s CONTROL window of the seats alone, then the session RETURNS (restored from host RAM; it decodes 512 tokens as an '
                         'eighth seat), a long (120k) fresh prompt (SPILL) and the returning turn\'s COLD TWIN (a fresh salt, the same body; strict pair). THE NEW THING HERE: the seven seats and the returning '
                         'eighth run on the OCTO block (QWEN_FAST_OCTO_MIN_LIVE=6), so the tier\'s restore and spill happen inside a scheduler step while octo rounds are the steady state. READ (the tier pack\'s TC1, '
                         'references/fusion-jobs/tier): the plan verdict PASS, every pair IDENTICAL, a "host tier restore" line for the returning turn at the oracle\'s Q, "host tier spill" lines inside the long '
                         'prompt\'s request, the failure counters (spill, restore, digest, read-back mismatch) ZERO, no pins held, every one of the seven seats decoding across the control, restore, restore_decode '
                         'and spill windows (else NOT_EXERCISED), and the lines "window <name>: ... round time p50/p99/mean ..., longest wait ... s". Also: "[OCTO]" round lines with shape=octo during the control '
                         'window (the seats ARE on the octo block) and none of the octo judge\'s hard problems (python3 scripts/ci/octo_judge.py --container-log <container log> --profile %s; its eight-live '
                         'lever rules do not apply at seven seats: read what it reports about shape and admission). The stall is PRINTED here, not judged (the audits are this arm\'s cost); the budget is judged '
                         'on TC2 to TC5. Unqualified: a measurement, never evidence.' % (AUDIT, AUDIT))))
    for index, (control_name, arm_name) in enumerate((('C1', 'L1'), ('C2', 'L2')), 1):
        out.append(dict(name='%s-control-%s' % (control_name, 'a' if index == 1 else 'b'), mode='soft', minutes=65, actions='reset smoke', profile=CONTROL, tests=TIMED_TESTS, telemetry=True, box=100,
                        needs=('C1 <- TC1',) if index == 1 else ('C2 <- L1',), extra=[],
                        why=('%s (TIMED, the CONTROL of the %s decode ABAB pair): %s, the approved fusion ship profile as production would run it after the fusion cutover (two M3 blocks, W2, engine reuse, every fusion '
                             'lever, no octo, no oneq, no grid13, no tier), eight live (concurrent8_steady), eight at 32k and 120k with 800-token answers (concurrent8_code_32k / _128k) and eight at depth for the whole '
                             'all-live window (%s). Every boot a fresh reset. %s %s READ: c2_smoke_check clean, no stall; this log is the reference of %s (octo_compare.py --control <this smoke log> --octo <%s smoke log>: '
                             'every answer equal, user by user) and of the floor (C1 against C2). %s'
                             % (control_name, 'first' if index == 1 else 'second', CONTROL, DEEP.replace(',', ', '), ARC, deep_note, arm_name, arm_name, TELEMETRY))))
        out.append(dict(name='%s-s4-%s' % (arm_name, 'a' if index == 1 else 'b'), mode='soft', minutes=65, actions='reset smoke', profile=SHIP, tests=TIMED_TESTS, telemetry=True, box=100,
                        needs=('L1 <- C1',) if index == 1 else ('L2 <- C2',), extra=[],
                        why=('%s (TIMED, the CANDIDATE of the %s decode ABAB pair): %s, the whole candidate as a traffic profile (unaudited): %s The same tests as %s. %s %s %s READ: c2_smoke_check clean AND %s Then the '
                             'pair: python3 scripts/ci/octo3_report.py --control-a <C1 smoke> --arm-a <L1 smoke> --control-b <C2 smoke> --arm-b <L2 smoke> --arm-container <L1 container> --arm-container <L2 container> '
                             '--profile %s: per test (concurrent8_steady, %s) the aggregate of the all-live window against the control\'s in BOTH pairs, the floors, GAIN / PARITY / LOSS; SHIP-GO needs no LOSS and a '
                             'GAIN in the steady window; octo_compare.py for the texts (EVERY answer equal; any difference is NO-GO). The prefill levers act on prefill, so the 4k, 32k and 120k first tokens '
                             '(prefill_ladder_solo is A1\'s) are read from the same logs\' ttft lines. %s'
                             % (arm_name, 'first' if index == 1 else 'second', SHIP, SIX, control_name, ARC, PENDING, deep_note, OCTO_READ % SHIP, SHIP, DEEP.replace(',', ', '), TELEMETRY))))
    for index, name in ((1, 'TC2-tier-concurrent-control-timed'), (2, 'TC3-tier-concurrent-s4-timed'), (3, 'TC4-tier-concurrent-control-repeat'), (4, 'TC5-tier-concurrent-s4-repeat')):
        control = index in (1, 3)
        out.append(dict(name=name, mode='soft', minutes=50, actions='reset prefix', profile=None, tests=None, telemetry=False, box=90,
                        needs=('TC2 <- G1',) if index == 1 else (('TC3 <- TC1 TC2',) if index == 2 else (('TC4 <- TC3',) if index == 3 else ('TC5 <- TC4',))),
                        extra=['C2_PREFIX_PLAN=tier-concurrent-timed', 'C2_PREFIX_PROFILE=%s' % (CONTROL if control else SHIP), 'C2_PREFIX_BASELINE=none'],
                        why=('TC%d (TIMED, the ABAB\'s %s): the %s of the concurrent tier plan (c2_prefix_gate.py --plan tier-concurrent-timed, scenario tier_concurrent_timed: no row digests, no extent audit, the tier\'s own audits off), '
                             'profile %s. %s THE ABAB: TC2 control, TC3 candidate, TC4 control, TC5 candidate, every boot a fresh reset. %s READ each: the plan verdict PASS (the stall budget is JUDGED here: no seat waited longer '
                             'than the Lever N decode gap floor, 8 s, between two tokens in any window), then the four together: python3 scripts/ci/tier_concurrent_compare.py <TC2 results> <TC3 results> <TC4 results> <TC5 results> '
                             '(A control, B candidate, C control repeat, D candidate repeat): exit 0 needs every seat IDENTICAL over its common tokens (the seven streams unchanged by the candidate), the returning, long-prompt '
                             'and cold-twin answers IDENTICAL across the profiles, and no longest wait over the budget; it prints the round p50/p99 of each arm window by window, the CANDIDATE arm\'s p99 against the '
                             'CONTROL arm\'s (percent) and the control-to-control FLOOR. THE CONTROL IS THE APPROVED FUSION SHIP PROFILE, which has no tier, no octo and the 19,968-block pool: the returning session is '
                             're-prefilled cold beside the seven seats there. The candidate\'s p99 therefore mixes the tier\'s restore stall with the octo rounds\' speed-up; the tier\'s own cost against its own control is '
                             'the tier pack\'s (references/fusion-jobs/tier). THE LEVER-FLAVOURED READ: in the candidate arm the lines "host tier restore" and "host tier spill" must be in the log (else NOT_EXERCISED) and the '
                             '[OCTO] round lines must show shape=octo with live 7-8 during the control window. Estimate: 50 minutes. %s'
                             % (index + 1, 'A' if index == 1 else ('B' if index == 2 else ('C' if index == 3 else 'D')), 'CONTROL' if control else 'CANDIDATE', CONTROL if control else SHIP,
                                'The control profile has no owner waiver other than the approved one; it needs no PENDING word.' if control else PENDING, ARC, TELEMETRY.split(' (')[0] + ': the prefix step has no sidecar.'))))
    for index, (arm_name, pair_name) in enumerate((('N1', 'L1'), ('N2', 'L2')), 1):
        out.append(dict(name='%s-s4-nogrid13-%s' % (arm_name, 'a' if index == 1 else 'b'), mode='soft', minutes=40, actions='reset smoke', profile=NOGRID13, tests=GRID_TESTS, telemetry=True, box=70,
                        needs=('N1 <- L1',) if index == 1 else ('N2 <- L2 N1',), extra=[],
                        why=('%s (TIMED, the SWITCHABLE LINE of the candidate): %s = the candidate WITHOUT QWEN_FAST_GRID13 (the 13x10 matmul grids), as a gate-only profile. Against %s it answers whether the grids pay inside the '
                             'whole stack: they are default-off, exact by argument and by audit (A1 audits them) and their own decode ABAB (the M1 programme\'s F2 window) decides them standalone. %s READ: c2_smoke_check '
                             'clean AND octo_judge clean on %s; then the pair: python3 scripts/ci/octo3_report.py --control-a <N1 smoke> --arm-a <L1 smoke> --control-b <N2 smoke> --arm-b <L2 smoke> '
                             '--arm-container <L1 container> --arm-container <L2 container> --profile %s (here the CONTROL is the no-grid boot and the arm the candidate: a GAIN means the grids win; the report prints UNREAD for concurrent8_code_128k_long, which the no-grid boots do not run). DECISION: a LOSS or '
                             'a PARITY on both pairs means delete the QWEN_FAST_GRID13 line from the ship entry of scripts/ci/fusion-wp/S4.json, regenerate (make_fusion_profiles.py, make_ship4_profiles.py, '
                             'make_ship4_jobs.py) and re-run only the gates that read the changed profile (A1 and G1 at least). %s'
                             % (arm_name, NOGRID13, pair_name, ARC, NOGRID13, SHIP, TELEMETRY))))
    out.append(dict(gates['G2-prefix-hit'], needs=('G2 <- G1',)))
    out.append(dict(name='TQ1-tier-quota-isolation', mode='soft', minutes=120, actions='reset prefix', profile=None, tests=None, telemetry=False, box=240, needs=('TQ1 <- TC1',),
                    extra=['C2_PREFIX_PLAN=tier-quota', 'C2_PREFIX_PROFILE=%s' % AUDIT, 'C2_PREFIX_BASELINE=none'],
                    why=('TQ1 (SOFT, GATE ONLY): the per-tenant quota gate (c2_prefix_gate.py --plan tier-quota; docs/prefix-store-hygiene.md "Tenant quotas") on the audited twin %s. THE QUOTAS ARE NOT IN THE CANDIDATE '
                         '(the contract refuses QWEN_PREFIX_TENANT_QUOTA_* on a traffic profile until a qualification record exists); this job keeps the gate-only evidence current on the combined stack, so that '
                         'a later quota decision does not need another window. Two arms, two boots, the same scenario as the tier pack\'s TQ1 (references/fusion-jobs/tier/TQ1-tier-quota-isolation.env has the full '
                         'read: both arms PASS, tenant B\'s return a device-resident hit on quota-isolated and not on quota-control, the quota\'s own evictions on the books, quota_dev_floor_overrides 0, the '
                         'failure counters zero, all four pairs IDENTICAL, then python3 scripts/ci/tier_concurrent_compare.py --quota <results>/quota-control <results>/quota-isolated). The flood is the 16,500-block '
                         'pool. Lowest value of the window; cut it first.' % AUDIT)))
    out.append(dict(name='Z-reset', mode='soft', minutes=10, actions='status reset', profile=None, tests=None, telemetry=False, box=None, needs=(), extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, '
                         'link measurement, topology republish, agent restart, deploy and the post-checks.')))
    ship_build = dict(gates['B0-ship-build'], name='B1-ship-build', needs=('B1 <- G1 GX1',), mode='soft')
    ship_build['why'] = ship_build['why'].replace('B0 (STOP): build the SHIP image', 'B1 (AFTER THE OWNER APPROVES; refused until then, by design): build the SHIP image', 1)
    out.append(ship_build)
    replay = dict(gates['SR-platform-replay'], needs=('SR <- B1 G1 GX1',), mode='soft')
    replay['why'] = replay['why'].replace('SR (STOP)', 'SR (AFTER THE OWNER APPROVES, on the image B1 builds)', 1)
    out.append(replay)
    return out


def render_env(row):
    lines = [HEADER % dict(image=IMAGE)]
    why = row['why'] if ('Duration' in row['why'] or 'Estimate' in row['why']) else row['why'] + ' Duration (estimate): %d minutes.' % row['minutes']
    lines += ['# ' + piece for piece in jobs.wrap(why)]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['profile']:
        lines.append('C2_PROFILE=%s' % row['profile'])
    if row.get('bake'):
        lines.append('C2_BAKE_DEFAULT_PROFILE=%s' % row['bake'])
    if row['tests']:
        lines.append('C2_SMOKE_TESTS=%s' % row['tests'])
    lines += row['extra']
    if row['telemetry']:
        lines.append('C2_TELEMETRY=1')
    if row.get('box') and not any(item.startswith('C2_BOX_MINUTES=') for item in row['extra']):
        lines.append('C2_BOX_MINUTES=%d' % row['box'])
    return '\n'.join(lines) + '\n'


def render_order(table, plan_):
    total = sum(row['minutes'] for row in table)
    names = [row['name'] for row in table]
    cards = total - next(row['minutes'] for row in table if row['name'].startswith('B0'))
    cards -= sum(row['minutes'] for row in table if row['name'].startswith(('B1', 'SR')))
    first, last = 'A1', 'TQ1'
    waiver = spec(plan_)['waiver']
    blocks = (('GATES', 'B0 X0 A1 GX0 GX1 G1 TC1'), ('DECODE ABAB', 'C1 L1 C2 L2'), ('TIER BESIDE OCTO, TIMED', 'TC2 TC3 TC4 TC5'), ('EXTRAS', 'N1 N2 G2 TQ1'), ('AFTER APPROVAL', 'Z B1 SR'))

    def minutes(tokens):
        return sum(row['minutes'] for row in table if row['name'].split('-')[0] in tokens.split())

    lines = [
        "# The order of the tp4/ship-4 four-card window (docs/tp4-ship4.md; scripts/ci/fusion-wp/S4.json, scripts/ci/make_ship4_profiles.py, scripts/ci/make_ship4_jobs.py): the COMBINED SHIP CANDIDATE %s, ONE image "
        "(%s, a placeholder: B0 builds it from the pushed tp4/ship-4 head WITH the K64j-OQ graft) and ONE window." % (SHIP, IMAGE),
        "# THE CANDIDATE: %s It is a TRAFFIC profile under ONE owner traffic waiver (id %s), decision PENDING: it boots in a gate or a smoke and logs the word, no build bakes it until the owner replaces the "
        "decision sentence with one starting APPROVED (python3 scripts/ci/make_fusion_profiles.py --approve-ship \"APPROVED ...\" --ship %s: it edits the ship entry of scripts/ci/fusion-wp/S4.json and nothing else), "
        "the CPU suite is green on that commit and B1 builds the ship image from it." % (SIX, waiver['id'], SPEC_NAME),
        "# One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop "
        "the window (the NEEDS lines skip what depended on it).",
        "# NOTHING IN THE CANDIDATE HAS RUN ON A CARD TOGETHER. What has run, on its own: octo-T8 and its levers on the Lever N stack (alternating, paired per round, 8 live: the draft +6.1%, the glue levers +28.6%, the "
        "bundle +2.5%, all three +46.6%, text identical 24 of 24); the host tier alone (a session restored in 3.8 s, 4.1 s and 6.5 s at 32k, 128k and 254k against 8.3 s, 37.6 s and 96.0 s cold, exact); the "
        "prefill levers alone (oneq byte-exact on card M; grid13 by argument and audit). Every number this pack produces for the combination is the first.",
        "# THE ARMS (gate-only twins, generated by make_ship4_profiles.py): %s (the candidate with every audit on and the tier's instruments, sampled) and %s (the candidate without the 13x10 grids). The control of every "
        "pair is the approved fusion ship profile %s on the same image." % (AUDIT, NOGRID13, CONTROL),
        "# THE ONE SWITCHABLE LINE: QWEN_FAST_GRID13=1. Its own decode ABAB (the M1 programme's F2 window) had not reported when this was written; N1 N2 answer it inside the whole stack. A loss is answered by deleting "
        "the line from the ship entry of S4.json and regenerating; nothing else in the candidate depends on it.",
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No "
        "other window driver may be alive and no privileged builder container up.",
        "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only "
        ".github/c2-serving-job.env). Push ONE tag at a time and wait for its run to finish; queued tags are cancelled. Never cancel a running card job. Every result is UNQUALIFIED.",
        "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
        "# ARC-ZERO %s %s" % (first, last),
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE %s and "
        "restores them AFTER %s, i.e. around every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0 and X0 do not need the pause; Z, B1 and SR do not." % (first, last),
        "# THE JOBS, in value order (cut the window at a block end): %s. B0 is the WINDOW build (rc 0 now); B1 builds the SHIP image and is REFUSED while the decision is PENDING (c2_serving_job.read_bake), by design: it is the "
        "post-approval build, followed by SR." % '; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for row in table:
        for need in row['needs']:
            lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean, whose fusion and octo rules read the container log, and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line (its K64j-OQ graft line and its qwen_kv_read 2 line), the image id." % PRODUCTION,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive.",
        "#   A1   (STOP) every lever's engaged line and an audit line with exact=True under the fusion rules, the octo rules and the sampling; the tier's attach lines; no fell-back line, no 'audit mismatch', the memory "
        "ledger's free DRAM per chip recorded. ANY failure is NO-GO for the candidate and halts the window.",
        "#   GX0 GX1   parked_compare.py (control, candidate): every content hash EQUAL, compared above 0, the parked judge with 0 problems; the octo, oneq and tier lines of the candidate's container log. Any difference is NO-GO.",
        "#   G1   every smoke rule clean, skew under 238 s, 0 crashes, parked judge 0 problems, the OCTO READ lines (octo_judge clean on the container log: the octo rounds ran, no program compiled on the first one).",
        "#   TC1  the tier-concurrent plan PASS (audited), every pair IDENTICAL, the restore and the spill on record, zero failure counters, all seven seats across every judged window, octo rounds in the control window.",
        "#   C1 L1 C2 L2   octo3_report.py (the all-live window of each test, both pairs, the floors), octo_compare.py (every answer equal) and octo_judge.py on both candidate logs. GO for the decode: no LOSS in any test, a GAIN in "
        "concurrent8_steady, every text equal.",
        "#   TC2 TC3 TC4 TC5   the tier-concurrent-timed plan PASS each (the stall budget judged: no wait over 8 s), then tier_concurrent_compare.py over the four (the texts IDENTICAL, the p99 side by side, the floor).",
        "#   N1 N2   octo3_report.py with the no-grid boots as the control; the decision is the QWEN_FAST_GRID13 line.   G2   the plan PASS (read a failure against its control before calling it a regression).   TQ1   the tier pack's read.",
        "#   B1   (after the approval) 'baking the serving default %s with a session cap of 8', the provenance report, the image id.   SR   'PLATFORM_REPLAY passed=True'." % SHIP,
        "# NO-GO for the candidate: any stall, any text difference against the control, any audit mismatch, any fallback line, an arm that logs no engaged line, a LOSS in a timed decode test, a seat that waited over the "
        "budget in the candidate's tier windows, a restore failure that was not a clean miss, a tier that latched off on its own, a program compiled by a restore or a spill. Adopting is the owner's decision.",
        "# Time (estimate, minutes): %s: %d minutes (%d h %d min) in all, %d minutes of cards before the ship build (X0 to Z); the gates block (B0 to TC1) is %d minutes."
        % (', '.join('%s %d' % (name.split('-')[0], row['minutes']) for name, row in zip(names, table)), total, total // 60, total % 60, cards, minutes('B0 X0 A1 GX0 GX1 G1 TC1')),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in table]
    return '\n'.join(lines) + '\n'


def generate(plan_=None):
    plan_ = plan_ or plan()
    table = rows(plan_)
    out = {}
    for row in table:
        out['%s.env' % row['name']] = render_env(row)
    out['ORDER.txt'] = render_order(table, plan_)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args(argv)
    if not (arguments.write or arguments.check):
        parser.error('--write or --check')
    wanted = generate()
    found = {path.name: path.read_text(encoding='utf-8') for path in FOLDER.glob('*') if path.is_file()} if FOLDER.is_dir() else {}
    stale = sorted(name for name in set(wanted) | set(found) if wanted.get(name) != found.get(name))
    if arguments.check:
        for name in stale:
            sys.stderr.write('%s/%s is not what make_ship4_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
