"""The tp4/next-1 window packs (scripts/ci/references/fusion-jobs/S5), generated (stdlib only).

    python3 scripts/ci/make_s5_jobs.py --write   regenerate the pack
    python3 scripts/ci/make_s5_jobs.py --check   exit 1 if the checked-in pack is not what this generates

The NEXT production candidate (docs/tp4-next1.md): the traffic profile `-fx-traffic-s5` (scripts/ci/fusion-wp/S5.json, owner traffic waiver PENDING) on top of the shipped ship-4 stack, its audited twins
(make_s5_profiles.py) and the approved ship-4 profile as the control of every pair, ONE image (tp4-next-1, a placeholder: B0 builds it from the pushed tp4/next-1 head WITH the K64j-OQ graft), the checkpoint
production serves (ukisai/Swift-1.5-Qwen3.8-27b, selected by the job keys C2_MODEL_REPO and C2_MODEL_REVISION: qwen_c2_checkpoint.py). Templates only: no tag is pushed from the branch.

THREE ORDER FILES share the folder (the window driver reads a fixed ORDER.txt, so a window other than the first is run from a commit where its file is ORDER.txt: `git mv ORDER-W2.txt ORDER.txt`):

  ORDER.txt        WINDOW 1 (about 3 h of cards after the build): B0 build (while production serves), X0, A1 the combined audited attach on `-fx-s5-audit` (STOP), A2 the audited prefix arm (exact resume),
                   T1A and T1B the timed pair (the approved ship-4 profile against the candidate at one, two and four live seats with eight beside them), Z.
  ORDER-W2.txt     WINDOW 2: NP9's long-answer jobs N1 N2 N3 (the 32,768-token ceiling) on the same image and the candidate, A3 the audited sliced restore (optional), Z.
  ORDER-SHIP.txt   AFTER THE OWNER APPROVES (and only then): B1 the ship build and SR the platform replay of the candidate, exactly as the ship-4 pack's: B1 is refused by the job reader while the waiver is PENDING.

The agent-turn test (smoke name `agent_turns`, built on another branch) is a placeholder: AGENT_TURNS_TEST below is empty; set it to the smoke name once it has landed in scripts/ci/c2_serving_smoke.py and run
this script with --write: A1, T1A and T1B then name it (and their minutes grow by AGENT_TURNS_MINUTES).
"""

import argparse
from pathlib import Path
import sys

import make_fusion_jobs as jobs
import make_s5_profiles as twins
import make_ship4_profiles as ship4

HERE = Path(__file__).resolve().parent
FOLDER = HERE / 'references' / 'fusion-jobs' / 'S5'
IMAGE = 'tp4-next-1'
REPO = 'ukisai/Swift-1.5-Qwen3.8-27b'
REVISION = 'b4c84d42903a8646b25857eb2827288d92ed85a4'
SERVED = 'ukisai/Swift-1.5-Qwen3.8-27b'
SHIP4 = ship4.SHIP                      # the approved ship-4 profile: production, the control of every pair, and what B0 bakes as the window image's default
SHIP5 = twins.SHIP                      # the candidate (a traffic profile, waiver PENDING)
AUDIT = twins.AUDIT                     # the audited twin of the candidate (every audit, W3 WC with its audit)
AUDIT_NOWC = twins.AUDIT_NOWC           # its fallback without W3 WC
A1_TESTS = 'warmup,coding,np1_live2,concurrent5_split,concurrent8_drain,concurrent8_steady'
TIMED_TESTS = 'warmup,np1_live1,np1_live2,np1_live4,concurrent8_steady'
NP9_SOLO_TESTS = 'warmup,concurrent8_steady,long_answer_prefix,long_answer_solo_4k,long_answer_solo_200k'
NP9_SHORT_TESTS = 'warmup,concurrent8_steady,long_answer8_4k'
NP9_DEEP_TESTS = 'warmup,concurrent8_steady,long_answer8_96k'
# The production-shaped multi-turn agent test is built on another branch (smoke name agent_turns). Empty: a comment placeholder in A1, T1A and T1B. Set to the smoke name once
# scripts/ci/c2_serving_smoke.py has it and run --write: it is appended to their C2_SMOKE_TESTS and their minutes grow by AGENT_TURNS_MINUTES.
AGENT_TURNS_TEST = ''
AGENT_TURNS_MINUTES = 25
MODEL = ['C2_MODEL_REPO=%s' % REPO, 'C2_MODEL_REVISION=%s' % REVISION]
ORDERS = ('ORDER.txt', 'ORDER-W2.txt', 'ORDER-SHIP.txt')

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the tp4/next-1 windows (the ship-5 candidate; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/next-1 and push a tag experiment/c2-serving-vN, ONE TAG AT A TIME (wait for the run to finish; never cancel a running card job).
# Nothing here names a rig, a card, a host or a registry: the cards are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt, ORDER-W2.txt and
# ORDER-SHIP.txt have the order, the stop rules, the minutes and the read rules of the three runs.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head WITH the K64j-OQ graft (sed -i 's/%(image)s/<tag>/' *.env). A commit of a file
# the image carries needs a new B0 build and a new tag in every template. This window never stops, starts or hands back the node agent.
# C2_MODEL_REPO / C2_MODEL_REVISION (qwen_c2_checkpoint.py): a job that carries them serves the Swift checkpoint (what production serves) from its own converted-weight directory and, since NP9, with the output
# ceiling of its catalog entry; the keys are the same in every smoke job of this pack, so each pair differs by the profile alone. The prefix, replay and build steps do not forward the selection (the job reader
# refuses the keys beside them): they run the base checkpoint, which is what their question (exactness of reuse against a cold twin of the same boot; the platform sequence) does not depend on."""

CHECKPOINT_READ = ('CHECKPOINT READ (the workflow passes c2_smoke_check.py --expect-checkpoint): the container log holds "[QWEN-C2] checkpoint %s@%s key=... weight_cache=.../weights-ckpt/<key> base_cache=... '
                   '(the base cache is not read or written)" once, a "weight cache ...: found" line (the cache the Swift window built; this pack never builds it) and the worker\'s "[QWEN-C2] checkpoint bound ..." line; '
                   'the models answer names %s.' % (REPO, REVISION, SERVED))
TELEMETRY = ('TELEMETRY (C2_TELEMETRY=1, scripts/ci/card_telemetry.py): the read-only ARC sidecar samples every chip once a second through the smoke step; the first [TELEMETRY] line must say decode check ok. '
             'A rate difference that comes with a different loaded AICLK between the arms is the clocks, not the lever.')
ARC = 'RUN WITH THE CI PAUSED: the ARC runner sets are at ZERO for this job (ORDER, ARC-ZERO); no build, no CI push, no other job beside it.'
def np1_lines(admitted):
    """The NP1 lines of the container log; `admitted` is the text the [SOLO-PACKED] line ends with on the arm read."""
    return ('the NP1 lines in the container log: "[NP1] kv scratch: Qwen36ForCausalLM.allocate_kv_cache allocates 1 block(s) beyond the pool vLLM sized (pool 16500 blocks, scratch ids 16500..16500)" once, '
            '"[SOLO-PACKED] admitted min_users=1 scratch_pages=16500 (%s)" once and, for each of the two M3 blocks, "packed padded block admitted min_users=1 users=4 max_idle=4 ... scratch_pages=16500"; '
            'the octo block\'s own admitted line is unchanged (min_users=6, max_idle=2, no scratch pages)' % admitted)


SHAPES = ('np1_live1, np1_live2 and np1_live4 (one, two and four users started together on about 3,500-token real-code prompts, every answer to EXACTLY 2,000 tokens with ignore_eos, so the all-live window is long '
          'enough to read at either rate) and concurrent8_steady (eight users: the octo block, which the lone-user round never touches, so a parity row and the boot-to-boot floor)')
CEILING_READ = ('CEILING READ: the container log holds "[QWEN-C2] request output budget 32768 for this checkpoint (its catalog entry; the profile says 16384)" and "[QWEN-C2] request contract installed: greedy, max_tokens <= 32768" '
                '(c2_smoke_check requires the second whenever a long-answer test ran), the checkpoint lines (c2_smoke_check --expect-checkpoint) and no "kv reservation too large" line.')
ANSWER_READ = ('ANSWER READ (c2_smoke_check, long_answer_problems): every user of a long_answer test ends with finish "length" and exactly 30,000 tokens (ignore_eos, so the stop is the ceiling, not the model); fewer tokens at '
               '"length" is a server whose ceiling clamped the request.')
NO_MEMORY = ('MEMORY READ: the output ceiling allocates no verifier capture, proposal bucket or page table (docs/tp4-np9-answer32k.md), so the DRAM lines are the candidate\'s own: no "dram backstop refused", no "dram admission '
             'hold" that the same prompts would not have drawn at 16,384, no engine-build failure, no "audit mismatch", no stall.')
LEVER_READS = ('LEVER READS (the candidate\'s additions, on top of the ship-4 reads): NP1: octo_judge.py --container-log <container log> --profile %s clean (the solo rule: at least one "packed padded round live=1" line, no '
               '"packed padded skipped live=1 eligible=1" line, beside the octo rules) and ' + np1_lines('owner traffic waiver <id>') + '; EXACT RESUME: "[PINDIAG] prefix: window carry engaged undrop=1 carry=1" once (the scheduler graft\'s install line; a '
               'smoke sends no salt, so the carried admissions are the prefix arms\'); SLICED RESTORE: "host tier restore slice on blocks=512 ms=200" once; GRID13: the "tp4 grid13 engaged" line names all three '
               'sub-switches; TIER: the "host tier on" line says the 48 GiB budget and the 16 GiB checkpoint store.')


def agent(tests):
    return tests + (',' + AGENT_TURNS_TEST if AGENT_TURNS_TEST else '')


def agent_minutes():
    return AGENT_TURNS_MINUTES if AGENT_TURNS_TEST else 0


AGENT_PLACEHOLDER = ('# AGENT TURNS (placeholder): the production-shaped multi-turn agent test, smoke name agent_turns, is built on another branch. When it has landed in scripts/ci/c2_serving_smoke.py, set '
                     'AGENT_TURNS_TEST in scripts/ci/make_s5_jobs.py and run it with --write: this job then names it in C2_SMOKE_TESTS and its estimate grows by %d minutes.' % AGENT_TURNS_MINUTES)


def rows():
    """[row]: every template of the folder; row['orders'] says which ORDER files list it."""
    out = []
    out.append(dict(name='B0-build-window', orders=('w1',), mode='stop', minutes=40, actions='build', profile=None, tests=None, telemetry=False, box=None, model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP4, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B0 (STOP): the WINDOW image %s, built from the pushed tp4/next-1 head (the shipped ship-4 stack and its checkpoint selection plus NP1, NP9, exact resume P0, the sliced restore and the W3 WC lever, all '
                         'default off): the APPROVED ship-4 profile %s baked as the serving default with a session cap of 8, plus C2_BAKE_GRAFT=K64j-OQ. The candidate %s is a profile of the file and is chosen by a job\'s C2_PROFILE; it '
                         'cannot be the baked default while its owner traffic waiver is PENDING (ORDER-SHIP.txt, B1). READ: "baking the serving default %s with a session cap of 8", the provenance report with no problem line, its '
                         'K64j-OQ graft line, the image id. A build, not a card job: it may run while production serves; no privileged builder container may be up and no card job may run during it. Estimate from the last builds: '
                         '35-40 minutes.' % (IMAGE, SHIP4, SHIP5, SHIP4))))
    out.append(dict(name='X0-status-rescan-reset', orders=('w1', 'w2'), mode='stop', minutes=15, actions='status rescan reset', profile=None, tests=None, telemetry=False, box=None, model=False, extra=[],
                    why=('X0: the card state before the window: status, a rescan of the topology, an all-four reset. READ: four cards present and healthy, the fabric as measured, the compute grid of the run '
                         '(AVAILABLE WORKER CORE COUNT; 13x10 since the firmware unlock), no other window driver alive. The Swift weight cache exists on the card host: "weight cache ...: found" is the read of every '
                         'smoke job below. Stop rule: anything else. The owner stops the node agent and unserves production BEFORE it and restores production after Z with the cutover pack\'s own steps.')))
    a1_minutes = 90 + agent_minutes()
    out.append(dict(name='A1-s5-audited-attach', orders=('w1',), mode='stop', minutes=a1_minutes, actions='reset smoke', profile=AUDIT, tests=agent(A1_TESTS), telemetry=True, box=150, model=True, extra=[],
                    comment=AGENT_PLACEHOLDER,
                    why=('A1 (EXACTNESS of the whole candidate, STOP): %s, the audited twin of the ship-5 candidate: every audit of the shipped ship-4 stack (the K/V page writer\'s audit of every unit it writes, the verify T2 audit, the '
                         'pre-stage diff and two-block pre-stage audits, the multi-user SDPA launch\'s audit, the vglue, conv-gates and octo-lever audits, the 13x10 grids with all three sub-switches, the tile-aligned drafter window, the '
                         'host tier with every instrument on, the host read-backs SAMPLED) plus what the candidate adds: the lone-user padded round (NP1), exact resume with its carry audit, the sliced restore at the 48 GiB tier and '
                         'the 16 GiB checkpoint store, and the ONE lever the candidate does not carry, W3 WC (the GDN layer\'s windows, conv and gates in one launch, with its audit; F1\'s audit is out beside it), on Swift. '
                         'Tests: warmup; coding (ONE user for 1,500 tokens: every round a lone-user padded round); np1_live2 (a pair: two idle segments, page 0 alone, the flag changes nothing); concurrent5_split (four users in block A '
                         'and a fifth ALONE in block B, then a sixth: the split the placement leaves); concurrent8_drain (eight users whose budgets end one by one: every live count from 8 to 1, every split of the two blocks); '
                         'concurrent8_steady (the octo rounds, required by the smoke check on this profile).%s READ: c2_smoke_check clean (its fusion, octo, grid13 and WC rules read the container log); every audit line exact=True with '
                         'entries above 0; octo_judge.py --container-log <container log> --profile %s clean; %s; NO "audit mismatch", NO "fell back" (the WC rule alone accepts the one "fell back reason=8 users" line, the octo '
                         'block: WC takes 2 or 4 users), NO "kv shared", NO "page0 ... refused" for an eligible lone user, no stall. The memory ledger\'s free DRAM per chip after the attach: RECORD it. THE FIRST COMPILE OF THE WC KERNELS '
                         'IS THIS JOB (they have never been through a compiler): when the log shows the attach stopping at a WC compile error or a "gdn wc" refusal and at nothing else, that is a fault of an UNSHIPPED lever, '
                         'not of the candidate: run A1N (the same job on the twin without WC) and keep the window going; ANY other failure is NO-GO for the candidate and halts the window. %s %s %s'
                         % (AUDIT, (' After those, %s (see the placeholder line below).' % AGENT_TURNS_TEST) if AGENT_TURNS_TEST else ' The agent-turn test is added when it lands (placeholder below).',
                            AUDIT, np1_lines('gate only'), CHECKPOINT_READ, TELEMETRY, ARC))))
    out.append(dict(name='A1N-s5-audited-attach-nowc', orders=(), mode='stop', minutes=a1_minutes, actions='reset smoke', profile=AUDIT_NOWC, tests=agent(A1_TESTS), telemetry=True, box=150, model=True, extra=[],
                    comment=AGENT_PLACEHOLDER,
                    why=('A1N (the FALLBACK of A1, not in any ORDER: pushed only when A1 stopped on W3 WC alone): %s, the audited twin without W3 WC and with F1\'s audit back, on Swift, the same tests and the same READ as A1 for '
                         'everything but WC. If A1N passes the candidate is audited whole; W3 WC stays unproven and is not in the candidate. %s %s %s' % (AUDIT_NOWC, CHECKPOINT_READ, TELEMETRY, ARC))))
    out.append(dict(name='A2-s5-prefix-exact-resume', orders=('w1',), mode='soft', minutes=55, actions='reset prefix', profile=None, tests=None, telemetry=False, box=120, model=False,
                    prefix=('exact-resume', AUDIT), extra=[],
                    why=('A2 (EXACT RESUME and the prefix store, the prefix arm the smoke harness cannot be: a smoke sends no salt, so no smoke test reaches the prefix store, the host tier or a returning turn): c2_prefix_gate.py '
                         '--plan exact-resume on the audited twin %s, no baseline: first prompts of about 9k, 33k and 120k tokens, then returning tool turns of 150, 400, 900, 1,700, 2,600 and 300 new tokens, EVERY turn a '
                         'cold/hit pair under a fresh salt, on sticky sessions with the undrop and the drafter-window carry and their audits, beside the lone-user padded round (each request here is alone: every round is a '
                         'lone-user padded round with every audit on), the preconverted checkpoints and the tier at its new size. It runs on the BASE checkpoint (the prefix step does not forward the selection; exactness is a '
                         'cold twin of the same boot, which does not depend on the weights). READ: the plan verdict PASS, every pair IDENTICAL, the oracle\'s Q equal to the hit\'s Q on every turn, "window carry carried" lines on the '
                         'turns that fit the window and "window carry audit ... exact=True" with the carried window\'s digest equal to the cold twin\'s, no "audit mismatch", no "fell back", no stall, no program compiled by a '
                         'hit after its cold twin. SOFT: a failure here is NO-GO for the exact-resume lines of the candidate (QWEN_PREFIX_STICKY_UNDROP and QWEN_PREFIX_WINDOW_CARRY in the ship entry of '
                         'scripts/ci/fusion-wp/S5.json), not for the rest; read it before the owner approves. Estimate 55 minutes (the ER pack\'s ERX estimated 50). %s' % (AUDIT, ARC))))
    out.append(dict(name='T1A-s4-control', orders=('w1',), mode='soft', minutes=30 + agent_minutes(), actions='reset smoke', profile=SHIP4, tests=agent(TIMED_TESTS), telemetry=True, box=80, model=True, extra=[],
                    comment=AGENT_PLACEHOLDER,
                    why=('T1A (TIMED, the CONTROL of the pair): the approved ship-4 profile %s on Swift, the stack production runs: a lone live user decodes on the 4-row per-request engine, two or three on the padded 64-row block, four on '
                         'the full block, six or more on the octo block. Tests: %s. Every boot a fresh reset. %s %s %s' % (SHIP4, SHAPES, ARC, CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='T1B-s5-candidate', orders=('w1',), mode='soft', minutes=30 + agent_minutes(), actions='reset smoke', profile=SHIP5, tests=agent(TIMED_TESTS), telemetry=True, box=80, model=True, extra=[],
                    comment=AGENT_PLACEHOLDER,
                    why=('T1B (TIMED, the CANDIDATE of the pair): the ship-5 candidate %s, a TRAFFIC profile whose owner waiver is PENDING (a profile whose decision does not start with APPROVED boots in a gate or a smoke and logs '
                         'the word), the same tests as T1A, on Swift. %s READ: c2_smoke_check clean AND python3 scripts/ci/octo_judge.py --container-log <container log> --profile %s clean (the solo rule beside the octo rules); %s. '
                         'Then: python3 scripts/ci/np1_report.py --control-a <T1A smoke> --arm-a <T1B smoke> --control-b <T1A smoke> --arm-b <T1B smoke> --arm-container <T1B container> --profile %s: np1_report is an ABAB '
                         'reader and this pair has ONE boot a side, so each smoke is passed twice and the boot-to-boot floors read exactly 1 (no noise allowance): the ratios per test (the aggregate of the all-live window of '
                         'the candidate over the control\'s) are the read, a GAIN at np1_live1 is the claim, and a LOSS the reader prints at a parity row for a difference of 2%% to 5%% is boot noise until a second pair '
                         '(T1A T1B again, the owner\'s call) says otherwise. The texts must be EQUAL, user by user: python3 scripts/ci/octo_compare.py --control <T1A smoke> --octo <T1B smoke> --octo-container <T1B container> --profile %s: the '
                         'same checkpoint, greedy, so a difference is a token the candidate changed and is NO-GO. %s %s %s' % (SHIP5, ARC, SHIP5, LEVER_READS % SHIP5, SHIP5, SHIP5, ARC, CHECKPOINT_READ, TELEMETRY))))
    out.append(dict(name='Z-reset', orders=('w1', 'w2'), mode='soft', minutes=5, actions='status reset', profile=None, tests=None, telemetry=False, box=None, model=False, extra=[],
                    why=('Z: the window\'s end: status and an all-four reset, no agent start (the cards stay under development). The owner restores the ARC runner sets to their normal scale, then production: card reset, link '
                         'measurement, topology republish, agent restart, deploy and the post-checks. Nothing in this window touched production\'s image, profile, alias or cache.')))
    # window 2: NP9's long-answer jobs on the candidate, and the audited sliced restore
    out.append(dict(name='N1-long-answer-solo', orders=('w2',), mode='stop', minutes=75, actions='reset smoke', profile=SHIP5, tests=NP9_SOLO_TESTS, telemetry=True, box=150, model=True, extra=[],
                    why=('N1 (STOP; NP9\'s N1 on the candidate): ONE request at a time, 30,000 tokens, on Swift. warmup; concurrent8_steady (the profile\'s required eight-user check, which also proves the octo rounds still run); '
                         'long_answer_prefix (one 4k real-code prompt asked at max_tokens 16,384 and then 30,000, both ignore_eos: the long answer must begin with the short one but for its last 512 characters - greedy decoding does not '
                         'change with the ceiling); long_answer_solo_4k (a 4k prompt, 30,000 tokens, ignore_eos) and long_answer_solo_200k (a 200,000-token prompt as the server counts it, 30,000 tokens, ignore_eos: the request reserves '
                         '3,596 of the pool\'s 16,499 blocks). Each request is alone: every round is a lone-user padded round (NP1). %s %s %s READ also: the solo decode rate is recorded, never gated; the stream never stalls for the '
                         'answer\'s 7 to 15 minutes (a gap over the profile\'s QWEN_FAST_LEVERN_MAX_DECODE_GAP_S, 8 s, is a fault). Stop rule: a request that ends short of 30,000 tokens, a prefix difference, a stall, any audit '
                         'mismatch or a refusal line. %s %s' % (CEILING_READ, ANSWER_READ, NO_MEMORY, ARC, TELEMETRY))))
    out.append(dict(name='N2-long-answer8-short', orders=('w2',), mode='soft', minutes=70, actions='reset smoke', profile=SHIP5, tests=NP9_SHORT_TESTS, telemetry=True, box=120, model=True, extra=[],
                    why=('N2 (NP9\'s N2 on the candidate): EIGHT requests at once, 4k prompts, 30,000 tokens each (ignore_eos), on Swift: eight seats decode together for the whole answer (octo rounds at eight live, the engine-reuse '
                         'rebinds, the page tables growing by 470 blocks each, the host KV tier idle). The KV reservation holds 535 blocks a seat, 4,280 of 16,499: nothing is held. %s %s %s READ also: the all-live window\'s '
                         'per-seat rate (live4_tok_s) is recorded, never gated; octo_judge.py --container-log <log> --profile %s is clean. %s %s' % (CEILING_READ, ANSWER_READ, NO_MEMORY, SHIP5, ARC, TELEMETRY))))
    out.append(dict(name='N3-long-answer8-deep', orders=('w2',), mode='soft', minutes=80, actions='reset smoke', profile=SHIP5, tests=NP9_DEEP_TESTS, telemetry=True, box=150, model=True, extra=[],
                    why=('N3 (NP9\'s N3 on the candidate): EIGHT requests at once, prompts of about 96,000 tokens (as the server counts them), 30,000 tokens each (ignore_eos), on Swift. The longest prompt at which eight 30,000-token '
                         'requests are all resident: 8 x 1,971 = 15,768 of the 16,499 blocks (docs/tp4-np9-answer32k.md), so no seat waits: the container log holds NO "kv reservation hold" line (a hold here means the pool arithmetic '
                         'is wrong). The eight prefills queue (about 26 s each), then eight seats decode at depth 96,000 to 126,000 together for 30,000 tokens. %s %s %s READ also: the per-seat rate at depth is recorded against '
                         'concurrent8_code_128k_long of the Swift window, never gated. %s %s' % (CEILING_READ, ANSWER_READ, NO_MEMORY, ARC, TELEMETRY))))
    out.append(dict(name='A3-s5-tier-slice-audited', orders=('w2',), mode='opt', minutes=100, actions='reset prefix', profile=None, tests=None, telemetry=False, box=180, model=False,
                    prefix=('tier-concurrent-sliced', AUDIT), extra=[],
                    why=('A3 (OPTIONAL, the sliced restore under every audit): c2_prefix_gate.py --plan tier-concurrent-sliced on the audited twin %s, no baseline: a session built to the largest size the profile admits, fresh-salt floods '
                         'the size of the KV pool so it leaves the device pool for host RAM, SEVEN SEATS decoding, a 10 s settle and a 60 s control window, then the session RETURNS (restored from the host tier in slices of 512 blocks '
                         'of at most 200 ms, the seats decoding between them), a long (120k) fresh prompt (SPILL), the returning turn\'s COLD TWIN. It runs on the base checkpoint (the prefix step does not forward the '
                         'selection). THE QUESTION: is a restore done in slices, at the 48 GiB tier, byte-identical to the whole one under every audit (verify=all, the read-back, the extent audit), beside the lone-user round? '
                         'READ: the plan verdict PASS, every pair IDENTICAL, "host tier restore slice on blocks=512 ms=200" once, a restore line with slices above 1 and held_steps above 0, no slice over 1,000 ms, no "slice aborted" '
                         'line the request\'s own end does not explain, no "fell back", zero failure counters, all seven seats across every judged window. A failure is NO-GO for the sliced-restore lines of the candidate '
                         '(QWEN_PREFIX_TIER_RESTORE_SLICE and _MS in the ship entry of scripts/ci/fusion-wp/S5.json), not for the rest. OPT: an optional job the driver admits only if it fits the window. Estimate 100 minutes '
                         '(the tier-slice pack\'s TS1). %s' % (AUDIT, ARC))))
    # after the approval
    out.append(dict(name='B1-ship-build', orders=('ship',), mode='soft', minutes=35, actions='build', profile=None, tests=None, telemetry=False, box=None, model=False,
                    extra=['C2_BAKE_DEFAULT_PROFILE=%s' % SHIP5, 'C2_BAKE_GRAFT=K64j-OQ'],
                    why=('B1 (AFTER THE OWNER APPROVES; refused until then, by design): build the SHIP image from the commit that records the owner\'s approval, with %s baked as the image\'s serving default (ENV QWEN_C2_PROFILE) and '
                         'THATCH_SERVING_SESSION_CAP baked as its max-num-seqs (8). REFUSED by c2_serving_job.read_bake and by build-c2-serving-image.sh while the profile\'s owner_traffic_waiver (id %s) has a decision that does '
                         'not start with APPROVED: the owner approves first, the approval is committed (make_fusion_profiles.py --approve-ship --ship traffic-s5), the CPU suite runs on that commit, and the tag is pushed on THAT '
                         'commit. READ: "baking the serving default %s with a session cap of 8", the provenance report "(c) the image default is the BAKED profile ...", no problem line, the image id. THE IMAGE BAKES THE '
                         'K64j-OQ OP GRAFT. Estimate 35 minutes.' % (SHIP5, twins_waiver_id(), SHIP5))))
    out.append(dict(name='SR-platform-replay', orders=('ship',), mode='soft', minutes=25, actions='reset replay', profile=None, tests=None, telemetry=False, box=60, model=False, extra=[],
                    replay=SHIP5,
                    why=('SR (AFTER THE OWNER APPROVES, on the image B1 builds): the four-card platform replay of the PRODUCTION image: the node agent\'s serving sequence in a copy of its container running the THIN LAYER built on the '
                         'B1 image (C2_PLATFORM_IMAGE=local:thin-layer: the step reads the image name from a file the operator writes on the runner; it is never committed). The replay sets no profile, so it boots the ship profile BY THE '
                         'IMAGE DEFAULT, which is why it needs the baked B1 image and so the APPROVED decision (it cannot run on the window image, whose default is the approved ship-4 profile). THE RECORDED SOURCE IMAGE: the replay copies '
                         'the node agent\'s container from a tracked record (scripts/ci/references/c2-serving, the step\'s --source) and needs that image present on the rig host (c2_platform_replay.source_problem); when the step '
                         'reports \'the source image ... is not on this host\', the operator pulls it first - the image name stays out of this file. READ the vLLM subprocess log: \'[QWEN-C2] profile %s: vLLM argv\', the OWNER '
                         'TRAFFIC WAIVER line, the P150x4 mesh, admission passed with no UNQUALIFIED admission line and no 262k evidence waiver, prefix installed, the Lever N install and warm lines, the parked build line, every op-fusion '
                         'lever\'s engaged line, the :tt alias before and after the restart, and no kill-switch line (no levern.off, prefix-reuse.off or parked.off left on the hub). PASS: \'PLATFORM_REPLAY passed=True\'. The replay '
                         'runs the base checkpoint (the replay step does not forward the selection); production picks Swift from its node environment, which the replay\'s copy of the agent\'s container records. '
                         'The replay turns the session cap off; the cap is read live after the deploy. %s Measured basis for the estimate: 25 minutes.' % (SHIP5, LEVER_READS % SHIP5))))
    return out


def twins_waiver_id():
    import json
    profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
    return profiles[SHIP5]['owner_traffic_waiver']['id']


def render_env(row):
    lines = [HEADER % dict(image=IMAGE)]
    why = row['why'] if ('Duration' in row['why'] or 'Estimate' in row['why']) else row['why'] + ' Duration (estimate): %d minutes.' % row['minutes']
    lines += ['# ' + piece for piece in jobs.wrap(why)]
    if row.get('comment'):
        lines += ['# ' + piece for piece in jobs.wrap(row['comment'][2:])]
    lines += ['C2_CARDS=quad', 'C2_ACTIONS=%s' % row['actions'], 'C2_IMAGE_TAG=%s' % IMAGE]
    if row['profile']:
        lines.append('C2_PROFILE=%s' % row['profile'])
    if row.get('prefix'):
        plan, profile = row['prefix']
        lines += ['C2_PREFIX_PLAN=%s' % plan, 'C2_PREFIX_PROFILE=%s' % profile, 'C2_PREFIX_BASELINE=none']
    if row.get('replay'):
        lines += ['C2_PLATFORM_IMAGE=local:thin-layer', 'C2_REPLAY_PROFILE=%s' % row['replay']]
    if row['model']:
        lines += MODEL
    if row['tests']:
        lines.append('C2_SMOKE_TESTS=%s' % row['tests'])
    lines += row['extra']
    if row['telemetry']:
        lines.append('C2_TELEMETRY=1')
    if row.get('box') and not any(item.startswith('C2_BOX_MINUTES=') for item in row['extra']):
        lines.append('C2_BOX_MINUTES=%d' % row['box'])
    return '\n'.join(lines) + '\n'


def needs_of(name):
    """The NEEDS directives of one ORDER file, by window key."""
    return {
        'w1': ['X0 <- B0', 'A1 <- X0', 'A2 <- A1', 'T1A <- A1', 'T1B <- T1A'],
        'w2': ['N1 <- X0', 'N2 <- N1', 'N3 <- N1', 'A3 <- X0'],
        'ship': ['SR <- B1'],
    }[name]


def table_of(table, key):
    return [row for row in table if key in row['orders']]


def total_lines(table):
    names = [row['name'] for row in table]
    total = sum(row['minutes'] for row in table)
    builds = sum(row['minutes'] for row in table if row['actions'] == 'build')
    return names, total, total - builds, builds


def blocks_text(table, blocks):
    def minutes(tokens):
        return sum(row['minutes'] for row in table if row['name'].split('-')[0] in tokens.split())
    return '; '.join('%s: %s (%d min)' % (title, tokens, minutes(tokens)) for title, tokens in blocks)


COMMON_P0 = [
    "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE X0 and restores it after Z; nothing in this pack does either. B0 is a build and may run while production serves. No other window driver may be alive and no privileged builder container up.",
    "# P0. The CPU suite is green on the pushed branch head (gh workflow run qwen-cpu-suite.yml -f ref=<sha>) and each tag is pushed on THAT commit (the throwaway commit carries only .github/c2-serving-job.env). Push ONE tag at a time and wait for its run "
    "to finish; queued tags are cancelled. Never cancel a running card job. Every result is UNQUALIFIED.",
    "# P0. SWIFT IS ON THE CARD HOST AND ITS CONVERTED CACHE IS BUILT: python3 scripts/ci/qwen_c2_checkpoint.py check --hub <the hub directory> --repo %s --revision %s exits 0, and the Swift window's W0 (or the production "
    "first boot) has built the keyed weight cache, so every smoke job here logs \"weight cache ...: found\". This pack builds nothing of it." % (REPO, REVISION),
    "# P0. GRID: the four cards expose a 13x10 compute grid since the firmware unlock. Read it from the run, never assume 11x10: every pair runs on one grid.",
]

COMMON_FORMAT = "# One template per line: <template name without .env> <stop|soft|opt> <image tag> <estimated minutes>. stop: the window halts at the first failure of this job. soft: a failure or a skip does not stop the window (the NEEDS lines skip what depended on it). opt: optional, admitted only if it fits."

CANDIDATE = ("# THE CANDIDATE (docs/tp4-next1.md): %s, a TRAFFIC profile under ONE owner traffic waiver (id %s, decision PENDING: the owner approves at ship time) = the approved ship-4 profile %s plus the lone-user padded round (NP1, "
             "QWEN_FAST_SOLO_PACKED=1), exact resume P0 (QWEN_PREFIX_STICKY_UNDROP=1, QWEN_PREFIX_WINDOW_CARRY=1), the sliced host-tier restore (QWEN_PREFIX_TIER_RESTORE_SLICE=512, _MS=200), the decode verify sub-switch of the 13x10 "
             "grids (QWEN_FAST_GRID13=1, was prefill,drafter) and the owner-approved host RAM budget (QWEN_PREFIX_HOST_TIER_GIB=48 beside QWEN_PREFIX_STORE_GIB=16). The image also carries NP9: the Swift catalog entry's 32,768-token output "
             "ceiling (qwen_c2_checkpoints.json, \"output_budget\"; deleting that one line restores 16,384 and changes nothing else). W3 WC is in the audited twin only; W3 MR and W3 keyed are not in this tree (the batched reads "
             "and the round-host keyed diff of the shipped stack are the same levers).")


def render_order_w1(table):
    names, total, cards, builds = total_lines(table)
    aud = 'A1'
    lines = [
        "# The order of the tp4/next-1 WINDOW 1 (docs/tp4-next1.md; scripts/ci/make_s5_jobs.py): the ship-5 candidate %s on top of the shipped ship-4 stack, the checkpoint production serves (%s, revision %s), ONE image (%s, a "
        "placeholder: B0 builds it from the pushed tp4/next-1 head WITH the K64j-OQ graft) and ONE window. The control of every pair is the approved ship-4 profile %s on the same image." % (SHIP5, REPO, REVISION[:12], IMAGE, SHIP4),
        COMMON_FORMAT,
        CANDIDATE % (SHIP5, twins_waiver_id(), SHIP4),
        "# WHAT THIS DECIDES (the owner's, never the window's): whether the candidate goes to production in place of the ship-4 profile. A1 is the exactness gate of everything the candidate adds and of the one lever it does not (W3 WC); "
        "A2 is the only job that reaches the prefix store, a returning turn and the sticky chain (a smoke sends no salt); T1A T1B time the candidate against production at one, two and four live seats with eight beside them. "
        "Expected: a lone user x2 to x3 (the arithmetic of docs/tp4-np1-fastlane.md, not a measurement), nothing at two or more seats, a returning agent turn 0.5 to 1.3 s sooner (docs/tp4-exact-resume.md, an estimate). Every result is UNQUALIFIED.",
        "# THE NEW DEVICE FACTS (first on a card here): NP1 gives the K/V writers and the extent reader a block id past vLLM's pool in a page table (K1 to K4 of docs/tp4-np1-fastlane.md); exact resume wraps vLLM's hit lookup and uploads carried "
        "drafter-window rows; the sliced restore holds a returning request between slices; W3 WC launches two kernels that have never been through a compiler. A1 and A2 are the first card to do any of it, with the audits on.",
        "# AGENT-TURN TEST: the production-shaped multi-turn agent test (smoke name agent_turns) is built on another branch. %s HOW TO ADD IT: set AGENT_TURNS_TEST = 'agent_turns' in scripts/ci/make_s5_jobs.py once c2_serving_smoke.py has it, "
        "run python3 scripts/ci/make_s5_jobs.py --write (A1, A1N, T1A and T1B then name it; their estimates grow by %d minutes each), commit." % ('It IS NAMED in A1, A1N, T1A and T1B.' if AGENT_TURNS_TEST else 'It is NOT yet named in any job (a placeholder comment sits in A1, A1N, T1A and T1B).', AGENT_TURNS_MINUTES),
    ] + COMMON_P0 + [
        "# ARC-ZERO A1 T1B",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO (minRunners and maxRunners 0, the normal values written down first, restored by the same patch with the old numbers) BEFORE A1 and restores them AFTER T1B, i.e. around "
        "every audited boot (a model load under CI load has hit the platform's load timeout) and every timed job. B0, X0 and Z do not need the pause.",
        "# THE JOBS, in value order (cut the window at a block end): %s. B0 is the WINDOW build and runs while production serves, so it is outside the card minutes." % blocks_text(table, (('GATES', 'X0 A1'), ('PREFIX ARM', 'A2'), ('DECODE PAIR', 'T1A T1B'), ('END', 'Z'))),
        "# 3 H CUT: A2 is the only job that is not the owner's list of window 1 (the smoke harness cannot reach the prefix store); X0 A1 T1A T1B Z is %d minutes. To run it later (as the first job of window 2, or alone) set SKIP_JOBS=A2 in the pack conf." % sum(row['minutes'] for row in table if row['name'].split('-')[0] in ('X0', 'A1', 'T1A', 'T1B', 'Z')),
        "# FALLBACK: A1N (a template in this folder, not in this ORDER) is A1 on %s, the audited twin without W3 WC. Push it, with the same placeholder tag, only when A1 stopped on W3 WC alone (a WC compile error or a 'gdn wc' refusal and nothing else). It takes A1's place in every NEEDS line below." % AUDIT_NOWC,
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for need in needs_of('w1'):
        lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean, whose fusion, octo, grid13, exact-resume and WC rules read the container log, and no stall; the templates' own headers have the whole text):",
        "#   B0   'baking the serving default %s with a session cap of 8', the provenance report with no problem line, its K64j-OQ graft line, the image id." % SHIP4,
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; the Swift weight cache found.",
        "#   A1   (STOP) every audit line exact=True with entries above 0, octo_judge.py clean under %s (the solo rule), the NP1 lines, the WC engaged line on the M3 blocks and its audit line (the octo block's one 'fell back reason=8 users' line is the design), "
        "no 'audit mismatch', no other 'fell back', no 'kv shared', the memory ledger recorded. A fault of W3 WC alone: run A1N. ANY other failure is NO-GO and halts the window." % AUDIT,
        "#   A2   (SOFT, the prefix arm) the exact-resume plan PASS: every pair IDENTICAL, the oracle's Q on every turn, the carry's audit lines exact=True, no stall; a failure is NO-GO for the exact-resume lines.",
        "#   T1A T1B   np1_report.py GO (a GAIN at np1_live1, no LOSS elsewhere), octo_compare.py (every answer equal) and octo_judge.py on the candidate's log; c2_smoke_check clean on both.",
        "# NO-GO for the candidate: any stall, any text difference against the control, any audit mismatch, any fallback line beyond the one named, an arm that logs no engaged line, a refusal at the attach (the cache not one block longer, a request block id "
        "past the pool, a restore that latched the tier off), a lone-user round that never ran or an eligible one that went to the engines, a LOSS in a timed test. A failing lever is taken out by deleting its lines from the ship entry of scripts/ci/fusion-wp/S5.json "
        "and regenerating (a new build). Adopting is the owner's decision: ORDER-SHIP.txt (B1, SR) runs after the approval, never before.",
        "# Time (estimate, minutes): %s: %d minutes (%d h %d min) in all, %d minutes of cards (X0 to Z), %d of build." % (', '.join('%s %d' % (name.split('-')[0], row['minutes']) for name, row in zip(names, table)), total, total // 60, total % 60, cards, builds),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in table]
    return '\n'.join(lines) + '\n'


def render_order_w2(table):
    names, total, cards, builds = total_lines(table)
    lines = [
        "# The order of the tp4/next-1 WINDOW 2 (docs/tp4-next1.md, docs/tp4-np9-answer32k.md; scripts/ci/make_s5_jobs.py): NP9's long-answer jobs on the ship-5 candidate %s and the SAME image as window 1 (%s, the tag B0 built), Swift "
        "(%s, revision %s) with the 32,768-token output ceiling of its catalog entry, and A3 the audited sliced restore. RUN IT FROM A COMMIT WHERE THIS FILE IS ORDER.txt (git mv ORDER-W2.txt ORDER.txt): the window driver reads a fixed ORDER.txt." % (SHIP5, IMAGE, REPO, REVISION[:12]),
        COMMON_FORMAT,
        CANDIDATE % (SHIP5, twins_waiver_id(), SHIP4),
        "# WHAT THIS DECIDES (the owner's, never the window's): whether the 32,768-token ceiling goes to production with the candidate. The CPU side is settled (docs/tp4-np9-answer32k.md: the ceiling allocates no device memory on the serving shape; it costs "
        "KV admission only); what only a card can show is that a seat runs 30,000 tokens, alone and eight at once, at a short and a long prompt, with the text unchanged by the ceiling. Every result is UNQUALIFIED.",
        "# WINDOW 1 AND THE CEILING: the catalog key reaches every Swift boot of the image, window 1's included. Window 1 asks at most 2,000 tokens (np1 shapes) and 12,288 (the coding set) and compares arms booted from the same image, so the key does not touch its "
        "reads. If window 1 must run on the old 16,384 ceiling, delete the one line '\"output_budget\": 32768' from the Swift entry of scripts/ci/qwen_c2_checkpoints.json before B0 (and put it back before this window): it is data and nothing else moves.",
    ] + COMMON_P0 + [
        "# ARC-ZERO N1 A3",
        "# ARC PAUSE / RESTORE: the owner scales the rig's ARC runner sets to ZERO BEFORE N1 and restores them AFTER A3 (the decodes here run 7 to 30 minutes and every boot is a model load under the platform's timeout).",
        "# THE JOBS, in value order (cut the window at a block end): %s." % blocks_text(table, (('ONE SEAT', 'X0 N1'), ('EIGHT SEATS', 'N2 N3'), ('SLICED RESTORE', 'A3'), ('END', 'Z'))),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for need in needs_of('w2'):
        lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES (each also needs c2_smoke_check clean, with the container log, and no stall; the templates' own headers have the whole text):",
        "#   X0   four cards present and healthy, the fabric as measured, the 13x10 grid in the run, no other window driver alive; Swift on the host.",
        "#   N1   (STOP) the ceiling lines (request output budget 32768; request contract installed ... <= 32768), the checkpoint lines, long_answer_prefix prefix_equal=True, every long_answer user finish length with 30000 tokens.",
        "#   N2   the same ceiling and answer rules at eight live; octo_judge clean; no hold line (4,280 of 16,499 blocks).",
        "#   N3   the same at 96,000-token prompts; NO 'kv reservation hold' line (15,768 of 16,499 blocks); the per-seat rate at depth recorded.",
        "#   A3   (OPT) the tier-concurrent-sliced plan PASS: every pair IDENTICAL, the slice lines, zero failure counters, all seven seats across every judged window.",
        "# NO-GO for the 32,768 ceiling: a request that ends short of its 30,000 tokens, a prefix difference in long_answer_prefix, a stall, an audit mismatch, an engine-build or DRAM refusal that 16,384 would not have drawn, a 'kv reservation hold' at N3. "
        "NO-GO for the sliced restore: a text that differs, a slice over 1,000 ms, a tier that latched off on its own.",
        "# Time (estimate, minutes): %s: %d minutes (%d h %d min) in all, %d minutes of cards (X0 to Z)." % (', '.join('%s %d' % (name.split('-')[0], row['minutes']) for name, row in zip(names, table)), total, total // 60, total % 60, cards),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in table]
    return '\n'.join(lines) + '\n'


def render_order_ship(table):
    names, total, cards, builds = total_lines(table)
    lines = [
        "# The order of the tp4/next-1 SHIP FLOW (docs/tp4-next1.md; scripts/ci/make_s5_jobs.py): B1 builds the ship image and SR replays the platform's serving sequence on it. BOTH WAIT FOR THE OWNER'S APPROVAL of the owner traffic waiver of %s (id %s): "
        "B1 is refused by c2_serving_job.read_bake while its decision starts with PENDING, and SR boots the profile BY THE IMAGE DEFAULT (c2_platform_replay), so it needs B1's image. Order: the owner approves, `python3 scripts/ci/make_fusion_profiles.py "
        "--approve-ship \"APPROVED ...\" --ship traffic-s5` regenerates the profile, the CPU suite runs on that commit, B1's tag is pushed on THAT commit, then SR. RUN IT FROM A COMMIT WHERE THIS FILE IS ORDER.txt (git mv ORDER-SHIP.txt ORDER.txt)." % (SHIP5, twins_waiver_id()),
        COMMON_FORMAT,
        "# P0. PRODUCTION: the owner stops the node agent and unserves production BEFORE SR and restores it after, with the cutover pack's own steps (B1 is a build and may run while production serves). The card jobs of the earlier windows are over.",
        "# P0. The CPU suite is green on the commit that records the approval, and each tag is pushed on THAT commit. Push ONE tag at a time and wait for its run to finish; never cancel a running card job.",
        "# THE JOBS: %s." % blocks_text(table, (('SHIP BUILD', 'B1'), ('PLATFORM REPLAY', 'SR'))),
        "# DEPENDENCIES (machine-greppable): '# NEEDS <jobs> <- <jobs>' means the jobs on the left run only if every job on the right completed and passed its READ rule; otherwise the driver skips them.",
    ]
    for need in needs_of('ship'):
        lines.append('# NEEDS ' + need)
    lines += [
        "# READ RULES:",
        "#   B1   'baking the serving default %s with a session cap of 8', the provenance report '(c) the image default is the BAKED profile ...', no problem line, the image id." % SHIP5,
        "#   SR   'PLATFORM_REPLAY passed=True'; the lever reads of the template's header (the NP1 lines, the carry's install line, the slice line, the grid13 line, the 48 GiB tier line).",
        "# Time (estimate, minutes): %s: %d minutes (%d h %d min) in all, %d minutes of cards." % (', '.join('%s %d' % (name.split('-')[0], row['minutes']) for name, row in zip(names, table)), total, total // 60, total % 60, cards),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in table]
    return '\n'.join(lines) + '\n'


def generate():
    table = rows()
    out = {}
    for row in table:
        out['%s.env' % row['name']] = render_env(row)
    out['ORDER.txt'] = render_order_w1(table_of(table, 'w1'))
    out['ORDER-W2.txt'] = render_order_w2(table_of(table, 'w2'))
    out['ORDER-SHIP.txt'] = render_order_ship(table_of(table, 'ship'))
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
            sys.stderr.write('%s/%s is not what make_s5_jobs.py generates: run it with --write\n' % (FOLDER, name))
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
