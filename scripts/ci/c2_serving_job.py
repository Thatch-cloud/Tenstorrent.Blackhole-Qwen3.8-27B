"""Read .github/c2-serving-job.env: the job the next experiment/c2-serving-vN tag runs.

    python3 scripts/ci/c2_serving_job.py .github/c2-serving-job.env >> "$GITHUB_OUTPUT"

prints one key=value line per workflow output (qwen-c2-serving.yml's 'Read the job file' step used
to parse the file inline; this is that parse, with the gate's keys added, where a CPU test can hold
it). Refuses (exit 1, the reason on stderr) anything the workflow must not act on: an unknown
action, a malformed image tag, platform image or replay model id, a profile the checkout's
qwen_c2_profiles.json does not define, an unknown gate plan, a prompt length that is not a positive
integer, and a card-M harness, argument or environment the cardm step must not run (read_cardm).

Keys (every one optional but C2_IMAGE_TAG):
  C2_ACTIONS          ACTIONS, run in the workflow's order (default: status); a job names them in that order, else it is
                      refused. agentstop (right after status) stops the host's node agent, and with it the serving
                      container it started, so the cards are free; agentstart (last, after push) starts the agent
                      again and waits for its container to answer /v1/models. Neither opens a card, so a job of only
                      status, agentstop, unserve and agentstart is valid with C2_CARDS=pair or quad.
  C2_RMI_TAGS         space-separated image tags the rmi action (right after status) deletes from the host's Docker, each
                      matching RMI_TAG and naming the image <registry>/tt-vllm:qwen38-c2-<tag> (the workflow forms the
                      name). Required with rmi, refused without it; a PROTECTED tag (a production lineage) is refused.
                      rmi opens no card, so a job of only rmi (or status rmi) is valid with C2_CARDS=pair or quad.
  C2_IMAGE_TAG        the image is zot.thatch.local:5000/tt-vllm:qwen38-c2-<tag>
  C2_FABRIC           the fabric config the fabric action opens the (1, 4) mesh under: FABRIC_1D (the default, what
                      the TT plugin sets) or FABRIC_1D_RING; needs C2_CARDS=quad when set
  C2_FABRIC_PROBE     what the fabric action runs on the (1, 4) mesh: fabric (the default: tp4_fabric_probe.py, the ring, the
                      link count and the collectives timed) or rs-tile (tp4_rs_tile_spike.py: the model's own all-reduce on a
                      64-row block against its one-tile calls, bit for bit, then the tile-split wrapper; the verdict line
                      TP4_RS_TILE) or mr (tp4_mr_probe.py: the verify readback's mesh-read arms at four chips, optimisation/ttnn-op/mr_probe's
                      arms and verdict, MR_PROBE verdict=GO|MESH-ONLY|NO-GO) or kvread (tp4_kv_read_probe.py: the prefix audit's region read,
                      ttnn.qwen_read_blocks, against the whole-cache read byte for byte on the pool-sized cache; KV_READ_PROBE verdict=PASS|FAIL, the exit
                      status is the verdict); needs the fabric action
  C2_SUPERSEDED_BY    set on every template of a pack that a later pack replaced (references/tp4-w2-jobs); read_job REFUSES such a template
  C2_PROFILE          the C2 profile smoke and gate serve (default: general)
  C2_CARDS            the card set the hardware steps open: pair (cards M and A, the default) or quad (every
                      Blackhole board present, the four-card (1, 4) mesh the TP4 profiles open). quad takes
                      status, platform, unserve, reset (all four together), fabric (the four-card fabric probe),
                      build, drift, probe, smoke, gate, prefix, replay (the agent's serving sequence on all four
                      boards: it needs C2_REPLAY_PROFILE, a P150x4 profile) and push - not cardm or priority, which
                      are pair-shaped - and only profiles that name mesh_device P150x4 (general-tp4 ...);
                      pair takes only profiles that name none. Anything else is refused here, before a card opens.
  C2_SMOKE_TESTS      comma-separated c2_serving_smoke.py tests, empty for the default set (agreement, bench and concurrent8_steady are
                      opt-in: only when named - tp_agreement.py collect and tp_decode_bench.py, results in agreement-<profile>.json
                      and bench-<profile>.json). A four-card smoke whose profile asks for the eight-seat quad
                      (QWEN_FAST_QUAD_DRAFT_BLOCKS=2) or the two-block pre-stage (QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE=1) MUST name
                      concurrent8_steady: c2_smoke_check.py judges those levers on that test alone and fails the run without it, so this
                      reader refuses the job before any card time is spent (steady_eight_problems)
  C2_SMOKE_PARTIAL    a reason (8..200 printable characters) that exempts a four-card smoke job from that refusal: the job KNOWS it omits
                      concurrent8_steady (a measurement or fault arm whose test list is fixed by its protocol). The smoke check is
                      unchanged, so such a run still ends red on its concurrent8_steady rule: read its logs, not its conclusion. A
                      committed template that needs the key is held to it by test_c2_serving_job; one that does not never carries it
  C2_BENCH_SHAPES     the bench test's shapes, comma-separated STREAMSxPROMPT (default: tp_decode_bench's ladder)
  C2_PLATFORM_IMAGE   the thatch-serving-tt image the replay runs
  C2_BAKE_DEFAULT_PROFILE  build only: bake this profile as the image's serving default (the image ENV QWEN_C2_PROFILE) and
                      THATCH_SERVING_SESSION_CAP as its max-num-seqs (the node agent forwards neither, so the image is what a platform launch
                      serves and what a rollback moves). Default, rendered empty: nothing is baked, the default is qwen_c2_profiles.json's
                      (production's four seats). Needs the build action, C2_CARDS=quad and a four-card (P150x4), not gate_only, profile; a
                      profile with an owner_traffic_waiver (serving_c2_contract.TRAFFIC_WAIVER) is baked only once its decision reads APPROVED
  C2_DRAFTER_CANDIDATES  build only: the pinned drafter candidate ids (scripts/ci/drafter_checkpoints.json) whose bytes the image build stages beside the default
                      checkpoint (build-c2-serving-image.sh). Space- or comma-separated; every id must be a candidate of the table (the default is never staged);
                      needs the build action. Default, rendered empty: only the placeholder (drafter_checkpoint.py, docs/tp4-combined-window.md)
  C2_BOX_MINUTES      the job's BOX in minutes (1..540, at most the longest step's own timeout; default, rendered empty: none): what the job may take when it runs to its limit, the number a
                      window driver admits the job on. The workflow enforces it from inside, never by cancelling the run: the four-card smoke step
                      stops waiting for the API at the box and runs its smoke client under `timeout` for what the box has left (the container
                      is removed by the step's own trap), and the replay step runs under `timeout --signal=INT`, so its own finally removes its container; the gate step hands the gate driver min(step, job, box) as --budget-seconds, which refuses before
                      any container starts a plan list whose worst case (every arm to its docker timeout) does not fit; the prefix step hands the prefix gate
                      --box-seconds, which clips every arm's docker timeout to what the box has left and starts no arm with under five minutes left (no plan is refused)
  C2_DRAFTER_MANIFEST build only: the drafter checkpoint the image serves, a manifest under scripts/ci/references/drafter-manifests (default,
                      rendered empty: the served drafter, dedf8df6). A candidate's fixtures must be staged on the build host first
                      (drafter_stage.py); the build lays a DRAFTER_MANIFEST marker beside them and the loader refuses any other revision.
                      Needs the build action
  C2_GATE_PLAN        GATE_PLANS, comma- or space-separated, run in order (default: bringup)
  C2_GATE_LENGTHS     the matrix's prompt lengths, one per user (default, rendered empty: the S1 G4
                      ladder, whose top rung the gate lowers to what the image's profile admits)
  C2_GATE_MAX_TOKENS  the matrix's answer budget per user (default 4096)
  C2_GATE_MEMORY_PROMPT  G5's prompt length (default: the profile's largest admitted prompt)
  C2_GATE_MEMORY_USERS   G5's concurrent streams (default, rendered empty: the profile's seats; M9a runs ONE cold 253,920-token prompt on an
                      eight-seat 262k profile); never more than the profile's seats
  C2_REPLAY_PROFILE   the profile the replay's container serves (default: the source container's)
  C2_REPLAY_SERVED_MODEL  the model id the replay's /v1/models must advertise and its requests after
                      the warmup name, org/name[:tag] (default, rendered empty: c2_platform_replay.py's,
                      Qwen/Qwen3.8-27B:tt); Qwen/Qwen3.8-27B replays an image without the alias
  C2_REPLAY_BUDGET_SMOKE  '' (default) or 1: the replay also runs the image's thinking-budget smoke
                      (python3 -m serving.budget_smoke) in its container before tearing it down, up to 90 minutes, and
                      fails the job if the smoke ran and failed; needs the replay action
  C2_CARDM_HARNESS    the cardm action's harness: a .sh under optimisation/ttnn-op/<dir>/ that embeds
                      scripts/ci/qual_card.sh byte for byte and selects its board by it (required with
                      cardm), e.g. optimisation/ttnn-op/k64j/run_card_b.sh
  C2_CARDM_ARGS       its extra arguments, handed over as CARD_B_ARGS: plain words, no quotes or globs
  C2_CARDM_ENV        its extra environment, space-separated NAME=value; never QUAL_* (QUAL_CARD is card
                      M's, set by the step), ALLOW_SERVING_CARD, RESULTS or CARD_B_ARGS (the step's), nor
                      what would redirect the shell, the loader or docker (CARDM_REFUSED_ENV)
  S2 (C2-packed-any, s2-design.md W11 and 6.3; C2_GATE_PLAN may name S2_GATE_PLANS too):
  C2_GATE_PAIRS       the A/B pairs control and control-below run (default, rendered empty: the gate's
                      two, ABAB), 1..MAX_PAIRS; control times its unaudited flag-on arms against the pairs'
                      flag-off arms and adds one audited flag-on arm (control-audit) after them - its rule
                      (c2_serving_gate.CONTROL_RULE) judges the net over exactly two pairs, so any other number
                      leaves control short, never a pass
  C2_GATE_FAMILIES    control-below's (G3b) families F, each a multiple of 256 in BELOW_FAMILY_RANGE
                      (default, rendered empty: the gate's 16640,4352)
  C2_GATE_JIT         whose kernel-cache growth fails an arm: auto (default: S2 arms and plans, never warm),
                      judge (every arm but warm: M2's exact bring-up), record (none)
  C2_GATE_POLICY      strict (default) or dc-i: the user's decision D-c(i) - only if K2 had failed - under
                      which a cross-path divergence reads NOT_COMPARABLE; dc-i needs C2_GATE_POLICY_DECISION,
                      where the user's decision is recorded (a plain word: a path, URL or id)
  C2_GATE_AUDITS      extent (default: QWEN_FAST_EXTENT_AUDIT on every S2 arm but control's timing arms) or
                      all (also the prestage, pair-mask and fused-commit audits on the G4 arms); mixed (M7) runs
                      all four either way
  TP4 op profile (docs/tp4-profile.md; C2_GATE_PLAN may name OPS_GATE_PLANS, after every other plan, the twin first):
                      ops-twin, ops-trace - four-card timed profile only (C2_CARDS=quad, QWEN_FAST_TP=4, both verify
                      audits off), attribution only; ops_profile_plan says what each runs and what it writes to ops/
  C2_GATE_SALT        a cache_salt for every gate stream (sticky sessions, harness item B3; default, rendered
                      empty: none sent, the payload exactly as before): fresh (each stream its own salt, minted
                      under a key the gate mounts, so every request takes the prefix route's capture path and
                      publishes, and none can hit) or none (said explicitly: every request unsalted, which a
                      prefix profile serves with reuse off). Only on a prefix profile (the gate refuses it on
                      any other before a container starts); warm on c2-packed-prefix then warms the prefix twins
  C2_PREFIX_PLAN      PREFIX_PLANS for the prefix action (c2_prefix_gate.py), in order (default: bringup);
                      each exactness and lifecycle arm is a plan too (PREFIX_ARM_PLANS): exactness-eager
                      re-runs that arm alone. Never beside its own plan, and no plan twice (one arm, one
                      results directory). exactness-shared (the four-agent shared-block arm) applies to the
                      sticky-session profiles only; on them exactness-traced and lifecycle-tiny are
                      NOT_APPLICABLE, and a plan with no applicable arm is refused by the gate. agent-turns
                      (tp4/packed-prefix: the agent-turn replay, eight conversations of several turns) runs the prefix
                      and the control arm; agent-turns-prefix and agent-turns-baseline run one arm each (the ABAB jobs)
  C2_PREFIX_PROFILE   the prefix-reuse profile it serves (default general-prefix; must be a checkout profile)
  C2_PREFIX_BASELINE  the no-reuse profile it compares against (default general; none: timing without
                      the baseline arm)
  C2_PREFIX_AGENTS    the timing and agent-turns plans' busy-agent counts, one phase each (default 1,4,5,6;
                      the agent-turn replay on eight seats: 8)
  C2_PREFIX_KVREAD_MOUNT  '' (default) or 1: the prefix gate's containers (and its anchor probe) mount the rig's opgraft-KVR (the region-read
                      extension qwen_kv_read.so and prod-audit/model.py, the production model.py staged with the narrowed audit) over an
                      image that lacks them, namely the production base tp4-serve-10: the P1-CTL re-run on production bytes with the cheap audit.
                      Leave it empty for tp4-serve-11, which bakes both. Refused unless the prefix action runs.
  The C2_PREFIX_* keys are read only when C2_ACTIONS has prefix; otherwise their defaults are output.
  C2_TAULAB_*         the taulab action (the W-T1 tau lab, scripts/ci/c2_tau_lab.py; C2_CARDS=quad only): ONE container of
                      the image on all four cards serving C2_TAULAB_PROFILE plus two log flags, the arms sent over its API.
                      Read only when C2_ACTIONS has taulab (otherwise defaults are output).
  C2_TAULAB_PROFILE   the production profile it serves (default c2-packed-tp4; a P150x4 profile of the checkout)
  C2_TAULAB_DATA      the rig-local data directory (default, rendered empty: kwork64/taulab/data under the runner's home);
                      relative paths are under the home, no '..'; the driver READS it and never writes it
  C2_TAULAB_ARMS      the arms, space or comma separated, run in their own order (default: A1 A2 A3 A4 A5)
  C2_TAULAB_DEADLINE  the whole run in minutes, container load included (default 270, at most 540)
  C2_TAULAB_IN_FLIGHT requests in flight per independent-turn arm (default 8: the four seats and four queued; 1..16)
  C2_TAULAB_MAX_TOKENS  the answer budget of A1, A2, A4 and A5 (default, rendered empty: 2048; A3 always keeps the lanes' 2400)
  C2_TAULAB_DRAFTER_ARM  which drafter the run serves (default, rendered empty: the lab as it was): control (the served drafter),
                      b16-bf8 (the block-16 candidate), b32-bf8 (the block-32 candidate, measurement only) or dedf-bf16 (the
                      served drafter with bfloat16 projection weights). The image (C2_IMAGE_TAG) carries the drafter's fixtures;
                      a candidate arm leaves A3 out of the default arms (A3 is
                      the control arm's calibration) and refuses a list that names it
  C2_TAULAB_PAIR_CONTROL  the run id of the control arm's taulab run (digits): after the lab, drafter_pair_report.py pairs this
                      run's turns with that run's, on the runner, and holds the verdict to the pre-registered GO rule
  C2_TAULAB_COUNTERS  production's spec-decode counters, a rig-local JSON of aggregates (default: none; positions 1-3 are then
                      NOT_ESTABLISHED in the report)
  C2_TT_GRID          clamp the compute grid tt-metal exposes, as insurance for cards whose firmware no longer soft-harvests Tensix
                      columns (tensix_col_disable_count 2 -> 0, so the unharvested 13x10 core descriptor is selected instead of 11x10):
                      exactly 10,9 (11x10, today's grid, valid on a harvested and an unharvested card), 11,9 (12x10) or 12,9 (13x10),
                      or unset (default, rendered empty: the variable is not passed at all, today's behaviour). The value is the
                      container's TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE (the end coordinate of compute_with_storage; tt-metal is
                      fatal only when it exceeds the descriptor). Every step that opens a card forwards it: the fabric probe, the pair
                      and four-card smokes, the gate, prefix and tau lab drivers (--tt-grid), the replay (--tt-grid) and the cardm
                      harness (QUAL_TT_GRID, a step-owned name). A cardm harness that does not forward QUAL_TT_GRID is refused with the
                      key set, before any card time. reset and the node agent's own container (agentstart) are not covered

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import json
import os
import re
import sys

ACTIONS = ('status', 'rmi', 'agentstop', 'platform', 'unserve', 'priority', 'rescan', 'reset', 'fabric', 'cardm', 'drift', 'build', 'probe', 'smoke', 'gate',
           'prefix', 'taulab', 'replay', 'push', 'agentstart')
CARD_SETS = ('pair', 'quad')
# What a four-card job may run: the pair-shaped steps (a single-card harness on card M, the M+A smoke and replay, the
# CPU-priority measurement of the M+A container) do not apply to it, and fabric applies to nothing else.
QUAD_ACTIONS = ('status', 'rmi', 'agentstop', 'platform', 'unserve', 'rescan', 'reset', 'fabric', 'drift', 'build', 'probe', 'smoke', 'gate', 'prefix',
                'taulab', 'replay', 'push', 'agentstart')
TP4_MESH_DEVICE = 'P150x4'
# The mesh_device values of a pair profile: none (the image's P300 under upstream's four-channel p150_x2) and P300
# (general-2link: the same pair under the two-channel descriptor this cabling needs).
PAIR_MESH_DEVICES = (None, 'P300')
FABRIC_CONFIGS = ('FABRIC_1D', 'FABRIC_1D_RING')
FABRIC_PROBES = ('fabric', 'rs-tile', 'mr', 'kvread')
DRAFTER_ID = re.compile(r'[a-z0-9][a-z0-9.-]{2,63}')
GATE_PLANS = ('bringup', 'matrix', 'memory', 'lifecycle')
# S2 (s2-design.md 6.3), run on the S2 image (graft K64j) and its c2-packed profiles; c2_serving_gate.py says what
# each runs. warm and warm-off are M1 (never judged for kernel-cache growth); control and forced-cap M3-M4 (G3);
# control-below M5 (G3b); lifecycle-arrival M6's added event; mixed, short, boundaries and staggered M7-M10 (G4);
# churn M11 (G5); permuted is built but required only under decision D-c(i).
S2_GATE_PLANS = ('warm', 'warm-off', 'control', 'forced-cap', 'control-below', 'lifecycle-arrival', 'mixed', 'short',
                 'boundaries', 'staggered', 'churn', 'permuted')
# The lanes window (c2_lanes_plans): D0 and one fast lane beside standard lanes on the four cards. lanes-exact: every lane's text equals
# that user's text alone on the per-request engines, every audit on; round-timing: the packed-round times at 4, 3, 2 and 1 live users
# and D0 against the per-request engines for a lone user; lanes-timing (and -more): the ratio swept on one boot with one fast user
# beside 3, 2 and 1 standard users, both bars read.
LANES_GATE_PLANS = ('lanes-exact', 'round-timing', 'lanes-timing', 'lanes-timing-more')
# The TP4 op-level device profile (ops_profile_plan; docs/tp4-profile.md): an unprofiled twin, then tracy at op-support
# 20000 on the four-card timed profile. They profile the device and use a scratch kernel cache, so they run after every
# judged plan of a job, the twin first (check_ops_order).
OPS_GATE_PLANS = ('ops-twin', 'ops-trace')
ALL_GATE_PLANS = GATE_PLANS + S2_GATE_PLANS + LANES_GATE_PLANS + OPS_GATE_PLANS
MAX_PAIRS = 4
BELOW_FAMILY_RANGE = (4096, 16640)   # capacities the pinned (flag-off) mask admits: attention_mask_replay.py:18,26
JIT_MODES = ('auto', 'judge', 'record')
POLICIES = ('strict', 'dc-i')
AUDIT_SETS = ('extent', 'all')
# C2_GATE_SALT (sticky sessions B3): what cache_salt the gate's streams carry; unset sends none, as before.
SALT_MODES = ('none', 'fresh')
# C2_TT_GRID: the clamp for the compute grid (tt_metal/llrt/core_descriptor.cpp reads the end coordinate of compute_with_storage from this variable
# and is fatal only when it exceeds the descriptor, so 10,9 is valid on a harvested 11x10 and an unharvested 13x10 card). Exactly these values or unset.
TT_GRID_ENV = 'TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE'
TT_GRIDS = ('10,9', '11,9', '12,9')
# The name the cardm step hands its harness, which puts it into its container as TT_GRID_ENV (the QUAL_ prefix is the step's: C2_CARDM_ENV refuses it).
CARDM_TT_GRID_VARIABLE = 'QUAL_TT_GRID'
DECISION = re.compile(r'[A-Za-z0-9_.,:=/+@%#-]{3,200}')
# The prefix-reuse G1 gates (TT prefix-reuse design 2.2; c2_prefix_gate.py).
PREFIX_PLANS = ('bringup', 'exactness', 'lifecycle', 'timing', 'agent-turns', 'levern-faults', 'levern-hit', 'read-qualify')
# (arm, its plan): each exactness and lifecycle arm is a plan of its own, named as the arm, that runs
# only it, judged as inside its plan (neither plan has a cross-arm check; c2_prefix_gate.PLAN_ARMS).
# G1 v47 (run 36246961161) needed the eager arm again without the traced and audit arms' hour.
PREFIX_ARM_PLANS = (('exactness-traced', 'exactness'), ('exactness-audit', 'exactness'),
                    ('exactness-eager', 'exactness'), ('exactness-shared', 'exactness'),
                    ('lifecycle-evict', 'lifecycle'), ('lifecycle-store', 'lifecycle'),
                    ('lifecycle-tiny', 'lifecycle'),
                    # tp4/packed-prefix: the agent-turn replay's two arms, each as a job of its own (the ABAB pairs)
                    ('agent-turns-prefix', 'agent-turns'), ('agent-turns-baseline', 'agent-turns'))
# The W-T1 tau lab (c2_tau_lab.py): its arms, the production profile it serves and its time box.
TAULAB_ARMS = ('A1', 'A2', 'A3', 'A4', 'A5')
TAULAB_PROFILE = 'c2-packed-tp4'
TAULAB_DRAFTER_ARMS = ('control', 'b16-bf8', 'b32-bf8', 'dedf-bf16')
DEFAULT_DRAFTER_MANIFEST = 'dedf8df6'
DRAFTER_MANIFEST_NAME = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}')
TAULAB_DEADLINE_MINUTES = 270
MAX_TAULAB_DEADLINE_MINUTES = 540
TAULAB_IN_FLIGHT = 8
MAX_TAULAB_IN_FLIGHT = 16
# A rig-local path: plain characters, relative to the runner's home or absolute, never a '..' component or a leading '-'.
PLAIN_PATH = re.compile(r'(?!-)[A-Za-z0-9_./-]{1,200}')
PREFIX_PROFILE = 'general-prefix'
PREFIX_BASELINE = 'general'
PREFIX_AGENTS = (1, 4, 5, 6)
# S1's G4 ladder (c2-serve-for-real-plan 2.2, gate table row G4 part 1): both sides of every page and
# chunk boundary the fast path has (2048 = the draft window and the prefill chunk), a short prompt
# far below any of them, and long ones up to the ~123k prompt cap.
LADDER = (60, 255, 2047, 2048, 2049, 4096, 32768, 60000, 120000)
DEFAULT_MAX_TOKENS = 4096
PROFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen_c2_profiles.json')
TAG = re.compile(r'[0-9a-z.-]{3,40}')
PLATFORM_IMAGE = re.compile(r'[0-9a-z.:/@_-]{10,200}')
PROFILE_NAME = re.compile(r'[a-z0-9_-]{1,64}')
# A Hugging Face style id, org/name, and an optional deployment tag: Qwen/Qwen3.8-27B, Qwen/Qwen3.8-27B:tt.
MODEL_ID = re.compile(r'[0-9A-Za-z._-]{1,96}/[0-9A-Za-z._-]{1,96}(?::[0-9A-Za-z._-]{1,40})?')

HERE = os.path.dirname(os.path.abspath(__file__))
# The checkout this file is in: the workflow runs it from there, and the harness is read from it.
ROOT = os.path.dirname(os.path.dirname(HERE))
QUAL_CARD_LIBRARY = os.path.join(HERE, 'qual_card.sh')
QUAL_CARD_BEGIN, QUAL_CARD_END = '# >>> qual_card.sh', '# <<< qual_card.sh'
# The cardm action (card B is reserved for other work, so the single-card qualification harnesses run on card
# M): a .sh one or more directories under optimisation/ttnn-op, no component starting with '.' (so no '..').
CARDM_HARNESS = re.compile(r'optimisation/ttnn-op/(?:[A-Za-z0-9_][A-Za-z0-9_.-]*/)+[A-Za-z0-9_][A-Za-z0-9_.-]*\.sh')
ENV_NAME = re.compile(r'[A-Z][A-Z0-9_]*')
# A harness argument or environment value: paths, digests, numbers, lists. No quote, glob, space or shell
# character: the harness word-splits CARD_B_ARGS unquoted, and nothing here is ever evaluated.
PLAIN_WORD = re.compile(r'[A-Za-z0-9_.,:=/+@%-]+')
# The step sets these itself (QUAL_CARD to card M's board id); the job file may not.
CARDM_STEP_ENV = ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'RESULTS', 'CARD_B_ARGS')
# Nor what would redirect the harness's shell, its loader, sudo or docker (another daemon has other
# containers: the harness's holder check would look at the wrong one), or its default paths.
CARDM_REFUSED_ENV = CARDM_STEP_ENV + ('PATH', 'HOME', 'ENV', 'SHELLOPTS', 'IFS', 'PS4', 'CDPATH', 'GLOBIGNORE')
CARDM_REFUSED_PREFIXES = ('QUAL_', 'BASH', 'LD_', 'DOCKER_', 'SUDO_')


# The smoke test c2_smoke_check.py judges the eight-seat levers on, and the two profile flags that need it. c2_smoke_check's rule (blocks_problems,
# hostgap_problems): a profile that asks for the eight-seat quad (QWEN_FAST_QUAD_DRAFT_BLOCKS=2) or the two-block pre-stage
# (QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE=1) fails the four-card smoke unless concurrent8_steady ran and did not error. concurrent8_steady is opt-in
# in c2_serving_smoke.py (it runs only when named), so an empty or short C2_SMOKE_TESTS lets a healthy run be marked failed after its card time
# (2026-10-09: six runs). test_c2_serving_job holds these constants and the profile rule against c2_smoke_check itself.
STEADY_EIGHT_TEST = 'concurrent8_steady'
SMOKE_PARTIAL = re.compile(r'[ -~]{8,200}')
STEADY_EIGHT_FLAGS = (('QWEN_FAST_QUAD_DRAFT_BLOCKS', '2', 'the eight-seat quad'),
                      ('QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE', '1', 'the two-block pre-stage'))


# Tags rmi must never delete. Production's base is tt-vllm:qwen38-c2-tp4-serve-7; the P8 base (qwen-fast-serving:ci-*) is a
# different repository, so no tag here can ever name it. The named tags are the lineages production has run on, the prefixes
# cover every other serve-N, and anything with 'prod' in it is refused whatever else it says.
PROTECTED = ('serve-7', 'tp4-serve-7', 'tp4-serve-6', 'tp4-serve-3', 'tp4-serve-2')
PROTECTED_PREFIXES = ('serve-', 'tp4-serve-')
PROTECTED_WORDS = ('prod',)
RMI_TAG = re.compile(r'[a-z0-9][a-z0-9.-]{1,60}')


class JobError(ValueError):
    """A job file the workflow must not act on."""


def parse_env(text):
    """KEY=VALUE lines, blank lines and # comments skipped, whitespace around both stripped."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            key, _, value = line.partition('=')
            values[key.strip()] = value.strip()
    return values


def profile_names(path=PROFILES):
    with open(path, encoding='utf-8') as handle:
        return sorted(json.load(handle)['profiles'])


def profile_meshes(path=PROFILES):
    """{profile: its mesh_device, or None}: what a profile opens (serving_c2_contract.mesh_of reads the same key)."""
    with open(path, encoding='utf-8') as handle:
        return dict((name, body.get('mesh_device')) for name, body in json.load(handle)['profiles'].items())


def refuse_fabric_with_serving(actions):
    """The fabric probe closes the mesh it opened, and a second open in one job is what the ethernet-core teardown
    wedge punishes: it runs in a job of its own, never beside a step that serves or gates."""
    beside = sorted(set(actions) & set(('smoke', 'gate', 'prefix', 'taulab')))
    if 'fabric' in actions and beside:
        raise JobError('C2_ACTIONS has fabric with %s: the probe closes the mesh and a second open in one job wedges '
                       'the ethernet cores; run the probe in its own job, reset first' % ', '.join(beside))


def profile_envs(path=PROFILES):
    """{profile: its env dict}: the served environment c2_smoke_check.py judges the log against (its profile_env reads the same key)."""
    with open(path, encoding='utf-8') as handle:
        return dict((name, body.get('env') or {}) for name, body in json.load(handle)['profiles'].items())


def steady_eight_problems(values, actions, cards, profile, envs):
    """[reason] for a four-card smoke job whose profile c2_smoke_check.py will judge on concurrent8_steady while C2_SMOKE_TESTS leaves it out.
    The smoke check fails such a run whatever it logged ('the profile asks for the eight-seat quad and the smoke did not run concurrent8_steady
    ... the blocks were not judged', and the same for the two-block pre-stage); the card time is spent before anything says so. The test list is
    split as c2_serving_smoke.py splits it (on commas, nothing trimmed), and empty means its default set, which leaves the opt-in test out."""
    if 'smoke' not in actions or cards != 'quad':
        return []
    env = envs.get(profile) or {}
    asked = [label for flag, value, label in STEADY_EIGHT_FLAGS if env.get(flag) == value]
    if not asked:
        return []
    tests = [name for name in values.get('C2_SMOKE_TESTS', '').split(',') if name]
    if STEADY_EIGHT_TEST in tests:
        return []
    partial = values.get('C2_SMOKE_PARTIAL', '')
    if partial and not SMOKE_PARTIAL.fullmatch(partial):
        return ['C2_SMOKE_PARTIAL is the reason this job omits %s: 8..200 printable characters, got %r' % (STEADY_EIGHT_TEST, partial)]
    if partial:
        return []
    return ['C2_SMOKE_TESTS does not name %s (it names %s): profile %s asks for %s, and the smoke check judges it on that test alone, so the run '
            'would fail after its card time; add %s (comma-separated, no spaces), or state in C2_SMOKE_PARTIAL why it is left out '
            '(the run then still ends red on that rule)'
            % (STEADY_EIGHT_TEST, ', '.join(tests) if tests else 'nothing: the default set leaves it out', profile, ' and '.join(asked),
               STEADY_EIGHT_TEST)]


def read_cards(values, actions, profile_of, named):
    """C2_CARDS, refused when an action or a profile of the job does not fit the card set. `named`: the profiles the
    job's steps serve, (key, name) pairs; profile_of maps a profile to its mesh_device."""
    cards = values.get('C2_CARDS') or 'pair'
    if cards not in CARD_SETS:
        raise JobError('C2_CARDS must be one of %s, got %r' % (', '.join(CARD_SETS), cards))
    if cards == 'pair' and 'fabric' in actions:
        raise JobError('C2_ACTIONS has fabric: the four-card fabric probe needs C2_CARDS=quad')
    refuse_fabric_with_serving(actions)
    if cards == 'quad':
        wrong = sorted(set(actions) - set(QUAD_ACTIONS))
        if wrong:
            raise JobError('C2_CARDS=quad: %s are pair-shaped steps (M+A, card M); quad takes %s' % (
                ', '.join(wrong), ' '.join(QUAD_ACTIONS)))
    if values.get('C2_FABRIC') and cards != 'quad':
        raise JobError('C2_FABRIC needs C2_CARDS=quad')
    if cards == 'quad' and 'replay' in actions and not values.get('C2_REPLAY_PROFILE'):
        raise JobError('C2_CARDS=quad with replay needs C2_REPLAY_PROFILE: the source container\'s profile is a pair '
                       'profile, and the four-card mesh is the profile\'s (mesh_device P150x4)')
    for key, name in named:
        if not name or name == 'none' or name not in profile_of:
            continue
        if (profile_of[name] != TP4_MESH_DEVICE) if cards == 'quad' else (profile_of[name] not in PAIR_MESH_DEVICES):
            raise JobError('%s %s opens %s, but C2_CARDS=%s gives %s' % (
                key, name, 'the four-card (1, 4) mesh' if profile_of[name] == TP4_MESH_DEVICE else 'the (1, 2) pair', cards,
                'all four cards' if cards == 'quad' else 'cards M and A'))
    return cards


BENCH_SHAPES = re.compile(r'[0-9]+x[0-9]+(?:,[0-9]+x[0-9]+)*')


def bench_shapes(values):
    shapes = values.get('C2_BENCH_SHAPES', '')
    if shapes and not BENCH_SHAPES.fullmatch(shapes):
        raise JobError('C2_BENCH_SHAPES is STREAMSxPROMPT shapes, comma-separated (4x4096,1x131072), got %r' % shapes)
    return shapes


def fabric_config(values):
    fabric = values.get('C2_FABRIC') or FABRIC_CONFIGS[0]
    if fabric not in FABRIC_CONFIGS:
        raise JobError('C2_FABRIC must be one of %s, got %r' % (', '.join(FABRIC_CONFIGS), fabric))
    return fabric


def fabric_probe(values, actions):
    probe = values.get('C2_FABRIC_PROBE') or FABRIC_PROBES[0]
    if probe not in FABRIC_PROBES:
        raise JobError('C2_FABRIC_PROBE must be one of %s, got %r' % (', '.join(FABRIC_PROBES), probe))
    if values.get('C2_FABRIC_PROBE') and 'fabric' not in actions:
        raise JobError('C2_FABRIC_PROBE names what the fabric action runs; C2_ACTIONS has no fabric')
    return probe


def positive_int(name, text):
    try:
        value = int(text)
    except ValueError:
        raise JobError('%s must be a positive integer, got %r' % (name, text))
    if value < 1:
        raise JobError('%s must be a positive integer, got %r' % (name, text))
    return value


def split_list(text):
    return [part for part in re.split(r'[\s,]+', text or '') if part]


def embeds_qual_card(text, library):
    """Whether a harness carries the canonical qual_card.sh block once, byte for byte, and selects its board
    with it right after (test_qual_card's rule for every embedding harness): then QUAL_CARD, which the cardm
    step sets to card M, is the board it resolves, holder-checks and maps."""
    begins = [m.start() for m in re.finditer('^' + re.escape(QUAL_CARD_BEGIN), text, flags=re.M)]
    ends = [m.start() for m in re.finditer('^' + re.escape(QUAL_CARD_END), text, flags=re.M)]
    if len(begins) != 1 or len(ends) != 1 or ends[0] < begins[0] or '\n' not in text[ends[0]:]:
        return False
    end = text.index('\n', ends[0]) + 1
    return text[begins[0]:end] == library and text[end:].startswith('qual_card_select\n')


def read_cardm(values, requested, root=ROOT, library=QUAL_CARD_LIBRARY, tt_grid=''):
    """(harness, args, env) for the cardm step, or JobError. Checked whenever set, required with cardm. With a C2_TT_GRID clamp (`tt_grid`) the
    harness must forward the step's QUAL_TT_GRID into its container: one that does not would run the card at the unclamped grid while the job
    says it is clamped."""
    harness = values.get('C2_CARDM_HARNESS', '')
    if not harness:
        if requested:
            raise JobError('C2_ACTIONS has cardm: C2_CARDM_HARNESS must name its harness')
    else:
        if not CARDM_HARNESS.fullmatch(harness):
            raise JobError('C2_CARDM_HARNESS must be a .sh under optimisation/ttnn-op/<dir>/, got %r' % harness)
        path = os.path.join(root, *harness.split('/'))
        ops = os.path.realpath(os.path.join(root, 'optimisation', 'ttnn-op'))
        if not os.path.realpath(path).startswith(ops + os.sep) or not os.path.isfile(path):
            raise JobError('C2_CARDM_HARNESS %s is not a file of this checkout under optimisation/ttnn-op' % harness)
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        with open(library, encoding='utf-8') as handle:
            canonical = handle.read()
        if not embeds_qual_card(text, canonical):
            raise JobError('C2_CARDM_HARNESS %s does not select its board by the canonical qual_card.sh block; '
                           'refusing to run it on card M' % harness)
        if tt_grid and CARDM_TT_GRID_VARIABLE not in text:
            raise JobError('C2_TT_GRID is set but C2_CARDM_HARNESS %s does not forward %s into its container; it would open card M '
                           'unclamped: unset C2_TT_GRID or use a harness that forwards it' % (harness, CARDM_TT_GRID_VARIABLE))
    args = values.get('C2_CARDM_ARGS', '').split()
    for word in args:
        if not PLAIN_WORD.fullmatch(word):
            raise JobError('C2_CARDM_ARGS words are plain (%s), got %r' % (PLAIN_WORD.pattern, word))
    pairs = values.get('C2_CARDM_ENV', '').split()
    names = []
    for pair in pairs:
        name, sep, value = pair.partition('=')
        if not sep or not ENV_NAME.fullmatch(name) or not (value == '' or PLAIN_WORD.fullmatch(value)):
            raise JobError('C2_CARDM_ENV entries are NAME=value with a plain value, got %r' % pair)
        if name in CARDM_REFUSED_ENV or name.startswith(CARDM_REFUSED_PREFIXES):
            raise JobError('C2_CARDM_ENV may not set %s: the cardm step runs card M only, and sets %s itself'
                           % (name, ', '.join(CARDM_STEP_ENV)))
        if name in names:
            raise JobError('C2_CARDM_ENV sets %s twice' % name)
        names.append(name)
    return harness, ' '.join(args), ' '.join(pairs)


def read_tt_grid(values):
    """C2_TT_GRID (module docstring) as the string to hand the container, '' when unset, or JobError for anything but exactly one of TT_GRIDS."""
    value = values.get('C2_TT_GRID', '')
    if value and value not in TT_GRIDS:
        raise JobError('C2_TT_GRID must be one of %s or unset, got %r' % (', '.join(TT_GRIDS), value))
    return value


def tt_grid_arguments(value):
    """The `docker run` arguments that put the clamp into a container: ['-e', 'TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=10,9'], or [] for
    None or '' (unset: the container is exactly what it was). A value outside TT_GRIDS is a JobError, never a guess."""
    if not value:
        return []
    if value not in TT_GRIDS:
        raise JobError('the tt grid must be one of %s or unset, got %r' % (', '.join(TT_GRIDS), value))
    return ['-e', '%s=%s' % (TT_GRID_ENV, value)]


def read_rmi_tags(values, actions):
    """C2_RMI_TAGS as a space-joined string: required with the rmi action, refused without it, every tag well formed
    and none of them protected."""
    tags = split_list(values.get('C2_RMI_TAGS', ''))
    if 'rmi' not in actions:
        if tags:
            raise JobError('C2_RMI_TAGS is set but C2_ACTIONS has no rmi')
        return ''
    if not tags:
        raise JobError('C2_ACTIONS has rmi: C2_RMI_TAGS must name the image tags to delete')
    for tag in tags:
        if not RMI_TAG.fullmatch(tag):
            raise JobError('C2_RMI_TAGS: %r does not match %s' % (tag, RMI_TAG.pattern))
        if tag in PROTECTED or tag.startswith(PROTECTED_PREFIXES) or any(word in tag for word in PROTECTED_WORDS):
            raise JobError('C2_RMI_TAGS: %r is protected (a production lineage: %s, any serve- or tp4-serve- tag, '
                           'anything containing prod); rmi refuses it' % (tag, ' '.join(PROTECTED)))
    if len(set(tags)) != len(tags):
        raise JobError('C2_RMI_TAGS names a tag twice')
    return ' '.join(tags)


def read_job(values, profiles, root=ROOT, meshes=None, envs=None):
    """The workflow outputs for a parsed job file, or JobError. `meshes`: profile_meshes() and `envs`: profile_envs() (the checkout's when not given)."""
    if values.get('C2_SUPERSEDED_BY'):
        raise JobError('this template belongs to a superseded pack (C2_SUPERSEDED_BY=%s): its rules are void, use that pack' % values['C2_SUPERSEDED_BY'])
    actions = split_list(values.get('C2_ACTIONS', 'status')) or ['status']
    unknown = sorted(set(actions) - set(ACTIONS))
    if unknown:
        raise JobError('C2_ACTIONS: unknown %s (known: %s)' % (', '.join(unknown), ' '.join(ACTIONS)))
    order = [ACTIONS.index(action) for action in actions]
    if order != sorted(order):
        raise JobError('C2_ACTIONS must follow the workflow\'s order: %s' % ' '.join(ACTIONS))
    rmi_tags = read_rmi_tags(values, actions)
    tag = values.get('C2_IMAGE_TAG', '')
    if not TAG.fullmatch(tag):
        raise JobError('C2_IMAGE_TAG must match %s, got %r' % (TAG.pattern, tag))
    profile = values.get('C2_PROFILE') or 'general'
    if profile not in profiles:
        raise JobError('C2_PROFILE %r is not a profile of qwen_c2_profiles.json (%s)' % (profile, ', '.join(profiles)))
    platform_image = values.get('C2_PLATFORM_IMAGE', '')
    if platform_image and not PLATFORM_IMAGE.fullmatch(platform_image):
        raise JobError('C2_PLATFORM_IMAGE must match %s' % PLATFORM_IMAGE.pattern)
    plans = split_list(values.get('C2_GATE_PLAN', 'bringup')) or ['bringup']
    unknown = sorted(set(plans) - set(ALL_GATE_PLANS))
    if unknown:
        raise JobError('C2_GATE_PLAN: unknown %s (known: %s)' % (', '.join(unknown), ' '.join(ALL_GATE_PLANS)))
    check_ops_order(plans)
    s2 = read_s2_gate(values)
    lengths_text = values.get('C2_GATE_LENGTHS', '')
    lengths = [positive_int('C2_GATE_LENGTHS', part) for part in split_list(lengths_text)]
    max_tokens = positive_int('C2_GATE_MAX_TOKENS', values['C2_GATE_MAX_TOKENS']) \
        if values.get('C2_GATE_MAX_TOKENS') else DEFAULT_MAX_TOKENS
    memory_prompt = positive_int('C2_GATE_MEMORY_PROMPT', values['C2_GATE_MEMORY_PROMPT']) \
        if values.get('C2_GATE_MEMORY_PROMPT') else ''
    memory_users = positive_int('C2_GATE_MEMORY_USERS', values['C2_GATE_MEMORY_USERS'])         if values.get('C2_GATE_MEMORY_USERS') else ''
    replay_profile = values.get('C2_REPLAY_PROFILE', '')
    if replay_profile and replay_profile not in profiles:
        raise JobError('C2_REPLAY_PROFILE %r is not a profile of qwen_c2_profiles.json' % replay_profile)
    replay_served_model = values.get('C2_REPLAY_SERVED_MODEL', '')
    if replay_served_model and not MODEL_ID.fullmatch(replay_served_model):
        raise JobError('C2_REPLAY_SERVED_MODEL must match %s, got %r' % (MODEL_ID.pattern, replay_served_model))
    budget_smoke = values.get('C2_REPLAY_BUDGET_SMOKE', '')
    if budget_smoke not in ('', '1'):
        raise JobError('C2_REPLAY_BUDGET_SMOKE must be empty or 1, got %r' % budget_smoke)
    if budget_smoke and 'replay' not in actions:
        raise JobError('C2_REPLAY_BUDGET_SMOKE runs inside the replay\'s container; C2_ACTIONS has no replay')
    prefix = read_prefix(values, profiles, 'prefix' in actions)
    taulab = read_taulab(values, profiles, 'taulab' in actions)
    if meshes is None:
        meshes = profile_meshes()
    named = [('C2_PROFILE', profile if set(actions) & set(('smoke', 'gate')) else ''),
             ('C2_REPLAY_PROFILE', replay_profile if 'replay' in actions else '')]
    if 'prefix' in actions:
        named += [('C2_PREFIX_PROFILE', prefix['prefix_profile']), ('C2_PREFIX_BASELINE', prefix['prefix_baseline'])]
    if 'taulab' in actions:
        named += [('C2_TAULAB_PROFILE', taulab['taulab_profile'])]
        if (values.get('C2_CARDS') or 'pair') != 'quad':
            raise JobError('C2_ACTIONS has taulab: the tau lab serves the four-card (1, 4) mesh and needs C2_CARDS=quad')
    cards = read_cards(values, actions, meshes, named)
    if 'smoke' in actions and cards == 'quad':
        problems = steady_eight_problems(values, actions, cards, profile, profile_envs() if envs is None else envs)
        if problems:
            raise JobError(problems[0])
    bake_profile = read_bake(values, actions, cards, root)
    drafter_manifest = read_drafter_manifest(values, actions, root)
    if drafter_manifest and 'push' in actions:
        raise JobError('C2_DRAFTER_MANIFEST names a candidate drafter that is not qualified: its image is not pushed')
    tt_grid = read_tt_grid(values)
    cardm_harness, cardm_args, cardm_env = read_cardm(values, 'cardm' in actions, root=root, tt_grid=tt_grid)
    drafter_candidates = read_drafter_candidates(values, actions)
    box_minutes = read_box(values, actions)
    outputs = dict(cards=cards, fabric=fabric_config(values), fabric_probe=fabric_probe(values, actions), bench_shapes=bench_shapes(values), actions=' '.join(actions), rmi_tags=rmi_tags, tag=tag, profile=profile, tests=values.get('C2_SMOKE_TESTS', ''),
                   platform_image=platform_image, gate_plan=','.join(plans),
                   gate_lengths=','.join(str(length) for length in lengths), gate_max_tokens=str(max_tokens),
                   gate_memory_prompt=str(memory_prompt), gate_memory_users=str(memory_users), replay_profile=replay_profile,
                   replay_served_model=replay_served_model, replay_budget_smoke=budget_smoke, cardm_harness=cardm_harness, cardm_args=cardm_args,
                   cardm_env=cardm_env, bake_default_profile=bake_profile, drafter_candidates=drafter_candidates, box_minutes=box_minutes, drafter_manifest=drafter_manifest, tt_grid=tt_grid)
    outputs.update(s2)
    outputs.update(prefix)
    outputs.update(taulab)
    return outputs


def check_ops_order(plans):
    """Refuse a plan list that runs an ops plan before a judged one (the profiled arm reserves device DRAM, compiles into
    a scratch cache and may end as INFRA), that runs ops-trace without ops-twin before it (the twin is the text
    reference), or that names one twice (one arm directory per ops plan)."""
    seen = None
    for plan in plans:
        if plan in OPS_GATE_PLANS:
            seen = seen or plan
        elif seen:
            raise JobError('C2_GATE_PLAN: %s runs after %s: the ops plans go after every judged plan' % (plan, seen))
    named = [plan for plan in plans if plan in OPS_GATE_PLANS]
    if len(set(named)) != len(named):
        raise JobError('C2_GATE_PLAN names an ops plan twice: one arm directory per ops plan')
    if 'ops-trace' in named and named[0] != 'ops-twin':
        raise JobError('C2_GATE_PLAN: ops-trace needs ops-twin before it (its texts are held against the twin texts)')
    return plans


def read_drafter_candidates(values, actions, table=None):
    """C2_DRAFTER_CANDIDATES (module docstring) as a space-separated string of pinned candidate ids, '' when unset, or JobError: an id the table does not pin,
    the default (its bytes are the image's own), a malformed or repeated id, or no build action."""
    text = values.get('C2_DRAFTER_CANDIDATES', '').strip()
    if not text:
        return ''
    if 'build' not in actions:
        raise JobError('C2_DRAFTER_CANDIDATES is staged at build time: C2_ACTIONS has no build')
    if table is None:
        import drafter_checkpoint
        table = drafter_checkpoint.load()
    ids = split_list(text)
    for name in ids:
        if not DRAFTER_ID.fullmatch(name):
            raise JobError('C2_DRAFTER_CANDIDATES: %r is not a plain lowercase id (%s)' % (name, DRAFTER_ID.pattern))
        if name == table['default']:
            raise JobError('C2_DRAFTER_CANDIDATES: %s is the default checkpoint, baked already' % name)
        if name not in table['checkpoints']:
            raise JobError('C2_DRAFTER_CANDIDATES: %s is not a pinned candidate of drafter_checkpoints.json (%s)' % (
                name, ', '.join(sorted(set(table['checkpoints']) - {table['default']})) or 'none yet'))
    if len(set(ids)) != len(ids):
        raise JobError('C2_DRAFTER_CANDIDATES names an id twice')
    return ' '.join(ids)


BOX_MAX_MINUTES = 540
# timeout-minutes of the workflow's smoke, gate and prefix steps (pinned by test_c2_serving_job against the workflow file).
STEP_MINUTES = {'smoke': 210, 'gate': 380, 'prefix': 380, 'replay': 180, 'fabric': 45}


def read_box(values, actions):
    """C2_BOX_MINUTES (module docstring) as a string of whole minutes, '' when unset, or JobError. Only the steps that run for a while take a
    box: a job with none of smoke, gate, prefix is refused one (a status or a reset has no box to enforce)."""
    text = values.get('C2_BOX_MINUTES', '').strip()
    if not text:
        return ''
    minutes = positive_int('C2_BOX_MINUTES', text)
    if minutes > BOX_MAX_MINUTES:
        raise JobError('C2_BOX_MINUTES must be at most %d (the 9 h cap of a window), got %d' % (BOX_MAX_MINUTES, minutes))
    if not set(actions) & set(STEP_MINUTES):
        raise JobError('C2_BOX_MINUTES is enforced by the smoke, gate, prefix, replay and fabric steps: C2_ACTIONS has none of them')
    # The workflow's own step timeouts (smoke 210 min, gate and prefix 380) are the hard ceiling the box sits under: a larger box would be cut by the step, not by the box.
    ceiling = max(STEP_MINUTES[action] for action in actions if action in STEP_MINUTES)
    if minutes > ceiling:
        raise JobError('C2_BOX_MINUTES %d is above the %d min step timeout of the job longest step (%s)' % (
            minutes, ceiling, ', '.join('%s %d' % (action, STEP_MINUTES[action]) for action in actions if action in STEP_MINUTES)))
    return str(minutes)


def read_drafter_manifest(values, actions, root=ROOT):
    """C2_DRAFTER_MANIFEST (module docstring), '' when unset or the default, or JobError: a name that is a committed manifest."""
    name = values.get('C2_DRAFTER_MANIFEST', '')
    if not name or name == DEFAULT_DRAFTER_MANIFEST:
        if name and 'build' not in actions:
            raise JobError('C2_DRAFTER_MANIFEST is applied at build time: C2_ACTIONS has no build')
        return ''
    if not DRAFTER_MANIFEST_NAME.fullmatch(name):
        raise JobError('C2_DRAFTER_MANIFEST must match %s, got %r' % (DRAFTER_MANIFEST_NAME.pattern, name))
    if 'build' not in actions:
        raise JobError('C2_DRAFTER_MANIFEST is applied at build time: C2_ACTIONS has no build')
    if not os.path.isfile(os.path.join(root, 'scripts', 'ci', 'references', 'drafter-manifests', name + '.json')):
        raise JobError('C2_DRAFTER_MANIFEST %r is not a manifest under scripts/ci/references/drafter-manifests' % name)
    return name


def read_bake(values, actions, cards, root=ROOT):
    """C2_BAKE_DEFAULT_PROFILE (module docstring), '' when unset, or JobError. Read against the profiles file of the checkout
    the job runs from: the build script reads the same file from the staged context."""
    name = values.get('C2_BAKE_DEFAULT_PROFILE', '')
    if not name:
        return ''
    if not PROFILE_NAME.fullmatch(name):
        raise JobError('C2_BAKE_DEFAULT_PROFILE must match %s, got %r' % (PROFILE_NAME.pattern, name))
    if 'build' not in actions:
        raise JobError('C2_BAKE_DEFAULT_PROFILE is baked at build time: C2_ACTIONS has no build')
    if cards != 'quad':
        raise JobError('C2_BAKE_DEFAULT_PROFILE is a four-card serving default: it needs C2_CARDS=quad')
    with open(os.path.join(root, 'scripts', 'ci', 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
        entry = json.load(handle)['profiles'].get(name)
    if entry is None:
        raise JobError('C2_BAKE_DEFAULT_PROFILE %r is not a profile of qwen_c2_profiles.json' % name)
    if entry.get('gate_only') is True or entry.get('mesh_device') != TP4_MESH_DEVICE:
        raise JobError('C2_BAKE_DEFAULT_PROFILE %s is not a four-card serving profile (%s, not gate_only): a gate arm is never '
                       'an image default' % (name, TP4_MESH_DEVICE))
    if bake_waiver_pending(entry):
        raise JobError('C2_BAKE_DEFAULT_PROFILE %s carries an %s whose decision is not APPROVED: the waiver is a request until the commit that is '
                       'built records the owner\'s approval (docs/tp4-ship-ln-w2-er.md)' % (name, BAKE_WAIVER_FIELD))
    return name


# serving_c2_contract.TRAFFIC_WAIVER and its decision words (this module stays stdlib only and does not import the contract; a test holds the two equal).
BAKE_WAIVER_FIELD = 'owner_traffic_waiver'


def bake_waiver_pending(entry):
    """Whether a profile entry carries an owner traffic waiver whose decision does not start with APPROVED (absent: False)."""
    waiver = entry.get(BAKE_WAIVER_FIELD)
    if waiver is None:
        return False
    decision = waiver.get('decision') if isinstance(waiver, dict) else None
    if not isinstance(decision, str) or not decision.strip():
        return True
    return decision.split(None, 1)[0].rstrip(':') != 'APPROVED'


def below_family(name, text):
    """One G3b family: a multiple of 256 inside BELOW_FAMILY_RANGE, or JobError."""
    family = positive_int(name, text)
    if family % 256 or not BELOW_FAMILY_RANGE[0] <= family <= BELOW_FAMILY_RANGE[1]:
        raise JobError('%s: %d is not a family the flag-off block serves (a multiple of 256 in %d..%d)'
                       % (name, family, BELOW_FAMILY_RANGE[0], BELOW_FAMILY_RANGE[1]))
    return family


def read_s2_gate(values):
    """The S2 gate keys (module docstring) as workflow outputs, each rendered empty when unset (the gate's
    own default), or JobError."""
    pairs = values.get('C2_GATE_PAIRS', '')
    if pairs:
        count = positive_int('C2_GATE_PAIRS', pairs)
        if count > MAX_PAIRS:
            raise JobError('C2_GATE_PAIRS: at most %d pairs, got %d' % (MAX_PAIRS, count))
        pairs = str(count)
    families = [below_family('C2_GATE_FAMILIES', part) for part in split_list(values.get('C2_GATE_FAMILIES', ''))]
    if len(set(families)) != len(families):
        raise JobError('C2_GATE_FAMILIES names a family twice')
    jit = values.get('C2_GATE_JIT', '')
    if jit and jit not in JIT_MODES:
        raise JobError('C2_GATE_JIT must be one of %s, got %r' % (', '.join(JIT_MODES), jit))
    policy = values.get('C2_GATE_POLICY', '')
    if policy and policy not in POLICIES:
        raise JobError('C2_GATE_POLICY must be one of %s, got %r' % (', '.join(POLICIES), policy))
    decision = values.get('C2_GATE_POLICY_DECISION', '')
    if decision and not DECISION.fullmatch(decision):
        raise JobError('C2_GATE_POLICY_DECISION must match %s' % DECISION.pattern)
    if policy == 'dc-i' and not decision:
        raise JobError('C2_GATE_POLICY=dc-i needs C2_GATE_POLICY_DECISION: only the user\'s D-c decision relaxes the '
                       'strict policy, and the gate records where it is')
    audits = values.get('C2_GATE_AUDITS', '')
    if audits and audits not in AUDIT_SETS:
        raise JobError('C2_GATE_AUDITS must be one of %s, got %r' % (', '.join(AUDIT_SETS), audits))
    salt = values.get('C2_GATE_SALT', '')
    if salt and salt not in SALT_MODES:
        raise JobError('C2_GATE_SALT must be one of %s, got %r' % (', '.join(SALT_MODES), salt))
    return dict(gate_pairs=pairs, gate_families=','.join(str(family) for family in families), gate_jit=jit,
                gate_policy=policy, gate_policy_decision=decision, gate_audits=audits, gate_salt=salt)


def read_prefix(values, profiles, running):
    """The prefix action's outputs. Its keys are checked only when the action runs: a job that does
    not run it (an exact or c2 run) is never refused over them, and gets the defaults. The profile
    must then be one the checkout defines (general-prefix lands in qwen_c2_profiles.json on another
    track, and the image's own profiles are what c2_prefix_gate.py finally checks)."""
    if not running:
        if values.get('C2_PREFIX_KVREAD_MOUNT'):
            raise JobError('C2_PREFIX_KVREAD_MOUNT mounts the region-read graft into the prefix gate: C2_ACTIONS has no prefix')
        return dict(prefix_plan='bringup', prefix_profile=PREFIX_PROFILE, prefix_baseline=PREFIX_BASELINE,
                    prefix_agents=','.join(str(count) for count in PREFIX_AGENTS), prefix_kvread_mount='')
    plans = split_list(values.get('C2_PREFIX_PLAN', 'bringup')) or ['bringup']
    known = PREFIX_PLANS + tuple(arm for arm, _ in PREFIX_ARM_PLANS)
    unknown = sorted(set(plans) - set(known))
    if unknown:
        raise JobError('C2_PREFIX_PLAN: unknown %s (known: %s)' % (', '.join(unknown), ' '.join(known)))
    parent = dict(PREFIX_ARM_PLANS)
    twice = sorted(set(plan for plan in plans if plans.count(plan) > 1 or parent.get(plan) in plans))
    if twice:
        raise JobError('C2_PREFIX_PLAN: %s would run an arm twice into one results directory (a plan named twice, '
                       'or an arm beside its own plan)' % ', '.join(twice))
    profile = values.get('C2_PREFIX_PROFILE') or PREFIX_PROFILE
    if profile not in profiles:
        raise JobError('C2_PREFIX_PROFILE %r is not a profile of qwen_c2_profiles.json (%s)' % (profile, ', '.join(profiles)))
    baseline = values.get('C2_PREFIX_BASELINE') or PREFIX_BASELINE
    if baseline != 'none' and baseline not in profiles:
        raise JobError('C2_PREFIX_BASELINE %r is not a profile of qwen_c2_profiles.json' % baseline)
    if baseline == 'none' and 'bringup' in plans:
        raise JobError('C2_PREFIX_BASELINE none: the bringup plan compares against a baseline profile')
    agents_text = values.get('C2_PREFIX_AGENTS', '')
    agents = [positive_int('C2_PREFIX_AGENTS', part) for part in split_list(agents_text)] or list(PREFIX_AGENTS)
    mount = values.get('C2_PREFIX_KVREAD_MOUNT', '')
    if mount not in ('', '1'):
        raise JobError('C2_PREFIX_KVREAD_MOUNT must be empty or 1, got %r' % mount)
    return dict(prefix_plan=','.join(plans), prefix_profile=profile, prefix_baseline=baseline,
                prefix_agents=','.join(str(count) for count in agents), prefix_kvread_mount=mount)


def read_taulab(values, profiles, running):
    """The taulab action's outputs (module docstring, C2_TAULAB_*), each rendered empty or defaulted when unset. Its keys are
    checked only when the action runs: a job that does not run it is never refused over them."""
    if not running:
        return dict(taulab_profile=TAULAB_PROFILE, taulab_data='', taulab_arms=' '.join(TAULAB_ARMS),
                    taulab_deadline=str(TAULAB_DEADLINE_MINUTES), taulab_in_flight=str(TAULAB_IN_FLIGHT),
                    taulab_max_tokens='', taulab_counters='', taulab_drafter_arm='', taulab_pair_control='')
    profile = values.get('C2_TAULAB_PROFILE') or TAULAB_PROFILE
    if profile not in profiles:
        raise JobError('C2_TAULAB_PROFILE %r is not a profile of qwen_c2_profiles.json (%s)' % (profile, ', '.join(profiles)))
    drafter_arm = values.get('C2_TAULAB_DRAFTER_ARM', '')
    if drafter_arm and drafter_arm not in TAULAB_DRAFTER_ARMS:
        raise JobError('C2_TAULAB_DRAFTER_ARM %r is not one of %s' % (drafter_arm, ', '.join(TAULAB_DRAFTER_ARMS)))
    pair_control = values.get('C2_TAULAB_PAIR_CONTROL', '')
    if pair_control and (not pair_control.isdigit() or len(pair_control) > 20):
        raise JobError('C2_TAULAB_PAIR_CONTROL must be the digits of a workflow run id, got %r' % pair_control)
    if pair_control and drafter_arm in ('', 'control'):
        raise JobError('C2_TAULAB_PAIR_CONTROL pairs a candidate arm with the control run: set C2_TAULAB_DRAFTER_ARM to a candidate')
    candidate = drafter_arm not in ('', 'control')
    if candidate and not pair_control:
        raise JobError('C2_TAULAB_DRAFTER_ARM %s is judged against the control arm run: set C2_TAULAB_PAIR_CONTROL to the '
                       'run id (digits) of the control arm job, or no verdict is produced' % drafter_arm)
    arms = split_list(values.get('C2_TAULAB_ARMS', '')) or [arm for arm in TAULAB_ARMS if not (candidate and arm == 'A3')]
    if candidate and 'A3' in arms:
        raise JobError('C2_TAULAB_ARMS names A3 for the candidate arm %s: A3 is the control arm\'s calibration' % drafter_arm)
    unknown = sorted(set(arms) - set(TAULAB_ARMS))
    if unknown or len(set(arms)) != len(arms):
        raise JobError('C2_TAULAB_ARMS: %s (known: %s, each once)' % (
            ('unknown ' + ', '.join(unknown)) if unknown else 'an arm named twice', ' '.join(TAULAB_ARMS)))
    paths = {}
    for key in ('C2_TAULAB_DATA', 'C2_TAULAB_COUNTERS'):
        path = values.get(key, '')
        if path and (not PLAIN_PATH.fullmatch(path) or '..' in path.split('/')):
            raise JobError('%s must be a plain path (%s) with no .. component, got %r' % (key, PLAIN_PATH.pattern, path))
        paths[key] = path
    minutes = (positive_int('C2_TAULAB_DEADLINE', values['C2_TAULAB_DEADLINE']) if values.get('C2_TAULAB_DEADLINE')
               else TAULAB_DEADLINE_MINUTES)
    if minutes > MAX_TAULAB_DEADLINE_MINUTES:
        raise JobError('C2_TAULAB_DEADLINE: at most %d minutes, got %d' % (MAX_TAULAB_DEADLINE_MINUTES, minutes))
    in_flight = (positive_int('C2_TAULAB_IN_FLIGHT', values['C2_TAULAB_IN_FLIGHT']) if values.get('C2_TAULAB_IN_FLIGHT')
                 else TAULAB_IN_FLIGHT)
    if in_flight > MAX_TAULAB_IN_FLIGHT:
        raise JobError('C2_TAULAB_IN_FLIGHT: at most %d, got %d' % (MAX_TAULAB_IN_FLIGHT, in_flight))
    max_tokens = positive_int('C2_TAULAB_MAX_TOKENS', values['C2_TAULAB_MAX_TOKENS']) if values.get('C2_TAULAB_MAX_TOKENS') else ''
    return dict(taulab_profile=profile, taulab_data=paths['C2_TAULAB_DATA'], taulab_arms=' '.join(arms),
                taulab_deadline=str(minutes), taulab_in_flight=str(in_flight), taulab_max_tokens=str(max_tokens),
                taulab_counters=paths['C2_TAULAB_COUNTERS'], taulab_drafter_arm=drafter_arm, taulab_pair_control=pair_control)


def render(outputs):
    """GITHUB_OUTPUT lines, in a fixed order."""
    for key, value in outputs.items():
        if '\n' in value or '\r' in value:
            raise JobError('%s carries a newline' % key)
    return ''.join('%s=%s\n' % (key, outputs[key]) for key in sorted(outputs))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or len(argv) > 2:
        sys.stderr.write('usage: c2_serving_job.py <job.env> [qwen_c2_profiles.json]\n')
        return 2
    try:
        with open(argv[0], encoding='utf-8') as handle:
            values = parse_env(handle.read())
        profiles = profile_names(argv[1] if len(argv) > 1 else PROFILES)
        sys.stdout.write(render(read_job(values, profiles, envs=profile_envs(argv[1] if len(argv) > 1 else PROFILES))))
    except (JobError, OSError, ValueError, KeyError) as error:
        sys.stderr.write('c2-serving-job.env refused: %s\n' % error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
