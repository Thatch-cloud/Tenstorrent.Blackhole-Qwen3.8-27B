"""Read .github/c2-serving-job.env: the job the next experiment/c2-serving-vN tag runs.

    python3 scripts/ci/c2_serving_job.py .github/c2-serving-job.env >> "$GITHUB_OUTPUT"

prints one key=value line per workflow output (qwen-c2-serving.yml's 'Read the job file' step used
to parse the file inline; this is that parse, with the gate's keys added, where a CPU test can hold
it). Refuses (exit 1, the reason on stderr) anything the workflow must not act on: an unknown
action, a malformed image tag, platform image or replay model id, a profile the checkout's
qwen_c2_profiles.json does not define, an unknown gate plan, a prompt length that is not a positive
integer, and a card-M harness, argument or environment the cardm step must not run (read_cardm).

Keys (every one optional but C2_IMAGE_TAG):
  C2_ACTIONS          ACTIONS, run in the workflow's order (default: status)
  C2_IMAGE_TAG        the image is zot.thatch.local:5000/tt-vllm:qwen38-c2-<tag>
  C2_PROFILE          the C2 profile smoke and gate serve (default: general)
  C2_SMOKE_TESTS      comma-separated c2_serving_smoke.py tests, empty for all
  C2_PLATFORM_IMAGE   the thatch-serving-tt image the replay runs
  C2_GATE_PLAN        GATE_PLANS, comma- or space-separated, run in order (default: bringup)
  C2_GATE_LENGTHS     the matrix's prompt lengths, one per user (default, rendered empty: the S1 G4
                      ladder, whose top rung the gate lowers to what the image's profile admits)
  C2_GATE_MAX_TOKENS  the matrix's answer budget per user (default 4096)
  C2_GATE_MEMORY_PROMPT  G5's prompt length (default: the profile's largest admitted prompt)
  C2_REPLAY_PROFILE   the profile the replay's container serves (default: the source container's)
  C2_REPLAY_SERVED_MODEL  the model id the replay's /v1/models must advertise and its requests after
                      the warmup name, org/name[:tag] (default, rendered empty: c2_platform_replay.py's,
                      Qwen/Qwen3.8-27B:tt); Qwen/Qwen3.8-27B replays an image without the alias
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
  C2_GATE_SALT        a cache_salt for every gate stream (sticky sessions, harness item B3; default, rendered
                      empty: none sent, the payload exactly as before): fresh (each stream its own salt, minted
                      under a key the gate mounts, so every request takes the prefix route's capture path and
                      publishes, and none can hit) or none (said explicitly: every request unsalted, which a
                      prefix profile serves with reuse off). Only on a prefix profile (the gate refuses it on
                      any other before a container starts); warm on c2-packed-prefix then warms the prefix twins
  C2_GATE_BALLAST_MB  parked-ballast's (G-E3 iii) ballast, MB per chip (QWEN_FAST_GATE_DRAM_BALLAST, gate only):
                      sized from the G-E0 warm arm's P7p reading so the arrival sits at the parked need's boundary
                      (c2_serving_gate.ballast_advice prints the figure). Needed by that plan, refused without it
                      and refused beside a job that runs no parked-ballast
  (C2_GATE_PLAN may also name QUICKWIN_GATE_PLANS: quickwin-ledger and quickwin-warm, the phase-1 quick wins' byte
                      comparisons, on any S2 profile; they take no extra key)
  C2_PREFIX_PLAN      PREFIX_PLANS for the prefix action (c2_prefix_gate.py), in order (default: bringup);
                      each exactness and lifecycle arm is a plan too (PREFIX_ARM_PLANS): exactness-eager
                      re-runs that arm alone. Never beside its own plan, and no plan twice (one arm, one
                      results directory). exactness-shared (the four-agent shared-block arm) applies to the
                      sticky-session profiles only; on them exactness-traced and lifecycle-tiny are
                      NOT_APPLICABLE, and a plan with no applicable arm is refused by the gate
  C2_PREFIX_PROFILE   the prefix-reuse profile it serves (default general-prefix; must be a checkout profile)
  C2_PREFIX_BASELINE  the no-reuse profile it compares against (default general; none: timing without
                      the baseline arm)
  C2_PREFIX_AGENTS    the timing plan's busy-agent counts, one phase each (default 1,4,5,6)
  The C2_PREFIX_* keys are read only when C2_ACTIONS has prefix; otherwise their defaults are output.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import json
import os
import re
import sys

ACTIONS = ('status', 'platform', 'unserve', 'priority', 'reset', 'cardm', 'drift', 'build', 'probe', 'smoke', 'gate',
           'prefix', 'replay', 'push')
GATE_PLANS = ('bringup', 'matrix', 'memory', 'lifecycle')
# S2 (s2-design.md 6.3), run on the S2 image (graft K64j) and its c2-packed profiles; c2_serving_gate.py says what
# each runs. warm and warm-off are M1 (never judged for kernel-cache growth); control and forced-cap M3-M4 (G3);
# control-below M5 (G3b); lifecycle-arrival M6's added event; mixed, short, boundaries and staggered M7-M10 (G4);
# churn M11 (G5); permuted is built but required only under decision D-c(i).
S2_GATE_PLANS = ('warm', 'warm-off', 'control', 'forced-cap', 'control-below', 'lifecycle-arrival', 'mixed', 'short',
                 'boundaries', 'staggered', 'churn', 'permuted')
# Stage E, parked per-slot engines (QWEN_FAST_PARKED_ENGINES): run on a parked gate profile (c2-packed-prefix-parked-gate; its
# flag-off twin, the profile less '-parked', is the reference); c2_serving_gate.py says what each runs. parked-exact is
# G-E1 (a) and (f), parked-drafter G-E1 (b) and (g) with both negative controls, parked-lifecycle G-E2, parked-churn
# G-E3 (i), parked-corner G-E3 (ii), parked-ballast G-E3 (iii).
PARKED_GATE_PLANS = ('parked-exact', 'parked-drafter', 'parked-lifecycle', 'parked-churn', 'parked-corner',
                     'parked-ballast')
# Phase-1 quick wins (docs/engine-warm-skip.md), each usable without Stage E and run on the sticky or an S2 profile:
# quickwin-ledger holds QWEN_FAST_MEMORY_LEDGER_OFF=1 against the ledger on, quickwin-warm QWEN_FAST_ENGINE_WARM_SKIP=1
# against the warm forwards run, each solo and four concurrent, real text, strict exactness.
QUICKWIN_GATE_PLANS = ('quickwin-ledger', 'quickwin-warm')
ALL_GATE_PLANS = GATE_PLANS + S2_GATE_PLANS + PARKED_GATE_PLANS + QUICKWIN_GATE_PLANS
MAX_PAIRS = 4
MAX_BALLAST_MB = 4096   # per chip: a ballast past the whole idle free (about 4.4 GB) could only kill the attach
BELOW_FAMILY_RANGE = (4096, 16640)   # capacities the pinned (flag-off) mask admits: attention_mask_replay.py:18,26
JIT_MODES = ('auto', 'judge', 'record')
POLICIES = ('strict', 'dc-i')
AUDIT_SETS = ('extent', 'all')
# C2_GATE_SALT (sticky sessions B3): what cache_salt the gate's streams carry; unset sends none, as before.
SALT_MODES = ('none', 'fresh')
DECISION = re.compile(r'[A-Za-z0-9_.,:=/+@%#-]{3,200}')
# The prefix-reuse G1 gates (TT prefix-reuse design 2.2; c2_prefix_gate.py).
PREFIX_PLANS = ('bringup', 'exactness', 'lifecycle', 'timing')
# (arm, its plan): each exactness and lifecycle arm is a plan of its own, named as the arm, that runs
# only it, judged as inside its plan (neither plan has a cross-arm check; c2_prefix_gate.PLAN_ARMS).
# G1 v47 (run 36246961161) needed the eager arm again without the traced and audit arms' hour.
PREFIX_ARM_PLANS = (('exactness-traced', 'exactness'), ('exactness-audit', 'exactness'),
                    ('exactness-eager', 'exactness'), ('exactness-shared', 'exactness'),
                    ('lifecycle-evict', 'lifecycle'), ('lifecycle-store', 'lifecycle'),
                    ('lifecycle-tiny', 'lifecycle'))
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
PROFILE_NAME = re.compile(r'[a-z0-9_-]{1,40}')
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


def read_cardm(values, requested, root=ROOT, library=QUAL_CARD_LIBRARY):
    """(harness, args, env) for the cardm step, or JobError. Checked whenever set, required with cardm."""
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


def read_job(values, profiles, root=ROOT):
    """The workflow outputs for a parsed job file, or JobError."""
    actions = split_list(values.get('C2_ACTIONS', 'status')) or ['status']
    unknown = sorted(set(actions) - set(ACTIONS))
    if unknown:
        raise JobError('C2_ACTIONS: unknown %s (known: %s)' % (', '.join(unknown), ' '.join(ACTIONS)))
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
    s2 = read_s2_gate(values)
    ballast_plan = 'parked-ballast' in plans
    if ballast_plan and 'gate' in actions and not s2['gate_ballast_mb']:
        raise JobError('C2_GATE_PLAN parked-ballast needs C2_GATE_BALLAST_MB: the ballast is sized from the G-E0 '
                       'warm arm P7p reading (the gate ballast_advice line)')
    if s2['gate_ballast_mb'] and not (ballast_plan and 'gate' in actions):
        raise JobError('C2_GATE_BALLAST_MB belongs to parked-ballast: it does nothing in a job that runs no gate plan of that name')
    lengths_text = values.get('C2_GATE_LENGTHS', '')
    lengths = [positive_int('C2_GATE_LENGTHS', part) for part in split_list(lengths_text)]
    max_tokens = positive_int('C2_GATE_MAX_TOKENS', values['C2_GATE_MAX_TOKENS']) \
        if values.get('C2_GATE_MAX_TOKENS') else DEFAULT_MAX_TOKENS
    memory_prompt = positive_int('C2_GATE_MEMORY_PROMPT', values['C2_GATE_MEMORY_PROMPT']) \
        if values.get('C2_GATE_MEMORY_PROMPT') else ''
    replay_profile = values.get('C2_REPLAY_PROFILE', '')
    if replay_profile and replay_profile not in profiles:
        raise JobError('C2_REPLAY_PROFILE %r is not a profile of qwen_c2_profiles.json' % replay_profile)
    replay_served_model = values.get('C2_REPLAY_SERVED_MODEL', '')
    if replay_served_model and not MODEL_ID.fullmatch(replay_served_model):
        raise JobError('C2_REPLAY_SERVED_MODEL must match %s, got %r' % (MODEL_ID.pattern, replay_served_model))
    cardm_harness, cardm_args, cardm_env = read_cardm(values, 'cardm' in actions, root=root)
    prefix = read_prefix(values, profiles, 'prefix' in actions)
    outputs = dict(actions=' '.join(actions), tag=tag, profile=profile, tests=values.get('C2_SMOKE_TESTS', ''),
                   platform_image=platform_image, gate_plan=','.join(plans),
                   gate_lengths=','.join(str(length) for length in lengths), gate_max_tokens=str(max_tokens),
                   gate_memory_prompt=str(memory_prompt), replay_profile=replay_profile,
                   replay_served_model=replay_served_model, cardm_harness=cardm_harness, cardm_args=cardm_args,
                   cardm_env=cardm_env)
    outputs.update(s2)
    outputs.update(prefix)
    return outputs


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
    ballast = values.get('C2_GATE_BALLAST_MB', '')
    if ballast:
        ballast = str(positive_int('C2_GATE_BALLAST_MB', ballast))
        if int(ballast) > MAX_BALLAST_MB:
            raise JobError('C2_GATE_BALLAST_MB: at most %d MB per chip, got %s' % (MAX_BALLAST_MB, ballast))
    return dict(gate_pairs=pairs, gate_families=','.join(str(family) for family in families), gate_jit=jit,
                gate_policy=policy, gate_policy_decision=decision, gate_audits=audits, gate_salt=salt,
                gate_ballast_mb=ballast)


def read_prefix(values, profiles, running):
    """The prefix action's outputs. Its keys are checked only when the action runs: a job that does
    not run it (an exact or c2 run) is never refused over them, and gets the defaults. The profile
    must then be one the checkout defines (general-prefix lands in qwen_c2_profiles.json on another
    track, and the image's own profiles are what c2_prefix_gate.py finally checks)."""
    if not running:
        return dict(prefix_plan='bringup', prefix_profile=PREFIX_PROFILE, prefix_baseline=PREFIX_BASELINE,
                    prefix_agents=','.join(str(count) for count in PREFIX_AGENTS))
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
    return dict(prefix_plan=','.join(plans), prefix_profile=profile, prefix_baseline=baseline,
                prefix_agents=','.join(str(count) for count in agents))


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
        sys.stdout.write(render(read_job(values, profiles)))
    except (JobError, OSError, ValueError, KeyError) as error:
        sys.stderr.write('c2-serving-job.env refused: %s\n' % error)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
