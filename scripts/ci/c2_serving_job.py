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

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import json
import os
import re
import sys

ACTIONS = ('status', 'platform', 'unserve', 'priority', 'reset', 'cardm', 'drift', 'build', 'smoke', 'gate', 'replay',
           'push')
GATE_PLANS = ('bringup', 'matrix', 'memory', 'lifecycle')
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
    unknown = sorted(set(plans) - set(GATE_PLANS))
    if unknown:
        raise JobError('C2_GATE_PLAN: unknown %s (known: %s)' % (', '.join(unknown), ' '.join(GATE_PLANS)))
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
    return dict(actions=' '.join(actions), tag=tag, profile=profile, tests=values.get('C2_SMOKE_TESTS', ''),
                platform_image=platform_image, gate_plan=','.join(plans),
                gate_lengths=','.join(str(length) for length in lengths), gate_max_tokens=str(max_tokens),
                gate_memory_prompt=str(memory_prompt), replay_profile=replay_profile,
                replay_served_model=replay_served_model, cardm_harness=cardm_harness, cardm_args=cardm_args,
                cardm_env=cardm_env)


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
