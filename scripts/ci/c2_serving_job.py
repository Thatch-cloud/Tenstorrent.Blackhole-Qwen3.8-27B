"""Read .github/c2-serving-job.env: the job the next experiment/c2-serving-vN tag runs.

    python3 scripts/ci/c2_serving_job.py .github/c2-serving-job.env >> "$GITHUB_OUTPUT"

prints one key=value line per workflow output (qwen-c2-serving.yml's 'Read the job file' step used
to parse the file inline; this is that parse, with the gate's keys added, where a CPU test can hold
it). Refuses (exit 1, the reason on stderr) anything the workflow must not act on: an unknown
action, a malformed image tag or platform image, a profile the checkout's qwen_c2_profiles.json does
not define, an unknown gate plan, a prompt length that is not a positive integer.

Keys (every one optional but C2_IMAGE_TAG):
  C2_ACTIONS          ACTIONS, run in the workflow's order (default: status)
  C2_IMAGE_TAG        the image is zot.thatch.local:5000/tt-vllm:qwen38-c2-<tag>
  C2_PROFILE          the C2 profile smoke and gate serve (default: general)
  C2_SMOKE_TESTS      comma-separated c2_serving_smoke.py tests, empty for all
  C2_PLATFORM_IMAGE   the thatch-serving-tt image the replay runs
  C2_GATE_PLAN        GATE_PLANS, comma- or space-separated, run in order (default: bringup)
  C2_GATE_LENGTHS     the matrix's prompt lengths, one per user (default: the S1 G4 ladder)
  C2_GATE_MAX_TOKENS  the matrix's answer budget per user (default 4096)
  C2_GATE_MEMORY_PROMPT  G5's prompt length (default: the profile's largest admitted prompt)
  C2_REPLAY_PROFILE   the profile the replay's container serves (default: the source container's)

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import json
import os
import re
import sys

ACTIONS = ('status', 'platform', 'unserve', 'priority', 'reset', 'build', 'smoke', 'gate', 'replay', 'push')
GATE_PLANS = ('bringup', 'matrix', 'memory')
# S1's G4 ladder (c2-serve-for-real-plan 2.2, gate table row G4 part 1): both sides of every page and
# chunk boundary the fast path has (2048 = the draft window and the prefill chunk), a short prompt
# far below any of them, and long ones up to the ~123k prompt cap.
LADDER = (60, 255, 2047, 2048, 2049, 4096, 32768, 60000, 120000)
DEFAULT_MAX_TOKENS = 4096
PROFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen_c2_profiles.json')
TAG = re.compile(r'[0-9a-z.-]{3,40}')
PLATFORM_IMAGE = re.compile(r'[0-9a-z.:/@_-]{10,200}')
PROFILE_NAME = re.compile(r'[a-z0-9_-]{1,40}')


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


def read_job(values, profiles):
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
    lengths = [positive_int('C2_GATE_LENGTHS', part) for part in split_list(lengths_text)] or list(LADDER)
    max_tokens = positive_int('C2_GATE_MAX_TOKENS', values['C2_GATE_MAX_TOKENS']) \
        if values.get('C2_GATE_MAX_TOKENS') else DEFAULT_MAX_TOKENS
    memory_prompt = positive_int('C2_GATE_MEMORY_PROMPT', values['C2_GATE_MEMORY_PROMPT']) \
        if values.get('C2_GATE_MEMORY_PROMPT') else ''
    replay_profile = values.get('C2_REPLAY_PROFILE', '')
    if replay_profile and replay_profile not in profiles:
        raise JobError('C2_REPLAY_PROFILE %r is not a profile of qwen_c2_profiles.json' % replay_profile)
    return dict(actions=' '.join(actions), tag=tag, profile=profile, tests=values.get('C2_SMOKE_TESTS', ''),
                platform_image=platform_image, gate_plan=','.join(plans),
                gate_lengths=','.join(str(length) for length in lengths), gate_max_tokens=str(max_tokens),
                gate_memory_prompt=str(memory_prompt), replay_profile=replay_profile)


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
