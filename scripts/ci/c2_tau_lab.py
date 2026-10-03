"""The W-T1 tau lab (Linear THA-268 tooling, THA-269 the run): Qwen3.8-27B's own greedy answers on agent turns, through
the PRODUCTION arithmetic on four cards, one container load for every arm (design: tau-on-tt/design.md section 1).

    python3 scripts/ci/c2_tau_lab.py --image <registry>/tt-vllm:qwen38-c2-<tag> \
        --profile c2-packed-tp4 --data ~/kwork64/taulab/data --results ~/kwork64/taulab/results/<run> \
        --public "$RUNNER_TEMP/c2-results/taulab" --deadline-seconds 16200

Run by qwen-c2-serving.yml's 'taulab' action (C2_CARDS=quad), after the window's other jobs. Host side: this file starts ONE
container through c2_prefix_gate.server_run (the node agent's shape) serving a DERIVED profile - the production profile with
exactly two arithmetic-neutral log flags added (QWEN_FAST_PACKED_AUDIT=1: one [PACKED] line per user per round;
QWEN_FAST_PHASE_TIMING=1: one fast_serving_phases record per request; real_text_compare.ARITHMETIC_NEUTRAL) - and drives the
arms over its OpenAI API from this host. The data directory is read (never written, never mounted into the model container,
which has no use for it); every private output goes to --results, and only AGGREGATES go to --public (what the workflow
uploads: the Qwen repo is PUBLIC, so its Actions logs and artifacts are public). Stdout carries counts and aggregates only.

THE ARMS (order, default concurrency; design table in section 1.1)
  A1  SWE-rebench held-out turns            240 turns, thinking ON, independent turns, IN_FLIGHT at once
  A2  our own Claude Code / Codex traces     96 turns, thinking ON, independent turns (rig-local data only)
  A3  calibration: the lanes 4 x 4k + 4 x 32k prompts, built from the IMAGE itself before the lab container starts (a
      throwaway container, no network, no cards: real_text_prompts.build_prompts), each set of 4 sent at once, thinking OFF
      (the references were measured thinking-off, so a thinking-ON calibration could not calibrate anything), max_tokens 2400
      (the budget of the v162/v179 runs: tau depends on the output position). Must reproduce the A3 reference (the report's
      own estimator over those runs' logs, references/tau-lab/a3-reference.json) within +/-7%; a missing reference is a refusal
  A4  chained synthetic agent sessions       20 sessions x 6 turns, thinking ON, the model's own answers fed back (content and
      tool calls; the reasoning is not sent back, as an agent replaying a transcript does not) and each session grown to the
      seed's prompt targets (16k to 60k) with prefix_agent_corpus.Conversation, IN_FLIGHT sessions at once
  A5  the thinking-OFF pair of the 100 turns a5_pairs.json names (the same turn ids as their A1 turns, paired)
  G   (optional, never in the default run) the drafter fine-tune's generation arm: the turns of a TRAINING data directory
      (ft_select.py: g_train.jsonl + ids, MANIFEST.json with a G table only), thinking ON, each answered by the target and KEPT in
      outputs.jsonl (as every arm's answers are); named explicitly: --arms G
THINKING. Production agents call Qwen3.8 with thinking ON, which is the chat template's default: an ON turn carries NO
chat_template_kwargs (the smoke's stream_reasoning sends none either); an OFF turn is rendered with enable_thinking false. The
data's token ids are exactly those renders: A1, A2 and A4 are thinking ON; A5 (and A3, as the lanes baked it) thinking OFF.
ANSWER BUDGET. A thinking-ON answer reasons first, so a short budget measures a reasoning prefix, not the agent turn production
serves (reasoning, then a tool call): A1, A2, A4 and A5 default to 2048 output tokens. The summary reports, per arm, the share of
turns that got past </think>, that made a tool call and how they finished (stop / length), so a reader sees what the tau covers;
the arms share the box by the design's wall weights and the pre-registered order trims what does not fit.

DATA (FORMAT.md beside the data files is the contract; this reader matches it). The directory holds MANIFEST.json (format
"FORMAT.md", file sizes, sha256, record counts, arm counts), one conversation per line in a1_swe_heldout.jsonl and
a2_own_sessions.jsonl (turns[] with turn_id, bucket, prompt_tokens_think_on and the inverse-probability inputs
eligible_per_bucket / chosen_in_bucket), the token ids of every turn's prompt in the matching *.ids.jsonl.gz (thinking ON),
a5_pairs.json + a5_think_off.ids.jsonl.gz (the OFF renders), a3_calibration.json (a build spec), a4_seeds.jsonl and
scrub_report.json. Every file the arms need is checked against MANIFEST.json (size and sha256), the arm counts against its
arm table (A1 240, A2 96, A3 8, A4 20 sessions / 120 turns, A5 100), and A2's scrub report must show a clean final check - and
the run REFUSES (before any container) on any shortfall; a missing arm file is never an empty arm. A turn is sent as its token
ids (/v1/completions): no transcript text is ever loaded into the driver, so there is no text to print or leak. Each turn carries
id = turn_id, cluster = the conversation's group (the bootstrap resamples it), its turn index, its prompt tokens and its
inverse-probability weight (eligible / chosen in its bucket), which the report applies to the pooled tau, p10 and buckets.

RESULTS (private, under --results; mode 0700): turns.jsonl (one line per turn, status/usage/chunk counts/ids - no text),
outputs.jsonl (the output token ids per turn; an A4 turn's answer, which its next turn is built from), server.log
(docker logs of the one container, whole), run.json, profiles.json, docker-run.json and, from tau_lab_report, tapes.jsonl and
report.private.json. A killed run RESUMES: turns already ok in turns.jsonl are skipped (and an A4 session continues from its
stored answers). --analyze-only reruns the report over a finished results directory.

HANGS. A request that gets no byte for REQUEST_READ_TIMEOUT s fails; BREAKER_FAILURES consecutive transport failures, or the
container no longer running, stop the lab (the report is still written). A CANARY after the first CANARY_TURNS ok turns requires
attributed [PACKED] rounds and fast_serving_phases records in the container's log, or stops the lab (the log flags are not in
effect: every later turn would measure nothing).

Python 3.7 syntax, stdlib only: it runs on the rig host.
"""
import argparse
import array
import copy
import gzip
import hashlib
import http.client
import json
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_prefix_gate as pg  # noqa: E402
import c2_serving_gate as gate  # noqa: E402
import prefix_replay as replay  # noqa: E402
import tau_lab_report as report  # noqa: E402

ARMS = ('A1', 'A2', 'A3', 'A4', 'A5')
OPTIONAL_ARMS = ('G',)           # G: the drafter fine-tune's generation arm (ft_select.py's data directory); never in the default run
LOG_FLAGS = (('QWEN_FAST_PACKED_AUDIT', '1'), ('QWEN_FAST_PHASE_TIMING', '1'))
AUDIT_FLAGS = (('QWEN_FAST_VERIFY_T1_AUDIT', '1'), ('QWEN_FAST_VERIFY_T2_AUDIT', '1'))   # production runs with the verify audits ON
PRODUCTION_TAG = 'tp4-serve-2'
CONTAINER = 'qwen-c2-taulab'
PORT = 8022
SEED = 20261001
READINESS_SECONDS = 1800
RESERVE_SECONDS = 900            # what the stop, the log and the report need after the last request
SETTLE_SECONDS = 5.0
TURN_GRACE_SECONDS = 300         # a turn sent just before its arm's end may run this long past it
REQUEST_READ_TIMEOUT = 600       # no byte for this long: the turn fails. A queued request gets no bytes while it waits for a
                                 # seat (about one turn of the four ahead of it), so this is generous, but a hung engine
                                 # (the THA-251 family) is then seen within 10 minutes, not 25
BREAKER_FAILURES = 4             # consecutive transport failures (no answer, a 5xx, a read timeout) that stop the lab
CANARY_TURNS = 6                 # after this many ok turns the container's log must show [PACKED] rounds and phases records
CANARY_WAIT_SECONDS = 60
DEFAULT_IN_FLIGHT = 8            # 4 seats + 4 queued: keeps the packed block full while one seat refills
DEFAULT_MAX_TOKENS = 2048
CALIBRATION_MAX_TOKENS = 2400    # the lanes' v162 / v179 padded-4k/32k arms (their m3native-gate.json)
ARM_SPEC = {
    # kind, default set, thinking (True/False/None = as baked), design wall weight (minutes), default max_tokens
    'A1': dict(kind='turns', set='swe', thinking=True, weight=85, max_tokens=DEFAULT_MAX_TOKENS),
    'A2': dict(kind='turns', set='own', thinking=True, weight=55, max_tokens=DEFAULT_MAX_TOKENS),
    'A3': dict(kind='calib', set='calib', thinking=False, weight=8, max_tokens=CALIBRATION_MAX_TOKENS),
    'A4': dict(kind='chain', set='chained', thinking=True, weight=60, max_tokens=DEFAULT_MAX_TOKENS),
    'A5': dict(kind='turns', set='swe', thinking=False, weight=20, max_tokens=DEFAULT_MAX_TOKENS),
    # G: target-regenerated answers for fine-tune training turns; thinking ON like production agents, the answers KEPT in outputs.jsonl
    'G': dict(kind='turns', set='train', thinking=True, weight=100, max_tokens=DEFAULT_MAX_TOKENS),
}
ID = re.compile(r'[A-Za-z0-9_.:-]{1,64}')
SETS = ('swe', 'own', 'chained', 'calib')
MANIFEST_NAME = 'MANIFEST.json'
SCRUB_REPORT_NAME = 'scrub_report.json'
FORMAT_NAME = 'FORMAT.md'
THINK_END_TOKEN = 248069         # </think> and <tool_call> in the served tokenizer (the manifest may say otherwise)
TOOL_CALL_TOKEN = 248058
A3_REFERENCE = os.path.join(HERE, 'references', 'tau-lab', 'a3-reference.json')
# What each arm reads: (conversation file, ids file) / (spec) / (seeds) / (pairs, OFF ids). A5 also reads A1's pair of files.
ARM_FILES = {
    'A1': ('a1_swe_heldout.jsonl', 'a1_swe_heldout.ids.jsonl.gz'),
    'A2': ('a2_own_sessions.jsonl', 'a2_own_sessions.ids.jsonl.gz'),
    'A3': ('a3_calibration.json',),
    'A4': ('a4_seeds.jsonl',),
    'A5': ('a5_pairs.json', 'a5_think_off.ids.jsonl.gz', 'a1_swe_heldout.jsonl', 'a1_swe_heldout.ids.jsonl.gz'),
    'G': ('g_train.jsonl', 'g_train.ids.jsonl.gz'),
}
SEED_FIELDS = ('id', 'name', 'seed', 'system', 'task_template', 'first_tokens', 'turns', 'prompt_targets')


class LabError(ValueError):
    """A lab the driver refuses before any container starts. The message names a file or a count, never a value of the data."""


# -- the data --------------------------------------------------------------------------------------------------------

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(data):
    """MANIFEST.json of the data directory (the exact name: the rig's file system is case sensitive). Refuses a missing file
    or any format but FORMAT.md."""
    path = os.path.join(data, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise LabError('%s is missing from the data directory' % MANIFEST_NAME)
    try:
        with open(path, encoding='utf-8') as handle:
            manifest = json.load(handle)
    except ValueError:
        raise LabError('%s is not JSON' % MANIFEST_NAME)
    if not isinstance(manifest, dict) or manifest.get('format') != FORMAT_NAME \
            or not isinstance(manifest.get('files'), dict) or not isinstance(manifest.get('arms'), dict):
        raise LabError('%s is not the %s manifest (format, files, arms)' % (MANIFEST_NAME, FORMAT_NAME))
    return manifest


PUBLIC_SOURCES = ('swe', 'chained', 'code')       # training sources that are public text: no scrub report is needed for them alone


def g_needs_scrub(data):
    """True when arm G's conversation file holds ANY conversation whose source is not a public one (our own traces, or a record that
    names no source at all: unknown is treated as own). Read before the manifest check; the same file's size and sha256 are verified in
    the same pass, so an edit that hides an `own` source is refused there."""
    path = os.path.join(data, ARM_FILES['G'][0])
    try:
        conversations = read_jsonl(path, ARM_FILES['G'][0])
    except OSError:
        return True
    return any(not isinstance(conv, dict) or conv.get('source') not in PUBLIC_SOURCES for conv in conversations)


def needs_scrub(data, arms):
    """A2 always (it is our own sessions); G whenever any of its conversations could be our own."""
    return 'A2' in arms or ('G' in arms and g_needs_scrub(data))


def files_needed(arms, data=None):
    names = []
    for arm in arms:
        for name in ARM_FILES[arm]:
            if name not in names:
                names.append(name)
    if 'A2' in arms or ('G' in arms and (data is None or g_needs_scrub(data))):
        names.append(SCRUB_REPORT_NAME)
    return names


def verify_files(data, manifest, arms):
    """Every file the arms read must be in the manifest at its recorded size and sha256 (a stale or edited copy is refused)."""
    entries = manifest['files']
    for name in files_needed(arms, data):
        entry = entries.get(name)
        path = os.path.join(data, name)
        if not isinstance(entry, dict) or not os.path.isfile(path):
            raise LabError('%s is missing or not in %s' % (name, MANIFEST_NAME))
        if os.path.getsize(path) != entry.get('bytes') or sha256_file(path) != entry.get('sha256'):
            raise LabError('%s does not match %s (size or sha256)' % (name, MANIFEST_NAME))


def check_scrub_report(data, arms):
    """The gate before an arm that could carry our own traces (A2, and G when any of its conversations is not a public source): the
    data agent's scrub report must record a clean FINAL check of the written files (no residual detector hit in the conversations or the
    decoded ids) and name the conversations kept. -> counts (never values)."""
    if not needs_scrub(data, arms):
        return {}
    try:
        with open(os.path.join(data, SCRUB_REPORT_NAME), encoding='utf-8') as handle:
            scrub = json.load(handle)
    except (OSError, ValueError):
        raise LabError('%s is missing or not JSON' % SCRUB_REPORT_NAME)
    hits = scrub.get('final_detector_hits_on_written_file')
    if not isinstance(hits, dict):
        raise LabError('%s has no final detector check of the written files' % SCRUB_REPORT_NAME)
    if any(hits.values()):
        raise LabError('%s records residual detector hits in the written files: %d' % (SCRUB_REPORT_NAME, sum(hits.values())))
    return dict(kept=scrub.get('conversations_kept'), dropped=scrub.get('conversations_dropped'))


def read_jsonl(path, what):
    opener = gzip.open if path.endswith('.gz') else open
    out = []
    with opener(path, 'rt', encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                raise LabError('%s line %d is not JSON' % (what, number))
    return out


def read_ids(path, think):
    """{turn_id: array of token ids} from an ids file; every line must say `think`, carry n == len(ids), a plain id once."""
    ids = {}
    name = os.path.basename(path)
    for number, line in enumerate(read_jsonl(path, name), 1):
        turn_id = line.get('turn_id') if isinstance(line, dict) else None
        if not isinstance(turn_id, str) or not ID.fullmatch(turn_id) or turn_id in ids:
            raise LabError('%s line %d: a turn_id that is not a plain, unique id' % (name, number))
        if line.get('think') != think or not isinstance(line.get('ids'), list) or line.get('n') != len(line['ids']):
            raise LabError('%s line %d: not a %s render with n == len(ids)' % (name, number, think))
        ids[turn_id] = array.array('i', line['ids'])
    return ids


def read_conversations(path):
    """The conversations without their text (messages and tools dropped on read: the lab sends the ids)."""
    name = os.path.basename(path)
    out = []
    for number, conv in enumerate(read_jsonl(path, name), 1):
        if not isinstance(conv, dict) or not isinstance(conv.get('turns'), list) or not conv['turns'] \
                or not isinstance(conv.get('group'), str) or not ID.fullmatch(conv['group']):
            raise LabError('%s line %d: a conversation needs group and turns' % (name, number))
        conv.pop('messages', None)
        conv.pop('tools', None)
        out.append(conv)
    return out


def inverse_probability_weight(turn, name):
    """eligible_per_bucket[b] / chosen_in_bucket[b] for the turn's bucket b (the data's own weight, FORMAT.md)."""
    bucket = turn.get('bucket')
    try:
        eligible = float((turn.get('eligible_per_bucket') or {})[bucket])
        chosen = float((turn.get('chosen_in_bucket') or {})[bucket])
    except (KeyError, TypeError, ValueError):
        raise LabError('%s: a turn without the inverse-probability inputs of its bucket' % name)
    if chosen <= 0 or eligible <= 0:
        raise LabError('%s: a turn whose bucket weight is not positive' % name)
    return eligible / chosen


def conversation_records(path, ids_path, default_set):
    """One record per turn of a conversation file: id = turn_id, cluster = group, the turn index, its prompt tokens, the ids of
    its thinking-ON render and its inverse-probability weight. -> [record]"""
    name = os.path.basename(path)
    ids = read_ids(ids_path, 'on')
    records, seen = [], set()
    for conv in read_conversations(path):
        for index, turn in enumerate(conv['turns']):
            turn_id = turn.get('turn_id')
            if not isinstance(turn_id, str) or not ID.fullmatch(turn_id) or turn_id in seen:
                raise LabError('%s: a turn_id that is not a plain, unique id' % name)
            seen.add(turn_id)
            if turn_id not in ids:
                raise LabError('%s: a turn without token ids in %s' % (name, os.path.basename(ids_path)))
            if turn.get('prompt_tokens_think_on') != len(ids[turn_id]):
                raise LabError('%s: a turn whose prompt_tokens_think_on is not its ids length' % name)
            records.append(dict(id=turn_id, set=default_set, cluster=conv['group'], turn=index, tokens=len(ids[turn_id]),
                                bucket=turn.get('bucket'), prompt_ids=ids[turn_id],
                                weight=inverse_probability_weight(turn, name)))
    extra = sorted(set(ids) - seen)
    if extra:
        raise LabError('%s holds %d turns the conversation file does not' % (os.path.basename(ids_path), len(extra)))
    return records


def pair_records(data, a1_records):
    """A5: exactly the turn ids a5_pairs.json names, carrying their thinking-OFF ids (prompt_ids_nothink) under the SAME ids as
    their A1 turns (paired), with A1's cluster and weight."""
    try:
        with open(os.path.join(data, 'a5_pairs.json'), encoding='utf-8') as handle:
            pairs = json.load(handle)
    except (OSError, ValueError):
        raise LabError('a5_pairs.json is missing or not JSON')
    wanted = pairs.get('turn_ids') if isinstance(pairs, dict) else None
    if not isinstance(wanted, list) or len(set(wanted)) != len(wanted):
        raise LabError('a5_pairs.json has no list of unique turn_ids')
    off = read_ids(os.path.join(data, 'a5_think_off.ids.jsonl.gz'), 'off')
    by_id = dict((record['id'], record) for record in a1_records)
    if set(off) != set(wanted) or not set(wanted) <= set(by_id):
        raise LabError('the A5 turn ids, the OFF ids and A1 do not agree')
    records = []
    for turn_id in wanted:
        mine = dict(by_id[turn_id])
        mine.update(prompt_ids_nothink=off[turn_id], tokens=len(off[turn_id]))
        records.append(mine)
    return records


def seed_records(path):
    """A4's session seeds (a4_seeds.jsonl): refused when a seed lacks a field the chain needs."""
    name = os.path.basename(path)
    records = []
    for number, seed in enumerate(read_jsonl(path, name), 1):
        if not isinstance(seed, dict) or any(field not in seed for field in SEED_FIELDS):
            raise LabError('%s line %d: a session seed lacks one of %s' % (name, number, ', '.join(SEED_FIELDS)))
        targets = seed['prompt_targets']
        if not (isinstance(seed['id'], str) and ID.fullmatch(seed['id']) and isinstance(seed['turns'], int) and seed['turns'] >= 1
                and isinstance(targets, list) and len(targets) == seed['turns'] and all(isinstance(t, int) for t in targets)
                and seed['system'] in ('full', 'compact') and isinstance(seed['first_tokens'], int)
                and isinstance(seed['task_template'], str)):
            raise LabError('%s line %d: a session seed with a malformed field' % (name, number))
        records.append(dict(seed, set='chained', cluster=seed['id'], weight=1.0))
    return records


def calibration_records(path):
    """A3's records from its build spec: one per user of every set, the prompt ids still to come from the image (`pending`)."""
    try:
        with open(path, encoding='utf-8') as handle:
            spec = json.load(handle)
        sets = spec['sets']
        records = []
        for wave, entry in enumerate(sets):
            for user in range(int(entry['users'])):
                records.append(dict(id='a3-%s-%d' % (entry['name'], user), calib_set=entry['name'], set='calib', cluster=entry['name'], turn=user,
                                    wave=wave, tokens=int(entry['target']), max_tokens=CALIBRATION_MAX_TOKENS, weight=1.0,
                                    pending=True, prompt_ids=None))
    except (OSError, ValueError, KeyError, TypeError):
        raise LabError('a3_calibration.json is not a build spec (sets of users and target)')
    return records


def expected_counts(manifest):
    """{arm: records the manifest says it holds} (A4: sessions, with the turns beside)."""
    arms = manifest['arms']
    want = {}
    try:
        if 'G' in arms:
            want['G'] = arms['G']['turns']
        if set(arms) != set(['G']):          # a data directory of training turns only has no A tables; any other keeps the full contract
            want.update(A1=arms['A1']['turns'], A2=arms['A2']['turns'], A3=arms['A3']['turns'], A4=arms['A4']['sessions'],
                        A5=arms['A5']['turns'], A4_turns=arms['A4']['turns'])
    except (KeyError, TypeError):
        raise LabError('%s lacks an arm table (A1 A2 A3 A4 A5 with their counts)' % MANIFEST_NAME)
    return want


def check_counts(data, manifest, arms):
    """Refuse (before any container) when an arm asked for holds fewer records than the manifest says."""
    want = expected_counts(manifest)
    for arm in arms:
        if arm not in want:
            raise LabError('%s lacks the arm table of %s' % (MANIFEST_NAME, arm))
        have = len(data.get(arm) or [])
        if have < want[arm]:
            raise LabError('arm %s holds %d records, the manifest says %d' % (arm, have, want[arm]))
    if 'A4' in arms and sum(seed['turns'] for seed in data['A4']) < want['A4_turns']:
        raise LabError('arm A4 holds fewer turns than the manifest says')


def load_data(data, arms):
    """({arm: [records]}, manifest, scrub counts): the real file layout (FORMAT.md), verified against MANIFEST.json, counted
    against its arm table. Raises LabError on any shortfall; nothing is read that no arm needs."""
    manifest = read_manifest(data)
    verify_files(data, manifest, arms)
    scrub = check_scrub_report(data, arms)
    loaded = {}
    a1 = None
    if 'A1' in arms or 'A5' in arms:
        a1 = conversation_records(os.path.join(data, ARM_FILES['A1'][0]), os.path.join(data, ARM_FILES['A1'][1]), 'swe')
    for arm in arms:
        if arm == 'A1':
            loaded[arm] = a1
        elif arm == 'A2':
            loaded[arm] = conversation_records(os.path.join(data, ARM_FILES['A2'][0]),
                                               os.path.join(data, ARM_FILES['A2'][1]), 'own')
        elif arm == 'G':
            loaded[arm] = conversation_records(os.path.join(data, ARM_FILES['G'][0]), os.path.join(data, ARM_FILES['G'][1]), 'train')
        elif arm == 'A3':
            loaded[arm] = calibration_records(os.path.join(data, ARM_FILES['A3'][0]))
        elif arm == 'A4':
            loaded[arm] = seed_records(os.path.join(data, ARM_FILES['A4'][0]))
        else:
            loaded[arm] = pair_records(data, a1)
    check_counts(loaded, manifest, arms)
    return loaded, manifest, scrub


def read_a3_reference(path=None):
    """The A3 reference the report compares with (per set and pooled, with its source): refused when it is missing or holds no
    pooled tau and per-set taus. -> dict"""
    path = path or A3_REFERENCE
    try:
        with open(path, encoding='utf-8') as handle:
            reference = json.load(handle)
        sets = reference['sets']
        if not (isinstance(reference['pooled_tau'], (int, float)) and reference['pooled_tau'] > 0 and reference.get('source')
                and sets and all(isinstance(value, (int, float)) and value > 0 for value in sets.values())):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise LabError('the A3 reference (references/tau-lab/a3-reference.json) is missing or incomplete')
    return reference


def order_records(records, seed, arm):
    """The pre-registered random order: a seeded shuffle of the ids sorted, so the order depends on the ids and the seed only."""
    ordered = sorted(records, key=lambda record: record['id'])
    random.Random('%s:%s' % (seed, arm)).shuffle(ordered)
    return ordered


def calibration_waves(records):
    """A3's waves of four (the lanes' four seats): by the record's `wave`, else sorted by prompt size in fours."""
    if records and all('wave' in record for record in records):
        waves = {}
        for record in records:
            waves.setdefault(record['wave'], []).append(record)
        return [waves[key] for key in sorted(waves)]
    ordered = sorted(records, key=lambda record: (record.get('tokens') or 0, record['id']))
    return [ordered[at:at + 4] for at in range(0, len(ordered), 4)]


CALIBRATION_SCRIPT = (
    'import json, sys; sys.path.insert(0, "/bench"); import real_text_prompts as r; '
    'sets = json.loads(sys.argv[1]); out = {}; info = None\n'
    'for name, users, target in sets:\n'
    '    built = r.build_prompts(users, target, model=sys.argv[2])\n'
    '    out[name] = [entry["tokens"] for entry in built["users"]]\n'
    '    info = built["corpus"]\n'
    'print(json.dumps(dict(prompts=out, files=info.get("files"), characters=info.get("characters"))))')


def build_calibration(image, spec_path, checkout, hub=gate.HUB, snapshot=gate.SNAPSHOT, runner=subprocess.run, timeout=1800):
    """A3's prompts from the image itself, in a throwaway container (no network, no devices, the contract's boot off):
    real_text_prompts.build_prompts(4, 4096) and (4, 32768) over the image's own installed vLLM source, exactly the
    lanes' prompts. -> {set name: [token ids per user]}, {files, characters}. Raises LabError when the build fails."""
    with open(spec_path, encoding='utf-8') as handle:
        spec = json.load(handle)
    sets = [[entry['name'], int(entry['users']), int(entry['target'])] for entry in spec['sets']]
    arguments = ['docker', 'run', '--rm', '--network', 'none', '-e', 'QWEN_C2_SERVING=0',
                 '--mount', 'type=bind,src=%s,dst=/bench/real_text_prompts.py,readonly' % os.path.join(
                     checkout, 'scripts', 'ci', 'real_text_prompts.py'),
                 '--mount', 'type=bind,src=%s,dst=/models,readonly' % hub,
                 '--entrypoint', 'python3', image, '-B', '-c', CALIBRATION_SCRIPT, json.dumps(sets), snapshot]
    try:
        result = runner(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        raise LabError('the A3 prompt build did not run: %s' % type(error).__name__)
    if result.returncode:
        raise LabError('the A3 prompt build exited %s' % result.returncode)
    try:
        built = json.loads(result.stdout.decode('utf-8').strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise LabError('the A3 prompt build printed no JSON')
    for name, users, target in sets:
        prompts = built['prompts'].get(name)
        if not prompts or len(prompts) != users or any(len(tokens) != target for tokens in prompts):
            raise LabError('the A3 prompt build is not %d prompts of %d tokens' % (users, target))
    return built['prompts'], dict(files=built.get('files'), characters=built.get('characters'))


def attach_calibration(records, built):
    """Fill A3's pending records with the image's prompts."""
    for record in records:
        record['prompt_ids'] = built[record['calib_set']][record['turn']]
        record['pending'] = False
    return records


def plan_counts(data, arms):
    return dict((arm, len(data.get(arm) or [])) for arm in arms)


# -- the profile and the container -----------------------------------------------------------------------------------

def derive_profile(profiles, base):
    """(derived name, a profiles document holding only it): `base` with the two log flags. Refuses a base the lab cannot
    mean: not a four-card profile, a gate-only one, or one that already sets a flag to something else."""
    names = profiles['profiles']
    if base not in names:
        raise LabError('profile %s is not one the image carries' % base)
    profile = copy.deepcopy(names[base])
    if profile.get('mesh_device') != 'P150x4':
        raise LabError('profile %s does not open the four-card mesh' % base)
    if profile.get('gate_only') is True:
        raise LabError('profile %s is gate-only: the lab serves the production profile' % base)
    env = profile.setdefault('env', {})
    for name, value in LOG_FLAGS:
        if str(env.get(name, value)) != value:
            raise LabError('profile %s sets %s to %s, not %s' % (base, name, env[name], value))
        env[name] = value
    derived = '%s+taulab' % base
    profile['description'] = '%s, derived by c2_tau_lab: %s' % (base, ', '.join('%s=%s' % pair for pair in LOG_FLAGS))
    return derived, dict(default=derived, profiles={derived: profile})


def audits_on(profiles, base):
    """Whether the base profile runs the verify T1 and T2 audits (production does: with them off the traffic profile hung, the
    THA-251 family, and a result without them is not production's tau)."""
    env = (profiles['profiles'].get(base) or {}).get('env') or {}
    return all(str(env.get(name)) == value for name, value in AUDIT_FLAGS)


def arithmetic_diff(profiles, base, derived_document):
    """The env names the derived profile adds or changes that real_text_compare does not list as arithmetic-neutral:
    [] is the design's 'derived profile's arithmetic diff is empty'."""
    import real_text_compare as compare
    before = {key: str(value) for key, value in (profiles['profiles'][base].get('env') or {}).items()}
    document = next(iter(derived_document['profiles'].values()))
    after = {key: str(value) for key, value in (document.get('env') or {}).items()}
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    return [key for key in changed
            if key not in compare.ARITHMETIC_NEUTRAL and not key.endswith(compare.ARITHMETIC_NEUTRAL_SUFFIXES)]


def image_tag(image):
    return image.rsplit(':', 1)[-1] if ':' in image.rsplit('/', 1)[-1] else ''


def is_production_image(image):
    """The design's label: the production image tp4-serve-2 (the tag ends with it); any other image's tau is labelled
    non-production in the report."""
    return image_tag(image).endswith(PRODUCTION_TAG)


# -- the requests ----------------------------------------------------------------------------------------------------

def request_body(job, thinking, max_tokens, model):
    """(path, body) for one turn. Greedy, streamed, with the continuous usage stats the third source needs and the token
    ids the tapes need. Thinking ON sends no chat_template_kwargs (the template's default is what production agents get)."""
    options = dict(include_usage=True, continuous_usage_stats=True)
    common = dict(model=model, max_tokens=int(max_tokens), temperature=0.0, top_p=1.0, stream=True,
                  stream_options=options, return_token_ids=True)
    if job.get('messages') is not None:
        body = dict(common, messages=job['messages'])
        if job.get('tools'):
            body['tools'] = job['tools']
            body['tool_choice'] = 'auto'
        if thinking is False:
            body['chat_template_kwargs'] = dict(enable_thinking=False)
        return '/v1/chat/completions', body
    ids = job.get('prompt_ids_nothink') if thinking is False and job.get('prompt_ids_nothink') else job['prompt_ids']
    return '/v1/completions', dict(common, prompt=list(ids))


class Chunks(object):
    """One streamed completion, fed server-sent-event lines: its id, the completion tokens per chunk (the third source),
    the kind of each chunk (reasoning, content or tool call, from the chat deltas), the output token ids, the usage and the
    finish reason. Never keeps text except `content` and the tool calls when asked (the A4 chain's answer, which the next
    turn is built from)."""

    def __init__(self, keep_content=False, clock=time.time):
        self.clock, self.keep_content = clock, keep_content
        self.started = clock()
        self.first_at = None
        self.request_id = None
        self.chunk_tokens, self.kinds, self.token_ids = [], [], []
        self.content, self.finish, self.error, self.usage, self.done = [], None, None, None, False
        self.calls = {}
        self.reported = 0

    def feed(self, raw):
        line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith('data:'):
            return self.done
        body = line[5:].strip()
        if body == '[DONE]':
            self.done = True
            return True
        try:
            chunk = json.loads(body)
        except ValueError:
            return False
        if chunk.get('error') is not None:
            self.error = 'server error'
            self.done = True
            return True
        self.request_id = self.request_id or chunk.get('id')
        usage = chunk.get('usage')
        if usage:
            self.usage = usage
        total = usage.get('completion_tokens') if usage else None
        for choice in chunk.get('choices') or ():
            ids = choice.get('token_ids') or []
            delta = choice.get('delta') or {}
            text = delta.get('content') or choice.get('text') or ''
            thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
            calls = delta.get('tool_calls')
            self.token_ids.extend(ids)
            kind = 't' if calls else ('r' if thought and not text else 'c')
            if self.keep_content:
                if text and not calls:
                    self.content.append(text)
                for call in calls or ():
                    entry = self.calls.setdefault(call.get('index', len(self.calls)), dict(id=None, name='', arguments=[]))
                    entry['id'] = call.get('id') or entry['id']
                    function = call.get('function') or {}
                    if function.get('name'):
                        entry['name'] += function['name']
                    if function.get('arguments'):
                        entry['arguments'].append(function['arguments'])
            # The continuous usage stats say how many tokens this chunk brought; without them the ids do.
            step = (total - self.reported) if total is not None else len(ids)
            if step > 0:
                if self.first_at is None:
                    self.first_at = self.clock()
                self.chunk_tokens.append(step)
                self.kinds.append(kind)
                self.reported += step
            if choice.get('finish_reason'):
                self.finish = choice['finish_reason']
        return False

    def result(self):
        tool_at = next((sum(self.chunk_tokens[:at]) for at, kind in enumerate(self.kinds) if kind == 't'), None)
        content_at = next((sum(self.chunk_tokens[:at]) for at, kind in enumerate(self.kinds) if kind != 'r'), None)
        calls = [dict(id=entry['id'], type='function', function=dict(name=entry['name'], arguments=''.join(entry['arguments'])))
                 for _, entry in sorted(self.calls.items())]
        return dict(request_id=self.request_id, chunk_tokens=self.chunk_tokens, kinds=''.join(self.kinds),
                    think_tokens=content_at if self.kinds and 'r' in self.kinds else 0, tool_at=tool_at,
                    finish=self.finish, error=self.error, usage=self.usage, output_ids=self.token_ids,
                    content=''.join(self.content), tool_calls=calls,
                    ttft=None if self.first_at is None else self.first_at - self.started)


def http_stream(port, path, body, timeout, host='127.0.0.1', keep_content=False, clock=time.time, deadline=None):
    """One streamed request to the container: the Chunks result plus status. The whole wait is bounded by `timeout` seconds
    (a queued request gets no bytes while it waits, so it is not a per-read bound alone)."""
    state = Chunks(keep_content, clock)
    connection = http.client.HTTPConnection(host, int(port), timeout=min(timeout, REQUEST_READ_TIMEOUT))
    status = None
    try:
        connection.request('POST', path, body=json.dumps(body).encode('utf-8'),
                           headers={'content-type': 'application/json'})
        response = connection.getresponse()
        status = response.status
        if status != 200:
            response.read()
            state.error = 'HTTP %s' % status
        else:
            ends = clock() + timeout
            while True:
                raw = response.readline()
                if not raw or state.feed(raw):
                    break
                if clock() > ends or (deadline is not None and clock() > deadline):
                    state.error = 'turn deadline'
                    break
    except (OSError, http.client.HTTPException, ValueError) as error:
        state.error = 'connection: %s' % type(error).__name__
    finally:
        connection.close()
    result = state.result()
    result['status'] = status
    return result


def turn_ok(result):
    return result.get('error') is None and result.get('finish') in ('stop', 'length', 'tool_calls') \
        and bool(result.get('chunk_tokens'))


def transport_failure(record):
    """A failed turn that says the engine may be hung or gone (no answer, a 5xx, a read timeout), as against a request the
    server refused (a 4xx): what the breaker counts."""
    if record.get('status') == 'ok':
        return False
    http_status = record.get('http')
    if http_status is not None and http_status < 500:
        return False
    return True


def regions_of(output_ids, thinking, result, think_end=THINK_END_TOKEN, tool_call=TOOL_CALL_TOKEN):
    """(think_tokens, tool_at, past_think, tool_call_made) of one answer, for the report's regions and the coverage numbers.
    From the output TOKEN IDS when the answer has them (</think> and <tool_call> are in them even on /v1/completions, where no
    parser splits the stream into kinds), else from the chat stream's chunk kinds. think_tokens is the count through </think>
    (the whole answer when thinking ON never closed it, 0 when thinking is not ON)."""
    if output_ids:
        ended = output_ids.index(think_end) + 1 if think_end in output_ids else None
        called = output_ids.index(tool_call) if tool_call in output_ids else None
        think = (ended if ended is not None else len(output_ids)) if thinking is True else 0
        return think, called, (thinking is not True or ended is not None), called is not None
    think, called = result.get('think_tokens') or 0, result.get('tool_at')
    kinds = result.get('kinds') or ''
    past = thinking is not True or ('c' in kinds or 't' in kinds)
    return think, called, past, called is not None


# -- the lab ---------------------------------------------------------------------------------------------------------

class Results(object):
    """The private results directory: append-only turns.jsonl and outputs.jsonl, read back on resume."""

    def __init__(self, directory):
        self.directory = directory
        self.lock = threading.Lock()
        os.makedirs(directory, mode=0o700, exist_ok=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        self.turns_path = os.path.join(directory, 'turns.jsonl')
        self.outputs_path = os.path.join(directory, 'outputs.jsonl')
        self.done = self._read_done()

    def _lines(self, path):
        if not os.path.isfile(path):
            return []
        out = []
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        return out

    def _read_done(self):
        return dict(((turn['arm'], turn['id']), turn) for turn in self._lines(self.turns_path) if turn.get('status') == 'ok')

    def outputs(self):
        return dict(((entry['arm'], entry['id']), entry) for entry in self._lines(self.outputs_path))

    def add(self, turn, output):
        with self.lock:
            with open(self.turns_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(turn, sort_keys=True) + '\n')
            with open(self.outputs_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(output, sort_keys=True) + '\n')
            if turn.get('status') == 'ok':
                self.done[(turn['arm'], turn['id'])] = turn


def new_counts():
    return dict(planned=0, ok=0, error=0, skipped=0, resumed=0, refused=0, fed_empty=0)


class Lab(object):
    """Runs the arms against one served container. `stream` (path, body, timeout, keep) -> result dict, `clock`, `sleep` and
    `say` are injectable for the CPU tests, as are `alive` (is the container still running) and `canary` (-> a problem
    string or None, run once after CANARY_TURNS ok turns). `corpus` is the prefix_agent_corpus the A4 sessions are filled from."""

    def __init__(self, results, stream, limits, seed=SEED, in_flight=DEFAULT_IN_FLIGHT, max_tokens=None, model=replay.SERVED_NAME,
                 clock=time.time, sleep=time.sleep, say=print, corpus=None, alive=None, canary=None,
                 think_end=THINK_END_TOKEN, tool_call=TOOL_CALL_TOKEN):
        self.results, self.stream, self.limits = results, stream, limits
        self.seed, self.in_flight, self.max_tokens, self.model = seed, in_flight, max_tokens, model
        self.clock, self.sleep, self.say = clock, sleep, say
        self.corpus, self.alive, self.canary = corpus, alive, canary
        self.think_end, self.tool_call = think_end, tool_call
        self.counts = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.tripped = None
        self.failures = 0
        self.ok_total = 0
        self.canary_done = False

    # the breaker and the canary
    def trip(self, reason):
        with self.lock:
            first = self.tripped is None
            self.tripped = self.tripped or reason
        self.stop.set()
        if first:
            self.say('[TAULAB] stopping the lab: %s' % reason)

    def note(self, record):
        """Called for every finished turn: the breaker's count of consecutive transport failures and the canary's trigger."""
        fire = False
        with self.lock:
            if record.get('status') == 'ok':
                self.failures = 0
                self.ok_total += 1
                if self.canary is not None and not self.canary_done and self.ok_total >= CANARY_TURNS:
                    self.canary_done = fire = True
            elif transport_failure(record):
                self.failures += 1
                breaker = self.failures >= BREAKER_FAILURES
            else:
                breaker = False
        if fire:
            problem = self.canary()
            if problem:
                self.trip('canary: %s' % problem)
            else:
                self.say('[TAULAB] canary ok after %d turns' % self.ok_total)
        elif record.get('status') != 'ok' and breaker:
            self.trip('breaker: %d consecutive transport failures' % BREAKER_FAILURES)

    # one turn
    def send(self, arm, job, thinking, max_tokens, ends, turn=None, extra=None, keep_content=False):
        """One turn: -> (turn record, output record). Never raises on a request failure: the turn is recorded failed."""
        spec = ARM_SPEC[arm]
        path, body = request_body(job, thinking, max_tokens, self.model)
        started = self.clock()
        timeout = max(60.0, ends - started + TURN_GRACE_SECONDS)
        try:
            result = self.stream(path, body, timeout, keep_content)
        except Exception as error:  # noqa: BLE001 - one failed request must not take its worker thread (and its seat) with it
            result = dict(error='client: %s' % type(error).__name__, chunk_tokens=[], output_ids=[], status=None)
        usage = result.get('usage') or {}
        think, tool_at, past_think, tool_call = regions_of(result.get('output_ids') or [], thinking, result,
                                                           self.think_end, self.tool_call)
        turn_record = dict(arm=arm, id=job['id'], set=job.get('set', spec['set']), cluster=job.get('cluster', job['id']),
                           turn=turn if turn is not None else job.get('turn'), declared_tokens=job.get('tokens'),
                           weight=job.get('weight', 1.0), thinking=thinking, max_tokens=int(max_tokens), path=path,
                           status='ok' if turn_ok(result) else 'error', finish=result.get('finish'),
                           error=result.get('error'), http=result.get('status'),
                           request_id=result.get('request_id'), chunk_tokens=result.get('chunk_tokens'),
                           kinds=result.get('kinds'), think_tokens=think, tool_at=tool_at, past_think=past_think,
                           tool_call=tool_call, prompt_tokens=usage.get('prompt_tokens'),
                           completion_tokens=usage.get('completion_tokens'),
                           ttft=result.get('ttft'), seconds=round(self.clock() - started, 3),
                           ref_tau=job.get('ref_tau'), started=round(started, 3))
        if extra:
            turn_record.update(extra)
        output = dict(arm=arm, id=job['id'], output_ids=result.get('output_ids') or [])
        if keep_content:
            output['answer'] = dict(content=result.get('content') or '', tool_calls=result.get('tool_calls') or [],
                                    finish=result.get('finish'))
        self.results.add(turn_record, output)
        self.note(turn_record)
        return turn_record, output

    def fits(self, job, max_tokens):
        limit = self.limits.get('prompt')
        declared = job.get('tokens')
        return limit is None or declared is None or declared <= limit

    def bump(self, arm, key, amount=1):
        with self.lock:
            self.counts.setdefault(arm, new_counts())[key] += amount

    def tally(self, arm, turn):
        self.bump(arm, 'ok' if turn['status'] == 'ok' else 'error')

    # an arm
    def arm_max_tokens(self, arm, entry):
        return int(self.max_tokens or entry.get('max_tokens') or ARM_SPEC[arm]['max_tokens'])

    def run_pool(self, arm, jobs, work, workers, ends):
        """Run `work(job)` for every job on `workers` threads, taking no new job after `ends`, once the lab is stopped or
        when the container is gone."""
        todo = queue.Queue()
        for job in jobs:
            todo.put(job)

        def loop():
            while not self.stop.is_set():
                try:
                    job = todo.get_nowait()
                except queue.Empty:
                    return
                if self.clock() >= ends:
                    self.bump(arm, 'skipped')
                    continue
                if self.alive is not None and not self.alive():
                    self.trip('the container is not running')
                    self.bump(arm, 'skipped')
                    return
                work(job)
            while True:                  # stopped: what is still queued is skipped, not sent
                try:
                    todo.get_nowait()
                except queue.Empty:
                    return
                self.bump(arm, 'skipped')

        threads = [threading.Thread(target=loop) for _ in range(max(1, min(workers, len(jobs) or 1)))]
        for thread in threads:
            thread.daemon = True
            thread.start()
        for thread in threads:
            thread.join()

    def run_turns(self, arm, records, thinking, entry, ends):
        max_tokens = self.arm_max_tokens(arm, entry)
        jobs = []
        for record in records:
            self.bump(arm, 'planned')
            if (arm, record['id']) in self.results.done:
                self.bump(arm, 'resumed')
            elif not self.fits(record, max_tokens):
                self.bump(arm, 'refused')
            elif thinking is False and record.get('messages') is None and not record.get('prompt_ids_nothink'):
                self.bump(arm, 'refused')
            else:
                jobs.append(record)

        def work(job):
            self.tally(arm, self.send(arm, job, thinking, max_tokens, ends)[0])

        self.run_pool(arm, jobs, work, int(entry.get('in_flight') or self.in_flight), ends)

    def run_calibration(self, arm, records, entry, ends):
        max_tokens = self.arm_max_tokens(arm, entry)
        for wave in calibration_waves(records):
            if self.stop.is_set():
                for record in wave:
                    self.bump(arm, 'planned')
                    self.bump(arm, 'skipped')
                continue
            jobs = []
            for record in wave:
                self.bump(arm, 'planned')
                if (arm, record['id']) in self.results.done:
                    self.bump(arm, 'resumed')
                elif not self.fits(record, max_tokens):
                    self.bump(arm, 'refused')
                else:
                    jobs.append(record)
            self.run_pool(arm, jobs, lambda job: self.tally(arm, self.send(arm, job, False, job.get('max_tokens') or max_tokens,
                                                                        ends)[0]), len(jobs), ends)

    # A4: one chained session
    def start_session(self, session):
        """The session's conversation, filled from the corpus as FORMAT.md says: a real .py file and one of its functions into
        the seed's task template, a real excerpt attached when first_tokens > 0, then prefix_agent_corpus.Conversation."""
        import prefix_agent_corpus as corpus_module
        rng = random.Random('%s|%s|task' % (session['seed'], session['name']))
        source = self.corpus.pick(rng, suffixes=('.py',))
        name = self.corpus.function_name(rng, source)
        text = session['task_template'].format(path=source.path, name=name)
        if session['first_tokens'] > 0:
            text += '\n\nHere is the part of %s I am looking at:\n```\n%s\n```' % (
                source.path, self.corpus.excerpt(rng, corpus_module.token_chars(session['first_tokens']), hint=source.path))
        return corpus_module.Conversation(self.corpus, session['name'], session['seed'], session['system'],
                                          task_messages=[dict(role='user', content=text)])

    def run_chains(self, arm, records, entry, ends):
        import prefix_agent_corpus as corpus_module
        max_tokens = self.arm_max_tokens(arm, entry)
        stored = self.results.outputs()

        def work(session):
            conversation = self.start_session(session)
            targets = session['prompt_targets']
            for turn in range(session['turns']):
                turn_id = '%s.t%d' % (session['id'], turn)
                self.bump(arm, 'planned')
                done = self.results.done.get((arm, turn_id))
                if done is not None:
                    self.bump(arm, 'resumed')
                    answer = (stored.get((arm, turn_id)) or {}).get('answer') or {}
                    prompt_tokens, completion_tokens = done.get('prompt_tokens'), done.get('completion_tokens')
                else:
                    if self.clock() >= ends or self.stop.is_set():
                        self.bump(arm, 'skipped')
                        return
                    body = conversation.body()
                    job = dict(id=turn_id, set='chained', cluster=session['cluster'], messages=body['messages'],
                               tools=body['tools'], weight=1.0)
                    record, output = self.send(arm, job, True, max_tokens, ends, turn=turn, keep_content=True)
                    self.tally(arm, record)
                    if record['status'] != 'ok':
                        return          # the chain cannot go on without its answer
                    answer = output.get('answer') or {}
                    prompt_tokens, completion_tokens = record.get('prompt_tokens'), record.get('completion_tokens')
                conversation.add_answer(dict(content=answer.get('content') or '', tool_calls=answer.get('tool_calls') or [],
                                             finish=answer.get('finish')))
                last = conversation.last_assistant() or {}
                if not (last.get('content') or '').strip() and not last.get('tool_calls'):
                    self.bump(arm, 'fed_empty')       # the answer ended inside its reasoning: the chain goes on from nothing
                if turn + 1 < session['turns']:
                    conversation.extend(corpus_module.growth_input(targets[turn + 1], prompt_tokens or 0, completion_tokens or 0))

        self.run_pool(arm, records, work, int(entry.get('in_flight') or self.in_flight), ends)

    def run(self, arms, data, manifest, send_ends):
        """Every arm in order; each gets the time left in proportion to the design's wall weights of the arms still to run,
        so a slow arm cannot starve the later ones, and time an arm does not use goes to the next. -> {arm: counts}."""
        entries = manifest.get('arms') or {}
        pending = [arm for arm in arms]
        for arm in arms:
            entry = entries.get(arm) or {}
            if not isinstance(entry, dict):
                entry = {}
            entry = dict((key, value) for key, value in entry.items() if key in ('max_tokens', 'in_flight'))
            now = self.clock()
            left = max(0.0, send_ends - now)
            weight = sum(ARM_SPEC[name]['weight'] for name in pending)
            ends = now + left * ARM_SPEC[arm]['weight'] / float(weight)
            pending.remove(arm)
            records = data.get(arm) or []
            started = self.clock()
            self.counts.setdefault(arm, new_counts())
            kind = ARM_SPEC[arm]['kind']
            if kind == 'turns':
                self.run_turns(arm, order_records(records, self.seed, arm), ARM_SPEC[arm]['thinking'], entry, ends)
            elif kind == 'calib':
                self.run_calibration(arm, records, entry, ends)
            else:
                self.run_chains(arm, order_records(records, self.seed, arm), entry, ends)
            self.say('[TAULAB] arm %s: %s in %.0f s of %.0f' % (arm, ' '.join('%s=%d' % pair for pair in sorted(
                self.counts[arm].items())), self.clock() - started, ends - started))
            self.sleep(SETTLE_SECONDS)
            if self.stop.is_set():
                break
        return self.counts


# -- the container and the job ---------------------------------------------------------------------------------------

def docker_run(arguments, timeout=600):
    try:
        result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        return result.returncode, (result.stdout or b'').decode('utf-8', 'replace')
    except (OSError, subprocess.SubprocessError) as error:
        return None, repr(error)[:300]


def wait_ready(client, container, clock, sleep, seconds=READINESS_SECONDS):
    ends = clock() + seconds
    while clock() < ends:
        if not container.running():
            return 'the container exited before its API answered'
        if client.ready():
            return None
        sleep(10)
    return 'no /v1/models answer in %d s' % seconds


def launched_problems(log_lines, derived):
    """What the launched container's own log says against what was asked (memory read-the-launched-argv): the contract's
    '[QWEN-C2]' lines must name the derived profile. [str]."""
    text = '\n'.join(log_lines)
    problems = []
    if derived not in text:
        problems.append('the container log never names the derived profile %s' % derived)
    return problems


def make_canary(log_path, results, clock=time.time, sleep=time.sleep, wait=CANARY_WAIT_SECONDS):
    """The canary the lab runs once after its first ok turns: the container's log must hold [PACKED] rounds that belong to
    turns the lab sent and fast_serving_phases records (the two log flags are then in effect), within `wait` s (the log
    follower lags the container). -> a callable returning a problem string or None."""
    def check():
        ends = clock() + wait
        problem = 'no log'
        while True:
            text = ''
            if os.path.isfile(log_path):
                with open(log_path, encoding='utf-8', errors='replace') as handle:
                    text = handle.read()
            index = dict((turn['request_id'], key) for key, turn in list(results.done.items()) if turn.get('request_id'))
            rounds = report.parse_rounds(text)
            attributed = sum(1 for request_id, entries in rounds.items()
                             if report.owner_of(request_id, index) is not None
                             and any(entry['kind'] == 'P' and entry.get('emitted') for entry in entries))
            phases = text.count('"stage": "fast_serving_phases"')
            if attributed and phases:
                return None
            problem = 'no [PACKED] rounds of the lab\'s turns in the log' if not attributed else 'no fast_serving_phases records'
            if clock() >= ends:
                return problem + ' (the log flags are not in effect)'
            sleep(5)
    return check


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', default=None)
    parser.add_argument('--profile', default='c2-packed-tp4')
    parser.add_argument('--data', required=True, help='the rig-local data directory (read only)')
    parser.add_argument('--results', required=True, help='the private results directory (rig-local)')
    parser.add_argument('--public', default=None, help='where the aggregates-only summary goes (the uploaded artifact)')
    parser.add_argument('--arms', default=' '.join(ARMS))
    parser.add_argument('--deadline-seconds', type=int, default=16200,
                        help='the whole run, the container load included; the last %d s are the stop and the report'
                        % RESERVE_SECONDS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--in-flight', type=int, default=DEFAULT_IN_FLIGHT)
    parser.add_argument('--max-tokens', type=int, default=None, help='every arm\'s answer budget (default per arm; A3 keeps its '
                                                                    'own 2400, the lanes\' budget)')
    parser.add_argument('--counters', default=None, help='production\'s spec-decode counters (a JSON file; aggregates only)')
    parser.add_argument('--profiles', default=None, help='a profiles JSON instead of the image\'s own')
    parser.add_argument('--cards', choices=('quad',), default='quad')
    parser.add_argument('--hub', default=gate.HUB)
    parser.add_argument('--port', type=int, default=PORT)
    parser.add_argument('--dry-run', action='store_true', help='check the data and print the plan (counts) and the docker argv; '
                                                              'run nothing; exits nonzero on any shortfall')
    parser.add_argument('--analyze-only', action='store_true', help='rerun the report over --results; no container')
    return parser


def parse_arms(text):
    arms = [arm for arm in re.split(r'[,\s]+', text.strip()) if arm]
    known = ARMS + OPTIONAL_ARMS
    unknown = sorted(set(arms) - set(known))
    if not arms or unknown:
        raise LabError('unknown arm(s): %s (known: %s)' % (', '.join(unknown or ['(none)']), ' '.join(known)))
    if len(set(arms)) != len(arms):
        raise LabError('an arm named twice')
    return [arm for arm in known if arm in arms]


def load_corpus(checkout):
    """The corpus the A4 sessions are filled from: this checkout's own scripts/ci, docs, docker and workflows (public text)."""
    import prefix_agent_corpus as corpus_module
    try:
        return corpus_module.Corpus(corpus_module.load_sources(checkout))
    except (OSError, ValueError):
        raise LabError('the A4 corpus (the checkout\'s own files) could not be read')


def analyze(options, say, run_info=None):
    """The report over the results directory -> (code, public summary)."""
    info = run_info
    if info is None:
        try:
            with open(os.path.join(options.results, 'run.json'), encoding='utf-8') as handle:
                info = json.load(handle)
        except (OSError, ValueError):
            info = {}
    manifest = {}
    try:
        manifest['a3_reference'] = read_a3_reference()
    except LabError:
        pass            # the report then reads NOT_ESTABLISHED; a RUN refuses before it starts (main)
    counters = None
    if options.counters:
        with open(options.counters, encoding='utf-8') as handle:
            counters = json.load(handle)
    return report.build_and_write(options.results, options.public, manifest=manifest, counters=counters,
                                  seed=options.seed, say=say, info=info)


def main(argv=None, say=print, make_stream=None, make_client=None, make_container=None, make_log=None, docker=None,
         devices=None, clock=time.time, sleep=time.sleep, profiles=None, build_a3=None, corpus=None, make_canary_check=None):
    options = build_parser().parse_args(argv)
    try:
        arms = parse_arms(options.arms)
    except LabError as error:
        say('refused: %s' % error)
        return 2
    if options.analyze_only:
        summary = analyze(options, say)
        return 0 if summary and summary.get('complete') else 1
    # Everything the run needs is checked BEFORE a container starts: the data against its manifest and its arm counts, the A3
    # reference, the A4 corpus. Any shortfall is a refusal (and --dry-run exits nonzero on it): an empty arm is never a run.
    checkout = os.path.dirname(os.path.dirname(HERE))
    try:
        if not os.path.isdir(options.data):
            raise LabError('the data directory does not exist')
        data, manifest, scrub = load_data(options.data, arms)
        reference = read_a3_reference() if 'A3' in arms else None
        if 'A4' in arms and corpus is None:
            corpus = load_corpus(checkout)
    except (LabError, OSError, ValueError) as error:
        say('refused: %s' % (error if isinstance(error, LabError) else 'the data could not be read: %s' % type(error).__name__))
        return 2
    manifest = dict(manifest, a3_reference=reference) if reference else manifest
    marker = manifest.get('tokens') if isinstance(manifest.get('tokens'), dict) else {}
    think_end = int(marker.get('think_end_token_id') or THINK_END_TOKEN)
    tool_call = int(marker.get('tool_call_token_id') or TOOL_CALL_TOKEN)
    if not options.image:
        say('refused: --image is required')
        return 2
    started = clock()
    if profiles is None:
        try:
            if options.profiles:
                with open(options.profiles, encoding='utf-8') as handle:
                    profiles = json.load(handle)
            else:
                profiles = gate.image_profiles(options.image)
        except Exception as error:  # noqa: BLE001 - the traceback could name an image or a host: only its type goes out
            say('refused: the image\'s profiles could not be read (%s)' % type(error).__name__)
            return 2
    try:
        derived, document = derive_profile(profiles, options.profile)
    except LabError as error:
        say('refused: %s' % error)
        return 2
    extra = arithmetic_diff(profiles, options.profile, document)
    if extra:
        say('refused: the derived profile changes arithmetic: %s' % ', '.join(extra))
        return 2
    audits = audits_on(profiles, options.profile)
    production = is_production_image(options.image) and audits
    limits = dict(prompt=pg.prompt_limit(profiles, options.profile))
    say('[TAULAB] data: %s' % ' '.join('%s=%d' % pair for pair in sorted(plan_counts(data, arms).items())))
    if scrub:
        say('[TAULAB] A2 scrub report: kept=%s dropped=%s, final check clean' % (scrub.get('kept'), scrub.get('dropped')))
    say('[TAULAB] profile %s -> %s, image %s, verify audits on=%s, production=%s' % (
        options.profile, derived, 'tag ' + image_tag(options.image), audits, production))
    try:
        devices = devices if devices is not None else (None if options.dry_run else gate.devices_for(options.cards))
    except Exception as error:  # noqa: BLE001 - the card-set error lists the boards' ids: only its type goes out
        say('refused: the card set could not be resolved (%s)' % type(error).__name__)
        return 2
    results_dir = os.path.abspath(options.results)
    derived_path = os.path.join(results_dir, 'profiles.json')
    arguments = pg.server_run(options.image, CONTAINER, derived, devices or ['<card %d>' % n for n in range(4)],
                              options.port, options.hub, derived_path)
    if options.dry_run:
        say(json.dumps(arguments))
        return 0
    if gate.platform_containers():
        say('refused: a platform serving container exists')
        return 2
    docker = docker or docker_run
    calibration = {}
    if 'A3' in arms:
        # The prompts are built from the IMAGE itself (its installed vLLM source), before the lab container takes the cards.
        try:
            built, calibration = (build_a3 or (lambda: build_calibration(
                options.image, os.path.join(options.data, ARM_FILES['A3'][0]), checkout, options.hub)))()
        except LabError as error:
            say('refused: %s' % error)
            return 2
        attach_calibration(data['A3'], built)
        say('[TAULAB] A3 prompts built from the image: %s' % ' '.join('%s=%d' % (name, len(users)) for name, users in sorted(built.items())))
    results = Results(results_dir)
    with open(derived_path, 'w', encoding='utf-8') as handle:
        json.dump(document, handle, indent=1, sort_keys=True)
    with open(os.path.join(results_dir, 'docker-run.json'), 'w') as handle:
        json.dump(arguments, handle, indent=1)
    client = (make_client or (lambda: replay.Client(options.port)))()
    container = (make_container or (lambda name: replay.Container(name)))(CONTAINER)
    log_path = os.path.join(results_dir, 'server.log')
    follower = (make_log or (lambda name, path: replay.LogFollower(name, path)))(CONTAINER, log_path)
    info = dict(image_tag=image_tag(options.image), production=production, audits_on=audits, profile=options.profile,
                derived=derived, seed=options.seed, arms=arms, arithmetic_extra=extra, scrub=scrub, data_counts=plan_counts(data, arms),
                calibration_corpus=calibration, started=started, counts={})
    status = 1
    lab = None
    try:
        docker(['docker', 'rm', '-f', CONTAINER], 120)
        code, output = docker(arguments, 300)
        if code != 0:
            raise RuntimeError('docker run exited %s' % code)
        follower.start()
        problem = wait_ready(client, container, clock, sleep, max(60, min(READINESS_SECONDS, options.deadline_seconds)))
        if problem:
            raise RuntimeError(problem)
        info['ready_seconds'] = round(clock() - started, 1)
        say('[TAULAB] ready after %.0f s' % info['ready_seconds'])
        context = client.context_tokens()
        info['context_tokens'] = context
        problems = launched_problems(follower.lines(), derived)
        info['launched_problems'] = problems
        for problem in problems:
            say('[TAULAB] WARNING: %s' % problem)
        stream = (make_stream or (lambda: lambda path, body, timeout, keep: http_stream(
            options.port, path, body, timeout, keep_content=keep)))()
        lab = Lab(results, stream, limits, seed=options.seed, in_flight=options.in_flight, max_tokens=options.max_tokens,
                  clock=clock, sleep=sleep, say=say, corpus=corpus, alive=container.running, think_end=think_end,
                  tool_call=tool_call)
        lab.canary = (make_canary_check or make_canary)(log_path, results)
        send_ends = started + options.deadline_seconds - RESERVE_SECONDS
        lab.run(arms, data, manifest, send_ends)
        status = 1 if lab.tripped else 0
    except Exception as error:  # noqa: BLE001 - the container must still be stopped and the report still written
        info['error'] = type(error).__name__
        say('[TAULAB] stopped: %s' % type(error).__name__)
    finally:
        if lab is not None:
            lab.stop.set()
            info['counts'] = lab.counts          # whatever ran, so a crash cannot read as complete
            info['tripped'] = lab.tripped
        sleep(SETTLE_SECONDS)
        docker(['docker', 'stop', '-t', '120', CONTAINER], 300)
        sleep(SETTLE_SECONDS)
        docker(['docker', 'rm', '-f', CONTAINER], 120)
        info['finished'] = clock()
        with open(os.path.join(results_dir, 'run.json'), 'w', encoding='utf-8') as handle:
            json.dump(info, handle, indent=1, sort_keys=True, default=str)
    summary = analyze(options, say, info)
    complete = bool(summary and summary.get('complete')) and status == 0
    say('C2_TAULAB profile=%s complete=%s' % (options.profile, complete))
    return 0 if complete else 1


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main())
