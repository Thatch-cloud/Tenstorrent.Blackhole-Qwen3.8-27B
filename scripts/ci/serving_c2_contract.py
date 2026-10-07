"""Qwen C2 as a served model: what a self-contained serving image enforces for itself.

The Thatch node agent starts the serving image's own entrypoint (serving.server), which
launches vLLM with arguments built from platform kwargs and forwards only allow-listed
environment. The C2 fast path was measured under one exact vLLM argv and environment
(gate tags v235/v238, lever_n_m3native_run_arm.sh), so this module - booted from a .pth
hook in every python process of the image when QWEN_C2_SERVING=1 - makes the served
process match it:

1. sys.path: the fast path's tree (the base image's PYTHONPATH), which the Thatch layer
   replaces with its own.
2. environment: the selected profile's values (context, output budget), the p150_x2 mesh
   descriptor over the agent's p300 default, and QWEN36_BATCHED_DECODE_MODE removed (the
   gate never set it; the Thatch layer must bake it).
3. argv: in the vLLM API server, the engine arguments the profile owns replace whatever
   the platform passed. The served model name, host, port and the reasoning and tool-call
   parsers stay the platform's.
4. requests: the fast path decodes greedily and holds OUTPUT_BUDGET tokens per request, so
   the API edge coerces sampling to greedy, clamps max_tokens, and refuses what it cannot
   serve before the engine sees it. A refusal inside the engine (serving_fast_policy.
   validate_request_sampling) fails the engine, not the request - except under the c2
   profiles (QWEN_FAST_ANY_REQUEST=1), where a host-side refusal of one request's own terms
   (its sampling contract, budget or page table) ends only that request, as FINISHED_ABORTED
   (serving_request_quarantine); every other in-engine refusal still fails the engine.
5. prefix reuse (the TT prefix-reuse design, G1): the profile alone owns QWEN_PREFIX_REUSE, the switch
   every prefix-reuse graft in the image reads. A profile that sets it must also carry the engine
   flags reuse is exact under (prefix_reuse_problems), or the boot refuses; a profile that does not
   set it gets it removed, so an inherited value cannot turn half of reuse on under exact, c2 or
   general. Under a prefix profile the API server serves the registry's metrics, which the engine
   process exports (qwen_prefix_metrics); it keeps a request's cache_salt only when the salt verifies
   against the platform's salt key (salt_verdict), so a client cannot choose the cache partition it
   shares; it refuses a launched KV connector and warns on a server default that strips past
   reasoning (prefix_launch_problems); and every process logs where the model entry was imported
   from (the bring-up check, qwen_prefix_stage bringup). The fast path joins prefix reuse only
   through sticky sessions (QWEN_FAST_STICKY_SESSIONS, the c2-packed-prefix profiles): the profile
   alone owns that switch too, it needs QWEN_PREFIX_REUSE=1 and the fast path beside it, and the
   one speculative shape reuse then accepts is DFlash with 15 proposals.
6. gate-only profiles (gate_only: true) boot only with QWEN_C2_GATE=1: they exist for a gate and
   must never take traffic.
7. streaming parsers: C2 commits 3-16 tokens per engine step, so the API server's reasoning and
   tool-call parsers see multi-token deltas, which they parse differently from one token per step
   (R21: a real </think> or <tool_call> bound to a later lookalike, whitespace after a tool call).
   A profile with "parser_rechunk": true (c2, c2-gate) arms c2_parser_rechunk (fix M) in the API
   server: when vLLM imports vllm.parser.abstract_parser, DelegatingParser.parse_delta is wrapped
   to feed the parsers one token per sub-delta where that matters and merge the results into one
   message per step. general, the general-prefix profiles and exact never arm it (general and
   general-prefix decode one token per step).
8. meshes: a profile that names no mesh_device serves the pair exactly as before - the image's MESH_DEVICE
   (P300, a (1, 2) mesh) under the p150_x2 descriptor. A profile with mesh_device P150x4 (the general-tp4
   family) serves all four cards as one (1, 4) mesh: the contract sets MESH_DEVICE from it, the profile names
   the four-card ring's descriptor (tp4_mesh: 2x2, two channels, laid by the overlay), and every process
   installs the ring check on the TT plugin's open_mesh_device (tp4_mesh.install_ring_check: a broken ring
   stops the engine, a degraded one is logged; QWEN_TP4_RING_CHECK=0 skips it). mesh_problems refuses a mesh
   and descriptor that disagree, the fast path anywhere but the pair - or the four-card mesh under a profile that
   sets QWEN_FAST_TP=4 (the c2-packed-tp4 profiles: tp_shapes, its own kernels' siblings and its own evidence,
   packed_any_evidence_tp4.json, which admission requires) - and on-device sampling on a mesh whose vocabulary shard
   exceeds the sampler's 65,536
   logits (the pair's 124,160). The TP4 profiles sample on device (sample_on_device_mode decode_only; the TT
   plugin still falls back to host sampling per batch for what the device sampler cannot do); the file
   HOST_SAMPLING_FILE or QWEN_HOST_SAMPLING=1 drops it at boot, so every batch samples on the host.

Nothing here changes a gate: every step is off unless QWEN_C2_SERVING=1.
"""

import hashlib
import hmac
import json
import os
import re
import sys

PROFILES = '/opt/qwen-c2/profiles.json'
FAST_PATHS = ('/experiment-scripts/ci', '/speculative-decoding/harness', '/opt/tt-metal/ttnn', '/opt/tt-metal')
API_SERVER = 'vllm.entrypoints.openai.api_server'
INPUT_PROCESSOR = 'vllm.v1.engine.input_processor'
# The prefix-reuse switch (design section 2.0.1): the model graft's supports_prefix_caching, the
# scheduler graft's install and the runner patch all read it. Only a profile sets it.
PREFIX_SWITCH = 'QWEN_PREFIX_REUSE'
# Sticky sessions (serving_fast_policy.STICKY_SESSIONS_FLAG): the fast path's side of prefix reuse. Only a
# profile sets it, and only beside PREFIX_SWITCH and the fast path (prefix_reuse_problems).
STICKY_SWITCH = 'QWEN_FAST_STICKY_SESSIONS'
# The one speculative shape sticky sessions serve (the fast path's policy, serving_fast_policy).
STICKY_SPECULATION = dict(method='dflash', num_speculative_tokens=15)
# vLLM's default (config/cache.py:95), pinned in the prefix profiles: the block-hash chain is what makes
# a cached block's content its prompt's, and the exactness argument leans on it (design L2).
PREFIX_HASH_ALGO = 'sha256'
# The model entry every process logs the source of under a prefix profile (the bring-up check).
MODEL_ENTRY = 'models.demos.blackhole.qwen36.tt.qwen36_vllm'
# A profile with gate_only: true boots only with this set to 1.
GATE_SWITCH = 'QWEN_C2_GATE'
# Lever N at TP4 (levern_policy.FLAG and ALL_FLAGS, pinned equal by test_levern_contract): chunked prefill interleaved with the packed
# decode rounds. A gate-only profile's own switch; the platform's chunking policy is wrapped, the scheduler caps and alternates, and the
# model route continues its own suspended scratch. LEVERN_AUDIT is the digest instrument and also stands alone, as the NON-interleaved control.
LEVERN_SWITCH = 'QWEN_FAST_LEVER_N'
LEVERN_AUDIT = 'QWEN_FAST_LEVERN_AUDIT'
LEVERN_ENV_FLAGS = ('QWEN_FAST_LEVER_N', 'QWEN_FAST_LEVERN_AUDIT', 'QWEN_FAST_LEVERN_STEP_TOKENS', 'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS',
                    'QWEN_FAST_LEVERN_PREFILL_SHARE', 'QWEN_FAST_LEVERN_ROUNDS', 'QWEN_FAST_LEVERN_MAX_ROUNDS', 'QWEN_FAST_LEVERN_FAULT',
                    'QWEN_FAST_LEVERN_TTFT_TARGET_S', 'QWEN_FAST_LEVERN_SHORT_TOKENS', 'QWEN_FAST_LEVERN_PARK', 'QWEN_FAST_LEVERN_PARK_SLOTS',
                    'QWEN_FAST_LEVERN_MAX_PARK_S', 'QWEN_FAST_LEVERN_EPOCH_SCOPE')
LEVERN_PLATFORM_MODULE = 'vllm_tt_plugin.platform'
# The sources the merged route must carry (levern_route.SOURCES, pinned equal by test_levern_prefix_contract).
MERGED_SOURCES = ('COLD', 'CHECKPOINT', 'SCRATCH', 'PARKED')

# The meshes a profile may open (item 8), keyed by its mesh_device: None is every profile that names none (the
# pair, under the image's own MESH_DEVICE=P300), P150x4 the four-card (1, 4) ring (tp4_mesh.DESCRIPTOR_PATH,
# where docker/qwen-c2-overlay.txt lays scripts/ci's descriptor).
PAIR_DESCRIPTOR = '/opt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.textproto'
RING_DESCRIPTOR = '/opt/qwen-c2/mesh/qwen_p150x4_ring_mesh_graph_descriptor.textproto'
# The pair at the two links this cabling trains (M-A: 2, where p150_x2 declares 4, which STRICT_INIT refuses at
# fabric init): only the general-2link profile, the TP2 reference and baseline for TP4, names it. mesh_device
# P300 is the image's own MESH_DEVICE for the pair, so nothing else about the open changes.
PAIR_2LINK_DESCRIPTOR = '/opt/qwen-c2/mesh/qwen_p150x2_2link_mesh_graph_descriptor.textproto'
MESHES = {
    None: dict(shape=(1, 2), descriptor=PAIR_DESCRIPTOR),
    'P300': dict(shape=(1, 2), descriptor=PAIR_2LINK_DESCRIPTOR),
    'P150x4': dict(shape=(1, 4), descriptor=RING_DESCRIPTOR),
}
# The on-device sampler takes at most this many logits per device (qwen36 model.py; vocabulary 248,320).
VOCABULARY = 248320
SAMPLER_MAX_LOGITS = 65536
SAMPLE_ON_DEVICE_MODES = ('decode_only', 'all')
# Host sampling for every batch of a profile that samples on device: this file on the persistent mount (an
# operator's switch, like the prefix-reuse kill switch) or QWEN_HOST_SAMPLING=1 drops sample_on_device_mode.
HOST_SAMPLING_FILE = '/models/.qwen-c2/device-sampling.off'
HOST_SAMPLING_ENV = 'QWEN_HOST_SAMPLING'
# The ring check on the TT plugin's open_mesh_device (tp4_mesh.install_ring_check); 0 skips it.
RING_CHECK_ENV = 'QWEN_TP4_RING_CHECK'
# The fast path's tensor-parallel width (tp_shapes.TP_SWITCH): only the four-card fast profiles set it, to 4.
FAST_TP_ENV = 'QWEN_FAST_TP'
PLUGIN_WORKER = 'vllm_tt_plugin.worker'

# Client cache_salt under a prefix profile (design 2.2 need 4, decision D-P2). A salt partitions vLLM's
# prefix cache and the checkpoint registry: requests with one salt share KV blocks and checkpoints, and
# each can time the other's hits. The platform must choose it per tenant, never the client (an SDK's
# constant salt would join tenants). So the API server keeps a request's cache_salt only when it
# verifies against the platform's salt key, and drops it otherwise: an unsalted request gets no hit and
# publishes nothing (the scheduler graft's fail-closed rule). A verifiable salt is
# 'qps1.<tag>.<mac>' - tag 8-128 of [A-Za-z0-9_-], the gateway's opaque per-tenant value (for example
# HMAC(tenant secret, tenant_id); never the displayed tenant id), mac the hex HMAC-SHA256 of 'qps1.<tag>'
# under the key (mint_salt). The key is the file SALT_KEY_ENV names (default SALT_KEY_FILE, on the
# persistent mount beside the kill switch), at least SALT_KEY_MIN_BYTES bytes, read once at boot. Without
# it every salt is dropped and reuse is inert: general-prefix then serves as general does, with no hit.
# The format is this image's PROPOSAL for D-P2; the ADM gateway must mint it before reuse does anything.
SALT_KEY_FILE = '/models/.qwen-c2/prefix-salt.key'
SALT_KEY_ENV = 'QWEN_PREFIX_SALT_KEY_FILE'
SALT_VERSION = 'qps1'
SALT_TAG = re.compile(r'\A[A-Za-z0-9_-]{8,128}\Z')
SALT_MAC = re.compile(r'\A[0-9a-f]{64}\Z')
SALT_KEY_MIN_BYTES = 32

# Engine flags a profile owns, and whether each takes a value. A platform value for any of
# these is dropped: the fast path is only qualified under the profile's.
OWNED_FLAGS = {
    'model': True, 'dtype': True, 'max-model-len': True, 'max-num-seqs': True,
    'max-num-batched-tokens': True, 'block-size': True, 'num-gpu-blocks-override': True,
    'limit-mm-per-prompt': True, 'shutdown-timeout': True, 'additional-config': True,
    'speculative-config': True, 'gpu-memory-utilization': True, 'kv-cache-dtype': True,
    'enable-prefix-caching': False, 'no-enable-prefix-caching': False,
    'async-scheduling': False, 'no-async-scheduling': False,
    'enable-chunked-prefill': False, 'no-enable-chunked-prefill': False,
    'enforce-eager': False, 'no-enforce-eager': False,
    'prefix-caching-hash-algo': True,
}


class ContractError(ValueError):
    """A request the C2 fast path cannot serve; vLLM returns it to the client as a 400."""


def log(message, *values):
    try:
        sys.stderr.write('[QWEN-C2] ' + (message % values if values else message) + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def load_profile(path=PROFILES, name=None):
    with open(path, encoding='utf-8') as handle:
        profiles = json.load(handle)
    name = name or os.environ.get('QWEN_C2_PROFILE') or profiles['default']
    if name not in profiles['profiles']:
        raise ValueError('QWEN_C2_PROFILE %r is not one of %s' % (name, sorted(profiles['profiles'])))
    profile = dict(profiles['profiles'][name])
    profile['name'] = name
    return profile


def fix_sys_path(path=None):
    """Put the fast path's tree back, after the Thatch runtime's own entries so its
    `serving` package still wins, and ahead of site-packages as PYTHONPATH had it."""
    path = sys.path if path is None else path
    anchor = 1 if path and path[0] in ('', os.getcwd()) else 0
    for index, entry in enumerate(path):
        if entry.rstrip('/') == '/opt/thatch/py':
            anchor = index + 1
    for offset, entry in enumerate(entry for entry in FAST_PATHS if entry not in path):
        path.insert(anchor + offset, entry)
    return path


def resolve_snapshot(profile, exists=os.path.isdir):
    for candidate in profile['snapshots']:
        if exists(candidate):
            return candidate
    raise ValueError('None of the pinned target snapshots is mounted: %s' % ', '.join(profile['snapshots']))


def mesh_of(profile):
    """The MESHES entry of a profile (its mesh_device, or the pair's when it names none)."""
    device = profile.get('mesh_device')
    if device not in MESHES:
        raise ValueError('mesh_device %r is not one of %s' % (device, ', '.join(sorted(
            name for name in MESHES if name is not None))))
    return MESHES[device]


def tt_config(profile):
    return ((profile.get('engine') or {}).get('additional-config') or {}).get('tt') or {}


def mesh_problems(profile):
    """Every way the profile's mesh, descriptor, fast path and sampling disagree (item 8), [] when none."""
    try:
        mesh = mesh_of(profile)
    except ValueError as error:
        return [str(error)]
    problems = []
    if profile.get('mesh_graph_descriptor') != mesh['descriptor']:
        problems.append('mesh %s needs the descriptor %s, not %s' % (
            profile.get('mesh_device') or 'P300 (the pair)', mesh['descriptor'], profile.get('mesh_graph_descriptor')))
    if 'MESH_DEVICE' in (profile.get('env') or {}):
        problems.append("MESH_DEVICE is the profile's mesh_device, never an env value")
    rows, cols = mesh['shape']
    fast = ((profile.get('engine') or {}).get('additional-config') or {}).get('qwen_fast_t16')
    four_card = fast and profile.get('mesh_device') == 'P150x4' and (profile.get('env') or {}).get(FAST_TP_ENV) == '4'
    if fast and profile.get('mesh_device') and not four_card:
        problems.append('the fast path (qwen_fast_t16) serves the p150_x2 pair only, or the four-card (1, 4) mesh '
                        'under %s=4: its kernels, per-chip widths, link policy (sampling_link_policy pins that '
                        'descriptor) and qualification evidence are two-chip at four links; a (%d, %d) mesh under %s '
                        'serves the general profiles' % (FAST_TP_ENV, rows, cols, profile['mesh_device']))
    if (profile.get('env') or {}).get(FAST_TP_ENV) == '4' and (profile.get('mesh_device') != 'P150x4' or not fast):
        problems.append('%s=4 is the four-card width of the fast path: it needs mesh_device P150x4 and qwen_fast_t16, '
                        'and serving_startup refuses a width the opened mesh does not have' % FAST_TP_ENV)
    if four_card and tt_config(profile).get('sample_on_device_mode') is not None:
        problems.append('the four-card fast path verifies with its own shard argmax (62,080 columns per chip): a '
                        'profile that names sample_on_device_mode would sample twice')
    mode = tt_config(profile).get('sample_on_device_mode')
    if mode is not None:
        if mode not in SAMPLE_ON_DEVICE_MODES:
            problems.append('sample_on_device_mode %r is not one of %s' % (mode, ', '.join(SAMPLE_ON_DEVICE_MODES)))
        per_device = -(-VOCABULARY // (rows * cols))
        if per_device > SAMPLER_MAX_LOGITS:
            problems.append('on-device sampling needs at most %d logits per device; a (%d, %d) mesh has %d'
                            % (SAMPLER_MAX_LOGITS, rows, cols, per_device))
    return problems


def ring_mesh(profile):
    """Whether the profile opens a mesh of more than two devices (the four-card ring)."""
    rows, cols = mesh_of(profile)['shape']
    return rows * cols > 2


def host_sampling_forced(environ=None, exists=os.path.exists):
    environ = os.environ if environ is None else environ
    return environ.get(HOST_SAMPLING_ENV) == '1' or exists(HOST_SAMPLING_FILE)


def without_device_sampling(profile):
    """The profile with sample_on_device_mode removed from its engine's tt config (a copy; the input is kept)."""
    copied = json.loads(json.dumps(profile))
    tt = ((copied.get('engine') or {}).get('additional-config') or {}).get('tt')
    if tt is not None:
        tt.pop('sample_on_device_mode', None)
    return copied


def install_ring_check(environ=None):
    """Under a ring profile, in every process: tp4_mesh's check on the TT plugin's open_mesh_device, when the
    plugin's worker module is imported (the engine process opens the mesh). -> whether it was armed."""
    environ = os.environ if environ is None else environ
    if environ.get(RING_CHECK_ENV) == '0':
        log('mesh: the four-card ring check is off (%s=0)', RING_CHECK_ENV)
        return False
    import tp4_mesh

    sys.meta_path.insert(0, PostImportHook(PLUGIN_WORKER, lambda module: tp4_mesh.install_ring_check(
        module, lambda line: log('%s', line))))
    return True


def apply_environment(profile, environ=None):
    environ = os.environ if environ is None else environ
    for key, value in profile['env'].items():
        environ[key] = str(value)
    environ['TT_MESH_GRAPH_DESC_PATH'] = profile['mesh_graph_descriptor']
    # Item 8: a profile that names its mesh sets MESH_DEVICE over the image's and the platform's P300; one that
    # names none leaves MESH_DEVICE as it found it, exactly as before.
    if profile.get('mesh_device'):
        environ['MESH_DEVICE'] = profile['mesh_device']
    # The fast path was measured without it; the stock decode path (general profile) is what
    # it configures, and the platform bakes it =host, so that profile keeps it.
    if profile.get('drop_batched_decode_mode', True):
        environ.pop('QWEN36_BATCHED_DECODE_MODE', None)
    # Prefix reuse is the profile's to switch on: an inherited value under any other profile would
    # give the model the capability while the profile's argv leaves prefix caching off.
    if PREFIX_SWITCH not in profile['env']:
        environ.pop(PREFIX_SWITCH, None)
    # The same for the fast path's side of it (sticky sessions).
    if STICKY_SWITCH not in profile['env']:
        environ.pop(STICKY_SWITCH, None)
    # Lever N is the profile's to switch on: an inherited flag under any other profile would chunk a prefill the profile's argv keeps whole.
    for name in LEVERN_ENV_FLAGS:
        if name not in profile['env']:
            environ.pop(name, None)
    # The drafter checkpoint likewise: an inherited id under a profile that names none would pin a candidate the profile's argv does not point at.
    if DRAFTER_CHECKPOINT_FLAG not in profile['env']:
        environ.pop(DRAFTER_CHECKPOINT_FLAG, None)
    return environ


def prefix_reuse(profile):
    """Whether the profile turns conversation prefix reuse on (general-prefix and its gate variant)."""
    return str(profile.get('env', {}).get(PREFIX_SWITCH, '')) == '1'


def sticky_sessions(profile):
    """Whether the profile turns the fast path's sticky sessions on (c2-packed-prefix and its gate twin)."""
    return str(profile.get('env', {}).get(STICKY_SWITCH, '')) == '1'


def sticky_problems(profile):
    """Every way the profile's sticky-session switch is set without what it needs, [] when none: it is
    the fast path's side of prefix reuse, so it needs the fast path (qwen_fast_t16) and
    QWEN_PREFIX_REUSE=1 beside it."""
    engine = profile.get('engine', {})
    value = profile.get('env', {}).get(STICKY_SWITCH)
    problems = []
    if value is not None and str(value) not in ('0', '1'):
        problems.append('%s=%r is neither 1 nor 0' % (STICKY_SWITCH, value))
    if not sticky_sessions(profile):
        return problems
    if not prefix_reuse(profile):
        problems.append('%s=1 needs %s=1: the fast path would admit hits no checkpoint makes exact'
                        % (STICKY_SWITCH, PREFIX_SWITCH))
    if not (engine.get('additional-config') or {}).get('qwen_fast_t16'):
        problems.append('%s=1 is the fast path\'s switch, but the profile does not run the fast path (qwen_fast_t16)'
                        % STICKY_SWITCH)
    return problems


def prefix_reuse_problems(profile):
    """Every way the profile's engine flags break prefix reuse's exactness assumptions, [] when none.

    Reuse is exact only under the engine qwen_prefix_scheduler_patch.install_problems accepts (async
    scheduling off, a whole-prompt token budget, 64-token blocks, no speculative lookahead) and
    with vLLM's own prefix cache on; vLLM's align-mode assertion also needs chunked prefill on in
    the argv, which the TT platform turns off again for qwen3_5 (design 2.0.1 item 5; P0a checks
    1-2). Prefix caching without the switch would serve hits no checkpoint makes exact, and the
    fast path admits a hit only under sticky sessions (serving_lifecycle refuses num_computed_tokens
    != 0 without QWEN_FAST_STICKY_SESSIONS=1), which is then also the only way a speculative config
    passes: DFlash with 15 proposals, whose lookahead the scheduler graft accepts under that switch."""
    engine = profile.get('engine', {})
    value = profile.get('env', {}).get(PREFIX_SWITCH)
    caching = engine.get('enable-prefix-caching') is True
    problems = sticky_problems(profile)
    if value is not None and str(value) not in ('0', '1'):
        problems.append('%s=%r is neither 1 nor 0' % (PREFIX_SWITCH, value))
    if caching and engine.get('no-enable-prefix-caching'):
        problems.append('both enable-prefix-caching and no-enable-prefix-caching')
    if not prefix_reuse(profile):
        if caching:
            problems.append('enable-prefix-caching without %s=1: vLLM would serve cached blocks without the '
                            'checkpoint trim that makes a hit exact' % PREFIX_SWITCH)
        return problems
    if not caching or engine.get('no-enable-prefix-caching'):
        problems.append('%s=1 needs enable-prefix-caching' % PREFIX_SWITCH)
    if engine.get('enable-chunked-prefill') is not True or engine.get('no-enable-chunked-prefill'):
        problems.append('enable-chunked-prefill is required: vLLM asserts it for a hybrid model with prefix '
                        'caching, and the TT platform turns chunking off again')
    if engine.get('no-async-scheduling') is not True or engine.get('async-scheduling'):
        problems.append('no-async-scheduling is required: blocks hashed at allocation would be read unwritten')
    if engine.get('block-size') != 64:
        problems.append('block-size must be 64 (the page tables and the 2048-token chunk arithmetic), not %r'
                        % engine.get('block-size'))
    budget, context = engine.get('max-num-batched-tokens'), engine.get('max-model-len')
    if type(budget) is not int or type(context) is not int or budget < context:
        problems.append('max-num-batched-tokens %r must cover max-model-len %r: the vLLM budget must never split a prefill; only the Lever N '
                        'cap may, at 2,048-token boundaries with the drafter window inside the final step' % (budget, context))
    sticky = sticky_sessions(profile)
    if (engine.get('additional-config') or {}).get('qwen_fast_t16') and not sticky:
        problems.append('the fast path (qwen_fast_t16) cannot admit a prefix hit without %s=1 (sticky sessions)'
                        % STICKY_SWITCH)
    speculative = engine.get('speculative-config')
    if speculative:
        if not sticky:
            problems.append('speculative decoding: the scheduler graft refuses lookahead')
        elif not isinstance(speculative, dict) or any(speculative.get(key) != want
                                                      for key, want in sorted(STICKY_SPECULATION.items())):
            problems.append('speculative decoding: sticky sessions serve only %s with %d proposals, not %r'
                            % (STICKY_SPECULATION['method'], STICKY_SPECULATION['num_speculative_tokens'],
                               speculative))
    if engine.get('prefix-caching-hash-algo') != PREFIX_HASH_ALGO:
        problems.append('prefix-caching-hash-algo must be %s (the block-hash chain a hit\'s exactness leans on), '
                        'not %r' % (PREFIX_HASH_ALGO, engine.get('prefix-caching-hash-algo')))
    return problems


def levern_on(profile):
    """Whether the profile turns Lever N (interleaved chunked prefill) on."""
    return str((profile.get('env') or {}).get(LEVERN_SWITCH, '')) == '1'


DRAFTER_CHECKPOINT_FLAG = 'QWEN_FAST_DRAFTER_CHECKPOINT'


def drafter_problems(profile):
    """The profile's drafter selection (drafter_checkpoint): a candidate id must be pinned and the engine's draft paths must be its baked copy's; a profile that
    names no id must point at the default's (today's) paths. A candidate is a gate-only profile: nothing serves traffic on a drafter that has not been judged."""
    import drafter_checkpoint

    problems = list(drafter_checkpoint.profile_problems(profile))
    if DRAFTER_CHECKPOINT_FLAG in (profile.get('env') or {}) and profile.get('gate_only') is not True:
        problems.append('%s needs a gate-only profile (nothing serves traffic on a drafter that has not been judged)' % DRAFTER_CHECKPOINT_FLAG)
    return problems


MULTI_SWITCH = 'QWEN_FAST_TP4_SDPA'
MULTI_AUDIT = 'QWEN_FAST_TP4_SDPA_AUDIT'
F1_SWITCH = 'QWEN_FAST_TP4_CONV_GATES_SPREAD'
F1_AUDIT = 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'
SDPA_OFF_VALUES = ('', '0', 'off')


def multi_problems(profile):
    """The wave-2 levers (docs/tp4-w2.md, docs/tp4-combined-window.md) on a traffic profile: refused. The multi-user SDPA launch (QWEN_FAST_TP4_SDPA
    set to anything but off, one G16 flags 0x21 program) is outside the baked 262k evidence (packed_any_evidence_tp4_262144.json covers the served
    0x23 programs), the F1 conv-gates spread has never compiled on a card, and both audits are gate instruments. A traffic profile that names any of
    the four needs a pinned qualification record for its capacity, which does not exist yet: the refusal names what is missing."""
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    if profile.get('gate_only') is True:
        return []
    problems = []
    if env.get(MULTI_SWITCH, '').strip().lower() not in SDPA_OFF_VALUES:
        problems.append('%s=%s needs a gate-only profile: the multi-user SDPA launch is outside the 262k evidence and has no pinned multi '
                        'qualification record' % (MULTI_SWITCH, env[MULTI_SWITCH]))
    for key in (MULTI_AUDIT, F1_SWITCH, F1_AUDIT):
        if env.get(key, '0') not in ('', '0'):
            problems.append('%s=%s needs a gate-only profile (it has never served traffic and has no pinned qualification record)'
                            % (key, env[key]))
    return problems


def levern_problems(profile):
    """Every way the profile's Lever N flags break what the lever needs, [] when none (docs/lever-n-tp4-design-2026-10-04.md section 4).

    The flags parse (levern_policy: a step that is not a multiple of 2,048 is refused, a sibling without the master switch is a typo). The
    audit switch alone is the non-interleaved control and needs only a gate-only profile. The master switch needs, beside it:
    - a gate-only profile for the gate instruments (QWEN_FAST_LEVERN_AUDIT, QWEN_FAST_LEVERN_FAULT) and for the stage-1 shape (no prefix reuse); the
      master switch on the merged route alone, with neither instrument, is a TRAFFIC profile's (its gates ran in the windows that precede its cutover);
    - QWEN_FAST_ANY_REQUEST=1 (the one-fresh-prefill cap and the lifecycle's continuation routing live under it) and the fast path;
    - enable-chunked-prefill in place of no-enable-chunked-prefill, max-num-batched-tokens equal to max-model-len (nothing splits except
      through the cap), no-async-scheduling (the engine is synchronous: a step's wall time is the interval between two schedule() calls),
      64-token blocks and text only;
    - QWEN_FAST_KV_RESERVATION=1: a partial prefill is never preempted only under the reservation's worst-case admission;
    - no per-admission decode credit (the alternation subsumes it) and no fast lane (its gate does not know prefill chunks);
    - prefix reuse and sticky sessions BOTH on or BOTH off (the merged route, docs/lever-n-prefix-merged-route.md: start_pos > 0 then means one
      thing, and every step names where its state comes from). Both on, the profile must also pass prefix_reuse_problems and the image must carry
      the merged route (levern_route.SOURCES names all four sources: a capability, not a flag, so a stage-1 image cannot boot such a profile). The
      merged flags (TTFT target, short class, host parking, park slots, park age, epoch scope) are refused where they would do nothing: parking and
      the route epoch scope without the merged route, the route epoch scope without the two-block pre-stage and its per-block epochs."""
    import levern_policy

    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    engine = profile.get('engine') or {}
    problems = ['Lever N: ' + problem for problem in levern_policy.config_problems(env)]
    audit = env.get(LEVERN_AUDIT, '0') == '1'
    if not levern_on(profile):
        if audit and profile.get('gate_only') is not True:
            problems.append('Lever N: %s=1 is a gate instrument and needs a gate-only profile' % LEVERN_AUDIT)
        return problems
    if profile.get('gate_only') is not True:
        # The traffic arm (stage 1 of the short-window plan): the master switch alone, with its sibling flags and the merged route, may serve traffic
        # (the kill switch levern.off stops it with no restart); the gate instruments never may (they are the gates' own, and a fault is a negative control).
        for name in (LEVERN_AUDIT, 'QWEN_FAST_LEVERN_FAULT'):
            if str(env.get(name, '0')) not in ('0', ''):
                problems.append('Lever N: %s is a gate instrument and needs a gate-only profile (a traffic profile carries the master switch and the '
                                'policy flags only)' % name)
        if env.get(PREFIX_SWITCH, '0') != '1' or env.get(STICKY_SWITCH, '0') != '1':
            problems.append('Lever N: a traffic profile runs the merged route only (%s=1 and %s=1 beside %s=1): the stage-1 shape without prefix reuse '
                            'is a gate arm and needs a gate-only profile' % (PREFIX_SWITCH, STICKY_SWITCH, LEVERN_SWITCH))
    if env.get('QWEN_FAST_ANY_REQUEST') != '1':
        problems.append('Lever N: %s=1 needs QWEN_FAST_ANY_REQUEST=1' % LEVERN_SWITCH)
    if env.get(FAST_TP_ENV) != '4':
        problems.append('Lever N: %s=1 is the four-card path (%s=4, whose attach warms the eager prefill before the packed traces)'
                        % (LEVERN_SWITCH, FAST_TP_ENV))
    if (engine.get('additional-config') or {}).get('qwen_fast_t16') is not True:
        problems.append('Lever N: %s=1 is the fast path\'s switch, but the profile does not run the fast path (qwen_fast_t16)' % LEVERN_SWITCH)
    if engine.get('enable-chunked-prefill') is not True or engine.get('no-enable-chunked-prefill'):
        problems.append('Lever N: enable-chunked-prefill is required in place of no-enable-chunked-prefill')
    if engine.get('no-async-scheduling') is not True or engine.get('async-scheduling'):
        problems.append('Lever N: no-async-scheduling is required (a step\'s wall time is the interval between two schedule() calls)')
    budget, context = engine.get('max-num-batched-tokens'), engine.get('max-model-len')
    if type(budget) is not int or type(context) is not int or budget != context:
        problems.append('Lever N: max-num-batched-tokens %r must equal max-model-len %r (nothing splits except through the cap)'
                        % (budget, context))
    if engine.get('block-size') != 64:
        problems.append('Lever N: block-size must be 64, not %r' % (engine.get('block-size'),))
    mm = engine.get('limit-mm-per-prompt')
    if not isinstance(mm, dict) or mm.get('image') != 0 or mm.get('video') != 0:
        problems.append('Lever N: text only is required (limit-mm-per-prompt image 0 and video 0), not %r' % (mm,))
    if env.get('QWEN_FAST_KV_RESERVATION') != '1':
        problems.append('Lever N: QWEN_FAST_KV_RESERVATION=1 is required (a partial prefill is never preempted only under the '
                        'reservation\'s worst-case admission)')
    if env.get('QWEN_FAST_DECODE_STEPS_PER_ADMISSION', '0') != '0':
        problems.append('Lever N: QWEN_FAST_DECODE_STEPS_PER_ADMISSION must be unset or 0 (the alternation subsumes the credit)')
    if env.get('QWEN_FAST_LANE', '0') != '0':
        problems.append('Lever N: QWEN_FAST_LANE must be unset or 0 (the lane gate does not know prefill chunks)')
    problems += merged_route_problems(profile, env)
    return problems


def merged_route_problems(profile, env):
    """The merged route's requirements on a profile whose master switch is on (see levern_problems): prefix reuse and sticky sessions both on or both
    off, and with both on everything prefix reuse needs plus an image that carries all four sources. [] when the profile is the stage-1 shape
    (both off) and sets no merged-only flag."""
    problems = []
    prefix, sticky = env.get(PREFIX_SWITCH, '0'), env.get(STICKY_SWITCH, '0')
    for name, value in ((PREFIX_SWITCH, prefix), (STICKY_SWITCH, sticky)):
        if value not in ('0', '1'):
            problems.append('Lever N: %s=%r is neither 1 nor 0' % (name, value))
    merged = prefix == '1' and sticky == '1'
    if (prefix == '1') != (sticky == '1'):
        problems.append('Lever N: %s and %s are both on or both off beside %s=1 (the merged route is the sticky-session route; one of them alone '
                        'is a hit with no fast-path admission, or sticky admission with no checkpoint)' % (PREFIX_SWITCH, STICKY_SWITCH, LEVERN_SWITCH))
    if merged:
        problems += ['Lever N (merged route): ' + problem for problem in prefix_reuse_problems(profile)]
        import levern_route

        have = tuple(getattr(levern_route, 'SOURCES', ()))
        missing = [source for source in MERGED_SOURCES if source not in have]
        if missing:
            problems.append('Lever N (merged route): the route this image carries carries %s, not %s (missing %s): a stage-1 image cannot boot prefix reuse '
                            'beside Lever N' % (list(have), list(MERGED_SOURCES), ', '.join(missing)))
    else:
        for name in ('QWEN_FAST_LEVERN_PARK', 'QWEN_FAST_LEVERN_EPOCH_SCOPE'):
            if env.get(name, '0' if name.endswith('PARK') else 'global') not in ('0', 'global'):
                problems.append('Lever N: %s=%s needs the merged route (%s=1 and %s=1): it would do nothing' % (name, env[name], PREFIX_SWITCH, STICKY_SWITCH))
    if env.get('QWEN_FAST_LEVERN_EPOCH_SCOPE', 'global') == 'route':
        for name in ('QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE', 'QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS'):
            if env.get(name) != '1':
                problems.append('Lever N: QWEN_FAST_LEVERN_EPOCH_SCOPE=route needs %s=1 (the disjoint-writer class engages only with the per-block '
                                'epochs, whose staging destinations it is checked against)' % name)
    return problems


def install_levern_platform(on_import=None, environ=None):
    """Under a Lever N profile, in every process: wrap the TT platform's chunked-prefill policy when the plugin imports it
    (levern_platform.install), so the profile's enable-chunked-prefill survives check_and_update_config. -> whether it was armed."""
    environ = os.environ if environ is None else environ
    import levern_platform

    if on_import is None:
        def on_import(name, callback):
            loaded = sys.modules.get(name)
            if loaded is not None:
                # Already imported (an earlier hook or the launcher pulled the platform in): a post-import hook would never fire.
                callback(loaded)
                return
            sys.meta_path.insert(0, PostImportHook(name, callback))
    on_import(LEVERN_PLATFORM_MODULE, levern_platform.install)
    return True


def gate_problems(profile, environ):
    """A gate-only profile outside a gate."""
    if profile.get('gate_only') is True and environ.get(GATE_SWITCH) != '1':
        return ['profile %s is gate only: it boots only with %s=1, never for traffic' % (profile['name'], GATE_SWITCH)]
    return []


# The 262k evidence waiver (page_width_tp4.WAIVER_ENV): a gate-only profile's own env may set it, a traffic profile's never.
EVIDENCE_WAIVER = 'QWEN_FAST_262K_EVIDENCE_WAIVER'
GATE_PROFILE_MARKER = 'QWEN_C2_GATE_PROFILE'


def waiver_problems(profile, environ):
    """QWEN_FAST_262K_EVIDENCE_WAIVER set (by the profile's env or the process's) outside a gate run of a gate-only profile."""
    env = profile.get('env') or {}
    values = [str(source.get(EVIDENCE_WAIVER)) for source in (env, environ) if source.get(EVIDENCE_WAIVER) not in (None, '', '0')]
    if not values:
        return []
    problems = []
    if profile.get('gate_only') is not True:
        problems.append('profile %s is not gate only: %s must not be set for it (a traffic profile never waives the 262k evidence)'
                        % (profile['name'], EVIDENCE_WAIVER))
    if environ.get(GATE_SWITCH) != '1':
        problems.append('%s=1 needs %s=1 (a gate run)' % (EVIDENCE_WAIVER, GATE_SWITCH))
    if str(env.get(GATE_PROFILE_MARKER, environ.get(GATE_PROFILE_MARKER))) != '1':
        problems.append('%s=1 needs the own marker of the profile, %s=1' % (EVIDENCE_WAIVER, GATE_PROFILE_MARKER))
    if any(value != '1' for value in values):
        problems.append('%s must be 1 or unset, not %s' % (EVIDENCE_WAIVER, '/'.join(values)))
    return problems


def launched_values(argv, flag):
    """Every value argv gives the engine flag `flag` (--flag v or --flag=v; underscores read as dashes)."""
    tokens, values = list(argv), []
    for index, token in enumerate(tokens):
        if not token.startswith('--'):
            continue
        name, separator, value = token[2:].partition('=')
        if name.replace('_', '-') != flag:
            continue
        values.append(value if separator else (tokens[index + 1] if index + 1 < len(tokens) else ''))
    return values


def prefix_launch_problems(argv):
    """(refusals, warnings) of a prefix profile's launched argv, for engine flags the contract does not own."""
    refusals, warnings = [], []
    if launched_values(argv, 'kv-transfer-config'):
        refusals.append('--kv-transfer-config: the scheduler graft refuses a KV connector (external tokens would '
                        'move start_pos past Q)')
    for value in launched_values(argv, 'default-chat-template-kwargs'):
        try:
            kwargs = json.loads(value)
        except ValueError:
            warnings.append('--default-chat-template-kwargs %r is not JSON' % value)
            continue
        if isinstance(kwargs, dict) and 'preserve_thinking' in kwargs and kwargs['preserve_thinking'] is not True:
            warnings.append('--default-chat-template-kwargs sets preserve_thinking=%s: the Qwen3.8 template then '
                            'drops past reasoning at every new user message, so a turn stops extending the last '
                            'one and reuse collapses to older checkpoints (P0b 8.3)'
                            % json.dumps(kwargs['preserve_thinking']))
    return refusals, warnings


def read_salt_key(environ=None):
    """(key bytes or None, the file read or why there is no key)."""
    environ = os.environ if environ is None else environ
    path = environ.get(SALT_KEY_ENV) or SALT_KEY_FILE
    try:
        with open(path, 'rb') as handle:
            key = handle.read().strip()
    except (OSError, IOError) as error:
        return None, '%s: %s' % (path, getattr(error, 'strerror', None) or type(error).__name__)
    if len(key) < SALT_KEY_MIN_BYTES:
        return None, '%s holds %d bytes, fewer than %d' % (path, len(key), SALT_KEY_MIN_BYTES)
    return key, path


def salt_mac(key, tag):
    return hmac.new(key, ('%s.%s' % (SALT_VERSION, tag)).encode('ascii'), hashlib.sha256).hexdigest()


def mint_salt(key, tag):
    """The cache_salt the platform sends for the tenant whose opaque tag this is."""
    if not isinstance(tag, str) or not SALT_TAG.match(tag):
        raise ValueError('a salt tag is 8-128 of [A-Za-z0-9_-], got %r' % (tag,))
    return '%s.%s.%s' % (SALT_VERSION, tag, salt_mac(key, tag))


def salt_verdict(salt, key):
    """'unset', 'verified', 'dropped-no-key' or 'dropped-unverified' for one request's cache_salt."""
    if salt is None or salt == '':
        return 'unset'
    if key is None:
        return 'dropped-no-key'
    parts = salt.split('.') if isinstance(salt, str) else ()
    if (len(parts) != 3 or parts[0] != SALT_VERSION or not SALT_TAG.match(parts[1])
            or not SALT_MAC.match(parts[2])):
        return 'dropped-unverified'
    return 'verified' if hmac.compare_digest(salt_mac(key, parts[1]), parts[2]) else 'dropped-unverified'


def install_salt_policy(module, key, counter=None):
    """Wrap InputProcessor.process_inputs - every request of every OpenAI route passes it on the way to the
    engine (v1/engine/async_llm.py:349) - so the EngineCoreRequest it builds keeps cache_salt only when
    salt_verdict says 'verified'. Not guarded: if this cannot install, the API server must not serve
    (fail closed); counter(verdict) is observability and never raises into a request."""
    processor = module.InputProcessor
    if getattr(processor, '_qwen_prefix_salt', False):
        return False
    original = processor.process_inputs

    def process_inputs(self, *args, **kwargs):
        request = original(self, *args, **kwargs)
        verdict = salt_verdict(getattr(request, 'cache_salt', None), key)
        if verdict.startswith('dropped'):
            request.cache_salt = None
        if counter is not None:
            try:
                counter(verdict)
            except Exception:
                pass
        return request

    processor.process_inputs = process_inputs
    processor._qwen_prefix_salt = True
    log('prefix: cache_salt kept only when it verifies against the salt key (%s)',
        'present' if key is not None else 'ABSENT: every salt is dropped, no request can hit')
    return True


def salt_counter():
    """qwen_prefix_metrics.count_salt, or None when the metrics module is not importable."""
    try:
        import qwen_prefix_metrics

        return qwen_prefix_metrics.count_salt
    except Exception:
        return None


def log_model_tree(module):
    """The bring-up check's evidence: which file this process imported the model entry from."""
    try:
        log('prefix: model tree %s', os.path.realpath(getattr(module, '__file__', None) or '?'))
    except Exception:
        pass


def install_prefix_metrics(api_server, on_import=None):
    """The registry's metrics under a prefix profile: the API server collects what the engine process
    exports (qwen_prefix_metrics). Every process also starts an exporter in the children it forks
    (vLLM forks the EngineCore by default). Observability only: a failure is logged, never raised."""
    try:
        import qwen_prefix_metrics

        if api_server:
            if on_import is None:
                def on_import(name, callback):
                    sys.meta_path.insert(0, PostImportHook(name, callback))
            qwen_prefix_metrics.install_collector(on_import)
        else:
            qwen_prefix_metrics.start_exporter()
        qwen_prefix_metrics.export_in_forked_children()
        return True
    except Exception as error:
        log('prefix metrics not installed: %s: %s', type(error).__name__, error)
        return False


def engine_arguments(profile, snapshot):
    """The profile's engine argv, with the snapshot filled in where the profile names it."""
    arguments = []
    for key, value in profile['engine'].items():
        value = json.loads(json.dumps(value).replace('@SNAPSHOT@', snapshot))
        if value is True:
            arguments.append('--' + key)
        elif value is False or value is None:
            continue
        elif isinstance(value, (dict, list)):
            arguments.extend(('--' + key, json.dumps(value)))
        else:
            arguments.extend(('--' + key, str(value)))
    return arguments


def rewrite_argv(argv, profile, snapshot):
    """argv[0] kept; every owned flag (and its value) dropped; the profile's appended."""
    kept, skip = [argv[0]], False
    for token in argv[1:]:
        if skip:
            skip = False
            continue
        if token.startswith('--'):
            name, separator, _ = token[2:].partition('=')
            name = name.replace('_', '-')
            if name in OWNED_FLAGS:
                skip = OWNED_FLAGS[name] and not separator
                continue
        kept.append(token)
    return kept + engine_arguments(profile, snapshot)


def is_api_server(orig_argv):
    argv = list(orig_argv or ())
    return '-m' in argv and argv.index('-m') + 1 < len(argv) and argv[argv.index('-m') + 1] == API_SERVER


def prompt_length(prompt):
    if isinstance(prompt, dict):
        tokens = prompt.get('prompt_token_ids')
        if tokens is None and isinstance(prompt.get('decoder'), dict):
            tokens = prompt['decoder'].get('prompt_token_ids')
        if tokens is not None:
            return len(tokens)
    return None


# The drafter's last position: dflash_proposal_inputs.proposal_contexts takes a prefill position of at most 262,111 and a request
# budget of at most 262,144 - position - 32, so a request's prompt plus its answer may reach 262,112 and no further
# (test_serving_c2_contract holds this constant to those two literals). A profile whose max-model-len is above it names
# drafter_headroom_tokens: the window the clamp leaves unused at the end (32 at max-model-len 262,144), so the 32 positions the
# drafter's last proposal reaches past a request's final token exist.
DRAFTER_END_LIMIT = 262112


def prompt_room(max_model_len, budget, max_prompt_tokens=None, min_answer_tokens=None, headroom=0):
    """The longest prompt the edge admits.

    By default a prompt must leave room for the whole output budget, max_model_len - budget,
    and a profile's max_prompt_tokens can only lower that: the KV cache is sized for
    max-num-seqs x (max_prompt_tokens + budget), and the fast path has no preemption to fall
    back on when a request outgrows it. That blocks a 123,136-token cap under a 16,384 ceiling
    (the largest prompt would be 114,944), so a profile may name min_answer_tokens instead: every
    admitted prompt then keeps at least that much answer room, the cap is max_model_len -
    min_answer_tokens (lowered, never raised, by max_prompt_tokens), and max_tokens is clamped
    to what is left (enforce_request). A prompt plus its clamped answer never exceeds
    max_model_len either way, so a cache of max-num-seqs x max_model_len still holds every
    admitted request at once.

    `headroom` (a profile's drafter_headroom_tokens, 0 for every profile that does not name one) is the tail of the window the
    clamp never hands out: the room is max_model_len - headroom less the answer room, so prompt + answer <= max_model_len -
    headroom (262,112 at max-model-len 262,144 and 32)."""
    if min_answer_tokens is not None and (type(min_answer_tokens) is not int
                                          or not 1 <= min_answer_tokens <= budget):
        raise ValueError('min_answer_tokens must be an integer from 1 to the output budget %d, got %r'
                         % (budget, min_answer_tokens))
    room = max_model_len - headroom - (budget if min_answer_tokens is None else min_answer_tokens)
    return room if max_prompt_tokens is None else min(room, max_prompt_tokens)


def omitted_max_tokens(max_tokens, *, prompt_tokens, max_model_len):
    """Whether the client left max_tokens unset. The engine API passes None; vLLM's OpenAI
    server fills an omitted max_tokens with the whole remaining context, max_model_len - prompt
    (its get_max_tokens), before this contract sees the request - UNVERIFIED for the pinned
    vLLM 0.25.1 and for any platform get_max_output_tokens override, in which case an omitted
    value simply reads as explicit and is clamped as before. A client that explicitly asks for
    exactly the remaining context is read as omitting it."""
    return max_tokens is None or (prompt_tokens is not None and max_tokens == max_model_len - prompt_tokens)


def enforce_request(params, *, prompt_tokens, max_model_len, budget, eos_ids, max_prompt_tokens=None,
                    min_answer_tokens=None, default_max_tokens=None, drafter_headroom_tokens=0, kv_pool_blocks=None):
    """Refuse what the fast path cannot serve; coerce the rest to its greedy contract.

    min_answer_tokens and default_max_tokens are the c2 profile's (both None elsewhere, and
    then this is exactly the contract every earlier profile ran): the answer room every
    admitted prompt keeps (prompt_room), and the max_tokens a request gets when the client
    omits it (omitted_max_tokens) - within the same clamp. drafter_headroom_tokens (0 elsewhere: nothing changes) lowers the
    clamp's window to max_model_len - headroom; omitted_max_tokens still reads max_model_len (what vLLM fills in).
    kv_pool_blocks (None elsewhere: nothing changes) is the KV pool a profile with the reservation admission serves from
    (QWEN_FAST_KV_RESERVATION=1, serving_kv_reservation): a request whose worst-case blocks, after the clamp, exceed the WHOLE pool
    could never be admitted and is refused here (a 400) instead of waiting forever behind the scheduler's hold."""
    if getattr(params, 'n', 1) != 1:
        raise ContractError('n must be 1 on this model')
    if getattr(params, 'logprobs', None) is not None or getattr(params, 'prompt_logprobs', None) is not None:
        raise ContractError('logprobs are not supported on this model')
    if getattr(params, 'structured_outputs', None) is not None:
        raise ContractError('structured output (response_format, or tool_choice "required" or a named tool) '
                            'is not supported on this model; use tool_choice "auto"')
    if (getattr(params, 'logit_bias', None) or getattr(params, 'allowed_token_ids', None)
            or getattr(params, 'bad_words', None)):
        raise ContractError('logit_bias, allowed_token_ids and bad_words are not supported on this model')
    if getattr(params, 'stop', None):
        raise ContractError('stop strings are not supported on this model')
    if getattr(params, 'min_tokens', 0):
        raise ContractError('min_tokens is not supported on this model')
    if any(token not in eos_ids for token in (getattr(params, 'stop_token_ids', None) or ())):
        raise ContractError('stop_token_ids other than the model end-of-sequence tokens are not supported')
    # The profile may cap prompts below context less budget: the KV cache is sized for
    # max-num-seqs x (max_prompt_tokens + budget), not x max_model_len, and the fast path
    # has no preemption to fall back on when a request outgrows it (prompt_room).
    room = prompt_room(max_model_len, budget, max_prompt_tokens, min_answer_tokens, drafter_headroom_tokens)
    if prompt_tokens is not None and prompt_tokens > room:
        if min_answer_tokens is None:
            raise ContractError('prompt of %d tokens exceeds the %d-token prompt limit of this model (%d-token '
                                'output budget)' % (prompt_tokens, room, budget))
        raise ContractError('prompt of %d tokens exceeds the %d-token prompt limit of this model (at least %d '
                            'tokens of answer room)' % (prompt_tokens, room, min_answer_tokens))
    # Greedy: the fast path's verifier commits argmax tokens whatever these say, and the
    # first token is sampled by vLLM from them, so they must agree with it.
    params.temperature = 0.0
    params.top_p = 1.0
    params.top_k = 0
    params.min_p = 0.0
    params.presence_penalty = 0.0
    params.frequency_penalty = 0.0
    params.repetition_penalty = 1.0
    params.seed = None
    limit = budget if prompt_tokens is None else min(budget, max_model_len - drafter_headroom_tokens - prompt_tokens)
    if default_max_tokens is not None and omitted_max_tokens(params.max_tokens, prompt_tokens=prompt_tokens,
                                                             max_model_len=max_model_len):
        params.max_tokens = min(default_max_tokens, limit)
    elif params.max_tokens is None or params.max_tokens > limit:
        params.max_tokens = limit
    if kv_pool_blocks is not None and prompt_tokens is not None:
        from serving_kv_reservation import request_blocks

        needed = request_blocks(prompt_tokens, params.max_tokens)
        if needed > kv_pool_blocks:
            raise ContractError('a request of %d prompt tokens and up to %d answer tokens needs %d KV blocks of 64 tokens, '
                                'and the server holds %d in all' % (prompt_tokens, params.max_tokens, needed, kv_pool_blocks))
    return params


KV_RESERVATION_FLAG = 'QWEN_FAST_KV_RESERVATION'
KV_NULL_BLOCKS = 1                 # serving_kv_reservation.NULL_BLOCKS: vLLM's block 0 is never handed out


POOL_TOKENS_ENV = 'QWEN36_MAX_TOKENS_ALL_USERS'
LEGACY_WINDOW = 131328             # the windows at or under this keep the override as their pool (the 131k profiles, unchanged)


def real_pool_blocks(profile):
    """The block count vLLM really builds. The TT worker overwrites num-gpu-blocks-override with ceil(max_tokens_all_users / 64) +
    max_num_seqs (plugin worker.py:388-390), so a profile that names QWEN36_MAX_TOKENS_ALL_USERS owns its pool through that; one
    that does not has the override's value, as every 131k profile has always been read. None when neither is an integer."""
    engine, env = profile.get('engine', {}), profile.get('env', {})
    override, seats = engine.get('num-gpu-blocks-override'), engine.get('max-num-seqs')
    tokens = env.get(POOL_TOKENS_ENV)
    if tokens is None:
        return override if type(override) is int else None
    if type(seats) is not int or not str(tokens).isdigit() or int(tokens) < 1:
        return None
    return -(-int(tokens) // 64) + seats


def kv_pool_problem(profile):
    """What is wrong with how a profile sizes its pool; None when nothing. With QWEN36_MAX_TOKENS_ALL_USERS the worker's pool must
    equal num-gpu-blocks-override (the number the reservation, the boot rule and the DRAM arithmetic are all written against); a
    pooled profile past the legacy window must name it, since without it the worker builds seats x window blocks and the override
    is ignored."""
    engine, env = profile.get('engine', {}), profile.get('env', {})
    override, window = engine.get('num-gpu-blocks-override'), engine.get('max-model-len')
    if POOL_TOKENS_ENV in env:
        real = real_pool_blocks(profile)
        if real is None:
            return '%s must be a positive integer and the profile must name max-num-seqs, got %r' % (POOL_TOKENS_ENV, env[POOL_TOKENS_ENV])
        if type(override) is int and real != override:
            return ('%s=%s builds %d blocks (ceil(tokens / 64) + max-num-seqs) but num-gpu-blocks-override is %r: the worker '
                    'overwrites the override, so the two must agree' % (POOL_TOKENS_ENV, env[POOL_TOKENS_ENV], real, override))
    elif kv_pooled(profile) and type(window) is int and window > LEGACY_WINDOW:
        return ('profile %s pools its KV cache past the %d-token window and must name %s: the TT worker overwrites '
                'num-gpu-blocks-override' % (profile.get('name'), LEGACY_WINDOW, POOL_TOKENS_ENV))
    return None


def kv_pooled(profile):
    """Whether a fast-path (S2 extent) profile's KV cache is smaller than seats x its window: num-gpu-blocks-override below
    max-num-seqs x ceil(max-model-len / 64), so its seats cannot all be resident at full length. False for every other profile."""
    engine, env = profile.get('engine', {}), profile.get('env', {})
    override, seats, window = (engine.get(key) for key in ('num-gpu-blocks-override', 'max-num-seqs', 'max-model-len'))
    if env.get('QWEN_FAST_EXTENT_REPLAY') != '1' or any(type(value) is not int for value in (override, seats, window)):
        return False
    return override < seats * -(-window // 64)


def kv_reservation_problem(profile):
    """What is wrong with a profile's KV reservation setting; None when nothing. A pooled fast-path profile must turn the
    reservation admission on (the fast path cannot preempt, and a pool smaller than its seats' windows preempts without it); the
    flag is '0' or '1' only, and '1' needs a pool to read (num-gpu-blocks-override)."""
    value = profile.get('env', {}).get(KV_RESERVATION_FLAG)
    if value not in (None, '0', '1'):
        return '%s must be 0 or 1, got %r' % (KV_RESERVATION_FLAG, value)
    if kv_pooled(profile) and value != '1':
        return ('profile %s pools its KV cache (%s blocks for %s seats of %s tokens) and so needs %s=1: the fast path cannot '
                'preempt' % (profile.get('name'), profile['engine']['num-gpu-blocks-override'],
                             profile['engine']['max-num-seqs'], profile['engine']['max-model-len'], KV_RESERVATION_FLAG))
    if value == '1' and type(profile.get('engine', {}).get('num-gpu-blocks-override')) is not int:
        return '%s=1 needs num-gpu-blocks-override: the reservation is made against that pool' % KV_RESERVATION_FLAG
    return None


def kv_pool_blocks(profile):
    """The blocks the reservation admission hands out: num-gpu-blocks-override less vLLM's null block, for a profile that turns
    the reservation on; None for every other."""
    if profile.get('env', {}).get(KV_RESERVATION_FLAG) != '1':
        return None
    return real_pool_blocks(profile) - KV_NULL_BLOCKS


def request_limits(profile):
    """The request contract's numbers from a profile, checked once at boot: the output budget,
    and the optional max_prompt_tokens, min_answer_tokens and default_max_tokens."""
    budget = int(profile['env']['QWEN_FAST_OUTPUT_BUDGET'])
    limits = dict(budget=budget, max_prompt_tokens=profile.get('max_prompt_tokens'),
                  min_answer_tokens=profile.get('min_answer_tokens'),
                  default_max_tokens=profile.get('default_max_tokens'))
    default = limits['default_max_tokens']
    if default is not None and (type(default) is not int or not 1 <= default <= budget):
        raise ValueError('default_max_tokens must be an integer from 1 to the output budget %d, got %r'
                         % (budget, default))
    headroom = profile.get('drafter_headroom_tokens')
    if headroom is not None:
        # Named only by the profiles whose window is past the drafter's last position; absent from the key set otherwise, so
        # every earlier profile's limits are exactly what they were.
        if type(headroom) is not int or not 0 <= headroom <= 4096:
            raise ValueError('drafter_headroom_tokens must be an integer from 0 to 4096, got %r' % (headroom,))
        limits['drafter_headroom_tokens'] = headroom
    headroom = headroom or 0
    problem = kv_reservation_problem(profile) or kv_pool_problem(profile)
    if problem:
        raise ValueError(problem)
    pool = kv_pool_blocks(profile)
    if pool is not None:
        limits['kv_pool_blocks'] = pool
    max_model_len = profile.get('engine', {}).get('max-model-len')
    if type(max_model_len) is int:
        if max_model_len - headroom > DRAFTER_END_LIMIT:
            raise ValueError('max-model-len %d less drafter_headroom_tokens %d is past the drafter\'s last position %d '
                             '(dflash_proposal_inputs): name a drafter_headroom_tokens of at least %d'
                             % (max_model_len, headroom, DRAFTER_END_LIMIT, max_model_len - DRAFTER_END_LIMIT))
        # Validates min_answer_tokens; the cap must leave a prompt of at least one token.
        if prompt_room(max_model_len, budget, limits['max_prompt_tokens'], limits['min_answer_tokens'], headroom) < 1:
            raise ValueError('The profile admits no prompt at all')
    return limits


def install_request_contract(module, *, budget, eos_ids, max_prompt_tokens=None, min_answer_tokens=None,
                             default_max_tokens=None, drafter_headroom_tokens=0, kv_pool_blocks=None):
    processor = module.InputProcessor
    if getattr(processor, '_qwen_c2_contract', False):
        return
    original = processor.process_inputs

    def process_inputs(self, request_id, prompt, params, *args, **kwargs):
        if hasattr(params, 'temperature'):
            enforce_request(params, prompt_tokens=prompt_length(prompt),
                            max_model_len=self.model_config.max_model_len, budget=budget, eos_ids=eos_ids,
                            max_prompt_tokens=max_prompt_tokens, min_answer_tokens=min_answer_tokens,
                            default_max_tokens=default_max_tokens, drafter_headroom_tokens=drafter_headroom_tokens,
                            kv_pool_blocks=kv_pool_blocks)
        return original(self, request_id, prompt, params, *args, **kwargs)

    processor.process_inputs = process_inputs
    processor._qwen_c2_contract = True
    log('request contract installed: greedy, max_tokens <= %d, prompt <= %s, eos %s', budget,
        max_prompt_tokens or 'context - budget', sorted(eos_ids))
    if min_answer_tokens is not None or default_max_tokens is not None:
        log('request contract: every prompt keeps >= %s answer tokens, max_tokens defaults to %s when omitted',
            min_answer_tokens if min_answer_tokens is not None else budget,
            default_max_tokens if default_max_tokens is not None else 'the clamp')
    if drafter_headroom_tokens:
        log('request contract: prompt + answer <= context - %d (the drafter\'s last position)', drafter_headroom_tokens)
    if kv_pool_blocks is not None:
        log('request contract: a request needing more than the %d KV blocks the reservation admission hands out is refused',
            kv_pool_blocks)


def exit_without_device_teardown(modules=None, parent=None, exit=None, streams=None):
    """At interpreter exit, end an engine process before tt-metal's C++ teardown runs.

    tt-metal registers MetalContext::destroy_all_instances with on_exit when a process opens the
    mesh. On this rig that teardown fails to bring device 0's active ethernet core back
    (llrt.cpp:594, "Timed out while waiting for active ethernet core 31-25 to become active
    again"), and every later open of the card then fails the same way until the pair is reset -
    the m3native gate resets M+A before each run for exactly this reason. The Thatch runtime
    restarts the engine inside one container (a release-first load, a health-monitor recovery,
    a docker stop), so on 2026-09-25 the first graceful exit wedged card A and all five
    restarts after it died (job 01M3BK9NQ1WQM3JTJSX2V03D1M). A process that is killed never runs
    the teardown, and the next open succeeds (smokes v2-v5, each container removed with
    docker rm -f). Python atexit handlers run before C on_exit handlers, so os._exit here
    gives every exit of an engine process that end state.

    Only a multiprocessing child that has imported ttnn - the vLLM EngineCore, the one process
    that opens the mesh - is ended this way; the exit status is 0 (the parent watches the
    child's sentinel, not its status)."""
    modules = sys.modules if modules is None else modules
    if 'ttnn' not in modules:
        return False
    if parent is None:
        import multiprocessing

        parent = getattr(multiprocessing, 'parent_process', lambda: None)()
    if parent is None:
        return False
    log('engine process exiting without tt-metal device teardown')
    for stream in (streams if streams is not None else (sys.stdout, sys.stderr)):
        try:
            stream.flush()
        except Exception:
            pass
    (os._exit if exit is None else exit)(0)
    return True


def install_teardown_skip():
    import atexit

    atexit.register(exit_without_device_teardown)


class PostImportHook(object):
    """Run a callback on a module right after it executes, without importing it early."""

    def __init__(self, name, callback):
        self.name, self.callback = name, callback

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        import importlib.util

        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        loader, callback = spec.loader, self.callback
        execute = loader.exec_module

        def exec_module(module):
            execute(module)
            callback(module)

        loader.exec_module = exec_module
        return spec


def parser_rechunk(profile):
    """Whether the profile arms fix M (c2_parser_rechunk); off unless it says true."""
    value = profile.get('parser_rechunk', False)
    if value not in (True, False):
        raise ValueError('parser_rechunk must be true or false, got %r' % (value,))
    return value


def arm_parser_rechunk(profile_name):
    """Wrap DelegatingParser.parse_delta with fix M when vLLM imports its module. The module is
    imported here, at boot, so an image without it fails the API server's boot (status 78)
    instead of serving the c2 profile without M; install refuses a parser that is not vLLM
    0.25.1's, which fails the server's own import of the parsers."""
    import c2_parser_rechunk

    sys.meta_path.insert(0, PostImportHook(c2_parser_rechunk.MODULE, c2_parser_rechunk.install))
    log('profile %s: parser M armed: %s.%s.parse_delta re-chunks multi-token deltas at marker tokens (R21)',
        profile_name, c2_parser_rechunk.MODULE, c2_parser_rechunk.CLASS)


def boot(environ=None, orig_argv=None):
    environ = os.environ if environ is None else environ
    if environ.get('QWEN_C2_SERVING') != '1':
        return None
    fix_sys_path()
    profile = load_profile(environ.get('QWEN_C2_PROFILES', PROFILES))
    problems = gate_problems(profile, environ) + waiver_problems(profile, environ)
    if problems:
        raise ValueError('; '.join(problems))
    problems = prefix_reuse_problems(profile)
    if problems:
        raise ValueError('profile %s cannot serve prefix reuse exactly: %s' % (profile['name'], '; '.join(problems)))
    problems = levern_problems(profile)
    if problems:
        raise ValueError('profile %s cannot serve Lever N exactly: %s' % (profile['name'], '; '.join(problems)))
    problems = multi_problems(profile)
    if problems:
        raise ValueError('profile %s cannot serve the wave-2 levers: %s' % (profile['name'], '; '.join(problems)))
    problems = drafter_problems(profile)
    if problems:
        raise ValueError('profile %s cannot serve its drafter checkpoint: %s' % (profile['name'], '; '.join(problems)))
    problems = mesh_problems(profile)
    if problems:
        raise ValueError('profile %s cannot open its mesh: %s' % (profile['name'], '; '.join(problems)))
    apply_environment(profile, environ)
    if DRAFTER_CHECKPOINT_FLAG in profile['env']:
        # A candidate drafter: its baked bytes must be the pinned ones (docs/tp4-combined-window.md), or the attach is refused before an engine starts.
        import drafter_checkpoint

        drafter_checkpoint.attach_check(environ, log=lambda line: log('%s', line))
    if levern_on(profile):
        install_levern_platform()
        log('profile %s: Lever N armed (%s=1): the platform keeps chunked prefill, the scheduler caps and alternates', profile['name'],
            LEVERN_SWITCH)
    if ring_mesh(profile):
        install_ring_check(environ)
    if profile.get('skip_device_teardown', True):
        # Registered at interpreter start, so it runs after every other atexit handler.
        install_teardown_skip()
    limits = request_limits(profile)
    budget = limits['budget']
    eos_ids = frozenset(int(token) for token in profile['eos_ids'])
    orig_argv = getattr(sys, 'orig_argv', None) if orig_argv is None else orig_argv
    api_server = is_api_server(orig_argv)
    if prefix_reuse(profile):
        install_prefix_metrics(api_server)
        sys.meta_path.insert(0, PostImportHook(MODEL_ENTRY, log_model_tree))
    if api_server:
        snapshot = resolve_snapshot(profile)
        if tt_config(profile).get('sample_on_device_mode') is not None and host_sampling_forced(environ):
            profile = dict(without_device_sampling(profile), name=profile['name'])
            log('profile %s: host sampling for every batch (%s=1 or %s): sample_on_device_mode dropped',
                profile['name'], HOST_SAMPLING_ENV, HOST_SAMPLING_FILE)
        sys.argv[:] = rewrite_argv(sys.argv, profile, snapshot)
        log('profile %s: vLLM argv %s', profile['name'], json.dumps(sys.argv[1:]))
        log('mesh %s, output budget %d, context %s', environ['TT_MESH_GRAPH_DESC_PATH'], budget,
            profile['engine'].get('max-model-len'))
        if profile.get('mesh_device'):
            log('mesh device %s %s, fabric %s, sampling %s', profile['mesh_device'], list(mesh_of(profile)['shape']),
                tt_config(profile).get('fabric_config', 'the plugin default'),
                tt_config(profile).get('sample_on_device_mode') or 'host')
        if prefix_reuse(profile):
            refusals, warnings = prefix_launch_problems(sys.argv[1:])
            if refusals:
                raise ValueError('profile %s cannot serve prefix reuse with this launch: %s' % (
                    profile['name'], '; '.join(refusals)))
            for warning in warnings:
                log('prefix: WARNING %s', warning)
            key, where = read_salt_key(environ)
            log('profile %s: prefix reuse on (%s=1, vLLM prefix caching; the platform turns chunking off again); '
                'salt key %s', profile['name'], PREFIX_SWITCH, where)
            counter = salt_counter()
            sys.meta_path.insert(0, PostImportHook(
                INPUT_PROCESSOR, lambda module: install_salt_policy(module, key, counter)))
        if parser_rechunk(profile):
            arm_parser_rechunk(profile['name'])
        if profile.get('request_contract', True) is False:
            log('profile %s: no request contract (the fast path is off)', profile['name'])
            return profile
        sys.meta_path.insert(0, PostImportHook(
            INPUT_PROCESSOR, lambda module: install_request_contract(
                module, budget=budget, eos_ids=eos_ids, max_prompt_tokens=limits['max_prompt_tokens'],
                min_answer_tokens=limits['min_answer_tokens'], default_max_tokens=limits['default_max_tokens'],
                drafter_headroom_tokens=limits.get('drafter_headroom_tokens', 0),
                kv_pool_blocks=limits.get('kv_pool_blocks'))))
    return profile
