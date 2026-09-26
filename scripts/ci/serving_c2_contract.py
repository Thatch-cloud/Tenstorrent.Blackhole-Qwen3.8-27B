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
   from (the bring-up check, qwen_prefix_stage bringup).
6. gate-only profiles (gate_only: true) boot only with QWEN_C2_GATE=1: they exist for a gate and
   must never take traffic.

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
# vLLM's default (config/cache.py:95), pinned in the prefix profiles: the block-hash chain is what makes
# a cached block's content its prompt's, and the exactness argument leans on it (design L2).
PREFIX_HASH_ALGO = 'sha256'
# The model entry every process logs the source of under a prefix profile (the bring-up check).
MODEL_ENTRY = 'models.demos.blackhole.qwen36.tt.qwen36_vllm'
# A profile with gate_only: true boots only with this set to 1.
GATE_SWITCH = 'QWEN_C2_GATE'

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


def apply_environment(profile, environ=None):
    environ = os.environ if environ is None else environ
    for key, value in profile['env'].items():
        environ[key] = str(value)
    environ['TT_MESH_GRAPH_DESC_PATH'] = profile['mesh_graph_descriptor']
    # The fast path was measured without it; the stock decode path (general profile) is what
    # it configures, and the platform bakes it =host, so that profile keeps it.
    if profile.get('drop_batched_decode_mode', True):
        environ.pop('QWEN36_BATCHED_DECODE_MODE', None)
    # Prefix reuse is the profile's to switch on: an inherited value under any other profile would
    # give the model the capability while the profile's argv leaves prefix caching off.
    if PREFIX_SWITCH not in profile['env']:
        environ.pop(PREFIX_SWITCH, None)
    return environ


def prefix_reuse(profile):
    """Whether the profile turns conversation prefix reuse on (general-prefix and its gate variant)."""
    return str(profile.get('env', {}).get(PREFIX_SWITCH, '')) == '1'


def prefix_reuse_problems(profile):
    """Every way the profile's engine flags break prefix reuse's exactness assumptions, [] when none.

    Reuse is exact only under the engine prefix_scheduler_graft.install_problems accepts (async
    scheduling off, a whole-prompt token budget, 64-token blocks, no speculative lookahead) and
    with vLLM's own prefix cache on; vLLM's align-mode assertion also needs chunked prefill on in
    the argv, which the TT platform turns off again for qwen3_5 (design 2.0.1 item 5; P0a checks
    1-2). Prefix caching without the switch would serve hits no checkpoint makes exact, and the
    fast path cannot admit a hit yet (C1: serving_lifecycle refuses num_computed_tokens != 0)."""
    engine = profile.get('engine', {})
    value = profile.get('env', {}).get(PREFIX_SWITCH)
    caching = engine.get('enable-prefix-caching') is True
    problems = []
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
        problems.append('max-num-batched-tokens %r must cover max-model-len %r: a prefill must never split'
                        % (budget, context))
    if (engine.get('additional-config') or {}).get('qwen_fast_t16'):
        problems.append('the fast path (qwen_fast_t16) cannot admit a prefix hit yet (C1)')
    if engine.get('speculative-config'):
        problems.append('speculative decoding: the scheduler graft refuses lookahead')
    if engine.get('prefix-caching-hash-algo') != PREFIX_HASH_ALGO:
        problems.append('prefix-caching-hash-algo must be %s (the block-hash chain a hit\'s exactness leans on), '
                        'not %r' % (PREFIX_HASH_ALGO, engine.get('prefix-caching-hash-algo')))
    return problems


def gate_problems(profile, environ):
    """A gate-only profile outside a gate."""
    if profile.get('gate_only') is True and environ.get(GATE_SWITCH) != '1':
        return ['profile %s is gate only: it boots only with %s=1, never for traffic' % (profile['name'], GATE_SWITCH)]
    return []


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


def prompt_room(max_model_len, budget, max_prompt_tokens=None, min_answer_tokens=None):
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
    admitted request at once."""
    if min_answer_tokens is not None and (type(min_answer_tokens) is not int
                                          or not 1 <= min_answer_tokens <= budget):
        raise ValueError('min_answer_tokens must be an integer from 1 to the output budget %d, got %r'
                         % (budget, min_answer_tokens))
    room = max_model_len - (budget if min_answer_tokens is None else min_answer_tokens)
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
                    min_answer_tokens=None, default_max_tokens=None):
    """Refuse what the fast path cannot serve; coerce the rest to its greedy contract.

    min_answer_tokens and default_max_tokens are the c2 profile's (both None elsewhere, and
    then this is exactly the contract every earlier profile ran): the answer room every
    admitted prompt keeps (prompt_room), and the max_tokens a request gets when the client
    omits it (omitted_max_tokens) - within the same clamp."""
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
    room = prompt_room(max_model_len, budget, max_prompt_tokens, min_answer_tokens)
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
    limit = budget if prompt_tokens is None else min(budget, max_model_len - prompt_tokens)
    if default_max_tokens is not None and omitted_max_tokens(params.max_tokens, prompt_tokens=prompt_tokens,
                                                             max_model_len=max_model_len):
        params.max_tokens = min(default_max_tokens, limit)
    elif params.max_tokens is None or params.max_tokens > limit:
        params.max_tokens = limit
    return params


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
    max_model_len = profile.get('engine', {}).get('max-model-len')
    if type(max_model_len) is int:
        # Validates min_answer_tokens; the cap must leave a prompt of at least one token.
        if prompt_room(max_model_len, budget, limits['max_prompt_tokens'], limits['min_answer_tokens']) < 1:
            raise ValueError('The profile admits no prompt at all')
    return limits


def install_request_contract(module, *, budget, eos_ids, max_prompt_tokens=None, min_answer_tokens=None,
                             default_max_tokens=None):
    processor = module.InputProcessor
    if getattr(processor, '_qwen_c2_contract', False):
        return
    original = processor.process_inputs

    def process_inputs(self, request_id, prompt, params, *args, **kwargs):
        if hasattr(params, 'temperature'):
            enforce_request(params, prompt_tokens=prompt_length(prompt),
                            max_model_len=self.model_config.max_model_len, budget=budget, eos_ids=eos_ids,
                            max_prompt_tokens=max_prompt_tokens, min_answer_tokens=min_answer_tokens,
                            default_max_tokens=default_max_tokens)
        return original(self, request_id, prompt, params, *args, **kwargs)

    processor.process_inputs = process_inputs
    processor._qwen_c2_contract = True
    log('request contract installed: greedy, max_tokens <= %d, prompt <= %s, eos %s', budget,
        max_prompt_tokens or 'context - budget', sorted(eos_ids))
    if min_answer_tokens is not None or default_max_tokens is not None:
        log('request contract: every prompt keeps >= %s answer tokens, max_tokens defaults to %s when omitted',
            min_answer_tokens if min_answer_tokens is not None else budget,
            default_max_tokens if default_max_tokens is not None else 'the clamp')


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


def boot(environ=None, orig_argv=None):
    environ = os.environ if environ is None else environ
    if environ.get('QWEN_C2_SERVING') != '1':
        return None
    fix_sys_path()
    profile = load_profile(environ.get('QWEN_C2_PROFILES', PROFILES))
    problems = gate_problems(profile, environ)
    if problems:
        raise ValueError('; '.join(problems))
    problems = prefix_reuse_problems(profile)
    if problems:
        raise ValueError('profile %s cannot serve prefix reuse exactly: %s' % (profile['name'], '; '.join(problems)))
    apply_environment(profile, environ)
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
        sys.argv[:] = rewrite_argv(sys.argv, profile, snapshot)
        log('profile %s: vLLM argv %s', profile['name'], json.dumps(sys.argv[1:]))
        log('mesh %s, output budget %d, context %s', environ['TT_MESH_GRAPH_DESC_PATH'], budget,
            profile['engine'].get('max-model-len'))
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
        if profile.get('request_contract', True) is False:
            log('profile %s: no request contract (the fast path is off)', profile['name'])
            return profile
        sys.meta_path.insert(0, PostImportHook(
            INPUT_PROCESSOR, lambda module: install_request_contract(
                module, budget=budget, eos_ids=eos_ids, max_prompt_tokens=limits['max_prompt_tokens'],
                min_answer_tokens=limits['min_answer_tokens'], default_max_tokens=limits['default_max_tokens'])))
    return profile
