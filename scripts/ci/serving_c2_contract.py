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
   general. The checkpoint registry's store policies and telemetry (QWEN_PREFIX_EVICT, _SUPERSEDE,
   _TELEMETRY, _GHOST_ENTRIES) are the profile's too: strict values, and only beside the switch
   (prefix_policy_problems). Under a prefix profile the API server serves the registry's metrics, which the engine
   process exports (qwen_prefix_metrics); it keeps a request's cache_salt only when the salt verifies
   against the platform's salt key (salt_verdict), so a client cannot choose the cache partition it
   shares; it refuses a launched KV connector and warns on a server default that strips past
   reasoning (prefix_launch_problems); and every process logs where the model entry was imported
   from (the bring-up check, qwen_prefix_stage bringup). The fast path joins prefix reuse only
   through sticky sessions (QWEN_FAST_STICKY_SESSIONS, the c2-packed-prefix profiles): the profile
   alone owns that switch too, it needs QWEN_PREFIX_REUSE=1 and the fast path beside it, and the
   one speculative shape reuse then accepts is DFlash with 15 proposals.
6. gate-only profiles (gate_only: true) boot only with QWEN_C2_GATE=1: they exist for a gate and
   must never take traffic. A TRAFFIC profile may carry a lever whose traffic refusal rests on a missing
   pinned qualification record (the W2 levers, engine reuse) only through its OWNER TRAFFIC WAIVER
   (owner_traffic_waiver, traffic_waiver_problems): a named record of the owner's decision that lists each
   waived lever with its exact value, what is waived and the card evidence; the boot logs it. A gate
   instrument (an audit, a fault, a negative control) is never waivable.
7. streaming parsers: C2 commits 3-16 tokens per engine step, so the API server's reasoning and
   tool-call parsers see multi-token deltas, which they parse differently from one token per step
   (R21: a real </think> or <tool_call> bound to a later lookalike, whitespace after a tool call).
   A profile with "parser_rechunk": true (c2, c2-gate) arms c2_parser_rechunk (fix M) in the API
   server: when vLLM imports vllm.parser.abstract_parser, DelegatingParser.parse_delta is wrapped
   to feed the parsers one token per sub-delta where that matters and merge the results into one
   message per step. general, the general-prefix profiles and exact never arm it (general and
   general-prefix decode one token per step). Every profile, in the same API server, arms
   c2_toolcall_whitespace (QWEN_C2_TOOLCALL_WS=0 or "toolcall_whitespace": false turns it off): vLLM 0.25.1's
   qwen3 tool parser strip()s every <parameter=...> value, which loses the trailing newline of a written file
   and the indentation of an edit; the fix removes only the template's one framing newline at each end.
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

9. another checkpoint of the same architecture (qwen_c2_checkpoint.py, qwen_c2_checkpoints.json; default off): QWEN_C2_MODEL_REPO and QWEN_C2_MODEL_REVISION, both, a pair the
   catalog lists and that differs from the base, move the profile's snapshot to that checkpoint's, key the converted-weight directory (TT_CACHE_PATH) by the pair so two
   checkpoints never share a cache, and refuse a snapshot whose config, tokenizer or chat template are not the base's byte for byte. Unset, the base is served exactly as before.

Nothing here changes a gate: every step is off unless QWEN_C2_SERVING=1.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import sys

PROFILES = '/opt/qwen-c2/profiles.json'
FAST_PATHS = ('/experiment-scripts/ci', '/speculative-decoding/harness', '/opt/tt-metal/ttnn', '/opt/tt-metal')
API_SERVER = 'vllm.entrypoints.openai.api_server'
INPUT_PROCESSOR = 'vllm.v1.engine.input_processor'
# The prefix-reuse switch (design section 2.0.1): the model graft's supports_prefix_caching, the
# scheduler graft's install and the runner patch all read it. Only a profile sets it.
PREFIX_SWITCH = 'QWEN_PREFIX_REUSE'
# The checkpoint registry's store policies and telemetry (qwen_prefix_registry.ENV_EVICT, ENV_SUPERSEDE, ENV_TELEMETRY, ENV_GHOST): a profile's
# own env like the switch, strict in their values, and only meaningful beside it (prefix_policy_problems). Unset they are the registry's defaults:
# plain LRU, no supersession, telemetry on.
PREFIX_POLICY_FLAGS = {'QWEN_PREFIX_EVICT': ('lru', 'fair'), 'QWEN_PREFIX_SUPERSEDE': ('0', '1'), 'QWEN_PREFIX_TELEMETRY': ('0', '1'),
                       'QWEN_PREFIX_CKPT_PRECONVERTED': ('0', '1'), 'QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT': ('0', '1'),
                       'QWEN_PREFIX_HOST_TIER_AUDIT': ('0', '1'), 'QWEN_PREFIX_HOST_TIER_VERIFY': ('sample', 'all', 'off'),
                       'QWEN_PREFIX_HOST_TIER_SPILL_MODE': ('cap', 'complete')}
PREFIX_GHOST_FLAG = 'QWEN_PREFIX_GHOST_ENTRIES'
# The host KV tier's numbers (qwen_prefix_registry.tier_config) and the files and switches that only a gate sets. None of the tier or the preconverted
# checkpoints has run on a card: a traffic profile may not name them active until a qualification record lifts this.
PREFIX_TIER_GIB = 'QWEN_PREFIX_HOST_TIER_GIB'
# (the complete-spill mode's three knobs are tier numbers too: docs/tp4-sched-1.md section 3; they mean something only beside QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete)
PREFIX_SPILL_MODE = 'QWEN_PREFIX_HOST_TIER_SPILL_MODE'
PREFIX_SPILL_KNOBS = ('QWEN_PREFIX_HOST_TIER_SPILL_HARD_MAX_BLOCKS', 'QWEN_PREFIX_HOST_TIER_SPILL_BATCH_BLOCKS', 'QWEN_PREFIX_HOST_TIER_SPILL_WARN_MS')
PREFIX_TIER_NUMBERS = (PREFIX_TIER_GIB, 'QWEN_PREFIX_HOST_TIER_SPILL_MAX_BLOCKS', 'QWEN_PREFIX_HOST_TIER_MIN_TOKENS',
                       'QWEN_PREFIX_HOST_TIER_MIN_AVAILABLE_GIB') + PREFIX_SPILL_KNOBS
PREFIX_TIER_OFF_PATH = 'QWEN_PREFIX_HOST_TIER_OFF_PATH'
PREFIX_GATE_ONLY_VALUES = (('QWEN_PREFIX_CKPT_PRECONVERTED', '1'), ('QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT', '1'),
                           ('QWEN_PREFIX_HOST_TIER_AUDIT', '1'), (PREFIX_SPILL_MODE, 'complete'))
# Every name of the two features: the profile's to switch on, so a value inherited from the process under a profile that does not name it is dropped
# (apply_environment), as Lever N's, engine reuse's and W2's are.
PREFIX_TIER_ENV_NAMES = (tuple(name for name, _value in PREFIX_GATE_ONLY_VALUES) + PREFIX_TIER_NUMBERS
                         + (PREFIX_TIER_OFF_PATH, 'QWEN_PREFIX_HOST_TIER_VERIFY'))
# The per-tenant quotas of the prefix store (qwen_prefix_registry.QUOTA_ENV_NAMES; docs/prefix-store-hygiene.md "Tenant quotas"): default off and byte-identical off, strict in
# their values, meaningful only beside the switch, and gate-only until a qualification record exists (a traffic profile may not name one). The device numbers are tokens in
# whole 64-token blocks (a cap at least one 2048-token chunk, a floor at most the cap); the host numbers are GiB and need the host tier.
PREFIX_QUOTA_TOKENS = ('QWEN_PREFIX_TENANT_QUOTA_DEVICE_CAP_TOKENS', 'QWEN_PREFIX_TENANT_QUOTA_DEVICE_FLOOR_TOKENS')
PREFIX_QUOTA_GIB = ('QWEN_PREFIX_TENANT_QUOTA_HOST_CAP_GIB', 'QWEN_PREFIX_TENANT_QUOTA_HOST_FLOOR_GIB')
PREFIX_QUOTA_NAMES = PREFIX_QUOTA_TOKENS + PREFIX_QUOTA_GIB
# The sliced restore of the host KV tier (qwen_prefix_registry.ENV_TIER_RESTORE_SLICE; docs/prefix-store-hygiene.md "Sliced restore"): default off and byte-identical off, strict, only
# beside the tier (it changes HOW a returning session's blocks come back, never which), and a traffic profile may name it only when its owner traffic waiver names the tier.
PREFIX_SLICE_BLOCKS = 'QWEN_PREFIX_TIER_RESTORE_SLICE'
PREFIX_SLICE_MS = 'QWEN_PREFIX_TIER_RESTORE_SLICE_MS'
PREFIX_SLICE_NAMES = (PREFIX_SLICE_BLOCKS, PREFIX_SLICE_MS)
# The upload levers of the tier's host side (qwen_tier_upload.NAMES, pinned equal by test_qwen_tier_upload; docs/tp4-upload1.md): threaded staging copies of the raw block IO and
# parallel digest checks. Like the sliced restore: default off and byte-identical off, strict, only beside the tier (they change how long the same bytes take to move, never which),
# and a traffic profile may name them only when its owner traffic waiver names the tier.
PREFIX_UPLOAD_NAMES = ('QWEN_PREFIX_TIER_IO_THREADS', 'QWEN_PREFIX_TIER_IO_CHUNK_BLOCKS', 'QWEN_PREFIX_TIER_VERIFY_THREADS')
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
LEVERN_ENV_FLAGS = ('QWEN_FAST_LEVER_N', 'QWEN_FAST_LEVERN_AUDIT', 'QWEN_FAST_LEVERN_BUILD_MS', 'QWEN_FAST_LEVERN_STEP_TOKENS', 'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS',
                    'QWEN_FAST_LEVERN_PREFILL_SHARE', 'QWEN_FAST_LEVERN_ROUNDS', 'QWEN_FAST_LEVERN_MAX_ROUNDS', 'QWEN_FAST_LEVERN_FAULT',
                    'QWEN_FAST_LEVERN_TTFT_TARGET_S', 'QWEN_FAST_LEVERN_SHORT_TOKENS', 'QWEN_FAST_LEVERN_PARK', 'QWEN_FAST_LEVERN_PARK_SLOTS',
                    'QWEN_FAST_LEVERN_MAX_PARK_S', 'QWEN_FAST_LEVERN_EPOCH_SCOPE',
                    'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S',
                    'QWEN_FAST_LEVERN_KV_READ', 'QWEN_FAST_LEVERN_KV_CROSS_STEPS',
                    'QWEN_FAST_LEVERN_ADAPTIVE', 'QWEN_FAST_LEVERN_ADAPTIVE_DECODERS', 'QWEN_FAST_LEVERN_ADAPTIVE_SHARE', 'QWEN_FAST_LEVERN_ADAPTIVE_TTFT_S',
                    'QWEN_FAST_LEVERN_OWED_CAP_MS', 'QWEN_FAST_LEVERN_SHORT_SHARE', 'QWEN_FAST_LEVERN_SHORT_WAIT_S')
# The adaptive governor (levern_policy.ADAPTIVE_FLAGS) and the production-shape pacing (levern_policy.SCHED_FLAGS, docs/tp4-sched-1.md): gate instruments, never traffic.
LEVERN_ADAPTIVE_FLAGS = ('QWEN_FAST_LEVERN_ADAPTIVE', 'QWEN_FAST_LEVERN_ADAPTIVE_DECODERS', 'QWEN_FAST_LEVERN_ADAPTIVE_SHARE', 'QWEN_FAST_LEVERN_ADAPTIVE_TTFT_S',
                         'QWEN_FAST_LEVERN_OWED_CAP_MS', 'QWEN_FAST_LEVERN_SHORT_SHARE', 'QWEN_FAST_LEVERN_SHORT_WAIT_S')
LEVERN_PLATFORM_MODULE = 'vllm_tt_plugin.platform'
# Engine reuse, parked per-slot engines (serving_fast_policy.PARKED_ENGINES_FLAG; default off). A gate-only profile's own switches: the fast
# path's engine is rebound to each request instead of built (serving_parked_engines), and with PARKED_DRAFTS the pair and quad drafter traces are
# bound to the slots too. Only a profile sets them, and only on the four-card S2 shape they need (parked_problems). The audit knob may be a gate
# profile's own (the audit twin sets it, and its flag-off control carries it too), as are the negative controls, the injected faults and the ballast
# (each its own gate-only twin, as QWEN_FAST_LEVERN_FAULT is): a profile that is not a gate's carries none of them, and outside a gate profile the boot
# refuses them in the process environment.
PARKED_SWITCH = 'QWEN_FAST_PARKED_ENGINES'
PARKED_DRAFTS = 'QWEN_FAST_PARKED_DRAFTS'
PARKED_PREFIX = 'QWEN_FAST_PARKED_'
PARKED_NAMES = ('QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_PARKED_DRAFTS', 'QWEN_FAST_PARKED_PROJECT_ROWS', 'QWEN_FAST_PARKED_AUDIT',
                'QWEN_FAST_PARKED_NEGATIVE', 'QWEN_FAST_PARKED_FAULT', 'QWEN_FAST_PARKED_OFF_AFTER', 'QWEN_FAST_PARKED_OFF_PATH')
PARKED_GATE_ONLY = ('QWEN_FAST_PARKED_AUDIT', 'QWEN_FAST_PARKED_NEGATIVE', 'QWEN_FAST_PARKED_FAULT', 'QWEN_FAST_GATE_DRAM_BALLAST',
                    'QWEN_FAST_PARKED_OFF_AFTER', 'QWEN_FAST_PARKED_OFF_PATH')
# The W2 runtime kill switch (w2_switch: the file /models/.qwen-c2/w2.off beside levern.off, prefix-reuse.off and parked.off). QWEN_FAST_W2_OFF_PATH moves the
# file (any profile may name it; absolute path), QWEN_FAST_W2_OFF_AFTER is the gate-only drill that writes the file itself after n packed rounds. Pinned equal to
# w2_switch's by test_w2_switch. Any other QWEN_FAST_W2_ name is refused, so a typo never reads as "off".
W2_PREFIX = 'QWEN_FAST_W2_'
W2_NAMES = ('QWEN_FAST_W2_OFF_PATH', 'QWEN_FAST_W2_OFF_AFTER')
W2_GATE_ONLY = ('QWEN_FAST_W2_OFF_AFTER',)
# octo-T8 and its three levers (serving_octo, octo_draft_tp, octo_attn_bundle, octo_glue8; tp4/octo-3, docs/tp4-octo3.md). Pinned equal to the modules' names by test_octo3_contract. A GATE-ONLY profile
# carries them behind its gate marker (QWEN_C2_GATE_PROFILE=1: serving_octo admits them in a gate run only). A TRAFFIC profile carries the four levers only under its owner traffic waiver
# (octo_problems): the shape and the levers have no pinned qualification record (the K64j 0x21 eight-row group per bundle, B > 1 entries of it, the batched GDN launch at eight users, the
# quarter-tile glue movers), so the waiver names each at its exact value, and the profile's own QWEN_FAST_OCTO_WAIVER carries the waiver's id into the process (the one name by which
# serving_octo.waiver_run admits a non-gate run; the contract refuses it anywhere else and an inherited value is removed). The bundle's audit and the audit sampling are a gate's own.
OCTO_SWITCH = 'QWEN_FAST_OCTO'
OCTO_WAIVER_ENV = 'QWEN_FAST_OCTO_WAIVER'
OCTO_LEVERS = ('QWEN_FAST_OCTO', 'QWEN_FAST_OCTO_DRAFT', 'QWEN_FAST_OCTO_ATTN_BUNDLE', 'QWEN_FAST_OCTO_GLUE8')
OCTO_KNOBS = ('QWEN_FAST_OCTO_MIN_LIVE', 'QWEN_FAST_OCTO_DRAFT_CONV')
OCTO_GATE_ONLY = ('QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT', 'QWEN_FAST_SOLO_PACKED', 'QWEN_FAST_AUDIT_FIRST', 'QWEN_FAST_AUDIT_EVERY')
SOLO_PACKED_SWITCH = 'QWEN_FAST_SOLO_PACKED'        # NP1 (serving_octo.SOLO_PACKED_FLAG): gate-only, or named by a traffic profile's owner waiver (NEXT_WAIVABLE_LEVERS)
OCTO_NAMES = OCTO_LEVERS + OCTO_KNOBS + OCTO_GATE_ONLY + (OCTO_WAIVER_ENV,)
# The engine-start upload levers (qwen_device_zeros: the KV pool and the buffer pool's zero buffers filled on the device; qwen_lazy_shard: a weight-cache hit
# without the host transpose). Pinned equal to the modules' NAMES by test_upload_p0. A profile sets them (default off, strict '0'/'1'); the audit twins are a gate
# profile's own (the audit reads pages back and costs engine-start time); an inherited value never turns a lever on under a profile that did not name it.
UPLOAD_NAMES = ('QWEN_FAST_DEVICE_ZEROS', 'QWEN_FAST_DEVICE_ZEROS_AUDIT', 'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES', 'QWEN_FAST_DEVICE_ZEROS_AUDIT_TENSORS',
                'QWEN_FAST_DEVICE_ZEROS_AUDIT_BLOCKS', 'QWEN_FAST_LAZY_SHARD_W', 'QWEN_FAST_LAZY_SHARD_W_AUDIT', 'QWEN_FAST_LAZY_SHARD_W_AUDIT_LOADS')
UPLOAD_GATE_ONLY = ('QWEN_FAST_DEVICE_ZEROS_AUDIT', 'QWEN_FAST_LAZY_SHARD_W_AUDIT')
# The sources the merged route must carry (levern_route.SOURCES, pinned equal by test_levern_prefix_contract).
MERGED_SOURCES = ('COLD', 'CHECKPOINT', 'SCRATCH', 'PARKED')

# Sampled verification (docs/tp4-sampled-verify.md; default off, gate-only until a card has judged it). With QWEN_FAST_SAMPLED=1 the request contract honours
# a request's temperature, top_p and top_k (the model's own defaults for what it omits; temperature 0 stays greedy) instead of forcing greedy, and carries the
# request seed (the client's, else a random one) to the engine in SamplingParams.extra_args. The verifier that DRAWS the tokens is the device step's: until it
# exists the engine-side gate (serving_fast_policy.validate_request_sampling) still refuses a non-greedy request, so the switch fails closed. Only a profile
# sets the switch (apply_environment drops an inherited value); no shipped profile does.
SAMPLED_SWITCH = 'QWEN_FAST_SAMPLED'
SAMPLED_K_MAX = 64                  # sampled_accept.K_MAX: the candidates per row the filter sees (a test holds both equal; not imported: this module cannot import numpy at boot)
SAMPLED_MODEL_DEFAULTS = dict(temperature=1.0, top_p=0.95, top_k=20)   # generation_config.json of the base snapshot = sampled_accept.MODEL_DEFAULTS (held equal by a test)
SAMPLED_SEED_KEY = 'qwen_sampled_seed'     # the SamplingParams.extra_args key that carries the request seed to the engine

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


# The names configure_checkpoint reads (qwen_c2_checkpoint.REPO_ENV, REVISION_ENV and the derived ones it sets); a test holds them equal to the module's.
CHECKPOINT_ENV = ('QWEN_C2_MODEL_REPO', 'QWEN_C2_MODEL_REVISION', 'QWEN_C2_CHECKPOINT_KEY', 'QWEN_C2_CHECKPOINT_SNAPSHOT', 'QWEN_C2_WEIGHT_CACHE_BASE')


def configure_checkpoint(profile, environ, api_server, exists=os.path.isdir):
    """Item 9. -> the snapshot the engine serves instead of the profile's own, or None (nothing selected, or the base itself: the environment is left as it was).

    Every process sets the environment (the keyed TT_CACHE_PATH, the derived names) so a child inherits and re-derives the same values; the identity checks, the cache marker and
    the log line are the API server's alone. Any refusal is a ValueError: the API server exits (status 78) rather than serve another checkpoint's bytes under the base's name."""
    if not any(environ.get(name) for name in CHECKPOINT_ENV):
        return None      # nothing asked and nothing inherited: no import, no change (the off path is the code that was here before)
    import qwen_c2_checkpoint as checkpoint

    try:
        sel = checkpoint.selection(environ)
        if sel is None or sel.is_base:
            checkpoint.configure_environment(environ, sel, None)
            if sel is not None and api_server:
                log('model is the base %s@%s: nothing moved (the base snapshot, its own converted weights)', sel.repo, sel.revision)
            return None
        catalog = checkpoint.load_catalog()
        snapshot = checkpoint.resolve_snapshot(profile['snapshots'], sel, catalog, exists)
        facts = checkpoint.configure_environment(environ, sel, snapshot)
        budget = checkpoint.request_overrides(sel).get('output_budget')
        if budget is not None and not checkpoint_budget_problems(profile, request_limits(profile), budget):
            # Every process, like the keyed cache: serving_fast_policy reads QWEN_FAST_OUTPUT_BUDGET when the worker imports it, after this boot has laid the checkpoint's
            # ceiling over the profile's value (apply_environment ran first and wrote the profile's). A profile that cannot carry it keeps its own (checkpoint_limits says so, once).
            environ[OUTPUT_BUDGET_ENV] = str(budget)
        if api_server:
            problems = checkpoint.verify_snapshot(snapshot, sel, catalog)
            if problems:
                raise checkpoint.CheckpointError('; '.join(problems))
            claim = checkpoint.claim_cache(environ, sel)
            log('%s', checkpoint.log_text(facts, (len(catalog['identity_files']), len(sel.entry['shards']))))
            log('weight cache %s: %s%s', facts['weight_cache'], 'created' if claim['created'] else 'found',
                '' if claim['first_boot_complete'] or claim['created'] else ' (its first boot never completed: a truncated tensor file fails the load loudly; delete the directory to rebuild)')
        return snapshot
    except checkpoint.CheckpointError as error:
        raise ValueError('profile %s cannot serve the asked checkpoint: %s' % (profile['name'], error))


def checkpoint_limits(limits, environ, profile=None):
    """Item 9: request_limits(profile) with the selected checkpoint's catalog numbers laid over it (qwen_c2_checkpoints.json "request": default_max_tokens, the max_tokens a
    request that omits it gets, and output_budget, the ceiling on max_tokens, both within the same clamp, prompt cap and KV admission). No selection, or the base: `limits` itself, nothing read.
    An output_budget needs `profile` (checkpoint_budget_problems reads its window and flags) and is laid over a profile that can carry it only: any other keeps its own ceiling, and the log says why.
    configure_checkpoint exports the same number as QWEN_FAST_OUTPUT_BUDGET in every process, by the same rule."""
    if not any(environ.get(name) for name in CHECKPOINT_ENV[:2]):
        return limits
    import qwen_c2_checkpoint as checkpoint

    try:
        changes = checkpoint.request_overrides(checkpoint.selection(environ))
    except checkpoint.CheckpointError as error:
        raise ValueError(str(error))
    if not changes:
        return limits
    out = dict(limits)
    budget = changes.get('output_budget')
    if budget is not None:
        if profile is None:
            raise ValueError('the checkpoint\'s output_budget %r is laid over a profile, and none was given' % (budget,))
        refusals = checkpoint_budget_problems(profile, limits, budget)
        if refusals:
            log('request output budget stays %d: profile %s cannot carry the checkpoint\'s output_budget %d (%s)', limits['budget'], profile.get('name'), budget, '; '.join(refusals))
        else:
            log('request output budget %d for this checkpoint (its catalog entry; the profile says %d); the prompt cap %s and the KV admission are unchanged: a request reserves its prompt + max_tokens blocks',
                budget, limits['budget'], limits['max_prompt_tokens'])
            out['budget'] = budget
    default = changes.get('default_max_tokens')
    if default is not None:
        if not 1 <= default <= out['budget']:
            raise ValueError('the checkpoint\'s default_max_tokens %r is not an integer from 1 to the output budget %d' % (default, out['budget']))
        log('request default max_tokens %d for this checkpoint (its catalog entry; the profile says %s); the clamp, the %d-token output budget and the KV admission are unchanged',
            default, limits['default_max_tokens'], out['budget'])
        out['default_max_tokens'] = default
    return out


BASE_MODEL_ID = 'Qwen/Qwen3.8-27B'


def platform_model(argv):
    """The value of vLLM's --model in an argv (the platform's, before rewrite_argv drops it), or None."""
    for index, token in enumerate(argv):
        if token == '--model':
            return argv[index + 1] if index + 1 < len(argv) else None
        if token.startswith('--model='):
            return token[len('--model='):]
    return None


def check_platform_model(argv, environ):
    """Item 9: the model id the platform asks for must be the model this image serves (qwen_c2_checkpoint.platform_model_problems). The usual case, the base id or no id, costs nothing."""
    asked = platform_model(argv)
    selected = any(environ.get(name) for name in CHECKPOINT_ENV[:2])
    if not selected and (asked is None or asked == BASE_MODEL_ID):
        return
    import qwen_c2_checkpoint as checkpoint

    try:
        problems = checkpoint.platform_model_problems(asked, checkpoint.selection(environ), checkpoint.load_catalog())
    except checkpoint.CheckpointError as error:
        problems = [str(error)]
    if problems:
        raise ValueError('; '.join(problems))


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
    # The admission accounting trim (tp4/sched-1, docs/tp4-sched-1.md) likewise: an inherited switch must not skip a check under a profile that never named it.
    for name in SCHED_NAMES:
        if name not in profile['env']:
            environ.pop(name, None)
    # The drafter checkpoint likewise: an inherited id under a profile that names none would pin a candidate the profile's argv does not point at.
    if DRAFTER_CHECKPOINT_FLAG not in profile['env']:
        environ.pop(DRAFTER_CHECKPOINT_FLAG, None)
    # ...and engine reuse: an inherited value must not turn the parked engines (or their bound drafter traces) on under a profile that never asked,
    # nor a gate instrument (the audit, a negative control, a fault, the ballast, the kill switch's trigger) into an arm that did not name it.
    for name in sorted(set(PARKED_NAMES) | set(PARKED_GATE_ONLY) | set((PARKED_SWITCH, PARKED_DRAFTS))):
        if name not in profile['env']:
            environ.pop(name, None)
    # ...and the W2 kill switch's override and drill trigger (w2_switch): an inherited path would move the switch's file, an inherited trigger would
    # kill W2 by itself after n rounds in an arm that never named it.
    for name in W2_NAMES:
        if name not in profile['env']:
            environ.pop(name, None)
    # ...and the preconverted checkpoints and the host KV tier (default off, gate only until a qualification record lifts it): an inherited
    # switch must not turn either on under a profile that never named it.
    for name in PREFIX_TIER_ENV_NAMES + PREFIX_QUOTA_NAMES + PREFIX_SLICE_NAMES + PREFIX_UPLOAD_NAMES:
        if name not in profile['env']:
            environ.pop(name, None)
    # ...and the engine-start upload levers: an inherited switch must not move the KV pool's or the weights' load path under a profile that never asked.
    for name in UPLOAD_NAMES:
        if name not in profile['env']:
            environ.pop(name, None)
    # ...and octo-T8: an inherited flag must not turn the shape (or the waiver id that admits it in a non-gate run, or the audit sampling that thins a gate's audits) on under a profile that
    # never named it.
    for name in OCTO_NAMES:
        if name not in profile['env']:
            environ.pop(name, None)
    # ...and sampled verification: an inherited switch must not make the request contract stop forcing greedy under a profile that never named it.
    if SAMPLED_SWITCH not in profile['env']:
        environ.pop(SAMPLED_SWITCH, None)
    # ...and exact resume P0: an inherited switch must not lift vLLM's drop or carry a window under a profile that never asked.
    for name in EXACT_RESUME_NAMES:
        if name not in profile['env']:
            environ.pop(name, None)
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


def prefix_policy_problems(profile):
    """Every way the profile sets the checkpoint registry's store policies or telemetry (PREFIX_POLICY_FLAGS, PREFIX_GHOST_FLAG) wrongly, [] when
    none: a value the registry would refuse to start on, or any of them without QWEN_PREFIX_REUSE=1 (they would be inert, and a profile that
    names a policy it does not run is a misread of what the engine does)."""
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    problems = []
    for name, allowed in sorted(PREFIX_POLICY_FLAGS.items()):
        if name in env and env[name] not in allowed:
            problems.append('%s=%r is not one of %s' % (name, env[name], ', '.join(allowed)))
    if PREFIX_GHOST_FLAG in env:
        value = env[PREFIX_GHOST_FLAG]
        if not value.isdigit():
            problems.append('%s=%r is not a number of entries' % (PREFIX_GHOST_FLAG, value))
    numbers = {}
    for name in PREFIX_TIER_NUMBERS:
        if name in env:
            try:
                numbers[name] = float(env[name])
            except ValueError:
                numbers[name] = None
            if numbers[name] is None or not numbers[name] >= 0.0 or numbers[name] == float('inf'):
                problems.append('%s=%r is not a non-negative number' % (name, env[name]))
    if numbers.get(PREFIX_TIER_GIB):
        store = env.get('QWEN_PREFIX_STORE_GIB')
        try:
            store_gib = float(store) if store not in (None, '') else 8.0
        except ValueError:
            store_gib = None
        if store_gib is not None and numbers[PREFIX_TIER_GIB] <= store_gib:
            problems.append('%s=%s is the whole prefix state\'s host budget and must exceed QWEN_PREFIX_STORE_GIB (%s): nothing would be left '
                            'for KV pages' % (PREFIX_TIER_GIB, env[PREFIX_TIER_GIB], store_gib))
    if PREFIX_TIER_OFF_PATH in env and not re.fullmatch(r'(?!-)[A-Za-z0-9_./-]{1,200}', env[PREFIX_TIER_OFF_PATH]):
        problems.append('%s=%r is not a plain path' % (PREFIX_TIER_OFF_PATH, env[PREFIX_TIER_OFF_PATH]))
    slice_blocks = 0
    if PREFIX_SLICE_BLOCKS in env:
        if not re.fullmatch(r'[0-9]{1,5}', env[PREFIX_SLICE_BLOCKS]) or int(env[PREFIX_SLICE_BLOCKS]) > 65536:
            problems.append('%s=%r is not a whole number of blocks from 0 to 65536 (digits only)' % (PREFIX_SLICE_BLOCKS, env[PREFIX_SLICE_BLOCKS]))
        else:
            slice_blocks = int(env[PREFIX_SLICE_BLOCKS])
    if PREFIX_SLICE_MS in env:
        try:
            millis = float(env[PREFIX_SLICE_MS])
        except ValueError:
            millis = None
        if millis is None or not (0.0 < millis <= 60000.0):
            problems.append('%s=%r is not a number of milliseconds above 0 and at most 60000' % (PREFIX_SLICE_MS, env[PREFIX_SLICE_MS]))
        elif not slice_blocks:
            problems.append('%s is set and %s is not: the time budget would be inert' % (PREFIX_SLICE_MS, PREFIX_SLICE_BLOCKS))
    if slice_blocks and not numbers.get(PREFIX_TIER_GIB):
        problems.append('%s names a sliced restore but %s is not set: the restore would be inert' % (PREFIX_SLICE_BLOCKS, PREFIX_TIER_GIB))
    # The upload levers (qwen_tier_upload.upload_plan refuses all of these at boot): strict values, a call size only beside the IO threads, and the host tier beside them.
    upload_on = []
    if any(name in env for name in PREFIX_UPLOAD_NAMES):
        import qwen_tier_upload

        problems += qwen_tier_upload.flag_problems(env)
        upload_on = [name for name in PREFIX_UPLOAD_NAMES if env.get(name) not in (None, '', '0')]
        if upload_on and not numbers.get(PREFIX_TIER_GIB):
            problems.append('%s name the upload levers of the host tier but %s is not set: there would be nothing to move' % (', '.join(upload_on), PREFIX_TIER_GIB))
    # The complete-spill mode (qwen_prefix_registry.spill_plan refuses all of these at boot): its knobs are whole numbers in plain digits, mean something only beside the mode,
    # and the mode needs the tier it spills to.
    for name in PREFIX_SPILL_KNOBS:
        if name in env and not re.fullmatch(r'[0-9]+', env[name]):
            problems.append('%s=%r is not a whole number in plain digits' % (name, env[name]))
    spill_knobs = [name for name in PREFIX_SPILL_KNOBS if name in env]
    if spill_knobs and env.get(PREFIX_SPILL_MODE) != 'complete':
        problems.append('%s set without %s=complete: the cap mode would ignore it' % (', '.join(spill_knobs), PREFIX_SPILL_MODE))
    if env.get(PREFIX_SPILL_MODE) == 'complete' and not numbers.get(PREFIX_TIER_GIB):
        problems.append('%s=complete needs the host KV tier (%s > 0): there would be nothing to spill to' % (PREFIX_SPILL_MODE, PREFIX_TIER_GIB))
    quota_tokens = {}
    for name in PREFIX_QUOTA_TOKENS:
        if name in env:
            if not re.fullmatch(r'[0-9]+', env[name]) or int(env[name]) % 64:
                problems.append('%s=%r is not a whole number of 64-token blocks (digits only)' % (name, env[name]))
            else:
                quota_tokens[name] = int(env[name])
    cap, floor = quota_tokens.get(PREFIX_QUOTA_TOKENS[0], 0), quota_tokens.get(PREFIX_QUOTA_TOKENS[1], 0)
    if cap and cap < 2048:
        problems.append('%s=%d is below one 2048-token chunk' % (PREFIX_QUOTA_TOKENS[0], cap))
    if cap and floor > cap:
        problems.append('%s=%d exceeds %s=%d' % (PREFIX_QUOTA_TOKENS[1], floor, PREFIX_QUOTA_TOKENS[0], cap))
    quota_gib = {}
    for name in PREFIX_QUOTA_GIB:
        if name in env:
            try:
                quota_gib[name] = float(env[name])
            except ValueError:
                quota_gib[name] = None
            if quota_gib[name] is None or not quota_gib[name] >= 0.0 or quota_gib[name] == float('inf'):
                problems.append('%s=%r is not a finite non-negative number of GiB' % (name, env[name]))
                del quota_gib[name]
    if any(quota_gib.values()) and not numbers.get(PREFIX_TIER_GIB):
        problems.append('%s names a host quota but %s is not set: the quota would be inert' % (', '.join(sorted(name for name in quota_gib if quota_gib[name])), PREFIX_TIER_GIB))
    if quota_gib.get(PREFIX_QUOTA_GIB[0]) and quota_gib.get(PREFIX_QUOTA_GIB[1], 0.0) > quota_gib[PREFIX_QUOTA_GIB[0]]:
        problems.append('%s exceeds %s' % (PREFIX_QUOTA_GIB[1], PREFIX_QUOTA_GIB[0]))
    if not gate_profile(profile):
        # A traffic profile may name the tier and the preconverted checkpoints only under its owner traffic waiver (TIER_WAIVABLE_LEVERS); the audits and
        # the kill-switch path are a gate's own.
        unqualified = [name for name, value in PREFIX_GATE_ONLY_VALUES
                       if env.get(name) == value and not ((name in TIER_WAIVABLE_LEVERS or name in NEXT_WAIVABLE_LEVERS) and waived(profile, name, env))]
        if numbers.get(PREFIX_TIER_GIB) and not waived(profile, PREFIX_TIER_GIB, env):
            unqualified.append(PREFIX_TIER_GIB)
        if PREFIX_TIER_OFF_PATH in env:
            unqualified.append(PREFIX_TIER_OFF_PATH)
        if slice_blocks and not waived(profile, PREFIX_TIER_GIB, env):
            problems.append('%s: the sliced restore is the host tier\'s, so a traffic profile may name it only when its %s names the tier (%s)' % (PREFIX_SLICE_BLOCKS, TRAFFIC_WAIVER, PREFIX_TIER_GIB))
        if upload_on and not waived(profile, PREFIX_TIER_GIB, env):
            problems.append('%s: the upload levers are the host tier\'s, so a traffic profile may name them only when its %s names the tier (%s)' % (', '.join(upload_on), TRAFFIC_WAIVER, PREFIX_TIER_GIB))
        quotas = [name for name in PREFIX_QUOTA_NAMES if name in env and env[name] not in ('', '0') and (quota_tokens.get(name) or quota_gib.get(name))]
        if quotas:
            problems.append('%s: no qualification record exists for the per-tenant quotas of the prefix store, so only a gate-only profile may name them' % ', '.join(sorted(quotas)))
        if unqualified:
            problems.append('%s: no qualification record exists for the host KV tier or the preconverted checkpoints, so only a gate-only profile may name them '
                            'active, or a traffic profile whose %s names QWEN_PREFIX_HOST_TIER_GIB and QWEN_PREFIX_CKPT_PRECONVERTED at their values (the audits and '
                            'the kill-switch path are a gate\'s own)' % (', '.join(sorted(unqualified)), TRAFFIC_WAIVER))
    named = [name for name in sorted(list(PREFIX_POLICY_FLAGS) + [PREFIX_GHOST_FLAG, PREFIX_TIER_OFF_PATH] + list(PREFIX_TIER_NUMBERS) + list(PREFIX_QUOTA_NAMES) + list(PREFIX_SLICE_NAMES) + list(PREFIX_UPLOAD_NAMES))
             if name in env]
    if named and not prefix_reuse(profile):
        problems.append('%s set without %s=1: the registry does not exist, so the policy would be inert'
                        % (', '.join(named), PREFIX_SWITCH))
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
    problems = sticky_problems(profile) + prefix_policy_problems(profile)
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


def parked_engines(profile):
    """Whether the profile turns engine reuse on (the gate-only ...-parked profiles)."""
    return str((profile.get('env') or {}).get(PARKED_SWITCH, '')) == '1'


def gate_profile(profile):
    """Whether the profile is a gate's: its name ends in -gate, or it is gate_only."""
    return str(profile.get('name', '')).endswith('-gate') or profile.get('gate_only') is True


# tp4/sched-1 (docs/tp4-sched-1.md): QWEN_FAST_ADMISSION_LEAN takes three admission costs off a returning turn's rebind (the conv-slice readback after a slot's first adoption, the
# 'dram after engine' line of a rebound request, the post-prefill backstop's allocator read when the admission's own hold reading clears it by a wide margin). Strict 0 or 1, and a
# gate-only profile's until the owner decides (serving_prefill_admission.lean_problems names a malformed value, as the process reads it leniently inside an admission).
LEAN_SWITCH = 'QWEN_FAST_ADMISSION_LEAN'
SCHED_NAMES = (LEAN_SWITCH,)


def sched_problems(profile):
    """Every way the profile sets the tp4/sched-1 admission switch wrongly, [] when none: a value that is not 0 or 1, or 1 in a profile that serves traffic."""
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    problems = []
    value = env.get(LEAN_SWITCH)
    if value is not None and value not in ('0', '1'):
        problems.append('%s must be 0 or 1, got %r' % (LEAN_SWITCH, value))
    elif value == '1' and not gate_profile(profile) and not waived(profile, LEAN_SWITCH, env):
        problems.append('%s=1 is a gate arm (it skips admission checks that no card run has shown unnecessary) and needs a gate-only profile, or a traffic profile whose %s names it' % (LEAN_SWITCH, TRAFFIC_WAIVER))
    return problems


# tp4/sched-2 (docs/tp4-sched2.md; admission_pipeline.py): admission pipelining and the memory-ledger trim. Three switches and the hold's three knobs, all gate arms until the owner decides: a
# traffic profile carries none of them, whatever its waiver names. Strict, and the same arithmetic as admission_pipeline.problems() (the profile reads its own env here, the process reads
# its own there; test_sched2_pipeline pins the two to the same verdicts).
SCHED2_SWITCHES = ('QWEN_FAST_ADMISSION_SEED_FIRST', 'QWEN_FAST_ADMISSION_HOLD_REUSE', 'QWEN_FAST_ADMISSION_LEDGER_TRIM')
SCHED2_HOLD_KNOBS = {'QWEN_FAST_ADMISSION_HOLD_REUSE_MAX_AGE_MS': (100, 60000), 'QWEN_FAST_ADMISSION_HOLD_REUSE_MAX_REUSES': (1, 16),
                     'QWEN_FAST_ADMISSION_HOLD_REUSE_DRIFT_MB': (16, 1024)}


def sched2_problems(profile):
    """Every way the profile sets the tp4/sched-2 switches wrongly, [] when none: a value that is not 0 or 1, a knob outside its range or without the hold's switch, a switch without what it rides
    on (the hold needs the extent replay, seed first engine reuse, the lean admission and Lever N), or any of them in a profile that serves traffic."""
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    problems = []
    for name in SCHED2_SWITCHES:
        value = env.get(name)
        if value is not None and value not in ('0', '1'):
            problems.append('%s must be 0 or 1, got %r' % (name, value))
        elif value == '1' and not gate_profile(profile) and not (name in SCHED2_WAIVABLE_LEVERS and waived(profile, name, env)):
            problems.append('%s=1 is a gate arm (docs/tp4-sched2.md: it changes when work happens at an admission, and no card has run it) and needs a gate-only profile%s'
                            % (name, ', or a traffic profile whose %s names it' % TRAFFIC_WAIVER if name in SCHED2_WAIVABLE_LEVERS else ''))
    for name, (low, high) in sorted(SCHED2_HOLD_KNOBS.items()):
        value = env.get(name)
        if value is None:
            continue
        if not value.isascii() or not value.isdigit() or not low <= int(value) <= high:
            problems.append('%s must be a whole number from %d to %d in plain digits, got %r' % (name, low, high, value))
        if env.get('QWEN_FAST_ADMISSION_HOLD_REUSE') != '1':
            problems.append('%s is set without QWEN_FAST_ADMISSION_HOLD_REUSE=1: it would be ignored' % name)
    if env.get('QWEN_FAST_ADMISSION_HOLD_REUSE') == '1' and env.get('QWEN_FAST_EXTENT_REPLAY') != '1':
        problems.append('QWEN_FAST_ADMISSION_HOLD_REUSE=1 needs QWEN_FAST_EXTENT_REPLAY=1: the DRAM hold it reuses is registered only there')
    if env.get('QWEN_FAST_ADMISSION_SEED_FIRST') == '1':
        problems += ['QWEN_FAST_ADMISSION_SEED_FIRST=1 needs %s=1' % name for name in ('QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_ADMISSION_LEAN', 'QWEN_FAST_LEVER_N') if env.get(name) != '1']
    return problems


def parked_problems(profile, environ=None):
    """Every way engine reuse is set without what it needs, [] when none: the switch is the fast path's (qwen_fast_t16), the four-card mesh's
    and a gate-only profile's (nothing serves traffic on it until its exactness, lifecycle, memory and hang gates have run), and needs the S2
    shape beside it - C2-any, the extent readers, four or eight scheduler requests. The gate instruments (the audit, the negative controls, the injected
    faults, the ballast) are refused in the env of a profile that is not a gate's, and in the process environment outside a gate profile. Any other QWEN_FAST_PARKED_ name is refused. The drafts switch (the bound pair and quad traces) needs the engines."""
    environ = {} if environ is None else environ
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    engine = profile.get('engine') or {}
    problems = ['%s is not a parked-engines setting' % name for name in sorted(env) if name.startswith(PARKED_PREFIX) and name not in PARKED_NAMES]
    value = env.get(PARKED_SWITCH)
    if value is not None and value not in ('0', '1'):
        problems.append('%s=%r is neither 1 nor 0' % (PARKED_SWITCH, value))
    # The negative controls, the injected faults and the ballast, and the audit, are a gate-only profile's own env (as QWEN_FAST_LEVERN_FAULT is): a
    # profile that is not a gate's must carry none of them.
    problems += ['%s is a gate instrument: only a gate-only profile sets it' % name for name in PARKED_GATE_ONLY
                 if name in env and not gate_profile(profile)]
    if not gate_profile(profile):
        problems += ['%s is set outside a gate profile: it exists for a gate and must never reach traffic' % name
                     for name in PARKED_GATE_ONLY if name in environ]
    if env.get(PARKED_DRAFTS) not in (None, '0', '1'):
        problems.append('%s=%r is neither 1 nor 0' % (PARKED_DRAFTS, env.get(PARKED_DRAFTS)))
    if env.get(PARKED_DRAFTS) == '1' and not parked_engines(profile):
        problems.append('%s=1 needs %s=1' % (PARKED_DRAFTS, PARKED_SWITCH))
    if not parked_engines(profile):
        return problems
    if profile.get('gate_only') is not True:
        # A traffic profile serves engine reuse only under the owner's waiver (TRAFFIC_WAIVER), which must name the engines and, when the profile
        # binds the drafter traces too, the drafts switch: each at its exact value.
        for name in (PARKED_SWITCH, PARKED_DRAFTS):
            if env.get(name) == '1' and not waived(profile, name, env):
                problems.append('%s=1 needs a gate-only profile (nothing serves traffic on engine reuse until its gates have run and the owner\'s '
                                '%s names it)' % (name, TRAFFIC_WAIVER))
    if not (engine.get('additional-config') or {}).get('qwen_fast_t16'):
        problems.append('%s=1 is a fast path switch, but the profile does not run the fast path (qwen_fast_t16)' % PARKED_SWITCH)
    for name, wanted in (('QWEN_FAST_ANY_REQUEST', '1'), ('QWEN_FAST_EXTENT_REPLAY', '1'), (FAST_TP_ENV, '4')):
        if env.get(name) != wanted:
            problems.append('%s=1 needs %s=%s in the profile (the four-card S2 shape the parked engines serve)' % (PARKED_SWITCH, name, wanted))
    seats = engine.get('max-num-seqs')
    if seats not in (4, 8):
        problems.append('%s=1 needs max-num-seqs 4 or 8 (one parked engine per packed-block user), not %r' % (PARKED_SWITCH, seats))
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


def sampled_enabled(environ=None):
    """Whether QWEN_FAST_SAMPLED=1. Read per call, like any_request_enabled; anything but unset, '0' or '1' is a configuration error, never a silent off."""
    environ = os.environ if environ is None else environ
    value = environ.get(SAMPLED_SWITCH, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (SAMPLED_SWITCH, value))
    return value == '1'


def sampled_on(profile):
    """Whether the profile turns sampled verification on."""
    return str((profile.get('env') or {}).get(SAMPLED_SWITCH, '')) == '1'


def sampled_problems(profile):
    """Every way the profile's sampled-verify switch is wrong, [] when it names none (every shipped profile) or names 0. On, it needs C2-any (a host-side
    refusal of one request's own terms ends that request, not the engine), the request contract (the only place the sampling terms are read) and a gate-only
    profile: no sampled verifier has run on a card, so nothing serves traffic on it yet."""
    env = profile.get('env') or {}
    if SAMPLED_SWITCH not in env:
        return []
    value = str(env[SAMPLED_SWITCH])
    if value not in ('0', '1'):
        return ['%s must be 0 or 1, got %r' % (SAMPLED_SWITCH, env[SAMPLED_SWITCH])]
    if value == '0':
        return []
    problems = []
    if str(env.get('QWEN_FAST_ANY_REQUEST', '0')) != '1':
        problems.append('%s=1 needs QWEN_FAST_ANY_REQUEST=1 (a refusal of one request\'s sampling terms must end that request, not the engine)' % SAMPLED_SWITCH)
    if profile.get('request_contract', True) is False:
        problems.append('%s=1 needs the request contract (request_contract is false: nothing would read the sampling terms)' % SAMPLED_SWITCH)
    if profile.get('gate_only') is not True:
        problems.append('%s=1 needs a gate-only profile (no sampled verifier has been judged on a card, so nothing serves traffic on it)' % SAMPLED_SWITCH)
    return problems


def sampled_launch_problems(argv):
    """Refusals of a sampled profile's launched argv. The contract cannot tell an omitted top_p/top_k from an explicit one (vLLM has filled the omitted ones by
    the time the request reaches it), so the model's own defaults reach a request that omits them only through vLLM's generation-config fill, which
    `--generation-config vllm` (or a path that is not the model's) turns off."""
    refusals = []
    for value in launched_values(argv, 'generation-config'):
        if value != 'auto':
            refusals.append('--generation-config %s: the model\'s defaults (temperature, top_p, top_k) reach a request that omits them only through "auto"' % (value or '(none)'))
    if launched_values(argv, 'override-generation-config'):
        refusals.append('--override-generation-config: it would move the defaults a request that omits its sampling terms gets away from the model\'s')
    return refusals


MULTI_SWITCH = 'QWEN_FAST_TP4_SDPA'
MULTI_AUDIT = 'QWEN_FAST_TP4_SDPA_AUDIT'
F1_SWITCH = 'QWEN_FAST_TP4_CONV_GATES_SPREAD'
F1_AUDIT = 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'
SDPA_OFF_VALUES = ('', '0', 'off')


# The OWNER TRAFFIC WAIVER (docs/tp4-ship-ln-w2-er.md): the one way a TRAFFIC profile (not gate_only) carries a lever that multi_problems or
# parked_problems refuses on traffic for want of a pinned qualification record. It is a profile field, never an environment variable, so it
# travels with the profile into the image that bakes it (the image is the unit of rollback) and no process environment can grant it. It
# names every lever it waives WITH its exact value (a lever set and not listed is still refused; a lever listed and not set is a stale
# waiver, refused too), says in words what is waived, cites the card evidence and records the owner's decision. Only the four levers below
# are waivable: the audits, the faults, the negative controls, the ballast and the kill-switch trigger stay a gate's own.
TRAFFIC_WAIVER = 'owner_traffic_waiver'
TRAFFIC_WAIVER_FIELDS = ('id', 'levers', 'waives', 'evidence', 'decision')
TRAFFIC_WAIVER_ID = re.compile(r'[a-z0-9][a-z0-9.-]{2,62}')
WAIVABLE_LEVERS = {'QWEN_FAST_TP4_SDPA': 'multi', 'QWEN_FAST_TP4_CONV_GATES_SPREAD': '1', 'QWEN_FAST_PARKED_ENGINES': '1',
                   'QWEN_FAST_PARKED_DRAFTS': '1',
                   # tp4/octo-3 (octo_problems): the octo shape at every eligible eight-live round (never `alternate`, which exists to time two shapes on one boot), its draft pass, the
                   # bundled attention (four users a launch) and the glue levers at eight-row users; QWEN_FAST_OCTO_MIN_LIVE is the knob the shape's eligibility reads (6: the eight
                   # segments less the two idle ones page 0 can hold)
                   'QWEN_FAST_OCTO': 'live', 'QWEN_FAST_OCTO_DRAFT': '1', 'QWEN_FAST_OCTO_ATTN_BUNDLE': '4', 'QWEN_FAST_OCTO_GLUE8': '1', 'QWEN_FAST_OCTO_MIN_LIVE': '6'}
# The host KV tier's two levers a waiver may ALSO name (beside WAIVABLE_LEVERS, which stay the four the earlier ship waivers hold equal): a traffic profile serves the tier
# (QWEN_PREFIX_HOST_TIER_GIB) and the preconverted checkpoints (QWEN_PREFIX_CKPT_PRECONVERTED) only under an owner traffic waiver that names each at this value, as for W2
# and engine reuse (prefix_policy_problems). The audits, the digest-everything mode of a gate and the kill-switch path of a gate stay a gate's own.
TIER_WAIVABLE_LEVERS = {'QWEN_PREFIX_HOST_TIER_GIB': '32', 'QWEN_PREFIX_CKPT_PRECONVERTED': '1'}
# tp4/next-1 (the ship-5 candidate): the tier's budget the owner approved for it (the whole prefix state's host budget: a 32 GiB KV share beside a 16 GiB checkpoint store, so
# QWEN_PREFIX_STORE_GIB=16 and this 48), a waiver may ALSO name the tier at this value; the lone-user padded round (QWEN_FAST_SOLO_PACKED, serving_octo.solo_packed_admission, admitted
# in a waiver run like the octo shape) and exact resume P0 (the undrop and the drafter-window carry; the carry's audit stays a gate's own). Each has run on CPU alone: the waiver is the
# owner's decision that they run unqualified, exactly as for the levers above, and the audits and the gate-only names of both stay a gate's.
TIER_WAIVABLE_ALSO = {'QWEN_PREFIX_HOST_TIER_GIB': ('48',)}
# tp4/sched-1 (docs/tp4-sched-1.md), the two of its levers that return device time to the decoders and trade none away: the lean admission (QWEN_FAST_ADMISSION_LEAN=1: three admission
# costs off a returning turn's rebind) and the complete spills of the host tier (QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete, only beside the tier, whose default knobs the candidate keeps). The governor knobs
# (QWEN_FAST_LEVERN_ADAPTIVE*, _OWED_CAP_MS, _SHORT_SHARE, _SHORT_WAIT_S) trade decoder speed for first-token time and stay a gate's own, EXCEPT the one switch the ship-6 candidate carries
# (tp4/s6, docs/tp4-next2.md): QWEN_FAST_LEVERN_ADAPTIVE=1 at its default knobs (the three sub-knobs _DECODERS, _SHARE and _TTFT_S, and the sched-1 pacing knobs, stay a gate's own), only under an owner
# traffic waiver that names it, as for the levers above.
NEXT_WAIVABLE_LEVERS = {'QWEN_FAST_SOLO_PACKED': '1', 'QWEN_PREFIX_STICKY_UNDROP': '1', 'QWEN_PREFIX_WINDOW_CARRY': '1',
                        'QWEN_FAST_ADMISSION_LEAN': '1', 'QWEN_PREFIX_HOST_TIER_SPILL_MODE': 'complete', 'QWEN_FAST_LEVERN_ADAPTIVE': '1'}
# tp4/prefill-3 (docs/tp4-prefill3-ksplit.md): the K-split attention of the tail passes. Its output differs from the oneq call's in the last bits (a different association of the softmax: a
# tolerance-class numerics change), so a traffic profile carries it only under the owner's traffic waiver that names it; its audit (judged by a statistic against a bound, not bitwise) is a gate's own.
KSPLIT_SWITCH = 'QWEN_FAST_SDPA_PF_KSPLIT'
KSPLIT_AUDIT = KSPLIT_SWITCH + '_AUDIT'
KSPLIT_WAIVABLE_LEVERS = {KSPLIT_SWITCH: '1'}
NEXT_WAIVABLE_LEVERS.update(KSPLIT_WAIVABLE_LEVERS)
# tp4/s7 (docs/tp4-s7.md): two of the three tp4/sched-2 switches may serve traffic under an owner traffic waiver that names each at this value (the ship-7 candidate): seed first (the seed token leaves before
# the bind) and the memory-ledger trim. The hold reuse (QWEN_FAST_ADMISSION_HOLD_REUSE) and its knobs stay a gate's own until a ballast-boundary run has been read.
SCHED2_WAIVABLE_LEVERS = {'QWEN_FAST_ADMISSION_SEED_FIRST': '1', 'QWEN_FAST_ADMISSION_LEDGER_TRIM': '1'}
NEXT_WAIVABLE_LEVERS.update(SCHED2_WAIVABLE_LEVERS)
TRAFFIC_WAIVER_MARKER = 'OWNER TRAFFIC WAIVER'
# The decision is a sentence that starts with one of these words. PENDING is a REQUEST: the profile boots (gates and G1 provenance boot every
# profile) and logs the word, but no build may bake it as an image default (c2_serving_job.read_bake, build-c2-serving-image.sh) until the commit
# that is built records the owner's APPROVED decision.
TRAFFIC_WAIVER_DECISIONS = ('PENDING', 'APPROVED')


def traffic_waiver_problems(profile):
    """Every way the profile's owner traffic waiver is malformed or misplaced, [] when it has none or it is well formed."""
    waiver = profile.get(TRAFFIC_WAIVER)
    if waiver is None:
        return []
    label = '%s of %s' % (TRAFFIC_WAIVER, profile.get('name', 'the profile'))
    if not isinstance(waiver, dict):
        return ['%s must be an object with %s' % (label, ', '.join(TRAFFIC_WAIVER_FIELDS))]
    problems = []
    if profile.get('gate_only') is True:
        problems.append('%s: a gate-only profile needs no waiver (the gate refusals do not apply to it); only a traffic profile carries one' % label)
    missing = [field for field in TRAFFIC_WAIVER_FIELDS if field not in waiver]
    unknown = sorted(field for field in waiver if field not in TRAFFIC_WAIVER_FIELDS)
    if missing:
        problems.append('%s lacks %s' % (label, ', '.join(missing)))
    if unknown:
        problems.append('%s has unknown field(s) %s' % (label, ', '.join(unknown)))
    if 'id' in waiver and not (isinstance(waiver['id'], str) and TRAFFIC_WAIVER_ID.fullmatch(waiver['id'])):
        problems.append('%s: id must match %s, got %r' % (label, TRAFFIC_WAIVER_ID.pattern, waiver['id']))
    for field in ('waives', 'decision'):
        if field in waiver and not (isinstance(waiver[field], str) and waiver[field].strip()):
            problems.append('%s: %s must be a non-empty sentence' % (label, field))
    decision = waiver.get('decision')
    if isinstance(decision, str) and decision.strip() and decision.split(None, 1)[0].rstrip(':') not in TRAFFIC_WAIVER_DECISIONS:
        problems.append('%s: decision must start with %s, got %r' % (label, ' or '.join(TRAFFIC_WAIVER_DECISIONS), decision[:40]))
    evidence = waiver.get('evidence')
    if 'evidence' in waiver and not (isinstance(evidence, list) and evidence and all(isinstance(item, str) and item.strip() for item in evidence)):
        problems.append('%s: evidence must be a non-empty list of non-empty strings (the card runs the decision rests on)' % label)
    levers = waiver.get('levers')
    if 'levers' in waiver:
        if not (isinstance(levers, dict) and levers):
            problems.append('%s: levers must be a non-empty object {flag: exact value}' % label)
        else:
            env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
            waivable = dict(WAIVABLE_LEVERS, **dict(TIER_WAIVABLE_LEVERS, **NEXT_WAIVABLE_LEVERS))
            for flag in sorted(levers):
                value = levers[flag]
                if flag not in waivable:
                    problems.append('%s: %s is not waivable (only %s; a gate instrument never is)' % (label, flag, ', '.join(sorted(waivable))))
                elif value != waivable[flag] and value not in TIER_WAIVABLE_ALSO.get(flag, ()):
                    problems.append('%s: %s is waivable only at %s, not %r' % (label, flag, ' or '.join((waivable[flag],) + TIER_WAIVABLE_ALSO.get(flag, ())), value))
                elif env.get(flag) != value:
                    problems.append('%s names %s=%s and the profile env has %r: a waiver names exactly the levers the profile sets'
                                    % (label, flag, value, env.get(flag)))
    return problems


def waived_levers(profile):
    """{flag: value} of the profile's owner traffic waiver; {} when it has none or the waiver is malformed (a malformed waiver waives nothing)."""
    waiver = profile.get(TRAFFIC_WAIVER)
    if waiver is None or traffic_waiver_problems(profile):
        return {}
    return dict(waiver['levers'])


def traffic_waiver_pending(profile):
    """Whether the profile carries an owner traffic waiver the owner has not approved (malformed counts as not approved)."""
    waiver = profile.get(TRAFFIC_WAIVER)
    if waiver is None:
        return False
    if traffic_waiver_problems(profile):
        return True
    return waiver['decision'].split(None, 1)[0].rstrip(':') != 'APPROVED'


def waived(profile, flag, env):
    """Whether the profile's owner traffic waiver names `flag` at the value the env sets it to."""
    value = waived_levers(profile).get(flag)
    return value is not None and env.get(flag) == value


def octo_problems(profile, environ=None):
    """Every way the octo-T8 flags are set outside what the contract admits, [] when none (tp4/octo-3, docs/tp4-octo3.md). A profile with none of the names is clean without a read.
    A GATE-ONLY profile may set every name but the waiver's own env (it has no waiver). A TRAFFIC profile sets the four levers (OCTO_LEVERS) only under its owner traffic waiver, each
    named at its exact value (WAIVABLE_LEVERS) and set in the env, together with QWEN_FAST_OCTO_WAIVER equal to the waiver's id, and never the gate-only names (the bundle's audit, the
    audit sampling, the lone-user padded round); a lever set and not named is refused here, a waiver naming a lever the profile does not set (a stale waiver) by traffic_waiver_problems, as for W2.
    A traffic profile that sets QWEN_FAST_OCTO_WAIVER without the shape is refused too (a waiver id with nothing under it)."""
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    if not any(name in env for name in OCTO_NAMES):
        return []
    problems = []
    if profile.get('gate_only') is True:
        if OCTO_WAIVER_ENV in env:
            problems.append('%s is a traffic profile\'s: a gate-only profile admits the shape through its gate marker and needs no waiver id' % OCTO_WAIVER_ENV)
        return problems
    mode = env.get(OCTO_SWITCH, 'off')
    on = mode != 'off' or any(env.get(name, '0') not in ('', '0') for name in OCTO_LEVERS[1:])
    problems += ['%s=%s is a gate instrument: only a gate-only profile sets it%s' % (name, env[name], ' (or a traffic profile whose %s names it)' % TRAFFIC_WAIVER if name in NEXT_WAIVABLE_LEVERS else '')
                 for name in OCTO_GATE_ONLY if env.get(name, '0') not in ('', '0') and not (name in NEXT_WAIVABLE_LEVERS and waived(profile, name, env))]
    solo = env.get(SOLO_PACKED_SWITCH, '0') not in ('', '0') and waived(profile, SOLO_PACKED_SWITCH, env)
    if not on:
        if OCTO_WAIVER_ENV in env and not solo:
            problems.append('%s=%s without the octo shape (%s=live): a waiver id with nothing under it' % (OCTO_WAIVER_ENV, env[OCTO_WAIVER_ENV], OCTO_SWITCH))
        if solo:
            # the lone-user padded round under the owner's waiver without the octo shape: the process takes the waiver's id from the profile (serving_octo.waiver_run admits it)
            waiver = profile.get(TRAFFIC_WAIVER) or {}
            if env.get(OCTO_WAIVER_ENV) != waiver.get('id'):
                problems.append('%s=%r and the profile\'s %s id is %r: the process takes the waiver\'s id from the profile, and only that one' % (
                    OCTO_WAIVER_ENV, env.get(OCTO_WAIVER_ENV), TRAFFIC_WAIVER, waiver.get('id')))
        return problems
    for name in OCTO_LEVERS:
        value = env.get(name)
        if value in (None, '', '0', 'off'):
            continue
        if not waived(profile, name, env):
            problems.append('%s=%s needs a gate-only profile: octo-T8 and its levers have no pinned qualification record (the K64j 0x21 eight-row group per bundle and B > 1 entries of '
                            'it, the batched GDN launch at eight users, the quarter-tile glue movers) - or the owner\'s %s naming it' % (name, value, TRAFFIC_WAIVER))
    minimum = env.get('QWEN_FAST_OCTO_MIN_LIVE')
    if minimum is not None and not waived(profile, 'QWEN_FAST_OCTO_MIN_LIVE', env):
        problems.append('QWEN_FAST_OCTO_MIN_LIVE=%s on a traffic profile needs the owner\'s %s naming it (at 6)' % (minimum, TRAFFIC_WAIVER))
    waiver = profile.get(TRAFFIC_WAIVER) or {}
    if not isinstance(waiver, dict) or env.get(OCTO_WAIVER_ENV) != waiver.get('id'):
        problems.append('%s=%r and the profile\'s %s id is %r: the process takes the waiver\'s id from the profile, and only that one' % (
            OCTO_WAIVER_ENV, env.get(OCTO_WAIVER_ENV), TRAFFIC_WAIVER, waiver.get('id') if isinstance(waiver, dict) else None))
    return problems


def multi_problems(profile):
    """The wave-2 levers (docs/tp4-w2.md, docs/tp4-combined-window.md) on a traffic profile: refused unless the profile's owner traffic waiver
    names them. The multi-user SDPA launch (QWEN_FAST_TP4_SDPA set to anything but off, one G16 flags 0x21 program) is outside the baked 262k
    evidence (packed_any_evidence_tp4_262144.json covers the served 0x23 programs), the F1 conv-gates spread has no pinned qualification record,
    and both audits are gate instruments. A traffic profile that names the launch or F1 needs a pinned qualification record for its capacity,
    which does not exist, or the owner's waiver of it (TRAFFIC_WAIVER); the audits are never waivable. The refusal names what is missing."""
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    if profile.get('gate_only') is True:
        return []
    problems = []
    if env.get(MULTI_SWITCH, '').strip().lower() not in SDPA_OFF_VALUES and not waived(profile, MULTI_SWITCH, env):
        problems.append('%s=%s needs a gate-only profile: the multi-user SDPA launch is outside the 262k evidence and has no pinned multi '
                        'qualification record (or the owner\'s %s naming it)' % (MULTI_SWITCH, env[MULTI_SWITCH], TRAFFIC_WAIVER))
    for key in (MULTI_AUDIT, F1_SWITCH, F1_AUDIT):
        if env.get(key, '0') not in ('', '0') and not waived(profile, key, env):
            problems.append('%s=%s needs a gate-only profile (it has never served traffic and has no pinned qualification record)'
                            % (key, env[key]))
    return problems


def w2_problems(profile, environ=None):
    """Every way the W2 kill switch's names are misused, [] when none (w2_switch, docs/tp4-w2-kill-switch.md): an unknown QWEN_FAST_W2_ name; an override path that is
    not absolute; the drill trigger (QWEN_FAST_W2_OFF_AFTER) outside a gate-only profile, in the process environment outside a gate profile, not a positive whole number
    of packed rounds, or without W2 in the profile and a path of its own to write (an arm that kills what it never ran, or writes the production file)."""
    environ = {} if environ is None else environ
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    problems = ['%s is not a W2 kill switch setting (the names are %s)' % (name, ', '.join(W2_NAMES))
                for name in sorted(env) if name.startswith(W2_PREFIX) and name not in W2_NAMES]
    for name in W2_NAMES:
        path = env.get(name)
        if name == 'QWEN_FAST_W2_OFF_PATH' and path is not None and not (path.startswith('/') and not path.endswith('/')):
            problems.append('%s must be an absolute file path, got %r' % (name, path))
    problems += ['%s is a gate instrument: only a gate-only profile sets it' % name for name in W2_GATE_ONLY if name in env and not gate_profile(profile)]
    if not gate_profile(profile):
        problems += ['%s is set outside a gate profile: it exists for a gate and must never reach traffic' % name for name in W2_GATE_ONLY if name in environ]
    after = env.get('QWEN_FAST_W2_OFF_AFTER')
    if after is not None:
        if not (after.isascii() and after.isdigit() and after == str(int(after)) and int(after) > 0):
            problems.append('QWEN_FAST_W2_OFF_AFTER must be a positive whole number of packed rounds, got %r' % (after,))
        if env.get(MULTI_SWITCH, '').strip() != 'multi' and env.get(F1_SWITCH, '0') != '1':
            problems.append('QWEN_FAST_W2_OFF_AFTER needs W2 in the profile (%s=multi or %s=1): there is nothing to kill' % (MULTI_SWITCH, F1_SWITCH))
        path = env.get('QWEN_FAST_W2_OFF_PATH')
        if path is None or path.startswith('/models/'):
            problems.append('QWEN_FAST_W2_OFF_AFTER writes the flag file itself: QWEN_FAST_W2_OFF_PATH must name a scratch path in the container '
                            '(not the hub mount\'s %s), got %r' % ('/models/.qwen-c2/w2.off', path))
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
        if profile.get('gate_only') is not True:
            for name in ('QWEN_FAST_LEVERN_KV_READ', 'QWEN_FAST_LEVERN_KV_CROSS_STEPS'):
                if name in env:
                    problems.append('Lever N: %s is a gate instrument (the audit digest\'s read) and needs a gate-only profile' % name)
            for name in LEVERN_ADAPTIVE_FLAGS:
                if name in env:
                    problems.append('Lever N: %s is a gate instrument (the adaptive governor, docs/lever-n-adaptive-governor.md, or the production-shape pacing, docs/tp4-sched-1.md) and needs a gate-only profile' % name)
        return problems
    if profile.get('gate_only') is not True:
        # The traffic arm (stage 1 of the short-window plan): the master switch alone, with its sibling flags and the merged route, may serve traffic
        # (the kill switch levern.off stops it with no restart); the gate instruments never may (they are the gates' own, and a fault is a negative control).
        for name in (LEVERN_AUDIT, 'QWEN_FAST_LEVERN_FAULT', 'QWEN_FAST_LEVERN_KV_READ', 'QWEN_FAST_LEVERN_KV_CROSS_STEPS') + LEVERN_ADAPTIVE_FLAGS:
            if name in NEXT_WAIVABLE_LEVERS and waived(profile, name, env):
                continue                                # the adaptive governor's switch, named by the owner traffic waiver at its value (NEXT_WAIVABLE_LEVERS)
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


# Exact resume P0 (docs/tp4-exact-resume.md 5.2; qwen_prefix_registry): the undrop, the drafter-window carry and its audit. Gate-only until a
# qualification record exists, only beside sticky sessions with prefix reuse, the carry only with the undrop and the audit only with the
# carry; an inherited value under a profile that does not name one is dropped (apply_environment).
EXACT_RESUME_NAMES = ('QWEN_PREFIX_STICKY_UNDROP', 'QWEN_PREFIX_WINDOW_CARRY', 'QWEN_PREFIX_WINDOW_CARRY_AUDIT')


def exact_resume_problems(profile):
    """[problem] when a profile names the exact-resume switches wrongly (nothing is checked for a profile that names none)."""
    env = profile.get('env') or {}
    named = [name for name in EXACT_RESUME_NAMES if name in env]
    if not named:
        return []
    problems = []
    values = dict((name, str(env[name])) for name in named)
    problems += ['%s=%r: only 0 or 1' % (name, value) for name, value in sorted(values.items()) if value not in ('0', '1')]
    on = set(name for name, value in values.items() if value == '1')
    # a traffic profile may carry the undrop and the carry only under the owner's traffic waiver naming each (NEXT_WAIVABLE_LEVERS); the carry's audit is a gate's own
    unqualified = set(name for name in on if profile.get('gate_only') is not True and not (name in NEXT_WAIVABLE_LEVERS and waived(profile, name, {key: str(value) for key, value in env.items()})))
    if unqualified:
        problems.append('%s are gate-only until a qualification record exists (or the owner\'s %s names the undrop and the carry): %s is not gate_only' % (', '.join(sorted(unqualified)), TRAFFIC_WAIVER, profile.get('name')))
    if on and not (prefix_reuse(profile) and sticky_sessions(profile)):
        problems.append('%s need %s=1 and %s=1 (the scheduler graft serves the fast path\'s sticky hit only)'
                        % (', '.join(sorted(on)), PREFIX_SWITCH, STICKY_SWITCH))
    if EXACT_RESUME_NAMES[1] in on and EXACT_RESUME_NAMES[0] not in on:
        problems.append('%s=1 without %s=1' % (EXACT_RESUME_NAMES[1], EXACT_RESUME_NAMES[0]))
    if EXACT_RESUME_NAMES[2] in on and EXACT_RESUME_NAMES[1] not in on:
        problems.append('%s=1 without %s=1' % (EXACT_RESUME_NAMES[2], EXACT_RESUME_NAMES[1]))
    return problems


def ksplit_problems(profile):
    """[problem] when a profile names the K-split switches wrongly (nothing is checked for a profile that names none): a value that is not '0'/'1', the audit without the switch, the switch
    without the oneq call it splits (QWEN_FAST_SDPA_PF_ONEQ=1), the audit on a traffic profile (a gate's instrument), and - the numerics waiver - the switch on a TRAFFIC
    profile whose owner traffic waiver does not name it (a gate-only profile boots it)."""
    env = {key: str(value) for key, value in (profile.get('env') or {}).items()}
    named = [name for name in (KSPLIT_SWITCH, KSPLIT_AUDIT) if name in env]
    if not named:
        return []
    problems = ['%s=%r: only 0 or 1' % (name, env[name]) for name in named if env[name] not in ('0', '1')]
    on, audit = env.get(KSPLIT_SWITCH) == '1', env.get(KSPLIT_AUDIT) == '1'
    if audit and not on:
        problems.append('%s=1 without %s=1' % (KSPLIT_AUDIT, KSPLIT_SWITCH))
    if on and env.get('QWEN_FAST_SDPA_PF_ONEQ') != '1':
        problems.append('%s=1 splits the oneq call\'s K stream: it needs QWEN_FAST_SDPA_PF_ONEQ=1 (and so the chain, QWEN_FAST_SDPA_PF=1, which the image environment sets)' % KSPLIT_SWITCH)
    if profile.get('gate_only') is not True:
        if audit:
            problems.append('%s=1 is a gate instrument (it doubles the split calls): %s is not gate_only' % (KSPLIT_AUDIT, profile.get('name')))
        if on and not waived(profile, KSPLIT_SWITCH, env):
            problems.append('%s=1 is a numerics change (the softmax is merged from partial results: the output differs from the oneq call in the last bits) with no pinned qualification record: '
                            'gate-only, or the owner\'s %s names it (the numerics waiver is absent): %s is not gate_only' % (KSPLIT_SWITCH, TRAFFIC_WAIVER, profile.get('name')))
    return problems


def upload_problems(profile):
    """[problem] when a profile names the engine-start upload levers wrongly: a value that is not '0'/'1', a count out of range, an audit or a knob without its
    lever, an audit flag on a profile that is not a gate's."""
    env = profile.get('env') or {}
    if not any(name.startswith(('QWEN_FAST_DEVICE_ZEROS', 'QWEN_FAST_LAZY_SHARD_W')) for name in env):
        return []                                       # every profile without these names: nothing imported, nothing checked
    import qwen_device_zeros
    import qwen_lazy_shard

    problems = qwen_device_zeros.flag_problems(env) + qwen_lazy_shard.flag_problems(env)
    if profile.get('gate_only') is not True:
        problems += ['%s is a gate profile\'s own (it reads pages back at engine start): %s is not gate_only' % (name, profile.get('name'))
                     for name in UPLOAD_GATE_ONLY if env.get(name) == '1']
    for name in env:
        if name.startswith(('QWEN_FAST_DEVICE_ZEROS', 'QWEN_FAST_LAZY_SHARD_W')) and name not in UPLOAD_NAMES:
            problems.append('%s is not an upload lever name (%s)' % (name, ', '.join(UPLOAD_NAMES)))
    return problems


def install_upload_levers(environ=None, on_import=None):
    """In every process of a profile that sets QWEN_FAST_DEVICE_ZEROS=1 and/or QWEN_FAST_LAZY_SHARD_W=1: arm the post-import hooks that install the KV-pool twin
    (on the model module) and the lazy shard_w (on tp_common). Off by default: with neither switch nothing is imported or hooked. -> the names armed."""
    environ = os.environ if environ is None else environ
    if all(environ.get(name) in (None, '0') for name in ('QWEN_FAST_DEVICE_ZEROS', 'QWEN_FAST_LAZY_SHARD_W')):
        return []
    if on_import is None:
        def on_import(name, callback):
            loaded = sys.modules.get(name)
            if loaded is not None:
                # Already imported: a post-import hook would never fire.
                callback(loaded)
                return
            sys.meta_path.insert(0, PostImportHook(name, callback))
    import qwen_device_zeros
    import qwen_lazy_shard

    armed = []
    if qwen_device_zeros.arm(environ, on_import, log=log_pindiag):
        armed.append(qwen_device_zeros.FLAG)
    if qwen_lazy_shard.arm(environ, on_import, log=log_pindiag):
        armed.append(qwen_lazy_shard.FLAG)
    return armed


def log_pindiag(template, *values):
    """The levers' one-line diagnostics (brace-formatted, loguru when the process has it) - the form dflash_device.pindiag has."""
    try:
        from loguru import logger
    except ImportError:
        print(template.format(*values), flush=True)
        return
    logger.info(template, *values)


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
            # Tool-call argument whitespace, counts only (docs/tp4-exact-resume.md 5.3).
            counter = getattr(qwen_prefix_metrics, 'install_tool_arg_counter', None)
            if counter is not None:
                counter(on_import)
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


def sampled_terms(params):
    """QWEN_FAST_SAMPLED=1: the request's sampling terms as the sampled verifier applies them, written back into `params`; ContractError for what it cannot.

    - temperature 0 stays greedy and is coerced exactly as the greedy contract coerces it (temperature 0, top_p 1, top_k 0, no seed);
    - otherwise temperature, top_p and top_k are the request's; a field that is None takes the model's default (SAMPLED_MODEL_DEFAULTS). By the time a request
      reaches this hook vLLM has already filled the ones the client omitted (from the model's generation config, which sampled_launch_problems keeps on), so
      this default only reaches an offline-engine caller that left a field None;
    - top_k <= 0 (disabled) or above SAMPLED_K_MAX is clamped to SAMPLED_K_MAX: the one approximation (docs/tp4-sampled-verify.md section 5);
    - the request seed is the client's or, if it sent none, a random 64-bit one; it travels in extra_args[SAMPLED_SEED_KEY] and params.seed is cleared, because
      the pinned samplers refuse a block while a request seed is active in the plugin's seed manager (force_argmax.sample_rows, verify_trace_t1);
    - min_p and the three penalties are refused, greedy or not: the verifier applies neither, and dropping them silently (what the greedy contract does) would
      return text the client did not ask for."""
    if (getattr(params, 'presence_penalty', 0.0) or 0.0) != 0 or (getattr(params, 'frequency_penalty', 0.0) or 0.0) != 0 \
            or (1.0 if getattr(params, 'repetition_penalty', None) is None else params.repetition_penalty) != 1.0:
        raise ContractError('presence_penalty, frequency_penalty and repetition_penalty are not supported on this model')
    if (getattr(params, 'min_p', 0.0) or 0.0) != 0:
        raise ContractError('min_p is not supported on this model')
    defaults = SAMPLED_MODEL_DEFAULTS
    temperature = defaults['temperature'] if params.temperature is None else params.temperature
    top_p = defaults['top_p'] if getattr(params, 'top_p', None) is None else params.top_p
    top_k = defaults['top_k'] if getattr(params, 'top_k', None) is None else params.top_k
    if type(temperature) not in (int, float) or not 0 <= temperature < float('inf'):
        raise ContractError('temperature must be a finite number >= 0, got %r' % (temperature,))
    if temperature == 0:
        params.temperature = 0.0
        params.top_p = 1.0
        params.top_k = 0
        params.min_p = 0.0
        params.presence_penalty = 0.0
        params.frequency_penalty = 0.0
        params.repetition_penalty = 1.0
        params.seed = None
        return params
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise ContractError('top_p must be in (0, 1], got %r' % (top_p,))
    if type(top_k) is not int:
        raise ContractError('top_k must be an integer, got %r' % (top_k,))
    seed = getattr(params, 'seed', None)
    if seed is None:
        seed = secrets.randbits(64)
    elif type(seed) is not int:
        raise ContractError('seed must be an integer, got %r' % (seed,))
    params.temperature = float(temperature)
    params.top_p = float(top_p)
    params.top_k = SAMPLED_K_MAX if top_k <= 0 else min(top_k, SAMPLED_K_MAX)
    params.min_p = 0.0
    params.presence_penalty = 0.0
    params.frequency_penalty = 0.0
    params.repetition_penalty = 1.0
    params.seed = None
    extra = dict(getattr(params, 'extra_args', None) or {})
    extra[SAMPLED_SEED_KEY] = seed & ((1 << 64) - 1)
    params.extra_args = extra
    return params


def enforce_request(params, *, prompt_tokens, max_model_len, budget, eos_ids, max_prompt_tokens=None,
                    min_answer_tokens=None, default_max_tokens=None, drafter_headroom_tokens=0, kv_pool_blocks=None, sampled=None):
    """Refuse what the fast path cannot serve; coerce the rest to its greedy contract.

    sampled (None: QWEN_FAST_SAMPLED, off in every shipped profile) lifts the coercion to greedy for a request that asks for sampling (sampled_terms); off,
    which is every caller as it always was, nothing below it changes.

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
    if sampled_enabled() if sampled is None else sampled:
        sampled_terms(params)
    else:
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


LEGACY_OUTPUT_BUDGET = 16384        # the ceiling of every profile until NP9 (docs/tp4-np9-answer32k.md); a budget above it is asked more of
OUTPUT_BUDGET_STEP = 256            # serving_fast_policy.PACKED_FAMILY_TOKENS
MAXIMUM_OUTPUT_BUDGET = 32768       # serving_fast_policy.MAXIMUM_OUTPUT_BUDGET (a test holds both equal; not imported: that module reads QWEN_FAST_OUTPUT_BUDGET when it is imported)
OUTPUT_BUDGET_ENV = 'QWEN_FAST_OUTPUT_BUDGET'
MINIMUM_CONTEXT = 4096              # serving_fast_policy.MINIMUM_MODEL_LEN = this + the budget


def output_budget_problems(profile, budget):
    """[problem] with serving `budget` as the output ceiling of `profile`, [] when none. A budget at or under LEGACY_OUTPUT_BUDGET asks nothing new of any profile (every shipped profile holds one).

    Above it the budget is only free on the shape the c2-packed-tp4 serving profiles have: QWEN_FAST_ANY_REQUEST=1 (each request's own max_tokens is its budget, and the engines capture the
    sequential widths 1, 2 and 4) beside QWEN_FAST_EXTENT_REPLAY=1 (the pool refuses a replay width, serving_buffer_pool, so no verifier capture is allocated per 256-position mask family up
    to the budget); anywhere else a bigger budget is device memory per admitted request (verifier_engine.capture_bucket_rows). And the window must hold the budget:
    serving_fast_policy.MINIMUM_MODEL_LEN."""
    if type(budget) is not int or budget % OUTPUT_BUDGET_STEP or not OUTPUT_BUDGET_STEP <= budget <= MAXIMUM_OUTPUT_BUDGET:
        return ['the output budget must be a multiple of %d from %d to %d, got %r' % (OUTPUT_BUDGET_STEP, OUTPUT_BUDGET_STEP, MAXIMUM_OUTPUT_BUDGET, budget)]
    if budget <= LEGACY_OUTPUT_BUDGET:
        return []
    env = profile.get('env', {})
    problems = []
    if env.get('QWEN_FAST_ANY_REQUEST') != '1' or env.get('QWEN_FAST_EXTENT_REPLAY') != '1':
        problems.append('an output budget of %d (above %d) needs QWEN_FAST_ANY_REQUEST=1 and QWEN_FAST_EXTENT_REPLAY=1: on any other shape every request allocates verifier captures per 256-position '
                        'mask family up to the budget' % (budget, LEGACY_OUTPUT_BUDGET))
    if profile.get('request_contract', True) is False:
        problems.append('an output budget above %d needs the request contract, which clamps max_tokens to it' % LEGACY_OUTPUT_BUDGET)
    window = profile.get('engine', {}).get('max-model-len')
    if type(window) is int and window < MINIMUM_CONTEXT + budget:
        problems.append('an output budget of %d needs a window of at least %d tokens, the profile has %d' % (budget, MINIMUM_CONTEXT + budget, window))
    return problems


def checkpoint_budget_problems(profile, limits, budget):
    """[problem] with laying a checkpoint's `budget` over `limits` (request_limits(profile)): output_budget_problems, and the rule that it only RAISES the profile's ceiling and leaves the longest
    admitted prompt where it is (a profile with no min_answer_tokens keeps room for the whole budget, so a raise would shorten every prompt: not laid over that profile)."""
    problems = output_budget_problems(profile, budget)
    if problems:
        return problems
    if budget < limits['budget']:
        return ['it would lower the profile\'s output budget from %d to %d' % (limits['budget'], budget)]
    window = profile.get('engine', {}).get('max-model-len')
    if type(window) is int:
        headroom = limits.get('drafter_headroom_tokens', 0)
        before = prompt_room(window, limits['budget'], limits['max_prompt_tokens'], limits['min_answer_tokens'], headroom)
        after = prompt_room(window, budget, limits['max_prompt_tokens'], limits['min_answer_tokens'], headroom)
        if after != before:
            problems.append('it would lower the longest admitted prompt from %d to %d tokens (the profile keeps room for the whole budget: name min_answer_tokens)' % (before, after))
    return problems


def request_limits(profile):
    """The request contract's numbers from a profile, checked once at boot: the output budget,
    and the optional max_prompt_tokens, min_answer_tokens and default_max_tokens."""
    budget = int(profile['env']['QWEN_FAST_OUTPUT_BUDGET'])
    limits = dict(budget=budget, max_prompt_tokens=profile.get('max_prompt_tokens'),
                  min_answer_tokens=profile.get('min_answer_tokens'),
                  default_max_tokens=profile.get('default_max_tokens'))
    if budget > LEGACY_OUTPUT_BUDGET:
        problems = output_budget_problems(profile, budget)
        if problems:
            raise ValueError('; '.join(problems))
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
    if sampled_enabled():
        log('request contract: sampling honoured (%s=1; temperature 0 stays greedy, top_k clamped to %d)', SAMPLED_SWITCH, SAMPLED_K_MAX)
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


def arm_toolcall_whitespace(profile, environ):
    """Keep the edge whitespace of tool-call arguments in the vLLM API server (c2_toolcall_whitespace): vLLM 0.25.1's
    qwen3 parser strip()s every argument. ON for every profile unless QWEN_C2_TOOLCALL_WS=0 or the profile says
    "toolcall_whitespace": false. The module is imported here, at boot, so an image without it fails the API server's
    boot instead of serving the bug; install refuses a parser that is not vLLM 0.25.1's."""
    import c2_toolcall_whitespace

    return c2_toolcall_whitespace.arm(profile, environ)


def arm_idle_scratch(profile_name, environ):
    """NP1 (QWEN_FAST_SOLO_PACKED=1, gate only; idle_scratch.py, docs/tp4-np1-fastlane.md): when the model entry is imported, wrap Qwen36ForCausalLM.allocate_kv_cache so the worker's KV
    caches hold one block more than vLLM's pool - the scratch page the lone-user padded round parks its third idle segment's K/V in. The module is imported here, at boot, so an image
    without it fails the boot instead of serving the flag mounted and never executed; the wrapper raises at the model import if the class or the method is not there."""
    import idle_scratch

    count = idle_scratch.wanted(environ)
    if count:
        sys.meta_path.insert(0, PostImportHook(idle_scratch.MODEL_ENTRY, idle_scratch.install))
        log('profile %s: NP1 armed (%s=1): %s.%s.%s will allocate %d KV block(s) beyond the pool vLLM sizes, for the idle segments of a lone-user padded round (gate only)',
            profile_name, idle_scratch.SOLO_PACKED_FLAG, idle_scratch.MODEL_ENTRY, idle_scratch.MODEL_CLASS, idle_scratch.ALLOCATOR, count)
    return count


def boot(environ=None, orig_argv=None):
    environ = os.environ if environ is None else environ
    if environ.get('QWEN_C2_SERVING') != '1':
        return None
    fix_sys_path()
    profile = load_profile(environ.get('QWEN_C2_PROFILES', PROFILES))
    problems = gate_problems(profile, environ) + waiver_problems(profile, environ) + traffic_waiver_problems(profile)
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
    problems = sched_problems(profile)
    if problems:
        raise ValueError('profile %s misuses the tp4/sched-1 switches: %s' % (profile['name'], '; '.join(problems)))
    problems = sched2_problems(profile)
    if problems:
        raise ValueError('profile %s misuses the tp4/sched-2 switches: %s' % (profile['name'], '; '.join(problems)))
    problems = octo_problems(profile, environ)
    if problems:
        raise ValueError('profile %s cannot serve octo-T8: %s' % (profile['name'], '; '.join(problems)))
    problems = w2_problems(profile, environ)
    if problems:
        raise ValueError('profile %s misuses the W2 kill switch: %s' % (profile['name'], '; '.join(problems)))
    problems = upload_problems(profile)
    if problems:
        raise ValueError('profile %s misuses the engine-start upload levers: %s' % (profile['name'], '; '.join(problems)))
    problems = exact_resume_problems(profile)
    if problems:
        raise ValueError('profile %s misuses the exact-resume switches: %s' % (profile['name'], '; '.join(problems)))
    problems = ksplit_problems(profile)
    if problems:
        raise ValueError('profile %s misuses the K-split switches: %s' % (profile['name'], '; '.join(problems)))
    problems = drafter_problems(profile)
    if problems:
        raise ValueError('profile %s cannot serve its drafter checkpoint: %s' % (profile['name'], '; '.join(problems)))
    problems = parked_problems(profile, environ)
    if problems:
        raise ValueError('profile %s cannot serve parked engines: %s' % (profile['name'], '; '.join(problems)))
    problems = sampled_problems(profile)
    if problems:
        raise ValueError('profile %s misuses the sampled-verify switch: %s' % (profile['name'], '; '.join(problems)))
    problems = mesh_problems(profile)
    if problems:
        raise ValueError('profile %s cannot open its mesh: %s' % (profile['name'], '; '.join(problems)))
    levers = waived_levers(profile)
    if levers:
        waiver = profile[TRAFFIC_WAIVER]
        log('profile %s: %s %s: %s serve traffic with no pinned qualification record (%s) [decision: %s]', profile['name'], TRAFFIC_WAIVER_MARKER,
            waiver['id'], ' '.join('%s=%s' % (flag, levers[flag]) for flag in sorted(levers)), waiver['waives'], waiver['decision'])
    apply_environment(profile, environ)
    armed = install_upload_levers(environ)
    if armed:
        log('profile %s: engine-start upload levers armed: %s (default off; each logs engaged or refused at its first use)', profile['name'], ' '.join(armed))
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
    checkpoint_snapshot = configure_checkpoint(profile, environ, api_server)
    if api_server:
        limits = checkpoint_limits(limits, environ, profile)
        budget = limits['budget']
    if prefix_reuse(profile):
        install_prefix_metrics(api_server)
        sys.meta_path.insert(0, PostImportHook(MODEL_ENTRY, log_model_tree))
    if environ.get('QWEN_FAST_SOLO_PACKED', '0') != '0':
        arm_idle_scratch(profile['name'], environ)
    if api_server:
        check_platform_model(sys.argv[1:], environ)
        if sampled_on(profile):
            refusals = sampled_launch_problems(sys.argv[1:])
            if refusals:
                raise ValueError('profile %s cannot serve sampled verification with this launch: %s' % (profile['name'], '; '.join(refusals)))
        snapshot = checkpoint_snapshot or resolve_snapshot(profile)
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
        arm_toolcall_whitespace(profile, environ)
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
