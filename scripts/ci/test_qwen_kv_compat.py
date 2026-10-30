"""Which env names change what the NVMe tier stores (docs/tp4-kv-nvme.md, "The layout fingerprint").

A stored KV block or GDN checkpoint outlives a deploy, so the layout fingerprint must change exactly when the bytes' MEANING changes: qwen_kv_nvme.KV_EPOCH (by hand), the model,
the drafter, the KV layout, and the env switches in qwen_kv_nvme.FINGERPRINTED_ENV. This file is the guard on that last list. It classifies EVERY env name the serving image
sets or a profile can set - the names of qwen_c2_profiles.json and the ENV defaults of docker/qwen-c2-serving.Dockerfile - and it FAILS when a name is not classified, so a new
profile switch cannot reach a card without somebody deciding which class it is:

  kv       changes the stored bytes by design -> add it to FINGERPRINTED_ENV (this table's 'kv' class must equal that tuple).
  exact    touches the prefill path (or what it stores) and is qualified bit-exact against the path it replaces -> not fingerprinted; if it ever stops being exact, bump KV_EPOCH.
  others   decode, drafter, scheduling, memory, logging, audits, platform plumbing: nothing they compute reaches the tiers.

The classes were assigned from the readers of each name (the module docstrings that state their exactness claims and the paths they run on), not from the name alone. Run alone:
python3 -B -m unittest test_qwen_kv_compat
"""

import json
import os
import re
import unittest
from pathlib import Path

import qwen_kv_nvme as kvnvme

HERE = Path(__file__).resolve().parent
PROFILES = HERE / 'qwen_c2_profiles.json'
DOCKERFILE = HERE.parent.parent / 'docker' / 'qwen-c2-serving.Dockerfile'
ENV_INSTRUCTION = re.compile(r'^ENV ((?:.*\\\n)*.*)$', re.M)

CLASSES = {
    # kv: CHANGE the stored bytes by design: a different computation (a dtype, a summation order not proven exact) or a different stored form. In
    #    qwen_kv_nvme.FINGERPRINTED_ENV, so a value change is a fresh tier.
    'kv': '''
        QWEN_GDN_PREFILL_MMRS QWEN_PREFIX_CKPT_PRECONVERTED QWEN_SDPA_BF8
    ''',
    # covered: already a component of the fingerprint on its own (gdn_bf16).
    'covered': '''
        QWEN35_GDN_STATE_BF16
    ''',
    # drafter: names the drafter checkpoint: the fingerprint's 'drafter' component reads it (through the pinned table, drafter_checkpoint).
    'drafter': '''
        QWEN_FAST_DRAFTER_CHECKPOINT
    ''',
    # exact: run on the prefill path or in the stored bytes' neighbourhood and are qualified BIT-EXACT against the path they replace (audit twins, byte-for-byte; a
    #    route to the same bytes). Not fingerprinted: a deploy that flips one keeps the tier. If one of them stops being exact it is a numerics
    #    change, and KV_EPOCH must be bumped.
    'exact': '''
        QWEN_FAST_CCL_OPTIONS QWEN_FAST_CCL_TOPOLOGY QWEN_FAST_GDN_PREFILL_CONV QWEN_FAST_GRID13 QWEN_FAST_PREFILL_HANDOFF QWEN_FAST_SDPA_PF
        QWEN_FAST_SDPA_PF_FLAGS QWEN_FAST_SDPA_PF_ONEQ QWEN_FAST_TAP_WINDOW_ALIGNED QWEN_GDN_CONV_GATES QWEN_GDN_NORM_GATE QWEN_GDN_PACKED_QKV
        QWEN_GDN_PROJ_DIRECT QWEN_PREFIX_WINDOW_CARRY QWEN_PROJECTION_LINKS
    ''',
    # tier: the prefix reuse, host tier and store settings: where state is kept and how much, not what its bytes mean.
    'tier': '''
        QWEN_PREFIX_DIGESTS QWEN_PREFIX_HOST_TIER_GIB QWEN_PREFIX_HOST_TIER_OFF_PATH QWEN_PREFIX_HOST_TIER_SPILL_MODE QWEN_PREFIX_HOST_TIER_VERIFY
        QWEN_PREFIX_REUSE QWEN_PREFIX_STICKY_UNDROP QWEN_PREFIX_STORE_GIB QWEN_PREFIX_TIER_RESTORE_SLICE QWEN_PREFIX_TIER_RESTORE_SLICE_MS
    ''',
    # audit: gate-only comparisons and sampling cadences: they run a twin beside the served path and never change what is served.
    'audit': '''
        QWEN_FAST_AUDIT_EVERY QWEN_FAST_AUDIT_FIRST QWEN_FAST_BATCHED_READS_AUDIT QWEN_FAST_CCL_OPTIONS_AUDIT QWEN_FAST_DEVICE_ZEROS_AUDIT
        QWEN_FAST_DRAFT_GATEUP1_AUDIT QWEN_FAST_DRAFT_HEAD64_AUDIT QWEN_FAST_DRAFT_MM_GRID_AUDIT QWEN_FAST_DRAFT_PERMUTE_AUDIT
        QWEN_FAST_DRAFT_QKV1_AUDIT QWEN_FAST_DRAFT_REDUCE_AUDIT QWEN_FAST_DRAFT_SINGLES_AUDIT QWEN_FAST_DRAFT_TAIL_AUDIT QWEN_FAST_FUSED_COMMIT_AUDIT
        QWEN_FAST_GDN_PREFILL_CONV_AUDIT QWEN_FAST_GDN_SEQ_BLOCK_AUDIT QWEN_FAST_GRID13_AUDIT QWEN_FAST_KV_PAGE_WRITER_AUDIT
        QWEN_FAST_LAZY_SHARD_W_AUDIT QWEN_FAST_LEVERN_AUDIT QWEN_FAST_MLP_CFG_AUDIT QWEN_FAST_MLP_GATEUP_AUDIT QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT
        QWEN_FAST_PACKED_AUDIT QWEN_FAST_PARKED_AUDIT QWEN_FAST_PREFILL_HANDOFF_AUDIT QWEN_FAST_PRESTAGE_DIFF_AUDIT QWEN_FAST_REQUEST_SHARD_AUDIT
        QWEN_FAST_SDPA_PF_ONEQ_AUDIT QWEN_FAST_TAP_WINDOW_ALIGNED_AUDIT QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT QWEN_FAST_TP4_DRAFT_CONV_AUDIT
        QWEN_FAST_TP4_DRAFT_HEADS_AUDIT QWEN_FAST_TP4_GDN_WC_AUDIT QWEN_FAST_TP4_ROUND_HOST_AUDIT QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT
        QWEN_FAST_TP4_SDPA_AUDIT QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT
        QWEN_FAST_TP4_VGLUE_AUDIT QWEN_FAST_VERIFY_T1_AUDIT QWEN_FAST_VERIFY_T2_AUDIT QWEN_FAST_WRITE_PACKED_LEAN_AUDIT
        QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT QWEN_PREFIX_HOST_TIER_AUDIT QWEN_PREFIX_WINDOW_CARRY_AUDIT
    ''',
    # draft: the drafter and the speculative round (proposal, draft kernels, draft KV, the drafter's own precision): nothing it computes is stored (the stored
    #    window rows are the TARGET's taps).
    'draft': '''
        QWEN_DRAFT_KV_SLIDE_EXPERIMENT QWEN_DSPARK_REQUEST_CONTEXT QWEN_FAST_DRAFTER_BF16 QWEN_FAST_DRAFT_BF8 QWEN_FAST_DRAFT_GATEUP1
        QWEN_FAST_DRAFT_HEAD64 QWEN_FAST_DRAFT_MM_GRID QWEN_FAST_DRAFT_PERMUTE QWEN_FAST_DRAFT_QKV1 QWEN_FAST_DRAFT_REDUCE QWEN_FAST_DRAFT_TAIL
        QWEN_FAST_EARLY_DRAFT QWEN_FAST_LOOKUP_DRAFT QWEN_FAST_OCTO_DRAFT QWEN_FAST_PACKED_PROPOSAL QWEN_FAST_PAIR_MASK_REFRESH
        QWEN_FAST_PAIR_ROW_EXACT QWEN_FAST_PIPELINED_PROPOSALS QWEN_FAST_QUAD_DRAFT QWEN_FAST_QUAD_DRAFT_BLOCKS QWEN_FAST_TP4_DRAFT_CONV
        QWEN_FAST_TP4_DRAFT_HEADS QWEN_FAST_TP4_DRAFT_WIDE QWEN_FAST_TP4_WINDOW_VALIDATE QWEN_FAST_TP_KV_SLIDE QWEN_FROZEN_COMBINED_RUNTIME
        QWEN_GDN_SHARED_QK_EXPERIMENT
    ''',
    # decode: the decode / verify round and its collectives, commits and kernels: decode-generated state is never published to the tiers.
    'decode': '''
        QWEN_ATTN_PREP QWEN_FAST_BATCHED_READS QWEN_FAST_C1_EXACT QWEN_FAST_FAST_COMMIT QWEN_FAST_FUSED_COMMIT QWEN_FAST_FUSED_COMMIT_INPLACE
        QWEN_FAST_FUSED_COMMIT_LIVE_BANKS QWEN_FAST_GDN_AFTER_PAIRS QWEN_FAST_GDN_PAIR_SLICE QWEN_FAST_GDN_SEQ_BLOCK QWEN_FAST_GDN_SPLIT_V
        QWEN_FAST_GDN_USER_BATCH QWEN_FAST_KV_PAGE_WRITER QWEN_FAST_M3_BLOCKS QWEN_FAST_MLP_CFG QWEN_FAST_MLP_GATEUP QWEN_FAST_NATIVE_ATTN
        QWEN_FAST_PACKED_SAMPLER_IN_TRACE QWEN_FAST_PACKED_STEP QWEN_FAST_PADDED_BLOCK QWEN_FAST_PIPELINED_COMMITS QWEN_FAST_PIPELINED_PUBLISH
        QWEN_FAST_PRESTAGE QWEN_FAST_PRESTAGE_DIFF QWEN_FAST_REQUEST_SHARD_ARGMAX QWEN_FAST_ROUND_B1 QWEN_FAST_ROUND_FENCES QWEN_FAST_SDPA_MODES
        QWEN_FAST_SHARED_CCL QWEN_FAST_SINGLE_GATEUP QWEN_FAST_TP4_ATTN_FOLD QWEN_FAST_TP4_COMMIT_LANES QWEN_FAST_TP4_CONV_GATES_SPREAD
        QWEN_FAST_TP4_ENTRY_DIET QWEN_FAST_TP4_GDN_BLOCK_CONV QWEN_FAST_TP4_GDN_GLUE QWEN_FAST_TP4_GDN_WC QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS
        QWEN_FAST_TP4_ROUND_HOST_KEYED QWEN_FAST_TP4_ROUND_HOST_LEAN QWEN_FAST_TP4_ROUND_HOST_READ QWEN_FAST_TP4_ROUND_HOST_SELECT
        QWEN_FAST_TP4_RS_UNIT_MAJOR QWEN_FAST_TP4_SDPA QWEN_FAST_TP4_SHARD_ARGMAX QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2 QWEN_FAST_TP4_SHARD_VALUES
        QWEN_FAST_TP4_TRACED_PUBLISH QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE QWEN_FAST_TRACED_PUBLISH QWEN_FAST_VERIFY_T1 QWEN_FAST_VERIFY_T2
        QWEN_FAST_WRITE_PACKED_LEAN QWEN_GDN_DIRECT_WINDOW QWEN_GDN_FUSED_DECODE QWEN_GDN_FUSED_INPLACE QWEN_GDN_GATE_EXP_ABBA
        QWEN_GDN_GROUPED_GATHER_ABBA QWEN_MLP_BLOCK_STREAM_EXPERIMENT QWEN_SDPA_TREE_SCRATCH_ROUNDS
    ''',
    # sched: scheduling, admission, Lever N policy, lanes, budgets, deadlines, session caps and context limits.
    'sched': '''
        QWEN36_MAX_TOKENS_ALL_USERS QWEN_FAST_ADMISSION_DIAG_TRIM QWEN_FAST_ADMISSION_LEAN QWEN_FAST_ANY_REQUEST QWEN_FAST_BUDGET_CAP
        QWEN_FAST_DECODE_STEPS_PER_ADMISSION QWEN_FAST_FOUR_AS_TWO QWEN_FAST_LANE QWEN_FAST_LEVERN_ADAPTIVE QWEN_FAST_LEVERN_BUILD_MS
        QWEN_FAST_LEVERN_EPOCH_SCOPE QWEN_FAST_LEVERN_FAULT QWEN_FAST_LEVERN_KV_READ QWEN_FAST_LEVERN_MAX_DECODE_GAP_S QWEN_FAST_LEVERN_MAX_PARK_S
        QWEN_FAST_LEVERN_OWED_CAP_MS QWEN_FAST_LEVERN_PARK QWEN_FAST_LEVERN_PARK_SLOTS QWEN_FAST_LEVERN_PREFILL_SHARE QWEN_FAST_LEVERN_ROUNDS
        QWEN_FAST_LEVERN_SHORT_SHARE QWEN_FAST_LEVERN_SHORT_TOKENS QWEN_FAST_LEVERN_SHORT_WAIT_S QWEN_FAST_LEVERN_SOLO_STEP_TOKENS
        QWEN_FAST_LEVERN_STEP_TOKENS QWEN_FAST_LEVERN_TTFT_TARGET_S QWEN_FAST_LEVER_N QWEN_FAST_M3_REQUEST_WARM QWEN_FAST_MAX_POSITION QWEN_FAST_OCTO
        QWEN_FAST_OCTO_ATTN_BUNDLE QWEN_FAST_OCTO_GLUE8 QWEN_FAST_OCTO_MIN_LIVE QWEN_FAST_OCTO_WAIVER QWEN_FAST_OUTPUT_BUDGET
        QWEN_FAST_REPLAY_GROUP_ROWS QWEN_FAST_SEQ_DEADLINE_S QWEN_FAST_SEQ_PUBLISH_LOG QWEN_FAST_SEQ_STAGE_LOG QWEN_FAST_SHARD_CHECK
        QWEN_FAST_SOLO_LANE QWEN_FAST_SOLO_PACKED QWEN_FAST_STALL_BUILD_S QWEN_FAST_STALL_DEADLINE_S QWEN_FAST_STICKY_SESSIONS QWEN_FAST_W2_OFF_AFTER
        QWEN_FAST_W2_OFF_PATH THATCH_SERVING_SESSION_CAP
    ''',
    # memory: DRAM and host memory reservations, engine parking, startup and cache locations.
    'memory': '''
        QWEN_FAST_CAPTURE_PLUG QWEN_FAST_CAPTURE_PLUG_ENGINES QWEN_FAST_CAPTURE_PLUG_ENGINE_LEAVE_MB QWEN_FAST_CAPTURE_PLUG_LEAVE_MB
        QWEN_FAST_CAPTURE_PLUG_MIN_FREE_MB QWEN_FAST_CAPTURE_PLUG_RESERVE_MB QWEN_FAST_DEVICE_ZEROS QWEN_FAST_DRAM_ENGINE_BUILD_MB
        QWEN_FAST_DRAM_LARGEST_BUFFER_MB QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB QWEN_FAST_EXTENT_REPLAY QWEN_FAST_GATE_DRAM_BALLAST
        QWEN_FAST_KV_RESERVATION QWEN_FAST_LAZY_SHARD_W QWEN_FAST_MEMORY_LEDGER QWEN_FAST_MEMORY_LEDGER_L1 QWEN_FAST_PACKED_SAMPLER_PREWARM
        QWEN_FAST_PARKED_DRAFTS QWEN_FAST_PARKED_ENGINES QWEN_FAST_PARKED_FAULT QWEN_FAST_PARKED_NEGATIVE QWEN_FAST_PARKED_OFF_AFTER
        QWEN_FAST_PUBLISH_PREWARM QWEN_FAST_SKIP_BLOCK_STREAM TT_CACHE_PATH TT_METAL_CACHE TT_METAL_HOME VLLM_CACHE_ROOT
    ''',
    # diag: logging, tracing, census, fault handlers and probes.
    'diag': '''
        QWEN_FABRIC_LINK_PROBE QWEN_FAST_CARRY_LOG QWEN_FAST_CCL_HANDLE_GUARD QWEN_FAST_CCL_HANDLE_LOG QWEN_FAST_FAULTHANDLER
        QWEN_FAST_GDN_DISPATCH_DIAG QWEN_FAST_PHASE_LOG QWEN_FAST_PHASE_TIMING QWEN_FAST_RUNTIME_BINARY_SHA256 QWEN_FAST_TP4_HOSTGAP_LOG
        QWEN_FAST_TP4_HOSTGAP_PROBE QWEN_FAST_TP4_ROUND_HOST_LOG QWEN_FAST_TRACE_CENSUS QWEN_FAST_TRACE_CENSUS_GRAPH
    ''',
    # plumbing: platform plumbing and gate bookkeeping: offline flags, mesh selection, profile names, evidence waivers.
    'plumbing': '''
        HF_HUB_OFFLINE MESH_DEVICE OMP_NUM_THREADS QWEN_C2_GATE_PROFILE QWEN_C2_PROFILE QWEN_C2_SERVING QWEN_CARDS_ALLOCATED
        QWEN_FAST_262K_EVIDENCE_WAIVER QWEN_FAST_TP QWEN_HARDWARE_TESTS QWEN_SKIP_UNUSED_SINGLETON_POSITIONS TRANSFORMERS_OFFLINE
        VLLM_USE_V2_MODEL_RUNNER
    ''',
}

PARSED = {name: tuple(text.split()) for name, text in CLASSES.items()}


def profile_env_names():
    document = json.loads(PROFILES.read_text(encoding='utf-8'))
    names = set()
    for profile in document['profiles'].values():
        names.update((profile.get('env') or {}).keys())
    return names


def dockerfile_env_names():
    names = set()
    for match in ENV_INSTRUCTION.finditer(DOCKERFILE.read_text(encoding='utf-8')):
        for token in match.group(1).replace('\\\n', ' ').split():
            if '=' in token:
                names.add(token.split('=', 1)[0])
    return names


def classified():
    return {name for names in PARSED.values() for name in names}


def unclassified(names):
    return sorted(name for name in names if name not in classified() and name not in kvnvme.NAMES)


class EnvClassTests(unittest.TestCase):
    def test_every_env_name_of_the_profiles_and_the_image_is_classified(self):
        seen = profile_env_names() | dockerfile_env_names()
        self.assertGreater(len(seen), 150, 'the profile table or the Dockerfile did not parse')
        unknown = unclassified(seen)
        self.assertEqual(unknown, [], 'unclassified env names: put each in a class of CLASSES in scripts/ci/test_qwen_kv_compat.py (does it change the bytes the prefill stores?'
                         ' then it is a \'kv\' name and goes in qwen_kv_nvme.FINGERPRINTED_ENV too): %s' % ', '.join(unknown))

    def test_a_new_profile_switch_fails_until_it_is_classified(self):
        self.assertEqual(unclassified(profile_env_names() | {'QWEN_FAST_BRAND_NEW_SWITCH'}), ['QWEN_FAST_BRAND_NEW_SWITCH'])
        self.assertEqual(unclassified(['QWEN_SDPA_BF8', 'QWEN_PREFIX_NVME_TTL_S']), [])

    def test_no_name_is_in_two_classes(self):
        counts = {}
        for names in PARSED.values():
            for name in names:
                counts[name] = counts.get(name, 0) + 1
        self.assertEqual(sorted(name for name, count in counts.items() if count > 1), [])

    def test_the_kv_class_is_exactly_the_fingerprinted_tuple(self):
        self.assertEqual(sorted(PARSED['kv']), sorted(kvnvme.FINGERPRINTED_ENV))
        self.assertEqual(len(set(kvnvme.FINGERPRINTED_ENV)), len(kvnvme.FINGERPRINTED_ENV))

    def test_the_nvme_tiers_own_names_are_never_in_the_table(self):
        self.assertEqual(sorted(name for name in classified() if name in kvnvme.NAMES), [])

    def test_the_peer_to_peer_names_are_the_tiers_own_and_the_layout_component_carries_the_switch(self):
        # QWEN_PREFIX_NVME_P2P changes the stored bytes' ORDER (chip-major), so it is not a 'kv' env switch (those are read from the process environment by value): it is the plan's
        # `p2p`, and the fingerprint's 'layout' component says so. Its knobs (threads, staging size, the card list) change nothing stored.
        for name in ('QWEN_PREFIX_NVME_P2P', 'QWEN_PREFIX_NVME_P2P_THREADS', 'QWEN_PREFIX_NVME_P2P_STAGING_MIB', 'QWEN_PREFIX_NVME_P2PMEM', 'QWEN_PREFIX_NVME_P2P_ALLOW_UNVERIFIED_MAP'):
            self.assertIn(name, kvnvme.NAMES)
            self.assertNotIn(name, classified())
            self.assertEqual(unclassified([name]), [])
        self.assertNotIn('QWEN_PREFIX_NVME_P2P', kvnvme.FINGERPRINTED_ENV)
        self.assertIn('QWEN_PREFIX_NVME_P2P', kvnvme.GATE_ONLY)

    def test_the_two_components_with_their_own_env_name_are_in_the_table(self):
        self.assertIn('QWEN35_GDN_STATE_BF16', PARSED['covered'])
        self.assertIn(kvnvme.DRAFTER_FLAG, PARSED['drafter'])


class FingerprintEnvTests(unittest.TestCase):
    class Io(object):
        fingerprint = 'kv-layout'
        block_bytes = 2228224
        chips = 4

    BASE = {'QWEN_PREFIX_NVME': '1', 'QWEN_PREFIX_NVME_DEVICE': '/dev/null', 'QWEN_PREFIX_NVME_RETENTION_ACK': '1'}

    def components(self, environ=None, **flags):
        plan, problems = kvnvme.parse(dict(self.BASE, **flags))
        self.assertEqual(problems, [])
        fingerprint, components = kvnvme.layout_fingerprint(self.Io(), plan, dict(environ or {}))
        return fingerprint, components

    def fingerprint(self, environ=None, **flags):
        return self.components(environ, **flags)[0]

    def test_a_fingerprinted_switch_changes_the_fingerprint(self):
        base = self.fingerprint({})
        for name in kvnvme.FINGERPRINTED_ENV:
            self.assertNotEqual(self.fingerprint({name: '1'}), base, name)
            self.assertNotEqual(self.fingerprint({name: '1'}), self.fingerprint({name: '0'}), name)

    def test_the_other_classes_do_not(self):
        base = self.fingerprint({})
        for group in ('exact', 'tier', 'audit', 'draft', 'decode', 'sched', 'memory', 'diag', 'plumbing'):
            for name in PARSED[group]:
                self.assertEqual(self.fingerprint({name: 'x'}), base, '%s (%s) must not be in the fingerprint' % (name, group))

    def test_neither_the_profile_name_nor_the_image_build_is_in_it(self):
        base, components = self.components({})
        self.assertEqual(self.fingerprint({'QWEN_C2_PROFILE': 'one-profile'}), base)
        self.assertEqual(self.fingerprint({'QWEN_C2_PROFILE': 'another-profile'}), base)
        self.assertEqual(sorted(components), sorted(['layout', 'epoch', 'model', 'drafter', 'kv', 'block_bytes', 'page', 'chunk', 'mesh', 'gdn_bf16', 'ckpt_format', 'env', 'compat']))

    def test_the_peer_to_peer_switch_moves_the_layout_component_and_so_the_fingerprint_and_its_knobs_do_not(self):
        base, components = self.components({})
        on, on_components = self.components({}, QWEN_PREFIX_NVME_P2P='1')
        self.assertEqual((components['layout'], on_components['layout']), (kvnvme.LAYOUT, kvnvme.LAYOUT_CHIPMAJOR))
        self.assertNotEqual(on, base)
        for knob in ({'QWEN_PREFIX_NVME_P2P_THREADS': '2'}, {'QWEN_PREFIX_NVME_P2P_STAGING_MIB': '64'}, {'QWEN_PREFIX_NVME_P2PMEM': '0000:41:00.0'}):
            self.assertEqual(self.fingerprint({}, QWEN_PREFIX_NVME_P2P='1', **knob), on, knob)
        # (the env the process runs under cannot move it either: the switch is read from the plan)
        self.assertEqual(self.fingerprint({'QWEN_PREFIX_NVME_P2P': '1'}), base)

    def test_the_epoch_is_part_of_it_and_so_is_the_override_token(self):
        original = kvnvme.KV_EPOCH
        base = self.fingerprint({})
        try:
            kvnvme.KV_EPOCH = original + 1
            self.assertNotEqual(self.fingerprint({}), base)
        finally:
            kvnvme.KV_EPOCH = original
        self.assertEqual(self.fingerprint({}), base)
        self.assertNotEqual(self.fingerprint({}, QWEN_PREFIX_NVME_COMPAT='gate-a'), base)
        self.assertNotEqual(self.fingerprint({}, QWEN_PREFIX_NVME_COMPAT='gate-a'), self.fingerprint({}, QWEN_PREFIX_NVME_COMPAT='gate-b'))


if __name__ == '__main__':
    unittest.main()
