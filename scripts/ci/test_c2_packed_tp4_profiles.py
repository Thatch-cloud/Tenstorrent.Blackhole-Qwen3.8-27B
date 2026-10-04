"""The four-card S2 profiles (c2-packed-tp4, c2-packed-tp4-gate) and the pair's profiles they sit beside.

The pair's fast-path profiles are held byte for byte (a digest of each profile's canonical JSON, taken at the commit
before the four-card profiles existed): adding a width must not move what the pair serves. The four-card profiles are
their pair twins' engine and limits with exactly the documented differences - the mesh, the ring descriptor, QWEN_FAST_TP=4
and the fast path's four-card environment, FABRIC_1D - and pass the contract's mesh rules and the admission's environment
check when laid over the image's ENV."""

import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

ROOT = HERE.parent.parent
PROFILES = HERE / 'qwen_c2_profiles.json'
DOCKERFILE = ROOT / 'docker' / 'qwen-c2-serving.Dockerfile'
M3 = (True, 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1')

# sha256 of json.dumps(profile, sort_keys=True, separators=(',', ':')) for the pair's fast-path profiles at fd5ea503.
PAIR_DIGESTS = {
    'exact': 'cb8b837b64cf6be5a81559b482e21cbe2316e40d1d8e3394e2ec0635c0d1cd69',
    'c2': '41e2e2feb2e0e82b8a13f3f4948558f3e267d3454b47524f964a95655412e0e9',
    'c2-gate': 'b283978e048bbbcbfb9772d9d5b8e536b5d5e2e8c5484deb69a3ba7b3c34733d',
    'c2-packed': '5b2a6fe3ca26edb1ec819f36175b62f113a6de98896323e3f6f217e260b649e0',
    'c2-packed-gate': '9e07f12ffed7e83dc8bd98f493b8ee1ec7903795f6992258b8bc5a604f4892a0',
    'c2-packed-prefix': '4a3021630eeaae7e8aaec41206cf8277598715195afcda3c072f8b5b733ee655',
    'c2-packed-prefix-gate': '3fac0901d8a7ccbf0b74332ad7b9dd5558622803315859cf9458f086ac65e5c1',
}
FOUR_ENV = {'QWEN_FAST_TP': '4', 'QWEN_FAST_SDPA_MODES': 'tail,share', 'QWEN_PROJECTION_LINKS': '2',
            'QWEN_GDN_PREFILL_MMRS': '0', 'QWEN_FAST_GDN_PREFILL_CONV_AUDIT': '4', 'QWEN_FAST_TP_KV_SLIDE': '1'}
# The nine image flags the four-card profiles turn off: their code carries the pair's chip or head literals in text-patched,
# kernel or two-chip form (quad_draft, fused_commit, the draft K/V slide and the traced publish's K/V fusion, the MLP block
# stream, the GDN direct-window and shared-QK experiments) that the four-card port has not reached.
OFF_ENV = {'QWEN_FAST_QUAD_DRAFT': '0', 'QWEN_FAST_FUSED_COMMIT': '0', 'QWEN_FAST_FUSED_COMMIT_INPLACE': '0',
           'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS': '0', 'QWEN_DRAFT_KV_SLIDE_EXPERIMENT': '0',
           'QWEN_FAST_TRACED_PUBLISH': '0', 'QWEN_MLP_BLOCK_STREAM_EXPERIMENT': '0', 'QWEN_GDN_DIRECT_WINDOW': '0',
           'QWEN_GDN_SHARED_QK_EXPERIMENT': '0'}
# QWEN_C2_GATE_PROFILE is the admission waiver's marker: only the gate-only profile's own env carries it.
AUDIT_ENV = {'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1', 'QWEN_C2_GATE_PROFILE': '1'}
# tp4-serve-7: the traffic profile serves with the verify audits OFF and the pinned sampler recorded in the verify trace. This is the
# verified winning recipe (tp4-serve-6, the gate-only c2-packed-tp4-speed-strace): 5/5 audits-off passes of the deterministic hang,
# the engine-build hang shape (concurrent8_code) passed, the S3a matrix gate identical to solo (first run and rerun, POLICY PASS), about
# +23% four-user coding throughput over the audited profile. With the audits off and WITHOUT the in-trace sampler the profile hung
# twice at the packed-to-sequential tail of the first four-user answers (SR v163, SS v164); the in-trace sampler is what removes the hang.
# Only the gate profile takes the waiver.
VERIFY_AUDITS = {'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0',
                 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE': '1'}
# The environment the traffic profile carried through tp4-serve-5 (audits on, no in-trace sampler): the freeze twins (f2, f12) were
# measured on it and still carry it; they are asserted against production with exactly these keys put back.
PRE_SERVE7_TRAFFIC = {'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1'}
# The tail caps and the sequential-step watchdog (tp4-serve-3): the traffic profile and its audits-off speed twin carry them.
TAIL_CAPS = {'QWEN_FAST_BUDGET_CAP': '1', 'QWEN_FAST_SEQ_DEADLINE_S': '120'}
# The hang diagnosis's profiles: the speed twin as it stood before the caps (the environment of SS v164) with the caps OFF
# and the host-only instruments on; the factorial arms turn one verify audit back on each.
# (tp4-serve-4) plus the stall watch (stacks, then the pinned triage, then the end) and the handle guard in log mode.
DIAG_INSTRUMENTS = {'QWEN_FAST_BUDGET_CAP': '0', 'QWEN_FAST_SEQ_DEADLINE_S': '120', 'QWEN_FAST_SEQ_STAGE_LOG': '1',
                    'QWEN_FAST_TRACE_CENSUS': '1', 'QWEN_FAST_CCL_HANDLE_LOG': '1', 'QWEN_FAST_MEMORY_LEDGER_L1': '1',
                    'QWEN_FAST_STALL_DEADLINE_S': '120', 'QWEN_FAST_CCL_HANDLE_GUARD': 'log'}
# The graph census is off in every diag arm and says so explicitly: it is v173's fatal configuration (the graph processor reads tensors
# inside a trace capture, which refuses reads).
DIAG_PROFILES = {'c2-packed-tp4-diag': {'QWEN_FAST_TRACE_CENSUS_GRAPH': '0'},
                 'c2-packed-tp4-diag-t1': {'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_TRACE_CENSUS_GRAPH': '0'},
                 'c2-packed-tp4-diag-t2': {'QWEN_FAST_VERIFY_T2_AUDIT': '1', 'QWEN_FAST_TRACE_CENSUS_GRAPH': '0'}}
# The fix arm: the audits-off speed twin plus the capture plug (flags default off everywhere else), the stall watch and the guard.
# The plug sizes are per BANK (megabytes), written out so no default decides them; the guard is in log mode (fail mode can refuse a
# request on a pattern every audited run has).
FIX_ENV = {'QWEN_FAST_CAPTURE_PLUG': '1', 'QWEN_FAST_CAPTURE_PLUG_ENGINES': '1', 'QWEN_FAST_CAPTURE_PLUG_LEAVE_MB': '512',
           'QWEN_FAST_CAPTURE_PLUG_RESERVE_MB': '256', 'QWEN_FAST_CAPTURE_PLUG_ENGINE_LEAVE_MB': '256',
           'QWEN_FAST_CAPTURE_PLUG_MIN_FREE_MB': '128', 'QWEN_FAST_TRACE_CENSUS_GRAPH': '0',
           'QWEN_FAST_STALL_DEADLINE_S': '120', 'QWEN_FAST_CCL_HANDLE_GUARD': 'log'}
# The sampler isolation arms (tp4/sampler): each is its base profile plus exactly one flag (the audits-off hang's factorial found the T1
# audit's pinned sampler protects; these two keep the sampler and drop the audit's per-round readback and compare).
PREWARM, IN_TRACE = 'QWEN_FAST_PACKED_SAMPLER_PREWARM', 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE'
SAMPLER_PROFILES = {'c2-packed-tp4-diag-sprewarm': ('c2-packed-tp4-diag', PREWARM),
                    'c2-packed-tp4-diag-strace': ('c2-packed-tp4-diag', IN_TRACE),
                    'c2-packed-tp4-speed-sprewarm': ('c2-packed-tp4-speed', PREWARM),
                    'c2-packed-tp4-speed-strace': ('c2-packed-tp4-speed', IN_TRACE),
                    # the timed best with hang fix A: every timed audits-off profile must carry a hang fix
                    'c2-packed-tp4-best-strace': ('c2-packed-tp4-best', IN_TRACE)}
# The request-shard-argmax arms (tp4/next-2): each is its base profile plus exactly these env flags (the audited one also carries the gate
# waiver's marker, as every gate-only twin of the traffic profile does).
RSHARD, RSHARD_AUDIT = 'QWEN_FAST_REQUEST_SHARD_ARGMAX', 'QWEN_FAST_REQUEST_SHARD_AUDIT'
RSHARD_PROFILES = {'c2-packed-tp4-diag-rshard': ('c2-packed-tp4-diag', {RSHARD: '1'}),
                   'c2-packed-tp4-speed-rshard': ('c2-packed-tp4-speed', {RSHARD: '1'}),
                   'c2-packed-tp4-best-rshard': ('c2-packed-tp4-best', {RSHARD: '1'}),
                   'c2-packed-tp4-diag-t1-rshard-audit': ('c2-packed-tp4-diag-t1', {RSHARD: '1', RSHARD_AUDIT: '1'}),
                   'c2-packed-tp4-gate-rshard-audit': ('c2-packed-tp4', {RSHARD: '1', RSHARD_AUDIT: '1', 'QWEN_C2_GATE_PROFILE': '1'})}
# The cheap-levers timing twins (tp4/next-3-cheap): each is c2-packed-tp4-speed-strace plus exactly its lever (test_tp4_next3_cheap holds the rule);
# they carry the audits-off recipe, so the caps and the in-trace sampler are theirs as the strace arm's.
CHEAP_PROFILES = ('c2-packed-tp4-speed-strace-ring', 'c2-packed-tp4-speed-strace-draftwide')
# tp4/warm4 (test_tp4_warm4 holds each as its base plus exact deltas): the request-width warm twins, which carry the caps and the stall watch.
WARM4_PROFILES = ('c2-packed-tp4-warm4-diag', 'c2-packed-tp4-warm4-control', 'c2-packed-tp4-warm4-even-diag', 'c2-packed-tp4-warm4-gate',
                  'c2-packed-tp4-speed-warm4', 'c2-packed-tp4-warm4-diag-oldtail')
# The eight-seat twins (tp4/seats8) are production's recipe plus their deltas (test_c2_packed_tp4_seats8_profiles holds them): they carry the
# tail caps and the in-trace sampler, and the diag twin the stall watch and the handle guard, as c2-packed-tp4 and the diag arms do.
SEATS8_PROFILES = ('c2-packed-tp4-8', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate', 'c2-packed-tp4-8-diag-strace',
                   'c2-packed-tp4-8-diag-strace-nowarm', 'c2-packed-tp4-8-diag-strace-rshard',
                   # the 262k-window twins (tp4/seats8-262k, test_c2_packed_tp4_262k_profiles holds them): the same recipe
                   'c2-packed-tp4-8x262k', 'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-time-gate',
                   'c2-packed-tp4-8x262k-diag-strace')
# tp4/next-4 (test_tp4_next4 holds each as its base plus exact deltas): the traffic candidate, its warm4 twin and the eight-seat levers arm.
# ...and tp4/sdpa-long's gate-only twin of c2-packed-tp4-best-strace plus exactly QWEN_FAST_TP4_SDPA (test_sdpa_long_tp holds it to that).
NEXT4_PROFILES = ('c2-packed-tp4-best-ship', 'c2-packed-tp4-best-ship-warm4', 'c2-packed-tp4-8-best', 'c2-packed-tp4-best-sdpa')
# tp4/next-5 (test_tp4_next5 holds each as its base plus exact deltas): the tpub and pair-slice traffic candidates and the pair-slice gate twins.
NEXT5_PROFILES = ('c2-packed-tp4-best-dbf16', 'c2-packed-tp4-best-gate-dbf16', 'c2-packed-tp4-best-ship-tpub', 'c2-packed-tp4-best-ship-glue', 'c2-packed-tp4-best-strace-glue', 'c2-packed-tp4-best-gate-glue', 'c2-packed-tp4-8-best-quad', 'c2-packed-tp4-8-best-quad-dbf16', 'c2-packed-tp4-8-best-quad-gate')
# tp4/262k8 (test_tp4_262k8_best_profiles holds each as its 262k eight-seat twin plus the levers, the smaller pool and the lever audits): the eight-seat 262k best-lever arms.
BEST262K_PROFILES = ('c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit', 'c2-packed-tp4-8x262k-best-nosamp-audit', 'c2-packed-tp4-8x262k-best-stack-audit', 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-time-gate-d2', 'c2-packed-tp4-8x262k-best-time-gate-dbf16', 'c2-packed-tp4-8x262k-best-time-gate-lookup', 'c2-packed-tp4-8x262k-best-time-gate-nosamp', 'c2-packed-tp4-8x262k-best-time-gate-s1', 'c2-packed-tp4-8x262k-best-time-gate-stack', 'c2-packed-tp4-8x262k-best-time-gate-u1', 'c2-packed-tp4-8x262k-best-u1-audit', 'c2-packed-tp4-8x262k-ship')
# tp4/hostgap (test_tp4_hostgap holds each as the best-time-gate control plus flags): the eight-seat host-gap arms and their audited twins.
HOSTGAP_PROFILES = ('c2-packed-tp4-8x262k-hostgap-1', 'c2-packed-tp4-8x262k-hostgap-1-audit', 'c2-packed-tp4-8x262k-hostgap-2', 'c2-packed-tp4-8x262k-hostgap-2-audit')
# tp4/w1 (test_tp4_w1 holds each as the best-time-gate control minus the in-trace sampler plus the stack of levers, and its audited twin).
W1_PROFILES = ('c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w1-audit-nod1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-nod1')
NEXT5_PROFILES = NEXT5_PROFILES + BEST262K_PROFILES + HOSTGAP_PROFILES + W1_PROFILES
# tp4/sdpa-multi (test_sdpa_multi_tp holds each as the best-time-gate arm plus exactly the SDPA flag, and the audited one plus the audit flag).
SDPAMULTI_PROFILES = ('c2-packed-tp4-8x262k-best-sdpamulti', 'c2-packed-tp4-8x262k-best-sdpamulti-audit')
NEXT5_PROFILES = NEXT5_PROFILES + SDPAMULTI_PROFILES
# The GDN glue quick wins (tp4/gluefix, test_gdn_pair_slice holds the rule): the timed and diagnostic arms are c2-packed-tp4-speed-strace plus exactly one flag
# (same exemptions as the cheap twins); the audited arm is c2-packed-tp4-gate plus the flag and the vglue audit (a four-card profile that may carry the audit).
CHEAP_PROFILES = CHEAP_PROFILES + ('c2-packed-tp4-speed-strace-pairslice', 'c2-packed-tp4-speed-strace-dispatchdiag')
GLUEFIX_AUDITED = ('c2-packed-tp4-gate-pairslice',)
# tp4/samp-draft: the sampler and drafter lever twins (timed, on best-strace) and their audited twins (on best-gate); test_tp4_sampdraft holds each
# equal to its control plus exactly its flags. They carry the caps, the hang fix, the levers: every exemption NEXT4_PROFILES has, they have.
SAMPDRAFT_PROFILES = ('c2-packed-tp4-best-samp', 'c2-packed-tp4-best-d2', 'c2-packed-tp4-best-gate-samp', 'c2-packed-tp4-best-gate-d2')
NEXT5_PROFILES = NEXT5_PROFILES + SAMPDRAFT_PROFILES
# tp4/262k8-x (test_tp4_262k8_x holds each as the 262k eight-seat timed or audited twin plus/minus exactly its lever): the experiments image's per-lever arms.
X262K_TIMED = tuple('c2-packed-tp4-8x262k-best-time-gate-' + lever for lever in ('nosamp', 's1', 'd2', 'dbf16', 'lookup', 'stack'))
X262K_AUDITED = ('c2-packed-tp4-8x262k-best-nosamp-audit', 'c2-packed-tp4-8x262k-best-stack-audit')
X262K_PROFILES = X262K_TIMED + X262K_AUDITED
NEXT5_PROFILES = NEXT5_PROFILES + X262K_PROFILES
FIX_FLAGS = ('QWEN_FAST_CAPTURE_PLUG', 'QWEN_FAST_CAPTURE_PLUG_ENGINES', 'QWEN_FAST_CCL_HANDLE_GUARD', 'QWEN_FAST_STALL_DEADLINE_S')


# The admission-freeze window's twins (tp4/freeze): each is the traffic profile plus exactly these env flags, gate only.
TRIM, CREDIT = 'QWEN_FAST_ADMISSION_DIAG_TRIM', 'QWEN_FAST_DECODE_STEPS_PER_ADMISSION'
FREEZE_PROFILES = {'c2-packed-tp4-f2': {TRIM: '1'}, 'c2-packed-tp4-f12': {TRIM: '1', CREDIT: '1'}}


SPEED = 'QWEN_FAST_TP_KV_SLIDE'
# The speed window's profiles (tp4/speed): each is c2-packed-tp4-gate plus exactly these env differences and a description.
SPEED_PROFILES = {
    'c2-packed-tp4-gate-noslide': {SPEED: '0'},
    'c2-packed-tp4-speed': dict({'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0'}, **TAIL_CAPS),
    'c2-packed-tp4-speed-noslide': {'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0', SPEED: '0'},
}


# The batched-draft window's profiles (tp4/draft): each is its speed or gate base plus exactly these env differences.
QUAD, SINGLES = 'QWEN_FAST_QUAD_DRAFT', 'QWEN_FAST_DRAFT_SINGLES_AUDIT'
DRAFT_PROFILES = {
    'c2-packed-tp4-speed-quad': ('c2-packed-tp4-speed', {QUAD: '1'}),
    'c2-packed-tp4-speed-pairs': ('c2-packed-tp4-speed', {QUAD: '0'}),
    'c2-packed-tp4-gate-quad': ('c2-packed-tp4-gate', {QUAD: '1', SINGLES: 'all'}),
    'c2-packed-tp4-gate-pairs': ('c2-packed-tp4-gate', {QUAD: '0', SINGLES: 'all'}),
}


# The fused-commit window's profiles (tp4/fcommit): each is its gate or speed base plus exactly these env differences.
FUSED, INPLACE, LIVE, AUDIT = ('QWEN_FAST_FUSED_COMMIT', 'QWEN_FAST_FUSED_COMMIT_INPLACE', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS',
                               'QWEN_FAST_FUSED_COMMIT_AUDIT')
FCOMMIT_PROFILES = {
    'c2-packed-tp4-gate-fcommit': ('c2-packed-tp4-gate', {FUSED: '1', INPLACE: '1', AUDIT: '1', LIVE: '0', QUAD: '0'}),
    'c2-packed-tp4-gate-fcommit-live': ('c2-packed-tp4-gate', {FUSED: '1', INPLACE: '1', AUDIT: '1', LIVE: '1', QUAD: '0',
                                                              SINGLES: 'all'}),
    'c2-packed-tp4-gate-fcommit-quad': ('c2-packed-tp4-gate', {FUSED: '1', INPLACE: '1', AUDIT: '1', LIVE: '1', QUAD: '1',
                                                              SINGLES: 'all'}),
    'c2-packed-tp4-speed-fcommit': ('c2-packed-tp4-speed', {FUSED: '1', INPLACE: '1', LIVE: '1', QUAD: '0'}),
    'c2-packed-tp4-speed-fcommit-quad': ('c2-packed-tp4-speed', {FUSED: '1', INPLACE: '1', LIVE: '1', QUAD: '1'}),
    'c2-packed-tp4-speed-fcommit-oop': ('c2-packed-tp4-speed', {FUSED: '1'}),
}

# The combined best (tp4/next): the fused-commit quad arms with the five verify-glue levers added (see VGLUE_PROFILES).
BEST_PROFILES = ('c2-packed-tp4-best', 'c2-packed-tp4-best-gate')
# The timed best with a hang fix (tp4/next-2): c2-packed-tp4-best plus exactly one flag (SAMPLER_PROFILES, RSHARD_PROFILES hold the rule).
BEST_HANG_FIX = ('c2-packed-tp4-best-strace', 'c2-packed-tp4-best-rshard')
# The next-3 window's timed arm (tp4/next-3-scope): the production recipe (c2-packed-tp4-speed-strace: audits off, sampler in the verify
# trace, tail caps) plus the three fused-commit flags and nothing else (test_tp4_next3_window holds that rule).
NEXT3_PROFILES = ('c2-packed-tp4-speed-strace-fcommit',)
# The traced-publication window's arms (tp4/tpub): each is its best base plus exactly the traced-publication flags (TpubProfileTests).
TPUB, TPUB_AUDIT = 'QWEN_FAST_TP4_TRACED_PUBLISH', 'QWEN_FAST_TP4_TRACED_PUBLISH_AUDIT'
TPUB_PROFILES = {'c2-packed-tp4-best-strace-tpub': ('c2-packed-tp4-best-strace', {TPUB: '1'}),
                 'c2-packed-tp4-best-gate-tpub': ('c2-packed-tp4-best-gate', {TPUB: '1', TPUB_AUDIT: '1'})}
# The prompt-lookup window's timed arm (tp4/lookup): c2-packed-tp4-best-strace plus QWEN_FAST_LOOKUP_DRAFT=n3m12 and nothing else (LookupProfileTests).
LOOKUP, LOOKUP_POLICY = 'QWEN_FAST_LOOKUP_DRAFT', 'n3m12'
LOOKUP_PROFILES = ('c2-packed-tp4-best-lookup', 'c2-packed-tp4-best-gate-lookup')
# each lookup arm's base: the timed arm sits on best-strace, the audited smoke arm on best-gate
LOOKUP_BASE = {'c2-packed-tp4-best-lookup': 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-gate-lookup': 'c2-packed-tp4-best-gate'}
# The recurrence value split's twins (tp4/v5split, test_tp4_v5split_window holds the rules): the combined best (timed and audited) and the
# production recipe, each plus QWEN_FAST_GDN_SPLIT_V=2 (and the K5-A audit on the audited one).
V5_BEST = ('c2-packed-tp4-best-v5', 'c2-packed-tp4-best-v5-gate')
V5_PROFILES = V5_BEST + ('c2-packed-tp4-speed-strace-v5',)
FUSED_FAMILY = sorted(set(FCOMMIT_PROFILES) | set(BEST_PROFILES) | set(BEST_HANG_FIX) | set(NEXT3_PROFILES) | set(TPUB_PROFILES) | set(LOOKUP_PROFILES) | set(V5_BEST))


def without_caps(env):
    return {key: value for key, value in env.items() if key not in TAIL_CAPS}


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


def digest(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def image_env():
    """The C2 image's own ENV (the first ENV of the Dockerfile's final stage), as a dict."""
    text = DOCKERFILE.read_text(encoding='utf-8')
    block = text[text.index('ENV QWEN_ATTN_PREP=1'):]
    block = block[:block.index('\n\n')] if '\n\n' in block else block
    lines = []
    for line in block.splitlines():
        lines.append(line.rstrip('\\').strip())
        if not line.rstrip().endswith('\\'):
            break
    return dict(re.findall(r'(QWEN_[A-Z0-9_]+)=(\S+)', ' '.join(lines)))


class PairUntouchedTests(unittest.TestCase):
    def test_the_pairs_fast_path_profiles_are_byte_for_byte_what_they_were(self):
        found = profiles()
        for name, expected in PAIR_DIGESTS.items():
            self.assertEqual(digest(found[name]), expected, '%s changed: a four-card change must not move the pair' % name)

    def test_the_pairs_profiles_name_no_mesh_and_set_no_width(self):
        for name in PAIR_DIGESTS:
            profile = profiles()[name]
            self.assertNotIn('mesh_device', profile, name)
            self.assertNotIn('QWEN_FAST_TP', profile['env'], name)
            self.assertEqual(profile['mesh_graph_descriptor'], contract.PAIR_DESCRIPTOR, name)
            self.assertEqual(contract.mesh_problems(dict(profile, name=name)), [], name)


class FourCardProfileTests(unittest.TestCase):
    def pair(self, name):
        return profiles()[name]

    def test_each_four_card_profile_is_its_pair_twin_with_the_documented_differences(self):
        for four, twin, extra in (('c2-packed-tp4', 'c2-packed', dict(VERIFY_AUDITS)), ('c2-packed-tp4-gate', 'c2-packed-gate', AUDIT_ENV)):
            mine, theirs = profiles()[four], self.pair(twin)
            with self.subTest(profile=four):
                expected = dict(theirs['env'], **dict(FOUR_ENV, **dict(OFF_ENV, **extra)))
                if four == 'c2-packed-tp4':
                    expected.update(TAIL_CAPS)
                self.assertEqual(mine['env'], expected)
                engine = json.loads(json.dumps(mine['engine']))
                self.assertEqual(engine['additional-config']['tt'].pop('fabric_config'), 'FABRIC_1D')
                self.assertEqual(engine, theirs['engine'], 'the engine block is the twin\'s: 131,328 x 4 seats, 8,208 blocks')
                for key in set(mine) | set(theirs):
                    if key not in ('description', 'env', 'engine', 'mesh_device', 'mesh_graph_descriptor', 'gate_only', 'name'):
                        self.assertEqual(mine.get(key), theirs.get(key), key)
                self.assertEqual(mine['mesh_device'], 'P150x4')
                self.assertEqual(mine['mesh_graph_descriptor'], contract.RING_DESCRIPTOR)
                self.assertIs(mine.get('gate_only'), True if four.endswith('-gate') else None)
                self.assertNotIn('sample_on_device_mode', contract.tt_config(mine))
                self.assertTrue(mine['description'].startswith('GATE ONLY') == four.endswith('-gate'))

    def test_the_contract_accepts_the_fast_path_on_four_cards_only_under_the_width(self):
        for name in ('c2-packed-tp4', 'c2-packed-tp4-gate'):
            profile = dict(profiles()[name], name=name)
            self.assertEqual(contract.mesh_problems(profile), [], name)
            self.assertTrue(contract.ring_mesh(profile), name)
            environ = contract.apply_environment(profile, {'MESH_DEVICE': 'P300', 'TT_MESH_GRAPH_DESC_PATH': 'p300'})
            self.assertEqual((environ['MESH_DEVICE'], environ['TT_MESH_GRAPH_DESC_PATH'], environ['QWEN_FAST_TP']),
                             ('P150x4', contract.RING_DESCRIPTOR, '4'), name)
        base = json.loads(json.dumps(profiles()['c2-packed-tp4']))
        no_width = json.loads(json.dumps(base))
        del no_width['env']['QWEN_FAST_TP']
        self.assertTrue(any('fast path' in problem for problem in contract.mesh_problems(no_width)))
        sampler = json.loads(json.dumps(base))
        sampler['engine']['additional-config']['tt']['sample_on_device_mode'] = 'decode_only'
        self.assertTrue(any('sample twice' in problem for problem in contract.mesh_problems(sampler)))
        pair_with_width = json.loads(json.dumps(profiles()['c2-packed']))
        pair_with_width['env']['QWEN_FAST_TP'] = '4'
        self.assertTrue(any('needs mesh_device P150x4' in problem for problem in contract.mesh_problems(pair_with_width)))
        general_with_width = json.loads(json.dumps(profiles()['general-tp4']))
        general_with_width['env']['QWEN_FAST_TP'] = '4'
        self.assertTrue(any('qwen_fast_t16' in problem for problem in contract.mesh_problems(general_with_width)))

    def test_the_gate_profile_boots_only_in_a_gate_and_the_other_never_takes_the_waiver(self):
        self.assertIs(profiles()['c2-packed-tp4-gate']['gate_only'], True)
        self.assertNotIn('gate_only', profiles()['c2-packed-tp4'])

    def test_the_admission_accepts_each_profile_laid_over_the_image_environment(self):
        image = image_env()
        self.assertEqual(image['QWEN_FAST_SDPA_MODES'], 'tail,share,slice', 'the image ENV names the pair\'s modes')
        for name in ('c2-packed-tp4', 'c2-packed-tp4-gate'):
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])
        # and the pair's profile still passes the pair's check over the same image ENV
        environ = dict(image, **profiles()['c2-packed']['env'])
        self.assertEqual(admission.check_environment(environ, M3), [])

    def test_traffic_audits_off_with_the_sampler_in_trace_gate_audits_on_and_only_the_gate_takes_the_waiver(self):
        for name, waiver in (('c2-packed-tp4-gate', True), ('c2-packed-tp4', False)):
            env = profiles()[name]['env']
            if name == 'c2-packed-tp4':
                self.assertEqual({key: env.get(key) for key in VERIFY_AUDITS}, VERIFY_AUDITS, name)
            else:
                self.assertEqual({key: env.get(key) for key in PRE_SERVE7_TRAFFIC}, PRE_SERVE7_TRAFFIC, name)
            self.assertEqual('QWEN_C2_GATE_PROFILE' in env, waiver, name)

    def test_the_switched_off_flags_are_the_images_own_and_are_off(self):
        image = image_env()
        for name in OFF_ENV:
            self.assertIn(name, image, name)
            self.assertEqual(image[name], '1', 'the image turns it on: %s' % name)
            for profile in ('c2-packed-tp4', 'c2-packed-tp4-gate'):
                self.assertEqual(profiles()[profile]['env'][name], '0', (profile, name))

    def test_the_four_card_environment_names_no_value_the_pair_does(self):
        for name in ('c2-packed-tp4', 'c2-packed-tp4-gate'):
            env = profiles()[name]['env']
            self.assertEqual(env['QWEN_PROJECTION_LINKS'], '2', 'two trained links per edge (the image names the pair\'s 4)')
            self.assertEqual(env['QWEN_FAST_SDPA_MODES'], 'tail,share')
            self.assertEqual(env['QWEN_GDN_PREFILL_MMRS'], '0')


class SpeedProfileTests(unittest.TestCase):
    def test_the_slide_flag_is_the_four_card_profiles_and_no_pairs(self):
        found = profiles()
        for name in ('c2-packed-tp4', 'c2-packed-tp4-gate', 'c2-packed-tp4-gate-ring', 'c2-packed-tp4-gate-bf16'):
            self.assertEqual(found[name]['env'][SPEED], '1', name)
        for name in PAIR_DIGESTS:
            self.assertNotIn(SPEED, found[name]['env'], name)
        self.assertNotIn(SPEED, image_env(), 'the image leaves it unset: the eager chain is the default')

    def test_each_speed_profile_is_the_gate_with_only_its_documented_difference(self):
        found = profiles()
        gate = found['c2-packed-tp4-gate']
        for name, difference in SPEED_PROFILES.items():
            mine = found[name]
            with self.subTest(profile=name):
                self.assertEqual(mine['env'], dict(gate['env'], **difference))
                self.assertEqual(mine['engine'], gate['engine'])
                for key in set(mine) | set(gate):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), gate.get(key), key)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIs(mine['gate_only'], True)
                self.assertEqual(mine['env']['QWEN_C2_GATE_PROFILE'], '1', 'the admission waiver: no evidence section is filled yet')
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)

    def test_the_timed_arms_have_both_verify_audits_off_and_the_audited_arms_on(self):
        found = profiles()
        for name, audited in (('c2-packed-tp4-gate', True), ('c2-packed-tp4-gate-noslide', True),
                              ('c2-packed-tp4-speed', False), ('c2-packed-tp4-speed-noslide', False)):
            env = found[name]['env']
            for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                self.assertEqual(env[key], '1' if audited else '0', (name, key))

    def test_the_speed_pair_differs_only_in_the_slide(self):
        found = profiles()
        on, off = found['c2-packed-tp4-speed']['env'], found['c2-packed-tp4-speed-noslide']['env']
        self.assertEqual({key for key in on if on[key] != off.get(key)}, {SPEED} | set(TAIL_CAPS), 'the slide, and the caps the noslide twin predates')
        self.assertEqual((on[SPEED], off[SPEED]), ('1', '0'))
        on, off = found['c2-packed-tp4-gate']['env'], found['c2-packed-tp4-gate-noslide']['env']
        self.assertEqual({key for key in on if on[key] != off.get(key)}, {SPEED})

    def test_the_admission_accepts_each_speed_profile_over_the_image_environment(self):
        image = image_env()
        for name in SPEED_PROFILES:
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


class TailProfileTests(unittest.TestCase):
    """tp4-serve-3: the tail caps on the traffic profile and its speed twin, and the hang diagnosis's gate-only profiles."""

    def test_the_traffic_profile_and_its_speed_twin_differ_only_in_the_sampler_and_the_waiver(self):
        found = profiles()
        traffic, speed = found['c2-packed-tp4']['env'], found['c2-packed-tp4-speed']['env']
        self.assertEqual({key for key in set(traffic) | set(speed) if traffic.get(key) != speed.get(key)},
                         {'QWEN_FAST_PACKED_SAMPLER_IN_TRACE', 'QWEN_C2_GATE_PROFILE'})
        for env in (traffic, speed):
            for key, value in TAIL_CAPS.items():
                self.assertEqual(env[key], value, key)
        self.assertEqual((traffic['QWEN_FAST_VERIFY_T1_AUDIT'], traffic['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), 'audits off (serve-7)')
        self.assertEqual(traffic[IN_TRACE], '1')
        self.assertIn('verified winning recipe', found['c2-packed-tp4']['description'])

    def test_the_caps_are_absent_from_every_profile_that_predates_them(self):
        for name, profile in profiles().items():
            if name not in ('c2-packed-tp4', 'c2-packed-tp4-speed', 'c2-packed-tp4-speed-fix') + tuple(DIAG_PROFILES) + tuple(FREEZE_PROFILES) + tuple(SAMPLER_PROFILES) + tuple(RSHARD_PROFILES) + BEST_PROFILES + CHEAP_PROFILES + WARM4_PROFILES + NEXT3_PROFILES + SEATS8_PROFILES + NEXT4_PROFILES + NEXT5_PROFILES + tuple(TPUB_PROFILES) + LOOKUP_PROFILES + V5_PROFILES:
                self.assertNotIn('QWEN_FAST_BUDGET_CAP', profile['env'], name)
                self.assertNotIn('QWEN_FAST_SEQ_DEADLINE_S', profile['env'], name)

    def test_each_diag_profile_is_the_speed_twin_before_the_caps_with_only_host_only_instruments(self):
        found = profiles()
        speed = found['c2-packed-tp4-speed']
        before = {key: value for key, value in speed['env'].items() if key not in TAIL_CAPS}
        for name, difference in DIAG_PROFILES.items():
            mine = found[name]
            with self.subTest(profile=name):
                self.assertEqual(mine['env'], dict(before, **dict(DIAG_INSTRUMENTS, **difference)))
                self.assertEqual(mine['env']['QWEN_FAST_BUDGET_CAP'], '0', 'the old tail narrowing is what the diagnosis reproduces')
                self.assertEqual(mine['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertIs(mine['gate_only'], True)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                for key in set(mine) | set(speed):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), speed.get(key), key)
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)
                audits = (mine['env']['QWEN_FAST_VERIFY_T1_AUDIT'], mine['env']['QWEN_FAST_VERIFY_T2_AUDIT'])
                self.assertEqual(audits, {'c2-packed-tp4-diag': ('0', '0'),
                                          'c2-packed-tp4-diag-t1': ('1', '0'),
                                          'c2-packed-tp4-diag-t2': ('0', '1')}[name])

    def test_the_diag_instruments_are_the_host_only_flags_and_every_tail_flag_parses(self):
        import trace_census
        import stall_watch
        for name in ('c2-packed-tp4', 'c2-packed-tp4-speed', 'c2-packed-tp4-speed-fix') + tuple(DIAG_PROFILES):
            env = profiles()[name]['env']
            with self.subTest(profile=name):
                self.assertEqual(trace_census.seq_deadline(env), 120.0)
                self.assertIn(env['QWEN_FAST_BUDGET_CAP'], ('0', '1'))
                self.assertEqual(stall_watch.enabled(env), name in DIAG_PROFILES or name.endswith('-fix'))
                if stall_watch.enabled(env):
                    self.assertEqual(stall_watch.deadline('step', env), 120.0)
                self.assertIn(trace_census.handle_guard_mode(env), (None, 'log', 'fail'))
        diag = profiles()['c2-packed-tp4-diag']['env']
        for flag in ('QWEN_FAST_SEQ_STAGE_LOG', 'QWEN_FAST_TRACE_CENSUS', 'QWEN_FAST_CCL_HANDLE_LOG', 'QWEN_FAST_MEMORY_LEDGER_L1'):
            self.assertEqual(diag[flag], '1', flag)
        for name in DIAG_PROFILES:
            self.assertFalse(trace_census.graph_census_enabled(profiles()[name]['env']), name)
        for name in ('c2-packed-tp4-diag', 'c2-packed-tp4-diag-t1', 'c2-packed-tp4-diag-t2'):
            env = profiles()[name]['env']
            self.assertEqual(env['QWEN_FAST_TRACE_CENSUS_GRAPH'], '0', name)
            self.assertFalse(trace_census.graph_census_enabled(env), name)
            self.assertEqual(trace_census.handle_guard_mode(env), 'log', name)
        for name in ('c2-packed-tp4', 'c2-packed-tp4-speed', 'c2-packed-tp4-gate'):
            self.assertFalse(trace_census.graph_census_enabled(profiles()[name]['env']), name)

    def test_the_admission_accepts_each_tail_profile_over_the_image_environment_and_only_the_gate_ones_carry_the_waiver(self):
        image = image_env()
        for name in ('c2-packed-tp4', 'c2-packed-tp4-speed') + tuple(DIAG_PROFILES):
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])
        self.assertNotIn('QWEN_C2_GATE_PROFILE', profiles()['c2-packed-tp4']['env'])


class FixProfileTests(unittest.TestCase):
    """tp4-serve-4: the fix arm (c2-packed-tp4-speed-fix) and the flags that exist only in it and the diagnosis arms."""

    def test_the_fix_arm_is_the_audits_off_speed_twin_plus_exactly_the_fix_flags(self):
        found = profiles()
        mine, speed = found['c2-packed-tp4-speed-fix'], found['c2-packed-tp4-speed']
        self.assertEqual(mine['env'], dict(speed['env'], **FIX_ENV))
        self.assertEqual({key for key in set(mine['env']) | set(speed['env']) if mine['env'].get(key) != speed['env'].get(key)}, set(FIX_ENV))
        for key in set(mine) | set(speed):
            if key not in ('description', 'env'):
                self.assertEqual(mine.get(key), speed.get(key), key)
        self.assertIs(mine['gate_only'], True)
        self.assertTrue(mine['description'].startswith('GATE ONLY'))
        self.assertEqual((mine['env']['QWEN_FAST_VERIFY_T1_AUDIT'], mine['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), 'audits OFF')
        self.assertEqual(mine['env']['QWEN_FAST_BUDGET_CAP'], '1', 'the tail caps stay on')
        self.assertEqual(contract.mesh_problems(dict(mine, name='c2-packed-tp4-speed-fix')), [])
        self.assertIn('c2-packed-tp4-speed-fix', list(found)[list(found).index('c2-packed-tp4-speed'):][:2])

    def test_the_fix_flags_parse_and_the_graph_census_is_off_in_the_fix_arm(self):
        import capture_plug
        import stall_watch
        import trace_census
        env = profiles()['c2-packed-tp4-speed-fix']['env']
        settings = capture_plug.config(env)
        self.assertTrue(settings['engines'])
        self.assertFalse(settings['l1'], 'L1 stays opt-in on top of the plug')
        self.assertEqual(trace_census.handle_guard_mode(env), 'log')
        self.assertFalse(trace_census.graph_census_enabled(env))
        self.assertEqual(stall_watch.deadline('step', env), 120.0)

    def test_the_fix_arms_plug_sizes_fit_inside_what_was_free_per_bank_with_margin(self):
        """v174 before the packed capture: largest_free 20,138 MB per chip on 8 banks = about 2.4 GiB per bank (the sizes are per bank,
        not per chip: the first draft asked for 4096 + 2048 MB and could not attach). The measured need: packed block 224 MB per bank,
        verify capture 168, an engine 62."""
        import capture_plug
        mb = capture_plug.MB
        free_per_bank = 20138 // 8 * mb
        measured = dict(packed_block=224 * mb, verify_capture=168 * mb, engine=62 * mb)
        for label, env in (('profile', profiles()['c2-packed-tp4-speed-fix']['env']), ('defaults', {'QWEN_FAST_CAPTURE_PLUG': '1'})):
            settings = capture_plug.config(env)
            with self.subTest(settings=label):
                packed = settings['leave'] + settings['reserve'] + settings['min_free']
                self.assertLessEqual(packed * 2, free_per_bank, 'leave + reserve + floor must fit with 2x margin in a bank')
                self.assertLess(packed, free_per_bank)
                self.assertLess(settings['leave'] + settings['reserve'], 4240 * mb, 'never more than a bank holds')
                self.assertGreaterEqual(settings['leave'], 2 * measured['verify_capture'], 'the zone holds the verify capture twice over')
                self.assertGreaterEqual(settings['engine_leave'], 2 * measured['engine'], 'an engine zone holds an engine twice over')
                self.assertLessEqual(settings['engine_leave'], settings['leave'])
                self.assertLess(settings['min_free'], settings['reserve'])

    def test_no_other_profile_carries_the_plug_the_guard_in_fail_mode_or_the_stall_watch_except_the_diagnosis_arms(self):
        for name, profile in profiles().items():
            env = profile['env']
            with self.subTest(profile=name):
                self.assertNotIn('QWEN_FAST_CAPTURE_PLUG', env if name != 'c2-packed-tp4-speed-fix' else {})
                self.assertNotIn('QWEN_FAST_CAPTURE_PLUG_ENGINES', env if name != 'c2-packed-tp4-speed-fix' else {})
                if name not in DIAG_PROFILES and name != 'c2-packed-tp4-speed-fix' and not name.startswith('c2-packed-tp4-diag-s') and name not in ('c2-packed-tp4-diag-rshard', 'c2-packed-tp4-diag-t1-rshard-audit') and name not in WARM4_PROFILES + SEATS8_PROFILES:
                    for flag in ('QWEN_FAST_STALL_DEADLINE_S', 'QWEN_FAST_CCL_HANDLE_GUARD', 'QWEN_FAST_TRACE_CENSUS_GRAPH'):
                        self.assertNotIn(flag, env)

    def test_the_production_profile_is_untouched_by_this_window(self):
        env = profiles()['c2-packed-tp4']['env']
        for flag in FIX_FLAGS:
            self.assertNotIn(flag, env)

    def test_the_admission_accepts_the_fix_arm_over_the_image_environment(self):
        environ = dict(image_env(), **profiles()['c2-packed-tp4-speed-fix']['env'])
        self.assertEqual(admission.width(environ), 4)
        self.assertEqual(admission.check_environment(environ, M3), [])

    def test_the_pair_never_reads_the_plug_flag(self):
        import os
        import capture_plug
        for name in PAIR_DIGESTS:
            self.assertIsNone(capture_plug.config(profiles()[name]['env']), name)
        with open(HERE / 'packed_verifier.py', encoding='utf-8') as handle:
            source = handle.read()
        self.assertIn("os.environ.get('QWEN_FAST_TP', '2') != '4'", source, 'the packed block opens a plug at four cards only')
        with open(HERE / 'capture_plug.py', encoding='utf-8') as handle:
            self.assertIn("environ.get('QWEN_FAST_TP', '2') != '4'", handle.read(), 'and so does the engine plug')


class FreezeProfileTests(unittest.TestCase):
    """tp4/freeze: the traffic profile's gate-only twins that switch the admission-freeze flags on, one by one."""

    def test_each_twin_is_the_traffic_profile_plus_exactly_its_flags_and_gate_only(self):
        found = profiles()
        traffic = found['c2-packed-tp4']
        for name, flags in FREEZE_PROFILES.items():
            mine = found[name]
            with self.subTest(profile=name):
                audited = {k: v for k, v in traffic['env'].items() if k != IN_TRACE}
                self.assertEqual(mine['env'], dict(audited, **dict(PRE_SERVE7_TRAFFIC, **dict(flags, QWEN_C2_GATE_PROFILE='1'))))
                self.assertEqual(mine['engine'], traffic['engine'])
                for key in set(mine) | set(traffic):
                    if key not in ('description', 'env', 'gate_only'):
                        self.assertEqual(mine.get(key), traffic.get(key), key)
                self.assertIs(mine['gate_only'], True)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)
                self.assertEqual((mine['env']['QWEN_FAST_VERIFY_T1_AUDIT'], mine['env']['QWEN_FAST_VERIFY_T2_AUDIT']),
                                 ('1', '1'), 'the twins keep the audits on, as the traffic profile did when they were measured')

    def test_the_traffic_profile_and_every_other_profile_leave_the_flags_unset(self):
        for name, profile in profiles().items():
            if name not in FREEZE_PROFILES:
                self.assertNotIn(TRIM, profile['env'], name)
                self.assertNotIn(CREDIT, profile['env'], name)
        for flag in (TRIM, CREDIT):
            self.assertNotIn(flag, image_env(), 'the image leaves it unset: off is the default')

    def test_the_credit_is_on_in_the_f12_twin_alone_and_the_f2_twin_is_its_base(self):
        found = profiles()
        self.assertNotIn(CREDIT, found['c2-packed-tp4-f2']['env'])
        self.assertEqual(found['c2-packed-tp4-f12']['env'], dict(found['c2-packed-tp4-f2']['env'], **{CREDIT: '1'}))
        import serving_prefill_admission
        for name in FREEZE_PROFILES:
            self.assertIn(serving_prefill_admission.decode_steps_per_admission(found[name]['env']), (0, 1), name)

    def test_the_admission_accepts_each_twin_over_the_image_environment(self):
        image = image_env()
        for name in FREEZE_PROFILES:
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


class DraftProfileTests(unittest.TestCase):
    def test_each_draft_profile_is_its_base_with_only_its_documented_difference(self):
        found = profiles()
        for name, (base_name, difference) in DRAFT_PROFILES.items():
            mine, base = found[name], found[base_name]
            with self.subTest(profile=name):
                # the draft window's profiles predate the tail caps: they are the speed twin without them
                self.assertEqual(mine['env'], dict({key: value for key, value in base['env'].items() if key not in TAIL_CAPS},
                                                   **difference))
                self.assertEqual(mine['engine'], base['engine'])
                for key in set(mine) | set(base):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), base.get(key), key)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIs(mine['gate_only'], True)
                self.assertEqual(mine['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)
                self.assertEqual(mine['env']['QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'], '0',
                                 'the four-card quad copies the active banks: the live-banks flag stays off')

    def test_the_flag_pair_differs_in_the_quad_flag_alone(self):
        found = profiles()
        for on, off in (('c2-packed-tp4-speed-quad', 'c2-packed-tp4-speed-pairs'),
                        ('c2-packed-tp4-gate-quad', 'c2-packed-tp4-gate-pairs')):
            left, right = found[on]['env'], found[off]['env']
            self.assertEqual({key for key in set(left) | set(right) if left.get(key) != right.get(key)}, {QUAD}, on)
            self.assertEqual((left[QUAD], right[QUAD]), ('1', '0'))
        self.assertEqual(found['c2-packed-tp4-speed-pairs']['env'],
                         {key: value for key, value in found['c2-packed-tp4-speed']['env'].items() if key not in TAIL_CAPS},
                         'the -pairs profile is the speed profile (before the tail caps), stated as the pair of -quad')

    def test_the_quad_is_the_only_profile_family_with_the_flag_on_and_the_pairs_flags_are_the_images(self):
        found = profiles()
        on = sorted(name for name, profile in found.items() if profile['env'].get(QUAD) == '1'
                    and profile['env'].get('QWEN_FAST_TP') == '4')
        on = [name for name in on if name not in NEXT4_PROFILES + NEXT5_PROFILES]
        self.assertEqual(on, ['c2-packed-tp4-best', 'c2-packed-tp4-best-gate', 'c2-packed-tp4-best-gate-lookup', 'c2-packed-tp4-best-gate-tpub', 'c2-packed-tp4-best-lookup', 'c2-packed-tp4-best-rshard', 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-strace-tpub', 'c2-packed-tp4-best-v5', 'c2-packed-tp4-best-v5-gate', 'c2-packed-tp4-gate-fcommit-quad',
                              'c2-packed-tp4-gate-quad', 'c2-packed-tp4-speed-fcommit-quad', 'c2-packed-tp4-speed-quad'])
        self.assertEqual([name for name in on if 'fcommit' not in name and 'best' not in name],
                         ['c2-packed-tp4-gate-quad', 'c2-packed-tp4-speed-quad'], 'the fused-commit family is the other two')
        image = image_env()
        for name in ('QWEN_FAST_PACKED_PROPOSAL', 'QWEN_FAST_PAIR_ROW_EXACT', 'QWEN_FAST_ROUND_B1', 'QWEN_FAST_PACKED_AUDIT'):
            self.assertEqual(image[name], '1', 'the batched draft needs the image\'s own %s' % name)
        for name in DRAFT_PROFILES:
            self.assertNotIn('QWEN_FAST_PACKED_PROPOSAL', found[name]['env'], 'the image sets it; a profile never turns it off')
        for name in PAIR_DIGESTS:
            self.assertNotIn(SINGLES, found[name]['env'])
        self.assertNotIn(SINGLES, image, 'off in the image: only the audited draft profiles ask for it')

    def test_the_audit_is_on_the_audited_profiles_only_and_the_timed_ones_have_no_audit(self):
        found = profiles()
        for name, audited in (('c2-packed-tp4-gate-quad', True), ('c2-packed-tp4-gate-pairs', True),
                              ('c2-packed-tp4-speed-quad', False), ('c2-packed-tp4-speed-pairs', False)):
            env = found[name]['env']
            self.assertEqual(env.get(SINGLES), 'all' if audited else None, name)
            for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                self.assertEqual(env[key], '1' if audited else '0', (name, key))

    def test_the_admission_accepts_each_draft_profile_over_the_image_environment(self):
        image = image_env()
        for name in DRAFT_PROFILES:
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


class SamplerProfileTests(unittest.TestCase):
    """tp4/sampler: the four sampler isolation arms, each its base profile plus exactly one flag (audits stay off)."""

    def test_each_arm_is_its_base_profile_plus_exactly_its_flag(self):
        found = profiles()
        for name, (base, flag) in SAMPLER_PROFILES.items():
            mine = found[name]
            with self.subTest(profile=name):
                self.assertEqual(mine['env'], dict(found[base]['env'], **{flag: '1'}))
                self.assertEqual([key for key in mine['env'] if key not in found[base]['env']], [flag])
                for key in set(mine) | set(found[base]):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), found[base].get(key), key)
                self.assertIs(mine['gate_only'], True)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIn(flag, mine['description'])
                self.assertEqual((mine['env']['QWEN_FAST_VERIFY_T1_AUDIT'], mine['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), 'audits OFF')
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)
                other = IN_TRACE if flag == PREWARM else PREWARM
                self.assertNotIn(other, mine['env'], 'one arm per profile')

    def test_the_diag_twins_keep_the_old_tail_and_the_instruments_and_the_speed_twins_the_caps_and_no_instruments(self):
        found = profiles()
        for name, (base, flag) in SAMPLER_PROFILES.items():
            env = found[name]['env']
            with self.subTest(profile=name):
                if base.endswith('diag'):
                    self.assertEqual({key: env.get(key) for key in DIAG_INSTRUMENTS}, DIAG_INSTRUMENTS)
                    self.assertEqual(env['QWEN_FAST_TRACE_CENSUS_GRAPH'], '0')
                else:
                    self.assertEqual({key: env[key] for key in TAIL_CAPS}, TAIL_CAPS)
                    for key in ('QWEN_FAST_STALL_DEADLINE_S', 'QWEN_FAST_TRACE_CENSUS', 'QWEN_FAST_CCL_HANDLE_GUARD'):
                        self.assertNotIn(key, env)

    def test_no_other_profile_carries_a_sampler_flag_and_the_traffic_profile_is_the_strace_recipe(self):
        for name, profile in profiles().items():
            if name not in SAMPLER_PROFILES:
                self.assertNotIn(PREWARM, profile['env'], name)
                if name != 'c2-packed-tp4' and name not in CHEAP_PROFILES + NEXT3_PROFILES + SEATS8_PROFILES + NEXT4_PROFILES + NEXT5_PROFILES + tuple(TPUB_PROFILES) + LOOKUP_PROFILES + V5_PROFILES:
                    self.assertNotIn(IN_TRACE, profile['env'], name)
        # tp4-serve-7: production is the strace arm's recipe (audits off, sampler in the verify trace, no prewarm)
        env = profiles()['c2-packed-tp4']['env']
        self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT'], env[IN_TRACE]), ('0', '0', '1'))

    def test_production_is_the_verified_strace_recipe_less_the_gate_marker_and_gate_limits(self):
        """tp4-serve-7: c2-packed-tp4 is c2-packed-tp4-speed-strace (verified on tp4-serve-6) except the gate-only marker, the
        gate_only flag and the gate limits: production keeps the prompt cap and the 8,192-token answer room."""
        found = profiles()
        prod, strace = found['c2-packed-tp4'], found['c2-packed-tp4-speed-strace']
        self.assertEqual(strace['env'], dict(prod['env'], QWEN_C2_GATE_PROFILE='1'))
        for key in set(prod) | set(strace):
            if key not in ('description', 'env', 'gate_only', 'min_answer_tokens', 'max_prompt_tokens'):
                self.assertEqual(prod.get(key), strace.get(key), key)
        self.assertEqual((prod['min_answer_tokens'], prod['max_prompt_tokens']), (8192, 123136))
        self.assertEqual((strace['min_answer_tokens'], strace.get('max_prompt_tokens')), (256, None))
        self.assertNotIn('gate_only', prod)

    def test_the_admission_accepts_each_arm_over_the_image_environment(self):
        for name in SAMPLER_PROFILES:
            environ = dict(image_env(), **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


class RequestShardProfileTests(unittest.TestCase):
    """tp4/next-2: the request-shard-argmax arms, each its base profile plus exactly its flags, gate only."""

    def test_each_arm_is_its_base_plus_exactly_its_flags(self):
        found = profiles()
        for name, (base, flags) in RSHARD_PROFILES.items():
            mine = found[name]
            with self.subTest(profile=name):
                base_env = dict(found[base]['env'])
                if base == 'c2-packed-tp4':
                    # The audited correctness arm keeps the pre-serve-7 audited env explicitly: the audits are its purpose
                    # (the request shard audit is compared beside the T1/T2 audits), and with T1 on the sampler is already in the trace.
                    base_env.pop(IN_TRACE, None)
                    base_env.update(PRE_SERVE7_TRAFFIC)
                self.assertEqual(mine['env'], dict(base_env, **flags))
                for key in set(mine) | set(found[base]):
                    if key not in ('description', 'env', 'gate_only'):
                        self.assertEqual(mine.get(key), found[base].get(key), key)
                self.assertIs(mine['gate_only'], True)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIn(RSHARD + '=1', mine['description'])
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)

    def test_the_audits_and_tails_are_the_bases(self):
        found = profiles()
        for name in ('c2-packed-tp4-diag-rshard', 'c2-packed-tp4-speed-rshard'):
            env = found[name]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), name)
        self.assertEqual({key: found['c2-packed-tp4-diag-rshard']['env'].get(key) for key in DIAG_INSTRUMENTS}, DIAG_INSTRUMENTS)
        env = found['c2-packed-tp4-gate-rshard-audit']['env']
        self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))
        self.assertEqual({key: env[key] for key in TAIL_CAPS}, TAIL_CAPS)
        self.assertIn('audit', found['c2-packed-tp4-gate-rshard-audit']['description'].lower())

    def test_no_other_profile_and_not_the_image_carries_either_flag(self):
        for name, profile in profiles().items():
            if name not in RSHARD_PROFILES and name != 'c2-packed-tp4-8-diag-strace-rshard':
                self.assertNotIn(RSHARD, profile['env'], name)
                self.assertNotIn(RSHARD_AUDIT, profile['env'], name)
        self.assertNotIn(RSHARD, image_env())
        self.assertNotIn(RSHARD_AUDIT, image_env())
        for name in RSHARD_PROFILES:
            if name not in ('c2-packed-tp4-gate-rshard-audit', 'c2-packed-tp4-diag-t1-rshard-audit'):
                self.assertNotIn(RSHARD_AUDIT, profiles()[name]['env'], 'the audit is the audited arm alone')

    def test_the_admission_accepts_each_arm_over_the_image_environment(self):
        for name in RSHARD_PROFILES:
            environ = dict(image_env(), **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


class BestHangFixProfileTests(unittest.TestCase):
    """tp4/next-2: the timed best with a hang fix, each exactly c2-packed-tp4-best plus one flag, audits off, gate only."""

    def test_each_is_the_timed_best_plus_exactly_one_flag(self):
        found = profiles()
        for name, flag in (('c2-packed-tp4-best-strace', IN_TRACE), ('c2-packed-tp4-best-rshard', RSHARD)):
            with self.subTest(profile=name):
                mine, base = found[name], found['c2-packed-tp4-best']
                self.assertEqual(mine['env'], dict(base['env'], **{flag: '1'}))
                self.assertEqual([key for key in mine['env'] if key not in base['env']], [flag])
                for key in set(mine) | set(base):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), base.get(key), key)
                self.assertIs(mine['gate_only'], True)
                self.assertIn(flag + '=1', mine['description'])
                self.assertIn('audits OFF', mine['description'].replace('audits off', 'audits OFF'))
                self.assertEqual((mine['env']['QWEN_FAST_VERIFY_T1_AUDIT'], mine['env']['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))

    def test_each_differs_from_its_speed_twin_in_the_levers_and_the_one_flag_alone(self):
        found = profiles()
        levers = {key for key in set(found['c2-packed-tp4-best']['env']) | set(found['c2-packed-tp4-speed']['env'])
                  if found['c2-packed-tp4-best']['env'].get(key) != found['c2-packed-tp4-speed']['env'].get(key)}
        for name, twin in (('c2-packed-tp4-best-strace', 'c2-packed-tp4-speed-strace'), ('c2-packed-tp4-best-rshard', 'c2-packed-tp4-speed-rshard')):
            mine, other = found[name]['env'], found[twin]['env']
            self.assertEqual({key for key in set(mine) | set(other) if mine.get(key) != other.get(key)}, levers, name)


class TpubProfileTests(unittest.TestCase):
    """tp4/tpub: the traced publication of the sequential step's carry copies, each arm exactly its best base plus its flags, and
    the flag in no other profile, the image or production."""

    def test_each_arm_is_its_base_plus_exactly_its_flags_gate_only(self):
        found = profiles()
        for name, (base_name, difference) in TPUB_PROFILES.items():
            with self.subTest(profile=name):
                mine, base = found[name], found[base_name]
                self.assertEqual(mine['env'], dict(base['env'], **difference))
                self.assertEqual(sorted(key for key in mine['env'] if key not in base['env']), sorted(difference))
                for key in set(mine) | set(base):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), base.get(key), key)
                self.assertIs(mine['gate_only'], True)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                for flag in difference:
                    self.assertIn(flag + '=1', mine['description'])

    def test_the_timed_arm_is_audits_off_and_the_audited_arm_audits_everything_it_adds(self):
        found = profiles()
        timed, audited = found['c2-packed-tp4-best-strace-tpub']['env'], found['c2-packed-tp4-best-gate-tpub']['env']
        self.assertNotIn(TPUB_AUDIT, timed)
        self.assertEqual((timed['QWEN_FAST_VERIFY_T1_AUDIT'], timed['QWEN_FAST_VERIFY_T2_AUDIT'], timed[IN_TRACE]), ('0', '0', '1'))
        self.assertEqual((audited[TPUB], audited[TPUB_AUDIT]), ('1', '1'))
        self.assertIn('NOT QUALIFIED', found['c2-packed-tp4-best-gate-tpub']['description'])

    def test_no_other_profile_nor_the_image_carries_the_flags_and_production_is_unchanged(self):
        found = profiles()
        for name, profile in found.items():
            if name not in TPUB_PROFILES and name != 'c2-packed-tp4-best-ship-tpub':
                self.assertNotIn(TPUB, profile['env'], name)
                self.assertNotIn(TPUB_AUDIT, profile['env'], name)
        self.assertNotIn(TPUB, image_env())
        self.assertNotIn(TPUB_AUDIT, image_env())
        production = found['c2-packed-tp4']['env']
        self.assertEqual(production['QWEN_FAST_FUSED_COMMIT'], '0')
        self.assertEqual(production['QWEN_FAST_TRACED_PUBLISH'], '0')

    def test_the_admission_accepts_each_arm_over_the_image_environment(self):
        for name in TPUB_PROFILES:
            environ = dict(image_env(), **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


class LookupProfileTests(unittest.TestCase):
    """tp4/lookup: prompt-lookup drafting, the timed arm exactly its best base plus the one flag, the flag in no other profile, the image or production."""

    def test_the_arm_is_its_base_plus_exactly_the_flag_gate_only(self):
        found = profiles()
        for name in LOOKUP_PROFILES:
            with self.subTest(profile=name):
                mine, base = found[name], found[LOOKUP_BASE[name]]
                self.assertEqual(mine['env'], dict(base['env'], **{LOOKUP: LOOKUP_POLICY}))
                self.assertEqual(sorted(key for key in mine['env'] if key not in base['env']), [LOOKUP])
                for key in set(mine) | set(base):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), base.get(key), key)
                self.assertIs(mine['gate_only'], True)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIn(LOOKUP + '=' + LOOKUP_POLICY, mine['description'])

    def test_no_other_profile_nor_the_image_carries_the_flag_and_production_is_unchanged(self):
        found = profiles()
        for name, profile in found.items():
            if name not in LOOKUP_PROFILES + ('c2-packed-tp4-8x262k-best-time-gate-lookup',):
                self.assertNotIn(LOOKUP, profile['env'], name)
        self.assertNotIn(LOOKUP, image_env())
        self.assertNotIn(LOOKUP, found['c2-packed-tp4']['env'])

    def test_the_admission_accepts_the_arm_over_the_image_environment(self):
        for name in LOOKUP_PROFILES:
            environ = dict(image_env(), **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


class FusedCommitProfileTests(unittest.TestCase):
    def test_each_fused_commit_profile_is_its_base_with_only_its_documented_difference(self):
        found = profiles()
        for name, (base_name, difference) in FCOMMIT_PROFILES.items():
            mine, base = found[name], found[base_name]
            with self.subTest(profile=name):
                # the fused-commit window's arms predate the tail caps: they are the speed twin without them
                self.assertEqual(mine['env'], dict(without_caps(base['env']), **difference))
                self.assertEqual(mine['engine'], base['engine'])
                for key in set(mine) | set(base):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), base.get(key), key)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIs(mine['gate_only'], True)
                self.assertEqual(mine['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)
                self.assertIn('NOT QUALIFIED', mine['description'])

    def test_the_fused_commit_is_on_in_this_family_and_nowhere_else_at_four_cards(self):
        found = profiles()
        on = sorted(name for name, profile in found.items() if profile['env'].get(FUSED) == '1'
                    and profile['env'].get('QWEN_FAST_TP') == '4')
        on = [name for name in on if name not in NEXT4_PROFILES + NEXT5_PROFILES]
        self.assertEqual(on, FUSED_FAMILY)
        for name, profile in found.items():
            if name not in FUSED_FAMILY + list(NEXT4_PROFILES + NEXT5_PROFILES) and profile['env'].get('QWEN_FAST_TP') == '4':
                for flag in (FUSED, INPLACE, LIVE, AUDIT):
                    self.assertEqual(profile['env'].get(flag, '0'), '0', (name, flag))

    def test_the_sub_flags_never_stand_without_their_parents(self):
        """In place needs the fused commit; live banks need both (the quad twin refuses a live bank that can move); the audit is a
        gate-family arm only and never a timed one."""
        import quad_draft_tp

        for name in FUSED_FAMILY:
            env = profiles()[name]['env']
            with self.subTest(profile=name):
                self.assertEqual(env[FUSED], '1')
                if env.get(LIVE) == '1':
                    self.assertEqual(env[INPLACE], '1')
                self.assertEqual(quad_draft_tp.live_banks_missing(env), [])
                audited = 'gate' in name
                self.assertEqual(env.get(AUDIT), '1' if audited else None, name)
                for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                    self.assertEqual(env[key], '1' if audited else '0', (name, key))
                self.assertEqual(env.get(SINGLES), 'all' if audited and env[LIVE] == '1' else None, name)
                self.assertEqual(env['QWEN_FAST_TP_KV_SLIDE'], '1', 'the fused commit is the four-card slide scope')

    def test_the_timed_profiles_differ_from_their_unfused_twins_in_the_fused_flags_alone(self):
        found = profiles()
        for fused, plain, flags in (
                ('c2-packed-tp4-speed-fcommit', 'c2-packed-tp4-speed-pairs', {FUSED, INPLACE, LIVE}),
                ('c2-packed-tp4-speed-fcommit-quad', 'c2-packed-tp4-speed-quad', {FUSED, INPLACE, LIVE}),
                ('c2-packed-tp4-speed-fcommit-oop', 'c2-packed-tp4-speed', {FUSED})):
            left, right = found[fused]['env'], without_caps(found[plain]['env'])    # the fused arms predate the tail caps
            self.assertEqual({key for key in set(left) | set(right) if left.get(key) != right.get(key)}, flags, fused)

    def test_the_fallback_runs_the_projection_only_and_the_quad_and_pairs_bind_live_banks(self):
        found = profiles()
        oop = found['c2-packed-tp4-speed-fcommit-oop']['env']
        self.assertEqual((oop[FUSED], oop[INPLACE], oop[LIVE]), ('1', '0', '0'))
        for name in ('c2-packed-tp4-speed-fcommit', 'c2-packed-tp4-speed-fcommit-quad'):
            env = found[name]['env']
            self.assertEqual((env[FUSED], env[INPLACE], env[LIVE]), ('1', '1', '1'))
        self.assertEqual((found['c2-packed-tp4-speed-fcommit']['env'][QUAD], found['c2-packed-tp4-speed-fcommit-quad']['env'][QUAD]),
                         ('0', '1'))

    def test_the_images_own_fused_flags_are_the_pairs_and_the_audit_is_not_the_images(self):
        image = image_env()
        for flag in (FUSED, INPLACE, LIVE):
            self.assertEqual(image[flag], '1', flag)
        self.assertNotIn(AUDIT, image, 'off in the image: only the audited gate arms ask for it')

    def test_the_admission_accepts_each_fused_commit_profile_over_the_image_environment(self):
        image = image_env()
        for name in FCOMMIT_PROFILES:
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])


# The verify-glue window's profiles (tp4/vglue): each is its speed or gate base plus exactly these env differences.
C1A, V4A, V2, V1, V3A = ('QWEN_FAST_TP4_COMMIT_LANES', 'QWEN_FAST_TP4_SHARD_VALUES', 'QWEN_FAST_TP4_GDN_GLUE',
                         'QWEN_FAST_TP4_GDN_BLOCK_CONV', 'QWEN_FAST_TP4_ATTN_FOLD')
VGLUE_AUDIT = 'QWEN_FAST_TP4_VGLUE_AUDIT'
ALL_LEVERS = {C1A: '1', V4A: '1', V2: '1', V1: '1', V3A: '1'}
VGLUE_PROFILES = {
    'c2-packed-tp4-speed-vglue': ('c2-packed-tp4-speed', dict(ALL_LEVERS)),
    'c2-packed-tp4-gate-vglue': ('c2-packed-tp4-gate', dict(ALL_LEVERS, **{VGLUE_AUDIT: '1'})),
    'c2-packed-tp4-speed-vglue-c1a': ('c2-packed-tp4-speed', {C1A: '1'}),
    'c2-packed-tp4-speed-vglue-v4a': ('c2-packed-tp4-speed', {V4A: '1'}),
    'c2-packed-tp4-speed-vglue-v2': ('c2-packed-tp4-speed', {V2: '1'}),
    'c2-packed-tp4-speed-vglue-v1': ('c2-packed-tp4-speed', {V2: '1', V1: '1'}),
    'c2-packed-tp4-speed-vglue-v3a': ('c2-packed-tp4-speed', {V3A: '1'}),
    # the combined best (tp4/next-2): the speed twin (tail caps included) with the quad, the fused commit in place on live
    # banks and the five levers; its audited twin is the gate fused-commit quad arm with the same caps, the levers and the audit
    'c2-packed-tp4-best': ('c2-packed-tp4-speed', dict(ALL_LEVERS, **{QUAD: '1', FUSED: '1', INPLACE: '1', LIVE: '1'})),
    'c2-packed-tp4-best-gate': ('c2-packed-tp4-gate-fcommit-quad', dict(ALL_LEVERS, **dict(TAIL_CAPS, **{VGLUE_AUDIT: '1'}))),
}
VGLUE_AUDITED = ('c2-packed-tp4-gate-vglue', 'c2-packed-tp4-best-gate')


class VglueProfileTests(unittest.TestCase):
    def test_each_vglue_profile_is_its_base_with_only_its_documented_difference(self):
        found = profiles()
        for name, (base_name, difference) in VGLUE_PROFILES.items():
            mine, base = found[name], found[base_name]
            with self.subTest(profile=name):
                # the single-lever arms predate the tail caps (the speed twin without them); the combined best is built on the
                # speed twin as it stands, caps included, so the two timed arms schedule the same rounds
                base_env = base['env'] if name in BEST_PROFILES else without_caps(base['env'])
                self.assertEqual(mine['env'], dict(base_env, **difference))
                self.assertEqual(mine['engine'], base['engine'])
                for key in set(mine) | set(base):
                    if key not in ('description', 'env'):
                        self.assertEqual(mine.get(key), base.get(key), key)
                self.assertTrue(mine['description'].startswith('GATE ONLY'))
                self.assertIs(mine['gate_only'], True)
                self.assertEqual(mine['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertEqual(contract.mesh_problems(dict(mine, name=name)), [], name)

    def test_the_block_conv_lever_always_comes_with_the_glue_lever(self):
        for name, profile in profiles().items():
            env = profile['env']
            if env.get(V1) == '1':
                self.assertEqual(env.get(V2), '1', name)

    def test_the_audit_is_on_the_audited_profile_only_and_the_timed_ones_have_no_audit(self):
        found = profiles()
        for name in VGLUE_PROFILES:
            env = found[name]['env']
            self.assertEqual(env.get(VGLUE_AUDIT), '1' if name in VGLUE_AUDITED else None, name)
            for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                self.assertEqual(env[key], '1' if name in VGLUE_AUDITED else '0', (name, key))

    def test_the_levers_are_four_card_profiles_only_and_the_image_leaves_them_unset(self):
        found = profiles()
        image = image_env()
        flags = (C1A, V4A, V2, V1, V3A, VGLUE_AUDIT)
        for name, profile in found.items():
            if name in VGLUE_PROFILES or name in BEST_HANG_FIX or name in NEXT4_PROFILES + NEXT5_PROFILES or name in TPUB_PROFILES or name in LOOKUP_PROFILES or name in V5_BEST or name in GLUEFIX_AUDITED:
                self.assertEqual(profile['env']['QWEN_FAST_TP'], '4', name)
                continue
            for flag in flags:
                self.assertNotIn(flag, profile['env'], (name, flag))
        for flag in flags:
            self.assertNotIn(flag, image, 'the image leaves every lever off: a profile asks for it')

    def test_the_admission_accepts_each_vglue_profile_over_the_image_environment(self):
        image = image_env()
        for name in VGLUE_PROFILES:
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(admission.check_environment(environ, M3), [])

    def test_the_levers_need_the_verify_cuts_the_image_turns_on(self):
        image = image_env()
        for name in ('QWEN_FAST_VERIFY_T1', 'QWEN_FAST_VERIFY_T2', 'QWEN_FAST_GDN_USER_BATCH', 'QWEN_FAST_GDN_SEQ_BLOCK'):
            self.assertEqual(image.get(name), '1', 'the glue levers ride the verify trace these select: %s' % name)


class BestProfileTests(unittest.TestCase):
    """The combined best profiles (tp4/next) are the union of the fused-commit quad arm and the verify-glue arm."""

    def test_the_timed_best_is_the_speed_twin_plus_the_levers_and_nothing_else(self):
        found = profiles()
        speed, best = found['c2-packed-tp4-speed'], found['c2-packed-tp4-best']
        added = {key: value for key, value in best['env'].items() if speed['env'].get(key) != value}
        self.assertEqual(added, dict(ALL_LEVERS, **{QUAD: '1', FUSED: '1', INPLACE: '1', LIVE: '1'}))
        self.assertEqual({key for key in speed['env'] if key not in best['env']}, set(), 'no speed flag is dropped')
        self.assertEqual({key: best['env'][key] for key in TAIL_CAPS}, TAIL_CAPS, 'the tail caps ride the timed best')
        self.assertEqual(best['engine'], speed['engine'])
        for key in set(best) | set(speed):
            if key not in ('description', 'env'):
                self.assertEqual(best.get(key), speed.get(key), key)
        self.assertIn('c2-packed-tp4-speed', best['description'])

    def test_the_audited_best_is_the_gate_fused_commit_quad_arm_plus_the_caps_the_levers_and_their_audit(self):
        found = profiles()
        gate, audited, best = (found['c2-packed-tp4-gate-fcommit-quad'], found['c2-packed-tp4-best-gate'],
                               found['c2-packed-tp4-best'])
        added = {key: value for key, value in audited['env'].items() if gate['env'].get(key) != value}
        self.assertEqual(added, dict(ALL_LEVERS, **dict(TAIL_CAPS, **{VGLUE_AUDIT: '1'})))
        self.assertEqual({key for key in gate['env'] if key not in audited['env']}, set())
        # held against the timed best: the same scheduling (caps), arithmetic (every lever and fused flag) and slide; only the audits differ
        differing = {key for key in set(audited['env']) | set(best['env']) if audited['env'].get(key) != best['env'].get(key)}
        self.assertEqual(differing, {'QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT', 'QWEN_FAST_FUSED_COMMIT_AUDIT',
                                     SINGLES, VGLUE_AUDIT})

    def test_the_audited_best_is_the_traffic_profile_with_the_gate_switches_and_the_levers(self):
        found = profiles()
        traffic, audited = found['c2-packed-tp4']['env'], found['c2-packed-tp4-best-gate']['env']
        differing = {key for key in set(traffic) | set(audited) if traffic.get(key) != audited.get(key)}
        # tp4-serve-7: production is audits off with the sampler in the verify trace; the audited best keeps the audits on and no in-trace sampler
        self.assertEqual(differing, {'QWEN_C2_GATE_PROFILE', QUAD, FUSED, INPLACE, LIVE, AUDIT, SINGLES, VGLUE_AUDIT,
                                     'QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT', IN_TRACE} | set(ALL_LEVERS))

    def test_the_timed_best_has_every_audit_off_and_the_audited_best_every_audit_on(self):
        found = profiles()
        timed, audited = found['c2-packed-tp4-best']['env'], found['c2-packed-tp4-best-gate']['env']
        for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual((timed[key], audited[key]), ('0', '1'), key)
        for key in (AUDIT, VGLUE_AUDIT, SINGLES):
            self.assertNotIn(key, timed, key)
            self.assertIn(key, audited, key)

    def test_the_lone_user_lanes_are_not_on_the_best_profiles_because_the_solo_lane_refuses_the_fused_commit(self):
        """The D0 solo block is built for the M3 block beside an unfused publication: serving_solo_lane.UNSUPPORTED_FLAGS lists
        QWEN_FAST_FUSED_COMMIT, so the lanes profiles stay on their own gate base and the best profiles carry no lane flag."""
        import serving_solo_lane

        self.assertIn('QWEN_FAST_FUSED_COMMIT', serving_solo_lane.UNSUPPORTED_FLAGS)
        for name in BEST_PROFILES:
            env = profiles()[name]['env']
            self.assertNotIn('QWEN_FAST_SOLO_LANE', env, name)
            self.assertNotIn('QWEN_FAST_LANE', env, name)
            with self.assertRaisesRegex(ValueError, 'QWEN_FAST_FUSED_COMMIT=1'):
                serving_solo_lane.solo_lane_admission((True, 'm3'), dict(image_env(), **env, QWEN_FAST_SOLO_LANE='1'),
                                                      log=lambda *args: None)


if __name__ == '__main__':
    unittest.main()
