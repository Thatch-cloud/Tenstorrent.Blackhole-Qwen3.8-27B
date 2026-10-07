"""The combined development window's profiles (docs/tp4-combined-window.md): each is its parent plus exactly the keys named here.

The window gates W2 (SDPA multi + F1) and Lever N merged with prefix reuse together on the production profile family
(c2-packed-tp4-8x262k-ship-prefix). Every new profile derives from that family, is gate only, and carries no waiver marker, no
gate-profile marker and no host-gap log. The default profile is untouched.

Other census modules import the carrier sets below (ALL, MULTI, SDPA_AUDIT, F1, F1_AUDIT, LEVERN) instead of repeating them.
"""

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

R = 'c2-packed-tp4-8x262k-'
SHIP = R + 'ship-prefix'

MULTI_KEY = {'QWEN_FAST_TP4_SDPA': 'multi'}
F1_KEY = {'QWEN_FAST_TP4_CONV_GATES_SPREAD': '1'}
MA_KEY = {'QWEN_FAST_TP4_SDPA_AUDIT': '1'}
FA_KEY = {'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT': '1'}
LEAN_KEYS = {'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0'}
W1_AUDIT_KEYS = {'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1', 'QWEN_FAST_TP4_DRAFT_CONV_AUDIT': '1', 'QWEN_FAST_TP4_DRAFT_HEADS_AUDIT': '1',
                 'QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT': '1', 'QWEN_FAST_FUSED_COMMIT_AUDIT': '1', 'QWEN_FAST_DRAFT_SINGLES_AUDIT': 'all'}
POOL_KEYS = {'QWEN36_MAX_TOKENS_ALL_USERS': '1228288'}
POOL_ENGINE = {'num-gpu-blocks-override': 19200}


def S(suffix):
    return SHIP + suffix


# name -> (parent, env keys set, env keys removed, engine keys set)
SPEC = {
    S('-w2'): (SHIP, dict(MULTI_KEY, **F1_KEY), (), {}),
    S('-w2-audit'): (S('-audit'), dict(MULTI_KEY, **dict(MA_KEY, **dict(F1_KEY, **FA_KEY))), (), {}),
    S('-w2-nof1'): (S('-w2'), {}, ('QWEN_FAST_TP4_CONV_GATES_SPREAD',), {}),
    S('-w2-nof1-audit'): (S('-w2-audit'), {}, ('QWEN_FAST_TP4_CONV_GATES_SPREAD', 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'), {}),
    S('-levern-w2'): (S('-levern'), dict(MULTI_KEY, **F1_KEY), (), {}),
    S('-levern-w2-audit'): (S('-levern-audit'), dict(MULTI_KEY, **dict(MA_KEY, **dict(F1_KEY, **FA_KEY))), (), {}),
    S('-levern-w2-nof1'): (S('-levern-w2'), {}, ('QWEN_FAST_TP4_CONV_GATES_SPREAD',), {}),
    S('-levern-w2-nof1-audit'): (S('-levern-w2-audit'), {}, ('QWEN_FAST_TP4_CONV_GATES_SPREAD', 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'), {}),
    S('-w2-audit-lean'): (S('-w2-audit'), LEAN_KEYS, (), {}),
    S('-w2-nof1-audit-lean'): (S('-w2-nof1-audit'), LEAN_KEYS, (), {}),
    S('-levern-w2-audit-lean'): (S('-levern-w2-audit'), LEAN_KEYS, (), {}),
    S('-levern-w2-nof1-audit-lean'): (S('-levern-w2-nof1-audit'), LEAN_KEYS, (), {}),
    S('-w2-audit-pool'): (S('-w2-audit'), POOL_KEYS, (), POOL_ENGINE),
    S('-w2-nof1-audit-pool'): (S('-w2-nof1-audit'), POOL_KEYS, (), POOL_ENGINE),
    S('-levern-w2-audit-pool'): (S('-levern-w2-audit'), POOL_KEYS, (), POOL_ENGINE),
    S('-levern-w2-nof1-audit-pool'): (S('-levern-w2-nof1-audit'), POOL_KEYS, (), POOL_ENGINE),
    S('-levern-audit-nolna'): (S('-levern-audit'), {}, ('QWEN_FAST_LEVERN_AUDIT',), {}),
    S('-levern-w2-audit-nolna'): (S('-levern-w2-audit'), {}, ('QWEN_FAST_LEVERN_AUDIT',), {}),
    S('-levern-w2-nof1-audit-nolna'): (S('-levern-w2-nof1-audit'), {}, ('QWEN_FAST_LEVERN_AUDIT',), {}),
    S('-levern-epochglobal'): (S('-levern'), {'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'global'}, (), {}),
    S('-levern-audit-epochglobal'): (S('-levern-audit'), {'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'global'}, (), {}),
    S('-levern-w2-epochglobal'): (S('-levern-w2'), {'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'global'}, (), {}),
    S('-levern-w2-audit-epochglobal'): (S('-levern-w2-audit'), {'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'global'}, (), {}),
    S('-levern-w2-audit-sdpa'): (S('-levern-w2'), MA_KEY, (), {}),
    S('-levern-w2-audit-f1'): (S('-levern-w2'), dict(FA_KEY, QWEN_FAST_TP4_VGLUE_AUDIT='1'), (), {}),
    S('-levern-w2-audit-ln'): (S('-levern-w2'), {'QWEN_FAST_LEVERN_AUDIT': '1', 'QWEN_PREFIX_DIGESTS': '1',
                                                  'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT': '1'}, (), {}),
    # the fourth slice: the W1 audits (the shard audits of the verify trace and the draft, reduce-scatter, fused-commit and singles audits) that the other three leave out
    S('-levern-w2-audit-w1'): (S('-levern-w2'), W1_AUDIT_KEYS, (), {}),
    S('-pool'): (SHIP, POOL_KEYS, (), POOL_ENGINE),
    S('-dbf16'): (S('-pool'), {'QWEN_FAST_DRAFTER_BF16': '1'}, (), {}),
    S('-audit-digests'): (S('-audit'), {'QWEN_PREFIX_DIGESTS': '1'}, (), {}),
    S('-dckdefault'): (SHIP, {'QWEN_FAST_DRAFTER_CHECKPOINT': 'dflash2-dedf8df6'}, (), {}),
}

# ---- the swap closure --------------------------------------------------------------------------------------------------------------------------------------------------
# A swap is a one-line C2_PROFILE or C2_PREFIX_PROFILE change in the same image (ORDER.txt, SWAP rules). Every profile a candidate job names must have a committed target
# under every swap that applies to it and under every PAIR of swaps the window allows (F1 with LEAN, POOL or EPOCH; EPOCH with LEAN or POOL; LEAN and POOL never both: two
# fit failures end the candidate for the owner), so a second failure never finds the pack without a target. Names are canonical whatever the order the swaps are applied in:
#   <base> [-nof1 after -w2] [-audit] [-nolna] [-lean | -pool] [-epochglobal]
# and a generated profile is its parent (its name less its last swap token) plus that token's delta, exactly as the hand-written ones are.
EPOCH_KEY = {'QWEN_FAST_LEVERN_EPOCH_SCOPE': 'global'}
SWAP_COMBINATIONS = (('F1',), ('LEAN',), ('POOL',), ('EPOCH',), ('F1', 'LEAN'), ('F1', 'POOL'), ('F1', 'EPOCH'), ('LEAN', 'EPOCH'), ('POOL', 'EPOCH'))
SWAP_BASES = tuple(S(suffix) for suffix in ('-w2', '-w2-audit', '-levern', '-levern-audit', '-levern-audit-nolna', '-levern-w2', '-levern-w2-audit', '-levern-w2-audit-nolna'))


def swap(name, kind):
    """The profile a swap moves `name` to, or None when the swap does not apply to it (F1: W2 profiles that still carry F1; LEAN and POOL: audit twins; EPOCH: Lever N profiles)."""
    if kind == 'F1':
        return name.replace('-w2', '-w2-nof1', 1) if '-w2' in name and '-nof1' not in name else None
    if kind == 'EPOCH':
        return name + '-epochglobal' if '-levern' in name and not name.endswith('-epochglobal') else None
    if kind in ('LEAN', 'POOL') and '-audit' in name and not any(token in name for token in ('-lean', '-pool')):
        token = '-' + kind.lower()
        return name[:-len('-epochglobal')] + token + '-epochglobal' if name.endswith('-epochglobal') else name + token
    return None


def swapped(name, kinds):
    for kind in kinds:
        name = name and swap(name, kind)
    return name


def swap_closure(bases=SWAP_BASES, combinations=SWAP_COMBINATIONS):
    """Every profile reachable from `bases` by one swap or an allowed pair of swaps (the bases themselves excluded)."""
    found = set()
    for base in bases:
        for kinds in combinations:
            target = swapped(base, kinds)
            if target:
                found.add(target)
    return found


def generated(name):
    """(parent, env keys set, env keys removed, engine keys set) of a swap-closure profile that is not hand-written."""
    for token, keys, engine in (('-epochglobal', EPOCH_KEY, {}), ('-lean', LEAN_KEYS, {}), ('-pool', POOL_KEYS, POOL_ENGINE)):
        if name.endswith(token):
            return name[:-len(token)], keys, (), engine
    raise ValueError('%s ends in no swap token' % name)


for _name in sorted(swap_closure(), key=lambda value: (len(value), value)):
    if _name not in SPEC:
        SPEC[_name] = generated(_name)

ALL = tuple(SPEC)
# Carriers of each flag, derived from the spec (a name carries a flag when its own keys, or its parent's, set it).


def effective(name):
    parent, sets, drops, engine = SPEC[name]
    inherited = effective(parent) if parent in SPEC else {}
    env = dict(inherited)
    env.update(sets)
    for key in drops:
        env.pop(key, None)
    return env


def lever_audit(name):
    """Whether the profile carries QWEN_FAST_LEVERN_AUDIT: walk its parents down to the two hand-written Lever N profiles."""
    while name in SPEC:
        parent, sets, drops, engine = SPEC[name]
        if 'QWEN_FAST_LEVERN_AUDIT' in sets:
            return True
        if 'QWEN_FAST_LEVERN_AUDIT' in drops:
            return False
        name = parent
    return name == S('-levern-audit')


def carriers(key, value=None):
    return tuple(name for name in SPEC if key in effective(name) and (value is None or effective(name)[key] == value))


MULTI = carriers('QWEN_FAST_TP4_SDPA', 'multi')
SDPA_AUDIT = carriers('QWEN_FAST_TP4_SDPA_AUDIT')
F1 = carriers('QWEN_FAST_TP4_CONV_GATES_SPREAD')
F1_AUDIT = carriers('QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT')
LEVERN = tuple(name for name in SPEC if '-levern' in name)


def load():
    return json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))


def flat(profile):
    body = dict(profile)
    body.pop('description', None)
    return body


class ProfileTests(unittest.TestCase):
    def test_every_new_profile_is_in_the_file_and_nothing_else_is_new_in_the_family(self):
        found = load()['profiles']
        family = sorted(name for name in found if name.startswith(SHIP))
        known = sorted(name for name in family if name in SPEC)
        self.assertEqual(known, sorted(SPEC))
        older = ['ship-prefix', 'ship-prefix-audit', 'ship-prefix-levern', 'ship-prefix-levern-audit', 'ship-prefix-levern-traffic']
        self.assertEqual(sorted(name for name in family if name not in SPEC), sorted(R + name for name in older))

    def test_each_is_its_parent_plus_exactly_its_keys(self):
        found = load()['profiles']
        for name, (parent, sets, drops, engine) in SPEC.items():
            with self.subTest(name=name):
                want = flat(found[parent])
                want['env'] = dict(want['env'])
                want['env'].update(sets)
                for key in drops:
                    self.assertIn(key, found[parent]['env'], (name, key))
                    want['env'].pop(key)
                want['engine'] = dict(want['engine'])
                want['engine'].update(engine)
                want['gate_only'] = True
                self.assertEqual(flat(found[name]), want)

    def test_all_are_gate_only_with_no_waiver_no_gate_profile_marker_and_no_host_gap_log(self):
        found = load()['profiles']
        for name in SPEC:
            with self.subTest(name=name):
                self.assertIs(found[name]['gate_only'], True)
                for key in ('QWEN_FAST_262K_EVIDENCE_WAIVER', 'QWEN_C2_GATE_PROFILE', 'QWEN_FAST_TP4_HOSTGAP_LOG'):
                    self.assertNotIn(key, found[name]['env'])
                self.assertIn('never traffic', found[name]['description'])
                self.assertIn('docs/tp4-combined-window.md', found[name]['description'])

    def test_the_production_family_and_the_default_are_untouched(self):
        found = load()
        self.assertEqual(found['default'], 'c2-packed-tp4')
        for name in ('ship-prefix', 'ship-prefix-audit'):
            profile = found['profiles'][R + name]
            self.assertNotIn('QWEN_FAST_TP4_SDPA', profile['env'])
            self.assertNotIn('QWEN_FAST_TP4_CONV_GATES_SPREAD', profile['env'])
            self.assertNotIn('QWEN_FAST_LEVER_N', profile['env'])
        self.assertNotIn('gate_only', found['profiles'][SHIP])

    def test_the_window_profiles_keep_the_production_engine_but_the_pool_twins(self):
        found = load()['profiles']
        for name, (parent, sets, drops, engine) in SPEC.items():
            if name in (S('-pool'), S('-dbf16')) or '-pool' in name:
                self.assertEqual(found[name]['engine']['num-gpu-blocks-override'], 19200, name)
                self.assertEqual(found[name]['env']['QWEN36_MAX_TOKENS_ALL_USERS'], '1228288', name)
            else:
                self.assertEqual(found[name]['engine']['num-gpu-blocks-override'], 19968, name)
                self.assertEqual(found[name]['env']['QWEN36_MAX_TOKENS_ALL_USERS'], '1277440', name)
            self.assertEqual(found[name]['engine']['max-model-len'], 262144)
            self.assertEqual(found[name]['engine']['max-num-seqs'], 8)

    def test_the_lever_n_profiles_carry_exactly_the_ten_lever_n_keys_beside_the_parent_family(self):
        found = load()['profiles']
        plain = {key: value for key, value in found[S('-levern')]['env'].items() if key.startswith('QWEN_FAST_LEVER')}
        self.assertEqual(len(plain), 10)
        for name in LEVERN:
            with self.subTest(name=name):
                got = {key: value for key, value in found[name]['env'].items() if key.startswith('QWEN_FAST_LEVER')}
                want = dict(plain)
                if 'epochglobal' in name:
                    want['QWEN_FAST_LEVERN_EPOCH_SCOPE'] = 'global'
                if lever_audit(name):
                    want['QWEN_FAST_LEVERN_AUDIT'] = '1'
                self.assertEqual(got, want)

    def test_the_swap_closure_is_committed_and_every_generated_profile_says_what_it_swaps(self):
        found = load()['profiles']
        closure = swap_closure()
        self.assertGreaterEqual(len(closure), 38)
        for name in sorted(closure):
            with self.subTest(name=name):
                self.assertIn(name, found)
                self.assertIn(name, SPEC)
                self.assertIs(found[name]['gate_only'], True)
        # canonical names: the order of the swaps never matters
        self.assertEqual(swapped(S('-levern-w2-audit-nolna'), ('LEAN', 'EPOCH', 'F1')), swapped(S('-levern-w2-audit-nolna'), ('F1', 'EPOCH', 'LEAN')))
        self.assertEqual(swapped(S('-levern-w2-audit-nolna'), ('F1', 'LEAN', 'EPOCH')), S('-levern-w2-nof1-audit-nolna-lean-epochglobal'))
        self.assertIsNone(swap(S('-w2-audit'), 'EPOCH'), 'no Lever N in a W2-only twin')
        self.assertIsNone(swap(S('-levern-audit'), 'F1'), 'no F1 in a Lever N-only twin')
        self.assertIsNone(swap(S('-levern-w2-audit-lean'), 'POOL'), 'LEAN and POOL are never both applied')
        self.assertIsNone(swap(S('-levern-w2'), 'LEAN'), 'a timed arm carries no audits to drop')

    def test_the_four_split_slices_together_are_exactly_the_whole_audit_set(self):
        # SWAP SPLIT stands in for the combined audited attach only if nothing is left out: the union of the slices' keys is the audit twin's whole delta
        found = load()['profiles']
        whole = {key: value for key, value in found[S('-levern-w2-audit')]['env'].items() if found[S('-levern-w2')]['env'].get(key) != value}
        union = {}
        for suffix in ('-sdpa', '-f1', '-ln', '-w1'):
            parent = found[S('-levern-w2')]['env']
            slice_ = found[S('-levern-w2-audit' + suffix)]['env']
            union.update({key: value for key, value in slice_.items() if parent.get(key) != value})
        self.assertEqual(union, whole)

    def test_the_f1_audit_rides_the_verify_glue_audit_and_multi_audit_rides_multi(self):
        found = load()['profiles']
        for name in F1_AUDIT:
            self.assertEqual(found[name]['env']['QWEN_FAST_TP4_VGLUE_AUDIT'], '1', name)
            self.assertIn(name, F1, name)
        for name in SDPA_AUDIT:
            self.assertEqual(found[name]['env']['QWEN_FAST_TP4_SDPA'], 'multi', name)

    def test_only_the_audit_twins_with_the_sdpa_audit_can_take_an_extent_audited_arm(self):
        # the gates add QWEN_FAST_EXTENT_AUDIT=1 to every non-timed arm; multi refuses it without its own audit (sdpa_multi_tp.attach)
        found = load()['profiles']
        refused = sorted(name for name in MULTI if name not in SDPA_AUDIT)
        for name in refused:
            self.assertNotIn('QWEN_FAST_TP4_SDPA_AUDIT', found[name]['env'], name)
        for name in SDPA_AUDIT:
            self.assertEqual(found[name]['env']['QWEN_FAST_TP4_SDPA'], 'multi', name)


if __name__ == '__main__':
    unittest.main()
