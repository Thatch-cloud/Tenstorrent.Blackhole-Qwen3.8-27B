"""A slot_remap shorter than the GDN slot space must not index out of bounds.

Run 35717866188 (v71) got past the lifecycle refusal - the first run in which two users
ever decoded together - and died in the model:

    gdn/tp.py in remap_slots
        idx = [int(remap[i]) for i in range(self.B)]
    IndexError: index 4 is out of bounds for dimension 0 with size 4

self.B is args.max_batch_size, the full slot space. vLLM sizes slot_remap to the LIVE
batch. The caller's comment in qwen36_vllm.py asserts the opposite - "remap indexes the
full slot space (tokens / GDN)" - so the two sides disagreed, and a remap shorter than B
walked off its end.

Latent, not introduced: a condense only happens once a live decode batch has shrunk or
moved, and until v71 the engine always died earlier (v69 on the lifecycle refusal, v67
and v65 on the scheduled/prepared mismatch). Nothing had ever got two users decoding.

These tests DRIVE the patched method. The fixture is the shipped remap_slots and
_gather_indices verbatim, so the anchor the graft matches on is the real text; the graft
itself raises if that anchor is not found exactly once, which is what protects against
the fixture drifting away from the plugin.
"""

from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).parent))

from lever_n_model_patch import patch_gdn_slot_remap

# remap_slots and _gather_indices as they ship in
# /opt/tt-metal/models/demos/blackhole/qwen36/tt/gdn/tp.py, confirmed against the
# gdn/tp.py.orig captured by graft run 35717866188.
FIXTURE = '''
class GDN:
    def remap_slots(self, remap):
        """Reindex the batched decode state after a vLLM batch condense: slot i takes the state
        previously at slot remap[i] (identity entries are no-ops)."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        self._gather_indices(self.rec_state, idx, dim=0)
        for m in range(self.K):
            self._gather_indices(self.conv_states[m], idx, dim=1)
'''


def build(slots, gathered):
    module = types.ModuleType('patched_gdn')
    exec(compile(patch_gdn_slot_remap(FIXTURE), 'patched_gdn', 'exec'), module.__dict__)
    built = module.GDN()
    built.B = slots
    built.K = 2
    built.rec_state = 'rec'
    built.conv_states = ['conv0', 'conv1']
    built._gather_indices = lambda buf, idx, dim: gathered.append((buf, list(idx), dim))
    return built


class ShortSlotRemapTests(unittest.TestCase):
    def test_a_remap_shorter_than_the_slot_space_does_not_raise(self):
        """The v71 failure, as a unit test: four entries, five slots."""
        gathered = []
        build(slots=5, gathered=gathered).remap_slots([1, 0, 2, 3])
        self.assertTrue(gathered, 'a non-identity remap must still gather')

    def test_the_unmentioned_tail_is_identity(self):
        """Slots past the remap are not live after a condense, so their recurrent and
        conv state is left alone and re-initialised when a request takes them."""
        gathered = []
        build(slots=6, gathered=gathered).remap_slots([1, 0])
        self.assertEqual(gathered[0][1], [1, 0, 2, 3, 4, 5])

    def test_idx_is_padded_not_truncated(self):
        """_gather_indices concats one row per entry and copies in place, so an idx
        shorter than B would build a tensor narrower than the buffer."""
        gathered = []
        built = build(slots=6, gathered=gathered)
        built.remap_slots([1, 0])
        for buf, idx, _dim in gathered:
            self.assertEqual(len(idx), built.B, buf)

    def test_every_buffer_is_still_remapped(self):
        gathered = []
        build(slots=5, gathered=gathered).remap_slots([1, 0, 2, 3])
        self.assertEqual([entry[0] for entry in gathered], ['rec', 'conv0', 'conv1'])
        self.assertEqual([entry[2] for entry in gathered], [0, 1, 1])

    def test_a_full_width_remap_behaves_exactly_as_before(self):
        """The regression guard: the shipped case must be untouched."""
        gathered = []
        build(slots=4, gathered=gathered).remap_slots([3, 2, 1, 0])
        self.assertEqual(gathered[0][1], [3, 2, 1, 0])

    def test_an_identity_remap_still_returns_early(self):
        """Including a SHORT identity remap, which is the common condense."""
        for remap in ([0, 1, 2, 3], [0, 1]):
            gathered = []
            build(slots=4, gathered=gathered).remap_slots(remap)
            self.assertEqual(gathered, [], remap)

    def test_patching_twice_raises(self):
        once = patch_gdn_slot_remap(FIXTURE)
        with self.assertRaisesRegex(ValueError, 'already carries'):
            patch_gdn_slot_remap(once)

    def test_a_source_without_the_anchor_raises(self):
        """What protects the fixture from drifting away from the plugin: the graft job
        fails loudly rather than emitting an unpatched file."""
        with self.assertRaisesRegex(ValueError, 'matched 0 times'):
            patch_gdn_slot_remap('class GDN:' + chr(10) + '    pass' + chr(10))


class TheGraftTableAppliesItTests(unittest.TestCase):
    """patch_gdn_tp must carry this, or the fix never reaches the rig.

    The graft table maps one function per file, so a fix chained into the wrong one is
    silently absent - which is how the M2 alternation was mounted onto a class the
    platform never constructs (run 35707860782) and cost a rig cycle.
    """

    def test_patch_gdn_tp_output_carries_the_padded_index(self):
        import test_lever_n_m3native_patch as m3
        from lever_n_m3native_patch import patch_gdn_tp
        out = patch_gdn_tp(m3.GDN_TP)
        self.assertIn('if i < _qwen_n else i', out)

    def test_the_gdn_fixture_still_contains_the_anchor(self):
        """If the fixture loses remap_slots the graft raises rather than passing, but
        this says so directly instead of as a confusing failure elsewhere."""
        import test_lever_n_m3native_patch as m3
        self.assertIn('def remap_slots', m3.GDN_TP)
        self.assertIn('idx = [int(remap[i]) for i in range(self.B)]', m3.GDN_TP)


if __name__ == '__main__':
    unittest.main()
