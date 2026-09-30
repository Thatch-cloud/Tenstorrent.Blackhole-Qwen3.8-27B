"""The four-card quad (quad_draft_tp.py): one four-user, 64-row draft pass at 8 query / 2 KV heads per chip, in place of the
two packed pair passes, installed through tp_addresses.MODULE_TWINS at QWEN_FAST_TP=4 only.

What is held here, on the CPU (nothing runs on a card; the card-M probe at four-card shapes and the in-model singles audit are
the hardware half, scripts/ci/references/tp4-draft-jobs):

  - the seam: the twin replaces quad_draft in sys.modules at four cards and only there, the pair process never loads it, and the
    pinned quad_draft.py (and the kernel it pins) are byte for byte what they were;
  - the width: heads() is (16, 4, 4, 64, 16) at the pair and (8, 2, 4, 32, 8) at four cards; at the pair every function the twin
    redefines is the pinned function, call for call and bit for bit (its output and its op log);
  - the fold at 8 / 2 heads: folded head 16h + 4u + j is head 4h + j with user u's rows first, and the GQA group of four sends it
    to KV head 4h + u; the quad's rows are the pair fold's and each user's own single-user SDPA bit for bit, on two CPU
    stand-ins (a per-head float32 SDPA and a key-chunk-ordered online-softmax SDPA, whose firing controls show that a changed
    chunk order or a mis-mapped segment changes the bits), for the normal, peaked and quiet regimes and a hundredfold partner;
  - the K/V layout: the twelve-piece plan at two KV heads is the two pair assemblies' bytes (pads from the user's own pair rows);
  - the 64-row helpers are the four-card 32-row twins' calls at 64 rows; the attention and MLP branches run the pair's op
    sequence at 64 rows with per_core_M = 2, the SDPA at 32 query / 8 KV heads over 2080 keys, on a four-chip shape fake;
  - the head and the conv at four chips (two 32,768-column chunks of a 62,080 shard; one program per chip), the readback against
    four singles' rows (every user's features, candidates and scores), a refused block reported per chip;
  - the trace: the bucket owns FOUR users' placeholder banks (1, 2, 2048, 128), copies each user's ACTIVE bank into them every
    round (so a swap of the four-card slide is followed), and its host inputs are the two pairs' uploads row block for row block;
  - the coordinator at four cards: it reaches the twin, refuses what the four-card quad cannot serve (the live-banks flag, a
    (1, 2) mesh, a missing flag) with the reason, and the S2 pooled shapes carry the quad's slots;
  - shipping: both copy lists and the overlay carry the twin and the audit, and the CPU suite runs this file.

    py -3.11 -B -m unittest test_quad_draft_tp4      (from scripts/ci)
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import draft_singles_audit  # noqa: E402
import pair_row_exact  # noqa: E402
import pair_row_exact_tp  # noqa: E402
import quad_draft as pinned  # noqa: E402
import quad_draft_tp as twin  # noqa: E402
import tp_addresses  # noqa: E402
import tp_shapes  # noqa: E402
import test_quad_draft as base  # noqa: E402
from test_pair_row_exact import Device, TorchOps, coded, keep, same_bits  # noqa: E402
from test_dflash_proposal_trace import FakeShard, FakeTensor, RecordingOps  # noqa: E402
from tp_test_support import four_cards, pair  # noqa: E402

CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
FOUR_FLAGS = {'QWEN_FAST_QUAD_DRAFT': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1', 'QWEN_FAST_PAIR_ROW_EXACT': '1',
              'QWEN_FAST_ROUND_B1': '1', 'QWEN_FAST_TP': '4'}
# quad_draft.py and its neighbours at the commit the four-card twin branched from (tp4/speed 9cac8a84): the twin exists so
# that these bytes never move.
PINNED_SHA256 = {'quad_draft.py': 'f131a1f35780c939630546fe66ee0e8a55935f5257ff25e884c912b9a812156c',
                 'fused_commit.py': '2a434fb71af4d710c710a636c4ec10636b7c439264fa21f416a9f983a1f5eab8',
                 'quad_conv_io.cpp': '8c4e8f93f8ced3a3c8ef8f551b8f711e5fdc03e7bcd0aeeade3e15ee94e0492b',
                 'pair_row_exact.py': 'b3e31fde9682bf0b56443a8ffb38812a2c3195ccfc9bb9a715cc3d419f6ba34d'}
TWIN_OWN = {'heads', 'missing_requirements', 'note', 'validate_quad', 'quad_fold_query', 'quad_fold_keys',
            'quad_unfold_output', 'fold_attention', 'pairs_attention', 'quad_head_map', 'split_projected_heads',
            'project_key_value', 'concatenate_query_heads', 'quad_fused_convolution', 'halves_convolution',
            'head_candidates', 'QuadPass', 'read_quad_outputs', 'select_quad_outputs', 'PreparedQuadDFlashProposal',
            'refusal', '_tiled_dram'}


def clean_environment(**flags):
    """No QWEN_FAST_* flag but the ones given."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environment.update(flags)
    return patch.dict(os.environ, environment, clear=True)


# ---------------------------------------------------------------------------------------------
# The seam and the width.
# ---------------------------------------------------------------------------------------------

class SeamTests(unittest.TestCase):
    def test_the_seam_names_the_twin_and_installs_it_at_four_cards_only(self):
        self.assertIn(('quad_draft', 'quad_draft_tp'), tp_addresses.MODULE_TWINS)
        served = sys.modules['quad_draft']
        self.assertIs(served, pinned)
        with four_cards():
            self.assertIs(sys.modules['quad_draft'], twin)
            import quad_draft as reached

            self.assertIs(reached, twin, 'the coordinator imports it lazily, so it reaches the twin')
        self.assertIs(sys.modules['quad_draft'], pinned)
        with pair(), self.assertRaises(ValueError):
            tp_addresses.install()
        self.assertIs(sys.modules['quad_draft'], pinned)

    def test_the_pair_process_never_loads_the_twin(self):
        code = ('import sys; sys.path.insert(0, %r); import tp_addresses, quad_draft, dflash_packed_proposal_coordinator; '
                "assert 'quad_draft_tp' not in sys.modules, 'the pair loaded the four-card quad'" % str(HERE))
        environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_')}
        subprocess.run([sys.executable, '-B', '-c', code], env=environment, check=True, timeout=120)

    def test_the_pinned_quad_and_its_neighbours_are_the_bytes_they_were(self):
        for name, digest in PINNED_SHA256.items():
            data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest, '%s moved: the twin exists so it never does' % name)

    def test_the_twin_redefines_only_the_width_dependent_names_and_shares_the_rest(self):
        own = {name for name, value in vars(twin).items() if not name.startswith('__')
               and getattr(value, '__module__', None) == 'quad_draft_tp'}
        self.assertEqual(own, TWIN_OWN)
        for name in dir(pinned):
            if not name.startswith('__') and name not in TWIN_OWN | {'REQUIRED_FLAGS'}:
                with self.subTest(name=name):
                    self.assertIs(getattr(twin, name), getattr(pinned, name), 'the pinned module\'s own object')
        with patch.object(pinned, '_NOTED', ['seen']):
            self.assertIs(twin._NOTED, pinned._NOTED, 'one marker per process, whichever module logs it')
            self.assertFalse(twin.note([0, 1, 2, 3], 'fold', '110', log=lambda line: None))
        with self.assertRaises(AttributeError):
            twin.no_such_name

    def test_the_twin_has_no_import_time_width(self):
        """Read at each call: the same module object serves a pair-width test and a four-card one."""
        with pair():
            self.assertEqual(twin.heads(), (16, 4, 4, 64, 16))
        with four_cards():
            self.assertEqual(twin.heads(), (8, 2, 4, 32, 8))
        with pair():
            self.assertEqual(twin.heads(), (16, 4, 4, 64, 16))


class WidthTests(unittest.TestCase):
    def test_the_head_map_at_the_pair_is_the_pinned_one_and_at_four_cards_is_the_32_over_8_fold(self):
        with pair():
            self.assertEqual(twin.quad_head_map(), pinned.quad_head_map())
        with four_cards():
            head_map = twin.quad_head_map()
            self.assertEqual(len(head_map), 32)
            for folded, (h, u, j, kv) in head_map.items():
                self.assertEqual(folded, 16 * h + 4 * u + j)
                self.assertEqual(kv, 4 * h + u)
                self.assertEqual(folded // 4, kv, 'GQA group 32 / 8 = 4 sends the folded head to its KV head')
            self.assertEqual({kv for *_, kv in head_map.values()}, set(range(8)))

    def test_the_marker_names_the_width_and_is_logged_once(self):
        lines = []
        with four_cards(), patch.object(pinned, '_NOTED', []):
            self.assertTrue(twin.note([0, 1, 2, 3], 'fold', '110', log=lines.append))
            self.assertFalse(twin.note([0, 1, 2, 3], 'fold', '110', log=lines.append))
        with four_cards(), patch.object(pinned, '_NOTED', []):
            twin.note([0, 1, 2, 3], 'pairs', 'halves', log=lines.append)
        with pair(), patch.object(pinned, '_NOTED', []):
            twin.note([0, 1, 2, 3], 'fold', '110', log=lines.append)
        self.assertEqual(lines, ['[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=32/8 rows=64 sdpa=fold conv=110',
                                 '[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=16/4x2 rows=64 sdpa=pairs conv=halves',
                                 '[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=64/16 rows=64 sdpa=fold conv=110'])
        self.assertRegex(lines[0], base.quad_draft.MARKER.replace('[', r'\[').replace(']', r'\]'))

    def test_the_four_card_requirements_are_three_flags_and_no_live_banks(self):
        self.assertEqual(twin.REQUIRED_FLAGS, ('QWEN_FAST_PACKED_PROPOSAL', 'QWEN_FAST_PAIR_ROW_EXACT', 'QWEN_FAST_ROUND_B1'))
        self.assertEqual(twin.missing_requirements(FOUR_FLAGS), [])
        self.assertEqual(twin.missing_requirements(dict(FOUR_FLAGS, QWEN_FAST_ROUND_B1='0')), ['QWEN_FAST_ROUND_B1'])
        self.assertEqual(twin.missing_requirements({}), list(twin.REQUIRED_FLAGS))
        self.assertNotIn('QWEN_FAST_FUSED_COMMIT_LIVE_BANKS', twin.REQUIRED_FLAGS)


# ---------------------------------------------------------------------------------------------
# The fold at 8 / 2 heads.
# ---------------------------------------------------------------------------------------------

class Fixture:
    """Four users' operands at `heads` query and `kv` KV heads per chip, built three ways: as the quad assembles them (one
    64-row live block, the twelve-piece plan), as each pair assembles them (its own 32-row block, the pair plan) and as each
    user's single-user trace does (its own live block with pad rows, its own pad query rows)."""

    def __init__(self, heads, kv, seed=0, *, scale=1.0):
        from dflash_batched_mask import key_value_plan

        generator = torch.Generator().manual_seed(seed)

        def normal(*shape, factor=1.0):
            return (torch.randn(*shape, generator=generator) * factor).bfloat16()

        self.heads, self.kv = heads, kv
        self.cache = [{name: normal(1, kv, 2048, 128) for name in 'kv'} for _ in range(4)]
        self.live = [{name: normal(1, kv, 16, 128) for name in 'kv'} for _ in range(4)]
        self.live_pad = [{name: normal(1, kv, 16, 128) for name in 'kv'} for _ in range(4)]
        self.query = [normal(1, heads, 16, 128, factor=scale) for _ in range(4)]
        self.query_pad = [normal(1, heads, 16, 128, factor=scale) for _ in range(4)]
        self.rebuild()
        self.pair_plan = key_value_plan([2048, 2048], 16)[0]

    def rebuild(self):
        self.block = {name: torch.cat([self.live[user][name] for user in range(4)], dim=2) for name in 'kv'}
        self.quad_query = torch.cat(self.query, dim=2)
        self.quad_keys = self.assemble(pinned.quad_key_value_plan(), self.block, self.cache)
        self.pairs = []
        for index in range(2):
            users = (2 * index, 2 * index + 1)
            block = {name: torch.cat([self.live[user][name] for user in users], dim=2) for name in 'kv'}
            self.pairs.append(dict(query=torch.cat([self.query[user] for user in users], dim=2),
                                   keys=self.assemble(self.pair_plan_for(), block, [self.cache[user] for user in users])))

    @staticmethod
    def pair_plan_for():
        from dflash_batched_mask import key_value_plan

        return key_value_plan([2048, 2048], 16)[0]

    @staticmethod
    def assemble(plan, block, caches):
        """The branch's assembly loop on the host: a pad without a 'source' starts at row 0 (the pair's rule)."""
        out = {}
        for name in 'kv':
            pieces = []
            for part in plan:
                if part['kind'] == 'cached':
                    pieces.append(caches[part['user']][name])
                    continue
                start = part['source'].start if 'source' in part else 0
                pieces.append(block[name][:, :, start:start + part['rows']])
            out[name] = torch.cat(pieces, dim=2)
        return out

    def single(self, user):
        keys = {name: torch.cat([self.cache[user][name], self.live[user][name], self.live_pad[user][name]], dim=2)
                for name in 'kv'}
        return torch.cat([self.query[user], self.query_pad[user]], dim=2), keys['k'], keys['v']


class ChunkedOps(TorchOps):
    """TorchOps with the SDPA computed the way the kernel does: per head, key chunks of `chunk` in order, an online softmax
    (a running max, sum and output rescaled at each chunk, float32) - so a unit's bits depend on its chunk sequence, which is
    what the fold has to keep. A fully masked chunk leaves the state as it is."""

    chunk = 32

    def sdpa(self, query, key, value, *, attn_mask, is_causal, scale, program_config, compute_kernel_config,
             memory_config):
        self.calls.append(('sdpa', query.shape, key.shape, attn_mask.shape))
        q, k, v, mask = query.value, key.value, value.value, attn_mask.value
        heads, kv_heads = q.shape[1], k.shape[1]
        rows = []
        for head in range(heads):
            kv = head // (heads // kv_heads)
            rows.append(self.head(q[0, head].float(), k[0, kv].float(), v[0, kv].float(),
                                  mask[0, 0 if mask.shape[1] == 1 else head].float(), scale))
        return Device(torch.stack(rows)[None].bfloat16())

    def head(self, q, k, v, mask, scale):
        inf = float('inf')
        running = torch.full((q.shape[0],), -inf)
        total = torch.zeros(q.shape[0])
        out = torch.zeros(q.shape[0], v.shape[1])
        for start in range(0, k.shape[0], self.chunk):
            stop = start + self.chunk
            scores = q @ k[start:stop].T * scale + mask[:, start:stop]
            new = torch.maximum(running, scores.max(dim=-1).values)
            live = torch.isfinite(new)
            factor = torch.where(live, torch.exp(running - new), torch.ones_like(new))
            factor = torch.where(running == -inf, torch.zeros_like(factor), factor)
            weights = torch.where(live[:, None], torch.exp(scores - torch.where(live, new, torch.zeros_like(new))[:, None]),
                                  torch.zeros_like(scores))
            total = total * factor + weights.sum(dim=-1)
            out = out * factor[:, None] + weights @ v[start:stop]
            running = new
        return out / total[:, None]


def single_rows(operations, fixture, user, mask):
    query, keys, values = fixture.single(user)
    return operations.sdpa(Device(query), Device(keys), Device(values), attn_mask=Device(mask), is_causal=False,
                           scale=128 ** -0.5, program_config=None, compute_kernel_config=None,
                           memory_config='dram').value[:, :, :16]


def fold_mask():
    return pair_row_exact.fold_mask((2048, 2048), 16)


class FoldMappingTests(unittest.TestCase):
    def test_folded_query_head_16h_4u_j_is_head_4h_j_with_user_u_first_in_its_pairs_half(self):
        with four_cards():
            operations, owned = TorchOps(), []
            query = coded(8, 64)
            folded = twin.quad_fold_query(operations, Device(query), keep(owned)).value
            self.assertEqual(tuple(folded.shape), (1, 32, 32, 128))
            for h in range(2):
                for u in range(4):
                    for j in range(4):
                        half = query[0, 4 * h + j, 32 * (u // 2):32 * (u // 2) + 32]
                        expected = half if u % 2 == 0 else torch.cat([half[16:], half[:16]])
                        self.assertTrue(torch.equal(folded[0, 16 * h + 4 * u + j], expected), (h, u, j))

    def test_folded_kv_head_4h_u_is_user_us_segment_of_head_h(self):
        with four_cards():
            operations, owned = TorchOps(), []
            keys = coded(2, 8320)
            folded = twin.quad_fold_keys(operations, Device(keys), keep(owned)).value
            self.assertEqual(tuple(folded.shape), (1, 8, 2080, 128))
            for h in range(2):
                for u in range(4):
                    self.assertTrue(torch.equal(folded[0, 4 * h + u], keys[0, h, 2080 * u:2080 * (u + 1)]))
            self.assertEqual(operations.calls, [('reshape', (1, 2, 8320, 128), (1, 8, 2080, 128))], 'a view, no copy')

    def test_the_unfold_returns_every_users_rows_to_its_block_rows(self):
        with four_cards():
            operations, owned = TorchOps(), []
            output = coded(32, 32)
            unfolded = twin.quad_unfold_output(operations, Device(output), keep(owned)).value
            self.assertEqual(tuple(unfolded.shape), (1, 8, 64, 128))
            for h in range(2):
                for j in range(4):
                    for u in range(4):
                        self.assertTrue(torch.equal(unfolded[0, 4 * h + j, 16 * u:16 * u + 16],
                                                    output[0, 16 * h + 4 * u + j, :16]), (h, j, u))

    def test_fold_then_unfold_of_an_identity_attention_is_the_query(self):
        with four_cards():
            operations, owned = TorchOps(), []
            query = coded(8, 64)
            folded = twin.quad_fold_query(operations, Device(query), keep(owned))
            self.assertTrue(torch.equal(twin.quad_unfold_output(operations, folded, keep(owned)).value, query))

    def test_the_fold_is_built_from_the_pair_folds_own_functions(self):
        fixture = Fixture(8, 2)
        with four_cards():
            operations, owned = TorchOps(), []
            with patch('pair_row_exact_tp.fold_query', side_effect=pair_row_exact_tp.fold_query) as fold_query, \
                    patch('pair_row_exact_tp.unfold_output', side_effect=pair_row_exact_tp.unfold_output) as unfold, \
                    patch('pair_row_exact.folded_sdpa', side_effect=pair_row_exact.folded_sdpa) as sdpa:
                twin.fold_attention(operations, Device(fixture.quad_query), Device(fixture.quad_keys['k']),
                                    Device(fixture.quad_keys['v']), Device(fold_mask()), keep(owned), mask_validated=True)
            self.assertEqual((fold_query.call_count, unfold.call_count, sdpa.call_count), (2, 2, 1))
            self.assertEqual([call for call in operations.calls if call[0] == 'sdpa'],
                             [('sdpa', (1, 32, 32, 128), (1, 8, 2080, 128), (1, 1, 32, 2080))])

    def test_the_operands_are_validated_against_the_width(self):
        fixture = Fixture(8, 2, 1)
        with four_cards():
            operations, owned = TorchOps(), []
            query, keys, values = (Device(fixture.quad_query), Device(fixture.quad_keys['k']),
                                   Device(fixture.quad_keys['v']))
            mask = Device(fold_mask())
            with self.assertRaises(ValueError):
                twin.fold_attention(operations, query, keys, values, mask, keep(owned))
            wide = Fixture(16, 4, 1)
            cases = [(Device(wide.quad_query), Device(wide.quad_keys['k']), Device(wide.quad_keys['v']), mask),
                     (Device(fixture.quad_query[:, :, :32]), keys, values, mask),
                     (query, Device(fixture.quad_keys['k'][:, :, :4160]), values, mask),
                     (query, keys, values, Device(torch.zeros(1, 1, 32, 4160))),
                     (Device(fixture.quad_query, dtype='fp32'), keys, values, mask),
                     (query, Device(fixture.quad_keys['k'], layout='row'), values, mask),
                     (query, keys, values, Device(mask.value, memory='l1'))]
            for index, operands in enumerate(cases):
                with self.subTest(case=index), self.assertRaises(ValueError):
                    twin.validate_quad(operations, *operands)
            self.assertEqual(operations.calls, [])
        with pair():
            twin.validate_quad(TorchOps(), Device(wide.quad_query), Device(wide.quad_keys['k']),
                               Device(wide.quad_keys['v']), mask)


class FoldEqualsPairAndSingleTests(unittest.TestCase):
    """Every quad work unit is its pair-fold unit and its single-user unit: at 8 / 2 heads, bit for bit."""

    OPS = (TorchOps, ChunkedOps)

    def quad(self, ops, fixture, attend=None):
        owned = []
        return (attend or twin.fold_attention)(ops, Device(fixture.quad_query), Device(fixture.quad_keys['k']),
                                               Device(fixture.quad_keys['v']), Device(fold_mask()), keep(owned),
                                               mask_validated=True).value

    def pair(self, ops, fixture, index):
        owned = []
        pair_data = fixture.pairs[index]
        return pair_row_exact.fold_attention(ops, Device(pair_data['query']), Device(pair_data['keys']['k']),
                                             Device(pair_data['keys']['v']), Device(fold_mask()), keep(owned),
                                             mask_validated=True).value

    def test_every_users_rows_equal_the_pair_fold_and_the_single_user_bit_for_bit(self):
        for factory in self.OPS:
            for seed, scale in ((0, 1.0), (1, 4.0), (2, 0.25)):
                with self.subTest(ops=factory.__name__, seed=seed, scale=scale), four_cards():
                    fixture = Fixture(8, 2, seed, scale=scale)
                    ops = factory()
                    quad = self.quad(ops, fixture)
                    self.assertEqual(tuple(quad.shape), (1, 8, 64, 128))
                    pairs = [self.pair(factory(), fixture, index) for index in range(2)]
                    for user in range(4):
                        rows = quad[:, :, 16 * user:16 * user + 16]
                        paired = pairs[user // 2][:, :, 16 * (user % 2):16 * (user % 2) + 16]
                        self.assertTrue(same_bits(rows, paired), 'user %d against its pair fold' % user)
                        alone = single_rows(factory(), fixture, user, fold_mask())
                        self.assertTrue(same_bits(rows, alone), 'user %d against drafting alone' % user)

    def test_the_pairs_mode_gives_the_same_bits(self):
        with four_cards():
            fixture = Fixture(8, 2, 3)
            for factory in self.OPS:
                self.assertTrue(same_bits(self.quad(factory(), fixture, twin.pairs_attention),
                                          self.quad(factory(), fixture)), factory.__name__)
            ops = TorchOps()
            self.quad(ops, fixture, twin.pairs_attention)
            self.assertEqual([call for call in ops.calls if call[0] == 'sdpa'],
                             [('sdpa', (1, 16, 32, 128), (1, 4, 2080, 128), (1, 1, 32, 2080))] * 2)

    def test_a_hundredfold_partner_changes_nothing(self):
        with four_cards():
            fixture, loud = Fixture(8, 2, 5), Fixture(8, 2, 5)
            for user in (0, 2):
                for name in 'kv':
                    for part in ('cache', 'live'):
                        getattr(loud, part)[user][name] = (getattr(loud, part)[user][name].float() * 100).bfloat16()
            loud.rebuild()
            for factory in self.OPS:
                quad, louder = self.quad(factory(), fixture), self.quad(factory(), loud)
                for user in (1, 3):
                    rows = slice(16 * user, 16 * user + 16)
                    self.assertTrue(same_bits(louder[:, :, rows], quad[:, :, rows]), factory.__name__)
                self.assertTrue(same_bits(louder[:, :, 16:32], single_rows(factory(), fixture, 1, fold_mask())))

    def test_firing_controls_the_emulation_notices_a_changed_chunk_order_and_a_mis_mapped_segment(self):
        """The equalities above mean something only if the stand-in can tell a wrong computation from a right one."""
        with four_cards():
            fixture = Fixture(8, 2, 6, scale=4.0)
            good = self.quad(ChunkedOps(), fixture)
            wide = ChunkedOps()
            wide.chunk = 64
            self.assertFalse(same_bits(self.quad(wide, fixture), good), 'a different key-chunk size changes the bits')
            # a fold that hands user u the next user's segment (a wrong KV head map) is not that user's single
            operations, owned = ChunkedOps(), []
            folded = twin.quad_fold_query(operations, Device(fixture.quad_query), keep(owned))
            keys = twin.quad_fold_keys(operations, Device(fixture.quad_keys['k']), keep(owned)).value
            values = twin.quad_fold_keys(operations, Device(fixture.quad_keys['v']), keep(owned)).value
            shifted_keys = torch.roll(keys.reshape(1, 2, 4, 2080, 128), 1, dims=2).reshape(1, 8, 2080, 128)
            shifted_values = torch.roll(values.reshape(1, 2, 4, 2080, 128), 1, dims=2).reshape(1, 8, 2080, 128)
            wrong = twin.quad_unfold_output(operations, operations.sdpa(
                folded, Device(shifted_keys), Device(shifted_values), attn_mask=Device(fold_mask()), is_causal=False,
                scale=128 ** -0.5, program_config=None, compute_kernel_config=None, memory_config=None), keep(owned)).value
            alone = single_rows(ChunkedOps(), fixture, 1, fold_mask())
            self.assertFalse(same_bits(wrong[:, :, 16:32], alone), 'a mis-mapped segment is caught')
            self.assertTrue(same_bits(good[:, :, 16:32], alone))


class KeyValueLayoutTests(unittest.TestCase):
    def test_the_twelve_pieces_at_two_kv_heads_are_the_two_pair_assemblies_bytes(self):
        with four_cards():
            fixture = Fixture(8, 2, 4)
            for name in 'kv':
                expected = torch.cat([fixture.pairs[0]['keys'][name], fixture.pairs[1]['keys'][name]], dim=2)
                self.assertEqual(tuple(expected.shape), (1, 2, 8320, 128))
                self.assertTrue(same_bits(fixture.quad_keys[name], expected), name)
            plan, spans, key_rows = twin.key_value_plan([2048] * 4, 16)
            self.assertEqual(key_rows, 8320)
            self.assertEqual(len(plan), 12)
            self.assertEqual([piece['source'].start for piece in plan if piece['kind'] == 'pad'], [0, 0, 32, 32])
            self.assertEqual([(span['rows'].start, span['rows'].stop) for span in spans],
                             [(0, 16), (16, 32), (32, 48), (48, 64)])
            today = [{key: value for key, value in part.items() if key != 'source'} if part['kind'] == 'pad' else part
                     for part in plan]
            wrong = Fixture.assemble(today, fixture.block, fixture.cache)
            self.assertFalse(same_bits(wrong['k'], fixture.quad_keys['k']), 'every pad from row 0 would differ')
            for contexts, rows in (([2048] * 3, 16), ([2048] * 4, 8), ([2048, 2048, 2048, 1024], 16)):
                with self.subTest(contexts=contexts), self.assertRaises(ValueError):
                    twin.key_value_plan(contexts, rows)

    def test_the_quad_host_mask_is_the_single_user_mask(self):
        with four_cards():
            mask = twin.quad_host_mask()
            self.assertEqual(tuple(mask.shape), (1, 1, 32, 2080))
            self.assertTrue(same_bits(mask, fold_mask()))


# ---------------------------------------------------------------------------------------------
# Parity with the pinned functions at the pair.
# ---------------------------------------------------------------------------------------------

class PairParityTests(unittest.TestCase):
    """At two chips every function the twin redefines is the pinned one, on the CPU stand-ins: same bits, same calls."""

    def test_the_fold_functions(self):
        with pair():
            fixture = Fixture(16, 4, 7)
            mask = fold_mask()
            for name in ('fold_attention', 'pairs_attention'):
                results = []
                for module in (pinned, twin):
                    ops, owned = TorchOps(), []
                    output = getattr(module, name)(ops, Device(fixture.quad_query), Device(fixture.quad_keys['k']),
                                                   Device(fixture.quad_keys['v']), Device(mask), keep(owned),
                                                   mask_validated=True).value
                    results.append((output, ops.calls))
                with self.subTest(function=name):
                    self.assertTrue(same_bits(results[0][0], results[1][0]))
                    self.assertEqual(results[0][1], results[1][1])
            for name, args in (('quad_fold_query', (coded(16, 64),)), ('quad_fold_keys', (coded(4, 8320),)),
                               ('quad_unfold_output', (coded(64, 32),))):
                results = []
                for module in (pinned, twin):
                    ops, owned = TorchOps(), []
                    output = getattr(module, name)(ops, Device(args[0]), keep(owned)).value
                    results.append((output, ops.calls))
                with self.subTest(function=name):
                    self.assertTrue(torch.equal(results[0][0], results[1][0]))
                    self.assertEqual(results[0][1], results[1][1])

    def test_the_helpers_and_the_readback(self):
        with pair(), patch('mesh_link_policy.projection_links', return_value=1):
            for name, run in (('helpers', self.helpers),):
                self.assertEqual(run(pinned), run(twin), name)
            self.assertEqual(self.head(pinned), self.head(twin))

    @staticmethod
    def helpers(module):
        ops = base.ShapeOps()
        owned = []
        retain = lambda value: owned.append(value) or value
        rows = 64
        parameters = dict(operations=ops, native_head_layout=True, kernel='kernel',
                          projections=dict(k=base.Tensor((5120, 512)), v=base.Tensor((5120, 512))),
                          head_norms=dict(k=base.Tensor((1, 1, 4, 32))))
        mesh = SimpleNamespace(shape=[1, 2])
        module.project_key_value(ops, base.Tensor((1, 1, rows, 5120)), base.Tensor((1, 1, rows, 2048)),
                                 (base.Tensor((1, 1, rows, 128)), base.Tensor((1, 1, rows, 128))), retain,
                                 parameters=parameters)
        module.split_projected_heads(ops, base.Tensor((1, 1, rows, 2048)), base.Tensor((1, 1, rows, 512)),
                                     base.Tensor((1, 1, rows, 512)), retain)
        module.concatenate_query_heads(ops, base.Tensor((1, 16, rows, 128)), retain)
        return ops.events

    @staticmethod
    def head(module):
        ops = base.ShapeOps()
        owned = []
        model = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight='head')
        chunks = module.head_candidates(ops, model, base.Tensor((1, 1, 64, 5120)), owned,
                                        lambda value: owned.append(value) or value)
        return ops.events, [(chunk['start'], chunk['stop'], chunk['values'].shape) for chunk in chunks]

    def test_the_readback(self):
        generator = torch.Generator().manual_seed(11)
        pairs = [base.pair_outputs(generator) for _ in range(2)]
        quad_chunks = []
        for number in range(4):
            first, second = pairs[0][0][number], pairs[1][0][number]
            quad_chunks.append(dict(start=first['start'], stop=first['stop'],
                values=base.chips(*(torch.cat([first['values'][chip], second['values'][chip]], dim=2) for chip in range(2))),
                indices=base.chips(*(torch.cat([first['indices'][chip], second['indices'][chip]], dim=2)
                                     for chip in range(2)))))
        projected = torch.cat([pairs[0][1], pairs[1][1]], dim=2)
        outputs = SimpleNamespace(chunks=quad_chunks, projected=base.chips(projected, projected.clone()))
        device = SimpleNamespace(operations=base.HostOps())
        with pair():
            mine, theirs = twin.read_quad_outputs(device, outputs), pinned.read_quad_outputs(device, outputs)
        for user in range(4):
            for key in ('hidden', 'candidates', 'unary'):
                self.assertTrue(torch.equal(mine[user][key], theirs[user][key]) and mine[user][key].dtype == theirs[user][key].dtype)

    def test_the_conv_program(self):
        output = base.Tensor((1, 1, 64, 5120))

        class Ops(base.DescriptorOps):
            def empty(self, shape, **options):
                return output

        hidden, dynamic, base_kernels = base.conv_operands()
        programs = []
        with pair(), patch.object(pinned, '_KERNEL_CHECKED', []):
            for module in (pinned, twin):
                ops = Ops()
                module.quad_fused_convolution(ops, base.MESH, hidden, dynamic, base_kernels, boundaries=base.QUAD_SEAMS)
                programs.append(ops.calls[0][1])
        self.assertEqual(programs[0], programs[1])

    def test_the_refusal_is_a_four_card_refusal_only(self):
        self.assertIn('QWEN_FAST_TP=4 only', twin.refusal([], [], {}))
        self.assertIn('QWEN_FAST_TP=4 only', twin.refusal([], [], {'QWEN_FAST_TP': '2'}))


# ---------------------------------------------------------------------------------------------
# A four-chip shape fake for the branches, the head and the conv.
# ---------------------------------------------------------------------------------------------

class ShapeOps4(base.ShapeOps):
    """test_quad_draft's shape-tracking ops on `chips` chips."""

    def __init__(self, chips=4):
        super().__init__()
        self.chips = chips

    def get_device_tensors(self, tensor):
        return [base.Shard(tensor.address + chip) for chip in range(self.chips)]

    def all_gather(self, value, **options):
        shape = list(value.shape)
        shape[options['dim']] *= self.chips
        self.log('all_gather', value.shape, options['dim'])
        return base.Tensor(shape, value.dtype)

    def linear(self, left, right, **options):
        self.log('linear', left.shape)
        return base.Tensor(left.shape[:-1] + (tp_shapes.vocab_shard(),))


class DescriptorOps4(base.DescriptorOps):
    def get_device_tensors(self, tensor):
        return [base.Shard(tensor.address + chip) for chip in range(4)]


MESH4 = SimpleNamespace(shape=[1, 4], compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
COLLECTIVES = base.COLLECTIVES


class RecordingQuad(twin.QuadPass):
    """The twin's pass with the conv recorded (the E1b program has its own test below)."""

    def __init__(self, ops, **options):
        super().__init__(**options)
        self.ops = ops

    def convolution(self, operations, mesh, hidden, dynamic, base_kernels, **options):
        return base.recording_convolution(self.ops)(operations, mesh, hidden, dynamic, base_kernels, **options)


def attention_run4(*, quad, users=None, sdpa='fold', chips=4):
    """execute_attention_branch at the width of this process on a `chips`-chip shape fake: the packed pair (folded) or the
    twin's quad. Returns the event log."""
    import draft_attention_branch

    found = tp_shapes.active()
    ops = ShapeOps4(chips)
    users = users or (4 if quad else 2)
    rows = 16 * users if quad else 32
    rope = {'q': (base.Tensor((1, 1, rows, 128)), base.Tensor((1, 1, rows, 128))),
            'live_k': (base.Tensor((1, 1, rows, 128)), base.Tensor((1, 1, rows, 128)))}
    if not quad:
        rope['k'] = (base.Tensor((1, 1, 4160, 128)), base.Tensor((1, 1, 4160, 128)))
    caches = [{name: base.Tensor((1, found.draft_kv_heads, 2048, 128)) for name in 'kv'} for _ in range(users)]
    pack = [dict(position=4096 + 1000 * user, history_rows=2048) for user in range(users)]
    mask = base.Tensor((1, 1, 32, 2080))
    mesh = SimpleNamespace(shape=[1, chips])
    parameters = dict(operations=ops, mesh=mesh, block_rows=16, native_head_layout=True, native_proposal_attention=True,
                      kernel='kernel', norm=base.Tensor((1, 1, 160, 32)), convolution=base.Tensor((5120, 1280)),
                      bases=[base.Tensor((1, 1, 1, 5120)) for _ in range(4)],
                      projections=dict(q=base.Tensor((5120, found.draft_query)),
                                       k=base.Tensor((5120, found.draft_kv_heads * 128)),
                                       v=base.Tensor((5120, found.draft_kv_heads * 128))),
                      head_norms=dict(q=base.Tensor((1, 1, 4, 32)), k=base.Tensor((1, 1, 4, 32))),
                      output_projection=base.Tensor((found.draft_query, 5120)))
    extra = {}
    if quad:
        extra['quad'] = RecordingQuad(ops, sdpa=sdpa)
        convolve = extra['quad'].convolution
    else:
        convolve = base.recording_convolution(ops)
        extra['row_exact'] = True
    owned = []
    with patch('dflash_t16_native_scope.require_active', return_value=None), \
            patch('mesh_link_policy.projection_links', return_value=1), \
            patch('feature_collective_tp.projection_links', return_value=1), \
            patch('feature_collective.projection_links', return_value=1):
        output = draft_attention_branch.execute_attention_branch(ops, mesh, COLLECTIVES,
            base.Tensor((1, 1, rows, 5120)), None, mask, rope, lambda value: owned.append(value) or value,
            parameters=parameters, context=None, pack=pack, convolution_operation=convolve, cached_history=caches,
            native_proposal_mask_validated=True, **extra)
    ops.log('output', output.shape)
    return ops.events


def mlp_run4(*, quad, chips=4):
    import draft_mlp_branch

    ops = ShapeOps4(chips)
    rows = 64 if quad else 32
    weights, convolution = object(), object()
    mlp = tp_shapes.geometry(chips).mlp
    parameters = dict(operations=ops, mesh=SimpleNamespace(shape=[1, chips]), source_weights=weights,
                      source_convolution=convolution, kernel='kernel', device_norm=base.Tensor((1, 1, 160, 32)),
                      device_conv=base.Tensor((5120, 1280)), bases=[base.Tensor((1, 1, 1, 5120)) for _ in range(4)],
                      device_projections=[base.Tensor((5120, mlp)),
                                          base.Tensor((5120, mlp)), base.Tensor((mlp, 5120))],
                      shards='shards', norm_weight='norm', conv_weight='conv', base_weight='base')
    seams = tuple((start, start + 16) for start in range(0, rows, 16))
    extra = dict(quad=RecordingQuad(ops)) if quad else {}
    convolve = extra['quad'].convolution if quad else base.recording_convolution(ops)
    owned = []
    with patch('mesh_link_policy.projection_links', return_value=1), \
            patch('feature_collective_tp.projection_links', return_value=1), \
            patch('feature_collective.projection_links', return_value=1):
        output = draft_mlp_branch.execute_mlp_branch(ops, parameters['mesh'], COLLECTIVES, base.Tensor((1, 1, rows, 5120)),
            weights, convolution, lambda value: owned.append(value) or value, parameters=parameters, trace_safe=True,
            convolution_operation=convolve, boundaries=seams, **extra)
    ops.log('output', output['output'].shape)
    return ops.events


class BranchTests(unittest.TestCase):
    def test_the_attention_branch_at_four_cards_is_the_pairs_op_sequence_at_64_rows(self):
        with four_cards():
            pair_events = attention_run4(quad=False)
            quad = attention_run4(quad=True)
        names = lambda events: [event[0] for event in events if event[0] in base.CORE_OPS]
        self.assertEqual(names(quad), names(pair_events))
        programs = [event[1] for event in pair_events if event[0] == 'program']
        quad_programs = [event[1] for event in quad if event[0] == 'program']
        self.assertEqual([dict(item)['compute_with_storage_grid_size'] for item in programs],
                         [(8, 5), (8, 8), (8, 8), (8, 10)])
        self.assertEqual([dict(item)['per_core_M'] for item in programs], [1] * 4)
        self.assertEqual([dict(item)['per_core_M'] for item in quad_programs], [2] * 4)
        strip = lambda items: [tuple(pair_ for pair_ in item if pair_[0] != 'per_core_M') for item in items]
        self.assertEqual(strip(quad_programs), strip(programs), 'every other program field is the pair\'s')
        self.assertEqual([event for event in quad if event[0] == 'create_heads'],
                         [('create_heads', (1, 1, 64, 1024), (1, 1, 64, 512), 8, 2)], 'no query pad at 64 rows')
        self.assertEqual([event for event in quad if event[0] == 'concat_heads'], [('concat_heads', (1, 8, 64, 128))])
        self.assertEqual([event for event in quad if event[0] == 'all_gather'], [('all_gather', (1, 1, 64, 5120), 0)])
        self.assertEqual([event for event in quad if event[0] == 'sdpa'],
                         [('sdpa', (1, 32, 32, 128), (1, 8, 2080, 128), (1, 1, 32, 2080))],
                         'the pair fold\'s program shape: 32 query heads over 8 KV heads of 2080 keys')
        self.assertEqual([event[3] for event in quad if event[0] == 'convolve'],
                         [((0, 16), (16, 32), (32, 48), (48, 64))] * 2)
        self.assertEqual(quad[-1], ('output', (1, 1, 64, 5120)))

    def test_the_kv_assembly_is_twelve_pieces_of_two_heads_with_pair_pads(self):
        with four_cards():
            quad = attention_run4(quad=True)
        assembly = [event for event in quad if event[0] == 'concat' and len(event[1]) == 12]
        self.assertEqual(len(assembly), 2, 'one concat each for k and v')
        self.assertEqual(assembly[0][1], ((1, 2, 2048, 128), (1, 2, 16, 128), (1, 2, 16, 128)) * 4)
        live = [event for event in quad if event[0] == 'slice' and event[1] == (1, 2, 64, 128)]
        self.assertEqual([event[2][2] for event in live], [0, 0, 16, 0, 32, 32, 48, 32] * 2)

    def test_the_pairs_sdpa_mode_runs_two_pair_folds_at_16_over_4(self):
        with four_cards():
            events = attention_run4(quad=True, sdpa='pairs')
        self.assertEqual([event for event in events if event[0] == 'sdpa'],
                         [('sdpa', (1, 16, 32, 128), (1, 4, 2080, 128), (1, 1, 32, 2080))] * 2)

    def test_the_mlp_branch_at_four_cards_is_todays_at_64_rows_with_per_core_m_2(self):
        with four_cards():
            pair_events, quad = mlp_run4(quad=False), mlp_run4(quad=True)
        self.assertEqual([event[0] for event in quad], [event[0] for event in pair_events], 'op for op')
        programs = [event[1] for event in pair_events if event[0] == 'program']
        quad_programs = [event[1] for event in quad if event[0] == 'program']
        self.assertEqual([dict(item)['per_core_M'] for item in programs], [1] * 4)
        self.assertEqual([dict(item)['per_core_M'] for item in quad_programs], [2] * 4)
        strip = lambda items: [tuple(pair_ for pair_ in item if pair_[0] != 'per_core_M') for item in items]
        self.assertEqual(strip(quad_programs), strip(programs))
        self.assertEqual([event for event in quad if event[0] == 'all_gather'], [('all_gather', (1, 1, 64, 5120), 0)])

    def test_at_the_pair_the_twin_pass_runs_the_pinned_passes_op_log(self):
        """Same branch, same shapes at two chips: the twin's QuadPass and the pinned one produce one event log."""
        with pair():
            twin_events = attention_run4(quad=True, chips=2)
        with pair():
            found_events = base.attention_run(__import__('draft_attention_branch'), quad=True)
        self.assertEqual([event for event in twin_events if event[0] in base.CORE_OPS and event[0] != 'all_gather'],
                         [event for event in found_events if event[0] in base.CORE_OPS and event[0] != 'all_gather'])
        self.assertEqual([event for event in twin_events if event[0] == 'sdpa'],
                         [event for event in found_events if event[0] == 'sdpa'])


class HelperTests(unittest.TestCase):
    def test_the_64_row_helpers_are_the_four_card_32_row_twins_calls_widened(self):
        import draft_head_layout_tp
        import draft_kv_projection_tp
        import feature_collective_tp

        def run(project, split, concat, gather, rows):
            ops = ShapeOps4(4)
            owned = []
            retain = lambda value: owned.append(value) or value
            parameters = dict(operations=ops, native_head_layout=True, kernel='kernel',
                              projections=dict(k=base.Tensor((5120, 256)), v=base.Tensor((5120, 256))),
                              head_norms=dict(k=base.Tensor((1, 1, 4, 32))))
            with patch('mesh_link_policy.projection_links', return_value=1), \
                    patch('feature_collective_tp.projection_links', return_value=1):
                project(ops, base.Tensor((1, 1, rows, 5120)), base.Tensor((1, 1, rows, 1024)),
                        (base.Tensor((1, 1, rows, 128)), base.Tensor((1, 1, rows, 128))), retain, parameters=parameters)
                split(ops, base.Tensor((1, 1, rows, 1024)), base.Tensor((1, 1, rows, 256)),
                      base.Tensor((1, 1, rows, 256)), retain)
                concat(ops, base.Tensor((1, 8, rows, 128)), retain)
                ops.log('gathered', gather(ops, MESH4, COLLECTIVES, base.Tensor((1, 1, rows, 5120), 'fp32'),
                                           retain_temporaries=retain).shape)
            return ops.events

        with four_cards():
            today = run(draft_kv_projection_tp.project_key_value, draft_head_layout_tp.split_projected_heads,
                        draft_head_layout_tp.concatenate_query_heads, feature_collective_tp.gather_add_projection, 32)
            quad = run(twin.project_key_value, twin.split_projected_heads, twin.concatenate_query_heads,
                       twin.gather_add_projection, 64)
        self.assertGreater(len(today), 15)
        self.assertEqual(quad, [base.widen(event) for event in today])
        # four chips are added in chip order, ((p0 + p1) + p2) + p3
        self.assertEqual([event for event in quad if event[0] == 'add'].__len__(), 3)

    def test_the_helpers_refuse_the_pairs_widths_at_four_cards(self):
        ops = ShapeOps4(4)
        retain = lambda value: value
        with four_cards():
            with self.assertRaises(ValueError):
                twin.split_projected_heads(ops, base.Tensor((1, 1, 64, 2048)), base.Tensor((1, 1, 64, 512)),
                                           base.Tensor((1, 1, 64, 512)), retain)
            with self.assertRaises(ValueError):
                twin.concatenate_query_heads(ops, base.Tensor((1, 16, 64, 128)), retain)
            with self.assertRaises(ValueError):
                twin.split_projected_heads(ops, base.Tensor((1, 1, 32, 1024)), base.Tensor((1, 1, 32, 256)),
                                           base.Tensor((1, 1, 32, 256)), retain)


class HeadTests(unittest.TestCase):
    def test_head_candidates_at_four_cards_take_two_chunks_of_a_62080_shard_on_each_half(self):
        with four_cards():
            ops = ShapeOps4(4)
            owned = []
            model = SimpleNamespace(num_devices=4, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight='head')
            chunks = twin.head_candidates(ops, model, base.Tensor((1, 1, 64, 5120)), owned,
                                          lambda value: owned.append(value) or value)
        self.assertEqual([event for event in ops.events if event[0] == 'linear'], [('linear', (1, 1, 32, 5120))] * 2)
        self.assertEqual([event for event in ops.events if event[0] == 'topk'],
                         [('topk', (1, 1, 32, 32768))] * 4, 'two chunks per half, the second padded to 32,768')
        concats = [event for event in ops.events if event[0] == 'concat']
        self.assertEqual(concats, [('concat', ((1, 1, 32, 16), (1, 1, 32, 16)), 2)] * 4, 'values then indices, per chunk')
        self.assertEqual([(chunk['start'], chunk['stop'], chunk['values'].shape, chunk['indices'].dtype)
                          for chunk in chunks], [(0, 32768, (1, 1, 64, 16), 'u16'), (32768, 62080, (1, 1, 64, 16), 'u16')])

    def test_the_head_refuses_the_pairs_model(self):
        with four_cards():
            model = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight='head')
            with self.assertRaises(ValueError):
                twin.head_candidates(ShapeOps4(4), model, base.Tensor((1, 1, 64, 5120)), [], lambda value: value)


class ConvTests(unittest.TestCase):
    def program(self, conv='110'):
        ops = DescriptorOps4()
        hidden, dynamic, kernels = base.conv_operands()
        with patch.object(pinned, '_KERNEL_CHECKED', []):
            output = twin.quad_fused_convolution(ops, MESH4, hidden, dynamic, kernels, boundaries=base.QUAD_SEAMS,
                                                 conv=conv)
        self.assertEqual(len(ops.calls), 1)
        tensors, program = ops.calls[0]
        self.assertIs(tensors[-1], output)
        return tensors, program

    def test_e1b_runs_one_110_worker_program_per_chip(self):
        with four_cards():
            tensors, program = self.program('110')
        self.assertEqual(sorted(program), [((0, chip), (0, chip)) for chip in range(4)], 'one program per chip')
        for chip in range(4):
            descriptor = program[((0, chip), (0, chip))]
            reader, *computes = descriptor['kernels']
            self.assertTrue(reader['kernel_source'].endswith('quad_conv_io.cpp'))
            self.assertEqual([(kernel['compile_time_args'], kernel['core_ranges']) for kernel in computes],
                             [([2], (((10, 0), (10, 9)),)), ([3], tuple(((x, 0), (x, 9)) for x in range(10)))])
            addresses = [tensor.address + chip for tensor in tensors]
            for worker in range(110):
                x, y = worker // 10, worker % 10
                self.assertEqual(reader['runtime_args'][x][y], addresses + [64, worker, 110, 0x10001, 0x10001])

    def test_e1_runs_the_served_80_workers_on_every_chip(self):
        with four_cards():
            _, program = self.program('80')
        for chip in range(4):
            reader, *computes = program[((0, chip), (0, chip))]['kernels']
            self.assertEqual([kernel['compile_time_args'] for kernel in computes], [[4]])
            self.assertEqual(reader['runtime_args'][0][0][-5:], [64, 0, 80, 0x10001, 0x10001])

    def test_the_conv_refuses_the_pairs_mesh_and_what_the_kernel_cannot_serve(self):
        hidden, dynamic, kernels = base.conv_operands()
        with four_cards():
            ops = DescriptorOps4()
            with self.assertRaises(ValueError):
                twin.quad_fused_convolution(ops, base.MESH, hidden, dynamic, kernels, boundaries=base.QUAD_SEAMS)
            with self.assertRaises(ValueError):
                twin.quad_fused_convolution(ops, MESH4, *base.conv_operands(32), boundaries=((0, 16), (16, 32)))
            with self.assertRaises(ValueError):
                twin.quad_fused_convolution(ops, MESH4, hidden, dynamic, kernels, boundaries=((0, 64),))
            with self.assertRaises(ValueError):
                twin.quad_fused_convolution(ops, MESH4, hidden, dynamic, kernels, boundaries=base.QUAD_SEAMS,
                                            conv='halves')
            two = DescriptorOps4()
            two.get_device_tensors = lambda tensor: [base.Shard(tensor.address), base.Shard(tensor.address + 1)]
            with self.assertRaises(ValueError):
                twin.quad_fused_convolution(two, MESH4, hidden, dynamic, kernels, boundaries=base.QUAD_SEAMS)
            self.assertEqual(ops.calls + two.calls, [])

    def test_the_halves_mode_calls_the_four_card_served_kernel_per_half(self):
        with four_cards():
            owned, calls = [], []

            def served(operations, mesh, hidden, dynamic, kernels, *, boundaries=None):
                calls.append((hidden.shape, tuple(value.shape for value in dynamic), boundaries))
                return base.Tensor(hidden.shape)

            with patch('draft_convolution_fused_tp.fused_convolution', side_effect=served):
                output = twin.QuadPass(conv='halves').convolution(ShapeOps4(4), MESH4, *base.conv_operands(),
                    fp32_intermediates=True, retain_temporaries=lambda value: owned.append(value) or value,
                    boundaries=base.QUAD_SEAMS)
        self.assertEqual(calls, [((1, 1, 32, 5120), ((1, 1, 32, 320),) * 2, ((0, 16), (16, 32)))] * 2)
        self.assertEqual(output.shape, (1, 1, 64, 5120))
        self.assertIs(owned[-1], output)

    def test_the_conv_operation_keeps_checked_convolutions_contract(self):
        quad = twin.QuadPass()
        with four_cards():
            for options in (dict(fp32_intermediates=False, retain_temporaries=lambda value: value, boundaries=base.QUAD_SEAMS),
                            dict(fp32_intermediates=True, retain_temporaries=None, boundaries=base.QUAD_SEAMS),
                            dict(fp32_intermediates=True, retain_temporaries=lambda value: value)):
                with self.subTest(options=sorted(options)), self.assertRaises(ValueError):
                    quad.convolution(DescriptorOps4(), MESH4, *base.conv_operands(), **options)

    def test_the_pass_is_the_pinned_pass_with_this_modules_helpers(self):
        for name in ('project_key_value', 'concatenate_query_heads', 'head_candidates'):
            self.assertIs(getattr(twin.QuadPass, name), getattr(twin, name), name)
        self.assertIs(twin.QuadPass.key_value_plan, pinned.QuadPass.key_value_plan)
        self.assertIs(twin.QuadPass.gather_add_projection, pinned.QuadPass.gather_add_projection)
        self.assertEqual((twin.QuadPass.rows, twin.QuadPass.mask_rows), (64, 2080))
        with self.assertRaises(ValueError):
            twin.QuadPass(sdpa='dense')


# ---------------------------------------------------------------------------------------------
# The readback at four chips.
# ---------------------------------------------------------------------------------------------

class HostOps4:
    def get_device_tensors(self, tensor):
        return [SimpleNamespace(values=value) for value in tensor.chips]

    def to_torch(self, shard):
        return shard.values.clone()


def four_chip_users(generator, users=4, chips=4):
    """Each user's own single-user outputs as its capture leaves them - per chip and per candidate chunk, (1, 1, 16, 16)
    values and indices, and the (1, 1, 32, 256) replicated selector features (rows 16-31 the pad) - and the same rows
    stacked as the quad's 64-row outputs."""
    from draft_shared_head_tp import candidate_chunks

    singles = []
    for _ in range(users):
        chunks = []
        for start, stop in candidate_chunks():
            values, indices = [], []
            for _chip in range(chips):
                values.append(torch.randn(1, 1, 16, 16, generator=generator).bfloat16())
                rows = [torch.randperm(stop - start, generator=generator)[:16] for _ in range(16)]
                indices.append(torch.stack(rows).reshape(1, 1, 16, 16).to(torch.int32))
            chunks.append(dict(start=start, stop=stop, values=values, indices=indices))
        singles.append(dict(chunks=chunks, projected=torch.randn(1, 1, 32, 256, generator=generator).bfloat16()))
    quad_chunks = []
    for number in range(len(singles[0]['chunks'])):
        first = singles[0]['chunks'][number]
        quad_chunks.append(dict(start=first['start'], stop=first['stop'],
            values=SimpleNamespace(chips=[torch.cat([single['chunks'][number]['values'][chip] for single in singles], dim=2)
                                          for chip in range(chips)]),
            indices=SimpleNamespace(chips=[torch.cat([single['chunks'][number]['indices'][chip] for single in singles], dim=2)
                                           for chip in range(chips)])))
    projected = torch.cat([single['projected'][:, :, :16] for single in singles], dim=2)
    quad = SimpleNamespace(chunks=quad_chunks, projected=SimpleNamespace(chips=[projected.clone() for _ in range(chips)]))
    return singles, quad


def single_outputs(single, chips=4):
    return SimpleNamespace(
        chunks=[dict(start=chunk['start'], stop=chunk['stop'], values=SimpleNamespace(chips=chunk['values']),
                     indices=SimpleNamespace(chips=chunk['indices'])) for chunk in single['chunks']],
        projected=SimpleNamespace(chips=[single['projected'].clone() for _ in range(chips)]))


class ReadbackTests(unittest.TestCase):
    def test_every_users_parts_are_its_own_singles_rows_at_four_chips(self):
        generator = torch.Generator().manual_seed(21)
        with four_cards():
            singles, quad = four_chip_users(generator)
            device = SimpleNamespace(operations=HostOps4())
            parts = twin.read_quad_outputs(device, quad)
            self.assertEqual(len(parts), 4)
            operations = device.operations
            for user, single in enumerate(singles):
                expected = draft_singles_audit.read_parts(draft_singles_audit.user_rows(
                    draft_singles_audit.raw_outputs(operations, single_outputs(single)), 0))
                for key in ('hidden', 'candidates', 'unary'):
                    with self.subTest(user=user, key=key):
                        mine = parts[user][key]
                        self.assertEqual(tuple(mine.shape), tuple(expected[key].shape))
                        self.assertTrue(draft_singles_audit.same_bits(mine, expected[key]))

    def test_the_readback_reads_four_chips_by_two_chunks(self):
        generator = torch.Generator().manual_seed(22)
        with four_cards():
            _, quad = four_chip_users(generator)
            self.assertEqual(len(quad.chunks), 2)
            reads = Mock(wraps=HostOps4().to_torch)
            operations = HostOps4()
            operations.to_torch = reads
            twin.read_quad_outputs(SimpleNamespace(operations=operations), quad)
        # 2 chunks x 4 chips x (values + indices), and the features from each of the 4 chips
        self.assertEqual(reads.call_count, 2 * 4 * 2 + 4)

    def test_a_readback_with_the_pairs_two_chips_is_refused_at_four_cards(self):
        generator = torch.Generator().manual_seed(23)
        with four_cards():
            _, quad = four_chip_users(generator, chips=2)
            with self.assertRaises(AssertionError):
                twin.read_quad_outputs(SimpleNamespace(operations=HostOps4()), quad)

    def test_replicated_features_that_differ_on_any_chip_are_refused(self):
        generator = torch.Generator().manual_seed(24)
        with four_cards():
            _, quad = four_chip_users(generator)
            quad.projected.chips[3][0, 0, 5, 3] += 1
            with self.assertRaises(AssertionError):
                twin.read_quad_outputs(SimpleNamespace(operations=HostOps4()), quad)

    def test_a_refused_block_is_reported_per_chip_and_raises_the_same_error(self):
        generator = torch.Generator().manual_seed(25)
        with four_cards():
            _, quad = four_chip_users(generator)
            quad.chunks[1]['values'].chips[3][0, 0, 40:, :] = float('-inf')
            lines, loguru = base.logged()
            with loguru, self.assertRaises(ValueError) as raised:
                twin.read_quad_outputs(SimpleNamespace(operations=HostOps4(), pool_slot=SimpleNamespace(index=0)), quad)
        self.assertEqual(str(raised.exception), 'Finite complete-block top16 values and in-range integer indices required')
        reports = [line for line in lines if line.startswith('[PINDIAG] draft outputs rejected')]
        self.assertEqual(len(reports), 4, 'the refused half, one line per chip')
        self.assertIn('device_slot=0 chip=3 nonfinite=%d neg_inf=%d nan=0 ' % (24 * 16, 24 * 16), reports[3])

    def test_select_quad_outputs_selects_each_users_own_slice(self):
        generator = torch.Generator().manual_seed(26)
        with four_cards():
            _, quad = four_chip_users(generator)
            parts = [dict(user=user) for user in range(4)]
            with patch.object(twin, 'read_quad_outputs', return_value=parts), \
                    patch('dflash_packed_proposal.select_packed', return_value='tokens') as select:
                device = SimpleNamespace(predecessors='p', successors='s')
                self.assertEqual(twin.select_quad_outputs(device, quad, (1, 2, 3, 4), (15,) * 4), 'tokens')
            select.assert_called_once_with(parts, (1, 2, 3, 4), (15,) * 4, 'p', 's')


# ---------------------------------------------------------------------------------------------
# The trace on four-chip fakes.
# ---------------------------------------------------------------------------------------------

class FakeTensor4(FakeTensor):
    """A device tensor of four chips."""

    def __init__(self, value, dtype, layout, on_device, addresses):
        super().__init__(value, dtype, layout, on_device, addresses)
        if on_device:
            self.chips = [value.clone() for _ in range(4)]
            self.shards = [FakeShard(self, chip, next(addresses)) for chip in range(4)]

    def write(self, value):
        self.value = value.clone()
        if self.chips is not None:
            self.chips = [value.clone() for _ in range(4)]


class RecordingOps4(RecordingOps):
    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        tensor = FakeTensor4(value, dtype, layout, device is not None, self.addresses)
        self.event('from_torch', value, device is not None, dtype, layout, memory_config, mesh_mapper, tensor)
        return tensor


def quad_devices4(ops):
    """Four devices on one mesh, pooled slots 0-3, each with five layers of active K/V banks that are device tensors."""
    from test_dflash_proposal_trace import pair_devices

    made = [*pair_devices(ops, context=2048), *pair_devices(ops, context=2048)]
    mesh = made[0].mesh
    for index, device in enumerate(made):
        device.mesh = mesh
        device.pool_slot = SimpleNamespace(index=index)
        device.position = 4096 + 1000 * index + 7
        device.kv_history.active = fresh_banks(ops)
    return made


def fresh_banks(ops):
    return [{name: FakeTensor4(torch.zeros(1), 'bf16', 'tile', True, ops.addresses) for name in ('k', 'v')}
            for _ in range(5)]


class TraceTests(unittest.TestCase):
    def setUp(self):
        import dflash_proposal_trace

        environment = clean_environment(QWEN_FAST_QUAD_DRAFT='1', QWEN_FAST_PACKED_PROPOSAL='1',
                                        QWEN_FAST_PAIR_ROW_EXACT='1', QWEN_FAST_ROUND_B1='1')
        environment.start()
        self.addCleanup(environment.stop)
        cards = four_cards()
        cards.__enter__()
        self.addCleanup(cards.__exit__, None, None, None)
        for target, value in ((pinned, '_NOTED'), (dflash_proposal_trace, '_PAIR_MASK_REFRESH_NOTED')):
            patcher = patch.object(target, value, [])
            patcher.start()
            self.addCleanup(patcher.stop)

    def build(self, **options):
        ops = RecordingOps4()
        devices = quad_devices4(ops)
        return ops, devices, twin.PreparedQuadDFlashProposal(devices, **options)

    def test_the_bucket_owns_four_users_placeholder_banks_and_never_reads_the_pools(self):
        ops, devices, trace = self.build()
        lines, loguru = base.logged()
        with loguru, patch('memory_ledger.record') as ledger:
            self.assertTrue(trace.prepare_device((11, 22, 33, 44)))
        uploads = [event for event in ops.events if event[0] == 'from_torch' and event[2] is True]
        shapes = [event[1][1] for event in uploads]
        self.assertEqual(shapes[:6], [(1, 64), (1, 1, 32, 2080), (1, 1, 64, 128), (1, 1, 64, 128), (1, 1, 64, 128),
                                      (1, 1, 64, 128)])
        self.assertEqual(shapes[6:], [(1, 2, 2048, 128)] * 40, '4 users x 5 layers x k and v, two KV heads')
        bucket = trace.buckets[(2048,) * 4]
        self.assertEqual(set(bucket.rope), {'q', 'live_k'}, 'no rope.k: nothing reads it')
        self.assertTrue(same_bits(bucket.host_mask, fold_mask()))
        self.assertEqual(len(bucket.cached_history), 4)
        active = {id(layer[name]) for device in devices for layer in device.kv_history.active for name in ('k', 'v')}
        held = [layer[name] for cache in bucket.cached_history for layer in cache for name in ('k', 'v')]
        self.assertEqual(len(held), 40)
        self.assertFalse(active & {id(value) for value in held}, 'placeholders, never the devices\' own banks')
        self.assertFalse(getattr(bucket, 'live_banks', False))
        call = devices[0].execute_proposal.call_args
        self.assertIsInstance(call.kwargs['quad'], twin.QuadPass)
        self.assertNotIsInstance(call.kwargs['quad'], type(None))
        self.assertIs(call.kwargs['cached_history'], bucket.cached_history)
        self.assertEqual(call.kwargs['pack'], [dict(position=device.position, history_rows=2048) for device in devices])
        self.assertEqual([line for line in lines if line.startswith(pinned.MARKER)],
                         ['[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=32/8 rows=64 sdpa=fold conv=110'])
        self.assertFalse([line for line in lines if 'live banks' in line], 'no live banks are bound')
        self.assertEqual(ledger.call_args.kwargs['point'], 'slots=0,1,2,3')
        self.assertEqual(len(ledger.call_args.kwargs['quad_placeholders']), 6 + 40)
        self.assertTrue(trace.last_built)

    def test_every_round_copies_each_users_active_bank_into_its_placeholder_and_follows_a_swap(self):
        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'):
            trace.prepare_device((1, 2, 3, 4))
            bucket = trace.buckets[(2048,) * 4]
            placeholders = [ops.normalize(layer[name]) for cache in bucket.cached_history for layer in cache
                            for name in ('k', 'v')]
            before = len(ops.events)
            trace.prepare_device((5, 6, 7, 8))
            first_round = ops.events[before:]
            # the four-card slide's commit swaps the active bank: the next round reads the other one
            swapped = [fresh_banks(ops) for _ in devices]
            for device, banks in zip(devices, swapped):
                device.kv_history.active = banks
            before = len(ops.events)
            trace.prepare_device((9, 10, 11, 12))
            second_round = ops.events[before:]
        for events in (first_round, second_round):
            slices = [event for event in events if event[0] == 'slice']
            copies = [event for event in events if event[0] == 'copy']
            self.assertEqual(len(slices), 40, 'one slice of each bank')
            self.assertEqual({(event[2], event[3]) for event in slices}, {((0, 0, 0, 0), (1, 2, 2048, 128))})
            self.assertEqual([event[2] for event in copies], placeholders, 'into the placeholders, in user-major order')
        active_ids = [ops.normalize(layer[name]) for banks in swapped for layer in banks for name in ('k', 'v')]
        self.assertEqual([event[1] for event in second_round if event[0] == 'slice'], active_ids,
                         'the round after the swap reads the new active banks')
        self.assertNotEqual([event[1] for event in first_round if event[0] == 'slice'], active_ids)
        # the placeholders never moved
        self.assertEqual([ops.normalize(layer[name]) for cache in bucket.cached_history for layer in cache
                          for name in ('k', 'v')], placeholders)

    def test_every_rounds_host_inputs_are_the_two_pairs_uploads(self):
        """Rows [32p, 32p + 32) of the quad's ids, rope.q and live_k are pair p's, and the mask is the pair fold's, bit
        for bit, under the pair's own QWEN_FAST_ROUND_B1 build at four cards."""
        import dflash_proposal_trace

        ops, devices, trace = self.build()
        seeds = (101, 202, 303, 404)
        with base.logged()[1], patch('memory_ledger.record'):
            trace.prepare_device(seeds)
            pairs = []
            for first, second in twin.PAIRS:
                pair_trace = dflash_proposal_trace.PreparedPackedDFlashProposal(devices[first], devices[second])
                pair_trace.prepare_device(seeds[first], seeds[second])
                pairs.append(next(iter(pair_trace.buckets.values())))
        bucket = trace.buckets[(2048,) * 4]
        for index, pair_bucket in enumerate(pairs):
            rows = slice(32 * index, 32 * (index + 1))
            with self.subTest(pair=index):
                self.assertTrue(torch.equal(bucket.identifiers.value[:, rows], pair_bucket.identifiers.value))
                for name in ('q', 'live_k'):
                    for table, theirs in zip(bucket.rope[name], pair_bucket.rope[name]):
                        self.assertTrue(same_bits(table.value[:, :, rows], theirs.value), name)
                self.assertTrue(same_bits(bucket.mask.value, pair_bucket.mask.value))
                self.assertTrue(getattr(pair_bucket, 'row_exact', False))
                self.assertEqual(tuple(pair_bucket.cached_history[0][0]['k'].shape), (1, 2, 2048, 128))

    def test_prepare_collect_adopt_finish_use_this_modules_readback(self):
        ops, devices, trace = self.build()
        parts = [dict(user=user) for user in range(4)]
        with base.logged()[1], patch('memory_ledger.record'), \
                patch.object(twin, 'read_quad_outputs', return_value=parts) as read:
            trace.prepare_device((11, 22, 33, 44))
            collected = trace.collect()
            read.assert_called_once()
            self.assertEqual(collected, dict(parts=parts, seeds=(11, 22, 33, 44), counts=(15,) * 4))
            with self.assertRaises(ValueError):
                trace.adopt([(1,), (2,)])
            trace.adopt([(1, 2, 3), (4,), (5, 6), (7,)])
            with self.assertRaises(ValueError):
                trace.collect()
            self.assertEqual(trace.finish(2, 1), (5,))
            self.assertEqual(trace.finish(0, 2), (1, 2))
            self.assertEqual(trace.finish(1, 5), (4,))
            self.assertEqual(trace.finish(3, 1), (7,))
            self.assertIsNone(trace._pending)

    def test_finish_without_a_batched_selection_selects_with_this_modules_reader(self):
        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'), \
                patch.object(twin, 'select_quad_outputs', return_value=((1,), (2,), (3,), (4,))) as select:
            trace.prepare_device((11, 22, 33, 44))
            self.assertEqual(trace.finish(1, 1), (2,))
            self.assertEqual(select.call_args.args[2:], ((11, 22, 33, 44), (15,) * 4))
            self.assertEqual(trace.audit_selection(), ((1,), (2,), (3,), (4,)))

    def test_a_failed_build_releases_every_placeholder(self):
        ops, devices, trace = self.build()
        devices[0].execute_proposal.side_effect = RuntimeError('capture failed')
        with base.logged()[1], self.assertRaises(RuntimeError):
            trace.prepare_device((1, 2, 3, 4))
        self.assertEqual(trace.buckets, {})
        self.assertEqual(trace.owned, [])
        freed = [event for event in ops.events if event[0] == 'deallocate']
        self.assertEqual(len(freed), 6 + 40, 'ids, mask, the four rope tables and the 40 placeholder banks')
        self.assertEqual(devices[0].validated_native_proposal_masks, set())

    def test_close_releases_the_trace_and_the_placeholders(self):
        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'):
            trace.prepare_device((1, 2, 3, 4))
        trace.close()
        self.assertTrue(trace.closed)
        self.assertEqual([event[0] for event in ops.events].count('release_trace'), 1)
        self.assertEqual(trace.owned, [])
        self.assertFalse(trace.prepare_device((1, 2, 3, 4)))

    def test_a_device_mid_publication_or_closed_declines_and_the_constructor_refuses_what_it_cannot_serve(self):
        for attribute, value in (('pending', object()), ('closed', True), ('progress', object())):
            ops, devices, trace = self.build()
            setattr(devices[3], attribute, value)
            with self.subTest(attribute=attribute):
                self.assertFalse(trace.prepare_device((1, 2, 3, 4)))
                self.assertEqual(trace.buckets, {})
        ops = RecordingOps4()
        devices = quad_devices4(ops)
        with self.assertRaises(ValueError):
            twin.PreparedQuadDFlashProposal(devices[:3])
        devices[2].block_rows = 8
        with self.assertRaises(ValueError):
            twin.PreparedQuadDFlashProposal(devices)

    def test_a_pool_that_holds_the_quads_mask_and_outputs_is_borrowed_as_the_pinned_quad_does(self):
        """S2 M0 / v86 are the pinned trace helpers' (borrow_pooled_mask, pool_outputs): the twin calls them unchanged."""
        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'), \
                patch('dflash_proposal_trace.borrow_pooled_mask', wraps=__import__('dflash_proposal_trace').borrow_pooled_mask) as mask, \
                patch('dflash_proposal_trace.pool_outputs', wraps=__import__('dflash_proposal_trace').pool_outputs) as outputs:
            trace.prepare_device((1, 2, 3, 4))
        self.assertEqual(mask.call_args.args[1], [0, 1, 2, 3])
        self.assertEqual(outputs.call_args.args[1], [0, 1, 2, 3])


class AuditAtFourChipsTests(unittest.TestCase):
    """The pinned shadow audit's comparison is chip-generic (it zips every chip): held at four chips on a fixture in the shape
    base.CompareTests builds at two."""

    def fixture(self, chips=4):
        generator = torch.Generator().manual_seed(9)
        ops = HostOps4()
        pair_buckets, quad_chunks, quad_parts, pairs, tokens = [], [], [], [], []

        def shards(shape, integer=False):
            if integer:
                return SimpleNamespace(chips=[torch.randint(0, 999, shape, generator=generator) for _ in range(chips)])
            return SimpleNamespace(chips=[torch.randn(*shape, generator=generator) for _ in range(chips)])

        for index in range(2):
            chunks = [dict(values=shards((1, 1, 32, 16)), indices=shards((1, 1, 32, 16), True)) for _ in range(2)]
            projected = torch.randn(1, 1, 32, 256, generator=generator).bfloat16()
            parts = [dict(hidden=torch.randn(1, 15, 256, generator=generator).bfloat16(),
                          candidates=torch.randint(0, 999, (1, 15, 16), generator=generator),
                          unary=torch.randn(1, 15, 16, generator=generator)) for _ in range(2)]
            pair_bucket = SimpleNamespace(outputs=SimpleNamespace(chunks=chunks, projected=SimpleNamespace(
                chips=[projected.clone() for _ in range(chips)])))
            seeds = (10 * (2 * index + 1), 10 * (2 * index + 2))
            pair_trace = SimpleNamespace(device_a=SimpleNamespace(predecessors='p', successors='s'), discarded=0)
            pair_trace.collect = Mock(return_value=dict(parts=parts, seeds=seeds, counts=(15, 15)))
            pair_trace._pending = (seeds[0], seeds[1], pair_bucket, [])
            pairs.append(([2 * index, 2 * index + 1], pair_trace))
            quad_parts.extend(dict((key, value.clone()) for key, value in part.items()) for part in parts)
            pair_buckets.append(pair_bucket)
            tokens.extend([(index, 1), (index, 2)])
        for number in range(2):
            quad_chunks.append({key: SimpleNamespace(chips=[torch.cat(
                [pair_buckets[0].outputs.chunks[number][key].chips[chip],
                 pair_buckets[1].outputs.chunks[number][key].chips[chip]], dim=2) for chip in range(chips)])
                for key in ('values', 'indices')})
        wide = torch.cat([bucket.outputs.projected.chips[0] for bucket in pair_buckets], dim=2)
        bucket = SimpleNamespace(parts=quad_parts, tokens=tuple(tokens), outputs=SimpleNamespace(
            chunks=quad_chunks, projected=SimpleNamespace(chips=[wide.clone() for _ in range(chips)])))
        quad = SimpleNamespace(operations=ops, _pending=((10, 20, 30, 40), bucket, []))
        return quad, pairs

    def selected(self, parts, seeds, counts, predecessors, successors):
        return ((seeds[0] // 10 - 1 >= 2 and 1 or 0, 1), (seeds[0] // 10 - 1 >= 2 and 1 or 0, 2))

    def compare(self, quad, pairs):
        quad._audit_snapshot = pinned.raw_outputs(quad.operations, quad._pending[1].outputs)
        with patch('dflash_packed_proposal.select_packed_batched', side_effect=self.selected):
            return pinned.compare_with_pairs(quad, pairs)

    def test_equal_rounds_compare_every_chip(self):
        quad, pairs = self.fixture()
        equal, stage, checks = self.compare(quad, pairs)
        self.assertEqual((equal, stage), (True, 'all'))
        # the quad against its snapshot: 2 chunks x 2 keys x 4 chips and 4 feature reads; per pair: 2 users x 4 checks,
        # 2 chunks x 2 keys x 4 chips and 4 feature reads
        self.assertEqual(checks, (16 + 4) + 2 * (2 * 4 + 16 + 4))

    def test_a_difference_on_the_fourth_chip_is_named(self):
        quad, pairs = self.fixture()
        quad._pending[1].outputs.chunks[1]['values'].chips[3][0, 0, 40, 3] = 99.0
        quad._audit_snapshot = pinned.raw_outputs(quad.operations, quad._pending[1].outputs)
        pairs[1][1]._pending[2].outputs.chunks[1]['values'].chips[3][0, 0, 8, 3] += 1
        with patch('dflash_packed_proposal.select_packed_batched', side_effect=self.selected):
            self.assertEqual(pinned.compare_with_pairs(quad, pairs)[:2], (False, 'values:chunk1:chip3:pair1'))


# ---------------------------------------------------------------------------------------------
# The engage-time refusal.
# ---------------------------------------------------------------------------------------------

def refusal_devices(chips=4, **overrides):
    layers, predecessors, successors = object(), object(), object()
    mesh = SimpleNamespace(shape=[1, chips], compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
    operations = object()
    devices = []
    for _ in range(4):
        fields = dict(operations=operations, mesh=mesh, block_rows=16, fused_convolution=True, layers=layers,
                      predecessors=predecessors, successors=successors)
        fields.update(overrides)
        devices.append(SimpleNamespace(**fields))
    return devices


class RefusalTests(unittest.TestCase):
    def refuse(self, devices=None, batched=(), environ=None):
        return twin.refusal(refusal_devices() if devices is None else devices, [] if batched == () else batched,
                            dict(FOUR_FLAGS) if environ is None else environ)

    def test_a_four_card_quad_over_four_packable_devices_is_served(self):
        self.assertIsNone(self.refuse())

    def test_each_refusal_names_its_reason(self):
        self.assertEqual(self.refuse(environ={}), 'the four-card quad serves QWEN_FAST_TP=4 only')
        self.assertEqual(self.refuse(environ=dict(FOUR_FLAGS, QWEN_FAST_PACKED_PROPOSAL='0', QWEN_FAST_ROUND_B1='0')),
                         'requires QWEN_FAST_PACKED_PROPOSAL=1,QWEN_FAST_ROUND_B1=1')
        self.assertIn('needs the four-card fused commit', self.refuse(
            environ=dict(FOUR_FLAGS, QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='1')))
        self.assertIsNone(self.refuse(environ=dict(FOUR_FLAGS, QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='0')))
        self.assertEqual(self.refuse(batched=None), 'requires the batched selection (QWEN_FAST_ROUND_B1=1)')
        self.assertEqual(self.refuse(refusal_devices(chips=2)), 'the four devices are not on the (1, 4) mesh')
        self.assertEqual(self.refuse(refusal_devices(block_rows=8)), 'the four devices are not T16')
        self.assertEqual(self.refuse(refusal_devices(fused_convolution=False)), 'requires the fused learned convolution')
        other = refusal_devices()
        other[2].layers = object()
        self.assertEqual(self.refuse(other), 'the four devices do not share one draft weight set')
        split = refusal_devices()
        split[1].mesh = refusal_devices()[0].mesh
        self.assertEqual(self.refuse(split), 'the four devices do not share one mesh')
        small = refusal_devices()
        small[0].mesh.compute_with_storage_grid_size = lambda: SimpleNamespace(x=8, y=10)
        self.assertEqual(self.refuse(small), 'the compute grid cannot hold QWEN_FAST_QUAD_CONV=110')
        self.assertIsNone(self.refuse(small, environ=dict(FOUR_FLAGS, QWEN_FAST_QUAD_CONV='80')))
        with self.assertRaises(ValueError):
            self.refuse(environ=dict(FOUR_FLAGS, QWEN_FAST_QUAD_SDPA='dense'))


# ---------------------------------------------------------------------------------------------
# The coordinator at four cards.
# ---------------------------------------------------------------------------------------------

class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        flags = dict(FOUR_FLAGS, QWEN_FAST_PACKED_AUDIT='1')
        environment = clean_environment(**flags)
        environment.start()
        self.addCleanup(environment.stop)
        cards = four_cards()
        cards.__enter__()
        self.addCleanup(cards.__exit__, None, None, None)
        del base.LOG[:]
        base.FakeQuadTrace.instances, base.FakeQuadTrace.failures = [], []
        self.Pair = base.pair_trace_class()
        self.Pair.instances = []
        from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture

        for target, value in (('quad_draft_tp.PreparedQuadDFlashProposal', base.FakeQuadTrace),
                              ('dflash_proposal_trace.PreparedPackedDFlashProposal', self.Pair),
                              ('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture),
                              ('dflash_packed_proposal.select_packed_batched', Mock(side_effect=base.selected_tokens))):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: base.LOG.append(('sync',))))
        self.mesh = MESH4

    def coordinator(self):
        import dflash_packed_proposal_coordinator

        return dflash_packed_proposal_coordinator.PackedProposalCoordinator()

    def prepare(self, coordinator, bridges, **options):
        lines, loguru = base.logged()
        with loguru:
            prepared = coordinator.prepare(bridges, **options)
        return prepared, lines

    def test_four_steady_users_run_one_quad_through_the_twin_and_no_pair(self):
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        with patch.object(twin, 'refusal', wraps=twin.refusal) as refusal:
            prepared, lines = self.prepare(coordinator, bridges)
        refusal.assert_called_once()
        self.assertEqual(len(base.FakeQuadTrace.instances), 1)
        self.assertEqual(self.Pair.instances, [])
        self.assertEqual(base.FakeQuadTrace.instances[0].prepared, [(100, 101, 102, 103)])
        self.assertEqual(base.LOG, [('quad', (100, 101, 102, 103)), ('sync',), ('collect', 'quad'), ('audit', 1)])
        self.assertIn('[QUAD-DRAFT] round=1 built=1 ms=', lines[0])
        self.assertFalse([line for line in lines if 'fallback' in line or 'disabled' in line])
        self.assertEqual(coordinator.quad_rounds, 1)

    def test_the_live_banks_flag_the_image_sets_disables_it_once_with_the_reason_and_the_pairs_run(self):
        os.environ['QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'] = '1'
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        _, lines = self.prepare(coordinator, bridges)
        _, more = self.prepare(coordinator, bridges)
        disabled = [line for line in lines + more if line.startswith(pinned.DISABLED_MARKER)]
        self.assertEqual(disabled, ['[PINDIAG] quad draft disabled round=1 failures=0 reason='
                                    'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1_needs_the_four-card_fused_commit,_which_is_not_'
                                    'ported:_the_quad_copies_the_active_banks'])
        self.assertEqual(base.FakeQuadTrace.instances, [])
        self.assertEqual(len(self.Pair.instances), 2)

    def test_a_pair_mesh_is_refused_and_a_missing_requirement_is_named(self):
        bridges = base.quad_bridges(self.operations, SimpleNamespace(shape=[1, 2]))
        coordinator = self.coordinator()
        _, lines = self.prepare(coordinator, bridges)
        self.assertTrue(coordinator.quad_disabled)
        self.assertTrue(any('the_four_devices_are_not_on_the_(1,_4)_mesh' in line for line in lines), lines)
        os.environ['QWEN_FAST_PAIR_ROW_EXACT'] = '0'
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        _, lines = self.prepare(coordinator, bridges)
        self.assertTrue(any('reason=requires_QWEN_FAST_PAIR_ROW_EXACT=1' in line for line in lines), lines)

    def test_a_failure_falls_back_to_the_pairs_and_two_in_a_row_give_up(self):
        coordinator, bridges = self.coordinator(), base.quad_bridges(self.operations, self.mesh)
        base.FakeQuadTrace.failures = [True, True]
        _, lines = self.prepare(coordinator, bridges)
        self.assertEqual(len(self.Pair.instances), 2, 'the round fell back to the pairs')
        self.assertTrue(lines[0].startswith('[QUAD-DRAFT] fallback round=1 reason=RuntimeError:quad_capture_failed'))
        _, lines = self.prepare(coordinator, bridges)
        self.assertTrue(coordinator.quad_disabled)
        self.assertTrue(any(line.startswith('[PINDIAG] quad draft disabled round=2 failures=2 ') for line in lines))

    def test_three_live_and_two_live_rounds_are_todays_pairs_call_for_call(self):
        import dflash_packed_proposal_coordinator

        def run(quad, slots):
            del base.LOG[:]
            self.Pair.instances, base.FakeQuadTrace.instances = [], []
            with patch.dict(os.environ, {'QWEN_FAST_QUAD_DRAFT': quad}):
                coordinator = dflash_packed_proposal_coordinator.PackedProposalCoordinator()
                bridges = base.quad_bridges(self.operations, self.mesh, slots)
                _, lines = self.prepare(coordinator, bridges)
            return list(base.LOG), [base.strip_ms(line) for line in lines]

        for slots in ((0, 1, 3), (0, 1), (2, 3), (0, 2)):
            with self.subTest(slots=slots):
                self.assertEqual(run('1', slots), run('0', slots))
                self.assertEqual(base.FakeQuadTrace.instances, [])

    def test_the_pooled_shapes_carry_the_quad_slots_at_four_cards(self):
        import dflash_packed_proposal_coordinator as coordinator

        masks = coordinator.pooled_draft_mask_shapes(4, 16)
        self.assertEqual(masks[(0, 1, 2, 3)], (1, 1, 32, 2080))
        outputs = coordinator.pooled_draft_output_shapes(4, 16)
        spec = outputs[(0, 1, 2, 3)]
        self.assertEqual(spec['head'], (1, 1, 64, 16))
        self.assertEqual(spec['projected'], (1, 1, 64, 256))
        self.assertEqual(tuple(spec['chunks']), ((0, 32768), (32768, 62080)))
        with patch.dict(os.environ, {'QWEN_FAST_QUAD_DRAFT': '0'}):
            self.assertNotIn((0, 1, 2, 3), coordinator.pooled_draft_mask_shapes(4, 16))
            self.assertNotIn((0, 1, 2, 3), coordinator.pooled_draft_output_shapes(4, 16))


# ---------------------------------------------------------------------------------------------
# Shipping.
# ---------------------------------------------------------------------------------------------

class ShippingTests(unittest.TestCase):
    def test_the_twin_and_the_audit_reach_the_image_through_both_copy_lists_and_the_overlay(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        for name in ('quad_draft_tp.py', 'draft_singles_audit.py'):
            with self.subTest(module=name):
                self.assertIn(name, dockerfile_modules(dockerfile_text()))
                self.assertIn(name, context_modules())
                self.assertIn('scripts/ci/%s\n' % name, overlay)

    def test_the_cpu_suite_runs_these_files(self):
        text = CPU_WORKFLOW.read_text(encoding='utf-8')
        for name in ('test_quad_draft_tp4', 'test_draft_singles_audit', 'test_tp4_draft_window'):
            self.assertRegex(text, r'python -B -m unittest [^\n]*\b%s\b' % name)

    def test_every_new_file_is_lf(self):
        for name in ('quad_draft_tp.py', 'draft_singles_audit.py', 'test_quad_draft_tp4.py', 'test_draft_singles_audit.py',
                     'test_tp4_draft_window.py', 'speed_window_compare.py', 'c2_smoke_check.py', 'acceptance_report.py'):
            with self.subTest(name=name):
                self.assertNotIn(b'\r', (HERE / name).read_bytes())

    def test_the_twin_is_served_by_the_closure_scan(self):
        import test_tp4_closure_literals as closure

        self.assertIn('quad_draft_tp', closure.SERVED)
        self.assertIn('quad_draft', closure.NOT_SERVED)


if __name__ == '__main__':
    unittest.main()
