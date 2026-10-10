"""The integrator's hooks of the op-fusion programme (docs/tp4-fusion.md): the three WP6 levers in the drafter's ATTENTION branch (draft_attention_branch, WP7's file) and WP7's
permutation levers at the octo shapes (octo_draft_tp.octo_fold_query / octo_unfold_output), each hook lazy, strict and flag-guarded as the ones the packages wrote.

Held, for each:
  - flags off (unset and '0'): the op sequence is the one the file had BEFORE the hook (the merge commit's blob is run beside it on the same fake), bit for bit, and none of the
    new modules is imported;
  - flag on: the hook reaches the lever's module with the arguments the package's notes spell out (choose(gather, quad, 'attention'), residual(..., site='attention'),
    grid_for(..., (8, 10), 2, rows, kernel, site='wo'), fold_query / unfold_output at site='octo' with halves=2, users=4, block=8) and ONLY the replaced ops leave the sequence;
  - the octo fold and unfold launch for real on the executing fake and return the served fold's and unfold's bits.

    py -3.11 -B -m unittest test_fusion_hooks      (from scripts/ci)
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import draft_permute_tp as perm  # noqa: E402
import octo_draft_tp  # noqa: E402
import test_draft_permute_tp as permute_tests  # noqa: E402
import test_quad_draft as base  # noqa: E402
import test_quad_draft_tp4 as quad_tests  # noqa: E402
from test_pair_row_exact import Device, keep  # noqa: E402
from tp_test_support import four_cards  # noqa: E402

BEFORE_THE_HOOKS = '4f29696d'      # the merge of fx-wp7: the last commit before the integrator's hooks
ATTENTION_FLAGS = ('QWEN_FAST_DRAFT_REDUCE', 'QWEN_FAST_DRAFT_TAIL', 'QWEN_FAST_DRAFT_MM_GRID')
NEW_MODULES = ('draft_reduce_tp', 'draft_tail_tp', 'draft_mmgrid_tp', 'draft_fusion_tp')


def blob(path):
    try:
        return subprocess.run(['git', '-C', str(HERE), 'show', '%s:scripts/ci/%s' % (BEFORE_THE_HOOKS, path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=True).stdout.decode('utf-8')
    except (OSError, subprocess.CalledProcessError):
        return None


def load_before(path, name):
    """The module `path` as it was before the hooks, loaded under `name` (None where the history is not there)."""
    text = blob(path)
    if text is None:
        return None
    module = types.ModuleType(name)
    module.__file__ = str(HERE / path)
    exec(compile(text, '%s@%s' % (path, BEFORE_THE_HOOKS), 'exec'), module.__dict__)
    return module


def clean_environment(**values):
    environment = {name: v for name, v in os.environ.items() if name not in ATTENTION_FLAGS + ('QWEN_FAST_DRAFT_PERMUTE',)
                   and not name.endswith('_AUDIT')}
    environment.update(values)
    return patch.dict(os.environ, environment, clear=True)


def forget_new_modules():
    for name in NEW_MODULES:
        sys.modules.pop(name, None)


def attention(**flags):
    """The op log of execute_attention_branch on the quad pass under `flags` (the event tuples)."""
    with four_cards(), clean_environment(**flags):
        return quad_tests.attention_run4(quad=True)


class AttentionOffTests(unittest.TestCase):
    def test_flags_off_is_the_op_sequence_the_file_had_before_the_hooks_and_imports_none_of_the_new_modules(self):
        before = load_before('draft_attention_branch.py', '_attention_before_the_hooks')
        if before is None:
            self.skipTest('the history of %s is not here' % BEFORE_THE_HOOKS)
        forget_new_modules()
        unset = attention()
        zero = attention(**{name: '0' for name in ATTENTION_FLAGS})
        self.assertEqual(unset, zero)
        self.assertEqual([name for name in NEW_MODULES if name in sys.modules], [], 'a flags-off process loads none of the lever modules')
        with patch.dict(sys.modules, {'draft_attention_branch': before}):
            beside = attention()
        self.assertEqual(unset, beside)
        self.assertGreater(len(unset), 100)

    def test_the_pair_and_the_single_user_pass_are_unchanged_too(self):
        before = load_before('draft_attention_branch.py', '_attention_before_the_hooks')
        if before is None:
            self.skipTest('the history of %s is not here' % BEFORE_THE_HOOKS)
        with four_cards(), clean_environment():
            now = quad_tests.attention_run4(quad=False)
        # (the old module is in sys.modules BEFORE the four-card install, which rebinds the pinned helpers in every loaded module, as it does in the served process)
        with patch.dict(sys.modules, {'draft_attention_branch': before}), four_cards(), clean_environment():
            then = quad_tests.attention_run4(quad=False)
        self.assertEqual(now, then)

    def test_each_flag_is_strict_and_an_audit_needs_its_lever(self):
        for name in ATTENTION_FLAGS:
            with self.subTest(flag=name):
                with self.assertRaises(ValueError) as caught:
                    attention(**{name: '2'})
                self.assertIn(name, str(caught.exception))
                with self.assertRaises(ValueError) as caught:
                    attention(**{name + '_AUDIT': '1'})
                self.assertIn('needs', str(caught.exception))


class AttentionHookTests(unittest.TestCase):
    def test_reduce_hands_the_quads_gather_to_the_lever_with_the_attention_site(self):
        import draft_reduce_tp

        seen = []

        def choose(served, quad, site):
            seen.append((served, quad, site))
            return served

        off = attention()
        with patch.object(draft_reduce_tp, 'choose', side_effect=choose):
            on = attention(QWEN_FAST_DRAFT_REDUCE='1')
        self.assertEqual(len(seen), 1)
        served, quad, site = seen[0]
        self.assertEqual(site, 'attention')
        self.assertIsNotNone(quad, 'the quad pass')
        self.assertEqual(served.__name__, 'gather_add_projection')
        self.assertEqual(on, off, 'choose returning the served gather leaves the sequence alone')

    def test_reduce_on_the_single_user_pass_is_asked_with_no_quad_and_answers_the_served_function(self):
        import draft_reduce_tp

        with four_cards(), clean_environment(QWEN_FAST_DRAFT_REDUCE='1'):
            with patch.object(draft_reduce_tp, 'choose', side_effect=lambda served, quad, site: served) as choose:
                quad_tests.attention_run4(quad=False)
        self.assertEqual((choose.call_args.args[1], choose.call_args.args[2]), (None, 'attention'))

    def test_the_real_choose_is_the_served_function_itself_without_a_quad(self):
        import draft_reduce_tp

        with clean_environment(QWEN_FAST_DRAFT_REDUCE='1'):
            self.assertIs(draft_reduce_tp.choose(len, None, 'attention'), len)

    def test_tail_replaces_the_residual_tail_with_one_launch_at_the_attention_site(self):
        import draft_tail_tp

        off = attention()
        seen = {}

        def residual(operations, mesh, finished, hidden, retain, *, site):
            seen.update(site=site, finished=finished.shape, hidden=hidden.shape, retain=retain)
            operations.log('residual-launch', finished.shape)
            return base.Tensor(finished.shape, operations.bfloat16)

        with patch.object(draft_tail_tp, 'residual', side_effect=residual):
            on = attention(QWEN_FAST_DRAFT_TAIL='1')
        self.assertEqual((seen['site'], seen['finished'], seen['hidden']), ('attention', (1, 1, 64, 5120), (1, 1, 64, 5120)))
        tail = [('typecast', (1, 1, 64, 5120), 'fp32'), ('typecast', (1, 1, 64, 5120), 'fp32'), ('add', (1, 1, 64, 5120), (1, 1, 64, 5120), 'fp32'),
                ('typecast', (1, 1, 64, 5120), 'bf16')]
        self.assertEqual(off[-5:-1], tail)
        self.assertEqual(on, off[:-5] + [('residual-launch', (1, 1, 64, 5120)), off[-1]], 'only the four tail ops leave the sequence')

    def test_mm_grid_widens_only_the_wo_projections_program_grid(self):
        import draft_mmgrid_tp

        off = attention()
        seen = {}

        def grid_for(operations, mesh, value, weight, served_grid, columns, rows, kernel, site=None):
            seen.update(served=served_grid, columns=columns, rows=rows, kernel=kernel, site=site, weight=weight.shape)
            return (13, 10)

        with patch.object(draft_mmgrid_tp, 'grid_for', side_effect=grid_for):
            on = attention(QWEN_FAST_DRAFT_MM_GRID='1')
        self.assertEqual((seen['served'], seen['columns'], seen['rows'], seen['site']), ((8, 10), 2, 64, 'wo'))
        self.assertEqual(seen['kernel'], 'kernel')
        self.assertEqual(seen['weight'][1], 5120, 'the output projection (heads x 128, 5120)')
        changed = [(a, b) for a, b in zip(off, on) if a != b]
        self.assertEqual(len(off), len(on))
        self.assertEqual(len(changed), 2, 'the program and the matmul that carries it')
        for a, b in changed:
            self.assertEqual(a[0], b[0])
            self.assertIn("('compute_with_storage_grid_size', (8, 10))", repr(a))
            self.assertIn("('compute_with_storage_grid_size', (13, 10))", repr(b))
            self.assertEqual(repr(a).replace('(8, 10)', '(13, 10)'), repr(b), 'nothing but the grid field differs')

    def test_all_three_together_reach_all_three_modules_once(self):
        import draft_mmgrid_tp
        import draft_reduce_tp
        import draft_tail_tp

        with patch.object(draft_reduce_tp, 'choose', side_effect=lambda served, quad, site: served) as choose, \
                patch.object(draft_tail_tp, 'residual', side_effect=lambda operations, mesh, finished, hidden, retain, site: base.Tensor(finished.shape)) as residual, \
                patch.object(draft_mmgrid_tp, 'grid_for', side_effect=lambda *a, **k: (13, 10)) as grid:
            attention(QWEN_FAST_DRAFT_REDUCE='1', QWEN_FAST_DRAFT_TAIL='1', QWEN_FAST_DRAFT_MM_GRID='1')
        self.assertEqual((choose.call_count, residual.call_count, grid.call_count), (1, 1, 1))

    def test_the_audit_flags_reach_the_modules_through_the_branch_check(self):
        """The audits run inside the lever modules (draft_fusion_tp): the branch only holds that an audit has its lever and each flag parses."""
        import draft_reduce_tp

        with patch.object(draft_reduce_tp, 'choose', side_effect=lambda served, quad, site: served) as choose:
            attention(QWEN_FAST_DRAFT_REDUCE='1', QWEN_FAST_DRAFT_REDUCE_AUDIT='1')
        self.assertEqual(choose.call_count, 1)


class OctoHookTests(permute_tests.Quiet):
    GEOMETRY = dict(halves=2, users=4, block=8)

    def query(self, seed=1):
        return permute_tests.bits(torch.Generator().manual_seed(seed), (1, permute_tests.HEADS, 64, permute_tests.DIM))

    def folded(self, seed=2):
        heads = permute_tests.KV * 2 * 4 * (permute_tests.HEADS // permute_tests.KV)
        return permute_tests.bits(torch.Generator().manual_seed(seed), (1, heads, 32, permute_tests.DIM))

    def test_flags_off_the_fold_and_the_unfold_are_the_served_ops_and_the_lever_is_never_asked(self):
        before = load_before('octo_draft_tp.py', '_octo_before_the_hooks')
        with four_cards():
            logs = []
            for module in (octo_draft_tp, before):
                if module is None:
                    continue
                for value in (None, '0'):
                    environment = {} if value is None else {perm.FLAG: value}
                    with clean_environment(**environment), patch.object(perm, 'fold_query', side_effect=AssertionError('flag off')), \
                            patch.object(perm, 'unfold_output', side_effect=AssertionError('flag off')):
                        ops, owned = permute_tests.CanonOps(), []
                        folded = module.octo_fold_query(ops, Device(self.query()), keep(owned))
                        module.octo_unfold_output(ops, folded, keep(owned))
                        logs.append(ops.calls)
        self.assertGreater(len(logs[0]), 20)
        self.assertTrue(all(log == logs[0] for log in logs), 'unset, 0 and the file before the hook: the same ops')

    def test_flag_on_each_site_calls_the_module_with_the_octo_geometry_and_a_served_reference(self):
        with four_cards(), clean_environment(**{perm.FLAG: '1'}):
            cases = ((octo_draft_tp.octo_fold_query, Device(self.query()), 'fold_query'), (octo_draft_tp.octo_unfold_output, Device(self.folded()), 'unfold_output'))
            for function, tensor, target in cases:
                with self.subTest(function=function.__name__), patch.object(perm, target, return_value='engaged') as hook:
                    ops, owned = permute_tests.CanonOps(), []
                    self.assertEqual(function(ops, tensor, keep(owned)), 'engaged')
                    hook.assert_called_once()
                    self.assertEqual(hook.call_args.kwargs['site'], 'octo')
                    self.assertEqual({key: hook.call_args.kwargs[key] for key in ('halves', 'users', 'block')}, self.GEOMETRY)
                    self.assertEqual(ops.calls, [], 'nothing ran before the hook')
                    with perm.served_only():
                        self.assertIsNotNone(hook.call_args.kwargs['served']())
                    self.assertTrue(ops.calls, 'the served reference runs the served ops')

    def test_flag_on_the_fold_and_the_unfold_launch_and_return_the_served_bits(self):
        query, folded = self.query(), self.folded()
        for function, tensor, served in ((octo_draft_tp.octo_fold_query, query, permute_tests.fold_served('octo', query)),
                                         (octo_draft_tp.octo_unfold_output, folded, permute_tests.unfold_served('octo', folded))):
            with self.subTest(function=function.__name__):
                ops, owned = permute_tests.ExecutingOperations(), []
                with four_cards(), clean_environment(**{perm.FLAG: '1'}):
                    result = function(ops, ops.from_logical(tensor), keep(owned))
                mine = ops.to_logical(result)
                self.assertEqual(tuple(mine.shape), tuple(served.shape))
                self.assertTrue(permute_tests.same(mine, served))
                self.assertEqual(len(ops.programs), 1, 'one launch')

    def test_the_served_reference_inside_the_hook_does_not_launch_again(self):
        """octo_fold_query's lambda calls octo_fold_query itself: inside perm.served_only() hook_enabled() is false, so the recursion ends at the served body."""
        with four_cards(), clean_environment(**{perm.FLAG: '1'}), perm.served_only():
            ops, owned = permute_tests.CanonOps(), []
            with patch.object(perm, 'fold_query', side_effect=AssertionError('launched inside a served reference')):
                octo_draft_tp.octo_fold_query(ops, Device(self.query()), keep(owned))
            self.assertTrue(ops.calls)

    def test_the_lever_flag_is_read_the_way_the_quad_hook_reads_it(self):
        guard = "os.environ.get('QWEN_FAST_DRAFT_PERMUTE', '0') != '0'"
        self.assertEqual((HERE / 'octo_draft_tp.py').read_text(encoding='utf-8').count(guard), 2)
        self.assertEqual((HERE / 'quad_draft_tp.py').read_text(encoding='utf-8').count(guard), 2)


if __name__ == '__main__':
    unittest.main()
