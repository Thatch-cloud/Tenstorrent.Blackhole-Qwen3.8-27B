"""The identity census of the multi-user SDPA launch's bindings (docs/tp4-combined-window.md, section on the interaction with Lever N).

sdpa_multi_tp binds, per segment reader, three buffers by Python identity (segment.positions, segment.metadata[0][1] and segment.cur_pos[0])
and bakes their addresses into its gather and mask programs at attach. Nothing re-checks them after a trace is captured (a replay re-issues
the device program and never comes back to Python), so the invariant is: no code path of the serving image assigns, rebinds, mutates in place or
re-constructs those attributes after the attach. Every site that does is listed here with the reason it is safe; a new one fails this test until
someone has audited it against the multi launch (and against Lever N's alternation of prefill steps with packed decode rounds, which is the
code that runs between two replays).

The census reads every non-test module the C2 overlay names (docker/qwen-c2-overlay.txt), by syntax tree:
  - an assignment, augmented assignment or deletion whose target is `<x>.positions`, `<x>.cur_pos` or `<x>.metadata` (also under subscripts:
    `<x>.metadata[:] = ...`, `<x>.cur_pos[i] = ...`);
  - `setattr(<x>, 'positions' | 'cur_pos' | 'metadata', ...)`;
  - a mutating call on one of them (append, extend, insert, pop, clear, remove);
  - a construction of PackedVerifierEngine, PackedExtentReplayReader or ExtentSegmentReader.
"""

import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NAMES = ('positions', 'cur_pos', 'metadata')
MUTATING = ('append', 'extend', 'insert', 'pop', 'clear', 'remove', '__setitem__')
CONSTRUCTED = ('PackedVerifierEngine', 'PackedExtentReplayReader', 'ExtentSegmentReader')

# (module, enclosing function, kind, attribute): why it is safe. The multi launch's own buffers are its own attributes (MultiBlock builds them).
AUDITED = {
    ('pooled_attention_replay.py', 'apply_sdpa_modes', 'subscript-assign', 'metadata'): 'the pair path\'s config rewrite at construction',
    ('serving_runtime.py', 'attach_combined_runtime', 'construct', 'PackedVerifierEngine'): 'the attach: every block is built once, before any trace',
    ('extent_attention_replay.py', 'ExtentSegmentReader.__init__', 'assign', 'cur_pos'): 'the segment reader constructor (the G1 reader)',
    ('extent_attention_replay.py', 'ExtentSegmentReader.__init__', 'assign', 'metadata'): 'the segment reader constructor (the G1 reader)',
    ('extent_attention_replay.py', 'ExtentSegmentReader.__init__', 'assign', 'positions'): 'the segment reader constructor (the G1 reader)',
    ('extent_attention_replay.py', 'ExtentSegmentReader.__init__', 'call-append', 'metadata'): 'the segment reader constructor (the G1 reader)',
    ('extent_attention_replay.py', 'PackedExtentReplayReader.__init__', 'construct', 'ExtentSegmentReader'): 'the packed reader constructor',
    ('serving_buffer_pool.py', 'PackedExtentStorage.__init__', 'assign', 'cur_pos'): 'the lent storage built once per block at the attach',
    ('model_batch.py', 'prepare_inputs', 'assign', 'positions'): 'the batch\'s own host positions tensor, not a reader attribute',
    ('model_batch.py', 'ModelBatch.__init__', 'assign', 'positions'): 'the batch constructor',
    ('model_batch.py', 'ModelBatch.__init__', 'construct', 'PackedExtentReplayReader'): 'the one packed reader per fixture, at construction',
    ('extent_attention_replay_tp.py', 'ExtentSegmentReader.__init__', 'assign', 'cur_pos'): 'the segment reader constructor: the identities multi binds',
    ('extent_attention_replay_tp.py', 'ExtentSegmentReader.__init__', 'assign', 'metadata'): 'the segment reader constructor: the identities multi binds',
    ('extent_attention_replay_tp.py', 'ExtentSegmentReader.__init__', 'assign', 'positions'): 'the segment reader constructor: the identities multi binds',
    ('extent_attention_replay_tp.py', 'ExtentSegmentReader.__init__', 'call-append', 'metadata'): 'the segment reader constructor: the identities multi binds',
    ('extent_attention_replay_tp.py', 'PackedExtentReplayReader.__init__', 'construct', 'ExtentSegmentReader'): 'the packed reader constructor',
    ('attention_replay_tp.py', 'ReplayAttentionReader.__init__', 'assign', 'metadata'): 'the pair path\'s reader constructor',
    ('attention_replay_tp.py', 'ReplayAttentionReader.__init__', 'assign', 'positions'): 'the pair path\'s reader constructor',
    ('attention_replay_tp.py', 'ReplayAttentionReader.__init__', 'call-append', 'metadata'): 'the pair path\'s reader constructor',
    ('packed_ordered_cache.py', 'ChainedOrderedCacheWriter.__init__', 'assign', 'positions'): 'the writer\'s own positions, built at construction',
    ('sdpa_long_tp.py', 'apply', 'subscript-assign', 'metadata'): 'the grid configurations rewrite the program config entries in place at the attach; never for multi',
    ('sdpa_multi_tp.py', 'MultiBlock.__init__', 'assign', 'cur_pos'): 'multi\'s OWN stacked cur_pos buffer (not a reader attribute)',
}
# The modules that run between two replays or around a prefill step: none of them may name a binding at all.
GUARDED = ('levern_policy.py', 'levern_scheduler.py', 'levern_platform.py', 'levern_route.py', 'serving_lifecycle.py',
           'serving_prefill_admission.py', 'qwen_prefix_scheduler_patch.py', 'qwen_prefix_registry.py', 'qwen_prefix_model_patch.py',
           'qwen_prefix_runner_patch.py', 'qwen_prefix_metrics.py', 'verify_prestage.py', 'serving_page_binding.py', 'serving_packed_step.py')


def root_attribute(node):
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Attribute) and node.attr in NAMES:
        return node.attr
    return None


def sites(source):
    """[(enclosing function path, kind, attribute)] of every binding-touching site of a module's source."""
    found = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def scope(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = scope

        def note(self, kind, attribute):
            found.append(('.'.join(self.stack), kind, attribute))

        def visit_Assign(self, node):
            for target in node.targets:
                for element in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                    attribute = root_attribute(element)
                    if attribute:
                        self.note('subscript-assign' if isinstance(element, ast.Subscript) else 'assign', attribute)
            self.generic_visit(node)

        def visit_AugAssign(self, node):
            attribute = root_attribute(node.target)
            if attribute:
                self.note('augassign', attribute)
            self.generic_visit(node)

        visit_AnnAssign = visit_AugAssign

        def visit_Delete(self, node):
            for target in node.targets:
                attribute = root_attribute(target)
                if attribute:
                    self.note('delete', attribute)
            self.generic_visit(node)

        def visit_Call(self, node):
            function = node.func
            if (isinstance(function, ast.Name) and function.id == 'setattr' and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value in NAMES):
                self.note('setattr', node.args[1].value)
            if isinstance(function, ast.Attribute) and function.attr in MUTATING and root_attribute(function.value):
                self.note('call-' + function.attr, root_attribute(function.value))
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else None
            if name in CONSTRUCTED:
                self.note('construct', name)
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return found


def overlay_modules():
    names = []
    for line in (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            source = line.split()[0]
            if source.startswith('scripts/ci/') and source.endswith('.py') and not Path(source).name.startswith('test_'):
                names.append(Path(source).name)
    return names


class IdentityCensusTests(unittest.TestCase):
    def test_every_binding_touching_site_of_the_image_is_audited_and_none_is_new(self):
        found = {}
        for name in overlay_modules():
            path = HERE / name
            if not path.exists():
                continue
            for function, kind, attribute in sites(path.read_text(encoding='utf-8')):
                found[(name, function, kind, attribute)] = found.get((name, function, kind, attribute), 0) + 1
        self.assertEqual(sorted(found), sorted(AUDITED),
                         'a module of the image gained or lost a site that assigns, mutates or constructs the readers\' positions, cur_pos or '
                         'metadata: audit it against sdpa_multi_tp (its programs bake those addresses at the attach) and Lever N, then update AUDITED')

    def test_the_modules_that_run_between_replays_name_no_binding_at_all(self):
        names = overlay_modules()
        for name in GUARDED:
            with self.subTest(name=name):
                self.assertIn(name, names)
                self.assertEqual(sites((HERE / name).read_text(encoding='utf-8')), [])

    def test_the_census_sees_every_form_it_claims(self):
        text = '\n'.join([
            'def a(r):', '    r.positions = 1', '    r.metadata[:] = []', '    r.cur_pos[0] = 3', '    r.cur_pos += 1', '    del r.positions',
            '    setattr(r, "metadata", [])', '    r.metadata.append(1)', '    PackedVerifierEngine()', '    mod.PackedExtentReplayReader()'])
        self.assertEqual(sorted(kind for _, kind, _ in sites(text)),
                         sorted(['assign', 'subscript-assign', 'subscript-assign', 'augassign', 'delete', 'setattr', 'call-append',
                                 'construct', 'construct']))

    def test_the_multi_launch_binds_the_three_identities_and_says_why_nothing_rechecks_them(self):
        source = (HERE / 'sdpa_multi_tp.py').read_text(encoding='utf-8')
        for word in ('segment.positions', 'segment.metadata[0][1]', 'segment.cur_pos[0]', 'self.bound = (positions, lent_tables, lent_positions)'):
            self.assertIn(word, source)


class ReboundGuardTests(unittest.TestCase):
    def test_the_reader_twin_asks_the_multi_launch_and_is_silent_without_it(self):
        sys.path.insert(0, str(HERE))
        import extent_attention_fold_tp as fold

        reader = object.__new__(fold.PackedExtentReplayReader)
        self.assertIsNone(reader.rebound_reason())                       # multi is None: the class attribute
        reader.multi = type('Multi', (), {'rebound': staticmethod(lambda: 'rebound after attach')})()
        self.assertEqual(reader.rebound_reason(), 'rebound after attach')
        reader.multi = type('Multi', (), {'rebound': staticmethod(lambda: None)})()
        self.assertIsNone(reader.rebound_reason())

    def test_the_verify_asks_before_every_replay_and_before_the_trace_runs(self):
        source = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        verify = source[source.index('    def verify(self, entries):'):]
        ask = verify.index('rebound_reason')
        self.assertLess(ask, verify.index('execute_trace(self.mesh, self.trace'))
        self.assertIn('raise RuntimeError(rebound)', verify[ask:ask + 600])


if __name__ == '__main__':
    unittest.main()
