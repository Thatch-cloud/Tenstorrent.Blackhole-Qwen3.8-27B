"""The G1 model graft's stage: pinned inputs, pinned output, scoped edits, refusals, the probe.

qwen_prefix_model_patch stages conversation prefix reuse into the image's model.py and
qwen36_vllm.py (TT prefix-reuse design section 2.0.1 item 4, section 2.2 "Model graft"). This
holds the stage itself, on the real files (fixtures/qwen36_model.py and fixtures/qwen36_vllm.py,
the IMG bytes: md5 e4ba08d9 and b5230935, the sha256 SOURCE_SHA256 pins):

  pins     the fixtures are the pinned originals; the stage's output is the pinned graft; stage()
           refuses any other input and writes nothing it did not pin, and a second run refuses;
  scope    only the three chunk-prefill methods of model.py and three methods of qwen36_vllm.py
           change - prefill_paged_slots and the RoPE line stay byte for byte - and everything else
           is added (the adapter after the imports, the new methods, the constants);
  reuse    the loops' resumable edit is lever_n_model_patch.patch_tp_replay's, called once;
  refuse   a Lever N tree (F5), an already staged tree, a drifted anchor and the bundle's copy of
           lever_n_model_patch, which lacks the eager-loop edit;
  names    the registry key is the scheduler graft's, the request-id kwarg and markers are what the
           runner patch and the gate harness use;
  probe    --probe reports original / staged / unknown per file (and the C2 graft files against a
           graft.sha256), with exit codes for the bring-up anchor probe.

Execution (exactness, guards, warmup, off-is-stock) is test_qwen_prefix_model_runtime's.
"""

import ast
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lever_n_model_patch  # noqa: E402
import qwen_prefix_model_patch as patcher  # noqa: E402
import qwen_prefix_registry  # noqa: E402
import qwen_prefix_runner_patch  # noqa: E402

MODEL = HERE / 'fixtures' / 'qwen36_model.py'
VLLM = HERE / 'fixtures' / 'qwen36_vllm.py'


def model_text():
    return MODEL.read_text(encoding='utf-8')


def vllm_text():
    return VLLM.read_text(encoding='utf-8')


_STAGED = []


def staged():
    if not _STAGED:   # each edit re-parses the 175 KB model file: stage once per process
        _STAGED.append((patcher.patch_model_source(model_text()), patcher.patch_vllm_source(vllm_text())))
    return _STAGED[0]


def methods(source):
    """Every method's source text (decorators to its last line), keyed Class.method."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    first = min([member.lineno] + [d.lineno for d in member.decorator_list])
                    out[node.name + '.' + member.name] = ''.join(lines[first - 1:member.end_lineno])
    return out


def members(source, cls):
    prefix = cls + '.'
    return {name[len(prefix):]: text for name, text in methods(source).items() if name.startswith(prefix)}


def model_methods(source):
    return members(source, 'Qwen36Model')


def vllm_methods(source):
    return members(source, 'Qwen36ForCausalLM')


class Pins(unittest.TestCase):
    def test_the_fixtures_are_the_pinned_originals(self):
        for path, name in ((MODEL, patcher.MODEL_FILE), (VLLM, patcher.VLLM_FILE)):
            data = path.read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), patcher.SOURCE_SHA256[name])
            self.assertNotIn(b'\r', data)
        self.assertEqual(hashlib.md5(MODEL.read_bytes()).hexdigest()[:8], 'e4ba08d9')
        self.assertEqual(hashlib.md5(VLLM.read_bytes()).hexdigest()[:8], 'b5230935')

    def test_the_output_is_the_pinned_graft(self):
        model, vllm = patcher.patch_tree_bytes(MODEL.read_bytes(), VLLM.read_bytes())
        got = {patcher.MODEL_FILE: hashlib.sha256(model).hexdigest(),
               patcher.VLLM_FILE: hashlib.sha256(vllm).hexdigest()}
        self.assertEqual(got, patcher.PATCHED_SHA256,
                         'the staged bytes moved; if on purpose, pin PATCHED_SHA256 = %r' % (got,))
        for text in (model, vllm):
            self.assertNotIn(b'\r', text)
            ast.parse(text)

    def stage_into(self, root):
        tree = Path(root) / 'tt'
        tree.mkdir()
        (tree / patcher.MODEL_FILE).write_bytes(MODEL.read_bytes())
        (tree / patcher.VLLM_FILE).write_bytes(VLLM.read_bytes())
        return tree

    def test_stage_writes_both_files_in_place_and_a_second_run_refuses(self):
        with tempfile.TemporaryDirectory() as root:
            tree = self.stage_into(root)
            self.assertEqual(patcher.stage(tree), patcher.PATCHED_SHA256)
            model, vllm = staged()
            self.assertEqual((tree / patcher.MODEL_FILE).read_bytes(), model.encode('utf-8'))
            self.assertEqual((tree / patcher.VLLM_FILE).read_bytes(), vllm.encode('utf-8'))
            with self.assertRaisesRegex(ValueError, 'is not the pinned original'):
                patcher.stage(tree)

    def test_a_drifted_source_is_refused_and_nothing_is_written(self):
        with tempfile.TemporaryDirectory() as root:
            tree = self.stage_into(root)
            drifted = MODEL.read_bytes().replace(b'_PREFILL_MASK_BUCKETS', b'_PREFILL_MASK_BUCKETS ', 1)
            (tree / patcher.MODEL_FILE).write_bytes(drifted)
            with self.assertRaisesRegex(ValueError, 'model.py sha256 .* is not the pinned original'):
                patcher.stage(tree)
            self.assertEqual((tree / patcher.MODEL_FILE).read_bytes(), drifted)
            self.assertEqual((tree / patcher.VLLM_FILE).read_bytes(), VLLM.read_bytes())

    def test_a_drifted_output_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as root:
            tree = self.stage_into(root)
            with mock.patch.dict(patcher.PATCHED_SHA256, {patcher.VLLM_FILE: '0' * 64}):
                with self.assertRaisesRegex(ValueError, 'is not the pinned graft'):
                    patcher.stage(tree)
            self.assertEqual((tree / patcher.MODEL_FILE).read_bytes(), MODEL.read_bytes())

    def test_output_dir_leaves_the_tree_alone(self):
        with tempfile.TemporaryDirectory() as root:
            tree = self.stage_into(root)
            patcher.stage(tree, Path(root) / 'out')
            self.assertEqual((tree / patcher.MODEL_FILE).read_bytes(), MODEL.read_bytes())
            self.assertEqual(hashlib.sha256((Path(root) / 'out' / patcher.MODEL_FILE).read_bytes()).hexdigest(),
                             patcher.PATCHED_SHA256[patcher.MODEL_FILE])


class Scope(unittest.TestCase):
    def test_model_changes_only_the_three_chunk_prefill_methods(self):
        everything = methods(staged()[0])
        self.assertEqual({name for name in everything if not name.startswith('Qwen36Model.')},
                         {'_QwenPrefixRow.__init__'})
        before, after = model_methods(model_text()), model_methods(staged()[0])
        changed = {name for name in before if before[name] != after[name]}
        self.assertEqual(changed, {'prefill_traced_chunked', '_prefill_chunked_eager_tp', '_prefill_traced_chunked_tp'})
        added = set(after) - set(before)
        self.assertEqual(added, {'_qwen_prefix_prefill_slots', '_qwen_prefix_gdn_layers',
                                 '_qwen_prefix_program_cache_entries', '_qwen_prefix_read_scratch',
                                 '_qwen_prefix_capture', '_qwen_prefix_restore', '_qwen_prefix_warm_restore',
                                 '_qwen_prefix_audit'})
        self.assertEqual(before['prefill_paged_slots'], after['prefill_paged_slots'])

    def test_the_batched_prefill_entries_are_the_stock_ones(self):
        """The C2 fast path's prefill capture enumerates prefill_paged_slots* on the model and
        refuses any name outside BATCHED_PREFILL_ENTRIES, on every profile of the image (review
        finding 1). The stage adds none; test_qwen_prefix_model_runtime runs the capture itself."""
        import dflash_prefill_window
        before, after = model_methods(model_text()), model_methods(staged()[0])
        entries = lambda names: {name for name in names if name.startswith('prefill_paged_slots')}  # noqa: E731
        self.assertEqual(entries(after), entries(before))
        self.assertLessEqual(entries(after), set(dflash_prefill_window.BATCHED_PREFILL_ENTRIES))
        self.assertNotIn('def prefill_paged_slots', patcher.MODEL_METHODS)

    def test_the_new_guards_raise_and_are_not_bare_asserts(self):
        """python -O strips assert statements; every guard the stage adds must survive it."""
        blocks = {'MODEL_ADAPTER': patcher.MODEL_ADAPTER,
                  'MODEL_METHODS': 'class _Methods:\n' + patcher.MODEL_METHODS,
                  'VLLM_WARM_METHOD': 'class _Methods:\n' + patcher.VLLM_WARM_METHOD}
        for name, block in blocks.items():
            asserts = [node.lineno for node in ast.walk(ast.parse(block)) if isinstance(node, ast.Assert)]
            self.assertEqual(asserts, [], name)
        for name in ('ENTRY_RESUME', 'EAGER_TAIL_NEW', 'TRACED_CAPTURE_NEW', 'VLLM_ENTRY_NEW',
                     'VLLM_BATCHED_CALL_NEW', 'VLLM_SLOTS_NEW', 'VLLM_WARM_NEW'):
            self.assertNotIn('assert ', getattr(patcher, name), name)

    def test_model_module_level_additions_sit_after_the_imports(self):
        stock, graft = model_text(), staged()[0]
        head = stock.index(patcher.IMPORT_ANCHOR) + len(patcher.IMPORT_ANCHOR)
        self.assertEqual(graft[:head], stock[:head])
        self.assertTrue(graft[head:].startswith(patcher.MODEL_ADAPTER))
        self.assertTrue(graft[head + len(patcher.MODEL_ADAPTER):].startswith(stock[head:stock.index('class Qwen36Model')]))

    def test_rope_stays_staged_for_the_whole_prompt(self):
        entry = model_methods(staged()[0])['prefill_traced_chunked']
        self.assertEqual(entry.count(patcher.ENTRY_ROPE), 1)
        self.assertNotIn('if start == 0:', entry)

    def test_the_resume_kwargs_reach_the_loops_only_when_reuse_engages(self):
        entry = model_methods(staged()[0])['prefill_traced_chunked']
        self.assertEqual(entry.count('**_qwen_resume'), 2)
        self.assertIn('        _qwen_resume = {}\n        if start or capture_at:\n', entry)

    def test_both_loops_are_resumable_and_capture_and_the_eager_tail_is_guarded(self):
        after = model_methods(staged()[0])
        for name in ('_prefill_chunked_eager_tp', '_prefill_traced_chunked_tp'):
            body = after[name]
            self.assertEqual(body.count('for c in range(chunk_from, chunk_to):'), 1, name)
            self.assertEqual(body.count('capture_at=None, on_capture=None'), 1, name)
            self.assertEqual(body.count('on_capture((c + 1) * chunk_size)'), 1, name)
        self.assertIn(patcher.EAGER_TAIL_NEW, after['_prefill_chunked_eager_tp'])
        self.assertIn(patcher.TRACED_CAPTURE_NEW, after['_prefill_traced_chunked_tp'])

    def test_vllm_changes_only_the_entry_the_batched_prefill_and_the_warmup(self):
        stock, graft = methods(vllm_text()), methods(staged()[1])
        self.assertEqual({k: v for k, v in stock.items() if not k.startswith('Qwen36ForCausalLM.')},
                         {k: v for k, v in graft.items() if not k.startswith('Qwen36ForCausalLM.')})
        before, after = vllm_methods(vllm_text()), vllm_methods(staged()[1])
        changed = {name for name in before if before[name] != after[name]}
        self.assertEqual(changed, {'prefill_forward', '_prefill_forward_tp_batched', 'warmup_model_prefill'})
        self.assertEqual(set(after) - set(before), {'_qwen_prefix_warm'})
        graft = staged()[1]
        self.assertEqual(graft.count(patcher.VLLM_CONSTANTS), 1)
        self.assertEqual(graft.count('"supports_prefix_caching": _QWEN_PREFIX_REUSE,'), 1)
        self.assertNotIn('"supports_prefix_caching": False', graft)

    def test_the_stock_calls_survive_on_the_off_path(self):
        after = vllm_methods(staged()[1])
        self.assertIn(patcher.VLLM_BATCHED_CALL_OLD, after['prefill_forward'])
        self.assertIn('            host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)\n',
                      after['_prefill_forward_tp_batched'])


class Reuse(unittest.TestCase):
    def test_the_loops_come_from_lever_n_patch_tp_replay_once(self):
        with mock.patch.object(lever_n_model_patch, 'patch_tp_replay',
                               wraps=lever_n_model_patch.patch_tp_replay) as spy:
            patcher.patch_model_source(model_text())
        spy.assert_called_once()

    def test_the_bundle_copy_of_lever_n_model_patch_is_refused(self):
        saved = lever_n_model_patch._patch_replay_loop
        try:
            del lever_n_model_patch._patch_replay_loop
            with self.assertRaisesRegex(ValueError, 'the bundle copy, not HEAD'):
                patcher.patch_model_source(model_text())
        finally:
            lever_n_model_patch._patch_replay_loop = saved


class Refusals(unittest.TestCase):
    def test_a_lever_n_model_tree_is_refused(self):
        lever_n = lever_n_model_patch.patch_model(model_text())
        with self.assertRaisesRegex(ValueError, 'refuses a Lever N tree'):
            patcher.patch_model_source(lever_n)
        loops_only = lever_n_model_patch.patch_tp_replay(model_text())
        with self.assertRaisesRegex(ValueError, 'resumable chunk loops'):
            patcher.patch_model_source(loops_only)

    def test_a_lever_n_vllm_entry_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'refuses a Lever N tree'):
            patcher.patch_vllm_source(lever_n_model_patch.patch_vllm_entry(vllm_text()))

    def test_an_already_staged_tree_is_refused(self):
        model, vllm = staged()
        with self.assertRaisesRegex(ValueError, 'already carries the prefix-reuse graft'):
            patcher.patch_model_source(model)
        with self.assertRaisesRegex(ValueError, 'already carries the prefix-reuse graft'):
            patcher.patch_vllm_source(vllm)

    def test_a_drifted_anchor_names_the_edit(self):
        drifted = model_text().replace('        if tail_real > 0:\n            ttnn.deallocate(last_hidden)\n',
                                       '        if tail_real > 0:\n            ttnn.deallocate(last_hidden)  # moved\n', 1)
        self.assertNotEqual(drifted, model_text())
        with self.assertRaisesRegex(ValueError, 'eager capture hook and tail guard: expected one'):
            patcher.patch_model_source(drifted)
        drifted = vllm_text().replace('"supports_prefix_caching": False,', '"supports_prefix_caching": True,', 1)
        with self.assertRaisesRegex(ValueError, 'capability matched 0 times'):
            patcher.patch_vllm_source(drifted)


class SwitchOff(unittest.TestCase):
    def test_switch_off_the_staged_files_run_as_the_stock_ones(self):
        """qwen_prefix_stage's switch-off rule, held here: with QWEN_PREFIX_REUSE unset the staged
        model.py and qwen36_vllm.py make the stock files' ttnn calls and results (prefill traced and
        eager, warmup) and the capability is off. The executed comparison is
        test_qwen_prefix_model_runtime.OffIsStock, run from this module too."""
        import test_qwen_prefix_model_runtime as runtime

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(runtime.OffIsStock)
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.failures + result.errors, [])
        self.assertGreaterEqual(result.testsRun, 6)

    def test_the_image_stage_entries_are_the_pinned_edits(self):
        """qwen_prefix_stage.STAGES runs patch_model / patch_vllm_entry: the same bytes stage() writes,
        refused on an input or output off the pins."""
        model, vllm = model_text(), vllm_text()
        self.assertEqual(patcher.patch_model(model), patcher.patch_model_source(model))
        self.assertEqual(patcher.patch_vllm_entry(vllm), patcher.patch_vllm_source(vllm))
        with self.assertRaisesRegex(ValueError, 'not the pinned original'):
            patcher.patch_model(model + '\n')
        with mock.patch.object(patcher, 'PATCHED_SHA256', dict(patcher.PATCHED_SHA256, **{patcher.VLLM_FILE: '0' * 64})):
            with self.assertRaisesRegex(ValueError, 'not the pinned graft'):
                patcher.patch_vllm_entry(vllm)


class Names(unittest.TestCase):
    def test_the_registry_key_is_the_scheduler_grafts(self):
        self.assertEqual(patcher.REGISTRY_KEY, qwen_prefix_registry.REGISTRY_KEY)
        self.assertIn('_QWEN_PREFIX_REGISTRY_KEY = "%s"' % patcher.REGISTRY_KEY, staged()[0])
        self.assertEqual(patcher.CHUNK, qwen_prefix_registry.CHUNK)
        self.assertIn('_QWEN_PREFIX_CHUNK = %d' % patcher.CHUNK, staged()[0])

    def test_the_request_id_kwarg_is_the_one_the_runner_passes(self):
        # submit_prefill hands the row ids over as this kwarg (qwen_prefix_runner_patch.SUBMIT_NEW);
        # a different name here leaves every row without a request id: a hit would assert.
        self.assertEqual(patcher.REQ_IDS_KWARG, qwen_prefix_runner_patch.REQUEST_IDS_KWARG)
        self.assertEqual(patcher.REQ_IDS_KWARG, qwen_prefix_registry.REQUEST_IDS_KWARG)
        self.assertIn('kwargs["%s"] = ' % patcher.REQ_IDS_KWARG, qwen_prefix_runner_patch.SUBMIT_NEW)

    def test_the_request_id_kwarg_and_the_markers_are_in_the_staged_files(self):
        model, vllm = staged()
        self.assertIn('_QWEN_PREFIX_REQ_IDS_KWARG = "%s"' % patcher.REQ_IDS_KWARG, vllm)
        self.assertIn('f"' + patcher.MARKER_ROW + '{u} ', model)
        self.assertIn('f"' + patcher.MARKER_WARM + '{chosen} ', model)
        self.assertEqual(model.count('f"' + patcher.MARKER_AUDIT + ' req='), 2)

    def test_the_stage_is_python_3_7_syntax_and_the_graft_python_3_10(self):
        source = (HERE / 'qwen_prefix_model_patch.py').read_text(encoding='utf-8')
        ast.parse(source, feature_version=(3, 7))
        for text in staged():
            ast.parse(text, feature_version=(3, 10))


class Probe(unittest.TestCase):
    def run_main(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = patcher.main(list(argv))
        return code, out.getvalue()

    def test_original_then_staged(self):
        with tempfile.TemporaryDirectory() as root:
            tree = Pins().stage_into(root)
            code, out = self.run_main('--tree', str(tree), '--probe', '--expect', 'original')
            self.assertEqual(code, 0)
            self.assertIn('original %s model.py' % patcher.SOURCE_SHA256[patcher.MODEL_FILE], out)
            self.assertEqual(self.run_main('--tree', str(tree), '--probe', '--expect', 'staged')[0], 1)
            code, out = self.run_main('--tree', str(tree))
            self.assertEqual(code, 0)
            self.assertIn('staged %s qwen36_vllm.py' % patcher.PATCHED_SHA256[patcher.VLLM_FILE], out)
            code, out = self.run_main('--tree', str(tree), '--probe', '--expect', 'staged')
            self.assertEqual(code, 0)
            (tree / patcher.VLLM_FILE).write_text('# something else\n', encoding='utf-8')
            code, out = self.run_main('--tree', str(tree), '--probe')
            self.assertEqual(code, 0)
            self.assertIn('unknown', out)
            self.assertEqual(self.run_main('--tree', str(tree), '--probe', '--expect', 'staged')[0], 1)

    def test_graft_sums_check_the_c2_graft_files(self):
        with tempfile.TemporaryDirectory() as root:
            tree = Pins().stage_into(root)
            (tree / 'gdn').mkdir()
            (tree / 'gdn' / 'tp.py').write_bytes(b'grafted gdn\n')
            (tree / 'layer.py').write_bytes(b'image layer\n')
            sums = Path(root) / 'graft.sha256'
            sums.write_text('%s  graft/gdn/tp.py\n%s  graft/layer.py\n%s  graft/mlp.py\n%s  graft/gdn/tp.py.orig\n' % (
                hashlib.sha256(b'grafted gdn\n').hexdigest(), hashlib.sha256(b'grafted layer\n').hexdigest(),
                '0' * 64, '1' * 64), encoding='utf-8')
            report = patcher.probe(tree, sums)
            self.assertEqual(report['gdn/tp.py']['state'], 'grafted')
            self.assertEqual(report['layer.py']['state'], 'unknown')
            self.assertEqual(report['mlp.py']['state'], 'missing')
            self.assertNotIn('gdn/tp.py.orig', report)
            self.assertEqual(self.run_main('--tree', str(tree), '--probe', '--graft-sums', str(sums))[0], 1)

    def test_the_repository_graft_sums_parse(self):
        sums = HERE.parent.parent / 'docker' / 'qwen-c2-graft' / 'graft.sha256'
        with tempfile.TemporaryDirectory() as root:
            tree = Pins().stage_into(root)
            report = patcher.probe(tree, sums)
        self.assertIn('gdn/tp.py', report)
        self.assertIn('attention/tp.py', report)


if __name__ == '__main__':
    unittest.main()
