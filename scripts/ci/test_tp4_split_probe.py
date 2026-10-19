"""tp4_split_probe on the CPU: the split-invariance probe's arms, comparisons and verdict on an in-memory fake ttnn and a toy model.

The toy model processes tokens one position at a time into a GDN-like state and writes one K and V row per position through the page
table, exactly where the image's model writes them. 'exact' mode is split-invariant by construction (the probe must say IDENTICAL);
'sensitive' rounds the state where a call ends off a 2048 boundary and perturbs the rows the masked bucket writes at a start off 2048
(the probe must say DIFFERS and place the KV differences in [R, H)); 'noisy' makes the cold run non-deterministic (the controls must FAIL).

Run at py 3.11 with torch: `python3 -B -m unittest test_tp4_split_probe` from scripts/ci."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import tp4_split_probe as probe  # noqa: E402

CHIPS = 4
HEAD_DIM = 8
VOCAB = 97
TEMPLATE = HERE / 'references' / 'fusion-jobs' / 'ER' / 'SP0-split-probe.env'


class Dev(object):
    """A device tensor of the fake: its host value is every chip's, already composed."""

    def __init__(self, data):
        self.data = data
        self.dtype = data.dtype
        self.shape = tuple(data.shape)


class Grid(object):
    x, y = 13, 10


class FakeMesh(object):
    def get_num_devices(self):
        return CHIPS

    def compute_with_storage_grid_size(self):
        return Grid()


class FakeTTNN(object):
    bfloat16 = torch.bfloat16
    FabricConfig = type('FabricConfig', (), {'FABRIC_1D': 'f1d', 'FABRIC_1D_RING': 'f1dr'})

    def __init__(self):
        self.calls = []
        self.mesh = FakeMesh()

    def MeshShape(self, *shape):
        return shape

    def set_fabric_config(self, value):
        self.calls.append(('fabric', value))

    def open_mesh_device(self, shape, **keywords):
        self.calls.append(('open', tuple(shape)))
        return self.mesh

    def close_mesh_device(self, mesh):
        self.calls.append(('close',))

    def synchronize_device(self, mesh):
        pass

    def deallocate(self, tensor):
        pass

    def ConcatMeshToTensor(self, mesh, dim):
        return ('concat', dim)

    def to_torch(self, tensor, mesh_composer=None):
        return tensor.data.clone()


class DeltaNet(object):
    def __init__(self, index):
        self.index = index
        self.rec_state = Dev(torch.zeros(CHIPS, 2, 4, 4, dtype=torch.float32))
        self.conv_carry = Dev(torch.zeros(CHIPS, 3, 8, dtype=torch.bfloat16))
        self.conv_states = [Dev(torch.zeros(CHIPS, 1, 8, dtype=torch.bfloat16)) for _ in range(4)]


class Attention(object):
    pass


class Layer(object):
    def __init__(self, index, full):
        self.is_full_attention = full
        self.attention = Attention() if full else DeltaNet(index)


class ToyModel(object):
    """Just the surface tp4_split_probe touches, with the image model's resume rules (start % 2048 == 0 on the chunked entry)."""

    def __init__(self, mode='exact'):
        self.mode = mode
        self.vocab_size = VOCAB
        self.args = types.SimpleNamespace(vocab_size=VOCAB, n_local_kv_heads=1, head_dim=HEAD_DIM)
        self.layers = [Layer(0, False), Layer(1, False), Layer(2, False), Layer(3, True)]
        self._paged_kv_caches = None
        self.bound = False
        self.cold_calls = 0
        self.calls = []
        self.masked_buckets = []

    # -- the image model's surface ----------------------------------------------------------------------------------------------
    def allocate_kv_caches(self, shape, dtype, batch_size=1):
        blocks, heads, block, width = shape
        assert block == 64 and heads == 1 and width == HEAD_DIM
        self._paged_kv_caches = [[Dev(torch.zeros(blocks, heads * CHIPS, block, width, dtype=torch.bfloat16)),
                                  Dev(torch.zeros(blocks, heads * CHIPS, block, width, dtype=torch.bfloat16))]]
        return self._paged_kv_caches

    def _bind_gdn_prefill_scratch(self):
        assert not self.bound
        self.bound = True
        return 'previous'

    def _unbind_gdn_prefill_scratch(self, previous):
        assert self.bound and previous == 'previous'
        self.bound = False

    def _build_request_rope(self, token_ids, vision_tokens):
        self.calls.append(('rope', int(token_ids.shape[1])))

    def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None, start=0, capture_at=None, on_capture=None):
        assert self.bound, 'prefill outside the bound scratch'
        if start % 2048 or not 0 <= start < actual_len:
            raise AssertionError('prefix reuse: start_pos %d is not a chunk boundary inside the prompt (L=%d)' % (start, actual_len))
        self.calls.append(('chunked', start, actual_len))
        if start == 0:
            self.reset()
        # 'bucket' mode (the programs differ): the tail of a prompt that is not the cold run's own runs in another bucket than the cold run's
        shifted = actual_len // 2048 * 2048 if self.mode == 'bucket' and actual_len != self.final_length else None
        return self.process(token_ids[:, start:actual_len], page_table, start, actual_len, masked_off_chunk=False, shifted_from=shifted)

    def prefill_masked_bucket(self, token_ids, page_table, actual_len, chunk_start=0, bucket=None, flex_sdpa=True, vision_tokens=None,
                              vis_row_offset=0):
        assert self.bound
        assert chunk_start % 64 == 0, 'a fill off a block boundary'
        self.calls.append(('masked', chunk_start, actual_len))
        self.masked_buckets.append((chunk_start, actual_len, bucket))
        if chunk_start == 0:
            self.reset()
        # 'bucket' mode: a masked program in the natural bucket (bucket=None) is not the cold run's program; one forced into 2048 is
        shifted = chunk_start if self.mode == 'bucket' and bucket is None else None
        return self.process(token_ids[:, :actual_len], page_table, chunk_start, chunk_start + actual_len,
                            masked_off_chunk=chunk_start % 2048 != 0 and self.mode == 'sensitive', shifted_from=shifted)

    # -- the toy arithmetic -----------------------------------------------------------------------------------------------------
    def reset(self):
        self.cold_calls += 1
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.rec_state.data.zero_()
                layer.attention.conv_carry.data.zero_()
                for state in layer.attention.conv_states:
                    state.data.zero_()

    def process(self, tokens, page_table, begin, end, masked_off_chunk, shifted_from=None):
        k_cache, v_cache = self._paged_kv_caches[0]
        values = tokens.reshape(-1).to(torch.float32)
        for offset in range(end - begin):
            position = begin + offset
            token = float(values[offset])
            for layer in self.layers:
                if layer.is_full_attention:
                    continue
                dn = layer.attention
                dn.rec_state.data.mul_(0.75).add_(token / VOCAB + layer.attention.index)
            row = torch.full((CHIPS, HEAD_DIM), token / VOCAB + position * 1e-3, dtype=torch.float32)
            if (self.mode == 'sensitive' and masked_off_chunk) or (shifted_from is not None and position >= shifted_from):
                row = row + 0.5
            block = int(page_table[0, position // 64])
            k_cache.data[block, :, position % 64, :] = row.to(torch.bfloat16)
            v_cache.data[block, :, position % 64, :] = (row * 2).to(torch.bfloat16)
        if self.mode in ('sensitive', 'handoff') and end % 2048 and end != self.final_length:
            for layer in self.layers:
                if not layer.is_full_attention:
                    dn = layer.attention
                    dn.rec_state.data.copy_(dn.rec_state.data.to(torch.bfloat16).to(torch.float32))
                    # the store at the boundary leaves a trace the decay cannot wash out (the toy's stand-in for a changed rounding)
                    dn.conv_carry.data.add_(1)
        if self.mode == 'noisy' and begin == 0:
            self.layers[0].attention.rec_state.data.add_(1e-3 * self.cold_calls)
        state = self.layers[0].attention.rec_state.data
        logits = torch.zeros(CHIPS, 1, 1, VOCAB, dtype=torch.float32)
        logits[..., int(state.abs().sum()) % VOCAB] = float(state.sum())
        return Dev(logits)


def factory(mode):
    model = ToyModel(mode)

    def build(ttnn, mesh, options, environ, log):
        model.final_length = None
        info = dict(layer_indices=[0, 1, 2, 3], layer_types=['linear_attention'] * 3 + ['full_attention'], n_local_kv_heads=1,
                    head_dim=HEAD_DIM, vocab_size=VOCAB)
        return model, info

    return model, build


class FinalLength(object):
    """The toy's 'sensitive' rounding skips the prompt's own end: a split that ends at L is the cold run's end."""

    @staticmethod
    def attach(model, length):
        model.final_length = length


def run_probe(mode, lengths=(5000,), extra=(), attribute=False):
    model, build = factory(mode)
    ttnn = FakeTTNN()
    lines = []
    argv = ['--output', 'unused', '--lengths', ','.join(str(length) for length in lengths)]
    for split in extra:
        argv += ['--extra-split', str(split)]
    if attribute:
        argv.append('--attribute')
    options = probe.build_parser().parse_args(argv)
    original_measure = probe.measure_length

    def measure(ttnn_, torch_, model_, mesh, length, width, seed, extras, log, *more):
        FinalLength.attach(model_, length)
        return original_measure(ttnn_, torch_, model_, mesh, length, width, seed, extras, log, *more)

    probe.measure_length = measure
    try:
        report = probe.run(options, ttnn=ttnn, torch=torch, model_factory=build, environ={}, log=lines.append)
    finally:
        probe.measure_length = original_measure
    return report, ttnn, model, lines


class PlanTests(unittest.TestCase):
    def test_the_arms_of_a_length(self):
        arms = dict(probe.plan_arms(9000))
        self.assertEqual(arms, {'A': None, 'A2': None, 'C': 4096, 'B+128': 4224, 'B+1024': 5120, 'B+1920': 6016, 'Bend': 8576})
        for name, split in arms.items():
            if split is not None:
                self.assertIsNone(probe.check_split(9000, split), name)

    def test_the_end_split_never_lands_on_a_chunk_boundary(self):
        arms = dict(probe.plan_arms(4096 + 300))
        self.assertEqual(arms['Bend'] % 2048, 4096 % 2048 + 2048 - 128)
        self.assertNotEqual(arms['Bend'] % 2048, 0)

    def test_a_resume_off_128_is_refused(self):
        self.assertIn('not a multiple of 128', probe.check_split(9000, 4100))
        self.assertIn('not a multiple of 128', probe.check_split(9000, 4160))   # 64-aligned is still wrong SDPA
        self.assertIn('not inside the prompt', probe.check_split(9000, 9000))
        self.assertIn('not inside the prompt', probe.check_split(9000, 0))
        with self.assertRaises(ValueError):
            probe.plan_arms(9000, extra=(4100,))
        with self.assertRaises(ValueError):
            probe.plan_arms(probe.MIN_LENGTH - 1)

    def test_regions(self):
        self.assertEqual(probe.regions(4224, 9000), [('below_floor2048_R', 0, 4096), ('floor2048_R_to_R', 4096, 4224),
                                                     ('R_to_H', 4224, 6144), ('H_to_L', 6144, 9000)])
        self.assertEqual(probe.regions(8576, 9000)[2:], [('R_to_H', 8576, 9000), ('H_to_L', 9000, 9000)])
        self.assertEqual(probe.regions(4096, 9000)[1:3], [('floor2048_R_to_R', 4096, 4096), ('R_to_H', 4096, 4096)])

    def test_the_layers_are_read_from_the_checkpoint_types(self):
        types_ = ['linear_attention'] * 3 + ['full_attention']
        self.assertEqual(probe.pick_layers(types_ * 16), [0, 1, 2, 3])
        self.assertEqual(probe.pick_layers(['linear_attention', 'full_attention'] * 4), [0, 1])
        with self.assertRaises(ValueError):
            probe.pick_layers(['linear_attention'] * 8)
        with self.assertRaises(ValueError):
            probe.pick_layers(['full_attention'] + ['linear_attention'] * 7)


class VerdictTests(unittest.TestCase):
    def test_an_invariant_model_is_identical(self):
        report, ttnn, model, lines = run_probe('exact')
        self.assertEqual(report['verdict'], 'IDENTICAL', report.get('reasons') or report.get('error'))
        self.assertEqual(report['grid'], [13, 10])
        self.assertEqual(report['chips'], 4)
        arms = report['results'][0]['arms']
        self.assertEqual([arm['name'] for arm in arms], ['A', 'A2', 'C', 'B+128', 'B+1024', 'B+1920', 'Bend'])
        self.assertTrue(all(arm['identical'] for arm in arms if arm['name'] != 'A'))
        self.assertEqual(ttnn.calls[-1], ('close',))
        self.assertFalse(model.bound)
        self.assertTrue(any(line.startswith('SPLIT_PROBE verdict=IDENTICAL') for line in lines))
        # every split arm took the head segment at R and, when H < L, resumed the chunked entry at H (a 2048 boundary)
        self.assertIn(('masked', 2176, 4096 - 2176), model.calls)
        self.assertIn(('chunked', 4096, 5000), model.calls)
        self.assertIn(('rope', 5000), model.calls)

    def test_a_split_sensitive_model_differs_and_the_rows_are_placed(self):
        report, ttnn, model, lines = run_probe('sensitive')
        self.assertEqual(report['verdict'], 'DIFFERS', report.get('reasons') or report.get('error'))
        arms = {arm['name']: arm for arm in report['results'][0]['arms']}
        self.assertTrue(arms['A2']['identical'])
        self.assertTrue(arms['C']['identical'])
        for name in ('B+128', 'B+1024', 'B+1920', 'Bend'):
            self.assertFalse(arms[name]['identical'], name)
            self.assertGreater(arms[name]['gdn']['differing'], 0, name)
            k = arms[name]['kv']['layers'][0]['k']
            self.assertFalse(k['equal'])
            self.assertEqual(k['regions']['below_floor2048_R'], 0)
            self.assertEqual(k['regions'].get('floor2048_R_to_R', 0), 0)
            low, high = [(l, h) for label, l, h in probe.regions(arms[name]['split'], 5000) if label == 'R_to_H'][0]
            self.assertEqual(k['regions']['R_to_H'], high - low)
        self.assertTrue(any('SPLIT_PROBE verdict=DIFFERS' in line for line in lines))

    def test_a_non_deterministic_control_fails_the_run(self):
        report, ttnn, model, lines = run_probe('noisy')
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertTrue(any('control A2 differs' in reason for reason in report['reasons']))

    def test_verdict_rules(self):
        same = dict(identical=True)
        good = [dict(length=5000, arms=[dict(name='A', split=None), dict(name='A2', split=None, **same), dict(name='C', split=4096, **same),
                                        dict(name='B+128', split=4224, **same)])]
        self.assertEqual(probe.verdict(good), ('IDENTICAL', []))
        bad_split = json.loads(json.dumps(good))
        bad_split[0]['arms'][3]['identical'] = False
        self.assertEqual(probe.verdict(bad_split)[0], 'DIFFERS')
        bad_control = json.loads(json.dumps(bad_split))
        bad_control[0]['arms'][2]['identical'] = False
        self.assertEqual(probe.verdict(bad_control)[0], 'FAIL')
        self.assertEqual(probe.verdict([])[0], 'FAIL')
        self.assertEqual(probe.verdict([dict(length=5000, error='boom')])[0], 'FAIL')


class MainTests(unittest.TestCase):
    def test_main_writes_the_report_and_exits_by_the_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'split.json')
            for mode, status in (('exact', 0), ('sensitive', 2), ('noisy', 1)):
                model, build = factory(mode)
                original = probe.measure_length

                def measure(ttnn_, torch_, model_, mesh, length, width, seed, extras, log, original=original):
                    FinalLength.attach(model_, length)
                    return original(ttnn_, torch_, model_, mesh, length, width, seed, extras, log)

                probe.measure_length = measure
                try:
                    code = probe.main(['--output', out, '--lengths', '5000'], ttnn=FakeTTNN(), torch=torch, model_factory=build, environ={},
                                      log=lambda text: None)
                finally:
                    probe.measure_length = original
                self.assertEqual(code, status, mode)
                self.assertIn(json.loads(Path(out).read_text())['verdict'], ('IDENTICAL', 'DIFFERS', 'FAIL'))

    def test_an_unaligned_extra_split_is_refused_before_the_mesh_opens(self):
        with tempfile.TemporaryDirectory() as directory:
            out = str(Path(directory) / 'split.json')
            ttnn = FakeTTNN()
            code = probe.main(['--output', out, '--lengths', '5000', '--extra-split', '2100'], ttnn=ttnn, torch=torch,
                              model_factory=factory('exact')[1], environ={}, log=lambda text: None)
            self.assertEqual(code, 1)
            self.assertEqual(ttnn.calls, [])
            report = json.loads(Path(out).read_text())
            self.assertEqual(report['verdict'], 'FAIL')
            self.assertIn('refused', report['error'])

    def test_an_aligned_extra_split_runs(self):
        report, ttnn, model, lines = run_probe('exact', extra=(2304,))
        self.assertIn('Bx2304', [arm['name'] for arm in report['results'][0]['arms']])
        self.assertEqual(report['verdict'], 'IDENTICAL')

    def test_without_a_ring_descriptor_nothing_opens(self):
        options = probe.build_parser().parse_args(['--output', 'x', '--lengths', '5000'])
        lines = []
        report = probe.run(options, environ={'TT_MESH_GRAPH_DESC_PATH': '/nowhere/p300.textproto'}, log=lines.append)
        self.assertEqual(report['verdict'], 'FAIL')
        self.assertIn('refused to open', report['error'])
        self.assertFalse(report['opened'])


class BuildModelTests(unittest.TestCase):
    """build_model against a stand-in models tree: the layer subset from the checkpoint's types, the real state dict, no tensor cache."""

    def fake_tree(self, refuse_none=False):
        built = []

        class Args(object):
            def __init__(self, mesh_device=None, max_batch_size=1, max_seq_len=2048):
                self.mesh_device, self.max_seq_len = mesh_device, max_seq_len
                self.attention_type_list = (['linear_attention'] * 3 + ['full_attention']) * 16
                self.n_layers, self.n_local_kv_heads, self.head_dim, self.vocab_size = 64, 1, 256, 248320
                self.dummy_weights = False

            def load_state_dict(self):
                return {'real': True}

        class Model(object):
            def __init__(self, mesh, args, state_dict, tensor_cache_path=None):
                if refuse_none and tensor_cache_path is None:
                    raise TypeError("unsupported operand type(s) for /: 'NoneType' and 'str'")
                built.append(dict(args=args, state_dict=state_dict, cache=tensor_cache_path))

        modules = {}
        for name in ('models', 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36', 'models.demos.blackhole.qwen36.tt'):
            modules[name] = types.ModuleType(name)
        modules['models.demos.blackhole.qwen36.tt.model'] = types.ModuleType('model')
        modules['models.demos.blackhole.qwen36.tt.model'].Qwen36Model = Model
        modules['models.demos.blackhole.qwen36.tt.model_config'] = types.ModuleType('model_config')
        modules['models.demos.blackhole.qwen36.tt.model_config'].Qwen36ModelArgs = Args
        return modules, built

    def build(self, refuse_none, environ):
        modules, built = self.fake_tree(refuse_none)
        saved = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            options = probe.build_parser().parse_args(['--output', 'x'])
            model, info = probe.build_model(FakeTTNN(), FakeMesh(), options, environ, lambda text: None)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return built, info

    def test_the_subset_and_no_cache(self):
        built, info = self.build(False, {})
        self.assertEqual(info['layer_indices'], [0, 1, 2, 3])
        self.assertEqual(info['layer_types'], ['linear_attention'] * 3 + ['full_attention'])
        self.assertEqual(info['tensor_cache'], 'none')
        args = built[0]['args']
        self.assertEqual((args.layer_indices, args.n_layers, args.dummy_weights), ([0, 1, 2, 3], 4, True))
        self.assertEqual(built[0]['state_dict'], {'real': True})
        self.assertIsNone(built[0]['cache'])
        self.assertEqual(args.max_seq_len, 65536)

    def test_a_module_that_needs_a_cache_path_gets_the_compile_cache_tmpfs(self):
        with tempfile.TemporaryDirectory() as directory:
            built, info = self.build(True, {'TT_METAL_CACHE': directory})
            self.assertEqual(str(built[0]['cache']), str(Path(directory) / 'split-probe-weights'))
            self.assertEqual(info['tensor_cache'], str(Path(directory) / 'split-probe-weights'))


class AttributionTests(unittest.TestCase):
    """--attribute: the F arms (both segments forced into the 2048 bucket) say WHY a B arm differs from the cold run (docs/tp4-exact-end.md 2)."""

    def test_the_default_plan_has_no_forced_arm(self):
        self.assertEqual([name for name, _ in probe.plan_arms(9000)], ['A', 'A2', 'C', 'B+128', 'B+1024', 'B+1920', 'Bend'])
        self.assertEqual(probe.plan_arms(9000), probe.plan_arms(9000, (), False))

    def test_the_attribution_plan_repeats_the_splits_forced(self):
        arms = dict(probe.plan_arms(9000, attribute=True))
        self.assertEqual({name: split for name, split in arms.items() if probe.is_forced(name)},
                         {'F+128': 4224, 'F+1024': 5120, 'F+1920': 6016, 'Fend': 8576})
        for forced, natural in (('F+128', 'B+128'), ('F+1024', 'B+1024'), ('F+1920', 'B+1920'), ('Fend', 'Bend')):
            self.assertEqual(arms[forced], arms[natural])
        self.assertEqual(probe.FORCED_BUCKET, 2048)
        self.assertFalse(any(probe.is_forced(name) for name in ('A', 'A2', 'C', 'B+128', 'Bend')))

    def test_an_invariant_model_is_identical_with_the_attribution_arms_and_the_verdict_ignores_them(self):
        report, ttnn, model, lines = run_probe('exact', attribute=True)
        self.assertEqual(report['verdict'], 'IDENTICAL', report.get('reasons') or report.get('error'))
        self.assertEqual({row['cause'] for row in report['attribution']}, {'none'})
        self.assertEqual(len(report['attribution']), 4)
        # the forced arms ran BOTH segments of the split in the 2048 bucket (the cold run's own program masked to the valid rows): for R=2176 of the 5000-token prompt,
        # the rows [2048, 2176) at chunk_start 2048 and the head [2176, 4096) at chunk_start 2176
        self.assertIn((2048, 128, 2048), model.masked_buckets)
        self.assertIn((2176, 1920, 2048), model.masked_buckets)
        # ... and the natural arms let the model pick the bucket (None here: the toy has no bucket table)
        self.assertIn((2176, 1920, None), model.masked_buckets)
        self.assertTrue(any(line.startswith('SPLIT_PROBE attribution L=5000 R=2176 B+128=IDENTICAL F+128=IDENTICAL cause=none') for line in lines), lines)

    def test_a_model_whose_programs_differ_is_attributed_to_the_bucket(self):
        report, ttnn, model, lines = run_probe('bucket', attribute=True)
        self.assertEqual(report['verdict'], 'DIFFERS')
        arms = {arm['name']: arm for arm in report['results'][0]['arms']}
        self.assertTrue(arms['C']['identical'])
        for name in ('B+128', 'B+1024', 'B+1920', 'Bend'):
            self.assertFalse(arms[name]['identical'], name)
        for name in ('F+128', 'F+1024', 'F+1920', 'Fend'):
            self.assertTrue(arms[name]['identical'], name)
        self.assertEqual({row['cause'] for row in report['attribution']}, {'bucket'})
        self.assertTrue(any('B+1024=DIFFERS F+1024=IDENTICAL cause=bucket' in line for line in lines), lines)

    def test_a_model_whose_store_at_the_split_differs_is_attributed_to_the_handoff(self):
        report, ttnn, model, lines = run_probe('handoff', attribute=True)
        self.assertEqual(report['verdict'], 'DIFFERS')
        arms = {arm['name']: arm for arm in report['results'][0]['arms']}
        for name in ('B+128', 'B+1024', 'B+1920', 'F+128', 'F+1024', 'F+1920'):
            self.assertFalse(arms[name]['identical'], name)
            self.assertGreater(arms[name]['gdn']['differing'], 0, name)
        self.assertEqual({row['cause'] for row in report['attribution']}, {'handoff'})
        self.assertTrue(any('B+128=DIFFERS F+128=DIFFERS cause=handoff' in line for line in lines), lines)

    def test_the_verdict_counts_the_natural_splits_only(self):
        report, _, _, lines = run_probe('bucket', attribute=True)
        line = [text for text in lines if text.startswith('SPLIT_PROBE verdict=')][0]
        self.assertIn('splits=4 differing=4', line)
        report, _, _, lines = run_probe('bucket')
        self.assertIn('splits=4 differing=4', [text for text in lines if text.startswith('SPLIT_PROBE verdict=')][0])
        self.assertNotIn('attribution', report)

    def test_attribution_reads_the_arms_it_is_given(self):
        def item(**flags):
            return dict(length=5000, arms=[dict(name='B+128', split=2176, identical=flags['b']), dict(name='F+128', split=2176, identical=flags['f'])])

        self.assertEqual(probe.attribution([item(b=False, f=True)])[0][-1], 'bucket')
        self.assertEqual(probe.attribution([item(b=False, f=False)])[0][-1], 'handoff')
        self.assertEqual(probe.attribution([item(b=True, f=False)])[0][-1], 'handoff')
        self.assertEqual(probe.attribution([item(b=True, f=True)])[0][-1], 'none')
        self.assertIsNone(probe.attribution([dict(length=5000, arms=[dict(name='B+128', split=2176), dict(name='F+128', split=2176, error='x')])])[0][-1])
        self.assertEqual(probe.attribution([dict(length=5000, arms=[dict(name='B+128', split=2176, identical=True)])]), [])


class WiringTests(unittest.TestCase):
    def test_the_job_and_the_workflow_know_the_probe(self):
        self.assertIn('split', job.FABRIC_PROBES)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('split) script=tp4_split_probe.py; report=split-probe.json', workflow)
        self.assertIn('SPLIT_PROBE', workflow)
        # the weights are mounted read-only, and only for the split probe
        self.assertIn('if [ "${PROBE:-fabric}" = split ] || [ "${PROBE:-fabric}" = split-attribute ]; then', workflow)
        self.assertIn(':/models:ro" -e HF_HOME=/models -e HF_HUB_CACHE=/models -e HF_HUB_OFFLINE=1)', workflow)
        self.assertLess(probe.WATCHDOG_S, 45 * 60)

    def test_the_attribution_probe_is_wired_like_the_split_probe(self):
        self.assertIn('split-attribute', job.FABRIC_PROBES)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('split-attribute) script=tp4_split_probe.py; report=split-probe.json; probe_args="--attribute --profile ', workflow)
        line = [text for text in workflow.splitlines() if 'split-attribute) script=' in text][0]
        profile = line.split('--profile ')[1].split('"')[0]
        profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))['profiles']
        self.assertIn(profile, profiles, 'the probe runs under the ship-4 candidate\'s own environment')
        import serving_c2_contract

        environ = {}
        serving_c2_contract.apply_environment(serving_c2_contract.load_profile(str(HERE / 'qwen_c2_profiles.json'), profile), environ)
        self.assertEqual(environ.get('QWEN_FAST_GRID13'), 'prefill,drafter')
        self.assertEqual(environ.get('QWEN_FAST_TAP_WINDOW_ALIGNED'), '1')

    def test_the_template_is_a_valid_job(self):
        values = job.parse_env(TEMPLATE.read_text(encoding='utf-8'))
        self.assertEqual(values['C2_ACTIONS'], 'reset fabric')
        self.assertEqual(values['C2_CARDS'], 'quad')
        self.assertEqual(values['C2_FABRIC_PROBE'], 'split')
        self.assertLessEqual(int(values['C2_BOX_MINUTES']), 45)
        self.assertGreater(int(values['C2_BOX_MINUTES']) * 60, probe.WATCHDOG_S)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = job.main([str(TEMPLATE), str(HERE / 'qwen_c2_profiles.json')])
        self.assertEqual(status, 0, err.getvalue())
        self.assertIn('fabric_probe=split', out.getvalue())


if __name__ == '__main__':
    unittest.main()
