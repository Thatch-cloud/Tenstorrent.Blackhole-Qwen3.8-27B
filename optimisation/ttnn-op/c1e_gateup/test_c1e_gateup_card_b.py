"""CPU tests for the C1e card-B harness (c1e_gateup_card_b.py) and its runner (run_card_b.sh): the raw
bf16 comparison and its classes, the double rounding that is why C1c cannot be exact, the inputs, the
weights built the served way, the verdict, every section end to end against a numeric fake ttnn, and
the runner (the qualification card, the pinned image, the op mounted file by file from the one table,
lever_n_m3native_patch.C1E_FILES). The device half runs on the rig only (run_card_b.sh)."""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
CI = ROOT / 'scripts' / 'ci'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CI))

import torch  # noqa: E402

import c1e_gateup_card_b as card  # noqa: E402
import lever_n_m3native_patch as patcher  # noqa: E402
import mlp_c1e_pack as pack  # noqa: E402
from test_mlp_c1e_pack import Grid, Ops, Record, prepare_model  # noqa: E402  (the fakes, not the TestCases)

SCRIPT = HERE / 'run_card_b.sh'


def script_text():
    return SCRIPT.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))


def bf16(bits):
    return torch.tensor(bits, dtype=torch.int16).view(torch.bfloat16)


class CompareTests(unittest.TestCase):
    def test_identical_is_exact(self):
        value = torch.randn(1056, 64).to(torch.bfloat16)
        result = card.bits_compare(torch, value, value.clone())
        self.assertTrue(result['exact'])
        self.assertEqual(result['per_slice'], [0, 0])            # 1024 + the partial last slice of 32

    def test_classes(self):
        # +0 vs -0, two NaN payloads, NaN vs a number, 1.0 vs the next bf16, and the two smallest
        # denormals of opposite sign (2 ulps apart through zero).
        left = bf16([[0x0000, 0x7FC0, 0x7FC0, 0x3F80, 0x0001, 0x4000]])
        right = bf16([[-0x8000, 0x7FC1, 0x3F80, 0x3F81, -0x7FFF, 0x4000]])
        result = card.bits_compare(torch, left, right)
        self.assertEqual((result['mismatches'], result['zero_sign'], result['nan_payload'], result['nan_one_side'],
                          result['numeric'], result['max_ulp']), (5, 1, 1, 1, 2, 2))
        self.assertEqual(result['first'], [0, 0])
        self.assertFalse(result['exact'])

    def test_per_slice_localises_differences(self):
        left = torch.zeros(1536, 8, dtype=torch.bfloat16)
        right = left.clone()
        right[1024, 3] = 1.0
        right[1535, 0] = 2.0
        result = card.bits_compare(torch, left, right)
        self.assertEqual(result['per_slice'], [0, 2])
        self.assertEqual(result['first'], [1024, 3])

    def test_shapes_and_dtypes_must_match(self):
        with self.assertRaises(ValueError):
            card.bits_compare(torch, torch.zeros(2, 2, dtype=torch.bfloat16), torch.zeros(2, 3, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            card.bits_compare(torch, torch.zeros(2, 2), torch.zeros(2, 2))

    def test_double_rounding_is_why_c1c_cannot_be_exact(self):
        generator = torch.Generator().manual_seed(7)
        gate = torch.randn(200000, generator=generator) * 1.43
        up = torch.randn(200000, generator=generator) * 1.43
        silu = torch.nn.functional.silu(gate)
        fused = (silu * up).to(torch.bfloat16)                                          # one rounding
        separate = (silu.to(torch.bfloat16).float() * up.to(torch.bfloat16).float()).to(torch.bfloat16)
        share = card.bits_compare(torch, fused.reshape(1, -1), separate.reshape(1, -1))['mismatches'] / 200000
        self.assertGreater(share, 0.25)


class HarnessTests(unittest.TestCase):
    def test_x_specials(self):
        x = card.make_x(torch, 128, 0).reshape(128, card.DIM)
        bits = x.view(torch.int16)
        self.assertTrue(bool((bits[0] == 0).all()))
        self.assertTrue(bool((bits[1] == -0x8000).all()))
        self.assertEqual(int(torch.isnan(x).sum()), 1)
        self.assertTrue(bool(torch.isnan(x[2, 17])))
        plain = card.make_x(torch, 128, 0, specials=False)
        self.assertEqual(int(torch.isnan(plain).sum()), 0)
        self.assertTrue(torch.equal(card.make_x(torch, 64, 3).view(torch.int16), card.make_x(torch, 64, 3).view(torch.int16)))

    def test_make_weights_with_the_documented_packing(self):
        # Small stand-ins for DIM / HIDDEN_TP: the same code path at 1/40 x 1/34 the size.
        with mock.patch.object(card, 'DIM', 128), mock.patch.object(card, 'HIDDEN_TP', 256):
            shards, identity = card.make_weights(torch, prepare_model, 0, [0, 1])
            self.assertEqual(identity, {0: True, 1: True})
            w1, w3, packed = shards[1]
            self.assertEqual((tuple(w1.shape), tuple(w3.shape), tuple(packed.shape)), ((128, 256), (128, 256), (128, 512)))
            self.assertEqual(w1.dtype, torch.bfloat16)
            broken = lambda gate_up, ndev, gate_is_first: gate_up          # noqa: E731 - a packing that is not the interleave
            self.assertEqual(card.make_weights(torch, broken, 0, [0, 1])[1], {0: False, 1: False})

    def test_the_interleave_is_the_kernels_page_map(self):
        """TILE-layout page (kt, nt) of a [K, N] tensor holds rows kt*32.., columns nt*32..: so
        mlp_c1e_pack's page map is exactly this harness's element interleave."""
        k, n = 64, 96
        w1 = torch.arange(k * n, dtype=torch.float32).reshape(k, n)
        w3 = -w1 - 1
        packed = card.interleave(torch, w1, w3)
        columns = n // 32
        for kt in range(k // 32):
            for nt in range(2 * columns):
                g, pair = pack.separate_page(kt * 2 * columns + nt, columns)
                source = (w1, w3)[g]
                self.assertTrue(torch.equal(packed[kt * 32:(kt + 1) * 32, nt * 32:(nt + 1) * 32],
                                            source[kt * 32:(kt + 1) * 32, (pair % columns) * 32:(pair % columns + 1) * 32]))

    def test_projections(self):
        self.assertEqual(card.projections(dict(pack_ms=0.3, fused_ms=4.0, c1c_ms=5.1)),
                         dict(c1e_penalty_ms_per_chunk=19.2, c1c_penalty_ms_per_chunk_gate_up_only=70.4))
        self.assertEqual(card.projections({}), {})

    def test_arguments(self):
        args = card.parse_args(['--out', 'x.json', '--quick'])
        self.assertEqual((args.rows, args.seeds, args.chips), ([2048, 1056], [0], [0]))
        self.assertEqual(args.op_dir, Path(card.OP_DIR))
        args = card.parse_args(['--out', 'x.json'])
        self.assertEqual(args.rows, list(card.ROWS))
        self.assertEqual(args.sections, list(card.SECTIONS))
        for bad in (['--rows', '32'], ['--rows', '100'], ['--chips', '2'], ['--sections', 'bogus']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                card.parse_args(['--out', 'x.json'] + bad)

    def test_rows_cover_the_buckets_and_the_c1c_partial_slices(self):
        self.assertTrue({128, 256, 512, 1024, 2048} <= set(card.ROWS))      # model._PREFILL_MASK_BUCKETS
        self.assertTrue(any(rows % card.SLICE for rows in card.ROWS if rows > card.SLICE))
        self.assertTrue(any(rows % 256 for rows in card.ROWS))               # a partial fused M block (M_block 8)
        self.assertEqual(card.SLICE, patcher.C1_SLICE_ROWS)                  # C1c's slice, as grafted

    def test_the_served_constants(self):
        """tp_common.all_gather_swiglu_prefill at TP2: M8 / K agmm_k_block_size(2560) = 8 / N16, subblock 1x4,
        grid (7, 9) widened to 8 for two links."""
        self.assertEqual(card.SERVED_BLOCKS, dict(M_block_size=8, K_block_size=8, N_block_size=16,
                                                  subblock_h=1, subblock_w=4))
        self.assertEqual(card.SERVED_GRID, (8, 9))
        self.assertEqual((card.DIM, card.HIDDEN_TP, card.TP), (5120, 8704, 2))

    def test_verdict_needs_cases_and_no_failures(self):
        self.assertFalse(card.verdict(dict(failures=[], cases_run=0)))
        self.assertFalse(card.verdict(dict(failures=['x'], cases_run=3)))
        self.assertTrue(card.verdict(dict(failures=[], cases_run=3)))


# ---------------------------------------------------------------------------------------------
# The harness end to end against a numeric fake ttnn (small shapes): every section runs, the
# verdict logic sees C1e exact, C1c different, and the negative controls bite.
# ---------------------------------------------------------------------------------------------

class NumericTensor:
    """A one-chip device tensor holding its logical value (bfloat4_b is modelled as bf16 values)."""

    def __init__(self, fake, value, dtype):
        self.fake, self.value, self.dtype, self.layout = fake, value, dtype, 'tile'
        self.address = fake.allocate(self)
        self.padded_shape = tuple(value.shape)
        self.tile = Record(tile_shape=(32, 32))

    @property
    def shape(self):
        return tuple(self.value.shape)

    def memory_config(self):
        return self.fake.DRAM_MEMORY_CONFIG

    def buffer_address(self):
        return self.address


class NumericDevice:
    shape = (1, 1)

    def __init__(self, fake):
        self.fake = fake

    def compute_with_storage_grid_size(self):
        return Grid()

    def num_program_cache_entries(self):
        return len(self.fake.programs)


class NumericTtnn(Ops):
    """Ops plus the numerics c1e_gateup_card_b drives: from/to_torch, matmuls, SwiGLU, slices."""
    float32 = 'f32'

    class MathFidelity:
        LoFi = 'LoFi'

    class UnaryOpType:
        SILU = 'SILU'

    def __init__(self):
        super().__init__()
        self.tensors, self.programs, self.packs = {}, set(), 0
        self.experimental = Record(minimal_matmul=self.minimal_matmul)

    def allocate(self, tensor):
        self.next += 0x1000
        self.tensors[self.next] = tensor
        return self.next

    # device and tensors
    def open_device(self, device_id):
        return NumericDevice(self)

    def close_device(self, device):
        pass

    def synchronize_device(self, device):
        pass

    def ShardTensorToMesh(self, mesh, dim):
        return ('shard', dim, 1)

    def from_torch(self, value, dtype, layout, device, memory_config, mesh_mapper=None):
        if dtype == self.bfloat16:
            return NumericTensor(self, value.clone(), dtype)
        return NumericTensor(self, value.reshape(-1, value.shape[-1]).clone(), dtype)

    def empty(self, shape, dtype, layout, device, memory_config):
        torch_dtype = torch.int64 if dtype == self.uint32 else torch.bfloat16
        return NumericTensor(self, torch.zeros(shape, dtype=torch_dtype), dtype)

    def to_torch(self, tensor):
        return tensor.value

    def deallocate(self, tensor):
        pass

    def get_device_tensors(self, tensor):
        return [tensor]

    def TensorAccessorArgs(self, tensor):
        class Args:
            def get_compile_time_args(self):
                return [3, 0]
        return Args()

    def WormholeComputeKernelConfig(self, **kwargs):
        return Record(**kwargs)

    def MinimalMatmulConfig(self, **kwargs):
        return Record(**kwargs)

    # compute
    @staticmethod
    def _silu(value):
        return torch.nn.functional.silu(value)

    def minimal_matmul(self, x, weight, config, compute_kernel_config, dtype, memory_config, fuse_swiglu=False):
        product = x.value.float().reshape(-1, x.value.shape[-1]) @ weight.value.float()
        if fuse_swiglu:
            rows, width = product.shape
            pairs = product.reshape(rows, width // 64, 2, 32)
            product = (self._silu(pairs[:, :, 0]) * pairs[:, :, 1]).reshape(rows, width // 2)
        product = product.to(torch.float32 if dtype == self.float32 else torch.bfloat16)
        return NumericTensor(self, product.reshape(1, 1, *product.shape), dtype)

    def linear(self, x, weight, compute_kernel_config, program_config, memory_config):
        product = x.value.float() @ weight.value.float()
        if program_config.fused_activation == self.UnaryOpType.SILU:
            product = self._silu(product)
        return NumericTensor(self, product.to(torch.bfloat16), self.bfloat16)

    def mul(self, left, right, memory_config, dtype=None):
        out = (left.value.float() * right.value.float()).to(torch.bfloat16)
        return NumericTensor(self, out, self.bfloat16)

    def silu(self, value, memory_config):
        return NumericTensor(self, self._silu(value.value.float()), self.float32)

    def slice(self, x, begins, ends):
        index = tuple(slice(b, e) for b, e in zip(begins, ends))
        return NumericTensor(self, x.value[index].clone(), x.dtype)

    def concat(self, parts, dim, memory_config):
        return NumericTensor(self, torch.cat([part.value for part in parts], dim=dim), parts[0].dtype)

    # generic_op: the two kernels, by source, reading their tensors through the runtime args
    def generic_op(self, tensors, program):
        super().generic_op(tensors, program)
        (kernels,) = [value.snapshot for value in program.values()]
        self.programs.add(tuple((k['source'], tuple(k['compile']), tuple(k['defines'])) for k in kernels))
        source = kernels[0]['source']
        if source == pack.KERNEL:
            gate, up, packed = (self.tensors[address] for address in kernels[0]['common'])
            packed.value = card.interleave(torch, gate.value, up.value).clone()
            self.packs += 1
        elif source == pack.CHECK_KERNEL:
            for (x, y), args in sorted(kernels[0]['runtime'].items()):
                packed, separate, output = (self.tensors[address] for address in args[:3])
                worker, workers, columns, pages, offset = args[3:]
                mismatches, compared = 0, 0
                for page in range(worker, pages, workers):
                    kt, c = divmod(page, columns)
                    left = packed.value[kt * 32:(kt + 1) * 32, (2 * c + offset) * 32:(2 * c + offset + 1) * 32]
                    right = separate.value[kt * 32:(kt + 1) * 32, c * 32:(c + 1) * 32]
                    mismatches += int((left.contiguous().view(torch.int16) != right.contiguous().view(torch.int16)).sum())
                    compared += 1
                tile = output.value.reshape(workers, 32, 32)
                tile[worker, 0, :6] = torch.tensor([mismatches, compared, 0, 0, pack.CHECK_SENTINEL,
                                                    mismatches ^ 0xFFFFFFFF])
        else:
            raise AssertionError('unknown kernel %s' % source)
        return tensors[-1]


class HarnessFlowTests(unittest.TestCase):
    def run_harness(self, fake, extra=()):
        tpc = types.SimpleNamespace(
            prefill_tuning=lambda tp: dict(tp=tp),
            create_prefill_mlp_matmul_program_config=lambda m, k, n, max_cols=None, tuning=None, fused_activation=None:
                Record(m=m, k=k, n=n, fused_activation=fused_activation))
        modules = {'ttnn': fake}
        for name in ('models', 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36',
                     'models.demos.blackhole.qwen36.tt', 'models.tt_dit', 'models.tt_dit.utils',
                     'models.tt_dit.utils.tensor'):
            modules[name] = types.ModuleType(name)
        modules['models.demos.blackhole.qwen36.tt'].tp_common = tpc
        modules['models.tt_dit.utils.tensor'].prepare_for_fused_swiglu = prepare_model
        pack.clear_cache()
        pack._SCRATCH.clear()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, modules), \
                mock.patch.object(card, 'DIM', 64), mock.patch.object(card, 'HIDDEN_TP', 64), \
                mock.patch.object(card, 'SLICE', 64), contextlib.redirect_stdout(io.StringIO()):
            out = Path(directory, 'report.json')
            status = card.main(['--out', str(out), '--op-dir', str(CI), '--rows', '64,96', '--seeds', '0',
                                '--iters', '2', '--warmup', '1', '--explore'] + list(extra))
            return status, json.loads(out.read_text())

    def test_every_section_runs_and_passes(self):
        fake = NumericTtnn()
        status, report = self.run_harness(fake)
        self.assertEqual((status, report['passed'], report['failures']), (0, True, []))
        self.assertEqual(report['cases_run'], 4)                                  # 2 chips x 2 row counts
        labels = [case['label'] for case in report['cases']]
        self.assertIn('chip1 rows96 seed0 C1e vs served', labels)
        self.assertIn('chip0 rows64 seed0 E3 vs served (explore)', labels)
        self.assertEqual(report['e3_errors'], [])
        self.assertTrue(all(case['exact'] for case in report['cases'] if case['expect_exact']))
        self.assertEqual(report['c1c_mismatch_cases'], 4)                         # the control sees the rounding
        self.assertEqual(report['negative']['program_cache_new_entries'], 0)
        self.assertGreater(report['negative']['swapped_mismatches'], 0)
        self.assertEqual(report['negative']['swapped_bytes_vs_w1_exact'], [False])
        self.assertEqual({entry['tag'] for entry in report['bytes']}, {'A'})
        self.assertTrue(all(entry['spec_equal'] for entry in report['bytes']))
        self.assertEqual(report['scratch_allocation'], 'served topology')
        self.assertIn('c1e_penalty_ms_per_chunk', report['timing'])
        self.assertEqual(report['host_identity'], dict(a={'0': True, '1': True}, b={'0': True}))
        self.assertEqual(sorted(report['op_files']), sorted(pack.RUNTIME_FILES))

    def test_a_copy_that_drops_a_page_fails_the_run(self):
        fake = NumericTtnn()
        original = fake.generic_op

        def lossy(tensors, program):
            result = original(tensors, program)
            if [value.snapshot for value in program.values()][0][0]['source'] == pack.KERNEL:
                tensors[-1].value[:32, 32:64] = 0                                 # packed page 1 (up tile 0) lost
            return result
        fake.generic_op = lossy
        status, report = self.run_harness(fake)
        self.assertEqual((status, report['passed']), (1, False))
        self.assertTrue(any('scratch pages differ from w3' in failure for failure in report['failures']))
        self.assertTrue(any('C1e vs served' in failure for failure in report['failures']))

    def test_a_harness_blind_to_c1c_fails_the_run(self):
        fake = NumericTtnn()

        def linear(x, weight, compute_kernel_config, program_config, memory_config):
            return NumericTensor(fake, x.value.float() @ weight.value.float(), fake.float32)

        def mul(left, right, memory_config, dtype=None):
            return NumericTensor(fake, (torch.nn.functional.silu(left.value) * right.value).to(torch.bfloat16),
                                 fake.bfloat16)
        fake.linear, fake.mul = linear, mul
        status, report = self.run_harness(fake, ['--sections', 'equality'])
        self.assertEqual(report['c1c_mismatch_cases'], 0)
        self.assertEqual(status, 1)
        self.assertTrue(any('cannot see the divergence' in failure for failure in report['failures']))

    def test_a_served_packing_that_is_not_the_interleave_is_caught_on_the_host_and_in_the_bytes(self):
        """If the image packed gate/up any other way (here: up first), C1e's scratch could not be the served
        weight: the host section says so before any device work, and the byte check says so on the device."""
        original = prepare_model

        def swapped(gate_up, ndev, gate_is_first=True):
            half = gate_up.shape[1] // 2
            return original(torch.cat([gate_up[:, half:], gate_up[:, :half]], dim=-1), ndev)
        with mock.patch.object(sys.modules[__name__], 'prepare_model', swapped):
            status, host = self.run_harness(NumericTtnn(), ['--sections', 'host'])
            _, device = self.run_harness(NumericTtnn(), ['--sections', 'bytes'])
        self.assertEqual(status, 1)
        self.assertEqual(host['host_identity']['a'], {'0': False, '1': False})
        self.assertTrue(any('not the per-chip tile-pair interleave' in failure for failure in host['failures']))
        self.assertFalse(any('not the per-chip tile-pair interleave' in failure for failure in device['failures']))
        self.assertTrue(any('served pages differ from w1' in failure for failure in device['failures']))
        self.assertFalse(any('scratch pages differ' in failure for failure in device['failures']))


class RunScriptTests(unittest.TestCase):
    def test_the_qualification_card_the_pinned_image_and_the_refusal(self):
        text = script_text()
        self.assertNotIn('CARD_M=', text)
        self.assertIn('QUAL_CARD_B=blackhole-F36F768B9A5CAFA0', text)          # the embedded qual_card.sh block
        self.assertIn('R=${RESULTS:-$HOME/kwork64/c1e/$QUAL_TAG}', text)
        # Image A5: the image of the v177 / v178 bisect (qwen-lever-n-m3native-gate.yml's v174-v181 pin).
        self.assertIn('IMAGE=${IMAGE:-sha256:126b30dfa72b0e008884f3a8a1cfcb5b7f79eadcdeaeda1cd0f91350dde6ee73}', text)
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml').read_text(encoding='utf-8')
        self.assertIn('*-v177|*-v178|', workflow)
        self.assertIn('image=sha256:126b30dfa72b0e008884f3a8a1cfcb5b7f79eadcdeaeda1cd0f91350dde6ee73', workflow)
        self.assertIn('--device "$node"', text)
        self.assertEqual(text.count('--device '), 1)
        self.assertIn('\nqual_card_resolve\nnode=$QUAL_NODE\nqual_refuse_holders\n', text)
        self.assertLess(text.index('\nqual_refuse_holders\n'), text.index('docker run --rm'))
        self.assertIn('qual_reset_hint >&2', text[text.index('docker run --rm'):])
        self.assertNotIn('tt-smi -r', [line.strip() for line in text.splitlines()
                                       if not line.lstrip().startswith(('#', 'echo', '"'))])
        self.assertNotIn(chr(13), SCRIPT.read_text(encoding='utf-8'))

    def test_the_op_is_mounted_file_by_file_from_the_table(self):
        text = script_text()
        self.assertIn('p.C1E_FILES', text)
        self.assertIn('OM+=(--mount "type=bind,src=$REPO/scripts/ci/$file,dst=%s/$file,readonly")' % card.OP_DIR, text)
        self.assertIn('--op-dir %s' % card.OP_DIR, text)
        self.assertIn('"${OM[@]}"', text)
        self.assertNotRegex(text, r'dst=/bench/c1e[,"]')
        self.assertNotRegex(text, r'dst=/experiment-scripts')
        self.assertIn('--mount "type=bind,src=$test_file,dst=/bench/c1e_gateup_card_b.py,readonly"', text)
        self.assertIn('-e TT_METAL_CACHE=/kcache', text)
        # The table is the op module and exactly the kernels it loads, beside mlp.py in the model.
        self.assertEqual(sorted(patcher.C1E_FILES.values()), sorted(pack.RUNTIME_FILES))

    def test_the_watcher_pass_and_the_explore_arm(self):
        text = script_text()
        self.assertIn('-e TT_METAL_WATCHER=5', text)
        block = text[text.index('if [ "${WATCHER:-}" = "1" ]; then'):]
        block = block[:block.index(chr(10) + 'fi' + chr(10))]
        self.assertIn('timeout_s=900', block)
        self.assertIn('--quick --no-timing', block)
        self.assertIn('args+=(--explore)', text)

    def test_bash_parses_it_and_the_table_snippet_yields_the_three_files(self):
        bash = shutil.which('bash')
        if bash is None or shutil.which('python3') is None:
            self.skipTest('no bash / python3')
        result = subprocess.run([bash, '-n', str(SCRIPT)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        line = [l for l in script_text().splitlines() if l.startswith('mapfile -t op_files')][0]
        with tempfile.TemporaryDirectory() as directory:
            ci = Path(directory, 'scripts', 'ci')
            ci.mkdir(parents=True)
            for name in ('lever_n_m3native_patch.py', 'gdn_prefill_conv_exact.py'):
                shutil.copy(CI / name, ci / name)
            script = ('set -euo pipefail' + chr(10) + 'REPO=' + Path(directory).as_posix() + chr(10) + line + chr(10)
                      + 'printf "RESULT|%s" "${op_files[*]}"' + chr(10))
            result = subprocess.run([bash, '-c', script], capture_output=True, text=True, timeout=120, cwd=directory,
                                    env=dict(os.environ))
        if result.returncode != 0 and ('No module named' in result.stderr or 'No such file' in result.stderr):
            self.skipTest('python3 here cannot see the temp tree: %s' % result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split('RESULT|')[-1].split(),
                         ['mlp_c1e_pack.cpp', 'mlp_c1e_pack.py', 'packed_weight_check.cpp'])

    def test_the_cpu_suite_runs_this_directory(self):
        cpu = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertIn("python -B -m unittest discover -s optimisation/ttnn-op/c1e_gateup -p 'test_*.py'", cpu)
        self.assertTrue(re.search(r'python -B -m unittest [^\n]*\btest_mlp_c1e_pack\b', cpu))


if __name__ == '__main__':
    unittest.main()
