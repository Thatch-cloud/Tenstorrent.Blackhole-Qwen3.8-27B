"""CPU checks for the [QWEN-SDPA-PF] G6 K/V chain (sdpa-prefill-share-spec.md 5.3); no device, no ttnn.

  1 patch integrity   apply_factory_pf refuses anything but fd8c0676 and, given it, reproduces
                      PF_FACTORY, inverts, keeps the qwen_draft_fp32_intermediates block and adds
                      nothing at namespace scope (fd8c0676 is the committed fixture
                      fixtures/sdpa_program_factory.fd8c0676.cpp, so CI runs these too; the probe
                      tree, when present, must be byte-identical to it); the committed chain reader hashes to READER_OUTPUT,
                      regenerates from the served reader (rebuilt from the committed K0 probe readers,
                      so CI needs no copy of the tree) and reverts to f97f5490; every constant crossing
                      a file boundary agrees (tag, flag bits, CT index, semaphore ids, 14-word RT block)
  2 topology mirror   pf_protocol_model.g6_topology: 16 x 6 at 2048 rows ({48g + 8s + m}), 8 x 6 at
                      1024, 4 x 6 at 512, idle cores out, identical unit lists per group, NoC order
                      deterministic and never costlier than raster
  3 round model       C = 0..1100: equal compact round lists in every group, reader k count = compute's
  4 model checker     pf_protocol_model.simulate: (a)-(c) on 'ok', (d) the planted hang, (e) members
                      with different C deadlock; plus the checker's teeth (three protocol mutations are
                      caught) and the equal-length divergence case (wrong_m) the spec's 4.2 misses.
                      QWEN_PF_MODEL_SEEDS sets the 'ok' seeds (default 40 here; README: 2,000)
  5 transcription     R7's K/V/Q calls are the served calls token for token but kv_bt / k_head_rd
  6 flags, envelope   decode_word's refusal table; every spec predicate in the F4 TT_FATAL
  7 Python opt-in     pf_optin (= lever_n_m3native_patch section I, the wired graft): env off = the
                      served config object; on = 0x5EFA0003 (Q2's default) only on the flexible bf8
                      qualified-S path; stock binary and bad flags refused first; the patch inverts,
                      alone and composed with the decode graft, on the committed image tp.py
  8 bench             ../sdpa_prefill_bench/test_prefill_chain_bench.py (arms, words, run_m1.sh)
  9 build script      build_k64g.sh's shas equal the Python's and build_k64f.sh's, its string checks are
                      a superset of build_k64f.sh's; run_card_m_pf.sh dry runs
  + the card-M test's host helpers and a dry run of all three roles against a fake ttnn, and the g++
    stub syntax check (stubcheck/stub_compile.py) when a compiler and the probe tree are present.

    py -3.11 -B -m pytest -q test_sdpa_prefill_chain_sources.py      (from this directory)
Env: QWEN_SDPA_PREFILL_SRC (probe-v25 src/device), QWEN_PF_MODEL_SEEDS, QWEN_PF_GXX.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / 'sdpa_prefill_bench'
DECODE = HERE.parent / 'sdpa_decode_qwen'
for path in (str(HERE), str(BENCH), str(DECODE), str(HERE / 'stubcheck')):
    if path not in sys.path:
        sys.path.insert(0, path)

import apply_factory_pf as factory  # noqa: E402
import make_k0_readers as k0  # noqa: E402
import make_pf_reader as gen  # noqa: E402
import pf_optin as optin  # noqa: E402
import pf_protocol_model as model  # noqa: E402
import stub_compile  # noqa: E402
import test_sdpa_prefill_chain_card_m as card  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - CI installs torch
    torch = None

NL = chr(10)
PROBE = Path(os.environ.get('QWEN_SDPA_PREFILL_SRC', 'C:/Users/liamb/.claude/jobs/8376c877/tmp/probe-v25/src/device'))
FACTORY_FIXTURE = HERE / 'fixtures' / 'sdpa_program_factory.fd8c0676.cpp'
# The served factory: the probe tree's when present (it must equal the fixture), else the committed
# fixture, so the factory-patch tests run in CI as well.
FACTORY_BASE = PROBE / 'sdpa_program_factory.cpp' if (PROBE / 'sdpa_program_factory.cpp').is_file() else FACTORY_FIXTURE
DUMP = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp/decode-sources-35503727180.txt')
CHAIN = HERE / gen.OUTPUT_NAME
MODEL_SEEDS = int(os.environ.get('QWEN_PF_MODEL_SEEDS', '40'))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def served_reader():
    """The served reader, rebuilt from the committed K0 probe reader k0a (it reverts to f97f5490)."""
    text = (BENCH / 'k0' / k0.OUTPUT_NAMES['k0a']).read_bytes().decode('utf-8')
    return k0.revert_edits(text, 'k0a').encode('utf-8')


def chain_text():
    return CHAIN.read_bytes().decode('utf-8')


def strip_comments(text):
    return re.sub(r'//[^\n]*', '', text)


def call_args(text, name, start=0):
    """Top-level arguments of the first call to `name` (template args skipped) at or after start."""
    index = text.index(name, start)
    open_index = text.index('(', index)
    depth, args, current = 0, [], []
    for position in range(open_index, len(text)):
        char = text[position]
        if char in '([{':
            depth += 1
            if depth == 1:
                continue
        elif char in ')]}':
            depth -= 1
            if depth == 0:
                args.append(''.join(current))
                return [' '.join(arg.split()) for arg in args if arg.strip()], position
        elif char == ',' and depth == 1:
            args.append(''.join(current))
            current = []
            continue
        current.append(char)
    raise AssertionError('unbalanced call to %s' % name)


def balanced(text):
    """Brace / paren balance of C++ text outside comments and string literals."""
    text = strip_comments(re.sub(r'"(?:[^"\\]|\\.)*"', '""', text))
    return text.count('{') - text.count('}'), text.count('(') - text.count(')')


# ---------------------------------------------------------------------------------------------
# 1. patch integrity and cross-file constants
# ---------------------------------------------------------------------------------------------

class FactoryPatchTests(unittest.TestCase):
    def test_the_committed_fixture_is_the_served_factory(self):
        data = FACTORY_FIXTURE.read_bytes()
        self.assertEqual(sha(data), factory.BASE_FACTORY)
        self.assertNotIn(b'\r', data)
        self.assertIn(b'SPDX-License-Identifier: Apache-2.0', data[:200])
        probe = PROBE / 'sdpa_program_factory.cpp'
        if probe.is_file():
            self.assertEqual(probe.read_bytes(), data)

    def test_the_noc_order_search_is_bounded(self):
        block = factory.F4[factory.F4.index('if (qwen_pf_flags & kQwenPfNocOrder) {'):]
        fatal = block.index('TT_FATAL(members.size() <= %d' % factory.NOC_ORDER_MAX_MEMBERS)
        self.assertLess(fatal, block.index('std::next_permutation'))
        self.assertIn(factory.NOC_ORDER_MARKER, block)
        self.assertIn(factory.NOC_ORDER_MARKER, factory.MARKERS)

    def test_any_other_input_is_refused(self):
        with self.assertRaisesRegex(factory.Refusal, 'unexpected factory'):
            factory.patch(b'// not the served factory' + NL.encode())

    def test_every_new_block_is_balanced_and_tagged(self):
        for label, _line, old, new in factory.EDITS:
            with self.subTest(edit=label):
                self.assertEqual(balanced(new), balanced(old))
                if label != 'F0':
                    self.assertTrue('QWEN-SDPA-PF' in new or 'qwen_kv_chain' in new)
                if new.startswith(old) or new.endswith(old):
                    continue                                # an insertion keeps its anchor verbatim
                # A replacement (F2, F3, F5, F6) only widens the served condition by `|| qwen_kv_chain`
                # or picks the chain reader under qwen_kv_chain: the served text stays reachable.
                self.assertIn('qwen_kv_chain', new)
                self.assertNotIn(chr(13), new)

    def test_the_base_gives_the_recorded_factory_and_inverts(self):
        base = FACTORY_BASE.read_bytes()
        self.assertEqual(sha(base), factory.BASE_FACTORY)
        patched = factory.patch(base)
        self.assertEqual(sha(patched), factory.PF_FACTORY)
        self.assertEqual(factory.unpatch(patched), base)
        text = patched.decode('utf-8')
        self.assertEqual(balanced(text), balanced(base.decode('utf-8')))
        self.assertEqual(factory.protected_block(text), factory.protected_block(base.decode('utf-8')))
        for marker in factory.MARKERS:
            self.assertIn(marker, text)
        # Nothing at namespace scope: every edit but the includes is inside create_descriptor, and the
        # anonymous namespace is byte-identical.
        anon = lambda t: t[t.index('namespace {'):t.index('}  // namespace', t.index('namespace {'))]
        self.assertEqual(anon(text), anon(base.decode('utf-8')))

    def test_anchors_are_unique_at_their_recorded_lines(self):
        text = FACTORY_BASE.read_text(encoding='utf-8')
        factory.check_anchors(text)
        self.assertEqual(text.count('if (!is_causal) {' + NL), 3)   # spec 3.2: why F2/F3/F5 anchor two lines each
        drifted = text.replace(factory.F1_ANCHOR, NL + factory.F1_ANCHOR)
        with self.assertRaisesRegex(factory.Refusal, 'F1 starts at line 528'):
            factory.check_anchors(drifted)

    def test_the_envelope_names_exist_at_function_scope_before_f4(self):
        text = FACTORY_BASE.read_text(encoding='utf-8')
        f4 = text.index(factory.F4_ANCHOR)
        body = text[text.index(factory.FUNCTION_LINE):f4]
        self.assertIsNotNone(re.search(r'^    const uint32_t qk_in0_num_subblocks = Sq_chunk_t / qk_out_subblock_h;$',
                                       body, re.M))

    def test_main_writes_reports_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'factory.cpp'
            shutil.copyfile(FACTORY_BASE, target)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(factory.main([str(target)]), 0)
                self.assertEqual(sha(target.read_bytes()), factory.PF_FACTORY)
                self.assertTrue((Path(directory) / ('factory.cpp.orig-' + factory.BASE_FACTORY[:8])).is_file())
                self.assertEqual(factory.main([str(target)]), 0)          # already patched: report, exit 0
                target.write_bytes(b'junk')
                self.assertEqual(factory.main([str(target)]), 1)


class ReaderPatchTests(unittest.TestCase):
    def test_the_committed_reader_is_the_recorded_output(self):
        data = CHAIN.read_bytes()
        self.assertEqual(sha(data), gen.READER_OUTPUT)
        self.assertNotIn(b'\r', data)

    def test_it_regenerates_from_the_served_reader_and_reverts_to_it(self):
        base = served_reader()
        self.assertEqual(sha(base), gen.BASE_SHA)
        self.assertEqual(gen.build(base), CHAIN.read_bytes())
        self.assertEqual(gen.revert_edits(chain_text(), gen.r7_old(base.decode('utf-8'))), base.decode('utf-8'))
        self.assertEqual(balanced(chain_text()), (0, 0))

    @unittest.skipUnless((PROBE / 'kernels/dataflow/reader_interleaved.cpp').is_file(), 'no probe-v25 tree')
    def test_the_probe_tree_reader_is_the_base(self):
        self.assertEqual((PROBE / 'kernels/dataflow/reader_interleaved.cpp').read_bytes(), served_reader())

    def test_a_wrong_base_and_anchor_drift_are_refused(self):
        with self.assertRaisesRegex(gen.Refusal, 'not the served reader'):
            gen.build(served_reader() + b' ')
        text = served_reader().decode('utf-8')
        with self.assertRaisesRegex(gen.Refusal, 'R2 envelope: anchor starts at line 98'):
            gen.check_anchors(text.replace(gen.R2_ANCHOR, NL + gen.R2_ANCHOR))
        with self.assertRaisesRegex(gen.Refusal, 'occurs 2 times'):
            gen.check_anchors(text + gen.R4_ANCHOR)

    def test_every_edit_is_balanced(self):
        text = served_reader().decode('utf-8')
        for name, _line, old, new in gen.edits(text):
            with self.subTest(edit=name):
                self.assertEqual(balanced(new), balanced(old))

    def test_the_r7_span_is_served_lines_406_to_709(self):
        text = served_reader().decode('utf-8')
        span = gen.r7_old(text)
        lines = text.split(NL)
        self.assertEqual(span, NL.join(lines[405:709]) + NL)
        self.assertIn('Forward V chunk to next core', span)
        self.assertEqual(lines[709].strip(), '}  // close k_chunk')

    def test_main_check_and_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            served = Path(directory) / 'reader_interleaved.cpp'
            served.write_bytes(served_reader())
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(gen.main(['--reader', str(served), '--check', str(HERE)]), 0)
                self.assertEqual(gen.main(['--reader', str(served), '--check', directory]), 1)
                self.assertEqual(gen.main(['--reader', str(served), '--out', directory]), 0)
                served.write_bytes(b'x')
                self.assertEqual(gen.main(['--reader', str(served), '--check', str(HERE)]), 2)
            self.assertEqual((Path(directory) / gen.OUTPUT_NAME).read_bytes(), CHAIN.read_bytes())


class ConstantTests(unittest.TestCase):
    def test_tag_and_flag_bits_agree_between_f1_r4_and_the_python(self):
        self.assertIn('kQwenPfTag = 0x%08Xu' % factory.PF_TAG, factory.F1)
        self.assertIn('kQwenPfTagMask = 0x%08Xu' % factory.PF_TAG_MASK, factory.F1)
        for name, bit in (('kQwenPfKvChain', factory.FLAG_KV_CHAIN), ('kQwenPfInjBatch', factory.FLAG_INJ_BATCH),
                          ('kQwenPfNocOrder', factory.FLAG_NOC_ORDER), ('kQwenPfTestMutate', factory.FLAG_TEST_MUTATE),
                          ('kQwenPfTestHang', factory.FLAG_TEST_HANG)):
            self.assertIn('%s = %#xu' % (name, bit), factory.F1)
        reader = chain_text()
        self.assertIn('pf_inj_batch = (qwen_pf_flags & %#xu) != 0' % factory.FLAG_INJ_BATCH, reader)
        self.assertIn('pf_test_mutate = (qwen_pf_flags & %#xu) != 0' % factory.FLAG_TEST_MUTATE, reader)
        self.assertIn('pf_test_hang = (qwen_pf_flags & %#xu) != 0' % factory.FLAG_TEST_HANG, reader)
        self.assertIn('std::getenv("%s")' % factory.TEST_ENV, factory.F1)
        for module in (card,):
            self.assertEqual(module.PF_TAG, factory.PF_TAG)
            self.assertEqual(module.TEST_ENV, factory.TEST_ENV)
        # The model's default is Q2's choice, the chain plus the injector read-ahead (0x3); the card-M
        # test's stress / trace / cache flag stays the 0x1 it was qualified with (Q1 swept all four).
        self.assertEqual(optin.DEFAULT_FLAGS, factory.FLAG_KV_CHAIN | factory.FLAG_INJ_BATCH)
        self.assertEqual(card.DEFAULT_FLAGS, factory.FLAG_KV_CHAIN)
        self.assertEqual(card.BINARY_MARKER, factory.LOG_MARKER.encode())
        self.assertEqual(optin.BINARY_MARKER, factory.LOG_MARKER.encode())

    def test_the_serving_graft_constants_are_the_factorys(self):
        """lever_n_m3native_patch section I (scripts/ci) cannot import this directory (it must stay
        standalone for the rig harnesses that copy it alone), so its constants are held here."""
        graft = optin.graft
        self.assertEqual(graft.SDPA_PF_TAG, factory.PF_TAG)
        self.assertEqual(graft.SDPA_PF_BINARY_MARKER, factory.LOG_MARKER)
        self.assertEqual(graft.SDPA_PF_ROWS, factory.QUALIFIED_ROWS)
        self.assertEqual(graft.SDPA_PF_CHUNK, factory.QUALIFIED_CHUNK)
        self.assertEqual(tuple(sorted(graft.SDPA_PF_PRODUCTION_FLAGS)), tuple(sorted(card.PRODUCTION_FLAGS)))
        for flags in graft.SDPA_PF_PRODUCTION_FLAGS:
            with self.subTest(flags=flags):
                self.assertEqual(flags & ~factory.PRODUCTION_FLAGS, 0)          # never a test or unknown bit
                self.assertEqual(factory.decode_word(graft.SDPA_PF_TAG | flags), (True, flags))
        self.assertIn(graft.SDPA_PF_DEFAULT_FLAGS, graft.SDPA_PF_PRODUCTION_FLAGS)
        import sdpa_prefill_bench as bench
        chain_b = bench.arm_by_name('chain_b')['program_word']
        self.assertEqual(chain_b, graft.SDPA_PF_TAG | graft.SDPA_PF_DEFAULT_FLAGS)   # the Q2 arm the default is

    def test_the_flags_ct_index_is_cb_arg_offset_plus_8(self):
        self.assertIn('get_compile_time_arg_val(cb_arg_offset + %d)' % factory.FLAGS_CT_OFFSET, chain_text())
        self.assertIn('reader CT cb_arg_offset + %d' % factory.FLAGS_CT_OFFSET, factory.F7)
        served = served_reader().decode('utf-8')
        # The served reader reads CB ids at +0..+7 and nothing beyond: +8 is free for the suffix.
        self.assertIn('get_compile_time_arg_val(cb_arg_offset + 7)', served)
        self.assertNotIn('cb_arg_offset + 8', served)
        # F7 appends after the CB-id insert, i.e. after every accessor block and CB id.
        self.assertTrue(factory.F7.startswith(factory.F7_ANCHOR))

    def test_semaphore_ids_are_the_reader_ct_29_30_31(self):
        served = served_reader().decode('utf-8')
        for name, index in (('sender_semaphore_id', 29), ('receiver_semaphore_id', 30), ('valid_semaphore_id', 31)):
            self.assertIn('constexpr uint32_t %s = get_compile_time_arg_val(%d);' % (name, index), served)
        self.assertEqual(factory.SEMAPHORE_IDS, dict(sender=0, receiver=1, valid=2))

    def test_the_factory_assigns_0_1_2_and_initial_invalid_invalid_valid(self):
        text = FACTORY_BASE.read_text(encoding='utf-8')
        block = text[text.index(factory.F2_OLD):text.index(factory.F2_OLD) + 400]
        self.assertIn('receiver_semaphore_id = 1;', block)
        self.assertIn('valid_semaphore_id = 2;', block)
        sems = text[text.index(factory.F3_OLD):text.index(factory.F3_OLD) + 900]
        self.assertEqual(re.findall(r'\.initial_value = (\w+)', sems), ['INVALID', 'INVALID', 'VALID'])

    def test_the_14_word_rt_block_order_agrees_with_the_reader_parse(self):
        reader = chain_text()
        block = reader[reader.index(gen.R3_NEW):reader.index('global_q_start = get_arg_val')]
        parsed = re.findall(r'(\w+) = get_arg_val<uint32_t>\(argidx\+\+\)|argidx \+= (\d+)', block)
        names = []
        for name, skip in parsed:
            names.extend([name] if name else ['<skip>'] * int(skip))
        self.assertEqual(names, ['is_chain_participant', 'is_injector', 'is_sink', 'chain_batch', 'chain_head',
                                 '<skip>', '<skip>', 'prev_physical_x', 'prev_physical_y', 'next_physical_x',
                                 'next_physical_y', 'next_core_q_chunks', 'mcast_num_dests', 'mcast_sender_wait'])
        self.assertEqual(len(factory.CHAIN_RT_WORDS), 14)
        self.assertEqual(len(model.RT_ORDER), 14)

    def test_f5_pushes_the_14_words_in_the_parse_order(self):
        text = FACTORY_BASE.read_text(encoding='utf-8')
        start = text.index(factory.F5_OLD)
        block = text[start:text.index('        }', start + len(factory.F5_OLD))]
        pushed = re.findall(r'reader_args\.push_back\((?:static_cast<uint32_t>\()?chain\.([\w.]+)\)', block)
        self.assertEqual(pushed, list(factory.CHAIN_RT_WORDS))


# ---------------------------------------------------------------------------------------------
# 2-3. topology mirror and round model
# ---------------------------------------------------------------------------------------------

class TopologyTests(unittest.TestCase):
    def test_2048_rows_is_16_groups_of_6_at_48g_plus_8s_plus_m(self):
        topo = model.g6_topology()
        self.assertEqual((topo['chains'], topo['members']), (16, 96))
        expected = sorted(sorted(48 * g + 8 * s + m for s in range(6)) for g in range(2) for m in range(8))
        self.assertEqual(sorted(sorted(members) for members in topo['groups'].values()), expected)
        self.assertEqual(topo['groups'][(0, 0, 0, 0, 0, 15)], [0, 8, 16, 24, 32, 40])   # (g=0, m=0)
        for core in range(96, 110):
            self.assertEqual(topo['info'][core]['participates'], 0)
            self.assertEqual(model.rt_words(topo['info'][core]), [0] * 14)

    def test_1024_and_512_rows(self):
        for rows, chains in ((1024, 8), (512, 4)):
            with self.subTest(rows=rows):
                topo = model.g6_topology(model.geometry(rows=rows))
                self.assertEqual((topo['chains'], topo['members']), (chains, 6 * chains))
                self.assertEqual(card.expected_chains(rows), (chains, 6 * chains))
        self.assertEqual(card.expected_chains(2048), (16, 96))

    def test_every_group_shares_one_unit_list_and_one_kv_head(self):
        geo = model.geometry()
        units = model.unit_lists(geo)
        for key, members in model.g6_topology(geo)['groups'].items():
            lists = {tuple((nb, nq // 6, q) for nb, nq, q in units[core]) for core in members}
            self.assertEqual(len(lists), 1)
            heads = {nq for core in members for _nb, nq, _q in units[core]}
            self.assertEqual(len(heads), 6)                      # six Q heads share one KV stream
            self.assertEqual(len({h // 6 for h in heads}), 1)

    def test_roles_and_neighbours(self):
        topo = model.g6_topology()
        coords = {i: (i % 11, i // 11) for i in range(110)}
        for members in topo['groups'].values():
            infos = [topo['info'][core] for core in members]
            self.assertEqual([info['is_injector'] for info in infos], [1, 0, 0, 0, 0, 0])
            self.assertEqual([info['is_sink'] for info in infos], [0, 0, 0, 0, 0, 1])
            self.assertEqual(members, sorted(members))           # raster: ascending linear index
            for p, core in enumerate(members):
                info = infos[p]
                if p + 1 < len(members):
                    self.assertEqual((info['next_x'], info['next_y']), coords[members[p + 1]])
                    self.assertEqual(info['next_core_q_chunks'], 2)
                if p > 0:
                    self.assertEqual((info['prev_x'], info['prev_y']), coords[members[p - 1]])
                self.assertEqual((info['mcast_num_dests'], info['mcast_sender_wait']), (0, 0))

    def test_zigzag_decode_and_ranges_match_the_kernel_formulas(self):
        self.assertEqual(model.decompose(0, 16, 12, True), (0, 0, 0))
        self.assertEqual(model.decompose(1, 16, 12, True), (0, 0, 15))
        self.assertEqual(model.decompose(33, 16, 12, True), (0, 2, 15))
        self.assertEqual(model.decompose(33, 16, 12, False), (0, 2, 1))
        ranges = model.core_ranges(110, 192, True)
        self.assertEqual(ranges[:2], [(0, 2), (2, 2)])
        self.assertEqual(ranges[95], (190, 2))
        self.assertEqual(ranges[96], (192, 0))

    def synthetic_coords(self):
        """A SYNTHETIC worker-coordinate map (not the card-M fixture): BH-like columns 1-7 and 10-16,
        rows 2-11, with a harvested-row style jump, so the NoC order has something to minimise."""
        columns = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]
        rows = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        return {i: (columns[i % 11], rows[i // 11]) for i in range(110)}

    def check_noc_order(self, coords):
        raster = model.g6_topology(coords=coords)
        noc = model.g6_topology(coords=coords, noc_order=True)
        again = model.g6_topology(coords=coords, noc_order=True)
        self.assertEqual(noc['groups'], again['groups'])           # deterministic
        for key, members in noc['groups'].items():
            self.assertEqual(sorted(members), sorted(raster['groups'][key]))
            phys = {core: coords[core] for core in members}
            cost = lambda order: model.order_cost([phys[c] for c in order], list(range(len(order))))
            self.assertLessEqual(cost(members), cost(raster['groups'][key]))
        return raster, noc

    def test_noc_order_on_a_synthetic_map(self):
        raster, noc = self.check_noc_order(self.synthetic_coords())
        self.assertNotEqual(raster['groups'], noc['groups'])       # the knob does something here

    def test_noc_order_on_the_card_m_fixture(self):
        fixture = HERE / 'fixtures' / 'cardm_worker_coords.json'
        if not fixture.is_file():
            self.skipTest('fixtures/cardm_worker_coords.json not captured yet (copy the K0 session\'s file)')
        data = json.loads(fixture.read_text(encoding='utf-8'))
        coords = {entry['i']: tuple(entry['worker']) for entry in data['cores']}
        self.assertEqual(len(coords), 110)
        self.check_noc_order(coords)

    def test_the_cost_wraps_like_uint32(self):
        self.assertEqual(model.order_cost([(0, 0), (16, 0)], [0, 1]), 1)     # min(16, 17 - 16)
        self.assertEqual(model.order_cost([(0, 0), (20, 0)], [0, 1]), 20)    # 17 - 20 wraps: min picks dx


class RoundModelTests(unittest.TestCase):
    def test_every_member_runs_the_same_rounds_for_c_0_to_1100(self):
        geo = model.geometry()
        units = model.unit_lists(geo)
        groups = model.g6_topology(geo)['groups'].values()
        for C in range(0, 1101):
            ext = model.extents(C)
            for members in groups:
                lists = {tuple(model.round_list(C, units[core], 6, ext)) for core in members}
                self.assertEqual(len(lists), 1, 'C=%d' % C)
                (compact,) = lists
                self.assertEqual(sum(n for *_rest, n in compact), 2 * C + 17)
                for nb, kvh, q, n in compact:
                    self.assertEqual(n, model.compute_k_chunks(C, q, ext))
                    self.assertEqual(model.kv_row_tiles(C, n - 1, ext), 4)   # full rows up to the last chunk

    def test_the_loop_count_equals_the_closed_form_at_small_c(self):
        for C in (0, 1, 15, 16, 33, 64, 513, 992):
            ext = model.extents(C)
            for q in range(16):
                self.assertEqual(model.reader_k_chunks(C, q, ext), model.compute_k_chunks(C, q, ext))
                self.assertEqual(model.reader_k_chunks(C, q, ext), model.reader_k_count(C, q, ext))
                for k in range(model.reader_k_chunks(C, q, ext)):
                    self.assertEqual(model.kv_row_tiles(C, k, ext), 4)

    def test_slot_parity_flips_identically_at_odd_c(self):
        for C in (1, 33, 513):
            rounds = model.served_group_rounds(C, 3)
            boundary = [n for n, (_u, _k, first, _c) in enumerate(rounds) if first]
            self.assertEqual(boundary, [0, C + 4])                     # unit 1 starts at round C + m + 1
            self.assertEqual(boundary[1] % 2, (C + 4) % 2)

    def test_page_width_follows_the_model_pad(self):
        self.assertEqual(model.page_blocks(992), 2016)
        self.assertEqual(card.page_width('fit', 2048, card.STARTS), 2016)
        self.assertEqual(card.page_width('model', 2048, card.STARTS), 2080)


# ---------------------------------------------------------------------------------------------
# 4. the protocol model checker
# ---------------------------------------------------------------------------------------------

class ModelCheckerTests(unittest.TestCase):
    def test_ok_terminates_consumes_the_right_chunks_and_resets(self):
        failures, runs = model.sweep(range(MODEL_SEEDS))
        self.assertEqual(runs, MODEL_SEEDS * 18)
        self.assertEqual(failures['ok'], [])

    def test_the_planted_hang_deadlocks_on_exactly_the_sink_and_p4(self):
        failures, _ = model.sweep(range(8), variants=('hang',))
        self.assertEqual(failures['hang'], [])
        result = model.simulate(16, 7, 3, 'hang')
        self.assertEqual(result.blocked, model.HANG_BLOCKED)
        self.assertEqual(result.errors, [])                         # nothing stale is ever pushed

    def test_a_withheld_credit_that_also_skips_the_reset_is_caught(self):
        """The first K64g reader skipped the whole credit_prev call under 0x200: the sink saw its
        previous round's VALID, pushed a stale slot and exited, and only P4 hung (review 0 finding 1)."""
        for C, m, seed in ((16, 7, 3), (0, 0, 1), (33, 3, 5)):
            with self.subTest(C=C, m=m, seed=seed):
                result = model.simulate(C, m, seed, 'hang', mutation='withhold_skips_reset')
                self.assertEqual(result.blocked, {'reader 4': 'credit'})
                self.assertTrue(any('K slot holds' in error for error in result.errors))
                self.assertTrue(model.check_hang(result))
        with self.assertRaises(ValueError):
            model.simulate(0, 0, 0, mutation='no_such_mutation')

    def test_members_with_different_c_deadlock(self):
        failures, _ = model.sweep(range(8), variants=('wrong_c',))
        self.assertEqual(failures['wrong_c'], [])

    def test_equal_length_divergence_is_silent_corruption(self):
        """The case the spec's 4.2 'a hang, not corruption' does not cover: why F4 keys on the whole list."""
        failures, _ = model.sweep(range(4), variants=('wrong_m',))
        self.assertEqual(failures['wrong_m'], [])
        result = model.simulate(33, 0, 1, 'wrong_m')
        self.assertTrue(result.terminated)
        self.assertTrue(any('expected' in error for error in result.errors))

    def test_the_checker_catches_protocol_mutations(self):
        for mutation in model.MUTATIONS[1:]:
            with self.subTest(mutation=mutation):
                caught = sum(1 for seed in range(10) if model.check_ok(model.simulate(15, 3, seed, mutation=mutation)))
                self.assertGreaterEqual(caught, 5, mutation)

    def test_runs_are_deterministic(self):
        first = model.simulate(16, 3, 11).as_dict()
        self.assertEqual(model.simulate(16, 3, 11).as_dict(), first)
        self.assertTrue(first['terminated'])


# ---------------------------------------------------------------------------------------------
# 5. reader transcription and structure
# ---------------------------------------------------------------------------------------------

class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.served = strip_comments(served_reader().decode('utf-8'))
        self.chain = strip_comments(chain_text())
        self.r7 = strip_comments(gen.R7_NEW)

    def served_call(self, name, marker):
        return call_args(self.served, name, self.served.index(marker))[0]

    def test_the_k_v_and_q_calls_are_the_served_ones_token_for_token(self):
        served_k = self.served_call('read_paged_chunk_with_padding<NKH', 'const uint32_t k_chunk_start_row_num')
        served_v = self.served_call('read_paged_chunk_with_padding<NVH', 'const uint32_t kv_chunk_start_row_num')
        served_q = self.served_call('read_q_subblock<q_tile_bytes>', 'if constexpr (use_q_subblock_push) {\n'
                                    '                    if (k_chunk == k_loop_start)')
        chain_k = call_args(self.r7, 'read_paged_chunk_with_padding<NKH')[0]
        chain_v = call_args(self.r7, 'read_paged_chunk_with_padding<NVH')[0]
        chain_q = call_args(self.r7, 'read_q_subblock<q_tile_bytes>')[0]
        self.assertEqual(len(served_k), 12)
        self.assertEqual([i for i, (a, b) in enumerate(zip(served_k, chain_k)) if a != b], [2, 9])
        self.assertEqual((served_k[2], chain_k[2]), ('k_head', 'k_head_rd'))
        self.assertEqual((served_k[9], chain_k[9]), ('barrier_threshold', 'kv_bt'))
        self.assertEqual(len(served_v), 13)
        self.assertEqual([i for i, (a, b) in enumerate(zip(served_v, chain_v)) if a != b], [9])
        self.assertEqual((served_v[9], chain_v[9]), ('barrier_threshold', 'kv_bt'))
        self.assertEqual(served_q, chain_q)
        self.assertEqual((len(served_k), len(served_v)), (len(chain_k), len(chain_v)))

    def test_the_push_order_is_k_then_q_then_v_then_forward(self):
        order = [self.r7.index(token) for token in (
            'cb_k.push_back(k_chunk_tiles)', 'read_paged_chunk_with_padding<NKH', 'read_q_subblock',
            'cb_v.push_back(v_chunk_tiles)', 'read_paged_chunk_with_padding<NVH', 'qwen_pf::forward_round')]
        self.assertEqual(order, sorted(order))
        self.assertLess(self.r7.index('qwen_pf::credit_prev'), self.r7.index('qwen_pf::wd_wait_valid'))
        self.assertLess(self.r7.index('cb_v.reserve_back(v_chunk_tiles)'), self.r7.index('qwen_pf::credit_prev'))
        self.assertLess(self.r7.index('noc.async_atomic_barrier()'), self.r7.index('read_q_subblock'))

    def test_the_slots_are_captured_before_any_reserve(self):
        self.assertLess(self.r7.index('k_slot = cb_k.get_write_ptr()'), self.r7.index('cb_k.reserve_back'))
        self.assertLess(self.r7.index('v_slot = cb_v.get_write_ptr()'), self.r7.index('read_paged_chunk_with_padding<NKH'))

    def test_the_forward_writes_both_slots_acks_then_relays(self):
        helper = strip_comments(gen.R1)
        body = helper[helper.index('FORCE_INLINE void forward_round'):]
        order = [body.index(token) for token in ('wd_wait_credit(sender_sem_id)', 'Semaphore<>(sender_sem_id).set(0)',
                                                 '.addr = k_addr', '.addr = v_addr', 'noc.async_write_barrier()',
                                                 'relay_unicast', 'noc.async_writes_flushed()')]
        self.assertEqual(order, sorted(order))
        credit = helper[helper.index('FORCE_INLINE void credit_prev'):helper.index('FORCE_INLINE void forward_round')]
        self.assertLess(credit.index('set(INVALID)'), credit.index('.up(noc, px, py, 1)'))
        # The reset is unconditional; only the credit sits under `withhold` (the planted hang).
        self.assertLess(credit.index('set(INVALID)'), credit.index('if (!withhold)'))
        self.assertLess(credit.index('if (!withhold)'), credit.index('.up(noc, px, py, 1)'))

    def test_bounded_waits_never_fall_through(self):
        helper = strip_comments(gen.R1)
        for name, waypoint in (('wd_wait_credit', 'QWDC'), ('wd_wait_valid', 'QWDV')):
            body = helper[helper.index('void %s' % name):]
            body = body[:body.index(NL + '}')]
            fire = body[body.index('if (n == kWdPolls)'):]
            self.assertLess(fire.index('WAYPOINT("%s")' % waypoint), fire.index('ASSERT(false)'))
            self.assertIn('for (;;)', fire)
            self.assertNotIn('return', fire)
            self.assertNotIn('break', fire)
        self.assertIn('kWdPolls = 1u << %d' % gen.WATCHDOG_POLLS_LOG2, helper)

    def test_the_exit_resets_after_barriers_for_participants_only(self):
        raw = chain_text()
        tail = strip_comments(raw[raw.index('}  // close phase'):])
        order = [tail.index(token) for token in ('if (is_chain_participant)', 'noc.async_write_barrier()',
                                                 'noc.async_atomic_barrier()', 'Semaphore<>(receiver_semaphore_id).set(INVALID)',
                                                 'Semaphore<>(sender_semaphore_id).set(INVALID)')]
        self.assertEqual(order, sorted(order))

    def test_roles_share_every_round_and_the_hang_withholds_only_the_sinks_last_credit(self):
        self.assertIn('const bool should_forward = is_chain_participant && !is_sink;', self.chain)
        self.assertIn('const bool should_receive = is_chain_participant && !is_injector;', self.chain)
        # One call on every round: credit_prev resets the flag whatever `withhold` is, so the sink's
        # last VALID wait really waits (QWDV) instead of consuming its previous round's VALID.
        args = call_args(self.r7, 'qwen_pf::credit_prev')[0]
        self.assertEqual(args[-1], 'pf_test_hang && is_sink && last_round')
        self.assertEqual(self.r7.count('qwen_pf::credit_prev'), 1)
        self.assertNotIn('if (!(pf_test_hang', self.r7)
        self.assertNotIn('set(INVALID)', self.r7)                   # no reset outside the helper
        self.assertIn('(pf_test_mutate && is_injector && k_chunk == 0) ? (k_head ^ 1u) : k_head', self.r7)
        self.assertIn('(pf_inj_batch && is_injector) ? k_chunk_tiles : barrier_threshold', self.chain)

    def test_q_is_always_the_subblock_read_behind_the_atomic_barrier(self):
        self.assertIn('static_assert(use_q_subblock_push, "[QWEN-SDPA-PF] the chain reader needs the Q subblock push");',
                      self.chain)
        self.assertLess(self.chain.index('constexpr bool use_q_subblock_push'), self.chain.index('static_assert(use_q_subblock_push'))
        self.assertIn('qk_in0_num_subblocks > 1', factory.ENVELOPE)

    def test_the_orphans_are_marked_and_nothing_else_is(self):
        for name, _line, decl in gen.R9_ORPHANS:
            self.assertIn('[[maybe_unused]] ' + decl.lstrip(), chain_text())
        served = served_reader().decode('utf-8').count('[[maybe_unused]]')
        self.assertEqual(chain_text().count('[[maybe_unused]]') - served, len(gen.R9_ORPHANS))


# ---------------------------------------------------------------------------------------------
# 6. flags and envelope
# ---------------------------------------------------------------------------------------------

class FlagsTests(unittest.TestCase):
    def test_the_refusal_table(self):
        self.assertEqual(factory.decode_word(16), (False, 0))          # the default never carries the tag
        self.assertEqual(factory.decode_word(0x5EFB0001), (False, 0))
        for flags in (0x1, 0x3, 0x5, 0x7):
            self.assertEqual(factory.decode_word(factory.PF_TAG | flags), (True, flags))
        for flags in (0x0, 0x2, 0x8, 0x1000, 0x1001, 0x401):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, 'unknown or incomplete flags'):
                factory.decode_word(factory.PF_TAG | flags)
        for flags in (0x101, 0x201, 0x303):
            with self.assertRaisesRegex(ValueError, 'test-only flags'):
                factory.decode_word(factory.PF_TAG | flags)
            with self.assertRaisesRegex(ValueError, 'test-only flags'):
                factory.decode_word(factory.PF_TAG | flags, test_env='yes')
            self.assertEqual(factory.decode_word(factory.PF_TAG | flags, test_env='1'), (True, flags))

    def test_every_envelope_predicate_is_in_the_f4_fatal(self):
        fatal = factory.F4[factory.F4.index('TT_FATAL('):factory.F4.index('"[QWEN-SDPA-PF] kv_chain outside')]
        condition = ' '.join(fatal.split())
        for predicate in factory.ENVELOPE:
            self.assertIn(predicate, condition)
        self.assertEqual(condition.count('&&') + 1, len(factory.ENVELOPE))
        self.assertEqual(len(factory.ENVELOPE), 19)                 # spec 3.2's eighteen + the Q subblock push

    def test_the_refusal_texts_the_card_m_test_expects_are_the_factorys(self):
        for _name, _change, needle in card.REFUSALS:
            self.assertTrue(any(needle in text for text in (factory.F1, factory.F4)), needle)

    def test_the_log_line_matches_every_parser(self):
        line = '[QWEN-SDPA-PF] flags=0x5 kv_chain=1 chains=16 members=96 order=noc'
        self.assertIn('"[QWEN-SDPA-PF] flags={:#x} kv_chain=1 chains={} members={} order={}"', factory.F4)
        self.assertEqual(card.factory_lines(line), [dict(flags=5, chains=16, members=96, order='noc')])
        import sdpa_prefill_bench as bench
        self.assertEqual(bench.pf_log_lines(line), [dict(flags=5, chains=16, members=96, order='noc')])


# ---------------------------------------------------------------------------------------------
# 7. the Python opt-in
# ---------------------------------------------------------------------------------------------

TP_EXCERPT = NL.join((
    'import os',
    '',
    'class TtAttention:',
    '    def __init__(self, args):',
    '        self.compute_cfg = None',
    '        self._sdpa_bf8 = os.environ.get("QWEN_SDPA_BF8", "0") == "1"',
    '',
    '    def forward_decode(self, x):',
    '        sdpa_cfg = ttnn.SDPAProgramConfig(',
    '            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),',
    '            exp_approx_mode=False,',
    '            q_chunk_size=0,',
    '            k_chunk_size=256,',
    '        )',
    '',
    '    def forward_prefill_paged(self, x, page_table, chunk_start_idx=0, chunk_start_idx_tensor=None):',
    '        S = x.shape[-2]',
    '        qk_chunk = 128',
    '        sdpa_cfg = ttnn.SDPAProgramConfig(',
    '            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),',
    '            exp_approx_mode=False,',
    '            q_chunk_size=qk_chunk,',
    '            k_chunk_size=qk_chunk,',
    '        )',
    '        return sdpa_cfg',
    ''))


def dumped_tp():
    """attention/tp.py from the probe 35503727180 dump (line-numbered), or None."""
    if not DUMP.is_file():
        return None
    dump = DUMP.read_text(encoding='utf-8')
    start = dump.index('/opt/tt-metal/models/demos/blackhole/qwen36/tt/attention/tp.py  sha256=')
    body = dump.index(NL, dump.index('=====', dump.index(NL, start))) + 1
    lines = []
    for line in dump[body:].split(NL):
        match = re.match(r'^\s*(\d+)  (.*)$', line) or re.match(r'^\s*(\d+)$', line)
        if not match:
            break
        lines.append(match.group(2) if match.lastindex == 2 else '')
    return NL.join(lines) + NL


FIXTURE_TP = HERE.parents[2] / 'scripts' / 'ci' / 'fixtures' / 'qwen36_attention_tp.py'
SERVED_KWARGS = dict(compute_with_storage_grid_size=(11, 10), exp_approx_mode=False, q_chunk_size=128, k_chunk_size=128)


class OptinTests(unittest.TestCase):
    """The opt-in is lever_n_m3native_patch section I (wired into the gate's attention/tp.py graft);
    pf_optin re-exports it. scripts/ci/test_lever_n_m3native_sdpa_pf.py drives the whole grafted
    forward_prefill_paged of the real tp.py; these check the helpers and the patch on an excerpt."""

    def setUp(self):
        self.built = []
        ttnn = SimpleNamespace(SDPAProgramConfig=lambda **kw: self.built.append(kw) or dict(kw))
        self.ns = optin.helpers_namespace(ttnn_module=ttnn)
        self.pindiag = []
        self.ns['_qwen_pf_log'] = self.pindiag.append

    def word(self, env, has_marker=True):
        calls = []

        def check():
            calls.append(1)
            return has_marker
        return self.ns['_qwen_pf_word'](env, check), calls

    def config(self, word, flexible=True, bf8=True, S=2048, qk_chunk=128):
        """(served, what the call site sends) for one prefill call."""
        layer = SimpleNamespace(_sdpa_pf_word=word, _sdpa_bf8=bf8,
                                mesh=SimpleNamespace(compute_with_storage_grid_size=lambda: (11, 10)))
        served = dict(SERVED_KWARGS, q_chunk_size=qk_chunk, k_chunk_size=qk_chunk)
        return served, self.ns['_qwen_pf_program_config'](layer, served, qk_chunk, flexible, S)

    def test_environment_off_is_the_served_config(self):
        word, calls = self.word({})
        self.assertIsNone(word)
        self.assertEqual(calls, [])                                     # no binary probe either
        self.assertEqual(self.pindiag, [])
        served, sent = self.config(None)
        self.assertIs(sent, served)                                     # the served object itself
        self.assertEqual(self.built, [])                                # and no second config built
        for value in ('0', '', 'true', '11'):
            with self.subTest(value=value):
                self.assertIsNone(self.word({optin.ENV: value})[0])

    def test_on_defaults_to_0x3_on_the_flexible_bf8_path_only(self):
        word, _ = self.word({optin.ENV: '1'})
        self.assertEqual(word, 0x5EFA0003)                              # Q2's choice: chain + read-ahead 32
        self.assertEqual(self.pindiag, ['[PINDIAG] sdpa prefill kvchain flags=0x5efa0003 rows=512/1024/2048 chunk=128'])
        served, sent = self.config(word)
        self.assertEqual(sent, dict(served, max_cores_per_head_batch=0x5EFA0003))
        for label, kwargs in (('legacy int chunk_start', dict(flexible=False)), ('odd q_num_chunks', dict(S=1920)),
                              ('bf16 mode', dict(bf8=False)), ('q/k chunk 64', dict(qk_chunk=64))):
            with self.subTest(label):
                served, sent = self.config(word, **kwargs)
                self.assertIs(sent, served)
        for text, expected in (('0x7', 0x5EFA0007), ('0x1', 0x5EFA0001), ('5', 0x5EFA0005), ('', 0x5EFA0003),
                               (' 0x3 ', 0x5EFA0003)):
            with self.subTest(flags=text):
                self.assertEqual(self.word({optin.ENV: '1', optin.FLAGS_ENV: text})[0], expected)

    def test_only_the_card_m_qualified_row_counts_opt_in(self):
        """Review 1 finding 4: S % 256 == 0 admitted 256-1792-row tail chunks and S > 2048 (unequal
        groups) that the Q1 sweep never ran. The opt-in sends exactly the swept row counts."""
        word = 0x5EFA0003
        for S in (512, 1024, 2048):
            self.assertEqual(self.config(word, S=S)[1]['max_cores_per_head_batch'], word)
        for S in (256, 768, 1280, 1536, 1792, 2304, 4096):
            with self.subTest(S=S):
                served, sent = self.config(word, S=S)
                self.assertIs(sent, served)
        self.assertEqual(tuple(sorted(card.ROWS)), factory.QUALIFIED_ROWS)
        self.assertEqual(self.ns['_QWEN_PF_ROWS'], factory.QUALIFIED_ROWS)
        self.assertEqual(optin.QUALIFIED_ROWS, factory.QUALIFIED_ROWS)
        self.assertEqual(self.ns['_QWEN_PF_FLAGS'], tuple(card.PRODUCTION_FLAGS))
        for rows in factory.QUALIFIED_ROWS:                        # every qualified S is a G6 shape
            topo = model.g6_topology(model.geometry(rows=rows))
            self.assertEqual(topo['members'], 6 * topo['chains'])
            self.assertEqual((topo['chains'], topo['members']), card.expected_chains(rows))

    def test_a_stock_binary_and_bad_flags_are_refused_first(self):
        with self.assertRaisesRegex(RuntimeError, 'lacks the \\[QWEN-SDPA-PF\\] chain factory'):
            self.word({optin.ENV: '1'}, has_marker=False)
        for flags in ('0x0', '0x2', '0x4', '0x6', '0x8', '0xf', '0x103', '0x203', '0x1003', 'junk'):
            with self.subTest(flags=flags):
                calls = []
                with self.assertRaisesRegex(RuntimeError, 'must be a production flag set \\(0x1, 0x3, 0x5, 0x7\\)'):
                    self.ns['_qwen_pf_word']({optin.ENV: '1', optin.FLAGS_ENV: flags}, lambda: calls.append(1) or True)
                self.assertEqual(calls, [])                             # refused before the binary probe
        self.assertEqual(self.pindiag, [])

    def test_the_process_environment_is_probed_once_per_process(self):
        """Every attention layer calls _qwen_pf_word() at construction: one binary probe, one marker."""
        environ = {optin.ENV: '1'}
        ns = optin.helpers_namespace(os_module=SimpleNamespace(environ=environ))
        probes, logged = [], []
        ns['_qwen_pf_binary_has_marker'] = lambda: probes.append(1) or True
        ns['_qwen_pf_log'] = logged.append
        self.assertEqual([ns['_qwen_pf_word']() for _ in range(16)], [0x5EFA0003] * 16)
        self.assertEqual((len(probes), len(logged)), (1, 1))
        off = optin.helpers_namespace(os_module=SimpleNamespace(environ={}))
        off['_qwen_pf_binary_has_marker'] = lambda: probes.append(1) or True
        self.assertIsNone(off['_qwen_pf_word']())
        self.assertEqual(off['_QWEN_PF_STATE'], {})
        self.assertEqual(len(probes), 1)

    def test_the_maps_probe_reads_the_one_mapped_ttnncpp(self):
        with tempfile.TemporaryDirectory() as directory:
            so = Path(directory) / '_ttnncpp.so'
            so.write_bytes(b'\0junk' + optin.BINARY_MARKER + b'\0')
            maps = Path(directory) / 'maps'
            maps.write_text('7f00 r-xp 0 0 0 %s\n7f01 r--p 0 0 0 %s\n' % (so.as_posix(), so.as_posix()))
            self.assertTrue(self.ns['_qwen_pf_binary_has_marker'](str(maps)))
            so.write_bytes(b'\0stock\0')
            self.assertFalse(self.ns['_qwen_pf_binary_has_marker'](str(maps)))
            maps.write_text('')
            with self.assertRaisesRegex(RuntimeError, 'exactly one mapped'):
                self.ns['_qwen_pf_binary_has_marker'](str(maps))

    def test_the_tp_patch_touches_only_its_two_sites_and_inverts(self):
        patched = optin.patch_tp(TP_EXCERPT)
        self.assertEqual(optin.unpatch_tp(patched), TP_EXCERPT)
        compile(patched, 'tp.py', 'exec')
        decode = patched[patched.index('def forward_decode'):patched.index('def forward_prefill_paged')]
        self.assertIn('q_chunk_size=0', decode)                         # the decode config untouched
        self.assertNotIn('_qwen_pf', decode)
        prefill = patched[patched.index('def forward_prefill_paged'):patched.index(optin.TP_HELPERS)]
        self.assertIn(optin.graft.SDPA_PF_CALL_OLD, prefill)            # the served statement, verbatim
        self.assertIn('_qwen_pf_program_config', prefill)
        self.assertIn(optin.INIT_NEW, patched)
        with self.assertRaisesRegex(ValueError, 'already grafted'):
            optin.patch_tp(patched)
        with self.assertRaisesRegex(ValueError, 'expected one occurrence'):
            optin.patch_tp(TP_EXCERPT.replace('k_chunk_size=qk_chunk,', 'k_chunk_size=64,'))
        with self.assertRaisesRegex(ValueError, 'expected one occurrence'):
            optin.patch_tp(TP_EXCERPT.replace('QWEN_SDPA_BF8', 'QWEN_SDPA_BF16'))
        with self.assertRaises(SyntaxError):                       # an unparsable input is refused
            optin.patch_tp(TP_EXCERPT.replace('        return sdpa_cfg', '        return sdpa_cfg)'))

    def test_the_cli_reports_whether_its_input_is_the_pinned_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'tp.py'
            source.write_text(TP_EXCERPT, encoding='utf-8', newline=NL)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(optin.main([str(source), '--out', str(Path(directory) / 'out.py')]), 0)
            self.assertIn('NOT the pinned image file %s' % optin.TP_PINNED_PREFIX, buffer.getvalue())
            self.assertEqual(optin.unpatch_tp((Path(directory) / 'out.py').read_text(encoding='utf-8')), TP_EXCERPT)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(optin.main([str(FIXTURE_TP), '--full', '--out', str(Path(directory) / 'full.py')]), 0)
            self.assertIn('(the pinned image file) -> the full graft', buffer.getvalue())
            self.assertEqual((Path(directory) / 'full.py').read_text(encoding='utf-8'),
                             optin.graft.PATCHES['attention/tp.py'](FIXTURE_TP.read_text(encoding='utf-8')))

    def test_the_patched_call_site_builds_the_served_config_when_off(self):
        patched = optin.patch_tp(TP_EXCERPT)
        seen = []
        namespace = dict(ttnn=SimpleNamespace(SDPAProgramConfig=lambda **kw: seen.append(kw) or kw))
        exec(compile(patched, 'tp.py', 'exec'), namespace)
        saved = os.environ.pop(optin.ENV, None)
        try:
            layer = namespace['TtAttention'](None)
        finally:
            if saved is not None:
                os.environ[optin.ENV] = saved
        self.assertIsNone(layer._sdpa_pf_word)
        layer.mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: (11, 10))
        layer._sdpa_bf8 = True
        out = layer.forward_prefill_paged(SimpleNamespace(shape=(1, 1, 2048, 5120)), None, chunk_start_idx_tensor=object())
        self.assertEqual(seen, [SERVED_KWARGS])                         # one config, the served one
        self.assertIs(out, seen[0])
        layer._sdpa_pf_word = 0x5EFA0003
        layer.forward_prefill_paged(SimpleNamespace(shape=(1, 1, 2048, 5120)), None, chunk_start_idx_tensor=object())
        self.assertEqual(seen[1:], [SERVED_KWARGS, dict(SERVED_KWARGS, max_cores_per_head_batch=0x5EFA0003)])
        out = layer.forward_prefill_paged(SimpleNamespace(shape=(1, 1, 2048, 5120)), None, chunk_start_idx=2048)
        self.assertNotIn('max_cores_per_head_batch', out)

    def test_the_patch_applies_to_the_pinned_tp(self):
        """The committed copy of the image's attention/tp.py (every graft through v143 staged it)."""
        text = FIXTURE_TP.read_text(encoding='utf-8')
        self.assertTrue(sha(text.encode('utf-8')).startswith(optin.TP_PINNED_PREFIX))
        patched = optin.patch_tp(text)
        self.assertEqual(optin.unpatch_tp(patched), text)
        compile(patched, 'attention/tp.py', 'exec')
        full = optin.patch_tp_full(text)
        self.assertEqual(optin.unpatch_tp(full), optin.graft.patch_attention_tp(text))
        if DUMP.is_file():
            self.assertEqual(dumped_tp().rstrip(NL), text.rstrip(NL))   # the probe dump is the same file


# ---------------------------------------------------------------------------------------------
# 9. build script and runner
# ---------------------------------------------------------------------------------------------

def script_constants(text):
    return dict(re.findall(r'^([A-Z0-9_]+)=([0-9a-f]{64})$', text, re.M))


class BuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.text = (HERE / 'build_k64g.sh').read_text(encoding='utf-8')
        self.k64f = (DECODE / 'build_k64f.sh').read_text(encoding='utf-8')
        self.shas = script_constants(self.text)

    def test_every_sha_equals_the_python_and_build_k64f(self):
        import apply_factory_qwen
        import make_qwen_kernels
        self.assertEqual(self.shas['PF_BASE'], factory.BASE_FACTORY)
        self.assertEqual(self.shas['PF_FACTORY'], factory.PF_FACTORY)
        self.assertEqual(self.shas['PF_READER_BASE'], gen.BASE_SHA)
        self.assertEqual(self.shas['PF_READER'], gen.READER_OUTPUT)
        self.assertEqual(self.shas['FACTORY_BASE'], apply_factory_qwen.BASE_FACTORY)
        self.assertEqual(self.shas['FACTORY_QWEN_STAGE1'], apply_factory_qwen.QWEN_FACTORY)
        self.assertEqual(self.shas['FACTORY_QWEN_STAGE3'], apply_factory_qwen.QWEN_FACTORY_STAGE3)
        self.assertEqual(self.shas['READER_QWEN_STAGE3'], make_qwen_kernels.OUTPUTS_STAGE3[make_qwen_kernels.READER_NAME])
        self.assertEqual(self.shas['READER_QWEN_STAGE1'], make_qwen_kernels.OUTPUTS[make_qwen_kernels.READER_NAME])
        self.assertEqual(self.shas['COMPUTE_QWEN'], make_qwen_kernels.OUTPUTS_STAGE3[make_qwen_kernels.COMPUTE_NAME])
        k64f = script_constants(self.k64f)
        for name in ('FACTORY_BASE', 'FACTORY_UNPATCHED', 'READER_ALL', 'COMPUTE_ALL', 'WRITER_ALL', 'DATAFLOW_COMMON',
                     'RT_ARGS_COMMON', 'READER_QWEN_STAGE1', 'COMPUTE_QWEN'):
            self.assertEqual(self.shas[name], k64f[name], name)
        self.assertEqual(self.shas['FACTORY_QWEN_STAGE3'], k64f['FACTORY_QWEN'])
        self.assertEqual(self.shas['READER_QWEN_STAGE3'], k64f['READER_QWEN'])
        self.assertEqual(self.shas['PF_BASE'], k64f['PREFILL_COMBINED'])
        served = dict(line.split('  ')[::-1] for line in re.findall(r'"([0-9a-f]{64}  \$\S+)"',
                                                                     (BENCH / 'run_m1.sh').read_text(encoding='utf-8')))
        for name, path in (('PF_COMPUTE', 'compute/sdpa.cpp'), ('PF_COMPUTE_COMMON', 'compute/compute_common.hpp'),
                           ('PF_DATAFLOW_COMMON', 'dataflow/dataflow_common.hpp'), ('PF_CHAIN_LINK', 'dataflow/chain_link.hpp'),
                           ('PF_WRITER', 'dataflow/writer_interleaved.cpp')):
            self.assertEqual(self.shas[name], served['$SDPA/kernels/' + path], name)

    def test_the_string_checks_are_a_superset_of_build_k64f(self):
        for marker in ('[QWEN-SDPA] flags=', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS', 'reader_decode_qwen.cpp',
                       'qwen_draft_fp32_intermediates', '[QWEN-SDPA] KV-share twin bands',
                       '[QWEN-SDPA] KV share is not in this build'):
            self.assertIn(marker, self.k64f)
            self.assertIn(marker, self.text)
        for marker in factory.MARKERS[:2] + (factory.READER_NAME,):
            self.assertIn(marker, self.text)
        self.assertIn("grep -E 'QWEN_|\\[QWEN-'", self.text)
        self.assertIn('comm -23 "$W/base-qwen-strings.txt" "$W/graft-qwen-strings.txt"', self.text)

    def test_the_steps_run_in_order_and_restore_both_factories(self):
        steps = [self.text.index('# ---------- %d.' % step) for step in range(8)]
        self.assertEqual(steps, sorted(steps))
        restore = self.text[self.text.index('restore_ttbuild() {'):self.text.index('on_exit() {')]
        for token in ('docker cp "$DS/sdpa_decode_program_factory.cpp.$FACTORY_BASE" ttbuild:$F',
                      'docker cp "$S/sdpa_program_factory.cpp.$PF_BASE" ttbuild:$PF',
                      'reader_interleaved_qwen_chain.cpp', 'docker exec ttbuild touch "$F" "$PF"'):
            self.assertIn(token, restore)
        self.assertIn('trap on_exit EXIT', self.text)
        self.assertLess(self.text.index('armed=1'), self.text.index('docker cp "$W/sdpa_program_factory.cpp" ttbuild:$PF'))
        self.assertIn('--check "$S"', self.text)
        self.assertIn('cp "$S/sdpa_program_factory.cpp.$PF_BASE" "$G/sdpa/device/sdpa_program_factory.cpp"', self.text)
        self.assertIn('for image in $IMAGES; do', self.text)
        images = re.search(r'IMAGES=\$\{K64G_IMAGES:-([^}]*)\}', self.text).group(1).split()
        run_card = (HERE / 'run_card_m_pf.sh').read_text(encoding='utf-8')
        card_image = re.search(r'IMAGE=\$\{IMAGE:-(sha256:[0-9a-f]{64})\}', run_card).group(1)
        self.assertIn(card_image, images)                       # the card-M image is always compared
        self.assertEqual(len(images), 3)
        self.assertIn("printf 'Only in %s: %s' \"$G/sdpa/device/kernels/dataflow\" reader_interleaved_qwen_chain.cpp", self.text)
        # Review 2 L5: every image is checked present at step 0, before ninja, and the gate image stays.
        step0, ninja = self.text.index('# ---------- 0.'), self.text.index('ninja -C build_Release')
        presence = self.text.index('docker image inspect "$image" >/dev/null 2>&1')
        self.assertLess(step0, presence)
        self.assertLess(presence, ninja)
        self.assertEqual(re.search(r'^GATE_IMAGE=(\S+)', self.text, re.M).group(1), card_image)
        self.assertIn('*" $GATE_IMAGE "*) ;;', self.text)
        # Review 2 L4: each image's own _ttnncpp.so QWEN strings must survive in the graft .so.
        self.assertIn('docker cp -L "$cid:/opt/tt-metal/build_Release/lib/_ttnncpp.so" "$W/image-ttnncpp.so"', self.text)
        self.assertIn('comm -23 "$W/image-qwen-strings.txt" "$W/graft-qwen-strings.txt"', self.text)
        # run_m1.sh refuses a KOPGRAFT_PF run in any image the build did not compare (review 2 M5).
        run_m1 = (BENCH / 'run_m1.sh').read_text(encoding='utf-8')
        self.assertEqual(re.search(r'^PF_GRAFT_IMAGES="([^"]*)"', run_m1, re.M).group(1).split(), images)

    def test_a_kept_staged_rerun_accepts_the_patched_prefill_factory(self):
        """Review 1 finding 6: step 2 required fd8c0676, so step 3b's 'already patched' branch was dead."""
        step2 = self.text[self.text.index('# ---------- 2.'):self.text.index('# ---------- 3.')]
        self.assertIn('if [ "$pf_now" = "$PF_FACTORY" ]; then', step2)
        self.assertIn('need "ttbuild prefill factory" "$pf_now" $PF_BASE', step2)
        self.assertLess(step2.index('"$PF_FACTORY"'), step2.index('need "ttbuild prefill factory"'))
        step3 = self.text[self.text.index('# 3b. the prefill factory'):self.text.index('# 3c. stage')]
        self.assertIn('elif [ "$pcur" = "$PF_FACTORY" ]; then', step3)
        self.assertIn('K64G_TTNNCPP_SHA256=', self.text)
        for op in ('attn_prep', 'nlp_concat_heads_decode', 'sdpa_decode'):
            self.assertIn(op, self.text[self.text.index('for op in attn_prep nlp_concat_heads_decode sdpa_decode'):][:80])

    def test_shell_files_are_lf_and_parse(self):
        for name in ('build_k64g.sh', 'run_card_m_pf.sh'):
            data = (HERE / name).read_bytes()
            self.assertNotIn(b'\r', data)
            if BASH:
                result = subprocess.run([BASH, '-n', (HERE / name).as_posix()], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        for path in HERE.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.sh', '.cpp', '.hpp', '.h', '.md', '.json'):
                self.assertNotIn(b'\r', path.read_bytes(), str(path))


def find_bash():
    candidates = []
    if os.name == 'nt':
        for root in (os.environ.get('ProgramW6432'), os.environ.get('ProgramFiles'), 'C:/Program Files'):
            if root:
                candidates.append(Path(root) / 'Git' / 'bin' / 'bash.exe')
    found = shutil.which('bash')
    if found and not (os.name == 'nt' and ('system32' in found.lower() or 'windowsapps' in found.lower())):
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


BASH = find_bash()
SCRUB = ('KOPGRAFT_PF', 'IMAGE', 'RESULTS', 'PF_SRC', 'WATCHER', 'WATCHDOG_S', 'CARD_M_ARGS', 'PF_DRY_RUN', 'REFERENCE',
         'PF_PYTHON')
IMAGE_A2 = 'sha256:eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9'


@unittest.skipUnless(BASH, 'bash not found')
class CardMRunnerTests(unittest.TestCase):
    def run_script(self, role, graft=True, **env):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            g = root / 'graft'
            if graft:
                (g / 'sdpa/device/kernels/dataflow').mkdir(parents=True)
                (g / 'sdpa_decode').mkdir()
                (g / '_ttnn.so').write_bytes(b'so')
                (g / '_ttnncpp.so').write_bytes(b'cpp')
                (g / 'sdpa/device/kernels/dataflow/reader_interleaved_qwen_chain.cpp').write_bytes(b'r')
            full = dict(os.environ)
            for name in SCRUB:
                full.pop(name, None)
            full.update(PF_DRY_RUN='1', PF_SRC=HERE.as_posix(), RESULTS=(root / 'r').as_posix(), KOPGRAFT_PF=g.as_posix())
            full.update(env)
            return subprocess.run([BASH, (HERE / 'run_card_m_pf.sh').as_posix(), role], env=full, capture_output=True,
                                  text=True, encoding='utf-8', errors='replace', timeout=60)

    def argv(self, result):
        return [line for line in result.stdout.splitlines() if line.startswith('### argv:')][0]

    def test_reference_mounts_no_graft(self):
        result = self.run_script('reference')
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv(result)
        self.assertNotIn('_ttnncpp.so\\,readonly', argv)
        self.assertNotIn('QWEN_SDPA_PF_TEST', argv)
        self.assertIn('--role reference', argv)
        self.assertIn('eceb2daa744c3345368a638488a804f0ffe8b76f6b0680945a47bb527bdb55a9', argv)
        self.assertIn('--device /dev/tenstorrent/by-id/blackhole-CEF5729692C19E6D', argv)

    def test_candidate_mounts_the_graft_like_the_arm_and_sets_the_test_env(self):
        argv = self.argv(self.run_script('candidate'))
        for dst in ('/opt/tt-metal/ttnn/ttnn/_ttnn.so', '/opt/tt-metal/build_Release/ttnn/_ttnncpp.so',
                    '/opt/tt-metal/build_Release/lib/_ttnncpp.so',
                    '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa\\,readonly',
                    '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode\\,readonly'):
            self.assertIn('dst=' + dst, argv)
        self.assertIn('-e QWEN_SDPA_PF_TEST=1', argv)
        self.assertNotIn('attn_prep', argv)                  # only the op directories the graft has

    def test_the_watcher_pass_is_narrow_and_capped(self):
        result = self.run_script('candidate', WATCHER='1')
        argv = self.argv(result)
        self.assertIn('-e TT_METAL_WATCHER=5', argv)
        self.assertIn('--flags 0x1\\,0x3\\,0x5\\,0x7', argv)
        self.assertIn('--starts 0\\,2048', argv)
        self.assertIn('--alternations 0', argv)
        self.assertIn('container cap 900 s', result.stdout)

    def test_a_missing_graft_is_refused(self):
        result = self.run_script('candidate', graft=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('build it with build_k64g.sh', result.stderr)

    def test_the_hang_role_verdicts(self):
        text = (HERE / 'run_card_m_pf.sh').read_text(encoding='utf-8')
        hang = text[text.rindex('if [ "$role" = hang ]; then'):]
        self.assertIn("grep -qF 'planted hang RETURNED'", hang)
        self.assertIn("grep -qF '%s'" % card.HANG_ARMED, hang)          # never armed: inconclusive, not a pass
        self.assertIn('''grep -qE "WATCHDOG: '(%s)'"''' % '|'.join(card.HANG_LABELS), hang)
        self.assertIn('[ "$status" = 1 ] && grep -qF \'Timeout (\'', hang)   # the faulthandler backstop
        self.assertIn('[ "$status" = 124 ] || [ "$status" = 137 ]', hang)
        self.assertIn('QWD[CV]', hang)
        self.assertIn('grep -qF reader_interleaved_qwen_chain "$wlog"', hang)   # an assert must be the chain's
        self.assertIn('card_m_reset_hint', hang)
        self.assertLess(hang.index("grep -qF '%s'" % card.HANG_ARMED), hang.index('QWD[CV]'))

    def test_holders_and_the_reset_hint(self):
        text = (HERE / 'run_card_m_pf.sh').read_text(encoding='utf-8')
        for token in ('privileged={{.HostConfig.Privileged}}', '"Source":"/dev"', 'fuser -v "$node"', 'sudo -n true',
                      'refuse_device_holders "$node"'):
            self.assertIn(token, text)
        hint = text[text.index('card_m_reset_hint() {'):text.index('refuse_device_holders() {')]
        self.assertIn('/sys/dev/char/', hint)
        self.assertIn('not the', hint)
        self.assertIn('tt-smi -ls', hint)
        tail = text[text.rindex('hung=0'):]
        self.assertIn("1) grep -qF 'Timeout (' \"$log\" && hung=1 ;;", tail)

    def write_reference(self, root, name, mtime, **fields):
        path = root / 'r' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(dict(passed=True, image=IMAGE_A2, narrow=False), **fields)))
        os.utime(path, (mtime, mtime))
        return path

    def test_the_candidate_picks_the_newest_passing_same_image_reference(self):
        """Review 2 M6: not simply the newest file."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_reference(root, 'reference-1.json', 1000)                            # the one to use
            self.write_reference(root, 'reference-2.json', 2000, narrow=True)                # a watcher reference
            self.write_reference(root, 'reference-3.json', 3000, passed=False)               # a failed run
            self.write_reference(root, 'reference-4.json', 4000, image='sha256:' + 'f' * 64)  # another image
            env = dict(RESULTS=(root / 'r').as_posix(), PF_PYTHON=sys.executable)
            full = self.run_script('candidate', **env)
            self.assertEqual(full.returncode, 0, full.stderr)
            self.assertIn('--reference /results/reference-1.json', self.argv(full))
            watcher = self.run_script('candidate', WATCHER='1', **env)
            self.assertIn('--reference /results/reference-2.json', self.argv(watcher))
            self.assertIn('--image %s' % IMAGE_A2, self.argv(full))
            (root / 'r' / 'reference-1.json').unlink()
            none = self.run_script('candidate', **env)
            self.assertNotIn('--reference', self.argv(none))
            self.assertIn('no passing reference report', none.stdout)
        text = (HERE / 'run_card_m_pf.sh').read_text(encoding='utf-8')
        self.assertIn("refusing: no passing", text)


# ---------------------------------------------------------------------------------------------
# The card-M test's host helpers and a fake-ttnn dry run of its three roles.
# ---------------------------------------------------------------------------------------------

class CardHelperTests(unittest.TestCase):
    def test_matrix_and_labels(self):
        args = card.parse_args(['--role', 'reference', '--out', 'x.json'])
        self.assertEqual(len(card.groups(args)), 3 * 2 * 2 * 2 * 3 * 3)
        labels = {card.case_label(card.group_label(*group), start) for group in card.groups(args) for start in args.starts}
        self.assertEqual(len(labels), 216 * 8)
        self.assertEqual(args.flags, [0x1, 0x3, 0x5, 0x7])
        self.assertFalse(card.narrow(args))
        self.assertTrue(card.narrow(card.parse_args(['--role', 'reference', '--out', 'x.json', '--rows', '2048'])))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            card.parse_args(['--role', 'candidate', '--out', 'x.json', '--flags', '0x103'])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            card.parse_args(['--role', 'reference', '--out', 'x.json', '--rows', '1920'])
        with self.assertRaises(ValueError):
            card.validate_starts([100], 2048)

    @unittest.skipIf(torch is None, 'torch not installed')
    def test_unit_digests_and_the_mutation_verdict(self):
        base = torch.randn(1, 12, 2048, 256).to(torch.bfloat16)
        served = card.unit_digests(torch, base)
        self.assertEqual(len(served), 192)
        everything = base.clone()
        everything[:, :, ::128] += 1
        self.assertEqual(card.mutation_verdict(served, card.unit_digests(torch, everything))[:2], (True, 192))
        injectors = base.clone()
        for core in [c for g in range(2) for c in range(48 * g, 48 * g + 8)]:     # s = 0: the raster injectors
            head, m = core // 8, core % 8
            for q in (m, 15 - m):
                injectors[0, head, q * 128] += 1
        passed, differing, message = card.mutation_verdict(served, card.unit_digests(torch, injectors))
        self.assertEqual((passed, differing), (False, 32))
        self.assertIn('receivers read DRAM', message)

    def test_factory_line_checks(self):
        lines = card.factory_lines('\n'.join('[QWEN-SDPA-PF] flags=%#x kv_chain=1 chains=16 members=96 order=%s'
                                             % (flags, 'noc' if flags & 4 else 'raster') for flags in (1, 3, 7)))
        requested = {(2048, 2016, 'dram', 1), (2048, 2016, 'dram', 3), (2048, 2016, 'dram', 7)}
        self.assertEqual(card.check_factory_lines(lines, requested), [])
        self.assertTrue(card.check_factory_lines(lines + lines[:1], requested))           # a rebuilt program
        self.assertTrue(card.check_factory_lines(lines[1:], requested))                   # a program not built
        bad = [dict(lines[2], order='raster')]
        self.assertTrue(card.check_factory_lines(bad, {(2048, 2016, 'dram', 7)}))
        # Review 1 finding 1: a DRAM Q and an L1 Q are two programs, so two lines per flag set are right.
        both = {(2048, 2016, qmem, 1) for qmem in ('dram', 'l1')}
        self.assertEqual(card.check_factory_lines(lines[:1] * 2, both), [])
        self.assertTrue(card.check_factory_lines(lines[:1], both))

    def test_the_watchdog_fires_once_and_exits_3(self):
        clock = [0.0]
        fired, exits = [], []
        dog = card.Watchdog(5, on_fire=fired.append, stream=io.StringIO(), exit=exits.append, clock=lambda: clock[0],
                            backstop=False)
        with dog.op('call'):
            self.assertFalse(dog.check())
            clock[0] = 6.0
            self.assertTrue(dog.check())
        self.assertEqual((fired, exits), (['call'], [3]))
        with dog.op('compile', extra=100):
            clock[0] = 50.0
            self.assertFalse(dog.check())

    def test_every_op_arms_the_faulthandler_backstop(self):
        """Review 0 finding 2 / review 2 M2: the poll thread needs the GIL, so each op also arms the
        C-thread backstop (budget + grace), re-armed for the outer op on a nested exit, cancelled last."""
        clock = [0.0]
        dog = card.Watchdog(5, stream=io.StringIO(), exit=lambda code: None, clock=lambda: clock[0], grace=60.0)
        dog.arm_backstop = lambda seconds: dog.armed.append(seconds + dog.grace)   # record, arm no real timer
        dog.cancel_backstop = lambda: dog.armed.append(None)
        with dog.op('outer', extra=100):
            clock[0] = 10.0
            with dog.op('inner'):
                clock[0] = 12.0
        self.assertEqual(dog.armed, [165.0, 65.0, 60.0 + 93.0, None])
        source = card.Watchdog.arm_backstop.__code__.co_names
        self.assertIn('dump_traceback_later', source)
        off = card.Watchdog(0)
        with off.op('x'):
            pass
        self.assertEqual(off.armed, [])


class FakeTensor:
    def __init__(self, host, dtype=None, memory=None):
        self.host, self.dtype, self.memory = host, dtype, memory
        self.shape = tuple(host.shape)


class FakeDevice:
    def __init__(self):
        self.programs = set()

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=11, y=10)

    def num_program_cache_entries(self):
        return len(self.programs)


def fake_ttnn(device, broken=None):
    """Enough ttnn for the card-M flow. The op is exact on the chain by construction (the word only
    names a program); 0x100 changes every unit; refusals follow decode_word and the F4 envelope; a
    new chain program writes the factory line to fd 1 (as tt-logger does). As on the device, the
    factory (decode_word, the envelope, the log line) runs ONLY on a program-cache miss, and the
    program key includes the Q memory and the whole word. broken='receivers' changes only the 32
    injector units under 0x100 (the chain-not-active failure M-A must catch)."""
    traces = {}

    def output(q, k, start, program_word):
        host = q.host.float() * (start + 1) + float(k.host.float().mean())
        if program_word is not None and program_word & 0x100:
            if broken == 'receivers':
                host = host.clone()
                for core in [c for g in range(2) for c in range(48 * g, 48 * g + 8)]:
                    head, m = core // 8, core % 8
                    for chunk in (m, 15 - m):
                        host[0, head, chunk * 128] += 1
            else:
                host = host + 1
        return host.to(torch.bfloat16)

    def chunked(input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, compute_kernel_config,
                program_config, chunk_start_idx_tensor=None, chunk_start_idx=None):
        word = program_config.get('max_cores_per_head_batch')
        fp32 = compute_kernel_config['fp32_dest_acc_en']
        rows, width = input_tensor_q.shape[2], page_table_tensor.shape[1]
        key = (rows, width, input_tensor_q.memory, word, chunk_start_idx_tensor is None, fp32, input_tensor_k.dtype)
        program_word = word
        flags = 0
        if word is not None and (word & 0xFFFF0000) == factory.PF_TAG:
            flags = word & 0xFFFF
        else:
            program_word = None
        if key not in device.programs:              # the factory: only on a program-cache miss
            if word is not None:
                try:
                    chain, flags = factory.decode_word(word, os.environ.get(factory.TEST_ENV))
                except ValueError as error:
                    raise RuntimeError('TT_FATAL %s' % error)
                if chain and (chunk_start_idx_tensor is None or not fp32 or input_tensor_k.dtype != 'bf8'):
                    raise RuntimeError('TT_FATAL %s' % factory.ENVELOPE_MARKER)
            device.programs.add(key)
            if program_word is not None:
                chains, members = card.expected_chains(rows)
                os.write(1, ('[QWEN-SDPA-PF] flags=%#x kv_chain=1 chains=%d members=%d order=%s\n'
                             % (flags, chains, members, 'noc' if flags & 4 else 'raster')).encode())
        start_tensor = chunk_start_idx_tensor
        start = int(start_tensor.host[0]) if start_tensor is not None else chunk_start_idx
        result = FakeTensor(output(input_tensor_q, input_tensor_k, start, program_word))
        if traces.get('capturing') is not None:
            traces['capturing'].append((result, input_tensor_q, input_tensor_k, start_tensor, program_word))
        return result

    def plain(q, k, v, is_causal, program_config, compute_kernel_config):
        if program_config.get('max_cores_per_head_batch') is not None:
            raise RuntimeError('TT_FATAL %s' % factory.ENVELOPE_MARKER)
        return FakeTensor(q.host.clone())

    def begin(dev, cq_id):
        traces['capturing'] = []
        return 'trace'

    def end(dev, trace, cq_id):
        traces['ops'] = traces.pop('capturing')

    def execute(dev, trace, cq_id, blocking):
        for result, q, k, start_tensor, program_word in traces['ops']:
            result.host = output(q, k, int(start_tensor.host[0]), program_word)

    def copy(host, target):
        target.host = host.host.clone()

    return SimpleNamespace(
        bfloat8_b='bf8', bfloat16='bf16', int32='int32', TILE_LAYOUT='tile', ROW_MAJOR_LAYOUT='rm',
        DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1', MathFidelity=SimpleNamespace(HiFi2='hifi2'),
        open_device=lambda **kw: device, close_device=lambda dev: None, deallocate=lambda tensor: None,
        from_torch=lambda host, dtype=None, layout=None, device=None, memory_config=None: FakeTensor(host.clone(), dtype,
                                                                                                    memory_config),
        to_torch=lambda tensor: tensor.host, WormholeComputeKernelConfig=lambda **kw: kw,
        SDPAProgramConfig=lambda **kw: kw, begin_trace_capture=begin, end_trace_capture=end, execute_trace=execute,
        release_trace=lambda dev, trace: None, copy_host_to_device_tensor=copy,
        transformer=SimpleNamespace(chunked_scaled_dot_product_attention=chunked, scaled_dot_product_attention=plain))


@unittest.skipIf(torch is None, 'torch not installed')
class CardDryRunTests(unittest.TestCase):
    """test_sdpa_prefill_chain_card_m.main against the fake ttnn, a small pool and a narrow matrix."""

    SMALL = ['--rows', '512,2048', '--starts', '0,128,2048', '--seeds', '0', '--variants', 'normal,zeroq',
             '--tables', 'perm', '--widths', 'fit,model', '--q-memory', 'dram,l1', '--watchdog', '0', '--no-maps']

    def setUp(self):
        self.saved = (card.POOL_BLOCKS, os.environ.get(factory.TEST_ENV), sys.modules.get('ttnn'))
        card.POOL_BLOCKS = 96
        os.environ.pop(factory.TEST_ENV, None)

    def tearDown(self):
        card.POOL_BLOCKS = self.saved[0]
        if self.saved[1] is None:
            os.environ.pop(factory.TEST_ENV, None)
        else:
            os.environ[factory.TEST_ENV] = self.saved[1]
        if self.saved[2] is None:
            sys.modules.pop('ttnn', None)
        else:
            sys.modules['ttnn'] = self.saved[2]

    def main(self, directory, role, extra=(), broken=None, test_env=True):
        device = FakeDevice()
        sys.modules['ttnn'] = fake_ttnn(device, broken)
        if test_env:
            os.environ[factory.TEST_ENV] = '1'
        out = Path(directory) / ('%s.json' % role)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            status = card.main(['--role', role, '--out', str(out)] + self.SMALL + list(extra))
        return status, json.loads(out.read_text()), buffer.getvalue()

    def test_reference_then_candidate_passes_every_check(self):
        with tempfile.TemporaryDirectory() as directory:
            status, reference, _ = self.main(directory, 'reference', test_env=False)
            self.assertEqual(status, 0, reference)
            self.assertEqual(len(reference['reference_sha256']), 2 * 2 * 2 * 2 * 3)
            status, report, stdout = self.main(directory, 'candidate', ['--reference', str(Path(directory) / 'reference.json'),
                                                                        '--alternations', '12', '--trace-replays', '8'])
            self.assertEqual(status, 0, report['failures'] or report.get('error'))
            self.assertTrue(all(case['served_matches_reference'] for case in report['cases']))
            self.assertTrue(all(all(case['flags'].values()) for case in report['cases']))
            self.assertEqual(report['mutation']['differing_units'], 192)
            self.assertEqual(report['cache']['after_chain'], report['cache']['after_served'] + 1)
            self.assertTrue(all(entry['refused'] and entry['matched'] for entry in report['refusals'].values()))
            self.assertEqual(report['stress']['drifted'], 0)
            self.assertEqual(report['trace']['mismatches'], 0)
            self.assertEqual(sorted({line['flags'] for line in report['factory_lines']}), [0x1, 0x3, 0x5, 0x7, 0x103])
            # One program (and one factory line) per (rows, page width, Q memory, flags): 2 x 2 x 2 per flag set.
            for flags in (0x1, 0x3, 0x5, 0x7):
                self.assertEqual(sum(1 for line in report['factory_lines'] if line['flags'] == flags), 8)
            self.assertIn('rows=2048 width=64 q=l1 flags=0x5', report['requested_programs'])
            self.assertIn('SDPA_PF_CARD_M role=candidate passed=True', stdout)

    def bench(self, test_env=True):
        device = FakeDevice()
        sys.modules['ttnn'] = fake_ttnn(device)
        if test_env:
            os.environ[factory.TEST_ENV] = '1'
        args = card.parse_args(['--role', 'candidate', '--out', 'x.json'] + self.SMALL)
        return card.Bench(sys.modules['ttnn'], torch, device), args

    def test_the_refusals_run_before_the_mutation_builds_their_program(self):
        """Review 1 finding 2 / review 2 H1: the factory checks only on a program-cache miss, so after
        M-A has built 0x103 the test-flag refusal would hit the cache and return an output."""
        bench, args = self.bench()
        report = dict(failures=[])
        with contextlib.redirect_stdout(io.StringIO()):
            card.refusals(bench, args, report)
            card.mutation(bench, args, report)
        self.assertEqual(report['failures'], [])
        self.assertTrue(all(entry['refused'] and entry['matched'] for entry in report['refusals'].values()))
        # The old order: the refusal's program is already cached; it is reported as untestable, never as a pass.
        bench, args = self.bench()
        report = dict(failures=[])
        with contextlib.redirect_stdout(io.StringIO()):
            card.mutation(bench, args, report)
            card.refusals(bench, args, report)
        name = 'test flags without %s' % factory.TEST_ENV
        self.assertFalse(report['refusals'][name]['refused'])
        self.assertIn('untestable', report['refusals'][name]['message'])
        self.assertEqual([failure for failure in report['failures'] if 'untestable' in failure],
                         ['Q1.8: %s: %s' % (name, report['refusals'][name]['message'])])
        source = card.run.__code__.co_names
        run_text = Path(card.__file__).read_text(encoding='utf-8')
        body = run_text[run_text.index('def run(args, report):'):]
        self.assertLess(body.index('refusals(bench, args, report)'), body.index('mutation(bench, args, report)'))
        self.assertIn('refusals', source)

    def test_the_fake_factory_runs_only_on_a_cache_miss(self):
        bench, _ = self.bench()
        shape = (2048, 64, 'perm', 'dram', 0, 'normal', 2048)
        bench.call(*shape, card.word(0x103))
        os.environ.pop(factory.TEST_ENV)
        bench.call(*shape, card.word(0x103))                        # cached: no F1 check, as on the device
        with self.assertRaisesRegex(RuntimeError, 'test-only flags'):
            bench.call(*shape[:3], 'l1', *shape[4:], card.word(0x103))  # another program (L1 Q): F1 runs

    def test_a_reference_missing_swept_cases_fails_them(self):
        with tempfile.TemporaryDirectory() as directory:
            narrow = ['--rows', '2048']
            device = FakeDevice()
            sys.modules['ttnn'] = fake_ttnn(device)
            out = Path(directory) / 'reference.json'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(card.main(['--role', 'reference', '--out', str(out)] + self.SMALL + narrow), 0)
            status, report, _ = self.main(directory, 'candidate', ['--reference', str(out), '--no-controls',
                                                                   '--no-mutation', '--alternations', '0',
                                                                   '--trace-replays', '0'])
            self.assertEqual(status, 1)
            missing = [failure for failure in report['failures'] if 'not in the reference report' in failure]
            self.assertEqual(len(missing), 2 * 2 * 2 * 3)             # every r512 case (2 widths x 2 Q x 2 variants x 3 starts)
            self.assertTrue(all(failure.startswith('r512-') for failure in missing))

    def test_a_failed_or_foreign_reference_is_refused_before_the_device(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'ref.json'
            for payload, needle in ((dict(passed=False, image='img', reference_sha256={}), 'did not pass'),
                                    (dict(passed=True, image='other', reference_sha256={}), "ran in image 'other'")):
                reference.write_text(json.dumps(payload))
                opened = []
                sys.modules['ttnn'] = SimpleNamespace(open_device=lambda **kw: opened.append(kw))
                out = Path(directory) / 'cand.json'
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    status = card.main(['--role', 'candidate', '--out', str(out), '--reference', str(reference),
                                        '--image', 'img'] + self.SMALL)
                self.assertEqual((status, opened), (1, []))
                self.assertTrue(any(needle in failure for failure in json.loads(out.read_text())['failures']))
                self.assertIn('refused before opening the device', buffer.getvalue())

    def test_a_chain_that_is_not_active_fails_the_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _ = self.main(directory, 'candidate', ['--rows', '2048', '--alternations', '0',
                                                                   '--trace-replays', '0', '--no-controls'],
                                          broken='receivers')
            self.assertEqual(status, 1)
            self.assertEqual(report['mutation']['differing_units'], 32)
            self.assertTrue(any('receivers read DRAM' in failure for failure in report['failures']))

    def test_a_returning_planted_hang_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            status, report, _ = self.main(directory, 'hang')
            self.assertEqual(status, 1)
            self.assertTrue(report['hang']['returned'])
            self.assertTrue(report['hang']['armed'])
            self.assertTrue(report['hang']['precheck_served_sha256'])   # the served pre-check ran first
            self.assertTrue(any('RETURNED' in failure for failure in report['failures']))
        text = Path(card.__file__).read_text(encoding='utf-8')          # the line run_card_m_pf.sh greps
        body = text[text.index('def planted_hang('):text.index('def run(args, report):')]
        self.assertLess(body.index('served = bench.sha(*shape)'), body.index("print('%s: served pre-check"))
        self.assertLess(body.index("print('%s: served pre-check"), body.index('word(HANG_FLAGS)'))

    def fire_on(self, directory, predicate):
        """Run the hang role with a watchdog that fires on the first op whose label satisfies
        predicate: the report on_fire wrote (the real watchdog then _exits, so nothing overwrites it)."""
        out = Path(directory) / 'hang.json'
        snapshots = []

        class FiringDog:
            def __init__(self, seconds, on_fire=None, **_):
                self.on_fire = on_fire

            def start(self):
                return self

            @contextlib.contextmanager
            def op(self, label, extra=0.0):
                if predicate(label):
                    self.on_fire(label)
                    snapshots.append(json.loads(out.read_text()))
                    raise SystemExit(3)
                yield

        saved = card.Watchdog
        card.Watchdog = FiringDog
        try:
            device = FakeDevice()
            sys.modules['ttnn'] = fake_ttnn(device)
            os.environ[factory.TEST_ENV] = '1'
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                card.main(['--role', 'hang', '--out', str(out)] + self.SMALL)
        finally:
            card.Watchdog = saved
        return snapshots[0]

    def test_only_the_planted_call_or_its_read_back_passes_the_hang_role(self):
        """Review 2 M1: a watchdog on 'open device', an upload or the pre-check is not the planted hang."""
        with tempfile.TemporaryDirectory() as directory:
            for label in card.HANG_LABELS:
                report = self.fire_on(directory, lambda what, label=label: what == label)
                self.assertTrue(report['passed'], label)
                self.assertEqual(report['hang']['watchdog'], label)
            for early in ('open device', 'upload'):
                report = self.fire_on(directory, lambda what, early=early: what.startswith(early))
                self.assertFalse(report['passed'], early)
            report = self.fire_on(directory, lambda what: what.startswith('sdpa r2048'))   # the served pre-check
            self.assertFalse(report['passed'])


# ---------------------------------------------------------------------------------------------
# g++ syntax check against the stubs (local: the arm-none-eabi-g++ 13 toolchain; CI: g++ if present)
# ---------------------------------------------------------------------------------------------

GXX = stub_compile.find_gxx()


@unittest.skipUnless(GXX, 'no g++ (QWEN_PF_GXX)')
class StubCompileTests(unittest.TestCase):
    def test_the_factory_blocks_compile(self):
        code, output = stub_compile.check_factory(GXX)
        self.assertEqual(code, 0, output)

    @unittest.skipUnless((PROBE / 'kernels/dataflow/dataflow_common.hpp').is_file(), 'no probe-v25 tree')
    def test_a_single_q_subblock_shape_is_refused_at_compile_time(self):
        results = stub_compile.check_readers(GXX, PROBE, CHAIN, 0x1, {'qk_subblock_h': 4})   # q_num_subblocks == 1
        (_s, served_code, _so), (_c, chain_code, chain_out) = results
        self.assertEqual(served_code, 0)
        self.assertNotEqual(chain_code, 0)
        self.assertIn('the chain reader needs the Q subblock push', chain_out)

    @unittest.skipUnless((PROBE / 'kernels/dataflow/dataflow_common.hpp').is_file(), 'no probe-v25 tree')
    def test_the_chain_reader_compiles_with_no_warning_beyond_the_served_reader(self):
        for flags in (0x1, 0x5, 0x7, 0x303):
            with self.subTest(flags=hex(flags)):
                results = stub_compile.check_readers(GXX, PROBE, CHAIN, flags)
                ok, served, extra = stub_compile.reader_verdict(results)
                self.assertTrue(ok, (extra, results[1][2][:2000]))


if __name__ == '__main__':
    unittest.main()
