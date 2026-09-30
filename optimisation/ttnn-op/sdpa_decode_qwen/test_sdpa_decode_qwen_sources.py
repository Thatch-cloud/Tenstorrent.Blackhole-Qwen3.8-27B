"""CPU checks for the [QWEN-SDPA] graft sources, stage 1 (K64e) and stage 3 (K64f); no device, no ttnn.

  - the committed kernels are the recorded originals plus exactly their edits (reverting the
    edits reproduces the base sha), and hash to the recorded outputs: stage 1 in this directory
    (R1-R3 / C1-C2), stage 3 in stage3/ (plus R4-R5, compute unchanged), and reverting R5, R4
    from the stage-3 reader gives the stage-1 reader byte for byte;
  - apply_factory_qwen refuses anything but 3e0a69af and, given the base, reproduces the
    recorded F1-F8 (stage 1) and F1-F12 (stage 3) factories and inverts back to them;
  - every constant that crosses a file boundary agrees: the sentinel and flag values, the
    binary markers, the semaphore and runtime-arg slots of the K/V share, the factory's log
    format against the card-M parser and the gate markers, the TT_FATAL texts the card-M test
    expects per stage, the shas build_k64e.sh and build_k64f.sh enforce, the op directory the
    arm mounts, the run_card_m.sh environment and watcher mode;
  - the card-M test's host helpers (mask formula, mask/query builders, fold layout, program
    keys, log check, watchdog), and a full dry run of its device flow against a fake ttnn whose
    op has the hardware's share semantics (twins use the leader's page-table row).

The optional base inputs are found at QWEN_SDPA_SOURCES_DUMP (probe_sdpa_decode_sources.py
output) and QWEN_SDPA_FACTORY_BASE (a 3e0a69af factory); the tests that need them skip
without them.

    py -3.11 -B -m unittest test_sdpa_decode_qwen_sources      (from this directory)
"""

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
CI = HERE.parents[2] / 'scripts' / 'ci'
for path in (str(HERE), str(CI)):
    if path not in sys.path:
        sys.path.insert(0, path)

import apply_factory_qwen as factory  # noqa: E402
import make_qwen_kernels as kernels  # noqa: E402
import test_sdpa_decode_qwen_card_m as card  # noqa: E402

NL = chr(10)
JOB = Path('C:/Users/liamb/.claude/jobs/8376c877/tmp')
DUMP = Path(os.environ.get('QWEN_SDPA_SOURCES_DUMP', JOB / 'sdpa-decode-sources.txt'))
FACTORY_BASE = Path(os.environ.get('QWEN_SDPA_FACTORY_BASE', JOB / 'sdpa_decode_program_factory.cpp.3e0a69af'))
STAGE3 = HERE / 'stage3'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stage_text(stage):
    """The factory's new text per stage without the base file: F2 as built, F9 applied for stage 3."""
    text = ''.join(new for _label, _old, new in factory.EDITS)
    if stage == 3:
        text = text.replace(factory.F2_SHARE_REFUSED, factory.F9_NEW)
        text += ''.join(new for _label, _old, new in factory.STAGE3_EDITS)
    return text


class KernelSourceTests(unittest.TestCase):
    def test_the_committed_kernels_are_the_bases_plus_exactly_the_edits(self):
        for stage, directory in ((1, HERE), (3, STAGE3)):
            for name in (kernels.READER_NAME, kernels.COMPUTE_NAME):
                with self.subTest(stage=stage, kernel=name):
                    built = (directory / name).read_bytes()
                    self.assertEqual(sha(built), kernels.STAGE_OUTPUTS[stage][name])
                    self.assertNotIn(b'\r', built)
                    self.assertEqual(sha(kernels.revert_edits(name, built, stage)), kernels.BASES[name])

    def test_stage_three_is_stage_one_plus_r4_r5_and_the_same_compute(self):
        reader3 = (STAGE3 / kernels.READER_NAME).read_bytes()
        self.assertEqual(kernels.revert_edits(kernels.READER_NAME, reader3, 3, to_stage=1),
                         (HERE / kernels.READER_NAME).read_bytes())
        self.assertEqual((STAGE3 / kernels.COMPUTE_NAME).read_bytes(), (HERE / kernels.COMPUTE_NAME).read_bytes())
        self.assertEqual(kernels.OUTPUTS_STAGE3[kernels.COMPUTE_NAME], kernels.OUTPUTS[kernels.COMPUTE_NAME])
        self.assertEqual([label for label, _old, _new in kernels.STAGE_EDITS[3][kernels.READER_NAME]],
                         ['R1', 'R2', 'R3', 'R4', 'R5'])
        self.assertEqual(sorted(path.name for path in STAGE3.iterdir() if path.is_file()),
                         sorted((kernels.READER_NAME, kernels.COMPUTE_NAME)))

    def test_the_edits_carry_the_tail_predicate_on_both_sides(self):
        compute = (HERE / kernels.COMPUTE_NAME).read_text(encoding='utf-8')
        predicate = 'if (!mask_tail || k_chunk == k_num_chunks - 1) {'
        self.assertEqual(compute.count(predicate), 1)
        self.assertIn('constexpr bool mask_tail = get_compile_time_arg_val(32) == 1;', compute)
        self.assertEqual(compute.count('add_block_inplace<true>(cb_qk_im, cb_mask_in'), 1)
        # Stage 1: one paged mask read, under the predicate. Stage 3: the share branch has its own.
        for stage, directory, reads in ((1, HERE, 1), (3, STAGE3, 2)):
            with self.subTest(stage=stage):
                reader = (directory / kernels.READER_NAME).read_text(encoding='utf-8')
                self.assertEqual(reader.count(predicate), reads)
                self.assertIn('constexpr uint32_t qwen_cta = attention_sink_args.next_compile_time_args_offset();',
                              reader)
                self.assertEqual(reader.count('read_mask_chunk<'), reads)
                self.assertEqual(reader.count(
                    'mask_width_t, Sk_chunk_t_dynamic, mask_chunk_tiles, mask_start_tile_id, mask_reader);'), reads)

    def test_the_share_protocol_in_the_stage_three_reader(self):
        """Spec 7.2/7.3: the leader reads, waits READY from every twin, multicasts K and V, barriers,
        then flags VALID; a twin reserves both slots, resets VALID, signals READY, waits VALID;
        the mask comes after K and V; the kernel ends barriered with VALID at 0."""
        reader = (STAGE3 / kernels.READER_NAME).read_text(encoding='utf-8')
        share = reader[reader.index('if constexpr (kv_share) {'):reader.index('                } else {' + NL + '                    // Read K chunk')]
        leader = share[share.index('if (do_k_mcast) {'):share.index('                    } else {')]
        twin = share[share.index('// TWIN'):]
        order = ['read_k<', 'cb_v.get_write_ptr()', 'read_v<', 'kv_ready.wait(num_dests);', 'kv_ready.set(0);',
                 'CoreLocalMem<uint32_t>(k_slot)', 'CoreLocalMem<uint32_t>(v_slot)', 'noc.async_write_barrier();',
                 'kv_valid.set(1);', 'kv_valid.set_multicast(noc, mcast_x, mcast_y0, mcast_x, mcast_y1, num_dests);']
        self.assertEqual([leader.index(step) for step in order], sorted(leader.index(step) for step in order))
        self.assertEqual(leader.count('noc.async_write_multicast('), 2)
        self.assertIn('                            false,' + NL + '                            capacity_t>(', leader,
                      'the leader reads with use_mcast=false / reuse_k=false: the plain legacy DRAM reads')
        order = ['cb_k.reserve_back(k_chunk_tiles);', 'cb_v.reserve_back(v_chunk_tiles);', 'kv_valid.set(0);',
                 'Semaphore<>(kv_ready_semaphore_id).up(noc, mcast_x, mcast_y0, 1);', 'kv_valid.wait(1);',
                 'cb_k.push_back(k_chunk_tiles);', 'cb_v.push_back(v_chunk_tiles);']
        self.assertEqual([twin.index(step) for step in order], sorted(twin.index(step) for step in order))
        self.assertNotIn('read_k<', twin)
        self.assertNotIn('read_v<', twin)
        self.assertGreater(share.index('read_mask_chunk<'), share.index('cb_v.push_back(v_chunk_tiles);'))
        self.assertIn('Semaphore<> kv_valid(k_mcast_semaphore_id);', share)
        # R5: the last statement of kernel_main.
        tail = reader[reader.rindex('if constexpr (kv_share) {'):]
        self.assertTrue(tail.endswith('        Semaphore<>(k_mcast_semaphore_id).set(0);' + NL + '    }' + NL + '}' + NL))
        self.assertIn('noc.async_write_barrier();', tail)
        self.assertIn('noc.async_atomic_barrier();', tail)
        # Every early return comes before the head loop, i.e. before any semaphore traffic.
        loop = reader.index('for (uint32_t cur_head = cur_head_group * num_heads_per_core;')
        self.assertNotIn('return;', reader[loop:])
        self.assertNotIn('Semaphore<', reader[:loop])

    def test_a_wrong_base_is_refused(self):
        for stage in kernels.STAGES:
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, 'base is'):
                kernels.apply_edits(kernels.READER_NAME, b'// not the reader' + NL.encode(), stage)

    def test_the_kernels_regenerate_from_the_dump(self):
        if not DUMP.is_file():
            self.skipTest('no sources dump at %s' % DUMP)
        text = DUMP.read_bytes().decode('utf-8')
        for stage, directory in ((1, HERE), (3, STAGE3)):
            for name, path in kernels.DUMP_PATHS.items():
                with self.subTest(stage=stage, kernel=name):
                    base = kernels.from_dump(text, path)
                    self.assertEqual(sha(base), kernels.BASES[name])
                    self.assertEqual(kernels.apply_edits(name, base, stage), (directory / name).read_bytes())
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(kernels.main(['--dump', str(DUMP), '--stage', '3', '--out', out]), 0)
            self.assertEqual(kernels.main(['--dump', str(DUMP), '--stage', '3', '--check', out]), 0)
            self.assertEqual(kernels.main(['--dump', str(DUMP), '--check', out]), 1, 'stage 1 != the stage-3 files')
            self.assertEqual(kernels.main(['--dump', str(DUMP), '--check', str(HERE)]), 0)

    def test_the_share_reader_uses_the_ported_upstream_forms(self):
        """The multicast / semaphore forms R4 uses are the ones dataflow_common.hpp's MLA K
        multicast and the writer already use in this tt-metal, so the kernel compiles against it."""
        if not DUMP.is_file():
            self.skipTest('no sources dump at %s' % DUMP)
        text = DUMP.read_bytes().decode('utf-8')
        common = kernels.from_dump(text, kernels.DUMP_PREFIX + 'dataflow/dataflow_common.hpp').decode('utf-8')
        writer = kernels.from_dump(text, kernels.DUMP_PREFIX + 'dataflow/writer_decode_all.cpp').decode('utf-8')
        for form in ('noc.async_write_multicast(', 'MulticastEndpoint{},', '.noc_x_start = mcast_params.mcast_x,',
                     'mcast_sem.set_multicast(', 'mcast_sem.wait(1);', 'noc.async_atomic_barrier();', 'uint32_t read_k(',
                     'uint32_t k_base_read_ptr = k_write_ptr;', 'const KMcastParams& mcast_params = {}) {'):
            self.assertIn(form, common)
        self.assertIn('Semaphore<>(reducer_semaphore_id).up(noc, parent_noc_x, parent_noc_y,', writer)


class FactoryPatchTests(unittest.TestCase):
    def test_anything_but_the_tree_scratch_factory_is_refused(self):
        for stage in factory.STAGES:
            for source in (b'', b'int main() {}' + NL.encode()):
                with self.subTest(stage=stage, source=source), self.assertRaisesRegex(ValueError, 'unexpected factory'):
                    factory.patch(source, stage)
        with self.assertRaisesRegex(ValueError, 'unknown stage'):
            factory.patch(b'', 2)

    def test_the_base_factory_patches_to_the_recorded_bytes_and_back(self):
        if not FACTORY_BASE.is_file():
            self.skipTest('no 3e0a69af factory at %s' % FACTORY_BASE)
        base = FACTORY_BASE.read_bytes()
        self.assertEqual(sha(base), factory.BASE_FACTORY)
        text = base.decode('utf-8')
        for label, old, new in factory.EDITS:
            with self.subTest(edit=label):
                self.assertEqual(text.count(old), 1)
        patched = factory.patch(base)
        self.assertEqual(sha(patched), factory.QWEN_FACTORY)
        self.assertEqual(factory.unpatch(patched), base)
        stage3 = factory.patch(base, 3)
        self.assertEqual(sha(stage3), factory.QWEN_FACTORY_STAGE3)
        self.assertEqual(factory.unpatch(stage3, 3), base)
        self.assertEqual(factory.unpatch(stage3, 3, to_stage=1), patched)
        # F7 appends at index 32: the legacy compute arg list has exactly 32 entries.
        body = text[text.index('std::vector<uint32_t> compute_compile_time_args_common = {'):]
        body = body[body.index('{') + 1:body.index('};')]
        self.assertEqual(len([item for item in body.split(',') if item.strip()]), 32)
        # F6 appends after the last TensorAccessorArgs block of the reader list.
        reader = patched.decode('utf-8')
        f6 = reader.index('reader_compile_time_args_common.push_back(static_cast<uint32_t>(qwen_mask_tail));')
        self.assertGreater(f6, reader.rindex('.append_to(reader_compile_time_args_common);'))
        # Stage 3: F10's branch sits before the linear assignment's else; F12 inside the runtime-arg
        # loop's non-col-major branch, after the core_num_in_output it keeps.
        full = stage3.decode('utf-8')
        self.assertLess(full.index('} else if (qwen_kv_share) {'), full.index('// Q in DRAM, no sharding: simple linear assignment'))
        self.assertGreater(full.index('} else if (qwen_kv_share) {'), full.index('} else if ((is_q_sharded || is_output_sharded)'))
        f12 = full.index('                const CoreCoord leader{p % grid_size.x, (p / grid_size.x) * B};')
        self.assertGreater(f12, full.index('            core_num_in_output = i % num_cores_per_batch;'))
        self.assertLess(f12, full.index('uint32_t worker_id_for_reduce ='))
        # The idle-core reader vector keeps 20 slots: F12 reuses slots 15-19, adds none.
        self.assertIn('KernelDescriptor::CoreRuntimeArgs reader_rt_args(20, 0);', full)
        self.assertEqual(full.count('reader_rt_args.push_back('), text.count('reader_rt_args.push_back('))
        # The only semaphores are ids 0-2 and, with share, F11's id 3.
        self.assertEqual(full.count('desc.semaphores.push_back('), 4)

    def test_every_new_factory_branch_is_gated_on_qwen_mode(self):
        for label, old, new in factory.EDITS:
            added = new.replace(old, '', 1) if new.startswith(old) else new
            with self.subTest(edit=label):
                self.assertTrue('qwen_mode' in added or label in ('F2', 'F5') or 'kQwen' in added, label)
        self.assertIn('if (qwen_mode) {', factory.F2)
        self.assertIn('qwen_mode || (scratch_rounds_override', factory.F3_NEW)
        # Stage 3: every addition is gated on qwen_kv_share, which F2 derives from qwen_flags (zero
        # outside qwen_mode) and B > 1; F9's check sits inside F2's if (qwen_mode).
        self.assertIn('    const bool qwen_kv_share = (qwen_flags & kQwenKvShare) != 0 && B > 1;', factory.F2)
        self.assertIn('    const uint32_t qwen_flags = qwen_mode ? static_cast<uint32_t>', factory.F1)
        for label, old, new in factory.STAGE3_EDITS:
            kept = set(old.splitlines())
            added = [row for row in new.splitlines() if row not in kept]
            with self.subTest(edit=label):
                self.assertTrue(added[0].lstrip().startswith(('if (qwen_kv_share) {', '} else if (qwen_kv_share) {')),
                                (label, added[0]))
        f2 = factory.F2.replace(factory.F2_SHARE_REFUSED, factory.F9_NEW)
        self.assertGreater(f2.index(factory.F9_NEW), f2.index('    if (qwen_mode) {'))

    def test_the_twin_placement_keeps_every_role_and_fits(self):
        """F10's placement, mirrored in Python over the factory's own formulas (11 x 10 grid,
        max_cores_per_head_batch 16, 2 KV heads): every linear index keeps its role, the B twins
        of each position are vertically adjacent below their leader, no coordinate repeats, the
        F9 check admits B=2 and B=3 and refuses B=4."""
        gx, gy, heads = 11, 10, 2
        for batches in (2, 3, 4):
            uncapped = min(gx * gy, 16 * batches * heads) // batches
            per_head = max(1, uncapped // heads)
            per_batch = per_head * heads
            fits = ((per_batch + gx - 1) // gx) * batches <= gy
            with self.subTest(B=batches):
                self.assertEqual(fits, batches in (2, 3))
                if not fits:
                    continue
                cores = [(p % gx, (p // gx) * batches + b) for i in range(per_batch * batches)
                         for b, p in [(i // per_batch, i % per_batch)]]
                self.assertEqual(len(set(cores)), len(cores))
                self.assertTrue(all(x < gx and y < gy for x, y in cores))
                for p in range(per_batch):
                    column = [cores[b * per_batch + p] for b in range(batches)]
                    self.assertEqual({x for x, _y in column}, {p % gx})
                    self.assertEqual([y for _x, y in column], list(range(column[0][1], column[0][1] + batches)))
                self.assertEqual(gx * gy - len(cores), {2: 46, 3: 14}[batches])


class ContractTests(unittest.TestCase):
    """Values that cross file boundaries."""

    def f4_format(self):
        match = re.search(r'log_info\(tt::LogOp, "([^"]+)"', factory.F4)
        return match.group(1)

    def test_the_sentinel_and_flags_agree_everywhere(self):
        import pooled_attention_replay as replay
        magic = int(re.search(r'kQwenMagic = (0x[0-9A-F]+)u;', factory.F1).group(1), 16)
        self.assertEqual(magic, replay.QWEN_DECODE_MAGIC)
        self.assertEqual(int(re.search(r'kQwenMagicMask = (0x[0-9A-F]+)u;', factory.F1).group(1), 16), 0xFFFFFF00)
        self.assertEqual(int(re.search(r'kQwenMaskTail = (0x[0-9a-f]+)u;', factory.F1).group(1), 16), replay.QWEN_MASK_TAIL)
        self.assertEqual(int(re.search(r'kQwenKvShare = (0x[0-9a-f]+)u;', factory.F1).group(1), 16), replay.QWEN_KV_SHARE)
        self.assertEqual(card.MAGIC, magic)
        self.assertEqual(card.TAIL, magic | replay.QWEN_MASK_TAIL)
        self.assertEqual(card.SHARE, magic | replay.QWEN_KV_SHARE)
        self.assertEqual(card.TAIL_SHARE, magic | replay.QWEN_MASK_TAIL | replay.QWEN_KV_SHARE)
        self.assertEqual((card.SHARE_STAGE3, card.NO_FLAGS, card.QWEN_PLAIN), (card.SHARE, magic, magic))
        # 'slice' (0x4) and 'readahead' (0x8) are stage 4 (K64i): ../sdpa_decode_slice checks them against its factory;
        # 'extent' (0x20) is K64j's: ../k64j checks it against its own.
        self.assertEqual(replay.SDPA_MODE_NAMES, ('tail', 'share', 'slice', 'readahead', 'extent'))
        self.assertEqual((replay.QWEN_Q_SLICE, replay.QWEN_KV_READAHEAD), (0x4, 0x8))
        self.assertEqual(set(replay.SDPA_MODES_LATER), {'narrow'})

    def test_the_share_arguments_and_semaphores_line_up(self):
        """F5/F11's READY id is the reader's 4th suffix arg; VALID is k_mcast (id 2); F12 fills the
        reader's runtime slots 15-19 (do_k_mcast, mcast_x, mcast_y0, mcast_y1, num_dests)."""
        self.assertIn('const uint32_t kv_ready_semaphore_id = 3;', factory.F5_LINE)
        self.assertIn('.id = kv_ready_semaphore_id,', factory.F11)
        suffix = re.findall(r'reader_compile_time_args_common\.push_back\(([^;]+)\);', factory.F6)
        self.assertEqual(suffix, ['static_cast<uint32_t>(qwen_mask_tail)', 'qwen_mask_width_t',
                                  'static_cast<uint32_t>(qwen_kv_share)', 'kv_ready_semaphore_id'])
        reader = (STAGE3 / kernels.READER_NAME).read_text(encoding='utf-8')
        for index, name in enumerate(('mask_tail', 'mask_width_t', 'kv_share', 'kv_ready_semaphore_id')):
            self.assertRegex(reader, r'constexpr (bool|uint32_t) %s = get_compile_time_arg_val\(qwen_cta \+ %d\)' % (name, index))
        for name in ('do_k_mcast', 'mcast_x', 'mcast_y0', 'mcast_y1', 'num_dests'):
            self.assertIn(name + ' =', factory.F12)
        self.assertIn('static_assert(!(kv_share && use_k_mcast)', reader)
        self.assertIn('static_assert(!(kv_share && reuse_k)', reader)

    def test_the_binary_markers_agree_everywhere(self):
        import pooled_attention_replay as replay
        self.assertTrue(factory.F9_NEW.count(factory.SHARE_MARKER) == 1)
        self.assertIn(factory.STAGE1_SHARE_REFUSAL, factory.F2_SHARE_REFUSED)
        self.assertEqual(replay.QWEN_SDPA_SHARE_MARKER, factory.SHARE_MARKER.encode())
        self.assertEqual(card.SHARE_BINARY_MARKER, factory.SHARE_MARKER.encode())
        self.assertEqual(card.STAGE1_BINARY_MARKER, factory.STAGE1_SHARE_REFUSAL.encode())
        self.assertEqual(card.BINARY_MARKER, replay.QWEN_SDPA_BINARY_MARKER)
        self.assertNotIn(factory.SHARE_MARKER, stage_text(1))
        self.assertIn(factory.SHARE_MARKER, stage_text(3))
        self.assertNotIn(factory.STAGE1_SHARE_REFUSAL, stage_text(3))
        self.assertEqual(card.binary_stage(dict(flags=True, share=True, stage1=False)), 3)
        self.assertEqual(card.binary_stage(dict(flags=True, share=False, stage1=True)), 1)
        self.assertEqual(card.binary_stage(dict(flags=False, share=False, stage1=False)), 0)
        with self.assertRaises(RuntimeError):
            card.binary_stage(dict(flags=True, share=True, stage1=True))

    def test_the_log_line_the_factory_prints_is_what_the_parsers_and_the_gate_expect(self):
        import lever_n_m3native_gate as gate
        import pooled_attention_replay as replay
        text = self.f4_format()
        self.assertTrue(text.startswith(replay.QWEN_SDPA_BINARY_MARKER.decode()))
        self.assertTrue(text.startswith(card.BINARY_MARKER.decode()))
        # fmt's {:#x} and {} on these types format as Python's do; bool prints false/true.
        line = text.format(1, 3, 2, 1032, 1032, 'false', 4, 790592)
        self.assertEqual(line, '[QWEN-SDPA] flags=0x1 B=3 PNHt=2 St=1032 mask_width_t=1032 kv_share=false '
                               'scratch_slots=4 cb_bytes=790592')
        self.assertEqual(card.factory_lines('prefix ' + line + NL),
                         [dict(flags='0x1', B=3, PNHt=2, St=1032, mask_width_t=1032, kv_share='false',
                               scratch_slots=4, cb_bytes=790592)])
        self.assertIn(gate.SDPA_MODES_MARKERS[2], line + NL)
        share = text.format(3, 2, 3, 1032, 1032, 'true', 4, 1042496)
        markers = gate.sdpa_mode_markers({'tail', 'share'})
        self.assertIn(markers[2], share + NL)
        self.assertEqual(card.line_key(card.factory_lines(share)[0]), (33024, 2, 3, 1032, 3))
        self.assertIn('[QWEN-SDPA] flags=', factory.MARKERS)
        # The gate's modes line is apply_sdpa_modes' own format.
        import inspect
        self.assertIn("'%s modes=%s rows=%d capacity=%d bundles=%s flags=%s mask=%s'",
                      inspect.getsource(replay.apply_sdpa_modes))
        self.assertIn("'narrow' if 'extent' in modes else 'wide'", inspect.getsource(replay.apply_sdpa_modes))
        self.assertTrue(markers[1].startswith(replay.SDPA_MODES_MARKER + ' modes=share,tail '))

    def test_the_refusals_the_card_m_test_expects_are_the_factorys_texts(self):
        self.assertEqual(len(card.REFUSALS), 6)
        for name, _sentinel, needle, _mask, _batches, _causal, stages in card.REFUSALS:
            for stage in stages:
                with self.subTest(refusal=name, stage=stage):
                    self.assertIn(needle, stage_text(stage))
        self.assertEqual([entry[0] for entry in card.refusals_for(1)],
                         ['share flag (stage 1)', 'unknown flag 0x10', 'tail with a 512-wide mask',
                          'narrow mask without tail', 'sentinel on a causal call'])
        self.assertEqual([entry[0] for entry in card.refusals_for(3)][0], 'twin bands for B=4')
        self.assertNotIn('KV share is not in this build', stage_text(3))

    def test_the_kernel_names_the_factory_selects_are_the_committed_files(self):
        self.assertIn('"dataflow/%s"' % kernels.READER_NAME, factory.F8R_NEW)
        self.assertIn('"compute/%s"' % kernels.COMPUTE_NAME, factory.F8C_NEW)

    def build_values(self, name):
        script = (HERE / name).read_text(encoding='utf-8')
        self.assertNotIn(chr(13), script)
        return script, dict(re.findall(r'^([A-Z_0-9]+)=([0-9a-f]{8,64})$', script, flags=re.M))

    def test_build_k64e_enforces_the_recorded_shas(self):
        script, values = self.build_values('build_k64e.sh')
        self.assertEqual(values['FACTORY_BASE'], factory.BASE_FACTORY)
        self.assertEqual(values['FACTORY_QWEN'], factory.QWEN_FACTORY)
        self.assertEqual(values['READER_ALL'], kernels.READER_BASE)
        self.assertEqual(values['COMPUTE_ALL'], kernels.COMPUTE_BASE)
        self.assertEqual(values['READER_QWEN'], kernels.OUTPUTS[kernels.READER_NAME])
        self.assertEqual(values['COMPUTE_QWEN'], kernels.OUTPUTS[kernels.COMPUTE_NAME])
        import sdpa_tree_scratch
        self.assertEqual(values['FACTORY_BASE'], sdpa_tree_scratch.PATCHED_FACTORY_SHA256)
        self.assertEqual(values['FACTORY_UNPATCHED'], sdpa_tree_scratch.HASHES['sdpa_decode_program_factory.cpp'])
        self.assertEqual(values['WRITER_ALL'], sdpa_tree_scratch.HASHES['kernels/dataflow/writer_decode_all.cpp'])
        self.assertEqual(values['COMPUTE_ALL'], sdpa_tree_scratch.HASHES['kernels/compute/sdpa_flash_decode.cpp'])

    def test_build_k64f_enforces_the_stage_three_shas(self):
        script, values = self.build_values('build_k64f.sh')
        _script, k64e = self.build_values('build_k64e.sh')
        self.assertEqual(values['FACTORY_QWEN'], factory.QWEN_FACTORY_STAGE3)
        self.assertEqual(values['FACTORY_QWEN_STAGE1'], factory.QWEN_FACTORY)
        self.assertEqual(values['READER_QWEN'], kernels.OUTPUTS_STAGE3[kernels.READER_NAME])
        self.assertEqual(values['READER_QWEN_STAGE1'], kernels.OUTPUTS[kernels.READER_NAME])
        self.assertEqual(values['COMPUTE_QWEN'], kernels.OUTPUTS_STAGE3[kernels.COMPUTE_NAME])
        for name in ('FACTORY_BASE', 'FACTORY_UNPATCHED', 'READER_ALL', 'COMPUTE_ALL', 'WRITER_ALL', 'DATAFLOW_COMMON',
                     'RT_ARGS_COMMON', 'PREFILL_COMBINED', 'K64D_TTNNCPP_PREFIX'):
            self.assertEqual(values[name], k64e[name], name)
        self.assertIn('KS=$S/stage3', script)
        self.assertIn('make_qwen_kernels.py" --stage 3 --reader "$W/reader_decode_all.cpp" --compute '
                      '"$W/sdpa_flash_decode.cpp" --check "$KS"', script)
        self.assertIn('apply_factory_qwen.py" "$S/sdpa_decode_program_factory.cpp.$FACTORY_BASE" --stage 3 --out', script)
        self.assertIn('docker cp "$KS/$(basename "$kernel")" "ttbuild:$KD/$kernel"', script)
        self.assertIn("share=$(count \"$so\" '[QWEN-SDPA] KV-share twin bands')", script)
        self.assertIn('test "$share" -ge 1 ||', script)
        self.assertIn('test "$stage1" -eq 0 ||', script)
        self.assertIn('GRAFT=${K64F_GRAFT:-$HOME/opgraft-K64f}', script)
        self.assertIn('echo "K64F_TTNNCPP_SHA256=$(hsha "$G/_ttnncpp.so")"', script)
        self.assertEqual([line for line in script.splitlines() if 'K64E' in line and 'K64E_KEEP_STAGED' not in line], [])

    def test_the_build_scripts_leave_build_release_on_the_restored_factory(self):
        """docker cp keeps the saved factory's old mtime: without a touch and a rebuild, ninja
        calls the unity TU up to date and the next ttbuild graft links the qwen factory."""
        for name, staged_from in (('build_k64e.sh', 'elif [ "$cur" = "$FACTORY_BASE" ]'),
                                  ('build_k64f.sh', 'elif [ "$cur" = "$FACTORY_BASE" ] || [ "$cur" = "$FACTORY_QWEN_STAGE1" ]')):
            with self.subTest(script=name):
                script = (HERE / name).read_text(encoding='utf-8')
                restore = script[script.index('restore_ttbuild() {'):script.index('on_exit() {')]
                self.assertIn('docker exec ttbuild touch "$F"', restore)
                self.assertLess(restore.index('ttbuild:$F'), restore.index('touch "$F"'), 'touch after the copy')
                staged = script[script.index(staged_from):]
                staged = staged[:staged.index('else' + NL + '  echo "FAIL: unexpected ttbuild decode factory')]
                self.assertLess(staged.index('ttbuild:$F'), staged.index('touch "$F"'))
                step6 = script[script.index('# ---------- 6.'):script.index('# ---------- 7.')]
                ninja = 'ninja -C build_Release ttnn/_ttnncpp.so ttnn/_ttnn.so'
                self.assertLess(step6.index('restore_ttbuild'), step6.index(ninja), 'rebuild after the restore')
                self.assertIn("grep -caF -- '[QWEN-SDPA] flags='", step6)
                self.assertIn('rebuilt=1', step6)
                self.assertIn('trap on_exit EXIT', script)

    def test_the_arm_and_the_runner_mount_the_directory_the_builds_assemble(self):
        target = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode'
        for name in ('build_k64e.sh', 'build_k64f.sh'):
            build = (HERE / name).read_text(encoding='utf-8')
            self.assertIn('OPS=/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer' + NL, build)
            self.assertIn('D=$OPS/sdpa_decode' + NL, build)
            self.assertIn('docker cp ttbuild:$D "$G/sdpa_decode"', build)
        arm = (CI / 'lever_n_m3native_run_arm.sh').read_text(encoding='utf-8')
        self.assertIn('$KOPGRAFT64/sdpa_decode:%s:ro' % target, arm)
        # The arm's kernel cache is keyed by the two kernels' bytes (then any further *qwen*.cpp, stage 4 on):
        # K64f never reuses K64e's reader.
        self.assertIn('kernel_cache="/experiment-cache/kernels-qwen-$({ cat "$sdpa_kernels/dataflow/reader_decode_qwen.cpp"', arm)
        self.assertIn('-e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', arm)
        runner = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')
        self.assertIn('$G/sdpa_decode:%s:ro' % target, runner)

    def test_run_card_m_sets_the_arms_scratch_and_has_the_watcher_pass(self):
        runner = (HERE / 'run_card_m.sh').read_text(encoding='utf-8')
        self.assertNotIn(chr(13), runner)
        self.assertIn('  -e QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 ' + chr(92), runner)
        self.assertEqual(card.SCRATCH_ENV, 'QWEN_SDPA_TREE_SCRATCH_ROUNDS')
        self.assertIn('G=${KOPGRAFT64:-$HOME/opgraft-K64f}', runner)
        watcher = runner[runner.index('if [ "${WATCHER:-}" = "1" ]; then'):runner.index('# shellcheck disable=SC2206')]
        self.assertIn('timeout_s=900', watcher)
        self.assertIn('-e TT_METAL_WATCHER=5', watcher)
        self.assertIn('--watchdog "${WATCHDOG_S:-120}"', watcher)
        self.assertIn('--capacities 2304,33024', watcher)
        self.assertIn('--no-timing', watcher)
        self.assertIn('timeout -k 30 "$timeout_s" docker run', runner)
        self.assertIn('"${WM[@]}"', runner)
        self.assertEqual([line for line in runner.splitlines() if 'tt-smi' in line and not line.lstrip().startswith(('#', 'echo', '"'))], [])
        args = card.parse_args(['--out', 'x.json', '--capacities', '2304,33024', '--seeds', '0', '--variants',
                                'normal,zeroq', '--starts', '0,240', '--no-timing', '--watchdog', '120'])
        self.assertEqual((args.capacities, args.watchdog, args.no_timing, args.share_alternations, args.trace_replays),
                         ([2304, 33024], 120.0, True, 500, 200))


def inspect_source(function):
    import inspect
    return inspect.getsource(function)


class CardMHelperTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_mask_positions_follow_the_refresh_kernel(self):
        """attention_mask_replay.cpp: position = start + offset + batch * rows + (head % (rows * 6)) / 6."""
        source = (CI / 'attention_mask_replay.cpp').read_text(encoding='utf-8')
        self.assertIn('const uint32_t position = start + offset + batch * rows + (head % (rows * 6)) / 6;', source)
        self.assertIn('cache_position > position ? 0xff80 : 0', source)
        positions = card.mask_positions(1000, (0, 4, 8))
        self.assertEqual(len(positions), 3)
        self.assertEqual(positions[0][:6], [1000] * 6)
        self.assertEqual(positions[0][6:12], [1001] * 6)
        self.assertEqual(positions[0][24:30], [1000] * 6, 'the second KV group repeats the tokens')
        self.assertEqual(positions[2][47], 1000 + 0 + 8 + 3)
        self.assertEqual(card.mask_positions(1000, (12,))[0][23], 1000 + 12 + 3)
        eight = card.mask_positions(1000, card.G8_OFFSETS, rows=8)
        self.assertEqual([len(row) for row in eight], [96, 96])
        self.assertEqual(eight[1][47], 1000 + 8 + 7)
        self.assertEqual(eight[1][48], 1000 + 8, 'KV head 1 starts over at token 0')

    def test_the_masks_are_zero_but_the_tail_and_narrow_is_the_wide_tail(self):
        torch = self.torch
        capacity, start = 2304, 2304 - 256 + 7
        wide = card.build_mask(torch, capacity, start, (0, 4, 8))
        self.assertEqual(tuple(wide.shape), (3, 1, 48, capacity))
        self.assertEqual(wide.dtype, torch.bfloat16)
        self.assertTrue(bool((wide[..., :capacity - 256] == 0).all()))
        tail = wide[..., capacity - 256:].float()
        self.assertTrue(bool(((tail == 0) | (tail == float('-inf'))).all()))
        visible = (tail[0, 0, 0] == 0).nonzero().flatten()
        self.assertEqual(int(visible.max()) + capacity - 256, start)
        self.assertTrue(bool((tail[0, 0, 0, :7 + 1] == 0).all()))
        narrow = card.build_mask(torch, capacity, start, (0, 4, 8), width=256)
        self.assertTrue(torch.equal(narrow, wide[..., capacity - 256:]))
        half = card.build_mask(torch, capacity, start, (12,), width=512)
        self.assertEqual(tuple(half.shape), (1, 1, 48, 512))
        planted = card.build_mask(torch, capacity, start, (0, 4, 8), plant=True)
        self.assertTrue(bool((planted[..., 0].float() == float('-inf')).all()))
        self.assertTrue(torch.equal(planted[..., 1:], wide[..., 1:]))
        with self.assertRaises(ValueError):
            card.build_mask(torch, capacity, start, (0,), width=256, plant=True)
        g8 = card.build_mask(torch, capacity, start, card.G8_OFFSETS, rows=8)
        self.assertEqual(tuple(g8.shape), (2, 1, 96, capacity))

    def test_the_g8_and_g4_masks_are_the_same_per_token(self):
        """Unfolded to (token, head), the G8 mask equals the G4 bundles' masks: E4 compares like with like."""
        torch = self.torch
        capacity, start = 2304, 2304 - 256 + 240
        g8 = card.build_mask(torch, capacity, start, card.G8_OFFSETS, rows=8)[:, :, :, -256:]
        g4 = torch.cat([card.build_mask(torch, capacity, start, offsets)[:, :, :, -256:] for offsets in card.BUNDLES], dim=0)

        def tokens(mask, rows):
            return torch.cat([card.unfold_rows(mask[b:b + 1].reshape(1, 1, rows * 12, 256), rows)
                              for b in range(mask.shape[0])], dim=1)

        self.assertTrue(torch.equal(tokens(g8, 8), tokens(g4, 4)))

    def test_the_fold_helpers_are_attention_head_folds(self):
        import attention_head_fold
        torch = self.torch
        tokens = torch.randn(1, 16, 12, 256)
        for rows in (4, 8, 16):
            with self.subTest(rows=rows):
                folded = card.fold_tokens(tokens[:, :rows])
                self.assertTrue(torch.equal(folded, attention_head_fold.fold_query(tokens[:, :rows])))
                self.assertTrue(torch.equal(card.unfold_rows(folded, rows), attention_head_fold.unfold_output(folded, rows)))
        padded = torch.cat([card.fold_tokens(tokens[:, :4]), torch.zeros(1, 1, 16, 256)], dim=2)
        self.assertTrue(torch.equal(card.unfold_rows(padded, 4), tokens[:, :4]), 'padded rows are dropped')
        bundle = card.fold_entries(torch, tokens, (0, 4, 8), 4)
        self.assertEqual(tuple(bundle.shape), (1, 3, 48, 256))
        self.assertTrue(torch.equal(card.unfold_entries(torch, bundle, 4), tokens[:, :12]))

    def test_share_geometries_are_the_same_tokens(self):
        torch = self.torch
        keys = torch.randn(40, 2, 64, 256).to(torch.bfloat16)
        table = torch.randperm(40)[:36].to(torch.int32)
        for variant in card.VARIANTS:
            with self.subTest(variant=variant):
                (n8, q8, o8, r8), (n3, q3, o3, r3), (n1, q1, o1, r1) = card.share_geometries(torch, 2, variant, keys, table)
                self.assertEqual((n8, n3, n1), ('G8', 'G4B3', 'G4B1'))
                self.assertEqual((tuple(q8.shape), tuple(q3.shape), tuple(q1.shape)),
                                 ((1, 2, 96, 256), (1, 3, 48, 256), (1, 1, 48, 256)))
                self.assertEqual((o8, o3, o1, r8, r3, r1), (card.G8_OFFSETS, (0, 4, 8), (12,), 8, 4, 4))
                g8 = card.unfold_entries(torch, q8, 8)
                g4 = torch.cat([card.unfold_entries(torch, q3, 4), card.unfold_entries(torch, q1, 4)], dim=1)
                self.assertTrue(torch.equal(g8, g4))

    def test_queries_and_the_pool_geometry(self):
        torch = self.torch
        self.assertEqual(card.num_blocks(33024), 33024 // 64 + 64)
        self.assertEqual(card.num_blocks(131328), 2052 + 64)
        for bad in (33000, 0, 33024.0):
            with self.assertRaises(ValueError):
                card.num_blocks(bad)
        normal = card.build_query(torch, 3, 0, 'normal')
        self.assertEqual(tuple(normal.shape), (1, 3, 48, 256))
        self.assertTrue(torch.equal(normal, card.build_query(torch, 3, 0, 'normal')), 'seeded')
        stage1 = torch.randn(1, 3, 48, 256, generator=torch.Generator().manual_seed(1000)).to(torch.bfloat16)
        self.assertTrue(torch.equal(normal, stage1), 'rows=4 draws the stage-1 bytes')
        self.assertEqual(tuple(card.build_query(torch, 2, 0, 'normal', rows=8).shape), (1, 2, 96, 256))
        zero = card.build_query(torch, 1, 0, 'zeroq')
        self.assertTrue(bool((zero[0, :, ::4] == 0).all()))
        self.assertFalse(bool((zero[0, :, 1::4] == 0).all()))
        keys = torch.randn(10, 2, 64, 256).to(torch.bfloat16)
        table = torch.randperm(10)[:4].to(torch.int32)
        peaky = card.build_query(torch, 3, 0, 'peaky', keys, table).float()
        scores = torch.einsum('bhd,ntpd->bhntp', peaky[0], keys.float()) / 16
        self.assertGreater(float(scores.max()), 5.0, 'aligned keys dominate')
        with self.assertRaises(ValueError):
            card.build_query(torch, 3, 0, 'peaky')

    def test_program_keys_and_the_line_check(self):
        self.assertEqual(card.program_key(33024, 3, 48, None, card.TAIL), (33024, 3, 2, 1032, 1))
        self.assertEqual(card.program_key(33024, 2, 96, 256, card.TAIL_SHARE), (33024, 2, 3, 8, 3))

        def line(key, **changes):
            capacity, batches, pnht, width, flags = key
            values = dict(flags='0x%x' % flags, B=batches, PNHt=pnht, St=capacity // 32, mask_width_t=width,
                          kv_share='true' if flags & 2 and batches > 1 else 'false', scratch_slots=4,
                          cb_bytes=card.CB_BYTES.get((capacity, pnht), 123))
            values.update(changes)
            return values

        requested = {card.program_key(c, b, rows * 12, w, s) for c in (33024, 131328)
                     for b, rows in ((3, 4), (1, 4), (2, 8)) for w in (None, 256) for s in (card.TAIL, card.TAIL_SHARE)}
        requested |= {card.program_key(2304, 2, 96, None, card.SHARE)}
        complete = [line(key) for key in requested]
        self.assertEqual(card.check_program_lines(complete, requested), [])
        self.assertEqual(card.check_program_lines(complete + complete, requested), [], 'cache off: repeats')
        missing = sorted(requested)[0]
        self.assertEqual(card.check_program_lines([line(key) for key in requested if key != missing], requested),
                         ['%s: no [QWEN-SDPA] line' % card.describe(missing)])
        share_key = (33024, 2, 3, 1032, 3)
        wrong = [line(key, kv_share='false') if key == share_key else line(key) for key in requested]
        self.assertIn("'kv_share': ('false', 'true')", card.check_program_lines(wrong, requested)[0])
        b1_share = (33024, 1, 2, 1032, 3)
        wrong = [line(key, kv_share='true') if key == b1_share else line(key) for key in requested]
        self.assertIn('kv_share', card.check_program_lines(wrong, requested)[0], 'B=1 never shares')
        wrong = [line(key, cb_bytes=1) if key == share_key else line(key) for key in requested]
        self.assertIn('cb_bytes', card.check_program_lines(wrong, requested)[0])
        self.assertEqual(card.check_program_lines(complete + [line((4352, 3, 2, 136, 1))], requested)[-1],
                         '1 [QWEN-SDPA] programs nobody requested: cap=4352 B=3 PNHt=2 mask_width_t=136 flags=0x1')
        # The stage-1 form: tail programs per capacity x G4 batch x width.
        stage1 = [line(key) for key in {card.program_key(c, b, 48, w, card.TAIL) for c in (33024, 131328)
                                        for b in (3, 1) for w in (None, 256)}]
        self.assertEqual(card.check_factory_lines(stage1, (33024, 131328)), [])
        self.assertEqual(len(card.check_factory_lines(stage1[1:], (33024, 131328))), 1)
        self.assertEqual(card.check_factory_lines([entry for entry in stage1 if entry['mask_width_t'] != 8],
                                                  (33024, 131328), narrow=False), [])
        self.assertIn('scratch_slots', card.check_factory_lines([dict(stage1[0], scratch_slots=15)] + stage1[1:],
                                                                (33024, 131328))[0])

    def test_the_watchdog_fires_only_past_the_deadline_and_exits_3(self):
        now = [0.0]
        exits, fired, written = [], [], []

        class Stream:
            def write(self, text):
                written.append(text)

            def flush(self):
                pass

        dog = card.Watchdog(60, on_fire=fired.append, stream=Stream(), exit=exits.append, clock=lambda: now[0])
        self.assertFalse(dog.check(), 'nothing armed')
        with dog.op('sdpa B=2'):
            now[0] = 59.0
            self.assertFalse(dog.check())
            with dog.op('read back'):
                now[0] = 118.0
                self.assertFalse(dog.check(), 'the inner op has its own deadline')
            now[0] = 120.0
            self.assertTrue(dog.check(), 'the outer op is past its deadline again')
        self.assertEqual((exits, fired), ([3], ['sdpa B=2']))
        self.assertIn("WATCHDOG: 'sdpa B=2' did not return within 60s", written[0])
        self.assertFalse(dog.check(), 'disarmed on exit')
        off = card.Watchdog(0, exit=self.fail)
        with off.op('anything'):
            pass
        self.assertIsNone(off.start().thread)


# The paged decode binding, sdpa_decode_nanobind.cpp:82-106 in the dump: four positionals,
# then kw_only(). A nanobind call with any other keyword raises TypeError before validate.
PAGED_POSITIONAL = ('input_tensor_q', 'input_tensor_k', 'input_tensor_v', 'page_table_tensor')
PAGED_KEYWORDS = ('is_causal', 'attn_mask', 'cur_pos_tensor', 'attention_sink', 'scale', 'sliding_window_size',
                  'memory_config', 'program_config', 'compute_kernel_config', 'paged_cache_geometry',
                  'cache_position_modulo')


class StrictTtnn:
    """A ttnn stand-in whose paged decode refuses what the real binding and validate refuse."""
    int32, bfloat16, ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'int32', 'bf16', 'rm', 'tile', 'dram'

    class Tensor:
        def __init__(self, dtype, layout, length):
            self.dtype, self.layout, self.length = dtype, layout, length

    def __init__(self):
        self.calls = []
        self.transformer = self

    def SDPAProgramConfig(self, **options):  # noqa: N802 - mirrors ttnn
        return options

    def from_torch(self, host, *, device, dtype, layout, memory_config):
        return self.Tensor(dtype, layout, host.shape[-1])

    def paged_scaled_dot_product_attention_decode(self, *args, **options):
        if len(args) != len(PAGED_POSITIONAL):
            raise TypeError('incompatible function arguments: %d positionals' % len(args))
        unknown = sorted(set(options) - set(PAGED_KEYWORDS))
        if unknown:
            raise TypeError('incompatible function arguments: %s' % unknown)
        if options.get('is_causal', True):
            position = options.get('cur_pos_tensor')
            if position is None:
                raise RuntimeError('Must have cur_pos tensor for paged attention in causal mode')
            if position.dtype != self.int32 or position.layout != self.ROW_MAJOR_LAYOUT:
                raise RuntimeError('Expect cur_pos to be INT32 ROW_MAJOR')
            if position.length != args[3].rows:
                raise RuntimeError('cur_pos must have batch size equal to Q')
            if (options['program_config']['q_chunk_size'] & 0xFFFFFF00) == (card.TAIL & 0xFFFFFF00):
                raise RuntimeError('TT_FATAL [QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor')
        self.calls.append(dict(options, pages=args[3]))
        return 'out'


class CardMCallTests(unittest.TestCase):
    """Case.call must be a call the real paged binding accepts (the card-M run on hardware found
    a list-valued cur_pos= keyword the binding does not have)."""

    def setUp(self):
        import torch
        self.torch = torch
        self.ttnn = StrictTtnn()
        grid = type('Grid', (), {'x': 11, 'y': 10})()
        device = type('Device', (), {'compute_with_storage_grid_size': lambda self: grid})()
        self.case = card.Case.__new__(card.Case)
        self.case.ttnn, self.case.torch, self.case.device, self.case.owned = self.ttnn, torch, device, []
        self.case.k = self.case.v = 'kv'
        self.case.capacity, self.case.requested = 33024, None
        self.case.tables = {}
        for rows in (3, 1):
            self.case.tables[(rows, False)] = self.ttnn.Tensor(self.ttnn.int32, self.ttnn.ROW_MAJOR_LAYOUT, 2)
            self.case.tables[(rows, False)].rows = rows
        self.query = type('Query', (), {'shape': (1, 3, 48, 256)})()

    def test_the_fake_binding_is_the_dumped_binding(self):
        if not DUMP.is_file():
            self.skipTest('no sources dump')
        text = DUMP.read_text(encoding='utf-8', errors='replace')
        begin = text.index('nb::arg("page_table_tensor").noconvert(),')
        end = text.index('nb::arg("cache_position_modulo") = nb::none());', begin)
        names = re.findall(r'nb::arg\("([a-z_]+)"\)', text[begin:end + 60])
        self.assertEqual(tuple(names), PAGED_POSITIONAL[3:] + PAGED_KEYWORDS)
        self.assertNotIn('cur_pos', names)

    def test_the_equality_and_timing_calls_are_accepted(self):
        for sentinel in (card.LEGACY, card.TAIL, card.TAIL_SHARE):
            for mask in (None, 'mask'):
                self.assertEqual(self.case.call(self.query, mask, sentinel), 'out')
        self.assertEqual(len(self.ttnn.calls), 6)
        self.assertEqual(self.ttnn.calls[0]['is_causal'], False)
        self.assertIs(self.ttnn.calls[0]['pages'], self.case.tables[(3, False)])
        distinct = self.ttnn.Tensor(self.ttnn.int32, self.ttnn.ROW_MAJOR_LAYOUT, 2)
        self.case.call(self.query, 'mask', card.SHARE, pages=distinct)
        self.assertIs(self.ttnn.calls[-1]['pages'], distinct, 'N1 passes its own page table positionally')

    def test_the_causal_refusal_reaches_the_factorys_text(self):
        positions = self.case.upload(self.torch.full((3,), 33023, dtype=self.torch.int32), self.ttnn.int32)
        result = card.expect_refusal(self.case, self.query, None, card.TAIL, 'modes are non-causal',
                                     is_causal=True, cur_pos_tensor=positions)
        self.assertEqual((result['refused'], result['matched']), (True, True), result['message'])
        legacy = self.case.call(self.query, None, card.LEGACY, is_causal=True, cur_pos_tensor=positions)
        self.assertEqual(legacy, 'out', 'the same causal call without the sentinel is a valid call')

    def test_the_controls_pass_a_cur_pos_tensor(self):
        source = inspect_source(card.controls)
        self.assertNotIn('cur_pos=', source)
        self.assertIn('options.update(is_causal=True, cur_pos_tensor=positions)', source)
        self.assertIn('dtype=torch.int32), case.ttnn.int32)', source)


class FakeTensor:
    def __init__(self, data, dtype=None):
        self.data, self.dtype = data, dtype
        self.freed = False

    @property
    def shape(self):
        return tuple(self.data.shape)


class FakeDevice:
    """The ttnn surface the card-M test touches, with an SDPA stand-in that has the graft's
    semantics: legacy/plain read every entry's own page-table row and the whole mask; tail reads
    only the mask's last 256 columns; share (B>1) makes every entry use row 0; the stage-1
    factory refuses 0x2; each new qwen program prints the F4 line once to fd 1. Trace capture
    records the launches and execute_trace replays them into the same output tensors."""

    int32, bfloat16, bfloat8_b = 'int32', 'bf16', 'bf8'
    ROW_MAJOR_LAYOUT, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'rm', 'tile', 'dram'

    def __init__(self, torch, stage=3):
        self.torch, self.stage = torch, stage
        self.transformer = self
        self.programs, self.traces, self.capturing = set(), {}, None
        self.launches = 0

    # device -----------------------------------------------------------------------
    def open_device(self, **options):
        self.options = options
        return self

    def close_device(self, device):
        self.closed = True

    def compute_with_storage_grid_size(self):
        return type('Grid', (), {'x': 11, 'y': 10})()

    def synchronize_device(self, device):
        pass

    def SDPAProgramConfig(self, **options):  # noqa: N802
        return options

    def from_torch(self, host, *, device, dtype, layout, memory_config):
        return FakeTensor(host.clone(), dtype)

    def to_torch(self, tensor):
        if tensor.freed:
            raise RuntimeError('read of a freed tensor')
        return tensor.data.clone()

    def deallocate(self, tensor):
        tensor.freed = True

    # trace ------------------------------------------------------------------------
    def begin_trace_capture(self, device, cq_id):
        self.capturing = len(self.traces)
        self.traces[self.capturing] = []
        return self.capturing

    def end_trace_capture(self, device, trace, cq_id):
        self.capturing = None

    def execute_trace(self, device, trace, cq_id, blocking):
        for output, compute in self.traces[trace]:
            output.data = compute()

    def release_trace(self, device, trace):
        del self.traces[trace]

    # the op -----------------------------------------------------------------------
    def paged_scaled_dot_product_attention_decode(self, query, k, v, pages, *, is_causal, scale, program_config,
                                                  memory_config, attn_mask=None, cur_pos_tensor=None):
        torch = self.torch
        sentinel = program_config['q_chunk_size']
        qwen = (sentinel & 0xFFFFFF00) == 0x51DEC000
        flags = sentinel & 0xFF if qwen else 0
        batches, rows = query.shape[1], query.shape[2]
        capacity = pages.data.shape[1] * 64
        width = attn_mask.shape[3] if attn_mask is not None else capacity
        if qwen:
            if flags & ~0x3:
                raise RuntimeError('TT_FATAL [QWEN-SDPA] unknown flags 0x%x' % flags)
            if self.stage == 1 and flags & 0x2:
                raise RuntimeError('TT_FATAL [QWEN-SDPA] KV share is not in this build')
            if is_causal or cur_pos_tensor is not None:
                raise RuntimeError('TT_FATAL [QWEN-SDPA] modes are non-causal, full-window and take no cur_pos tensor')
            if flags & 0x2 and batches > 1 and ((32 + 10) // 11) * batches > 10:
                raise RuntimeError('TT_FATAL [QWEN-SDPA] KV-share twin bands do not fit the 11x10 grid for B=%d' % batches)
            if flags & 0x1 and width not in (capacity, 256):
                raise RuntimeError('TT_FATAL [QWEN-SDPA] tail mask must be full width (%d) or one chunk (8), got %d'
                                   % (capacity // 32, width // 32))
            if not flags & 0x1 and width != capacity:
                raise RuntimeError('TT_FATAL [QWEN-SDPA] a narrow mask needs the tail flag')
            key = (capacity, batches, (rows + 31) // 32, width // 32, flags)
            if key not in self.programs:
                self.programs.add(key)
                pnht = (rows + 31) // 32
                os.write(1, ('Op | INFO | [QWEN-SDPA] flags=0x%x B=%d PNHt=%d St=%d mask_width_t=%d kv_share=%s '
                             'scratch_slots=4 cb_bytes=%d\n' % (flags, batches, pnht, capacity // 32, width // 32,
                                                                'true' if flags & 2 and batches > 1 else 'false',
                                                                card.CB_BYTES.get((capacity, pnht), 777))).encode())
        share = bool(flags & 0x2) and batches > 1
        tail = bool(flags & 0x1)

        def compute():
            out = query.data.float().clone()
            for b in range(batches):
                row = pages.data[0 if share else b].long()
                keys = k.data[row].float().mean()
                out[0, b] += keys
                if attn_mask is not None:
                    mask = attn_mask.data[b % attn_mask.shape[0], 0].float()
                    seen = mask[:, -256:] if tail else mask
                    out[0, b] += torch.isinf(seen).sum(dim=1, keepdim=True).float() / 1000
            return out.to(torch.bfloat16)

        output = FakeTensor(compute())
        if self.capturing is not None:
            self.traces[self.capturing].append((output, compute))
        self.launches += 1
        return output


class CardMDryRunTests(unittest.TestCase):
    """The card-M script end to end on a fake device with the graft's semantics: every stage-3
    check (E5, E4, N1, N2, N3, N4, N5, the log check) passes, the reference/candidate sha round
    trip matches, and the controls are live - a fake whose twins read their own rows (share
    NOT honoured) fails N1, the program-cache alternation and the trace distinguishability."""

    ARGS = ['--capacities', '768,1024', '--seeds', '0', '--variants', 'normal,zeroq', '--starts', '0,240',
            '--alternations', '2', '--share-alternations', '3', '--trace-replays', '4', '--warmup', '1', '--iters', '2']

    def run_card(self, fake, extra, directory, name, stage_markers):
        import torch
        binary = Path(directory) / ('%s.so' % name)
        binary.write_bytes(b'fake')
        out = Path(directory) / ('%s.json' % name)
        with mock.patch.dict(sys.modules, {'ttnn': fake}), \
                mock.patch.object(card, 'loaded_binary', return_value=(str(binary), stage_markers)), \
                mock.patch.dict(os.environ, {card.SCRATCH_ENV: '1'}), \
                mock.patch.object(card, 'WATCHDOG', card.Watchdog(0)):
            self.assertIs(sys.modules['ttnn'], fake)
            status = card.main(['--out', str(out)] + self.ARGS + extra)
        del torch
        return status, json.loads(out.read_text())

    def test_the_stage_three_flow_passes_and_the_reference_round_trips(self):
        import torch
        stage3 = dict(flags=True, share=True, stage1=False)
        with tempfile.TemporaryDirectory() as directory:
            status, reference = self.run_card(FakeDevice(torch, stage=3), ['--legacy-only'], directory, 'reference',
                                              dict(flags=False, share=False, stage1=False))
            self.assertEqual((status, reference['failures'], reference.get('error')), (0, [], None))
            self.assertEqual(len(reference['share_cases']), 2 * 2 * 2)
            self.assertIn('share/cap768/seed0/normal/start+0/G8/legacy-distinct', reference['legacy_sha256'])
            status, report = self.run_card(FakeDevice(torch, stage=3),
                                           ['--reference', str(Path(directory) / 'reference.json')],
                                           directory, 'candidate', stage3)
        self.assertEqual((report.get('error'), report['failures']), (None, []))
        self.assertEqual(status, 0)
        self.assertTrue(report['passed'])
        self.assertEqual(report['binary']['stage'], 3)
        self.assertEqual(len(report['cases']), 2 * 2 * 2 * 2)
        self.assertEqual(len(report['share_cases']), 2 * 2 * 2)
        self.assertTrue(all(not any(entry['differing'].values()) for entry in report['e4']))
        first = report['share_cases'][0]
        self.assertEqual(first['G8']['n1_tail_share'], dict(differing_from_leader_row=0, twins_differ=[True]))
        self.assertEqual(first['G4B3']['n1_share']['twins_differ'], [True, True])
        self.assertEqual([row['modes'] for row in report['alternation']], ['legacy/tail', 'legacy/tail+share'] * 2)
        self.assertTrue(all(row.get('modes_distinguishable', True) for row in report['alternation']))
        self.assertEqual([(row['mismatch_count'], row['modes_distinguishable']) for row in report['trace']],
                         [(0, [True, True])] * 2)
        self.assertEqual(sorted(report['refusals']['768']), sorted(entry[0] for entry in card.refusals_for(3)))
        self.assertTrue(all(result['refused'] and result['matched'] for result in report['refusals']['768'].values()))
        self.assertEqual(len(report['factory_lines']), len(report['requested_programs']))
        self.assertIn('cap=768 B=2 PNHt=3 mask_width_t=24 flags=0x3', report['requested_programs'])
        self.assertIn('g8_tail_over_g4_legacy_layer', report['user_layer_us']['768'])

    def test_a_graft_whose_twins_read_their_own_rows_fails_the_share_controls(self):
        import torch

        class NoShare(FakeDevice):
            def paged_scaled_dot_product_attention_decode(self, query, k, v, pages, **options):
                config = dict(options['program_config'])
                output = super().paged_scaled_dot_product_attention_decode(query, k, v, pages, **options)
                if config['q_chunk_size'] & 0x2 and (config['q_chunk_size'] & 0xFFFFFF00) == 0x51DEC000:
                    # the twins read their own rows: recompute as if the flag were absent
                    options['program_config'] = dict(config, q_chunk_size=config['q_chunk_size'] & ~0x2)
                    plain = super().paged_scaled_dot_product_attention_decode(query, k, v, pages, **options)
                    output.data = plain.data
                return output

        with tempfile.TemporaryDirectory() as directory:
            status, report = self.run_card(NoShare(torch, stage=3), ['--capacities', '768', '--no-timing'], directory,
                                           'broken', dict(flags=True, share=True, stage1=False))
        self.assertEqual(status, 1)
        text = chr(10).join(report['failures'])
        self.assertIn('twin output equals legacy on its own row', text)
        self.assertIn('is not legacy on the leader row', text)
        self.assertIn('legacy and tail+share agree on distinct page-table rows (control dead)', text)
        self.assertIn('trace modes are not distinguishable', text)

    def test_a_stage_one_binary_skips_share_and_expects_its_refusal(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            status, report = self.run_card(FakeDevice(torch, stage=1), ['--capacities', '768', '--no-timing'], directory,
                                           'stage1', dict(flags=True, share=False, stage1=True))
        self.assertEqual((status, report['failures']), (0, []))
        self.assertEqual((report['binary']['stage'], report['share_section'], report['share_cases']), (1, False, []))
        self.assertEqual(list(report['refusals']['768'])[0], 'share flag (stage 1)')

    def test_the_share_section_needs_the_arms_compact_scratch(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {card.SCRATCH_ENV: '0'}):
                fake = FakeDevice(torch, stage=3)
                binary = Path(directory) / 'x.so'
                binary.write_bytes(b'x')
                out = Path(directory) / 'x.json'
                with mock.patch.dict(sys.modules, {'ttnn': fake}), \
                        mock.patch.object(card, 'loaded_binary',
                                          return_value=(str(binary), dict(flags=True, share=True, stage1=False))), \
                        mock.patch.object(card, 'WATCHDOG', card.Watchdog(0)):
                    self.assertEqual(card.main(['--out', str(out)] + self.ARGS), 1)
                report = json.loads(out.read_text())
        self.assertEqual(report['failures'], ['QWEN_SDPA_TREE_SCRATCH_ROUNDS=1 is required for the G8 legacy calls '
                                              '(the arm sets it; run_card_m.sh does)'])


if __name__ == '__main__':
    unittest.main()
