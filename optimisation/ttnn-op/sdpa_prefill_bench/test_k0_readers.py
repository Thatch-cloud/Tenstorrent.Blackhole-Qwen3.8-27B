"""CPU tests for K0 (sdpa-prefill-share-spec.md 7.1): the probe readers make_k0_readers.py writes to
k0/, the single-file M1_READER mount in run_m1.sh, and the k0_session.sh runner.

No ttnn, no device and no copy of the served reader are needed: every committed k0/ reader reverts
to the f97f5490 base byte for byte, so the base is rebuilt from them (and cross-checked against the
probe-v25 tree when it is on this machine). The shell tests need bash (Git Bash on Windows) and skip
without it."""

import difflib
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_k0_readers as k0  # noqa: E402

NL = chr(10)
K0_DIR = HERE / 'k0'
RUN_M1 = HERE / 'run_m1.sh'
SESSION = HERE / 'k0_session.sh'
PROBE = Path(os.environ.get('QWEN_SDPA_PREFILL_SRC', 'C:/Users/liamb/.claude/jobs/8376c877/tmp/probe-v25/src/device'))
READER_DST = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp'
STARTS = '0,32768,65536,126976'
# Image A': the v117-v127 model-gate image (the K0 session's default, spec 6 ground rules).
IMAGE_A1 = 'sha256:1b9b644549d4409c4fc80e2f92c183e665e7e693c37cccc95b4bb760a6d7d537'
# The caller's environment must not steer the scripts under test.
SCRUB = ('M1_READER', 'IMAGE', 'K64F_IMAGE', 'M1_ARGS', 'M1_DRY_RUN', 'M1_REQUIRE_SOURCES', 'K0_ONLY', 'K0_STARTS',
         'K0_ROUNDS', 'K0_WATCHDOG_S', 'K0_Q4096', 'K0_DRY_RUN', 'K0_DIR', 'RESULTS', 'M1_SRC')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def committed(variant):
    return (K0_DIR / k0.OUTPUT_NAMES[variant]).read_bytes()


def base_bytes():
    """The served reader, rebuilt from the committed k0a (the tests assert it hashes to f97f5490)."""
    return k0.revert_edits(committed('k0a').decode('utf-8'), 'k0a').encode('utf-8')


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


def posix(path):
    return Path(path).as_posix()


def run_bash(script, env=None, args=()):
    full = dict(os.environ)
    for name in SCRUB:
        full.pop(name, None)
    full.update(env or {})
    return subprocess.run([BASH, posix(script), *args], env=full, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=120)


# --- a small structural reader of the C++: CB operations in text order ---------------------------

CALL = re.compile(r'\b(read_paged_chunk_with_padding|read_q_subblock|read_chunk_with_padding|read_chunk_for_forwarding)'
                  r'\s*<[^>]*>\s*\(')
METHOD = re.compile(r'\b(\w+)\.(reserve_back|push_back|pop_front|wait_front|wait)\s*\(')


def strip_comments(text):
    return re.sub(r'//[^\n]*', '', text)


def call_args(text, open_index):
    """Top-level comma-separated arguments of the call whose '(' is at open_index."""
    depth, args, current = 0, [], []
    for index in range(open_index, len(text)):
        char = text[index]
        if char in '(<[{':
            depth += 1
            if depth == 1:
                continue
        elif char in ')>]}':
            depth -= 1
            if depth == 0:
                args.append(''.join(current))
                return [' '.join(arg.split()) for arg in args], index
        elif char == ',' and depth == 1:
            args.append(''.join(current))
            current = []
            continue
        current.append(char)
    raise AssertionError('unbalanced call')


def cb_objects(text):
    return dict(re.findall(r'CircularBuffer (\w+)\((\w+)\);', text))


def cb_ops(body, objects):
    """[(op, cb, count)] in text order. read_paged_chunk_with_padding reserves dst_rows * dst_cols
    (args 6 and 7) and pushes the same (DC:273/314; checked against the probe tree when present)."""
    body = strip_comments(body)
    events = []
    for match in CALL.finditer(body):
        args, _ = call_args(body, match.end() - 1)
        if match.group(1) == 'read_paged_chunk_with_padding':
            count = '%s * %s' % (args[6], args[7])
            events.append((match.start(), [('reserve', args[1], count), ('push', args[1], count)]))
        else:
            events.append((match.start(), [(match.group(1), args[1], '')]))
    names = dict(reserve_back='reserve', push_back='push', pop_front='pop', wait_front='wait', wait='wait')
    for match in METHOD.finditer(body):
        args, _ = call_args(body, match.end() - 1)
        events.append((match.start(), [(names[match.group(2)], objects.get(match.group(1), match.group(1)),
                                        ','.join(args))]))
    return [op for _, ops in sorted(events, key=lambda pair: pair[0]) for op in ops]


def k_loop(text):
    start = text.index('            for (uint32_t k_chunk = k_loop_start; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {')
    return text[start:text.index('            }  // close k_chunk', start)]


IF_LINE = '                        if %s {' % k0.READ_PREDICATE


def select(text, take):
    """Every K0 `if (qwen_k0_full_kv || k_chunk < 2) {THEN} else {ELSE}` replaced by one branch."""
    out, lines, index = [], text.split(NL), 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(IF_LINE):
            pad = line[:len(line) - len(line.lstrip())]
            middle = lines.index(pad + '} else {', index)
            end = lines.index(pad + '}', middle)
            out.extend(lines[index + 1:middle] if take == 'then' else lines[middle + 1:end])
            index = end + 1
            continue
        out.append(line)
        index += 1
    return NL.join(out)


def added_lines(variant):
    base = base_bytes().decode('utf-8').split(NL)
    new = committed(variant).decode('utf-8').split(NL)
    out = []
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, base, new, autojunk=False).get_opcodes():
        if tag in ('replace', 'insert'):
            out.extend(new[j1:j2])
    return out


class GeneratorTests(unittest.TestCase):
    def test_the_committed_readers_hash_to_the_recorded_outputs(self):
        self.assertEqual(set(k0.VARIANTS), {'k0a', 'k0b4', 'k0b32', 'k0c'})
        for variant in k0.VARIANTS:
            with self.subTest(variant=variant):
                data = committed(variant)
                self.assertEqual(sha(data), k0.OUTPUTS[variant])
                self.assertNotIn(b'\r', data)
                data.decode('utf-8')

    def test_every_committed_reader_reverts_to_the_served_base(self):
        for variant in k0.VARIANTS:
            with self.subTest(variant=variant):
                base = k0.revert_edits(committed(variant).decode('utf-8'), variant).encode('utf-8')
                self.assertEqual(sha(base), k0.BASE_SHA)
                self.assertTrue(k0.BASE_SHA.startswith('f97f5490'))

    def test_the_readers_regenerate_from_the_base_byte_for_byte(self):
        outputs = k0.build_all(base_bytes())
        for variant in k0.VARIANTS:
            self.assertEqual(outputs[variant], committed(variant), variant)

    def test_the_probe_tree_reader_is_the_base(self):
        reader = PROBE / 'kernels/dataflow/reader_interleaved.cpp'
        if not reader.is_file():
            self.skipTest('no probe-v25 tree at %s' % PROBE)
        self.assertEqual(reader.read_bytes(), base_bytes())

    def test_a_wrong_input_sha_is_refused_and_nothing_is_written(self):
        base = base_bytes()
        for bad in (base + b' ', base.replace(b'barrier_threshold', b'barrier_threshold ', 1), b'', base.replace(NL.encode(), b'\r\n')):
            with self.assertRaisesRegex(k0.Refusal, 'not the served reader f97f5490'):
                k0.build(bad, 'k0a')
        with tempfile.TemporaryDirectory() as directory:
            reader = Path(directory) / 'reader_interleaved.cpp'
            reader.write_bytes(base + b'// drift' + NL.encode())
            out = Path(directory) / 'out'
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = k0.main(['--reader', str(reader), '--out', str(out)])
            self.assertEqual(status, 2)
            self.assertFalse(out.exists())
            self.assertIn('sha256=%s' % sha(reader.read_bytes()), stdout.getvalue())   # the input sha is printed
            self.assertIn('REFUSING', stderr.getvalue())

    def test_anchor_drift_is_refused_even_past_the_sha_check(self):
        text = base_bytes().decode('utf-8')
        with self.assertRaisesRegex(k0.Refusal, 'starts at line 14, the spec cites R:13'):
            k0.apply_edits(NL + text, 'k0a')
        with self.assertRaisesRegex(k0.Refusal, 'occurs 2 times'):
            k0.apply_edits(text + k0.K_CALL, 'k0b4')
        with self.assertRaisesRegex(k0.Refusal, 'occurs 0 times'):
            k0.apply_edits(text.replace(k0.V_CALL, k0.V_CALL.replace('skip_src_cols);', 'skip_src_cols );')), 'k0c')

    def test_the_cited_served_lines(self):
        lines = base_bytes().decode('utf-8').split(NL)
        self.assertEqual(lines[12], '#include "dataflow_common.hpp"')                                  # R:13
        self.assertIn('barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();', lines[208])  # R:209
        self.assertIn('read_paged_chunk_with_padding<NKH, block_size_t, DHt>(', lines[425])             # R:426
        self.assertEqual(lines[435].strip(), 'barrier_threshold,')                                      # R:436
        self.assertIn('read_paged_chunk_with_padding<NVH, block_size_t, head_dim>(', lines[618])        # R:619
        self.assertEqual(lines[628].strip(), 'barrier_threshold,')                                      # R:629

    def test_main_writes_checks_and_reports_every_output_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = Path(directory) / 'reader_interleaved.cpp'
            reader.write_bytes(base_bytes())
            out = Path(directory) / 'k0'
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(k0.main(['--reader', str(reader), '--out', str(out)]), 0)
                self.assertEqual(k0.main(['--reader', str(reader), '--check', str(out)]), 0)
            for variant in k0.VARIANTS:
                self.assertEqual((out / k0.OUTPUT_NAMES[variant]).read_bytes(), committed(variant))
                self.assertIn('sha256=%s' % k0.OUTPUTS[variant], stdout.getvalue())
            (out / 'reader_k0c.cpp').write_bytes(committed('k0a'))
            with redirect_stdout(StringIO()):
                self.assertEqual(k0.main(['--reader', str(reader), '--check', str(out)]), 1)


class SpanTests(unittest.TestCase):
    """Each variant differs from served only inside the intended spans."""

    # served 0-based line ranges an edit may touch: insert after R:13, insert after R:209, R:426-439, R:619-632
    INSERTS = {13, 209}
    SPANS = ((425, 439), (618, 632))

    def opcodes(self, variant):
        base = base_bytes().decode('utf-8').split(NL)
        new = committed(variant).decode('utf-8').split(NL)
        return base, new, [op for op in difflib.SequenceMatcher(None, base, new, autojunk=False).get_opcodes()
                           if op[0] != 'equal']

    def test_every_change_lies_in_an_intended_span(self):
        for variant in k0.VARIANTS:
            with self.subTest(variant=variant):
                _, _, ops = self.opcodes(variant)
                self.assertTrue(ops)
                for tag, i1, i2, _, _ in ops:
                    if tag == 'insert':
                        self.assertTrue(i1 in self.INSERTS or any(lo <= i1 <= hi for lo, hi in self.SPANS), (tag, i1))
                    else:
                        self.assertTrue(any(lo <= i1 and i2 <= hi for lo, hi in self.SPANS), (tag, i1, i2))

    def test_k0c_changes_only_the_kv_barrier_threshold(self):
        base, new, ops = self.opcodes('k0c')
        replaced = [(i1, base[i1:i2], new[j1:j2]) for tag, i1, i2, j1, j2 in ops if tag == 'replace']
        self.assertEqual([i1 + 1 for i1, _, _ in replaced], [436, 629])
        for _, old, now in replaced:
            self.assertEqual([line.strip() for line in old], ['barrier_threshold,'])
            self.assertEqual([line.strip() for line in now], ['qwen_k0_kv_bt,'])
        inserted = [line for tag, _, _, j1, j2 in ops if tag == 'insert' for line in new[j1:j2]]
        code = [line.strip() for line in inserted if not line.strip().startswith('//')]
        self.assertEqual(code, ['constexpr uint32_t qwen_k0_kv_bt = 32;'])
        self.assertFalse([op for op in ops if op[0] == 'delete'])

    def test_added_code_never_waits_and_never_touches_the_noc(self):
        forbidden = re.compile(r'\b(wait\w*|\w*_barrier\w*|Semaphore\w*|noc|noc_\w+|get_semaphore|invalidate_l1_cache|'
                               r'async_\w+|get_arg_val|get_compile_time_arg_val)\b')
        for variant in k0.VARIANTS:
            with self.subTest(variant=variant):
                code = [strip_comments(line) for line in added_lines(variant)]
                self.assertEqual([line for line in code if forbidden.search(line)], [])
                methods = {pair for line in code for pair in re.findall(r'\b(\w+)\.(\w+)\s*\(', line)}
                self.assertLessEqual(methods, {('cb_k', 'reserve_back'), ('cb_k', 'push_back'),
                                               ('cb_v', 'reserve_back'), ('cb_v', 'push_back')})
                calls = set(re.findall(r'(?<![\w.])(\w+)\s*(?:<[^<>;]*>)?\s*\(', ' '.join(code))) - {'if'}
                self.assertLessEqual(calls, {'read_paged_chunk_with_padding'})

    def test_the_variant_predicates_and_cadences(self):
        texts = {variant: committed(variant).decode('utf-8') for variant in k0.VARIANTS}
        self.assertIn('    const bool qwen_k0_full_kv = false;' + NL, texts['k0a'])
        self.assertIn('    const uint32_t qwen_k0_kv_bt = barrier_threshold;' + NL, texts['k0a'])
        self.assertIn('    const bool qwen_k0_full_kv = ((core_id / 8) % 6) == 0;' + NL, texts['k0b4'])
        self.assertIn('    const uint32_t qwen_k0_kv_bt = barrier_threshold;' + NL, texts['k0b4'])
        self.assertIn('    const bool qwen_k0_full_kv = ((core_id / 8) % 6) == 0;' + NL, texts['k0b32'])
        self.assertIn('    const uint32_t qwen_k0_kv_bt = qwen_k0_full_kv ? 32u : barrier_threshold;' + NL, texts['k0b32'])
        self.assertNotIn('qwen_k0_full_kv', texts['k0c'])
        for variant in ('k0a', 'k0b4', 'k0b32'):
            self.assertEqual(texts[variant].count(IF_LINE), 2, variant)          # K and V, the paged branch only
        for variant in k0.VARIANTS:
            text = texts[variant]
            self.assertEqual(text.count('qwen_k0_kv_bt,'), 2, variant)           # the K and V read calls only
            q_calls = re.findall(r'read_q_subblock<q_tile_bytes>\([^;]*;', text)
            self.assertTrue(q_calls and all('barrier_threshold);' in call for call in q_calls))   # Q keeps 4
            self.assertEqual(text.count('[QWEN-SDPA-K0] %s:' % variant), 2)


class ProtocolTests(unittest.TestCase):
    """Per k chunk, every variant keeps the served CB protocol: the same reserve/push sequence (cb,
    tile count) whichever branch a core takes, and no wait the served reader does not have."""

    def test_the_k_loop_cb_sequence_equals_served_on_both_branches(self):
        base = base_bytes().decode('utf-8')
        objects = cb_objects(base)
        self.assertEqual(objects['cb_k'], 'cb_k_in')
        self.assertEqual(objects['cb_v'], 'cb_v_in')
        served = cb_ops(k_loop(base), objects)
        self.assertIn(('reserve', 'cb_k_in', 'Sk_chunk_t * DHt'), served)
        self.assertIn(('push', 'cb_v_in', 'Sk_chunk_t * vDHt'), served)
        for variant in k0.VARIANTS:
            text = committed(variant).decode('utf-8')
            self.assertEqual(cb_objects(text), objects)
            for take in ('then', 'else'):
                with self.subTest(variant=variant, branch=take):
                    self.assertEqual(cb_ops(k_loop(select(text, take)), objects), served)

    def test_the_paged_served_path_order_is_k_then_q_subblocks_then_v(self):
        text = committed('k0a').decode('utf-8')
        objects = cb_objects(text)
        for take in ('then', 'else'):
            body = strip_comments(k_loop(select(text, take)))
            k_paged = body[body.index('if constexpr (is_chunked) {'):body.index('} else {', body.index('if constexpr (is_chunked) {'))]
            v_start = body.index('if constexpr (is_chunked) {', body.index('// Read V chunk from DRAM') if '// Read V chunk from DRAM' in body else body.index('cb_v_start_address = 0;'))
            v_paged = body[v_start:body.index('} else {', v_start)]
            self.assertEqual(cb_ops(k_paged, objects), [('reserve', 'cb_k_in', 'Sk_chunk_t * DHt'), ('push', 'cb_k_in', 'Sk_chunk_t * DHt')])
            self.assertEqual(cb_ops(v_paged, objects), [('reserve', 'cb_v_in', 'Sk_chunk_t * vDHt'), ('push', 'cb_v_in', 'Sk_chunk_t * vDHt')])
            self.assertLess(body.index(k_paged), body.index('read_q_subblock<q_tile_bytes>('))
            self.assertLess(body.index('read_q_subblock<q_tile_bytes>('), body.index(v_paged))

    def test_the_skip_branch_counts_are_the_served_calls_dst_rows_times_dst_cols(self):
        for variant in ('k0a', 'k0b4', 'k0b32'):
            text = strip_comments(committed(variant).decode('utf-8'))
            for cb, served_call in (('cb_k', k0.K_CALL), ('cb_v', k0.V_CALL)):
                args, _ = call_args(served_call, served_call.index('('))
                count = '%s * %s' % (args[6], args[7])
                self.assertEqual(text.count('%s.reserve_back(%s);' % (cb, count)), 1, (variant, cb))
                self.assertEqual(text.count('%s.push_back(%s);' % (cb, count)), 1, (variant, cb))
                self.assertEqual(args[1], {'cb_k': 'cb_k_in', 'cb_v': 'cb_v_in'}[cb])

    def test_the_read_helper_reserves_and_pushes_dst_rows_times_dst_cols(self):
        dc = PROBE / 'kernels/dataflow/dataflow_common.hpp'
        if not dc.is_file():
            self.skipTest('no probe-v25 tree at %s' % PROBE)
        data = dc.read_bytes()
        self.assertEqual(sha(data), '554a0b282d2a36c7b129eef04df67a5becbb14f98220246e43e47fdd56118aa6')
        text = data.decode('utf-8')
        body = text[text.index('void read_paged_chunk_with_padding('):text.index('void copy_tile(')]
        signature, _ = call_args(body, body.index('('))
        names = [arg.split()[-1] for arg in signature]
        self.assertEqual(names[1], 'cb_id')
        self.assertEqual(names[6:10], ['dst_rows', 'dst_cols', 'tile_bytes', 'barrier_threshold'])
        self.assertIn('const uint32_t num_tiles = dst_rows * dst_cols;', body)
        self.assertEqual(body.count('cb.reserve_back(num_tiles);'), 1)
        self.assertEqual(body.count('cb.push_back(num_tiles);'), 1)
        self.assertLess(body.index('cb.reserve_back(num_tiles);'), body.index('cb.push_back(num_tiles);'))
        self.assertIn('return ((512 / num_readers) * (1024 + 128)) / tile_bytes;', text)

    def test_core_id_is_runtime_arg_7_and_barrier_threshold_is_4(self):
        text = base_bytes().decode('utf-8')
        head = text[:text.index('const uint32_t core_id = get_arg_val<uint32_t>(argidx++);')]
        self.assertEqual(head[head.index('uint32_t argidx = 0;'):].count('get_arg_val<uint32_t>(argidx++)'), 7)
        self.assertEqual(((512 // 110) * (1024 + 128)) // 1088, 4)          # bf8 Q tile, 110 cores

    def test_the_injector_predicate_picks_one_core_per_g6_group(self):
        injectors = [i for i in range(96) if (i // 8) % 6 == 0]
        self.assertEqual(injectors, list(range(8)) + list(range(48, 56)))
        groups = {(g, m): [48 * g + 8 * s + m for s in range(6)] for g in range(2) for m in range(8)}
        for members in groups.values():
            self.assertEqual(len([i for i in members if i in injectors]), 1)
        self.assertEqual(len({(i // 48, i % 8) for i in injectors}), 16)

    def test_k0a_fills_both_slots_with_real_data_before_any_stale_push(self):
        """The stale slot is always real K/V from an earlier read (finite bf8), for every chunk_start."""
        for c in range(0, 1100):
            for m in range(8):
                pushes = [k < 2 for q in (m, 15 - m) for k in range(c + q + 1)]
                first_stale = pushes.index(False) if False in pushes else len(pushes)
                slots = {index % 2 for index in range(first_stale) if pushes[index]}
                self.assertEqual(slots, {0, 1}, (c, m))


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def dry(self, reader=None, extra=None):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(M1_DRY_RUN='1', M1_SRC=posix(HERE), RESULTS=posix(directory), M1_ARGS='--arms baseline --sha')
            if reader is not None:
                env['M1_READER'] = reader
            env.update(extra or {})
            return run_bash(RUN_M1, env)

    @staticmethod
    def argv(stdout):
        lines = [line for line in stdout.splitlines() if line.startswith('### argv: ')]
        assert len(lines) == 1, stdout
        return shlex.split(lines[0][len('### argv: '):])

    @staticmethod
    def mounts(argv):
        return [argv[index + 1] for index, word in enumerate(argv) if word == '--mount']

    def test_run_m1_parses(self):
        result = subprocess.run([BASH, '-n', posix(RUN_M1)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_m1_reader_mounts_exactly_one_file_read_only_over_the_image_reader(self):
        result = self.dry(posix(K0_DIR / 'reader_k0a.cpp'))
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv(result.stdout)
        readers = [mount for mount in self.mounts(argv) if 'dst=%s' % READER_DST in mount.split(',')]
        self.assertEqual(len(readers), 1)
        fields = readers[0].split(',')
        self.assertEqual(fields[0], 'type=bind')
        self.assertTrue(fields[1].startswith('src=') and fields[1].endswith('/k0/reader_k0a.cpp'))
        self.assertEqual(fields[3], 'readonly')
        self.assertFalse([word for word in argv if word in ('-v', '--volume')])
        self.assertIn('sha256=%s' % k0.OUTPUTS['k0a'], result.stdout)
        script = argv[argv.index('-c') + 1]
        self.assertIn('.reader.sha256 || {', script)
        self.assertIn('exit 97', script)
        self.assertLess(script.index('sha256sum /opt/tt-metal'), script.index('exec python3 -B /bench/sdpa_prefill_bench.py --out'))
        self.assertEqual(argv[argv.index('-c') + 2:], ['m1', '--arms', 'baseline', '--sha'])

    def test_without_m1_reader_nothing_is_mounted_over_the_reader(self):
        result = self.dry()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv(result.stdout)
        self.assertEqual([m for m in self.mounts(argv) if 'reader_interleaved' in m], [])
        script = argv[argv.index('-c') + 1]
        self.assertNotIn('.reader.sha256', script)
        self.assertIn('sha256sum -c --quiet /results/m1-', script)
        self.assertIn('-e', argv)
        self.assertIn('M1_REQUIRE_SOURCES=0', argv)

    def test_the_mounted_bench_is_hashed_in_the_container_and_must_match_the_host_file(self):
        result = self.dry(posix(K0_DIR / 'reader_k0c.cpp'))
        self.assertEqual(result.returncode, 0, result.stderr)
        host = sha((HERE / 'sdpa_prefill_bench.py').read_bytes())
        self.assertIn('### bench %s/sdpa_prefill_bench.py sha256=%s' % (posix(HERE), host), result.stdout)
        argv = self.argv(result.stdout)
        script = argv[argv.index('-c') + 1]
        printed = script.index('/bench/sdpa_prefill_bench.py || echo')          # in the printed sha256sum list
        self.assertLess(script.index('sha256sum /opt/tt-metal'), printed)
        check = script.index('.bench.sha256 || {')
        self.assertIn("the in-container bench is not the host file'; exit 97; }", script[check:])
        self.assertLess(check, script.index('exec python3 -B /bench/sdpa_prefill_bench.py'))
        text = RUN_M1.read_text(encoding='utf-8')
        self.assertIn('"$bench_sha  /bench/sdpa_prefill_bench.py" > "$R/m1-$stamp.bench.sha256"', text)

    def test_a_missing_image_is_refused_before_the_launch_line(self):
        text = RUN_M1.read_text(encoding='utf-8')
        refusal = text.index('docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "refusing: image')
        self.assertLess(text.index('holds a Tenstorrent device'), refusal)
        self.assertLess(refusal, text.index('echo "### M1 $stamp node='))
        self.assertLess(text.index('echo "### M1 $stamp node='), text.index('timeout -k 30 1800 "${argv[@]}"'))

    def test_a_directory_or_a_missing_file_is_refused(self):
        for reader in (posix(K0_DIR), posix(K0_DIR / 'no_such_reader.cpp')):
            with self.subTest(reader=reader):
                result = self.dry(reader)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('refusing', result.stderr)
                self.assertNotIn('### argv:', result.stdout)

    def test_the_seven_served_shas_are_the_probe_v25_sources(self):
        text = RUN_M1.read_text(encoding='utf-8')
        shas = re.findall(r'^  "([0-9a-f]{64})  \$', text, flags=re.M)
        self.assertEqual([value[:8] for value in shas],
                         ['fd8c0676', 'f97f5490', 'a3f48af8', '3fb5da24', '554a0b28', '43a24c46', '2a0959ff'])
        self.assertEqual(shas[1], k0.BASE_SHA)
        self.assertEqual(text.count('dst=$READER_DST,readonly'), 1)
        self.assertNotIn(chr(13), text)
        if PROBE.is_dir():
            names = ['sdpa_program_factory.cpp', 'kernels/dataflow/reader_interleaved.cpp', 'kernels/compute/sdpa.cpp',
                     'kernels/compute/compute_common.hpp', 'kernels/dataflow/dataflow_common.hpp',
                     'kernels/dataflow/chain_link.hpp', 'kernels/dataflow/writer_interleaved.cpp']
            self.assertEqual([sha((PROBE / name).read_bytes()) for name in names], shas)


FAKE_RUNNER = '''#!/usr/bin/env bash
# stand-in for run_m1.sh: a canned bench table, M1 SHA and M1 KERNEL_ELF lines per mounted reader
v=stock
if [ -n "${M1_READER:-}" ]; then v=$(basename "$M1_READER" .cpp); v=${v#reader_}
else case " $M1_ARGS " in *" --coords-out "*|*q256_4096*) ;; *) v=stock2 ;; esac; fi
case $v in
  stock) slope=0.3400; sha=aaaa; elf=e1e1 ;;
  stock2) slope=${FAKE_STOCK2_SLOPE:-0.3450}; sha=${FAKE_STOCK2_SHA:-aaaa}; elf=${FAKE_STOCK2_ELF:-e1e1} ;;
  k0a) slope=0.2000; sha=${FAKE_K0A_SHA:-bbbb}; elf=e2e2 ;;
  k0b4) slope=0.3300; sha=cccc; elf=e3e3 ;;
  k0b32) slope=0.2050; sha=dddd; elf=e4e4 ;;
  k0c) slope=0.3350; sha=${FAKE_K0C_SHA:-aaaa}; elf=${FAKE_K0C_ELF:-e5e5} ;;
esac
if [ "$v" = "${FAKE_PRELAUNCH:-none}" ]; then echo "refusing: container /x holds a Tenstorrent device" >&2; exit 1; fi
echo "### M1 20260923T120000 node=/dev/tenstorrent/2 image=${IMAGE:7:12} reader=${M1_READER:-served}"
echo "### fake run $v image=$IMAGE args: $M1_ARGS"
if [ "$v" = "${FAKE_HANG:-none}" ]; then echo "WATCHDOG: baseline@0 warmup did not return within 900 s; exit 3 (reset card M before the next run)"; exit 3; fi
if [ "$v" = "${FAKE_FAIL:-none}" ]; then echo "ERROR baseline@0: RuntimeError: device fault"; echo "M1 VERDICT: incomplete | fake"; exit 1; fi
echo "arm           rows cores        @0k       @32k       @64k      @124k   ms/1k keys  per 2048 rows"
echo "baseline      2048    96      1.228     12.370     23.512     44.400       $slope         $slope"
for s in 0 32768 65536 126976; do echo "M1 SHA baseline@$s $sha$s stable=1"; done
echo "M1 KERNEL_ELF reader_interleaved $elf files=1"
echo "M1 VERDICT: incomplete | fake"
exit 0
'''


@unittest.skipUnless(BASH, 'bash not found')
class SessionTests(unittest.TestCase):
    def session(self, extra=None):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / 'fake_run_m1.sh'
            fake.write_bytes(FAKE_RUNNER.encode('utf-8'))
            env = dict(M1_SRC=posix(HERE), RESULTS=posix(Path(directory) / 'results'), K0_RUN_M1=posix(fake))
            env.update(extra or {})
            return run_bash(SESSION, env)

    @staticmethod
    def fake_runs(out):
        return re.findall(r'^### fake run (\S+)', out, flags=re.M)

    def test_the_session_parses_and_its_reader_shas_are_the_recorded_outputs(self):
        result = subprocess.run([BASH, '-n', posix(SESSION)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = SESSION.read_text(encoding='utf-8')
        self.assertEqual(dict(re.findall(r'^  \[(k0\w+)\]=([0-9a-f]{64})$', text, flags=re.M)), k0.OUTPUTS)
        self.assertIn('RUNS=(stock k0a k0b4 k0b32 k0c stock2)', text)
        self.assertIn('STARTS=${K0_STARTS:-%s}' % STARTS, text)
        self.assertIn('ROUNDS=${K0_ROUNDS:-7}', text)
        self.assertIn('IMAGE=${IMAGE:-$IMAGE_A1}', text)
        self.assertIn('IMAGE_A1=%s' % IMAGE_A1, text)
        self.assertNotIn(chr(13), text)
        # The session never resets a card itself: tt-smi appears only in comments and printed hints.
        for line in text.splitlines():
            if 'tt-smi' in line:
                self.assertTrue(line.lstrip().startswith(('#', 'echo ')), line)

    def test_the_image_a1_default_is_the_model_gate_image(self):
        gate = HERE.parents[2] / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml'
        if not gate.is_file():
            self.skipTest('no gate workflow at %s' % gate)
        self.assertIn('*-v126|*-v127) image=%s' % IMAGE_A1, gate.read_text(encoding='utf-8'))

    def test_the_real_runner_dry_runs_stock_the_four_variants_then_stock2(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_bash(SESSION, dict(M1_SRC=posix(HERE), RESULTS=posix(directory), K0_DRY_RUN='1'))
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = re.findall(r'^### K0 run (\S+): reader=(\S+) image=(\S+) args: (.*)$', result.stdout, flags=re.M)
        self.assertEqual([label for label, _, _, _ in runs], ['stock', 'k0a', 'k0b4', 'k0b32', 'k0c', 'stock2'])
        argvs = [shlex.split(line[len('### argv: '):]) for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        self.assertEqual(len(argvs), 6)
        for (label, reader, image, args), argv in zip(runs, argvs):
            with self.subTest(run=label):
                self.assertEqual(image, IMAGE_A1)
                self.assertEqual(argv[argv.index('--entrypoint') + 2], IMAGE_A1)     # what docker is given
                self.assertIn('--arms baseline --starts %s --rounds 7 --sha --watchdog-s 120 --no-fallback-scalar '
                              '--kernel-elf reader_interleaved' % STARTS, args)
                mounts = [argv[i + 1] for i, word in enumerate(argv) if word == '--mount' and 'reader_interleaved' in argv[i + 1]]
                self.assertIn('M1_REQUIRE_SOURCES=1', argv)
                if label in ('stock', 'stock2'):
                    self.assertEqual(reader, 'served')
                    self.assertEqual(mounts, [])
                    self.assertEqual('--coords-out /results/cardm_worker_coords-' in args, label == 'stock')
                else:
                    self.assertEqual(len(mounts), 1)
                    self.assertTrue(mounts[0].split(',')[1].endswith('/k0/reader_%s.cpp' % label))
                    self.assertNotIn('--coords-out', args)

    def test_an_explicit_image_reaches_every_run(self):
        other = 'sha256:' + '0648ca9a' * 8
        with tempfile.TemporaryDirectory() as directory:
            result = run_bash(SESSION, dict(M1_SRC=posix(HERE), RESULTS=posix(directory), K0_DRY_RUN='1', IMAGE=other,
                                            K0_ONLY='stock k0c'))
        self.assertEqual(result.returncode, 0, result.stderr)
        argvs = [shlex.split(line[len('### argv: '):]) for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        self.assertEqual([argv[argv.index('--entrypoint') + 2] for argv in argvs], [other, other])

    def test_the_summary_from_six_clean_runs(self):
        result = self.session()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        out = result.stdout
        self.assertEqual(self.fake_runs(out), ['stock', 'k0a', 'k0b4', 'k0b32', 'k0c', 'stock2'])
        self.assertIn('image %s' % IMAGE_A1, out)
        self.assertIn('stock drift across the session: stock2/stock slope = 1.015 (|drift| 1.5%)', out)
        self.assertIn('stock2 == stock at 4/4 starts (0 missing): reproducible across processes = yes', out)
        self.assertIn('t_c = K0a slope x 0.064 ms = 12.80 us per step   (served step 21.76 us at slope 0.340)' + NL, out)
        self.assertIn('K0b4/K0a = 1.650   K0b32/K0a = 1.025   K0b4/stock = 0.971   K0c/stock = 0.985', out)
        self.assertIn('K0c == stock byte for byte at 4/4 starts (0 missing): yes', out)
        for variant in ('k0a', 'k0b4', 'k0b32'):
            self.assertIn('%s ran (output differs from stock at every start): yes   (4 differ, 0 EQUAL, 0 missing of 4)' % variant, out)
        self.assertIn("k0c reader compiled (its ELF != stock's, stock's == stock2's): yes" + NL, out)
        self.assertIn('K-1/K-2 clear: t_c < 15 us', out)
        self.assertIn('K-3 clear', out)
        self.assertIn('K-4 (read): K0b4/stock = 0.971', out)
        self.assertIn('K-5 clear', out)
        self.assertNotIn('NOT DECISIVE', out)
        self.assertNotIn('withheld', out)

    def test_the_session_stops_at_a_watchdog_and_prints_the_recovery(self):
        result = self.session(dict(FAKE_HANG='k0b4'))
        self.assertEqual(result.returncode, 3)
        out = result.stdout
        self.assertEqual(self.fake_runs(out), ['stock', 'k0a', 'k0b4'])
        self.assertIn('K0 STOPPED at k0b4: a hang (exit 3): WATCHDOG: baseline@0 warmup', out)
        self.assertIn('tt-smi -r', out)
        smoke = [line for line in out.splitlines() if 'stock smoke' in line]
        self.assertEqual(len(smoke), 1)
        self.assertIn('IMAGE=%s M1_SRC=%s RESULTS=' % (IMAGE_A1, posix(HERE)), smoke[0])
        self.assertIn('--no-fallback-scalar', smoke[0])
        self.assertIn('kcache-m1-20260923T120000', out)          # the warmup hint names the run's cache
        self.assertIn('K0 SUMMARY', out)

    def test_any_other_failure_after_launch_stops_the_session_with_the_recovery(self):
        result = self.session(dict(FAKE_FAIL='k0a'))
        self.assertEqual(result.returncode, 4)
        out = result.stdout
        self.assertEqual(self.fake_runs(out), ['stock', 'k0a'])
        self.assertIn('K0 STOPPED at k0a: exit 1 after a container ran on card M', out)
        self.assertIn('tt-smi -r', out)
        self.assertIn('K0 SUMMARY', out)
        self.assertNotIn('continuing', out)

    def test_a_failure_before_launch_stops_without_a_reset(self):
        result = self.session(dict(FAKE_PRELAUNCH='k0b32'))
        self.assertEqual(result.returncode, 1)
        out = result.stdout
        self.assertEqual(self.fake_runs(out), ['stock', 'k0a', 'k0b4'])
        self.assertIn('K0 STOPPED at k0b32: exit 1 before any container ran on card M', out)
        self.assertNotIn('tt-smi', out)

    def test_kill_rules_are_withheld_when_k0a_is_not_shown_to_have_run(self):
        result = self.session(dict(FAKE_K0A_SHA='aaaa'))       # k0a's output equals stock: its reader never ran
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        out = result.stdout
        self.assertIn('k0a ran (output differs from stock at every start): NO', out)
        self.assertIn('UNPROVEN: k0a is not shown to have run', out)
        self.assertIn('K-1/K-2 withheld: k0a ran = NO', out)
        self.assertIn('K-3 withheld: k0a ran = NO, k0b32 ran = yes', out)
        self.assertIn('K-4 withheld', out)
        for verdict in ('K-1 TRIGGERED', 'K-2 TRIGGERED', 'K-1/K-2 clear', 'K-3 TRIGGERED', 'K-3 clear'):
            self.assertNotIn(verdict, out)
        self.assertIn('K-5 clear', out)                        # K0c's own proof stands

    def test_an_equal_start_outranks_a_missing_one(self):
        result = self.session(dict(FAKE_K0A_SHA='aaaa', K0_STARTS=STARTS + ',1024'))   # 1024: no sha anywhere
        out = result.stdout
        self.assertIn('k0a ran (output differs from stock at every start): NO   (0 differ, 4 EQUAL, 1 missing of 5)', out)
        self.assertIn('k0b4 ran (output differs from stock at every start): incomplete   (4 differ, 0 EQUAL, 1 missing of 5)', out)
        self.assertIn('K-5 incomplete', out)

    def test_k5_needs_k0c_proven_and_stock_rerun(self):
        cases = (
            (dict(K0_ONLY='stock k0c'), "K-5 unproven: K0c equals stock, but k0c reader compiled = unproven, "
                                         "stock reproducible across processes = incomplete"),
            (dict(FAKE_K0C_ELF='e1e1'), "K-5 unproven: K0c equals stock, but k0c reader compiled = NO"),
            (dict(FAKE_STOCK2_ELF='e9e9'), "K-5 unproven: K0c equals stock, but k0c reader compiled = unproven"),
            (dict(FAKE_K0C_SHA='ffff'), 'K-5 TRIGGERED'),
            (dict(FAKE_K0C_SHA='ffff', FAKE_STOCK2_SHA='9999'), 'K-5 inconclusive'),
            (dict(FAKE_K0C_SHA='ffff', K0_ONLY='stock k0a k0b4 k0b32 k0c'), 'K-5 provisional TRIGGER'),
        )
        for extra, expected in cases:
            with self.subTest(**extra):
                out = self.session(extra).stdout
                self.assertIn(expected, out)
                self.assertNotIn('K-5 clear', out)
        out = self.session(dict(FAKE_K0C_ELF='e1e1')).stdout
        self.assertIn("the mounted reader was not compiled", out)

    def test_a_ratio_within_the_stock_drift_of_its_threshold_is_not_decisive(self):
        out = self.session(dict(FAKE_STOCK2_SLOPE='0.3700')).stdout      # 8.8% drift; K0b32/K0a 1.025 vs 1.10
        self.assertIn('K-3 clear', out)
        self.assertIn('NOT DECISIVE: 1.025 is within the stock drift (8.8%) of the 1.10 threshold', out)

    def test_a_mismatched_reader_is_refused_before_any_run(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'k0'
            shutil.copytree(K0_DIR, bad)
            (bad / 'reader_k0b32.cpp').write_bytes(committed('k0b4'))
            result = self.session(dict(K0_DIR=posix(bad)))
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing', result.stderr)
        self.assertNotIn('### fake run', result.stdout)


if __name__ == '__main__':
    unittest.main()
