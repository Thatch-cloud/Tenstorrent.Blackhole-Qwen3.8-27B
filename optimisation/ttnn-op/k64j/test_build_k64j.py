"""CPU checks for build_k64j.sh, the K64j graft's build on the rig host through ttbuild (the card-B CI route).

  - the contract: every recorded sha in the script is the generators' (apply_factory_k64j, make_k64j_kernels, the
    K64i generators) and build_k64i.sh's (the prefill factory and reader, the stock decode sources), the K64j
    literals it greps for are apply_factory_k64j's, its kernel list is make_k64j_kernels', its image is the card
    tests' (k64j_probe/run_card_b.sh);
  - the text: LF, bash -n, LC_ALL=C, the canonical qual_card.sh block byte for byte followed by qual_card_select, no
    serving board id outside it, no device (no docker run, no --device, no qual_card_resolve);
  - the dry run (needs bash): a fake docker first on PATH that logs and exits 1 is never invoked, nothing is created
    under HOME or at the graft, and the printed commands are the build's in order (the K64j factory and the four
    kernels staged into ttbuild, ninja, the assembly, the restore, the rebuild, the move into place); the refusals
    (a serving QUAL_CARD, CARD_B_ARGS, a graft over the base or inside the checkout);
  - the full run (needs bash; on Windows only with K64J_BUILD_E2E=1, its process spawns are slow there) against a
    fake docker that emulates ttbuild as a directory tree, a fake ninja that links a .so from the staged factories'
    QWEN literals, a fake K64i graft from the committed K64i kernels and the vendored fixtures, and a fake P8 image:
    the build exits 0, prints K64J_TTNNCPP_SHA256, writes a verifying manifest, the four K64j kernels and the
    runtime-extent literal, saves the bases from ttbuild and leaves ttbuild's tree byte for byte as it found it;
    a K64i graft whose manifest does not verify, a ttbuild on the unpatched 05708e6d factory, a link that loses a
    K64i QWEN string and a failed ninja each stop with their FAIL line, ttbuild restored.

    py -3.11 -B -m unittest test_build_k64j      (from this directory)
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
ROOT = HERE.parents[2]
CI = ROOT / 'scripts' / 'ci'
for _path in (str(HERE), str(OPS / 'sdpa_decode_slice'), str(OPS / 'sdpa_decode_qwen'), str(OPS / 'sdpa_prefill_chain'),
              str(OPS / 'sdpa_prefill_bench')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import apply_factory_k64j as factory  # noqa: E402
import apply_factory_pf as pf_factory  # noqa: E402
import apply_factory_qwen as qwen_factory  # noqa: E402
import apply_factory_slice as slice_factory  # noqa: E402
import make_k0_readers as k0  # noqa: E402
import make_k64j_kernels as kernels  # noqa: E402
import make_pf_reader as pf_reader  # noqa: E402

NL = chr(10)
BUILD = HERE / 'build_k64j.sh'
BUILD_K64I = OPS / 'sdpa_decode_slice' / 'build_k64i.sh'
PROBE = OPS / 'k64j_probe' / 'probe_k64j_card_b.py'
PROBE_RUNNER = OPS / 'k64j_probe' / 'run_card_b.sh'
PROBE_FIXTURES = OPS / 'k64j_probe' / 'fixtures'
PF_FIXTURE = OPS / 'sdpa_prefill_chain' / 'fixtures' / 'sdpa_program_factory.fd8c0676.cpp'
QUAL_CARD_LIB = CI / 'qual_card.sh'
BEGIN, END = '# >>> qual_card.sh', '# <<< qual_card.sh'
CARD_M, CARD_A = 'blackhole-CEF5729692C19E6D', 'blackhole-3707293C249A5E67'
UNPATCHED = '05708e6d9ddeddfdf13303d8f8fa391941d73b742ea3a380beeb0883ce8d4792'
GATE_IMAGE = re.search(r'^GATE_IMAGE=(sha256:[0-9a-f]{64})', BUILD.read_text(encoding='utf-8'), flags=re.M).group(1)
SCRUB = ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'RESULTS', 'CARD_B_ARGS', 'MSYS', 'K64J_BASE_GRAFT', 'K64J_BASE_SHA256',
         'K64J_GRAFT', 'K64J_DECODE_BASE_DIR', 'K64J_PF_BASE_DIR', 'K64J_WORK_ROOT', 'K64J_IMAGES',
         'K64J_SKIP_IMAGE_COMPARE', 'K64J_KEEP_STAGED', 'K64J_BUILD_DRY_RUN', 'FAKE_NINJA_DROP', 'FAKE_NINJA_FAIL',
         'FAKE_TTBUILD_DOWN')
STOCK = {
    'dataflow/reader_decode_all.cpp': PROBE_FIXTURES / 'reader_decode_all.49a05926.cpp',
    'dataflow/writer_decode_all.cpp': PROBE_FIXTURES / 'writer_decode_all.734c90c0.cpp',
    'compute/sdpa_flash_decode.cpp': PROBE_FIXTURES / 'sdpa_flash_decode.d24769bd.cpp',
    'dataflow/dataflow_common.hpp': PROBE_FIXTURES / 'dataflow_common.e4623a22.hpp',
    'rt_args_common.hpp': PROBE_FIXTURES / 'rt_args_common.1b52c60d.hpp',
}
TRANSFORMER = 'opt/tt-metal/ttnn/cpp/ttnn/operations/transformer'
XTRANSFORMER = 'opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/transformer'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return Path(path).read_text(encoding='utf-8')


def recorded(text):
    return dict(re.findall(r'^([A-Z0-9_]+)=([0-9a-f]{64})\b', text, flags=re.M))


def shell_array(text, name):
    """The quoted words of a bash array assignment NAME=(...)."""
    body = re.search(r'^%s=\((.*?)\)$' % name, text, flags=re.M | re.S).group(1)
    return shlex.split(body.replace('"$SLICE_MARKER"', "'[QWEN-SDPA] q-slice rows_per_kv='")
                       .replace('"$EXTENT_MARKER"', "'[QWEN-SDPA] runtime-extent entries='"))


def block_span(text):
    start = text.index(BEGIN)
    end = text.index(NL, text.index(END)) + 1
    return start, end


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


def factory_base():
    """The audited 3e0a69af decode factory: the vendored fixture, QWEN_SDPA_FACTORY_BASE, or the job directory."""
    for candidate in (os.environ.get('QWEN_SDPA_FACTORY_BASE'), HERE / 'fixtures' / 'sdpa_decode_program_factory.3e0a69af.cpp',
                      'C:/Users/liamb/.claude/jobs/8376c877/tmp/sdpa_decode_program_factory.cpp.3e0a69af'):
        if candidate and Path(candidate).is_file() and sha(Path(candidate).read_bytes()) == factory.BASE_FACTORY:
            return Path(candidate).read_bytes()
    return None


def served_prefill_reader():
    """reader_interleaved.cpp f97f5490, rebuilt from the committed K0 probe reader k0a (it reverts on its own)."""
    text = (OPS / 'sdpa_prefill_bench' / 'k0' / k0.OUTPUT_NAMES['k0a']).read_bytes().decode('utf-8')
    return k0.revert_edits(text, 'k0a').encode('utf-8')


def unpatched_factory(base):
    """05708e6d, the factory before the tree-scratch patch (optimisation/sim/sdpa-tree-scratch.patch, reversed)."""
    text = base.decode('utf-8')
    text = text.replace('#include <cstdint>\n#include <cstdlib>\n#include <map>\n', '#include <cstdint>\n#include <map>\n', 1)
    text = text.replace(
        '    const char* scratch_rounds_override = std::getenv("QWEN_SDPA_TREE_SCRATCH_ROUNDS");\n'
        '    const bool compact_tree_scratch = scratch_rounds_override && std::string(scratch_rounds_override) == "1";\n'
        '    const uint32_t intermed_output_tiles =\n'
        '        (out_tiles + 2 * PNHt) * (compact_tree_scratch ? num_tree_reduction_rounds : num_cores_per_head - 1);\n',
        '    const uint32_t intermed_output_tiles = (out_tiles + 2 * PNHt) * (num_cores_per_head - 1);\n', 1)
    return text.encode('utf-8')


# ---------------------------------------------------------------------------------------------
# The contract and the text.
# ---------------------------------------------------------------------------------------------

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.text = read(BUILD)
        self.values = recorded(self.text)

    def test_the_recorded_shas_are_the_generators(self):
        expected = {
            'FACTORY_BASE': factory.BASE_FACTORY, 'FACTORY_QWEN_STAGE1': qwen_factory.QWEN_FACTORY,
            'FACTORY_QWEN_STAGE3': qwen_factory.QWEN_FACTORY_STAGE3, 'FACTORY_QWEN_STAGE4': slice_factory.STAGE4_FACTORY,
            'FACTORY_K64J': factory.K64J_FACTORY, 'PF_BASE': pf_factory.BASE_FACTORY, 'PF_FACTORY': pf_factory.PF_FACTORY,
            'PF_READER_BASE': pf_reader.BASE_SHA, 'PF_READER': pf_reader.READER_OUTPUT,
            'K64J_READER_QWEN': kernels.OUTPUTS[kernels.READER_QWEN], 'K64J_READER_SLICE': kernels.OUTPUTS[kernels.READER_SLICE],
            'K64J_COMPUTE_QWEN': kernels.OUTPUTS[kernels.COMPUTE_QWEN], 'K64J_WRITER_SLICE': kernels.OUTPUTS[kernels.WRITER_SLICE],
            'READER_QWEN': kernels.BASES[kernels.READER_QWEN], 'READER_SLICE': kernels.BASES[kernels.READER_SLICE],
            'COMPUTE_QWEN': kernels.BASES[kernels.COMPUTE_QWEN], 'WRITER_SLICE': kernels.BASES[kernels.WRITER_SLICE],
            'K64I_TTNNCPP': re.search(r"^K64I_TTNNCPP_SHA256 = '([0-9a-f]{64})'", read(PROBE), flags=re.M).group(1),
        }
        for name, value in expected.items():
            self.assertEqual(self.values.get(name), value, name)
        for name, path in STOCK.items():
            self.assertEqual(sha(path.read_bytes()), self.values[{'dataflow/reader_decode_all.cpp': 'READER_ALL',
                                                                   'dataflow/writer_decode_all.cpp': 'WRITER_ALL',
                                                                   'compute/sdpa_flash_decode.cpp': 'COMPUTE_ALL',
                                                                   'dataflow/dataflow_common.hpp': 'DATAFLOW_COMMON',
                                                                   'rt_args_common.hpp': 'RT_ARGS_COMMON'}[name]], name)

    def test_the_shared_shas_are_build_k64is(self):
        k64i = recorded(read(BUILD_K64I))
        shared = sorted(set(self.values) & set(k64i))
        self.assertEqual(len(shared), 18, shared)
        for name in shared:
            self.assertEqual(self.values[name], k64i[name], name)
        self.assertEqual(self.values['FACTORY_UNPATCHED'], UNPATCHED)
        self.assertEqual(sorted(set(self.values) - set(k64i)),
                         ['FACTORY_K64J', 'K64I_TTNNCPP', 'K64J_COMPUTE_QWEN', 'K64J_READER_QWEN', 'K64J_READER_SLICE',
                          'K64J_WRITER_SLICE'])

    def test_the_literals_are_the_factorys(self):
        self.assertEqual(shell_array(self.text, 'K64J_MARKERS'),
                         [factory.EXTENT_LOG_MARKER, factory.EXTENT_MASK_REFUSAL, factory.EXTENT_CUR_POS_REFUSAL,
                          factory.EXTENT_LAYOUT_REFUSAL, factory.CUR_POS_REFUSAL])
        k64i = read(BUILD_K64I)
        self.assertEqual(shell_array(self.text, 'STAGE4_MARKERS'), shell_array(k64i, 'STAGE4_MARKERS'))
        self.assertEqual(shell_array(self.text, 'PF_MARKERS'), shell_array(k64i, 'PF_MARKERS'))
        for name, value in (('SHARE_MARKER', qwen_factory.SHARE_MARKER), ('STAGE1_REFUSAL', qwen_factory.STAGE1_SHARE_REFUSAL),
                            ('SLICE_MARKER', slice_factory.SLICE_LOG_MARKER), ('EXTENT_MARKER', factory.EXTENT_LOG_MARKER)):
            self.assertIn("%s='%s'" % (name, value), self.text)
        base = factory_base()
        if base is None:
            self.skipTest('no 3e0a69af factory to patch')
        text = factory.patch(base).decode('utf-8')
        for marker in shell_array(self.text, 'K64J_MARKERS') + shell_array(self.text, 'STAGE4_MARKERS'):
            self.assertIn(marker, text)
        self.assertNotIn(qwen_factory.STAGE1_SHARE_REFUSAL, text)

    def test_the_kernels_and_the_image(self):
        self.assertEqual(shell_array(self.text, 'DECODE_KERNELS'), list(kernels.KERNELS))
        image = re.search(r'^IMAGE=\$\{IMAGE:-(sha256:[0-9a-f]{64})\}', read(PROBE_RUNNER), flags=re.M).group(1)
        self.assertIn('GATE_IMAGE=%s ' % image, self.text)
        self.assertIn('IMAGES=${K64J_IMAGES:-$GATE_IMAGE}', self.text)
        for kernel in kernels.KERNELS:
            self.assertTrue((HERE / 'kernels' / kernel).is_file(), kernel)
        # The K64i graft's .so is the probe's: the base the K64j build starts from.
        self.assertIn('BASE_SHA=${K64J_BASE_SHA256-$K64I_TTNNCPP}', self.text)

    def test_the_text(self):
        data = BUILD.read_bytes()
        self.assertNotIn(b'\r', data)
        self.assertTrue(self.text.startswith('#!/usr/bin/env bash\n'))
        for needle in ('set -euo pipefail', 'export LC_ALL=C', 'trap on_exit EXIT', 'K64J_BUILD_DRY_RUN',
                       'echo "K64J_TTNNCPP_SHA256=$so_sha"', '### dry run: nothing built', 'python3 -B'):
            self.assertIn(needle, self.text)
        start, end = block_span(self.text)
        self.assertEqual(self.text[start:end], read(QUAL_CARD_LIB))
        self.assertTrue(self.text[end:].startswith('qual_card_select' + NL))
        outside = self.text[:start] + self.text[end:]
        code = NL.join(line for line in outside.splitlines() if not line.lstrip().startswith('#'))
        for board in (CARD_M, CARD_A):
            self.assertNotIn(board, code)
        for word in ('docker run', '--device', 'qual_card_resolve', 'qual_refuse_holders', 'tt-smi', 'plink'):
            self.assertNotIn(word, code, word)
        # The EXIT trap is armed before the first ttbuild modification, and the restore runs before the verify.
        self.assertLess(self.text.index('trap on_exit EXIT'), self.text.index('armed=1\n'))
        self.assertLess(self.text.index('# ---------- 7. restore ttbuild'), self.text.index('# ---------- 6. verify'))
        self.assertLess(self.text.index('# ---------- 5. assemble'), self.text.index('# ---------- 7. restore ttbuild'))

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_script_parses(self):
        result = subprocess.run([BASH, '-n', BUILD.as_posix()], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


# ---------------------------------------------------------------------------------------------
# The dry run.
# ---------------------------------------------------------------------------------------------

def write_exe(path, text):
    path.write_bytes(text.encode('utf-8'))
    path.chmod(0o755)


def python_shim(directory):
    write_exe(directory / 'python3', '#!/bin/sh\nexec "%s" "$@"\n' % Path(sys.executable).as_posix())


def clean_env(**extra):
    env = {key: value for key, value in os.environ.items() if key not in SCRUB}
    env.update(extra)
    return env


def run_build(env, args=(), timeout=900):
    return subprocess.run([BASH, BUILD.as_posix()] + list(args), env=env, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=timeout)


def run_words(stdout):
    """The argv of every '### run:' line, in order."""
    out = []
    for line in stdout.splitlines():
        if line.startswith('### run: ') and not line.startswith('### run: (cd '):
            out.append(shlex.split(line[len('### run: '):]))
    return out


@unittest.skipUnless(BASH, 'bash not found')
class DryRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.bin = cls.dir / 'bin'
        cls.bin.mkdir()
        cls.home = cls.dir / 'home'
        cls.home.mkdir()
        cls.log = cls.dir / 'docker.log'
        write_exe(cls.bin / 'docker', '#!/bin/sh\necho "$@" >> "%s"\nexit 1\n' % cls.log.as_posix())
        python_shim(cls.bin)
        cls.env = clean_env(PATH=str(cls.bin) + os.pathsep + os.environ.get('PATH', ''), HOME=cls.home.as_posix(),
                            K64J_BUILD_DRY_RUN='1')
        cls.result = run_build(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_nothing_runs_and_nothing_is_written(self):
        result = self.result
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr)
        self.assertFalse(self.log.exists(), 'the dry run invoked docker')
        self.assertEqual(list(self.home.rglob('*')), [])
        self.assertEqual(result.stdout.rstrip().splitlines()[-1], '### dry run: nothing built')
        self.assertIn('### build_k64j DRY RUN', result.stdout)
        self.assertIn('does not exist here; the base graft was not checked', result.stdout)
        # Where it ran and whether every target is writable: what an offsite user learns from the dry run.
        self.assertRegex(result.stdout, r'(?m)^### host \S+ user \S+ HOME \S+$')
        for tail in ('/opgraft-K64j', '/kwork64/k64j', '/kwork64/k64f', '/kwork64/k64g'):
            self.assertRegex(result.stdout, r'(?m)^### \S*%s: its nearest existing directory \S+ is writable, ' % re.escape(tail))
        # The checkout's own read-only checks did run.
        for kernel in kernels.KERNELS:
            self.assertIn('ok    shipped K64j %s %s' % (kernel, kernels.OUTPUTS[kernel][:16]), result.stdout)
            self.assertIn('%s base %s -> %s matches' % (kernel, kernels.BASES[kernel][:16], kernels.OUTPUTS[kernel]),
                          result.stdout)
        self.assertNotIn('FAIL', result.stdout + result.stderr)

    def test_the_commands_are_the_builds_in_order(self):
        words = run_words(self.result.stdout)
        F = '/%s/sdpa_decode/device/sdpa_decode_program_factory.cpp' % TRANSFORMER
        PFP = '/%s/sdpa/device/sdpa_program_factory.cpp' % TRANSFORMER
        KD = '/%s/sdpa_decode/device/kernels/' % TRANSFORMER

        def index(predicate, after=-1, label=''):
            for position, argv in enumerate(words):
                if position > after and predicate(argv):
                    return position
            self.fail('no command %s after %d' % (label, after))

        def docker_cp(src_end, dst_end):
            return lambda argv: argv[:2] == ['docker', 'cp'] and argv[-2].endswith(src_end) and argv[-1].endswith(dst_end)

        stage = index(docker_cp('/sdpa_decode_program_factory.cpp', 'ttbuild:' + F), label='stage factory')
        self.assertIn('k64j-dry.', words[stage][2])                           # the patched copy in the work dir
        stage_pf = index(docker_cp('/sdpa_program_factory.cpp', 'ttbuild:' + PFP), label='stage prefill')
        steps = [stage, stage_pf]
        for kernel in kernels.KERNELS:
            steps.append(index(docker_cp('/k64j/kernels/' + kernel, 'ttbuild:' + KD + kernel), after=steps[-1],
                               label='stage ' + kernel))
        ninja = index(lambda argv: argv[:3] == ['docker', 'exec', 'ttbuild'] and 'ninja -C build_Release' in argv[-1],
                      after=steps[-1], label='ninja')
        steps.append(ninja)
        steps.append(index(lambda argv: argv[:2] == ['cp', '-a'] and argv[-1].endswith('/opgraft-K64j.partial/')
                           and argv[-2].endswith('/opgraft-K64i/.'), after=ninja, label='assemble K64i'))
        steps.append(index(docker_cp('ttbuild:/opt/tt-metal/build_Release/ttnn/_ttnncpp.so', '/opgraft-K64j.partial/_ttnncpp.so'),
                           after=steps[-1], label='assemble so'))
        for kernel in kernels.KERNELS:
            steps.append(index(lambda argv, kernel=kernel: argv[0] == 'cp' and argv[-2].endswith('/k64j/kernels/' + kernel)
                               and argv[-1].endswith('/opgraft-K64j.partial/sdpa_decode/device/kernels/' + kernel),
                               after=steps[-1], label='assemble ' + kernel))
        restore = index(docker_cp('/kwork64/k64f/sdpa_decode_program_factory.cpp.' + factory.BASE_FACTORY, 'ttbuild:' + F),
                        after=steps[-1], label='restore factory')
        steps.append(restore)
        for kernel in kernels.KERNELS:
            steps.append(index(lambda argv, kernel=kernel: argv[:5] == ['docker', 'exec', 'ttbuild', 'rm', '-f']
                               and argv[5] == KD + kernel, after=restore, label='restore ' + kernel))
        rebuild = index(lambda argv: argv[:3] == ['docker', 'exec', 'ttbuild'] and 'ninja -C build_Release' in argv[-1],
                        after=steps[-1], label='rebuild')
        steps.append(rebuild)
        steps.append(index(lambda argv: argv[:2] == ['docker', 'create'] and argv[-1] == GATE_IMAGE, after=rebuild,
                           label='image'))
        steps.append(index(lambda argv: argv[0] == 'mv' and argv[1].endswith('/opgraft-K64j.partial')
                           and argv[2].endswith('/opgraft-K64j'), after=steps[-1], label='move into place'))
        self.assertEqual(steps, sorted(steps))
        # Exactly two ninja runs, and the generators the build runs are printed, not run.
        self.assertEqual(sum(1 for argv in words if argv[:2] == ['docker', 'exec'] and 'ninja' in argv[-1]), 2)
        printed = [argv for argv in words if argv[:2] == ['python3', '-B']]
        self.assertEqual([Path(argv[2]).name for argv in printed],
                         ['make_k64j_kernels.py', 'make_pf_reader.py', 'apply_factory_k64j.py', 'apply_factory_pf.py'])
        self.assertIn('--reader', printed[0])
        self.assertIn('ttbuild restore: the commands above', self.result.stdout)

    def refused(self, result, needle):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertFalse(self.log.exists())

    def test_the_refusals(self):
        self.refused(run_build(dict(self.env, QUAL_CARD=CARD_M)), 'refusing: QUAL_CARD=%s is card M' % CARD_M)
        self.refused(run_build(dict(self.env, CARD_B_ARGS='--graft /tmp/x')),
                     "refusing: CARD_B_ARGS='--graft /tmp/x': build_k64j.sh takes no arguments from the job file")
        base = (self.home / 'opgraft-K64i').as_posix()
        self.refused(run_build(dict(self.env, K64J_GRAFT=base + '/nested')), 'and the base')
        self.refused(run_build(self.env, args=[(ROOT / 'graft-here').as_posix()]), 'is inside the checkout')
        self.refused(run_build(self.env, args=['a', 'b']), 'takes at most one argument')
        self.refused(run_build(dict(self.env, K64J_BUILD_DRY_RUN='yes')), 'refusing: K64J_BUILD_DRY_RUN=yes')
        self.assertEqual(list(self.home.rglob('*')), [])


# ---------------------------------------------------------------------------------------------
# The full run against a fake ttbuild.
# ---------------------------------------------------------------------------------------------

FAKE_NINJA = r'''
import os
from pathlib import Path
import re
import sys

LITERAL = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
BASE = [b'fake tt-metal _ttnncpp.so', b'qwen_draft_fp32_intermediates']
OPS = Path('ttnn/cpp/ttnn/operations/transformer')


def literals(*texts):
    return [value for text in texts for value in LITERAL.findall(text) if 'QWEN' in value or value.endswith('.cpp')]


def so_bytes(decode, prefill, drop=''):
    kept = [value.encode('utf-8') for value in literals(decode, prefill) if not (drop and drop in value)]
    return b'\x7fELF\x02\x01\x00' + b'\x00'.join(BASE + kept) + b'\x00'


def main(argv):
    if os.environ.get('FAKE_NINJA_FAIL') == '1':
        print('ninja: build stopped: subcommand failed.')
        return 1
    build = Path(argv[argv.index('-C') + 1])
    decode = (OPS / 'sdpa_decode/device/sdpa_decode_program_factory.cpp').read_text(encoding='utf-8')
    prefill = (OPS / 'sdpa/device/sdpa_program_factory.cpp').read_text(encoding='utf-8')
    data = so_bytes(decode, prefill, os.environ.get('FAKE_NINJA_DROP', ''))
    (build / 'ttnn').mkdir(parents=True, exist_ok=True)
    (build / 'ttnn' / '_ttnncpp.so').write_bytes(data)
    (build / 'ttnn' / '_ttnn.so').write_bytes(b'fake _ttnn.so over ' + str(len(data)).encode())
    print('[2/2] Linking CXX shared library ttnn/_ttnncpp.so')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
'''

FAKE_DOCKER = r'''
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

TTBUILD = Path(os.environ['FAKE_TTBUILD'])
IMAGES = Path(os.environ['FAKE_IMAGES'])
STATE = Path(os.environ['FAKE_STATE'])
LOG = Path(os.environ['FAKE_DOCKER_LOG'])


def host(path):
    # Under Git Bash the build script's host paths come unconverted (the shim turns MSYS conversion off): /c/... is a
    # drive, and /tmp/... is MSYS's mount of the Windows temp directory (FAKE_MSYS_TMP).
    if os.name == 'nt':
        tmp = os.environ.get('FAKE_MSYS_TMP')
        if tmp and (path == '/tmp' or path.startswith('/tmp/')):
            return Path(tmp + path[len('/tmp'):])
        match = re.match(r'^/([A-Za-z])(/.*)?$', path)
        if match:
            return Path(match.group(1).upper() + ':' + (match.group(2) or '/'))
    return Path(path)


def inside(root, path):
    if not path.startswith('/'):
        raise SystemExit('fake docker: a container path must be absolute: %r' % path)
    return root / path.lstrip('/')


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def image_dir(image):
    return IMAGES / image.replace(':', '_')


def is_container(spec):
    match = re.match(r'^([A-Za-z0-9_.-]+):/', spec)
    return bool(match) and len(match.group(1)) > 1


def resolve(spec):
    if not is_container(spec):
        return host(spec)
    name, _, path = spec.partition(':')
    if name == 'ttbuild':
        return inside(TTBUILD, path)
    cids = state()
    if name not in cids:
        raise SystemExit('fake docker: no container %s' % name)
    return inside(image_dir(cids[name]), path)


def copy(src, dst):
    if src.is_dir():
        if dst.exists():
            dst = dst / src.name
        shutil.copytree(src, dst)
        return 0
    if not src.is_file():
        sys.stderr.write('Error: No such container:path: %s\n' % src)
        return 1
    if dst.is_dir():
        dst = dst / src.name
    if not dst.parent.is_dir():
        sys.stderr.write('Error: no such directory %s\n' % dst.parent)
        return 1
    shutil.copyfile(src, dst)
    return 0


def main(argv):
    with LOG.open('a', encoding='utf-8') as log:
        log.write(json.dumps(argv) + '\n')
    command, args = argv[0], argv[1:]
    if command == 'ps':
        if os.environ.get('FAKE_TTBUILD_DOWN') != '1':
            print('ttbuild')
        print('qwen-other')
        return 0
    if command == 'image' and args[0] == 'inspect':
        return 0 if image_dir(args[1]).is_dir() else 1
    if command == 'create':
        cids = state()
        cid = 'fakecid%d' % len(cids)
        cids[cid] = args[-1]
        STATE.write_text(json.dumps(cids))
        print(cid)
        return 0
    if command == 'rm':
        cids = state()
        cids.pop(args[-1])
        STATE.write_text(json.dumps(cids))
        print(args[-1])
        return 0
    if command == 'cp':
        words = [word for word in args if word != '-L']
        return copy(resolve(words[0]), resolve(words[1]))
    if command == 'exec' and args[0] == 'ttbuild':
        tool, rest = args[1], args[2:]
        if tool == 'sha256sum':
            path = inside(TTBUILD, rest[0])
            if not path.is_file():
                sys.stderr.write('sha256sum: %s: No such file or directory\n' % rest[0])
                return 1
            print('%s  %s' % (hashlib.sha256(path.read_bytes()).hexdigest(), rest[0]))
            return 0
        if tool == 'test' and rest[0] == '-e':
            return 0 if inside(TTBUILD, rest[1]).exists() else 1
        if tool == 'rm' and rest[0] == '-f':
            for path in rest[1:]:
                inside(TTBUILD, path).unlink(missing_ok=True)
            return 0
        if tool == 'touch':
            for path in rest:
                inside(TTBUILD, path).touch()
            return 0
        if tool == 'bash' and rest[0] == '-c':
            script = rest[1].replace('/opt/tt-metal', (TTBUILD / 'opt' / 'tt-metal').as_posix())
            return subprocess.run([os.environ['FAKE_BASH'], '-c', script]).returncode
    sys.stderr.write('fake docker: unsupported %r\n' % (argv,))
    return 99


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
'''

FAKE_STRINGS = r'''
import re
import sys

data = open(sys.argv[-1], 'rb').read()
for match in re.finditer(rb'[\x20-\x7e\t]{4,}', data):
    sys.stdout.write(match.group().decode('ascii') + '\n')
'''

NINJA = {}
exec(compile(FAKE_NINJA, 'fake_ninja', 'exec'), NINJA)   # noqa: S102 - the fake linker, shared with the fake ttbuild


def write_manifest(graft):
    files = sorted('./' + path.relative_to(graft).as_posix() for path in graft.rglob('*')
                   if path.is_file() and path.name != 'MANIFEST.sha256')
    (graft / 'MANIFEST.sha256').write_bytes(''.join('%s  %s\n' % (sha((graft / name).read_bytes()), name)
                                                    for name in files).encode())


def msys_tmp():
    """The Windows spelling of Git Bash's /tmp (empty elsewhere)."""
    if os.name != 'nt':
        return ''
    return subprocess.run([BASH, '-c', 'cygpath -m /tmp'], capture_output=True, text=True, timeout=60).stdout.strip()


def snapshot(root):
    return {path.relative_to(root).as_posix(): sha(path.read_bytes()) for path in root.rglob('*') if path.is_file()}


RUN_E2E = os.name != 'nt' or os.environ.get('K64J_BUILD_E2E') == '1'


@unittest.skipUnless(BASH, 'bash not found')
@unittest.skipUnless(RUN_E2E, 'the full fake-ttbuild runs spawn slowly on Windows: set K64J_BUILD_E2E=1')
class FullRunTests(unittest.TestCase):
    """build_k64j.sh end to end against a fake ttbuild, a fake K64i graft and a fake P8 image."""

    maxDiff = None

    def setUp(self):
        self.base_factory = factory_base()
        if self.base_factory is None:
            self.skipTest('no 3e0a69af decode factory (k64j/fixtures, QWEN_SDPA_FACTORY_BASE)')
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.home = self.dir / 'home'
        self.bin = self.dir / 'bin'
        self.ttbuild = self.dir / 'ttbuild'
        self.images = self.dir / 'images'
        self.results = self.dir / 'results'
        for path in (self.home, self.bin, self.images, self.results):
            path.mkdir()
        self.log = self.dir / 'docker.log'
        self.make_tools()
        self.make_ttbuild()
        self.make_base_graft()
        self.make_image()
        self.before = snapshot(self.ttbuild)

    def tearDown(self):
        self.tmp.cleanup()

    def make_tools(self):
        python = Path(sys.executable).as_posix()
        for name, source in (('fake_docker.py', FAKE_DOCKER), ('fake_ninja.py', FAKE_NINJA), ('fake_strings.py', FAKE_STRINGS)):
            (self.dir / name).write_text(source, encoding='utf-8', newline=NL)
        python_shim(self.bin)
        # No MSYS argument conversion for the fake docker: it gets the script's paths as written (container paths
        # stay /opt/..., and it maps /c/... host paths itself).
        write_exe(self.bin / 'docker', '#!/bin/sh\nMSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" exec "%s" -S -B "%s" "$@"\n'
                  % (python, (self.dir / 'fake_docker.py').as_posix()))
        write_exe(self.bin / 'ninja', '#!/bin/sh\nexec "%s" -S -B "%s" "$@"\n' % (python, (self.dir / 'fake_ninja.py').as_posix()))
        if shutil.which('strings') is None:
            write_exe(self.bin / 'strings', '#!/bin/sh\nexec "%s" -S -B "%s" "$@"\n' % (python, (self.dir / 'fake_strings.py').as_posix()))

    def op(self, *parts):
        return self.ttbuild.joinpath(TRANSFORMER, *parts)

    def put(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def make_ttbuild(self):
        self.put(self.op('sdpa_decode', 'device', 'sdpa_decode_program_factory.cpp'), self.base_factory)
        self.put(self.op('sdpa_decode', 'sdpa_decode.cpp'), b'// the op\n')
        for name, path in STOCK.items():
            self.put(self.op('sdpa_decode', 'device', 'kernels', name), path.read_bytes())
        self.put(self.op('sdpa', 'device', 'sdpa_program_factory.cpp'), PF_FIXTURE.read_bytes())
        self.put(self.op('sdpa', 'device', 'kernels', 'dataflow', 'reader_interleaved.cpp'), served_prefill_reader())
        self.put(self.op('attn_prep', 'attn_prep.cpp'), b'// attn_prep\n')
        self.put(self.ttbuild.joinpath(XTRANSFORMER, 'nlp_concat_heads_decode', 'nlp_concat_heads_decode.cpp'), b'// concat\n')
        # build_Release as the fake ninja links it from the audited factories (the restore must bring it back).
        linked = NINJA['so_bytes'](self.base_factory.decode(), PF_FIXTURE.read_text(encoding='utf-8'))
        self.put(self.ttbuild / 'opt/tt-metal/build_Release/ttnn/_ttnncpp.so', linked)
        self.put(self.ttbuild / 'opt/tt-metal/build_Release/ttnn/_ttnn.so', b'fake _ttnn.so over ' + str(len(linked)).encode())

    def make_base_graft(self):
        base = self.home / 'opgraft-K64i'
        for name, src in (('attn_prep', self.op('attn_prep')), ('sdpa_decode', self.op('sdpa_decode')), ('sdpa', self.op('sdpa')),
                          ('nlp_concat_heads_decode', self.ttbuild.joinpath(XTRANSFORMER, 'nlp_concat_heads_decode'))):
            shutil.copytree(src, base / name)
        for kernel, path in kernels.COMMITTED.items():
            self.put(base / 'sdpa_decode' / 'device' / 'kernels' / kernel, path.read_bytes())
        self.put(base / 'sdpa' / 'device' / 'kernels' / 'dataflow' / 'reader_interleaved_qwen_chain.cpp',
                 (OPS / 'sdpa_prefill_chain' / 'reader_interleaved_qwen_chain.cpp').read_bytes())
        stage4 = slice_factory.patch(self.base_factory).decode('utf-8')
        prefill = pf_factory.patch(PF_FIXTURE.read_bytes()).decode('utf-8')
        self.put(base / '_ttnncpp.so', NINJA['so_bytes'](stage4, prefill))
        self.put(base / '_ttnn.so', b'fake K64i _ttnn.so\n')
        write_manifest(base)
        self.base = base
        self.base_so = sha((base / '_ttnncpp.so').read_bytes())

    def make_image(self):
        root = self.images / GATE_IMAGE.replace(':', '_')
        shutil.copytree(self.op('sdpa_decode'), root / TRANSFORMER / 'sdpa_decode')
        shutil.copytree(self.op('sdpa'), root / TRANSFORMER / 'sdpa')
        # An image may carry an older qwen kernel of its own: a qwen-named difference is allowed.
        self.put(root / TRANSFORMER / 'sdpa_decode' / 'device' / 'kernels' / kernels.READER_QWEN,
                 kernels.COMMITTED[kernels.READER_QWEN].read_bytes())
        self.put(root / 'opt/tt-metal/build_Release/lib/_ttnncpp.so', b'\x7fELF\x00QWEN_SDPA_TREE_SCRATCH_ROUNDS\x00image\x00')

    def run_build(self, **extra):
        env = clean_env(PATH=str(self.bin) + os.pathsep + os.environ.get('PATH', ''), HOME=self.home.as_posix(),
                        RESULTS=self.results.as_posix(), K64J_BASE_SHA256=self.base_so, FAKE_TTBUILD=str(self.ttbuild),
                        FAKE_IMAGES=str(self.images), FAKE_STATE=str(self.dir / 'state.json'),
                        FAKE_DOCKER_LOG=str(self.log), FAKE_BASH=BASH, FAKE_MSYS_TMP=msys_tmp())
        env.update(extra)
        return run_build(env)

    def docker_calls(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]

    def test_the_build_writes_the_k64j_graft_and_restores_ttbuild(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-3000:])
        graft = self.home / 'opgraft-K64j'
        self.assertTrue(graft.is_dir())
        self.assertFalse((self.home / 'opgraft-K64j.partial').exists())
        so = sha((graft / '_ttnncpp.so').read_bytes())
        self.assertIn('K64J_TTNNCPP_SHA256=%s' % so, result.stdout.splitlines())
        self.assertIn('next: the card tests with CARD_B_ENV=WATCHER=1 KOPGRAFT64=', result.stdout)
        self.assertNotEqual(so, self.base_so)
        manifest = (graft / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines()
        listed = {}
        for line in manifest:
            digest, name = re.match(r'^([0-9a-f]{64}) [ *](.+)$', line).groups()   # '  ' (Linux) or ' *' (Git Bash)
            listed[name] = digest
        self.assertEqual(listed, {'./' + key: value for key, value in snapshot(graft).items() if key != 'MANIFEST.sha256'})
        for kernel in kernels.KERNELS:
            self.assertEqual(sha((graft / 'sdpa_decode' / 'device' / 'kernels' / kernel).read_bytes()), kernels.OUTPUTS[kernel])
        self.assertEqual(sha((graft / 'sdpa_decode' / 'device' / 'sdpa_decode_program_factory.cpp').read_bytes()),
                         factory.BASE_FACTORY)
        binary = (graft / '_ttnncpp.so').read_bytes()
        for marker in (factory.EXTENT_LOG_MARKER, factory.EXTENT_MASK_REFUSAL, slice_factory.SLICE_LOG_MARKER, '[QWEN-SDPA-PF] flags='):
            self.assertIn(marker.encode(), binary)
        # ttbuild is back byte for byte (the restore rebuild relinked the audited factories), the bases were saved.
        self.assertEqual(snapshot(self.ttbuild), self.before)
        self.assertEqual(sha((self.home / 'kwork64' / 'k64f' / ('sdpa_decode_program_factory.cpp.' + factory.BASE_FACTORY)).read_bytes()),
                         factory.BASE_FACTORY)
        self.assertEqual(sha((self.home / 'kwork64' / 'k64g' / ('sdpa_program_factory.cpp.' + pf_factory.BASE_FACTORY)).read_bytes()),
                         pf_factory.BASE_FACTORY)
        self.assertEqual(sorted(path.name for path in self.results.iterdir()),
                         ['base-qwen-strings.txt', 'graft-qwen-strings.txt',
                          'image-%s-qwen-strings.txt' % GATE_IMAGE[7:19],
                          'k64j-MANIFEST.sha256', 'k64j-build-summary.txt'])
        self.assertIn('K64J_TTNNCPP_SHA256=%s' % so, (self.results / 'k64j-build-summary.txt').read_text(encoding='utf-8'))
        calls = self.docker_calls()
        self.assertFalse([call for call in calls if call[0] == 'run'])       # no device container, ever
        self.assertEqual(sum(1 for call in calls if call[:2] == ['exec', 'ttbuild'] and 'ninja' in call[-1]), 2)
        self.assertIn("with the four K64j kernels in place of K64i's", result.stdout)
        self.assertIn("ok    sdpa_decode = image %s's but for qwen kernels" % GATE_IMAGE[7:19], result.stdout)
        self.assertIn("ok    sdpa = image %s's + reader_interleaved_qwen_chain.cpp" % GATE_IMAGE[7:19], result.stdout)

    def assert_restored(self, result, before=None):
        self.assertEqual(snapshot(self.ttbuild), self.before if before is None else before, result.stdout[-2000:])
        self.assertFalse((self.home / 'opgraft-K64j').exists())

    def test_a_base_graft_whose_manifest_does_not_verify(self):
        (self.base / 'attn_prep' / 'attn_prep.cpp').write_bytes(b'// edited after the manifest\n')
        result = self.run_build()
        self.assertEqual(result.returncode, 1)
        self.assertRegex(result.stderr, r'FAIL: \S*/home/opgraft-K64i/MANIFEST\.sha256 does not verify')
        self.assertEqual(self.docker_calls(), [])
        self.assert_restored(result)

    def test_a_ttbuild_on_the_unpatched_factory(self):
        unpatched = unpatched_factory(self.base_factory)
        self.assertEqual(sha(unpatched), UNPATCHED)
        path = self.op('sdpa_decode', 'device', 'sdpa_decode_program_factory.cpp')
        path.write_bytes(unpatched)
        before = snapshot(self.ttbuild)
        result = self.run_build()
        self.assertEqual(result.returncode, 1)
        self.assertIn('FAIL: ttbuild decode factory is the unpatched 05708e6d', result.stderr)
        self.assert_restored(result, before)

    def test_a_link_that_loses_a_k64i_qwen_string(self):
        result = self.run_build(FAKE_NINJA_DROP='[QWEN-SDPA] q-slice saves no tile')
        self.assertEqual(result.returncode, 1)
        self.assertRegex(result.stderr, r'FAIL: QWEN strings of \S*/home/opgraft-K64i/_ttnncpp\.so missing from the new \.so')
        self.assertIn('[QWEN-SDPA] q-slice saves no tile', result.stderr)
        self.assert_restored(result)                                         # the restore ran before the verify
        self.assertTrue((self.home / 'opgraft-K64j.partial').is_dir())
        self.assertIn('is left for inspection', result.stderr)

    def test_a_failed_ninja_restores_the_sources(self):
        result = self.run_build(FAKE_NINJA_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ninja: build stopped', result.stdout)
        self.assertIn('WARN  ttbuild build_Release still holds the qwen objects', result.stderr)
        self.assert_restored(result)


if __name__ == '__main__':
    unittest.main()
