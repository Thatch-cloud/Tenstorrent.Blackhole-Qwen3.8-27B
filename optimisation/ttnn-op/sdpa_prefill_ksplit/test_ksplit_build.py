"""CPU tests of build_k64j_oq_ks.sh, the K64j-OQ graft plus the K-split edits, built on the rig host through ttbuild (the cardm CI route). NOT RUN as a build: the evidence is the text, the recorded
shas, the dry run (every command printed, none run) and the refusals; the first real run is the first compile of the C++.

Run from this directory:  python3 -B -m unittest test_ksplit_build
"""

import hashlib
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
for path in (HERE, OPS / 'sdpa_prefill_oneq', OPS / 'sdpa_prefill_chain', ROOT / 'scripts' / 'ci'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import apply_factory_pk as pk  # noqa: E402
import make_ks_compute as mc  # noqa: E402
import make_ks_reader as mr  # noqa: E402
import make_ks_writer as mw  # noqa: E402

BUILD = HERE / 'build_k64j_oq_ks.sh'
ONEQ_BUILD = OPS / 'sdpa_prefill_oneq' / 'build_k64j_oq.sh'
BASH = shutil.which('bash')
NL = chr(10)
SCRUB = ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'RESULTS', 'CARD_B_ARGS', 'KS_BASE_GRAFT', 'KS_BASE_SHA256', 'KS_GRAFT', 'KS_WORK_ROOT', 'KS_IMAGES', 'KS_SKIP_REPRODUCE', 'KS_REQUIRE_REPRODUCE',
         'KS_ALLOW_SYMBOL_DELTA', 'KS_KEEP_STAGED', 'KS_BUILD_DRY_RUN', 'K64J_DECODE_BASE_DIR', 'K64J_PF_BASE_DIR')
FAKE_CARD = 'blackhole-0000000000000001'      # a board id no library names: selects without the serving override
LIBRARY = (ROOT / 'scripts' / 'ci' / 'qual_card.sh').read_text(encoding='utf-8')
RESERVED = re.search(r"^QUAL_RESERVED_CARD=(\S+)", LIBRARY, flags=re.M).group(1)      # read from the library: a new file does not name a board


def sha(data):
    return hashlib.sha256(data).hexdigest()


def recorded(name):
    return re.search(r'^%s=([0-9a-f]{64})\b' % name, BUILD.read_text(encoding='utf-8'), flags=re.M).group(1)


class ContractTests(unittest.TestCase):
    def test_the_recorded_shas_are_the_generators_and_their_outputs(self):
        self.assertEqual(recorded('PK_FACTORY'), pk.PK_FACTORY)
        self.assertEqual(recorded('PK_SCRIPT'), sha((HERE / 'apply_factory_pk.py').read_bytes()))
        self.assertEqual(recorded('KS_READER'), mr.READER_OUTPUT)
        self.assertEqual(recorded('KS_WRITER'), mw.WRITER_OUTPUT)
        self.assertEqual(recorded('KS_COMPUTE'), mc.COMPUTE_OUTPUT)
        for name, module_file in (('KS_READER', 'reader_interleaved_qwen_ksplit.cpp'), ('KS_WRITER', 'writer_interleaved_qwen_ksplit.cpp'), ('KS_COMPUTE', 'sdpa_qwen_ksplit.cpp')):
            self.assertEqual(recorded(name), sha((HERE / module_file).read_bytes()))

    def test_the_literals_are_the_four_the_patch_adds(self):
        text = BUILD.read_text(encoding='utf-8')
        for index, literal in enumerate(pk.NEW_QWEN_STRINGS, 1):
            self.assertIn("KS_LITERAL_%d='%s'" % (index, literal), text)
        self.assertEqual(len(pk.NEW_QWEN_STRINGS), 4)

    def test_the_k64j_and_oneq_constants_come_from_their_own_headers_not_a_copy(self):
        text = BUILD.read_text(encoding='utf-8')
        for name in ('FACTORY_K64J', 'PF_FACTORY', 'K64I_TTNNCPP', 'PS_FACTORY', 'PS_SCRIPT'):
            self.assertNotRegex(text, r'(?m)^%s=' % name, "%s must not be copied here" % name)
        self.assertIn("sed -n '/^# ---- recorded shas/,/^# >>> qual/p' \"$K64J_BUILD\"", text)
        self.assertIn("/^PS_FACTORY=/p;/^PS_SCRIPT=/p;/^OQ_LITERAL_1=/p;/^OQ_LITERAL_2=/p", text)
        self.assertTrue(re.search(r'^PS_FACTORY=[0-9a-f]{64}', ONEQ_BUILD.read_text(encoding='utf-8'), flags=re.M))

    def test_the_expected_base_binary_comes_from_the_admission_not_from_this_file(self):
        text = BUILD.read_text(encoding='utf-8')
        self.assertIn("K64J_OQ_TTNNCPP_SHA256 = '[0-9a-f]{64}'", text)
        import packed_any_admission
        self.assertNotIn(packed_any_admission.K64J_OQ_TTNNCPP_SHA256, text)
        self.assertNotIn(packed_any_admission.K64J_TTNNCPP_SHA256, text)

    def test_the_pinned_shas_it_checks_in_ttbuild_are_the_fixtures(self):
        text = BUILD.read_text(encoding='utf-8')
        for fixture in ('writer_interleaved.2a0959ff.cpp', 'sdpa.a3f48af8.cpp', 'compute_common.3fb5da24.hpp'):
            digest = sha((HERE / 'fixtures' / fixture).read_bytes())
            self.assertIn(digest, text, fixture)

    def test_the_text(self):
        raw = BUILD.read_bytes()
        self.assertNotIn(b'\r', raw)
        text = raw.decode('utf-8')
        self.assertIn('LC_ALL=C', text)
        self.assertIn('refusing: $GRAFT exists', text)
        self.assertIn('a graft is never overwritten', text)
        self.assertIn('. "$lib"', text)
        self.assertNotIn('>>> qual_card.sh', text)                       # sourced, not embedded: the embedded copies carry board ids
        code = [line for line in text.splitlines() if not line.lstrip().startswith('#')]
        body = NL.join(code)
        self.assertNotIn('docker run', body)
        self.assertNotIn('--device', body)
        self.assertNotIn('qual_card_resolve', body)
        self.assertNotIn('qual_refuse_holders', body)
        self.assertNotRegex(text, r'blackhole-[0-9A-F]{8}')
        for number, line in enumerate(text.splitlines(), 1):
            self.assertIsNone(re.search(r'\d{1,3}(\.\d{1,3}){3}|th[a]tch\.loca[l]|zo[t]\.|sha256:[0-9a-f]{16}|/hom[e]/|ss[h] ', line), '%d: %s' % (number, line))

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_script_parses(self):
        self.assertEqual(subprocess.run([BASH, '-n', str(BUILD)], capture_output=True, text=True, timeout=60).returncode, 0)

    def test_the_output_is_a_new_directory_and_the_base_is_never_a_destination(self):
        text = BUILD.read_text(encoding='utf-8')
        self.assertIn('${KS_GRAFT:-$HOME/opgraft-K64j-OQ-KS}', text)
        self.assertIn('BASE=${KS_BASE_GRAFT:-$HOME/opgraft-K64j-OQ}', text)
        self.assertIn('x mv "$G" "$GRAFT"', text)
        self.assertNotRegex(text, r'(?m)^\s*x (rm|mv|cp)[^\n]*"\$BASE"/?\s*$')


def run_build(env, args=(), timeout=900):
    return subprocess.run([BASH, BUILD.as_posix()] + list(args), env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)


def clean_env(**extra):
    env = {key: value for key, value in os.environ.items() if key not in SCRUB}
    env.update(extra)
    return env


def run_words(stdout):
    return [shlex.split(line[len('### run: '):]) for line in stdout.splitlines() if line.startswith('### run: ') and not line.startswith('### run: (cd ')]


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
        docker = cls.bin / 'docker'
        docker.write_text('#!/bin/sh\necho "$@" >> "%s"\nexit 1\n' % cls.log.as_posix())
        docker.chmod(0o755)
        shim = cls.bin / 'python3'
        shim.write_text('#!/bin/sh\nexec "%s" "$@"\n' % Path(sys.executable).as_posix())
        shim.chmod(0o755)
        cls.env = clean_env(PATH=str(cls.bin) + os.pathsep + os.environ.get('PATH', ''), HOME=cls.home.as_posix(), KS_BUILD_DRY_RUN='1', QUAL_CARD=FAKE_CARD)
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
        self.assertIn('### build_k64j_oq_ks DRY RUN', result.stdout)
        self.assertIn('does not exist here; the base graft was not checked', result.stdout)
        for name, digest in (('apply_factory_ps.py', None), ('apply_factory_pk.py', recorded('PK_SCRIPT')), ('dataflow/reader_interleaved_qwen_ksplit.cpp', recorded('KS_READER')),
                             ('dataflow/writer_interleaved_qwen_ksplit.cpp', recorded('KS_WRITER')), ('compute/sdpa_qwen_ksplit.cpp', recorded('KS_COMPUTE'))):
            if digest:
                self.assertIn('ok    shipped %s %s' % (name, digest[:16]), result.stdout)
            else:
                self.assertRegex(result.stdout, r'ok    shipped %s [0-9a-f]{16}' % re.escape(name))
        self.assertNotIn('FAIL', result.stdout + result.stderr)

    def test_the_commands_stage_k64j_oq_then_the_ksplit_factory_and_kernels_in_order(self):
        words = run_words(self.result.stdout)

        def index(predicate, after=-1, label=''):
            for position, argv in enumerate(words):
                if position > after and predicate(argv):
                    return position
            self.fail('no command %s after %d' % (label, after))

        ninjas = [i for i, argv in enumerate(words) if argv[:3] == ['docker', 'exec', 'ttbuild'] and 'ninja -C build_Release' in argv[-1]]
        self.assertEqual(len(ninjas), 3, 'A (K64j-OQ alone), B (K64j-OQ + ksplit) and the restore rebuild')
        stage_ps = index(lambda a: a[:2] == ['docker', 'cp'] and a[-2].endswith('sdpa_program_factory.ps.cpp'), label='stage the oneq prefill factory')
        stage_pk = index(lambda a: a[:2] == ['docker', 'cp'] and a[-2].endswith('sdpa_program_factory.pk.cpp'), label='stage the ksplit prefill factory')
        self.assertLess(stage_ps, ninjas[0])
        self.assertLess(ninjas[0], stage_pk, 'K64j-OQ alone is built before the ksplit factory is staged')
        self.assertLess(stage_pk, ninjas[1])
        staged = [index(lambda a, name=name: a[:2] == ['docker', 'cp'] and a[-1].endswith('/kernels/' + name) and 'ksplit' in a[-1], after=ninjas[0], label=name)
                  for name in ('dataflow/reader_interleaved_qwen_ksplit.cpp', 'dataflow/writer_interleaved_qwen_ksplit.cpp', 'compute/sdpa_qwen_ksplit.cpp')]
        self.assertTrue(all(ninjas[0] < at < ninjas[1] for at in staged), 'the three kernels are staged between build A and build B')
        assemble = index(lambda a: a[:2] == ['cp', '-a'] and a[-2].endswith('/opgraft-K64j-OQ/.') and a[-1].endswith('/opgraft-K64j-OQ-KS.partial/'), after=ninjas[1], label='copy the base')
        replace = index(lambda a: a[0] == 'cp' and a[-2].endswith('B-ttnncpp.so') and a[-1].endswith('/opgraft-K64j-OQ-KS.partial/_ttnncpp.so'), after=assemble, label='replace the one file')
        added = [index(lambda a, name=name: a[0] == 'cp' and a[-1].endswith('/opgraft-K64j-OQ-KS.partial/sdpa/device/kernels/' + name), after=replace, label='add ' + name)
                 for name in ('dataflow/reader_interleaved_qwen_ksplit.cpp', 'dataflow/writer_interleaved_qwen_ksplit.cpp', 'compute/sdpa_qwen_ksplit.cpp')]
        restore = index(lambda a: a[:2] == ['docker', 'cp'] and a[-1].endswith('sdpa_program_factory.cpp') and 'fd8c0676' in a[-2], after=added[-1], label='restore the saved prefill base')
        removed = [index(lambda a, name=name: a[:3] == ['docker', 'exec', 'ttbuild'] and a[3:5] == ['rm', '-f'] and a[-1].endswith('/kernels/' + name), after=added[-1], label='remove ' + name)
                   for name in ('dataflow/reader_interleaved_qwen_ksplit.cpp', 'dataflow/writer_interleaved_qwen_ksplit.cpp', 'compute/sdpa_qwen_ksplit.cpp')]
        self.assertTrue(all(at < ninjas[2] for at in removed + [restore]), 'ttbuild is restored (the kernels removed, the factories put back) before the clean rebuild')
        move = index(lambda a: a[0] == 'mv' and a[1].endswith('/opgraft-K64j-OQ-KS.partial') and a[2].endswith('/opgraft-K64j-OQ-KS'), after=ninjas[2], label='move')
        self.assertGreater(move, ninjas[2])
        for argv in words:                                              # the base graft is never a destination of anything
            self.assertFalse(argv[-1].rstrip('/').endswith('/opgraft-K64j-OQ') and argv[0] in ('cp', 'mv', 'rm', 'mkdir'), argv)

    def refused(self, result, needle):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertFalse(self.log.exists())

    def test_the_refusals(self):
        self.refused(run_build(dict(self.env, QUAL_CARD='')), 'refusing: QUAL_CARD is not set')
        self.refused(run_build(dict(self.env, QUAL_CARD=RESERVED, ALLOW_SERVING_CARD='1')), 'reserved for another project')
        self.refused(run_build(dict(self.env, CARD_B_ARGS='--graft /tmp/x')), "refusing: CARD_B_ARGS='--graft /tmp/x'")
        base = (self.home / 'opgraft-K64j-OQ').as_posix()
        self.refused(run_build(dict(self.env, KS_GRAFT=base)), 'and the base')
        self.refused(run_build(dict(self.env, KS_GRAFT=base + '/nested')), 'and the base')
        self.refused(run_build(self.env, args=[(ROOT / 'graft-here').as_posix()]), 'is inside the checkout')
        self.refused(run_build(self.env, args=['a', 'b']), 'takes at most one argument')
        self.refused(run_build(dict(self.env, KS_BUILD_DRY_RUN='yes')), 'refusing: KS_BUILD_DRY_RUN=yes')
        existing = self.home / 'already'
        existing.mkdir()
        self.refused(run_build(dict(self.env, KS_GRAFT=existing.as_posix())), 'a graft is never overwritten')
        existing.rmdir()
        self.assertEqual(list(self.home.rglob('*')), [])


if __name__ == '__main__':
    unittest.main()
