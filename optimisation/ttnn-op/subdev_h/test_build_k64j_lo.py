"""CPU checks for build_k64j_lo.sh, the K64j graft plus the all-gather link offset, built on the rig host through ttbuild (the cardm CI route).

  - the contract: the recorded shas are the vendored fixture's, the patch's and the patched file's; K64j's shas, kernel lists and markers are read from
    build_k64j.sh's own header (one source) and its expected base binary from scripts/ci/packed_any_admission.py;
  - the text: LF, bash -n, the canonical qual_card.sh block byte for byte followed by qual_card_select, no device, no docker run;
  - the dry run (bash): a fake docker that logs and fails is never invoked, nothing is written, the printed commands are the build's in order;
  - the full run (bash) against the K64j build test's fake ttbuild/docker/strings and a fake ninja that links a .so from the staged factories' literals:
    the build exits 0, publishes a graft that is the K64j base with exactly one file replaced, leaves the base and ttbuild byte for byte as it found them,
    and every refusal stops with its FAIL line, ttbuild restored: a base that is not K64j, an unexpected all-gather factory, a link that loses a QWEN
    string or adds another, a symbol delta, a failed ninja, a K64j rebuild that does not reproduce the base, an existing output.

    py -3.11 -B -m unittest test_build_k64j_lo      (from this directory)
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
OPS = HERE.parent
ROOT = HERE.parents[2]
CI = ROOT / 'scripts' / 'ci'
for _path in (str(OPS / 'k64j'), str(CI)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import test_build_k64j as kb  # noqa: E402  (the fake ttbuild, docker and strings, the fixtures, the K64j generators)
import packed_any_admission  # noqa: E402

BUILD = HERE / 'build_k64j_lo.sh'
K64J_BUILD = OPS / 'k64j' / 'build_k64j.sh'
FIXTURE = HERE / 'fixtures' / 'all_gather_async_default_program_factory.a7ba1851.cpp'
PATCH = HERE / 'ag_link_offset.patch'
AG_REL = 'ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp'
LITERAL = 'QWEN_AG_LINK_OFFSET_SD1'
BASH = kb.BASH
NL = chr(10)
SCRUB = kb.SCRUB + ('LO_BASE_GRAFT', 'LO_BASE_SHA256', 'LO_GRAFT', 'LO_WORK_ROOT', 'LO_IMAGES', 'LO_SKIP_REPRODUCE', 'LO_REQUIRE_REPRODUCE',
                    'LO_ALLOW_SYMBOL_DELTA', 'LO_KEEP_STAGED', 'LO_BUILD_DRY_RUN', 'FAKE_NINJA_EXTRA', 'FAKE_NINJA_SYMBOL')
sha, read = kb.sha, kb.read


def recorded(name):
    return re.search(r'^%s=([0-9a-f]{64})\b' % name, BUILD.read_text(encoding='utf-8'), flags=re.M).group(1)


class ContractTests(unittest.TestCase):
    def test_the_recorded_shas_are_the_fixture_the_patch_and_the_patched_file(self):
        self.assertEqual(sha(FIXTURE.read_bytes()), recorded('AG_BASE'))
        self.assertEqual(sha(PATCH.read_bytes()), recorded('AG_PATCH'))
        self.assertTrue(FIXTURE.name.endswith('.%s.cpp' % recorded('AG_BASE')[:8]))
        if shutil.which('patch') is None:
            self.skipTest('no patch binary')
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / AG_REL
            target.parent.mkdir(parents=True)
            shutil.copyfile(FIXTURE, target)
            result = subprocess.run(['patch', '-s', '-p1', '-d', directory], stdin=PATCH.open('rb'), capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sha(target.read_bytes()), recorded('AG_LO'))
            self.assertIn('std::getenv("%s")' % LITERAL, target.read_text(encoding='utf-8'))

    def test_the_fixture_is_the_stock_factory_with_its_licence(self):
        text = FIXTURE.read_text(encoding='utf-8')
        self.assertIn('SPDX-License-Identifier: Apache-2.0', text)
        self.assertNotIn('QWEN', text)
        self.assertNotIn(chr(13), text)

    @unittest.skipUnless(BASH, 'bash not found')
    def test_k64js_constants_come_from_its_own_header_not_a_copy(self):
        probe = ('S=%s; K64J_BUILD=%s; eval "$(sed -n \'/^# ---- recorded shas/,/^# >>> qual/p\' "$K64J_BUILD" | sed \'$d\')"; '
                 'echo "$FACTORY_K64J $PF_FACTORY $PF_READER $(k64j_sha dataflow/reader_decode_qwen.cpp) ${#DECODE_KERNELS[@]} $EXTENT_MARKER"'
                 % (HERE.as_posix(), K64J_BUILD.as_posix()))
        result = subprocess.run([BASH, '-c', probe], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        words = result.stdout.strip().split(' ', 5)
        self.assertEqual(words[:5], [kb.factory.K64J_FACTORY, kb.pf_factory.PF_FACTORY, kb.pf_reader.READER_OUTPUT,
                                     kb.kernels.OUTPUTS[kb.kernels.READER_QWEN], '4'])
        self.assertEqual(words[5], '[QWEN-SDPA] runtime-extent entries=')
        text = BUILD.read_text(encoding='utf-8')
        for name in ('FACTORY_K64J', 'PF_FACTORY', 'K64I_TTNNCPP'):
            self.assertNotRegex(text, r'(?m)^%s=' % name, 'K64j\'s %s must not be copied here' % name)

    def test_the_expected_base_binary_is_the_one_the_repo_records_for_the_served_image(self):
        match = re.search(r"^K64J_TTNNCPP_SHA256 = '([0-9a-f]{64})'", (CI / 'packed_any_admission.py').read_text(encoding='utf-8'), flags=re.M)
        self.assertEqual(match.group(1), packed_any_admission.K64J_TTNNCPP_SHA256)
        self.assertIn('K64J_TTNNCPP_SHA256 = ', BUILD.read_text(encoding='utf-8'))
        self.assertNotIn(packed_any_admission.K64J_TTNNCPP_SHA256, BUILD.read_text(encoding='utf-8'), 'a public script names no served digest')

    def test_the_text(self):
        raw = BUILD.read_bytes()
        self.assertNotIn(b'\r', raw)
        text = raw.decode('utf-8')
        library = (CI / 'qual_card.sh').read_text(encoding='utf-8')
        start, end = kb.block_span(text)
        self.assertEqual(text[start:end], library)
        self.assertTrue(text[end:].startswith('qual_card_select' + NL))
        code = [line for line in text.splitlines() if not line.lstrip().startswith('#')]
        body = NL.join(code)
        self.assertNotIn('docker run', body)
        self.assertNotIn('--device', body[body.index('qual_card_select'):])
        self.assertNotIn('qual_card_resolve', text[end:])
        self.assertNotIn('qual_refuse_holders', text[end:])
        self.assertIn('LC_ALL=C', text)
        self.assertIn('refusing: $GRAFT exists', text)
        self.assertIn('a graft is never overwritten', text)
        for forbidden in (CARD for CARD in ('blackhole-CEF5729692C19E6D', 'blackhole-3707293C249A5E67')):
            self.assertNotIn(forbidden, text[:start] + text[end:])

    def test_the_tracked_script_names_no_host_address_registry_or_home(self):
        text = BUILD.read_text(encoding='utf-8')
        start, end = kb.block_span(text)
        text = text[:start] + text[end:]
        for number, line in enumerate(text.splitlines(), 1):
            self.assertIsNone(re.search(r'\d{1,3}(\.\d{1,3}){3}|thatch\.local|zot\.|sha256:[0-9a-f]{16}|/home/|ssh ', line), '%d: %s' % (number, line))

    @unittest.skipUnless(BASH, 'bash not found')
    def test_the_script_parses(self):
        result = subprocess.run([BASH, '-n', BUILD.as_posix()], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


def run_build(env, args=(), timeout=900):
    return subprocess.run([BASH, BUILD.as_posix()] + list(args), env=env, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=timeout)


def clean_env(**extra):
    env = {key: value for key, value in os.environ.items() if key not in SCRUB}
    env.update(extra)
    return env


def run_words(stdout):
    import shlex
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
        kb.write_exe(cls.bin / 'docker', '#!/bin/sh\necho "$@" >> "%s"\nexit 1\n' % cls.log.as_posix())
        kb.python_shim(cls.bin)
        cls.env = clean_env(PATH=str(cls.bin) + os.pathsep + os.environ.get('PATH', ''), HOME=cls.home.as_posix(), LO_BUILD_DRY_RUN='1',
                            QUAL_CARD=kb.CARD_X)
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
        self.assertIn('### build_k64j_lo DRY RUN', result.stdout)
        self.assertIn('does not exist here; the base graft was not checked', result.stdout)
        for tail in ('/opgraft-K64j-LO', '/kwork64/k64j-lo'):
            self.assertRegex(result.stdout, r'(?m)^### \S*%s: its nearest existing directory \S+ is writable, ' % re.escape(tail))
        self.assertIn('ok    the stock all-gather factory fixture %s' % recorded('AG_BASE')[:16], result.stdout)
        self.assertIn('ok    ag_link_offset.patch %s' % recorded('AG_PATCH')[:16], result.stdout)
        self.assertIn('ok    the fixture with the patch applied %s' % recorded('AG_LO')[:16], result.stdout)
        self.assertNotIn('FAIL', result.stdout + result.stderr)

    def test_the_commands_are_k64js_staging_then_the_all_gather_patch_in_order(self):
        words = run_words(self.result.stdout)
        agf = '/opt/tt-metal/' + AG_REL

        def index(predicate, after=-1, label=''):
            for position, argv in enumerate(words):
                if position > after and predicate(argv):
                    return position
            self.fail('no command %s after %d' % (label, after))

        ninjas = [i for i, argv in enumerate(words) if argv[:3] == ['docker', 'exec', 'ttbuild'] and 'ninja -C build_Release' in argv[-1]]
        self.assertEqual(len(ninjas), 3, 'A (K64j alone), B (K64j + link offset) and the restore rebuild')
        stage_factory = index(lambda a: a[:2] == ['docker', 'cp'] and a[-1].endswith('sdpa_decode_program_factory.cpp'), label='stage decode')
        patched_ag = index(lambda a: a[:2] == ['docker', 'cp'] and a[-1] == 'ttbuild:' + agf and a[-2].endswith('all_gather_async_default_program_factory.cpp')
                           and 'k64j-lo-dry' in a[-2], label='stage patched all-gather')
        self.assertLess(stage_factory, ninjas[0])
        self.assertLess(ninjas[0], patched_ag, 'K64j alone is built before the patch is staged')
        self.assertLess(patched_ag, ninjas[1])
        assemble = index(lambda a: a[:2] == ['cp', '-a'] and a[-2].endswith('/opgraft-K64j/.') and a[-1].endswith('/opgraft-K64j-LO.partial/'), after=ninjas[1],
                         label='copy the base')
        replace = index(lambda a: a[0] == 'cp' and a[-2].endswith('B-ttnncpp.so') and a[-1].endswith('/opgraft-K64j-LO.partial/_ttnncpp.so'), after=assemble,
                        label='replace the one file')
        restore = index(lambda a: a[:2] == ['docker', 'cp'] and a[-1] == 'ttbuild:' + agf and a[-2].endswith('all_gather_async_default_program_factory.a7ba1851.cpp'),
                        after=replace, label='restore the stock all-gather factory from the fixture')
        self.assertLess(restore, ninjas[2])
        move = index(lambda a: a[0] == 'mv' and a[1].endswith('/opgraft-K64j-LO.partial') and a[2].endswith('/opgraft-K64j-LO'), after=ninjas[2], label='move')
        self.assertGreater(move, ninjas[2])
        # The base graft is never a destination of anything.
        for argv in words:
            self.assertFalse(argv[-1].rstrip('/').endswith('/opgraft-K64j') and argv[0] in ('cp', 'mv', 'rm', 'mkdir'), argv)
            self.assertNotEqual(argv[:2], ['rm', '-rf'] if argv[-1].endswith('/opgraft-K64j') else [], argv)

    def refused(self, result, needle):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertFalse(self.log.exists())

    def test_the_refusals(self):
        self.refused(run_build(dict(self.env, QUAL_CARD=kb.CARD_M)), 'refusing: QUAL_CARD=%s is card M' % kb.CARD_M)
        self.refused(run_build(dict(self.env, QUAL_CARD='')), 'refusing: QUAL_CARD is not set')
        self.refused(run_build(dict(self.env, QUAL_CARD='blackhole-F36F768B9A5CAFA0', ALLOW_SERVING_CARD='1')), 'is card B, reserved for another project')
        self.refused(run_build(dict(self.env, CARD_B_ARGS='--graft /tmp/x')), "refusing: CARD_B_ARGS='--graft /tmp/x'")
        base = (self.home / 'opgraft-K64j').as_posix()
        self.refused(run_build(dict(self.env, LO_GRAFT=base)), 'and the base')
        self.refused(run_build(dict(self.env, LO_GRAFT=base + '/nested')), 'and the base')
        self.refused(run_build(self.env, args=[(ROOT / 'graft-here').as_posix()]), 'is inside the checkout')
        self.refused(run_build(self.env, args=['a', 'b']), 'takes at most one argument')
        self.refused(run_build(dict(self.env, LO_BUILD_DRY_RUN='yes')), 'refusing: LO_BUILD_DRY_RUN=yes')
        existing = self.home / 'already'
        existing.mkdir()
        self.refused(run_build(dict(self.env, LO_GRAFT=existing.as_posix())), 'a graft is never overwritten')
        existing.rmdir()
        self.assertEqual(list(self.home.rglob('*')), [])


FAKE_NINJA_LO = r'''
import os
from pathlib import Path
import re
import sys

LITERAL = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
BASE = [b'fake tt-metal _ttnncpp.so', b'qwen_draft_fp32_intermediates', b'SYM_core']
OPS = Path('ttnn/cpp/ttnn/operations/transformer')
AG = Path('ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp')


def literals(*texts):
    return [value for text in texts for value in LITERAL.findall(text) if 'QWEN' in value or value.endswith('.cpp')]


def so_bytes(decode, prefill, ag, drop='', extra='', symbol=False):
    kept = [value.encode('utf-8') for value in literals(decode, prefill, ag) if not (drop and drop in value)]
    tokens = BASE + kept + ([extra.encode()] if extra and 'QWEN_AG_LINK_OFFSET_SD1' in ag else []) + ([b'SYM_extra'] if symbol and 'QWEN_AG_LINK_OFFSET_SD1' in ag else [])
    return b'\x7fELF\x02\x01\x00' + b'\x00'.join(tokens) + b'\x00'


def main(argv):
    if os.environ.get('FAKE_NINJA_FAIL') == '1':
        print('ninja: build stopped: subcommand failed.')
        return 1
    build = Path(argv[argv.index('-C') + 1])
    decode = (OPS / 'sdpa_decode/device/sdpa_decode_program_factory.cpp').read_text(encoding='utf-8')
    prefill = (OPS / 'sdpa/device/sdpa_program_factory.cpp').read_text(encoding='utf-8')
    ag = AG.read_text(encoding='utf-8')
    data = so_bytes(decode, prefill, ag, os.environ.get('FAKE_NINJA_DROP', ''), os.environ.get('FAKE_NINJA_EXTRA', ''), os.environ.get('FAKE_NINJA_SYMBOL') == '1')
    (build / 'ttnn').mkdir(parents=True, exist_ok=True)
    (build / 'ttnn' / '_ttnncpp.so').write_bytes(data)
    (build / 'ttnn' / '_ttnn.so').write_bytes(b'fake _ttnn.so\n')       # a Python module links _ttnncpp.so by name: its bytes do not depend on it
    print('[2/2] Linking CXX shared library ttnn/_ttnncpp.so')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
'''

FAKE_NM = r'''
import re
import sys

data = open(sys.argv[-1], 'rb').read()
for match in re.finditer(rb'SYM_[A-Za-z0-9_]+', data):
    sys.stdout.write('0000000000001000 T %s\n' % match.group().decode('ascii'))
'''

NINJA = {}
exec(compile(FAKE_NINJA_LO, 'fake_ninja_lo', 'exec'), NINJA)   # noqa: S102 - the fake linker, shared with the fake ttbuild


@unittest.skipUnless(BASH, 'bash not found')
@unittest.skipUnless(kb.RUN_E2E, 'the full fake-ttbuild runs spawn slowly on Windows: set K64J_BUILD_E2E=1')
class FullRunTests(unittest.TestCase):
    """build_k64j_lo.sh end to end against a fake ttbuild, a fake K64j graft and a fake image."""

    maxDiff = None
    op = kb.FullRunTests.op
    put = kb.FullRunTests.put
    docker_calls = kb.FullRunTests.docker_calls

    def setUp(self):
        self.base_factory = kb.factory_base()
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
        self.before = kb.snapshot(self.ttbuild)
        self.base_before = kb.snapshot(self.base)

    def tearDown(self):
        self.tmp.cleanup()

    def make_tools(self):
        python = Path(sys.executable).as_posix()
        for name, source in (('fake_docker.py', kb.FAKE_DOCKER), ('fake_ninja.py', FAKE_NINJA_LO), ('fake_strings.py', kb.FAKE_STRINGS), ('fake_nm.py', FAKE_NM)):
            (self.dir / name).write_text(source, encoding='utf-8', newline=NL)
        kb.python_shim(self.bin)
        kb.write_exe(self.bin / 'docker', '#!/bin/sh\nMSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" exec "%s" -S -B "%s" "$@"\n'
                     % (python, (self.dir / 'fake_docker.py').as_posix()))
        kb.write_exe(self.bin / 'ninja', '#!/bin/sh\nexec "%s" -S -B "%s" "$@"\n' % (python, (self.dir / 'fake_ninja.py').as_posix()))
        kb.write_exe(self.bin / 'nm', '#!/bin/sh\nexec "%s" -S -B "%s" "$@"\n' % (python, (self.dir / 'fake_nm.py').as_posix()))
        if shutil.which('strings') is None:
            kb.write_exe(self.bin / 'strings', '#!/bin/sh\nexec "%s" -S -B "%s" "$@"\n' % (python, (self.dir / 'fake_strings.py').as_posix()))

    def linked(self, decode, prefill, ag, **kwargs):
        return NINJA['so_bytes'](decode, prefill, ag, **kwargs)

    def make_ttbuild(self):
        kb.FullRunTests.make_ttbuild(self)
        self.put(self.ttbuild / 'opt' / 'tt-metal' / AG_REL, FIXTURE.read_bytes())
        # build_Release as the fake ninja links it from the audited sources (the restore must bring it back)
        linked = self.linked(self.base_factory.decode(), kb.PF_FIXTURE.read_text(encoding='utf-8'), FIXTURE.read_text(encoding='utf-8'))
        self.put(self.ttbuild / 'opt/tt-metal/build_Release/ttnn/_ttnncpp.so', linked)
        self.put(self.ttbuild / 'opt/tt-metal/build_Release/ttnn/_ttnn.so', b'fake _ttnn.so\n')

    def make_base_graft(self, extra_token=b''):
        base = self.home / 'opgraft-K64j'
        for name, src in (('attn_prep', self.op('attn_prep')), ('sdpa_decode', self.op('sdpa_decode')), ('sdpa', self.op('sdpa')),
                          ('nlp_concat_heads_decode', self.ttbuild.joinpath(kb.XTRANSFORMER, 'nlp_concat_heads_decode'))):
            shutil.copytree(src, base / name)
        kernels_dir = OPS / 'k64j' / 'kernels'
        for kernel in kb.kernels.KERNELS:
            self.put(base / 'sdpa_decode' / 'device' / 'kernels' / kernel, (kernels_dir / kernel).read_bytes())
        self.put(base / 'sdpa' / 'device' / 'kernels' / 'dataflow' / 'reader_interleaved_qwen_chain.cpp',
                 (OPS / 'sdpa_prefill_chain' / 'reader_interleaved_qwen_chain.cpp').read_bytes())
        k64j = kb.factory.patch(self.base_factory).decode('utf-8')
        prefill = kb.pf_factory.patch(kb.PF_FIXTURE.read_bytes()).decode('utf-8')
        data = self.linked(k64j, prefill, FIXTURE.read_text(encoding='utf-8')) + extra_token
        self.put(base / '_ttnncpp.so', data)
        self.put(base / '_ttnn.so', b'fake _ttnn.so\n')
        kb.write_manifest(base)
        self.base = base
        self.base_so = sha(data)

    def run_build(self, args=(), **extra):
        env = clean_env(PATH=str(self.bin) + os.pathsep + os.environ.get('PATH', ''), HOME=self.home.as_posix(), RESULTS=self.results.as_posix(),
                        LO_BASE_SHA256=self.base_so, FAKE_TTBUILD=str(self.ttbuild), FAKE_IMAGES=str(self.images), FAKE_STATE=str(self.dir / 'state.json'),
                        FAKE_DOCKER_LOG=str(self.log), FAKE_BASH=BASH, FAKE_MSYS_TMP=kb.msys_tmp(), QUAL_CARD=kb.CARD_X)
        env.update(extra)
        return run_build(env, args)

    def ninja_runs(self):
        return sum(1 for call in self.docker_calls() if call[:2] == ['exec', 'ttbuild'] and 'ninja' in call[-1])

    def assert_untouched(self, result, ttbuild_before=None):
        self.assertEqual(kb.snapshot(self.ttbuild), self.before if ttbuild_before is None else ttbuild_before, result.stdout[-2000:])
        self.assertEqual(kb.snapshot(self.base), self.base_before, 'the base graft was written')
        self.assertFalse((self.home / 'opgraft-K64j-LO').exists())

    def test_the_build_publishes_k64j_with_exactly_one_file_replaced(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-3000:])
        graft = self.home / 'opgraft-K64j-LO'
        self.assertTrue(graft.is_dir())
        self.assertFalse((self.home / 'opgraft-K64j-LO.partial').exists())
        so = sha((graft / '_ttnncpp.so').read_bytes())
        self.assertIn('K64J_LO_TTNNCPP_SHA256=%s' % so, result.stdout.splitlines())
        self.assertNotEqual(so, self.base_so)
        # exactly one file differs from the base, and it is the binary
        base, new = kb.snapshot(self.base), kb.snapshot(graft)
        self.assertEqual(sorted(set(base) ^ set(new)), [])
        self.assertEqual(sorted(name for name in base if base[name] != new[name]), ['MANIFEST.sha256', '_ttnncpp.so'])
        manifest = {}
        for line in (graft / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
            digest, name = re.match(r'^([0-9a-f]{64}) [ *](.+)$', line).groups()
            manifest[name] = digest
        self.assertEqual(manifest, {'./' + key: value for key, value in new.items() if key != 'MANIFEST.sha256'})
        binary = (graft / '_ttnncpp.so').read_bytes()
        self.assertIn(LITERAL.encode(), binary)
        self.assertNotIn(LITERAL.encode(), (self.base / '_ttnncpp.so').read_bytes())
        for marker in ('[QWEN-SDPA] runtime-extent entries=', '[QWEN-SDPA-PF] flags=', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS'):
            self.assertIn(marker.encode(), binary)
        # the base and ttbuild are as they were; the bases were saved; no device container
        self.assert_untouched_except_graft(result)
        self.assertIn('ok    REPRODUCED: K64j rebuilt alone', result.stdout)
        self.assertIn('ok    the only new QWEN string is %s' % LITERAL, result.stdout)
        self.assertIn('ok    MANIFEST.sha256 differs from the base\'s in exactly ./_ttnncpp.so', result.stdout)
        self.assertIn('ok    the 1 exported symbols equal the base\'s', result.stdout.replace('the 1 exported', 'the 1 exported'))
        self.assertEqual(self.ninja_runs(), 3)
        self.assertFalse([call for call in self.docker_calls() if call[0] == 'run'])
        self.assertEqual(sorted(path.name for path in self.results.iterdir()),
                         ['base-qwen-strings.txt', 'base-symbols.txt', 'graft-qwen-strings.txt', 'graft-symbols.txt', 'k64j-lo-MANIFEST.sha256',
                          'k64j-lo-build-summary.txt'])
        self.assertIn('reproduced=yes', (self.results / 'k64j-lo-build-summary.txt').read_text(encoding='utf-8'))

    def assert_untouched_except_graft(self, result):
        self.assertEqual(kb.snapshot(self.ttbuild), self.before, result.stdout[-2000:])
        self.assertEqual(kb.snapshot(self.base), self.base_before, 'the base graft was written')

    def test_a_second_run_refuses_to_overwrite_the_graft(self):
        self.assertEqual(self.run_build().returncode, 0)
        calls = len(self.docker_calls())
        again = self.run_build()
        self.assertEqual(again.returncode, 1)
        self.assertIn('a graft is never overwritten', again.stderr)
        self.assertEqual(len(self.docker_calls()), calls)

    def test_skipping_the_reproduction_saves_one_ninja_run(self):
        result = self.run_build(LO_SKIP_REPRODUCE='1')
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-2000:])
        self.assertEqual(self.ninja_runs(), 2)
        self.assertIn('WARN  LO_SKIP_REPRODUCE=1', result.stdout)
        self.assertNotIn('REPRODUCED', result.stdout)

    def test_a_base_graft_that_is_not_the_recorded_k64j_binary(self):
        result = self.run_build(LO_BASE_SHA256='0' * 64)
        self.assertEqual(result.returncode, 1)
        self.assertRegex(result.stderr, r'FAIL: base graft _ttnncpp\.so \(K64j, the served image\'s\) is %s, expected 0{64}' % self.base_so)
        self.assertEqual(self.docker_calls(), [])
        self.assert_untouched(result)

    def test_a_base_graft_whose_manifest_does_not_verify(self):
        (self.base / 'attn_prep' / 'attn_prep.cpp').write_bytes(b'// edited after the manifest\n')
        self.base_before = kb.snapshot(self.base)
        result = self.run_build()
        self.assertEqual(result.returncode, 1)
        self.assertRegex(result.stderr, r'FAIL: \S*/home/opgraft-K64j/MANIFEST\.sha256 does not verify')
        self.assertEqual(self.docker_calls(), [])
        self.assert_untouched(result)

    def test_a_base_graft_that_already_has_the_link_offset(self):
        shutil.rmtree(self.base)
        self.make_base_graft(extra_token=LITERAL.encode() + b'\x00')
        self.base_before = kb.snapshot(self.base)
        result = self.run_build()
        self.assertEqual(result.returncode, 1)
        self.assertIn('already has %s (it is not K64j)' % LITERAL, result.stderr)
        self.assert_untouched(result)

    def test_a_base_graft_that_is_not_k64j(self):
        shutil.rmtree(self.base)
        self.make_base_graft()
        data = (self.base / '_ttnncpp.so').read_bytes().replace(b'[QWEN-SDPA] runtime-extent entries=', b'[QWEN-SDPA] runtime-extent entrieX=')
        (self.base / '_ttnncpp.so').write_bytes(data)
        kb.write_manifest(self.base)
        self.base_so = sha(data)
        self.base_before = kb.snapshot(self.base)
        result = self.run_build()
        self.assertEqual(result.returncode, 1)
        self.assertIn("lacks '[QWEN-SDPA] runtime-extent entries=' (not K64j)", result.stderr)
        self.assert_untouched(result)

    def test_a_ttbuild_whose_all_gather_factory_is_not_the_stock_one(self):
        path = self.ttbuild / 'opt' / 'tt-metal' / AG_REL
        path.write_bytes(path.read_bytes() + b'// another tt-metal\n')
        before = kb.snapshot(self.ttbuild)
        result = self.run_build()
        self.assertEqual(result.returncode, 1)
        self.assertIn('ttbuild all-gather factory is', result.stderr)
        self.assertIn("not the stock %s" % recorded('AG_BASE'), result.stderr)
        self.assert_untouched(result, before)
        self.assertEqual(self.ninja_runs(), 0)

    def test_a_leftover_patched_all_gather_factory_is_replaced_and_restored(self):
        path = self.ttbuild / 'opt' / 'tt-metal' / AG_REL
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / AG_REL
            target.parent.mkdir(parents=True)
            shutil.copyfile(FIXTURE, target)
            subprocess.run(['patch', '-s', '-p1', '-d', directory], stdin=PATCH.open('rb'), check=True)
            path.write_bytes(target.read_bytes())
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-2000:])
        self.assertIn('already the patched one', result.stdout)
        self.assertEqual(sha(path.read_bytes()), recorded('AG_BASE'), 'restored to the stock factory, from the fixture')

    def test_a_link_that_loses_a_k64j_qwen_string(self):
        result = self.run_build(FAKE_NINJA_DROP='[QWEN-SDPA] q-slice saves no tile')
        self.assertEqual(result.returncode, 1)
        # build A (K64j alone) already differs from the base; the weight is carried by the string check
        self.assertRegex(result.stderr, r'FAIL: QWEN strings of \S*/home/opgraft-K64j/_ttnncpp\.so missing from the new \.so')
        self.assertIn('[QWEN-SDPA] q-slice saves no tile', result.stderr)
        self.assertEqual(kb.snapshot(self.ttbuild), self.before)
        self.assertEqual(kb.snapshot(self.base), self.base_before)
        self.assertTrue((self.home / 'opgraft-K64j-LO.partial').is_dir())
        self.assertIn('is left for inspection', result.stderr)
        self.assertFalse((self.home / 'opgraft-K64j-LO').exists())

    def test_a_link_that_adds_a_second_qwen_string(self):
        result = self.run_build(FAKE_NINJA_EXTRA='QWEN_SURPRISE')
        self.assertEqual(result.returncode, 1)
        self.assertIn('beyond the base\'s are not exactly %s' % LITERAL, result.stderr)
        self.assertIn('QWEN_SURPRISE', result.stderr)
        self.assertEqual(kb.snapshot(self.ttbuild), self.before)
        self.assertFalse((self.home / 'opgraft-K64j-LO').exists())

    def test_a_different_set_of_exported_symbols(self):
        result = self.run_build(FAKE_NINJA_SYMBOL='1')
        self.assertEqual(result.returncode, 1)
        self.assertIn('the exported symbols of the new .so differ', result.stderr)
        self.assertIn('SYM_extra', result.stderr)
        self.assertFalse((self.home / 'opgraft-K64j-LO').exists())
        shutil.rmtree(self.home / 'opgraft-K64j-LO.partial')
        allowed = self.run_build(FAKE_NINJA_SYMBOL='1', LO_ALLOW_SYMBOL_DELTA='1')
        self.assertEqual(allowed.returncode, 0, allowed.stdout[-2000:] + allowed.stderr[-2000:])
        self.assertIn('WARN  the exported symbols of the new .so differ', allowed.stdout)

    def test_a_k64j_rebuild_that_does_not_reproduce_the_base_is_a_warning_unless_required(self):
        shutil.rmtree(self.base)
        self.make_base_graft(extra_token=b'built at 2026-10-09\x00')       # a non-QWEN difference a toolchain or a timestamp could make
        self.base_before = kb.snapshot(self.base)
        warned = self.run_build()
        self.assertEqual(warned.returncode, 0, warned.stdout[-3000:] + warned.stderr[-2000:])
        self.assertIn('is NOT byte-identical to the base graft\'s', warned.stdout)
        self.assertNotIn('REPRODUCED', warned.stdout)
        self.assertIn('reproduced=no', (self.results / 'k64j-lo-build-summary.txt').read_text(encoding='utf-8'))
        shutil.rmtree(self.home / 'opgraft-K64j-LO')
        required = self.run_build(LO_REQUIRE_REPRODUCE='1')
        self.assertEqual(required.returncode, 1)
        self.assertIn('and LO_REQUIRE_REPRODUCE=1', required.stderr)
        # the sources are back; build_Release still holds the K64j objects (the trap restores and touches, it does not rebuild) and says so
        sources = lambda snap: {k: v for k, v in snap.items() if '/build_Release/' not in k}  # noqa: E731
        self.assertEqual(sources(kb.snapshot(self.ttbuild)), sources(self.before))
        self.assertIn('WARN  ttbuild build_Release still holds the staged objects', required.stderr)

    def test_a_failed_ninja_restores_the_sources(self):
        result = self.run_build(FAKE_NINJA_FAIL='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ninja: build stopped', result.stdout)
        self.assertIn('WARN  ttbuild build_Release still holds the staged objects', result.stderr)
        self.assert_untouched(result)

    def test_an_image_whose_qwen_string_the_result_lacks(self):
        image = self.images / 'fakeimg_1' / 'opt/tt-metal/build_Release/lib/_ttnncpp.so'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'\x7fELF\x00QWEN_SDPA_TREE_SCRATCH_ROUNDS\x00QWEN_ONLY_THE_IMAGE_HAS\x00')
        result = self.run_build(LO_IMAGES='fakeimg:1')
        self.assertEqual(result.returncode, 1)
        self.assertIn("QWEN strings of image fakeimg:1's _ttnncpp.so missing from the new .so", result.stderr)
        self.assertIn('QWEN_ONLY_THE_IMAGE_HAS', result.stderr)
        image.write_bytes(b'\x7fELF\x00QWEN_SDPA_TREE_SCRATCH_ROUNDS\x00')
        shutil.rmtree(self.home / 'opgraft-K64j-LO.partial')
        ok = self.run_build(LO_IMAGES='fakeimg:1')
        self.assertEqual(ok.returncode, 0, ok.stdout[-2000:] + ok.stderr[-2000:])
        self.assertIn("ok    every QWEN string of image fakeimg:1's _ttnncpp.so is in the new .so", ok.stdout)


if __name__ == '__main__':
    unittest.main()
