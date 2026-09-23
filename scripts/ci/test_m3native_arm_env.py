"""Every env var the container-side scripts read must cross into the container.

Run 35679222511 is why this file exists. The v36 arm set M3NATIVE_PREFILL_CHUNK_TOKENS
=2048 on the runner host, mounted the three grafted M1 files, and passed
TT_M1_FORCE_CHUNKED_PREFILL=1 - everything needed for the resumable prefill path. The
docker run never passed CHUNK_TOKENS itself, so lever_n_m3native_gate.py inside the
container saw it unset, took the byte-identical-without-it branch, and launched the
server with --no-enable-chunked-prefill --max-num-batched-tokens 33024. Every prompt
was prefilled whole, the resumable path never ran, and the run looked like a Lever N
measurement while testing nothing about Lever N. The same hole had already made the two
TTFT ceilings unassertable.

The invariant is one-directional and derived, not listed: whatever the arm mounts at
/bench is read for env access, and every M3NATIVE_* name it reads must appear as a
docker -e passthrough. Adding a flag to a /bench script without wiring it fails here,
on CPU, in seconds - instead of on the rig, as a silently stock run.
"""

import ast
import hashlib
import os
from pathlib import Path
import re
import unittest

HERE = Path(__file__).parent
ARM = HERE / 'lever_n_m3native_run_arm.sh'

# A read this script performs on the HOST, deliberately not passed through: it selects
# mounts and other env vars rather than being consumed inside the container.
HOST_ONLY = frozenset()

# Image A capacity flags: host M3NATIVE_<X> becomes container QWEN_FAST_<X>=1.
CAPACITY_FLAGS = ('MEMORY_LEDGER', 'SKIP_BLOCK_STREAM', 'SINGLE_GATEUP', 'DRAFT_BF8')


def arm_text():
    return ARM.read_text(encoding='utf-8').replace(chr(13) + chr(10), chr(10))


def bench_scripts(text):
    """The scripts the arm mounts at /bench, from the arm itself so it cannot drift."""
    names = re.findall(r'dst=/bench/([A-Za-z0-9_]+[.]py)', text)
    if not names:
        raise AssertionError('no /bench mounts found in the arm - the regex has drifted')
    return sorted(set(names))


def passed_through(text):
    """Every M3NATIVE_* name the docker run hands the container as -e NAME=..."""
    return set(re.findall(r'-e (M3NATIVE_[A-Z0-9_]+)=', text))


def _is_environ(node):
    if isinstance(node, ast.Attribute):
        return node.attr == 'environ'
    return isinstance(node, ast.Name) and node.id == 'environ'


class _KeyParameters(ast.NodeVisitor):
    """Module-level functions that use a parameter as an environment key, by position.

    m3native_ttft_profile reads its two ceilings through a helper:

        def _threshold(environ, name):
            value = (os.environ if environ is None else environ).get(name)
        ...
        ceiling = _threshold(environ, FLAG_MAX_STALL)

    so the name never appears at an environ access. Resolving the helper is what makes
    the scan see them. Chasing the call is also what keeps the scan HONEST: a blanket
    "any module constant starting with M3NATIVE_" rule would drag in the gate's
    M3NATIVE_GATE_JSON_BEGIN stdout markers and demand a -e passthrough for a sentinel
    that is printed, never read.
    """

    def __init__(self):
        self.positions = {}

    def visit_FunctionDef(self, node):
        names = [a.arg for a in node.args.args]
        used = set()
        for inner in ast.walk(node):
            key = _environ_key(inner)
            if isinstance(key, ast.Name) and key.id in names:
                used.add(names.index(key.id))
        if used:
            self.positions[node.name] = used
        self.generic_visit(node)


def _environ_key(node):
    """The key expression of an environment access, or None."""
    if isinstance(node, ast.Subscript) and _is_environ(node.value):
        return node.slice
    if not isinstance(node, ast.Call) or not node.args:
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == 'getenv':
        return node.args[0]
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr == 'getenv' and isinstance(func.value, ast.Name) and func.value.id == 'os':
        return node.args[0]
    if func.attr != 'get':
        return None
    target = func.value
    if isinstance(target, ast.IfExp):
        if any(_is_environ(part) for part in (target.body, target.orelse)):
            return node.args[0]
        return None
    return node.args[0] if _is_environ(target) else None


class EnvReads(ast.NodeVisitor):
    """M3NATIVE_* names read from the environment, directly or through a helper."""

    def __init__(self, module):
        self.names = set()
        self.constants = {}
        for node in module.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                        self.constants[target.id] = node.value.value
        finder = _KeyParameters()
        finder.visit(module)
        self.positions = finder.positions

    def _record(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            self.names.add(node.value)
        elif isinstance(node, ast.Name):
            resolved = self.constants.get(node.id)
            if resolved is not None:
                self.names.add(resolved)

    def visit_Call(self, node):
        key = _environ_key(node)
        if key is not None:
            self._record(key)
        callee = node.func.id if isinstance(node.func, ast.Name) else None
        for index in self.positions.get(callee, ()):
            if index < len(node.args):
                self._record(node.args[index])
        self.generic_visit(node)

    def visit_Subscript(self, node):
        key = _environ_key(node)
        if key is not None:
            self._record(key)
        self.generic_visit(node)


def reads(path):
    module = ast.parse(path.read_text(encoding='utf-8'))
    visitor = EnvReads(module)
    visitor.visit(module)
    return {n for n in visitor.names if n.startswith('M3NATIVE_')}


class ArmEnvPassthroughTests(unittest.TestCase):
    def test_every_bench_env_read_is_passed_into_the_container(self):
        text = arm_text()
        through = passed_through(text)
        missing = {}
        for name in bench_scripts(text):
            for flag in sorted(reads(HERE / name) - through - HOST_ONLY):
                missing.setdefault(flag, []).append(name)
        self.assertEqual(missing, {}, 'read inside the container but never passed to it: %s'
                         % '; '.join('%s (%s)' % (k, ', '.join(v)) for k, v in sorted(missing.items())))

    def test_the_three_flags_that_run_35679222511_lost_are_wired(self):
        through = passed_through(arm_text())
        for name in ('M3NATIVE_PREFILL_CHUNK_TOKENS', 'M3NATIVE_TTFT_MAX_S',
                     'M3NATIVE_TTFT_MAX_STALL_S'):
            with self.subTest(flag=name):
                self.assertIn(name, through)

    def test_the_chunk_flag_actually_reaches_the_server_argv(self):
        """CHUNK_TOKENS is only worth passing because the gate turns it into the flag.
        If the gate stops building --enable-chunked-prefill from it, the passthrough is
        cargo and this says so."""
        gate = (HERE / 'lever_n_m3native_gate.py').read_text(encoding='utf-8')
        self.assertIn("os.environ.get('M3NATIVE_PREFILL_CHUNK_TOKENS')", gate)
        self.assertIn("'--enable-chunked-prefill'", gate)
        self.assertIn("'--no-enable-chunked-prefill'", gate)

    def test_the_scanner_finds_a_read_hidden_behind_a_constant(self):
        """The guard against the guard: a literal-only scan passes the broken arm."""
        found = reads(HERE / 'm3native_ttft_profile.py')
        self.assertIn('M3NATIVE_TTFT_MAX_STALL_S', found)
        self.assertIn('M3NATIVE_TTFT_MAX_S', found)

    def test_bench_scripts_are_derived_and_all_exist(self):
        names = bench_scripts(arm_text())
        self.assertIn('lever_n_m3native_gate.py', names)
        self.assertIn('m3native_ttft_profile.py', names)
        for name in names:
            with self.subTest(script=name):
                self.assertTrue((HERE / name).is_file(), '%s is mounted but absent' % name)


    def test_image_a_capacity_flags_reach_the_container_and_are_read_there(self):
        """The four 4 x 131k capacity flags (docs/four-streams-131k-feasibility-2026-09-23.md).
        Each host name must become its QWEN_FAST_* name at the docker run, and that name must
        be read by a module the image bakes - so a rename on either side fails here."""
        text = arm_text()
        baked = ''.join(path.read_text(encoding='utf-8') for path in sorted(HERE.glob('*.py'))
                        if not path.name.startswith('test_'))
        for name in CAPACITY_FLAGS:
            with self.subTest(flag=name):
                self.assertIn('${M3NATIVE_%s:+-e QWEN_FAST_%s=1}' % (name, name), text)
                self.assertIn("'QWEN_FAST_%s'" % name, baked)


class SdpaModesArmTests(unittest.TestCase):
    """M3NATIVE_SDPA_MODES and the sdpa_decode op-directory graft (optimisation/ttnn-op/
    sdpa_decode_qwen). The flag is translated, not passed by name, so the /bench scan above
    cannot see it; these pin both ends and the mount rules."""

    SDPA_DIR = '/opt/tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode'
    POOLED = 'dst=/experiment-scripts/ci/pooled_attention_replay.py,readonly'

    def test_the_modes_flag_becomes_the_env_var_the_replay_reader_and_the_gate_read(self):
        text = arm_text()
        self.assertIn('${M3NATIVE_SDPA_MODES:+-e QWEN_FAST_SDPA_MODES=$M3NATIVE_SDPA_MODES}', text)
        self.assertIn("SDPA_MODES_ENV = 'QWEN_FAST_SDPA_MODES'", (HERE / 'pooled_attention_replay.py').read_text(encoding='utf-8'))
        self.assertIn("environ.get('QWEN_FAST_SDPA_MODES')", (HERE / 'lever_n_m3native_gate.py').read_text(encoding='utf-8'))
        self.assertNotIn('M3NATIVE_SDPA_MODES', passed_through(text), 'translated to QWEN_FAST_SDPA_MODES, not passed by name')

    def test_the_reader_module_is_mounted_only_with_the_flag(self):
        text = arm_text()
        self.assertEqual(text.count(self.POOLED), 1)
        block = text[text.index('if [ -n "${M3NATIVE_SDPA_MODES:-}" ]; then'):]
        self.assertLess(block.index(self.POOLED), block.index(chr(10) + 'fi' + chr(10)))
        self.assertIn('"${sdpa_mode_mounts[@]}"', text)

    def test_the_op_directory_is_mounted_only_from_a_graft_that_has_one(self):
        text = arm_text()
        mount = '-v $KOPGRAFT64/sdpa_decode:%s:ro' % self.SDPA_DIR
        self.assertEqual(text.count(mount), 1)
        start = text.index('if [ -n "${KOPGRAFT64:-}" ]; then')
        end = text.index(chr(10) + 'fi' + chr(10), start)
        graft = text[start:end]
        self.assertIn('if [ -d "$KOPGRAFT64/sdpa_decode" ]; then', graft)
        self.assertLess(graft.index('if [ -d "$KOPGRAFT64/sdpa_decode" ]; then'), graft.index(mount))
        self.assertIn('-e TT_METAL_CACHE=$kernel_cache', text)
        self.assertIn('kernel_cache=/experiment-cache/kernels' + chr(10), text)
        self.assertNotIn('TT_METAL_CACHE=/experiment-cache/kernels ', text)

    def _run_graft_block(self, graft):
        """The arm's own KOPGRAFT64 block, executed by bash against a fake graft directory."""
        import shutil
        import subprocess
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        text = arm_text()
        start = text.index('KM=""')
        end = text.index(chr(10) + 'fi' + chr(10), text.index('if [ -n "${KOPGRAFT64:-}" ]; then')) + 4
        script = 'set -euo pipefail' + chr(10) + text[start:end] + 'printf "RESULT|%s|%s|%s" "$KM" "$kernel_cache" "$graft_binary_sha"' + chr(10)
        try:
            result = subprocess.run([bash, '-c', script], env=dict(PATH=os.environ.get('PATH', ''), KOPGRAFT64=graft),
                                    capture_output=True, text=True, timeout=60)
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)
        if result.returncode == 127 or 'sha256sum' in result.stderr and 'not found' in result.stderr:
            self.skipTest('bash lacks coreutils here')
        return result

    def test_the_graft_block_runs_unchanged_for_k64d_and_adds_the_directory_for_k64e(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).as_posix()
            old, new = root + '/k64d', root + '/k64e'
            for graft in (old, new):
                Path(graft).mkdir()
                Path(graft, '_ttnncpp.so').write_bytes(b'so')
            kernels = Path(new, 'sdpa_decode', 'device', 'kernels')
            for name, body in (('dataflow/reader_decode_qwen.cpp', b'reader'), ('compute/sdpa_flash_decode_qwen.cpp', b'compute')):
                (kernels / name).parent.mkdir(parents=True, exist_ok=True)
                (kernels / name).write_bytes(body)
            result = self._run_graft_block(old)
            if result.returncode != 0 and 'No such file' in result.stderr and ':' in root[:3]:
                self.skipTest('bash here does not share this filesystem view: %s' % result.stderr.strip())
            self.assertEqual(result.returncode, 0, result.stderr)
            km, cache, sha = result.stdout.split('RESULT|')[-1].split('|')
            self.assertNotIn('sdpa_decode', km)
            self.assertEqual(km.count(' -v '), 5)
            self.assertEqual(cache, '/experiment-cache/kernels')
            self.assertEqual(sha, hashlib.sha256(b'so').hexdigest())
            result = self._run_graft_block(new)
            self.assertEqual(result.returncode, 0, result.stderr)
            km, cache, sha = result.stdout.split('RESULT|')[-1].split('|')
            self.assertIn(' -v %s/sdpa_decode:%s:ro' % (new, self.SDPA_DIR), km)
            self.assertEqual(km.count(' -v '), 6)
            self.assertEqual(cache, '/experiment-cache/kernels-qwen-' + hashlib.sha256(b'readercompute').hexdigest()[:12])
            (kernels / 'compute/sdpa_flash_decode_qwen.cpp').unlink()
            result = self._run_graft_block(new)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('lacks compute/sdpa_flash_decode_qwen.cpp', result.stderr)


if __name__ == '__main__':
    unittest.main()
