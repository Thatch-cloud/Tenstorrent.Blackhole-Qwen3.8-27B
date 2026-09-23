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


class PrefillProfileArmTests(unittest.TestCase):
    """Prefill ranking M2: the profile block honours M3NATIVE_MAX_TOKENS (the single-user 131k
    prefill profile runs --max-tokens 1), takes its op-support count from
    M3NATIVE_PROFILE_OP_SUPPORT (default 20000), and passes QWEN_PREFILL_PROFILE_FLUSH=1 (only under
    M3NATIVE_PROFILE_FLUSH=1: its mid-prefill drain segfaulted in v131) so the
    grafted layer.py drains the profiler every 16 layers. The C1c/C1d switches cross as
    QWEN_FAST_C1_AGMM / QWEN_FAST_C1_LEGACY."""

    START = 'max_tokens="${M3NATIVE_MAX_TOKENS:-256}"'
    IF = 'if [ "${M3NATIVE_PROFILE:-}" = "1" ]; then'

    def _profile_block(self, environ):
        import shutil
        import subprocess
        import tempfile
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        text = arm_text()
        start = text.index(self.START)
        end = text.index(chr(10) + 'fi' + chr(10), text.index(self.IF, start)) + 4
        script = ('set -euo pipefail' + chr(10) + text[start:end]
                  + 'printf "RESULT|%s|%s" "$max_tokens" "${entry_args[*]}"' + chr(10))
        with tempfile.TemporaryDirectory() as directory:
            env = dict(PATH=os.environ.get('PATH', ''), **environ)
            try:
                result = subprocess.run([bash, '-c', script], env=env, cwd=directory,
                                        capture_output=True, text=True, timeout=60)
            except OSError as error:
                self.skipTest('bash unusable: %s' % error)
        return result

    def _run(self, **environ):
        result = self._profile_block(environ)
        self.assertEqual(result.returncode, 0, result.stderr)
        tokens, argv = result.stdout.split('RESULT|')[-1].split('|')
        return int(tokens), argv

    def test_max_tokens_is_honoured_in_profile_mode_and_defaults_are_kept(self):
        self.assertEqual(self._run()[0], 256)
        self.assertEqual(self._run(M3NATIVE_MAX_TOKENS='96')[0], 96)
        self.assertEqual(self._run(M3NATIVE_PROFILE='1')[0], 48)
        self.assertEqual(self._run(M3NATIVE_PROFILE='1', M3NATIVE_MAX_TOKENS='1')[0], 1)

    def test_op_support_count_defaults_to_20000_and_is_overridable(self):
        self.assertIn('--op-support-count 20000 ', self._run(M3NATIVE_PROFILE='1')[1])
        self.assertIn('--op-support-count 200000 ',
                      self._run(M3NATIVE_PROFILE='1', M3NATIVE_PROFILE_OP_SUPPORT='200000')[1])
        self.assertNotIn('tracy', self._run(M3NATIVE_PROFILE_OP_SUPPORT='200000')[1])
        code = [line for line in arm_text().splitlines() if not line.lstrip().startswith('#')]
        self.assertEqual([line for line in code if '--op-support-count' in line],
                         ['              --op-support-count "$op_support" -o /experiment-results-profile'])

    def test_a_bad_op_support_count_is_refused(self):
        for value in ('abc', '0', '-5', '2e5'):
            with self.subTest(value=value):
                result = self._profile_block(dict(M3NATIVE_PROFILE='1', M3NATIVE_PROFILE_OP_SUPPORT=value))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('M3NATIVE_PROFILE_OP_SUPPORT must be a positive integer', result.stderr)

    def test_the_flush_flag_and_the_c1_switches_cross_into_the_container(self):
        text = arm_text()
        self.assertNotIn('${M3NATIVE_PROFILE:+-e QWEN_PREFILL_PROFILE_FLUSH=1}', text)
        for line in ('${M3NATIVE_PROFILE_FLUSH:+-e QWEN_PREFILL_PROFILE_FLUSH=1}',
                     '${M3NATIVE_C1_AGMM:+-e QWEN_FAST_C1_AGMM=1}',
                     '${M3NATIVE_C1_LEGACY:+-e QWEN_FAST_C1_LEGACY=1}'):
            with self.subTest(line=line):
                self.assertEqual(text.count(line), 1)
                self.assertLess(text.index(line), text.index('--entrypoint python3'))
        graft = (HERE / 'lever_n_m3native_patch.py').read_text(encoding='utf-8')
        for name in ('QWEN_PREFILL_PROFILE_FLUSH', 'QWEN_FAST_C1_AGMM', 'QWEN_FAST_C1_LEGACY'):
            with self.subTest(read=name):
                self.assertIn("'%s'" % name, graft)


class RoundB1ArmTests(unittest.TestCase):
    """Build 1 of the round host-phase cuts: M3NATIVE_ROUND_B1=1 crosses as QWEN_FAST_ROUND_B1=1,
    on its own continued line right after C1_LEGACY's, before the entrypoint; unset, nothing
    crosses. The modules that read it are baked (test_dflash_round_b1.ShippingTests checks both
    image copy lists), so the flag does nothing on an image without build 1 - the gate's
    '[PINDIAG] round b1 engaged' requirement is what catches that."""

    LINE = '${M3NATIVE_ROUND_B1:+-e QWEN_FAST_ROUND_B1=1}'

    def test_the_switch_crosses_before_the_entrypoint_next_to_c1_legacy(self):
        text = arm_text()
        self.assertEqual(text.count(self.LINE), 1)
        self.assertLess(text.index(self.LINE), text.index('--entrypoint python3'))
        lines = text.split(chr(10))
        index = next(number for number, line in enumerate(lines) if self.LINE in line)
        self.assertEqual(lines[index - 1].strip(), '${M3NATIVE_C1_LEGACY:+-e QWEN_FAST_C1_LEGACY=1} ' + chr(92))
        self.assertEqual(lines[index].strip(), self.LINE + ' ' + chr(92), 'nothing else on the continued line')

    def test_its_audit_crosses_on_the_next_line(self):
        """M3NATIVE_ROUND_B1_AUDIT=1 crosses as QWEN_FAST_ROUND_B1_AUDIT=1 (the correctness arm's
        shadow check; it does nothing without QWEN_FAST_ROUND_B1), right after the B1 line."""
        audit = '${M3NATIVE_ROUND_B1_AUDIT:+-e QWEN_FAST_ROUND_B1_AUDIT=1}'
        text = arm_text()
        self.assertEqual(text.count(audit), 1)
        lines = text.split(chr(10))
        index = next(number for number, line in enumerate(lines) if audit in line)
        self.assertEqual(lines[index - 1].strip(), self.LINE + ' ' + chr(92))
        self.assertEqual(lines[index].strip(), audit + ' ' + chr(92), 'nothing else on the continued line')
        self.assertLess(text.index(audit), text.index('--entrypoint python3'))
        for module in ('dflash_device.py', 'draft_kv_history.py'):
            with self.subTest(module=module):
                self.assertIn("os.environ.get('QWEN_FAST_ROUND_B1_AUDIT') == '1'",
                              (HERE / module).read_text(encoding='utf-8'))
        self.assertIn("ROUND_B1_AUDIT_FLAG = 'QWEN_FAST_ROUND_B1_AUDIT'",
                      (HERE / 'dflash_packed_proposal.py').read_text(encoding='utf-8'))

    def test_baked_modules_read_it(self):
        for module in ('dflash_device.py', 'draft_kv_history.py', 'dflash_proposal_trace.py',
                       'dflash_traced_publish.py', 'serving_packed_step.py',
                       'dflash_packed_proposal_coordinator.py'):
            with self.subTest(module=module):
                source = (HERE / module).read_text(encoding='utf-8')
                self.assertIn("os.environ.get('QWEN_FAST_ROUND_B1') == '1'", source)
        self.assertIn("ROUND_B1_FLAG = 'QWEN_FAST_ROUND_B1'", (HERE / 'dflash_packed_proposal.py').read_text(encoding='utf-8'))


class LegacyContinuationArmTests(unittest.TestCase):
    """Lever N's negative control: M3NATIVE_LEGACY_CONTINUATION_ORDER=1 crosses as
    QWEN_FAST_LEGACY_CONTINUATION_ORDER=1, which serving_lifecycle (the pre-fix routing order)
    and serving_worker_hook (the mixed-step pass-through) read. Unset, nothing crosses."""

    LINE = '${M3NATIVE_LEGACY_CONTINUATION_ORDER:+-e QWEN_FAST_LEGACY_CONTINUATION_ORDER=1}'

    def test_the_switch_crosses_before_the_entrypoint(self):
        text = arm_text()
        self.assertEqual(text.count(self.LINE), 1)
        self.assertLess(text.index(self.LINE), text.index('--entrypoint python3'))

    def test_both_image_modules_read_it(self):
        for module in ('serving_lifecycle.py', 'serving_worker_hook.py'):
            with self.subTest(module=module):
                source = (HERE / module).read_text(encoding='utf-8')
                self.assertIn("os.environ.get('QWEN_FAST_LEGACY_CONTINUATION_ORDER') == '1'", source)


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
