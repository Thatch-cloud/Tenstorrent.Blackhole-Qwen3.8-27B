"""Which board a qualification harness runs on, and which boards the CI gate touches.

scripts/ci/qual_card.sh picks the target by QUAL_CARD, a board id (default card B, the PCIe-only
qualification card), refuses card M and card A (the serving pair) unless ALLOW_SERVING_CARD=1,
resolves the node by board id at launch and again right before docker run, refuses only while
something can reach THAT card, and prints a reset hint whose tt-smi -r command resolves the board id
when it is run (never a bare index, which tt-smi reads as its own renumbering board index). Every
single-card harness under optimisation/ttnn-op embeds it byte for byte; the scripts/ci rig runners
source it. scripts/ci/serving_pair.sh gives the m3native gate its holder-check and reset targets: card
M and card A only, by board id, checked against their PCI addresses, and the reset step's self-heal (a
driver re-probe of a serving card that came back from the reset without its by-id link, and a rescan of
the upstream port of one whose PCI device did not come back at all). The gate's two
steps are run here as the runner would run them, against a fake rig with fake sudo, fuser and tt-smi, a
fake /sys and a fake driver.

The shell tests need bash (Git Bash on Windows) and skip without it. The library tests stub the
device tree, readlink, device numbers, sysfs, docker and fuser with shell functions, so they run
anywhere; the sysfs test builds a fake /sys with real symlinks and skips where none can be made.

    py -3.11 -B scripts/ci/test_qual_card.py --sync    # re-copy qual_card.sh into every harness
"""

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
ROOT = HERE.parent.parent
LIBRARY = HERE / 'qual_card.sh'
SERVING_PAIR = HERE / 'serving_pair.sh'
ARM = HERE / 'lever_n_m3native_run_arm.sh'
GATE = ROOT / '.github' / 'workflows' / 'qwen-lever-n-m3native-gate.yml'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
OPS = ROOT / 'optimisation' / 'ttnn-op'

# Harnesses that carry the block (they are staged to the rig on their own).
EMBEDDING = [
    OPS / 'sdpa_prefill_bench' / 'run_m1.sh',
    OPS / 'sdpa_prefill_bench' / 'k0_session.sh',
    OPS / 'sdpa_prefill_chain' / 'run_card_m_pf.sh',
    OPS / 'gdn_prefill_conv' / 'run_card_m.sh',
    OPS / 'sdpa_decode_qwen' / 'run_card_m.sh',
    OPS / 'verify_t2' / 'run_card_m.sh',
    OPS / 'c1e_gateup' / 'run_card_b.sh',
    OPS / 'draft_slide_inplace' / 'run_card_b.sh',
    OPS / 'sdpa_decode_qwen' / 'run_probe_k1.sh',
    OPS / 'sdpa_decode_slice' / 'run_card_b.sh',
    OPS / 'pair_row_probe' / 'run_card_b.sh',
    OPS / 'kernels-batch64' / 'attn_prep' / 'build-and-test-b64.sh',
    OPS / 'kernels-batch64' / 'nlp_concat_heads_decode' / 'build-and-test-b64.sh',
]
# Rig runners in scripts/ci: they source the file beside them.
SOURCING = [HERE / 'verify-t1-g0-rig.sh', HERE / 'matmul64_sweep_rig.sh', HERE / 'gdn-user-batch-rig.sh',
            HERE / 'sdpa_bench_rig.sh']
# The line each embedding harness launches its device container with (k0_session.sh launches none:
# every run goes through run_m1.sh).
LAUNCH = {
    OPS / 'sdpa_prefill_bench' / 'run_m1.sh': r'^timeout -k 30 "\$timeout_s" "\$\{argv\[@\]\}"',
    OPS / 'sdpa_prefill_chain' / 'run_card_m_pf.sh': r'^timeout -k 30 "\$timeout_s" "\$\{argv\[@\]\}"',
    OPS / 'gdn_prefill_conv' / 'run_card_m.sh': r'^timeout -k 30 "\$timeout_s" docker run',
    OPS / 'sdpa_decode_qwen' / 'run_card_m.sh': r'^timeout -k 30 "\$timeout_s" docker run',
    OPS / 'verify_t2' / 'run_card_m.sh': r'^timeout -k 30 "\$timeout_s" docker run',
    OPS / 'c1e_gateup' / 'run_card_b.sh': r'^timeout -k 30 "\$timeout_s" docker run',
    OPS / 'draft_slide_inplace' / 'run_card_b.sh': r'^timeout -k 30 "\$timeout_s" docker run',
    OPS / 'sdpa_decode_qwen' / 'run_probe_k1.sh': r'^timeout -k 30 "\$timeout_s" "\$\{argv\[@\]\}"',
    OPS / 'sdpa_decode_slice' / 'run_card_b.sh': r'^timeout -k 30 "\$timeout_s" "\$\{argv\[@\]\}"',
    OPS / 'pair_row_probe' / 'run_card_b.sh': r'^timeout -k 30 "\$timeout_s" "\$\{argv\[@\]\}"',
    OPS / 'kernels-batch64' / 'attn_prep' / 'build-and-test-b64.sh': r'^docker run -d --name "\$CONTAINER"',
    OPS / 'kernels-batch64' / 'nlp_concat_heads_decode' / 'build-and-test-b64.sh': r'^timeout 900 docker run',
}

CARD_M = 'blackhole-CEF5729692C19E6D'
CARD_A = 'blackhole-3707293C249A5E67'
CARD_B = 'blackhole-F36F768B9A5CAFA0'
PCI_M, PCI_A, PCI_B = '0000:d1:00.0', '0000:f3:00.0', '0000:f4:00.0'
# Their upstream ports: card M's AMD root port, card A's and card B's PCIe switch downstream ports.
PORT_M, PORT_A, PORT_B = '0000:d0:01.1', '0000:f2:00.0', '0000:f2:01.0'
RUNNER = 'thatch-build-amd64-02-cp-temp'
BEGIN, END = '# >>> qual_card.sh', '# <<< qual_card.sh'
NL = chr(10)
SCRUB = ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'M1_READER', 'IMAGE', 'M1_ARGS', 'M1_DRY_RUN', 'M1_REQUIRE_SOURCES',
         'RESULTS', 'M1_SRC', 'KOPGRAFT_PF', 'PF_SRC', 'PF_DRY_RUN', 'WATCHER', 'REFERENCE', 'CARD', 'NAME',
         'CONTAINER', 'GDN_USER_BATCH_DEVICE', 'K64F_SRC', 'KOPGRAFT64', 'REPO', 'RUNNER_NAME', 'FAKE_HELD', 'MSYS',
         'PROBE_DRY_RUN', 'EXPECT_TTNNCPP_SHA256', 'K64I_DRY_RUN', 'KOPGRAFT64_REFERENCE', 'PAIR_ROW_DRY_RUN')


def read(path):
    return path.read_text(encoding='utf-8')


def block_span(text):
    """(start, end) of the embedded block: the BEGIN line through the END line, newline included."""
    starts = [m.start() for m in re.finditer('^' + re.escape(BEGIN), text, flags=re.M)]
    ends = [m.start() for m in re.finditer('^' + re.escape(END), text, flags=re.M)]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError('expected exactly one %s ... %s block' % (BEGIN, END))
    end = text.index(NL, ends[0]) + 1
    return starts[0], end


def canonical():
    return read(LIBRARY)


def sync():
    for path in EMBEDDING:
        text = read(path)
        start, end = block_span(text)
        updated = text[:start] + canonical() + text[end:]
        if updated != text:
            path.write_bytes(updated.encode('utf-8'))
            print('synced %s' % path.relative_to(ROOT).as_posix())


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


def clean_env(**extra):
    env = dict(os.environ)
    for name in SCRUB:
        env.pop(name, None)
    env.update(extra)
    return env


def run(argv, timeout=120, **env):
    return subprocess.run([BASH] + [str(part) for part in argv], env=clean_env(**env), capture_output=True,
                          text=True, encoding='utf-8', errors='replace', timeout=timeout)


def q(path):
    return shlex.quote(Path(path).as_posix())


def symlinks_work():
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.symlink(directory, os.path.join(directory, 'probe'), target_is_directory=True)
        except (OSError, NotImplementedError):
            return False
    return True


class FakeRig:
    """A stand-in /dev/tenstorrent with three boards, and shell stubs for everything the library
    reads: readlink (by-id entries are files naming their node), device numbers (a node is
    ea:<number>; any other file carries its own in <file>.majmin), sysfs (pci/<node> files: the node's
    PCI address), docker (canned containers), fuser (a log, and holders on the nodes in FAKE_HELD)
    and sleep (a fake clock: each call brings a board that is 'pending' one poll closer)."""

    def __init__(self, directory, nodes=None, pci=None):
        self.dir = Path(directory)
        self.tt = self.dir / 'dev' / 'tenstorrent'
        (self.tt / 'by-id').mkdir(parents=True)
        (self.dir / 'pci').mkdir()
        # Renumbered on purpose: card B is node 0, card M node 2, card A node 1 (not PCI order).
        self.nodes = nodes or {CARD_B: '0', CARD_A: '1', CARD_M: '2'}
        self.pci = pci or {CARD_M: PCI_M, CARD_A: PCI_A, CARD_B: PCI_B}
        for card, node in self.nodes.items():
            (self.tt / node).write_text('')
            (self.tt / 'by-id' / card).write_text(self.node(card))
            (self.dir / 'pci' / node).write_text(self.pci.get(card, '') + NL)
        self.containers = []

    def node(self, card):
        return (self.tt / self.nodes[card]).as_posix()

    def byid(self, card):
        return (self.tt / 'by-id' / card).as_posix()

    def container(self, name, privileged=False, devices=(), rules=(), requests=(), mounts=()):
        cid = 'c%d' % len(self.containers)
        lines = ['/%s %s' % (name, 'true' if privileged else 'false')]
        lines += ['dev ' + path for path in devices] + ['rule ' + rule for rule in rules]
        lines += ['req ' + request for request in requests] + ['mnt ' + path for path in mounts]
        (self.dir / ('container-' + cid)).write_text(NL.join(lines) + NL)
        self.containers.append(cid)
        return cid

    def gate_arm(self):
        """What lever_n_m3native_run_arm.sh gives its container: card M and card A, /dev/tenstorrent
        read-only (for device-owners.py), the hugepages mount."""
        return self.container('qwen-m3native-1-1', devices=(self.node(CARD_M), self.node(CARD_A)),
                              mounts=(self.tt.as_posix(), '/dev/hugepages-1G', '/home/thatch/hf-cache'))

    def device_copy(self, card):
        """A device file outside /dev/tenstorrent with the same numbers as the card's node."""
        path = self.dir / 'dev' / ('copy-of-' + self.nodes[card])
        path.write_text('')
        Path(str(path) + '.majmin').write_text('ea:%s' % self.nodes[card] + NL)
        return path.as_posix()

    def sysfs(self):
        """Shell lines that build the fake /sys the self-heal reads (FAKE_DIR/sys): a PCI device per board
        with an address, each bound to the tenstorrent driver, and the three boards' upstream ports. Built
        by bash, not Python: the addresses hold ':', which only the shell's own path mapping stores on
        Windows."""
        pci = self.dir / 'sys' / 'bus' / 'pci'
        lines = ['mkdir -p %s' % q(pci / 'drivers' / 'tenstorrent'),
                 ': > %s' % q(pci / 'drivers' / 'tenstorrent' / 'bind'),
                 ': > %s' % q(pci / 'drivers' / 'tenstorrent' / 'unbind')]
        for address in sorted(set(filter(None, self.pci.values()))):
            lines.append('mkdir -p %s %s' % (q(pci / 'devices' / address), q(pci / 'drivers' / 'tenstorrent' / address)))
        lines.append('mkdir -p %s' % ' '.join(q(pci / 'devices' / port) for port in (PORT_M, PORT_A, PORT_B)))
        return lines

    def stubs(self):
        return [
            'QUAL_TT_ROOT=%s' % q(self.tt),
            'QUAL_BYID_ROOT=$QUAL_TT_ROOT/by-id',
            'QUAL_SYS_ROOT=%s' % q(self.dir / 'sys'),
            'FAKE_DIR=%s' % q(self.dir),
            'qual_is_char() { [ -f "$1" ]; }',
            'readlink() { local p=${@: -1}; case $p in */by-id/*) [ -f "$p" ] && cat "$p" ;; *) printf "%s\\n" "$p" ;; esac; }',
            'qual_majmin_of() { case ${1:-} in "$QUAL_TT_ROOT"/by-id/*) qual_majmin_of "$(readlink -f -- "$1")" ;; '
            '"$QUAL_TT_ROOT"/[0-9]*) [ -f "$1" ] && echo "ea:${1##*/}" ;; ?*) [ -f "$1.majmin" ] && cat "$1.majmin" ;; esac; '
            'return 0; }',
            'qual_pci_of() { case ${1:-} in "$QUAL_TT_ROOT"/[0-9]*) cat "$FAKE_DIR/pci/${1##*/}" 2>/dev/null || true ;; esac; }',
            'sleep() { local n; [ -f "$FAKE_DIR/pending" ] || return 0; n=$(cat "$FAKE_DIR/pending"); '
            'if [ "$n" -le 1 ]; then rm -f "$FAKE_DIR/pending"; . "$FAKE_DIR/comeback.sh"; '
            'else echo $((n - 1)) > "$FAKE_DIR/pending"; fi; }',
        ]

    def preamble(self, library=LIBRARY):
        return NL.join([
            'set -euo pipefail',
            '. %s' % q(library),
        ] + self.stubs() + [
            'docker() { case $1 in ps) printf "%%s\\n" %s ;; inspect) cat "$FAKE_DIR/container-$2" ;; esac; }'
            % ' '.join(self.containers),
            'fuser() { echo "$*" >> "$FAKE_DIR/fuser.log"; local n=${@: -1}; '
            'case " ${FAKE_HELD:-} " in *" $n "*) echo "$n: thatch 4242 F.... python3" >&2; return 0 ;; esac; return 1; }',
            'sudo() { return 1; }',
            'id() { echo 1000; }',
            '',
        ])

    def pending(self, card, node, polls):
        """Card comes back as `node` after `polls` sleeps (it is gone until then)."""
        (self.tt / 'by-id' / card).unlink()
        target = (self.tt / node).as_posix()
        (self.dir / 'pending').write_text('%d' % polls + NL)
        (self.dir / 'comeback.sh').write_text(NL.join([
            ': > %s' % q(target),
            "printf '%%s' %s > %s" % (q(target), q(self.tt / 'by-id' / card)),
            'echo %s > %s' % (self.pci[card], q(self.dir / 'pci' / node)),
        ]) + NL)
        self.nodes[card] = node

    def run(self, body, library=LIBRARY, **env):
        script = self.dir / 'case.sh'
        script.write_bytes((self.preamble(library) + body + NL).encode('utf-8'))
        return run([script.as_posix()], **env)

    def fuser_calls(self):
        log = self.dir / 'fuser.log'
        return log.read_text().splitlines() if log.is_file() else []


class EmbeddingTests(unittest.TestCase):
    def test_every_harness_embeds_the_canonical_block_byte_for_byte(self):
        library = canonical()
        self.assertTrue(library.startswith(BEGIN))
        self.assertTrue(library.rstrip(NL).splitlines()[-1].startswith(END))
        for path in EMBEDDING:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                text = read(path)
                start, end = block_span(text)
                self.assertEqual(text[start:end], library,
                                 'drifted: run py -3.11 -B scripts/ci/test_qual_card.py --sync')
                self.assertTrue(text[end:].startswith('qual_card_select' + NL))    # selected before any use

    def test_the_scripts_ci_runners_source_it_and_select_the_card(self):
        for path in SOURCING:
            with self.subTest(path=path.name):
                text = read(path)
                self.assertIn('. "$here/qual_card.sh"', text)
                self.assertIn('qual_card_select', text)
                self.assertIn('qual_card_resolve', text)
                launch = re.search(r'^timeout [^\n]*docker run', text, flags=re.M).start()
                holders = re.search(r'^\s*qual_refuse_holders$', text, flags=re.M).start()
                recheck = re.search(r'^\s*qual_card_recheck\b', text, flags=re.M).start()
                self.assertLess(holders, recheck)
                self.assertLess(recheck, launch)
                self.assertNotIn('docker run', text[recheck:launch])
                self.assertRegex(text, r'(device=\$QUAL_NODE|devices\+=\(--device "\$QUAL_NODE"\))')

    def test_every_harness_rechecks_the_node_right_before_its_launch(self):
        for path, pattern in LAUNCH.items():
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                text = read(path)
                start, end = block_span(text)
                code = text[end:]
                launches = [m.start() for m in re.finditer(pattern, code, flags=re.M)]
                self.assertEqual(len(launches), 1, pattern)
                rechecks = [m.start() for m in re.finditer(r'^qual_card_recheck\b', code, flags=re.M)]
                self.assertEqual(len(rechecks), 1)
                holders = [m.start() for m in re.finditer(r'^\s*qual_refuse_holders$', code, flags=re.M)]
                self.assertTrue(holders and holders[-1] < rechecks[0] < launches[0])
                self.assertNotIn('docker run', code[rechecks[0]:launches[0]])
                self.assertNotIn('qual_card_resolve', code[rechecks[0]:launches[0]])

    def test_no_harness_names_a_board_or_a_node_number_outside_the_block(self):
        for path in EMBEDDING + SOURCING:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                text = read(path)
                if path in EMBEDDING:
                    start, end = block_span(text)
                    text = text[:start] + text[end:]
                code = [line for line in text.splitlines() if not line.lstrip().startswith('#')]
                for board in (CARD_M, CARD_A):
                    self.assertNotIn(board, NL.join(code))
                self.assertEqual([line for line in code if re.search(r'/dev/tenstorrent/[0-9]', line)], [])
                self.assertEqual([line for line in code if re.search(r'tt-smi -r [0-9]', line)], [])
                self.assertNotIn('CARD_M=', text)
                self.assertNotIn('ls /dev/tenstorrent', text)
                self.assertNotIn(chr(13), read(path))

    def test_the_library_never_resets_and_never_maps_a_device(self):
        text = canonical()
        for line in text.splitlines():
            if 'tt-smi' in line:
                self.assertTrue(line.lstrip().startswith(('#', 'echo ')), line)
        self.assertNotRegex(text, r'tt-smi -r [0-9<]')                 # never a bare index, not even a placeholder
        self.assertNotIn('--device ', text)            # the harness tests count their one --device
        self.assertNotIn('docker run', text)
        self.assertIn("QUAL_CARD_B=%s" % CARD_B, text)
        self.assertIn("QUAL_SERVING_CARDS='%s %s'" % (CARD_M, CARD_A), text)
        self.assertIn('{{range .HostConfig.DeviceRequests}}req {{.Driver}} {{println .DeviceIDs}}{{end}}', text)

    def test_the_serving_pair_is_the_arms_pair(self):
        arm = read(ARM)
        default = re.search(r'serving_cards="\$\{M3NATIVE_CARDS:-([^}]*)\}"', arm).group(1)
        library = re.search(r"^QUAL_SERVING_CARDS='([^']*)'", canonical(), flags=re.M).group(1)
        self.assertEqual(default.split(), library.split())
        self.assertEqual(library.split(), [CARD_M, CARD_A])

    @unittest.skipUnless(BASH, 'bash not found')
    def test_every_touched_script_parses(self):
        for path in EMBEDDING + SOURCING + [LIBRARY, SERVING_PAIR]:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                result = run(['-n', path.as_posix()])
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_cpu_suite_runs_this_module(self):
        self.assertRegex(read(CPU_WORKFLOW), r'python -B -m unittest [^\n]*\btest_qual_card\b')


@unittest.skipUnless(BASH, 'bash not found')
class SelectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rig = FakeRig(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    SHOW = 'qual_card_select; echo "card=$QUAL_CARD tag=$QUAL_TAG serving=$QUAL_SERVING byid=$QUAL_BYID"'

    def test_the_default_target_is_card_b(self):
        result = self.rig.run(self.SHOW)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('card=%s tag=card-b serving=0 byid=%s' % (CARD_B, self.rig.byid(CARD_B)), result.stdout)
        self.assertEqual(result.stderr, '')

    def test_the_serving_pair_is_refused_without_the_override(self):
        for card, label in ((CARD_M, 'card M'), (CARD_A, 'card A')):
            with self.subTest(card=card):
                result = self.rig.run(self.SHOW, QUAL_CARD=card)
                self.assertEqual(result.returncode, 1)
                self.assertIn('refusing: QUAL_CARD=%s is %s, half of the serving pair' % (card, label), result.stderr)
                self.assertIn('ALLOW_SERVING_CARD=1 overrides', result.stderr)
                self.assertNotIn('card=', result.stdout)
                zero = self.rig.run(self.SHOW, QUAL_CARD=card, ALLOW_SERVING_CARD='0')
                self.assertEqual(zero.returncode, 1)

    def test_the_override_runs_on_a_serving_card_with_a_loud_warning(self):
        result = self.rig.run(self.SHOW, QUAL_CARD=CARD_M, ALLOW_SERVING_CARD='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('card=%s tag=card-m serving=1' % CARD_M, result.stdout)
        self.assertIn('WARNING: ALLOW_SERVING_CARD=1: this run is on %s, card M, half of the serving pair' % CARD_M,
                      result.stderr)
        self.assertIn('qwen-two-p150a-exclusive', result.stderr)

    def test_a_path_or_a_node_number_is_not_a_board_id(self):
        for value in ('../2', '/dev/tenstorrent/2', 'by-id/%s' % CARD_M, '.%s' % CARD_B, 'a b', '..'):
            with self.subTest(value=value):
                result = self.rig.run(self.SHOW, QUAL_CARD=value)
                self.assertEqual(result.returncode, 1)
                self.assertIn('is not a board id', result.stderr)

    def test_resolve_finds_the_node_by_board_id_now(self):
        result = self.rig.run('qual_card_select; qual_card_resolve; echo "node=$QUAL_NODE pci=$QUAL_PCI"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('node=%s pci=%s' % (self.rig.node(CARD_B), PCI_B), result.stdout)
        self.assertIn('### target card: %s (card B, the qualification card) -> %s, PCI %s'
                      % (CARD_B, self.rig.node(CARD_B), PCI_B), result.stdout)
        # Only the m3native gate spares card B: the other workflows are named, not implied safe.
        self.assertIn('qwen-card-reset.yml and most other hardware workflows', result.stdout)

    def test_a_missing_board_is_refused(self):
        (self.rig.tt / 'by-id' / CARD_B).unlink()
        result = self.rig.run('qual_card_select; qual_card_resolve; echo resolved')
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing: %s (card B, the qualification card) has no device node here' % CARD_B, result.stderr)
        self.assertNotIn('resolved', result.stdout)

    def test_an_alias_of_a_serving_card_is_that_serving_card(self):
        (self.rig.tt / 'by-id' / 'blackhole-0000000000000000').write_text(self.rig.node(CARD_M))
        body = 'qual_card_select; qual_card_resolve; echo "serving=$QUAL_SERVING"'
        refused = self.rig.run(body, QUAL_CARD='blackhole-0000000000000000')
        self.assertEqual(refused.returncode, 1)
        self.assertIn('the node of %s (card M' % CARD_M, refused.stderr)
        self.assertIn('ALLOW_SERVING_CARD=1 overrides', refused.stderr)
        allowed = self.rig.run(body, QUAL_CARD='blackhole-0000000000000000', ALLOW_SERVING_CARD='1')
        self.assertEqual(allowed.returncode, 0, allowed.stderr)
        self.assertIn('serving=1', allowed.stdout)


@unittest.skipUnless(BASH, 'bash not found')
class RecheckTests(unittest.TestCase):
    """Right before docker run the board id is read again: a re-enumeration between the holder check
    and the launch (a switch event, the gate resetting card A beside card B) refuses the launch."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rig = FakeRig(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, move=''):
        return self.rig.run('qual_card_select; qual_card_resolve; %s qual_card_recheck; echo LAUNCH' % move)

    def test_an_unmoved_board_launches(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('### %s is still %s' % (CARD_B, self.rig.node(CARD_B)), result.stdout)
        self.assertIn('LAUNCH', result.stdout)

    def test_a_board_that_moved_is_refused(self):
        # Card B now on node 2, the node card M had at the holder check.
        result = self.check('printf %%s %s > "$QUAL_BYID";' % q(self.rig.node(CARD_M)))
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing: %s (card B, the qualification card) was %s at the holder check and is %s now'
                      % (CARD_B, self.rig.node(CARD_B), self.rig.node(CARD_M)), result.stderr)
        self.assertNotIn('LAUNCH', result.stdout)

    def test_a_board_that_vanished_is_refused(self):
        result = self.check('rm -f "$QUAL_BYID";')
        self.assertEqual(result.returncode, 1)
        self.assertIn('is gone now', result.stderr)
        self.assertNotIn('LAUNCH', result.stdout)

    def test_a_named_card_and_node(self):
        body = 'qual_card_recheck %s %s; echo LAUNCH' % (CARD_M, q(self.rig.node(CARD_M)))
        self.assertIn('LAUNCH', self.rig.run(body).stdout)
        wrong = self.rig.run('qual_card_recheck %s %s; echo LAUNCH' % (CARD_M, q(self.rig.node(CARD_A))))
        self.assertEqual(wrong.returncode, 1)
        self.assertIn('refusing: %s (card M, half of the serving pair) was %s' % (CARD_M, self.rig.node(CARD_A)),
                      wrong.stderr)


@unittest.skipUnless(BASH, 'bash not found')
class HolderTests(unittest.TestCase):
    """The holder check is scoped to the target: a CI gate on the serving pair does not block card B,
    anything that can reach card B does; with a serving target, any container on any card blocks."""

    BODY = 'qual_card_select; qual_card_resolve; qual_refuse_holders; echo CLEAR'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rig = FakeRig(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, **env):
        return self.rig.run(self.BODY, **env)

    def test_a_ci_gate_on_the_serving_pair_does_not_block_card_b(self):
        self.rig.gate_arm()
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CLEAR', result.stdout)
        self.assertIn('### containers: none can reach %s' % self.rig.node(CARD_B), result.stdout)
        self.assertEqual(self.rig.fuser_calls(), ['-v ' + self.rig.node(CARD_B)])   # the target's node only

    def test_the_same_gate_blocks_a_serving_target(self):
        self.rig.gate_arm()
        for card in (CARD_M, CARD_A):
            with self.subTest(card=card):
                result = self.check(QUAL_CARD=card, ALLOW_SERVING_CARD='1')
                self.assertEqual(result.returncode, 1)
                self.assertIn('refusing: container /qwen-m3native-1-1 has', result.stderr)
                self.assertIn('the target is a serving card', result.stderr)

    def test_any_container_on_any_card_blocks_a_serving_target(self):
        self.rig.container('qual-b', devices=(self.rig.node(CARD_B),))
        result = self.check(QUAL_CARD=CARD_M, ALLOW_SERVING_CARD='1')
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing: container /qual-b', result.stderr)

    def test_whatever_can_reach_card_b_blocks_card_b(self):
        cases = (
            (lambda rig: dict(devices=(rig.node(CARD_B),)), 'which is {node} (the target)'),
            (lambda rig: dict(devices=(rig.byid(CARD_B),)), 'which is {node} (the target)'),
            (lambda rig: dict(mounts=(rig.byid(CARD_B),)), 'among its mnts, which is {node}'),
            (lambda rig: dict(privileged=True), 'is --privileged'),
            (lambda rig: dict(rules=('c 234:* rwm',)), 'has a device cgroup rule (c 234:* rwm)'),
            (lambda rig: dict(requests=('cdi [tenstorrent.com/device=0]',)),
             'has a device request for a Tenstorrent device (cdi [tenstorrent.com/device=0])'),
            (lambda rig: dict(devices=(rig.device_copy(CARD_B),)), "a device with the target's numbers (ea:0)"),
            (lambda rig: dict(devices=(rig.tt.as_posix(),)), "a directory holding the target's node"),
            (lambda rig: dict(devices=(rig.tt.as_posix() + '/7',)), 'which is not a device node now'),
            (lambda rig: dict(devices=((rig.tt / 'by-id').as_posix(),)), 'which is not a device node now'),
        )
        for index, (options, why) in enumerate(cases):
            with self.subTest(case=index, why=why):
                rig = FakeRig(tempfile.mkdtemp(dir=self.tmp.name))
                rig.gate_arm()
                rig.container('holder', **options(rig))
                why = why.format(node=rig.node(CARD_B))
                result = rig.run(self.BODY)
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn('refusing: container /holder', result.stderr)
                self.assertIn(why, result.stderr)
                self.assertNotIn('CLEAR', result.stdout)

    def test_what_cannot_reach_card_b_does_not_block_it(self):
        self.rig.gate_arm()
        self.rig.container('gpu', requests=('nvidia [0]',))                       # not a Tenstorrent request
        self.rig.container('copy-of-m', devices=(self.rig.device_copy(CARD_M),))   # card M's numbers
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CLEAR', result.stdout)

    def test_a_host_holder_of_the_target_blocks_and_one_of_another_card_does_not(self):
        held = self.check(FAKE_HELD=self.rig.node(CARD_B))
        self.assertEqual(held.returncode, 1)
        self.assertIn('refusing: host processes hold %s' % self.rig.node(CARD_B), held.stderr)
        self.assertEqual(len(self.rig.fuser_calls()), 5)                    # retried, then refused
        other = FakeRig(tempfile.mkdtemp(dir=self.tmp.name))
        result = other.run(self.BODY, FAKE_HELD=other.node(CARD_M) + ' ' + other.node(CARD_A))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(other.fuser_calls(), ['-v ' + other.node(CARD_B)])


@unittest.skipUnless(BASH, 'bash not found')
class ResetHintTests(unittest.TestCase):
    """The hint never prints a bare number: tt-smi -r reads one as tt-smi's own board index, which
    renumbers across resets. It prints a command that resolves the board id when it is run, and the
    PCI address that identifies the board's row in tt-smi -ls."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rig = FakeRig(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def hint(self, rig=None, **env):
        result = (rig or self.rig).run('qual_card_select; qual_reset_hint', **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotRegex(result.stdout, r'tt-smi -r [0-9<]')
        return result.stdout

    def command(self, out):
        lines = [line.strip() for line in out.splitlines() if '~/.local/bin/tt-smi -r "$' in line]
        self.assertEqual(len(lines), 1, out)
        return lines[0]

    def execute(self, command):
        body = 'smi() { echo "ARGS $*"; }; %s; echo DONE' % command.replace('~/.local/bin/tt-smi', 'smi')
        result = self.rig.run(body)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('DONE', result.stdout)
        return [line for line in result.stdout.splitlines() if line.startswith('ARGS')]

    def test_card_b_is_reset_by_its_board_id_and_confirmed_by_its_pci_row(self):
        out = self.hint()
        self.assertIn('It is %s now, PCI %s;' % (self.rig.node(CARD_B), PCI_B), out)
        self.assertIn("CONFIRM its row (PCI BDF %s): ~/.local/bin/tt-smi -ls | grep -i 'f4:00.0'" % PCI_B, out)
        self.assertEqual(self.command(out), 'n=$(readlink -e %s) && ~/.local/bin/tt-smi -r "$n"' % self.rig.byid(CARD_B))
        self.assertIn('Never card M or card A', out)
        self.assertNotIn(CARD_M, out)

    def test_the_printed_command_resolves_the_board_id_when_it_is_run(self):
        command = self.command(self.hint())
        # The boards renumber after the hint was printed: card B is node 5 by the time it is run.
        (self.rig.tt / '5').write_text('')
        (self.rig.tt / 'by-id' / CARD_B).write_text((self.rig.tt / '5').as_posix())
        self.assertEqual(self.execute(command), ['ARGS -r %s' % (self.rig.tt / '5').as_posix()])

    def test_the_printed_command_never_runs_tt_smi_without_the_board(self):
        # An empty argument would make tt-smi -r reset every board: with the board id gone, no call at all.
        command = self.command(self.hint())
        (self.rig.tt / 'by-id' / CARD_B).unlink()
        self.assertEqual(self.execute(command), [])
        pair = self.command(self.hint(QUAL_CARD=CARD_M, ALLOW_SERVING_CARD='1'))
        self.assertEqual(self.execute(pair), ['ARGS -r %s %s' % (self.rig.node(CARD_M), self.rig.node(CARD_A))])
        (self.rig.tt / 'by-id' / CARD_A).unlink()
        self.assertEqual(self.execute(pair), [])                    # never card M alone

    def test_a_serving_target_resets_both_link_ends_together_in_one_call(self):
        out = self.hint(QUAL_CARD=CARD_A, ALLOW_SERVING_CARD='1')
        self.assertEqual(self.command(out), 'm=$(readlink -e %s) && a=$(readlink -e %s) && ~/.local/bin/tt-smi -r "$m" "$a"'
                         % (self.rig.byid(CARD_M), self.rig.byid(CARD_A)))                    # M then A, one call
        self.assertIn('TOGETHER, in one call', out)
        self.assertIn('%s (card M, half of the serving pair) is %s now, PCI %s.' % (CARD_M, self.rig.node(CARD_M), PCI_M),
                      out)
        self.assertIn('%s (card A, half of the serving pair) is %s now, PCI %s.' % (CARD_A, self.rig.node(CARD_A), PCI_A),
                      out)
        self.assertIn('qwen-two-p150a-exclusive', out)
        self.assertNotIn(CARD_B, out)

    def test_an_unknown_address_still_resets_by_board_id(self):
        rig = FakeRig(tempfile.mkdtemp(dir=self.tmp.name), pci={CARD_M: PCI_M, CARD_A: PCI_A, CARD_B: ''})
        out = self.hint(rig)
        self.assertIn('PCI unknown', out)
        self.assertIn('which row is %s (its PCI address is unknown here)' % rig.node(CARD_B), out)
        self.assertEqual(self.command(out), 'n=$(readlink -e %s) && ~/.local/bin/tt-smi -r "$n"' % rig.byid(CARD_B))


@unittest.skipUnless(BASH, 'bash not found')
class SysfsTests(unittest.TestCase):
    """qual_pci_of itself: a node's device numbers -> <sys>/dev/char/<major>:<minor>/device -> the PCI
    function's directory name, read from a fake /sys laid out as the kernel lays it out."""

    LAYOUT = [
        # (major:minor, the class device's directory under devices/, its device link or None)
        ('234:2', 'pci0000:d0/0000:d0:01.1/0000:d1:00.0/tenstorrent/2', '../../../0000:d1:00.0'),
        ('234:10', 'pci0000:f0/0000:f0:01.1/0000:f2:02.0/0000:f4:00.0/tenstorrent/10', '../../../0000:f4:00.0'),
        ('1:3', 'virtual/misc/nodevice', None),
        ('234:11', 'platform/thing/tenstorrent/11', '../..'),
    ]

    @classmethod
    def setUpClass(cls):
        if not symlinks_work():
            raise unittest.SkipTest('symlinks cannot be created here')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sys = Path(self.tmp.name, 'sys').as_posix()
        lines = ['set -euo pipefail', 'sys=%s' % q(self.sys), 'mkdir -p "$sys/dev/char"']
        # Relative targets, as sysfs has them; each link is made from its own directory (Cygwin mangles a
        # relative target holding ':' when the link path itself is a drive path).
        for majmin, directory, link in self.LAYOUT:
            lines.append('mkdir -p "$sys/devices/%s"' % directory)
            if link:
                lines.append('(cd "$sys/devices/%s" && ln -s %s device)' % (directory, shlex.quote(link)))
            lines.append('(cd "$sys/dev/char" && ln -s "../../devices/%s" %s)' % (directory, majmin))
        result = run(['-c', NL.join(lines)], MSYS='winsymlinks:nativestrict')
        if result.returncode != 0:
            self.skipTest('fake /sys not built: ' + result.stderr)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_pci_address_comes_from_the_device_numbers(self):
        body = NL.join([
            'set -euo pipefail',
            '. %s' % q(LIBRARY),
            'QUAL_SYS_ROOT=%s' % q(self.sys),
            # hex device numbers, as stat -c %t:%T prints them
            'qual_majmin_of() { case $1 in m) echo ea:2 ;; b) echo ea:a ;; v) echo 1:3 ;; p) echo ea:b ;; '
            'junk) echo zz ;; nocolon) echo ea ;; three) echo ea:2:1 ;; esac; }',
            'for n in m b v p junk nocolon three missing; do echo "$n=[$(qual_pci_of $n)]"; done',
        ])
        result = run(['-c', body], MSYS='winsymlinks:nativestrict')
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in ('m=[%s]' % PCI_M, 'b=[%s]' % PCI_B, 'v=[]', 'p=[]', 'junk=[]', 'nocolon=[]', 'three=[]',
                         'missing=[]'):
            self.assertIn(expected, result.stdout.splitlines())


@unittest.skipUnless(BASH, 'bash not found')
class ServingPairTests(unittest.TestCase):
    """The gate's holder check and reset act on card M and card A only, by board id, checked against
    their PCI addresses - never card B, never a /dev/tenstorrent number, never a tt-smi index."""

    SHOW = 'serving_pair_resolve; echo "nodes=${SERVING_PAIR_NODES[*]} pci=${SERVING_PAIR_PCI[*]}"'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def rig(self, **options):
        return FakeRig(tempfile.mkdtemp(dir=self.tmp.name), **options)

    def run_pair(self, rig, body, **env):
        return rig.run(body, library=SERVING_PAIR, **env)

    def test_the_pair_is_card_m_and_card_a_by_board_id_and_pci(self):
        rig = self.rig()
        result = self.run_pair(rig, self.SHOW)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nodes=%s %s pci=%s %s' % (rig.node(CARD_M), rig.node(CARD_A), PCI_M, PCI_A), result.stdout)
        self.assertIn('serving pair: %s (card M, half of the serving pair) -> %s, PCI %s' % (CARD_M, rig.node(CARD_M), PCI_M),
                      result.stdout)
        self.assertNotIn(rig.node(CARD_B), result.stdout)
        self.assertNotIn(CARD_B, result.stdout)

    def test_the_holder_check_needs_only_the_board_ids(self):
        rig = self.rig(pci={CARD_M: '', CARD_A: '', CARD_B: ''})                   # no sysfs at all
        result = self.run_pair(rig, 'serving_pair_nodes; echo "nodes=${SERVING_PAIR_NODES[*]}"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nodes=%s %s' % (rig.node(CARD_M), rig.node(CARD_A)), result.stdout)
        self.assertNotIn(rig.node(CARD_B), result.stdout)
        refused = self.run_pair(rig, self.SHOW)                                    # but the reset refuses
        self.assertEqual(refused.returncode, 1)
        self.assertIn('at PCI unknown, not %s' % PCI_M, refused.stderr)

    def test_a_moved_board_or_a_missing_one_resolves_nothing(self):
        moved = self.rig(pci={CARD_M: PCI_B, CARD_A: PCI_A, CARD_B: PCI_M})
        result = self.run_pair(moved, self.SHOW)
        self.assertEqual(result.returncode, 1)
        self.assertIn("refusing: %s is %s at PCI %s, not %s; the pair's mapping changed"
                      % (CARD_M, moved.node(CARD_M), PCI_B, PCI_M), result.stderr)
        missing = self.rig()
        (missing.tt / 'by-id' / CARD_A).unlink()
        result = self.run_pair(missing, self.SHOW)
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing: serving card %s (card A, half of the serving pair) has no device node' % CARD_A,
                      result.stderr)
        emptied = self.run_pair(missing, 'serving_pair_resolve || true; echo "left=${#SERVING_PAIR_NODES[@]}"')
        self.assertIn('left=0', emptied.stdout)                                   # nothing stale to reset

    def test_both_board_ids_on_one_node_is_refused(self):
        rig = self.rig()
        (rig.tt / 'by-id' / CARD_A).write_text(rig.node(CARD_M))
        result = self.run_pair(rig, 'serving_pair_nodes')
        self.assertEqual(result.returncode, 1)
        self.assertIn('both serving cards resolve to %s' % rig.node(CARD_M), result.stderr)

    def test_the_driver_write_names_only_card_m_and_card_a(self):
        # The self-heal's only write: any address but the pair's is refused before sudo is even called.
        rig = self.rig()
        body = NL.join([
            'sudo() { echo "sudo $* <- $(cat)" >> "$FAKE_DIR/sudo.log"; }',
            'for pci in %s "" "%s %s" "*" %s:extra; do' % (PCI_B, PCI_M, PCI_A, PCI_A),
            '  st=0; serving_pair_driver_write "$pci" unbind 2>> "$FAKE_DIR/err.log" || st=$?; echo "[$pci]=$st"',
            'done',
            'st=0; serving_pair_driver_write %s remove 2>> "$FAKE_DIR/err.log" || st=$?; echo "remove=$st"' % PCI_A,
            'serving_pair_driver_write %s unbind' % PCI_A,
            'serving_pair_driver_write %s bind' % PCI_M,
        ])
        result = self.run_pair(rig, body)
        self.assertEqual(result.returncode, 0, result.stderr)
        for pci in (PCI_B, '', PCI_M + ' ' + PCI_A, '*', PCI_A + ':extra'):
            self.assertIn('[%s]=2' % pci, result.stdout)
        self.assertIn('remove=2', result.stdout)
        sysfs = (rig.dir / 'sys' / 'bus' / 'pci' / 'drivers' / 'tenstorrent').as_posix()
        self.assertEqual((rig.dir / 'sudo.log').read_text().splitlines(),
                         ['sudo -n tee %s/unbind <- %s' % (sysfs, PCI_A), 'sudo -n tee %s/bind <- %s' % (sysfs, PCI_M)])
        self.assertIn("is not card M's or card A's PCI address", (rig.dir / 'err.log').read_text())

    def test_the_rescan_write_names_only_card_m_and_card_a_upstream_ports(self):
        # The self-heal's other write: card B's port, a device address, or anything but the pair's own
        # upstream ports is refused before sudo is even called.
        rig = self.rig()
        body = NL.join([
            'sudo() { echo "sudo $* <- $(cat)" >> "$FAKE_DIR/sudo.log"; }',
            'for port in %s %s %s "" "%s %s" "*" %s/../%s %s:extra; do' % (PORT_B, PCI_A, PCI_M, PORT_M, PORT_A,
                                                                          PORT_A, PORT_B, PORT_A),
            '  st=0; serving_pair_rescan_write "$port" 2>> "$FAKE_DIR/err.log" || st=$?; echo "[$port]=$st"',
            'done',
            'serving_pair_rescan_write %s' % PORT_A,
            'serving_pair_rescan_write %s' % PORT_M,
            'echo "upstream=[$(serving_pair_upstream_of %s)] [$(serving_pair_upstream_of %s)] [$(serving_pair_upstream_of %s)]"'
            % (PCI_M, PCI_A, PCI_B),
        ])
        result = self.run_pair(rig, body)
        self.assertEqual(result.returncode, 0, result.stderr)
        for port in (PORT_B, PCI_A, PCI_M, '', PORT_M + ' ' + PORT_A, '*', PORT_A + '/../' + PORT_B, PORT_A + ':extra'):
            self.assertIn('[%s]=2' % port, result.stdout)
        self.assertIn('upstream=[%s] [%s] []' % (PORT_M, PORT_A), result.stdout)
        devices = (rig.dir / 'sys' / 'bus' / 'pci' / 'devices').as_posix()
        self.assertEqual((rig.dir / 'sudo.log').read_text().splitlines(),
                         ['sudo -n tee %s/%s/rescan <- 1' % (devices, PORT_A), 'sudo -n tee %s/%s/rescan <- 1' % (devices, PORT_M)])
        self.assertIn("is not card M's or card A's upstream port", (rig.dir / 'err.log').read_text())

    def test_the_serving_pair_file_never_names_card_b(self):
        text = read(SERVING_PAIR)
        for name in (CARD_B, PCI_B, 'f4:00', PORT_B, 'f2:01'):
            self.assertNotIn(name, text)
        self.assertIn("SERVING_PAIR_EXPECTED_PCI='%s %s'" % (PCI_M, PCI_A), text)
        self.assertIn("SERVING_PAIR_UPSTREAM_PORTS='%s %s'" % (PORT_M, PORT_A), text)
        self.assertNotIn(chr(13), text)

    def test_the_heal_refuses_a_wait_or_a_limit_that_is_not_a_number(self):
        # A non-numeric wait would make its -ge test an error, which is false: the wait would never end.
        rig = self.rig()
        os.remove(rig.byid(CARD_A))
        body = NL.join([
            'sudo() { echo "sudo $*" >> "$FAKE_DIR/sudo.log"; }',
            'sleeps=0',
            'sleep() { sleeps=$((sleeps + 1)); [ "$sleeps" -lt 100 ] || { echo ENDLESS >&2; exit 99; }; }',
            'for pair in "abc:" "60:7x" "60:-1" "-5:" "6 0:"; do',
            '  st=0; serving_pair_heal "${pair%%:*}" "${pair#*:}" 2>> "$FAKE_DIR/err.log" || st=$?; echo "[$pair]=$st"',
            'done',
        ])
        result = self.run_pair(rig, body)
        self.assertEqual(result.returncode, 0, result.stderr)
        for pair in ('abc:', '60:7x', '60:-1', '-5:', '6 0:'):
            self.assertIn('[%s]=1' % pair, result.stdout)
        self.assertEqual((rig.dir / 'err.log').read_text().count('is not a number of seconds'), 5)
        self.assertFalse((rig.dir / 'sudo.log').exists())

    def test_a_board_coming_back_from_a_reset_is_waited_for(self):
        rig = self.rig()
        rig.pending(CARD_A, '4', polls=3)                                          # gone for three polls
        result = self.run_pair(rig, 'serving_pair_resolve 10; echo "nodes=${SERVING_PAIR_NODES[*]}"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nodes=%s %s' % (rig.node(CARD_M), rig.node(CARD_A)), result.stdout)
        self.assertIn('both board ids resolved after 6 s', result.stdout)
        late = self.rig()
        late.pending(CARD_A, '4', polls=5)
        result = self.run_pair(late, 'serving_pair_resolve 4')
        self.assertEqual(result.returncode, 1)
        self.assertIn('has no device node', result.stderr)


class GateWorkflowTests(unittest.TestCase):
    """The m3native gate's holder check and board reset name only the serving pair."""

    @classmethod
    def setUpClass(cls):
        import yaml
        cls.workflow = yaml.safe_load(read(GATE))
        cls.steps = {step.get('name'): step for step in cls.workflow['jobs']['arm']['steps'] if step.get('name')}
        cls.holder = cls.steps['Confirm no process holds a serving card']['run']
        cls.reset = cls.steps['Reset the serving pair (card M and card A together) before this job opens it']['run']

    def test_the_holder_check_names_only_the_pair(self):
        text = self.holder
        self.assertIn('. scripts/ci/serving_pair.sh', text)
        self.assertLess(text.index('serving_pair_nodes 60'), text.index('fuser -v'))     # by board id; no sysfs
        self.assertIn('paths=("${SERVING_PAIR_NODES[@]}")', text)
        self.assertLess(text.index('test "${#paths[@]}" = 2'), text.index('fuser -v'))
        self.assertIn('fuser -v "${paths[@]}"', text)
        self.assertNotIn('ls /dev/tenstorrent', text)

    def test_the_reset_names_the_pair_by_node_and_keeps_both_ends_together(self):
        text = self.reset
        self.assertIn('. scripts/ci/serving_pair.sh', text)
        together = text.index('"$smi" -r "${SERVING_PAIR_NODES[@]}"')
        self.assertLess(text.index('serving_pair_resolve 60'), together)
        self.assertLess(text.index('test "${#SERVING_PAIR_NODES[@]}" = 2'), together)
        fallback = text[together:]
        self.assertLess(fallback.index('serving_pair_resolve 120'), fallback.index('"$smi" -r "${SERVING_PAIR_NODES[$i]}"'))
        self.assertEqual(re.findall(r'"\$smi" -r (\S+)', text), ['"${SERVING_PAIR_NODES[@]}"', '"${SERVING_PAIR_NODES[$i]}"'])
        self.assertNotIn('SERVING_PAIR_SMI', text)                                    # no tt-smi index anywhere
        self.assertNotIn('"$smi" -f', text)
        self.assertNotIn('/dev/tenstorrent/', text)
        self.assertIn('card M and card A, BOTH link ends in one tt-smi call', text)
        self.assertIn('Never card B', text)

    def test_the_reset_heals_right_after_the_pair_reset_and_fails_when_the_heal_fails(self):
        text = self.reset
        heal = text.index('serving_pair_heal 60 720 2>&1 | tee -a experiment-results/reset.log || heal=$?')
        code = [line.strip() for line in text.splitlines() if not line.lstrip().startswith('#')]
        # Two heals: the fallback's, before each card is resolved again, and the one after the reset.
        fallback_heal = 'if ! serving_pair_heal 60 720 >> experiment-results/reset.log 2>&1; then status=1; break; fi'
        self.assertEqual([line for line in code if 'serving_pair_heal' in line],
                         [fallback_heal, 'serving_pair_heal 60 720 2>&1 | tee -a experiment-results/reset.log || heal=$?'])
        self.assertLess(text.index('"$smi" -r "${SERVING_PAIR_NODES[@]}"'), heal)       # after the pair reset
        self.assertLess(text.index('"$smi" -r "${SERVING_PAIR_NODES[$i]}"'), heal)      # and its fallback
        self.assertEqual(text[:heal].rstrip().splitlines()[-1].strip(), 'if [ "$status" = 0 ]; then')
        self.assertLess(heal, text.index('echo "heal_exit=$heal" | tee experiment-results/heal.status'))
        self.assertEqual(text.rstrip().splitlines()[-2:], ['test "$status" = 0', 'test "$heal" = 0'])
        # The fallback's heal is the first thing in its per-card loop, before the resolve and the reset.
        loop = code[code.index('for i in 0 1; do') + 1:code.index('done')]
        self.assertEqual(loop, [fallback_heal,
                                'if ! serving_pair_resolve 120 >> experiment-results/reset.log 2>&1; then status=1; break; fi',
                                'timeout -k 20 300 "${prefix[@]}" "$smi" -r "${SERVING_PAIR_NODES[$i]}" '
                                '>> experiment-results/reset.log 2>&1 || status=$?'])
        self.assertLess(text.index('"$smi" -r "${SERVING_PAIR_NODES[@]}"'), text.index(fallback_heal))
        step = self.steps['Reset the serving pair (card M and card A together) before this job opens it']
        self.assertGreaterEqual(step['timeout-minutes'], 14)
        # No rescan and no unbind may start in the step's last two minutes: the runner's timeout must
        # never land between an unbind and its bind.
        lasts = [int(last) for last in re.findall(r'serving_pair_heal 60 (\d+) ', text)]
        self.assertEqual(lasts, [step['timeout-minutes'] * 60 - 120] * 2)

    def test_no_step_of_the_gate_touches_every_node_or_card_b(self):
        text = read(GATE)
        self.assertNotIn('ls /dev/tenstorrent', text)
        self.assertNotIn(CARD_B, text)
        self.assertNotIn(PCI_B, text)
        self.assertNotIn(PORT_B, text)

    @unittest.skipUnless(BASH, 'bash not found')
    def test_both_steps_parse(self):
        for name, run_text in (('holder', self.holder), ('reset', self.reset)):
            with self.subTest(step=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'step.sh'
                path.write_bytes(run_text.encode('utf-8'))
                result = run(['-n', path.as_posix()])
                self.assertEqual(result.returncode, 0, result.stderr)


FAKE_TT_SMI = '''#!/usr/bin/env bash
# tt-smi as the gate calls it: logs every call; -r accepts only existing /dev/tenstorrent node paths.
# Scenario (FAKE_DIR/scenario): ok; renumber - the first reset fails its re-init and the pair
# re-enumerates, card M at once as node 3, card A only after three polls as node 4 (card B keeps
# node 0); gone - the same, but card A never comes back; untrained - the first reset fails (a traceback)
# and card A's switch port cannot train its link: its PCI device is gone, with its driver binding, node and
# by-id link (run v217), until a rescan of its port brings it back (FAKE_KERNEL). A reset that succeeds
# then runs FAKE_DIR/after-reset.sh, if there is one (what the boards look like when it returns).
FAKE_DIR={fake}
TT={tt}
echo "$*" >> "$FAKE_DIR/smi.log"
case ${{1:-}} in
  -ls) echo 'fake tt-smi -ls'; exit 0 ;;
  -r) shift ;;
  *) exit 2 ;;
esac
if [ $# = 0 ]; then echo 'BARE RESET: every board' >> "$FAKE_DIR/smi.log"; exit 3; fi
for arg in "$@"; do
  case $arg in
    "$TT"/[0-9]*) [ -f "$arg" ] || {{ echo "no such node $arg" >&2; exit 1; }} ;;
    *) echo "not a /dev/tenstorrent path: $arg" >&2; exit 1 ;;
  esac
done
calls=$(( $(cat "$FAKE_DIR/resets" 2>/dev/null || echo 0) + 1 ))
echo "$calls" > "$FAKE_DIR/resets"
scenario=$(cat "$FAKE_DIR/scenario" 2>/dev/null || echo ok)
if [ "$calls" = 1 ] && [ "$scenario" = untrained ]; then
  a=$(cat "$TT/by-id/{card_a}")
  rm -rf "$a" "$TT/by-id/{card_a}" "$FAKE_DIR/pci/${{a##*/}}" "$FAKE_DIR/sys/bus/pci/devices/{pci_a}" \\
    "$FAKE_DIR/sys/bus/pci/drivers/tenstorrent/{pci_a}"
  echo 'Traceback (most recent call last):'
  exit 1
fi
if [ "$calls" = 1 ] && [ "$scenario" != ok ]; then
  rm -f "$TT/1" "$TT/2" "$TT/by-id/{card_m}" "$TT/by-id/{card_a}" "$FAKE_DIR/pci/1" "$FAKE_DIR/pci/2"
  : > "$TT/3"; printf '%s' "$TT/3" > "$TT/by-id/{card_m}"; echo {pci_m} > "$FAKE_DIR/pci/3"
  if [ "$scenario" = renumber ]; then echo 3 > "$FAKE_DIR/pending"; else echo 100000 > "$FAKE_DIR/pending"; fi
  {{
    echo ": > '$TT/4'"
    echo "printf '%s' '$TT/4' > '$TT/by-id/{card_a}'"
    echo "echo {pci_a} > '$FAKE_DIR/pci/4'"
  }} > "$FAKE_DIR/comeback.sh"
  echo 'Error when re-initializing chips!'
  exit 1
fi
if [ -f "$FAKE_DIR/after-reset.sh" ]; then . "$FAKE_DIR/after-reset.sh"; fi
echo "reset $*"
exit 0
'''

FAKE_FUSER = '''#!/usr/bin/env bash
FAKE_DIR={fake}
echo "$*" >> "$FAKE_DIR/fuser.log"
held=$(cat "$FAKE_DIR/held" 2>/dev/null || true)
found=1
for n in "$@"; do
  case " $held " in *" $n "*) echo "$n: thatch 4242 F.... python3" >&2; found=0 ;; esac
done
exit $found
'''

FAKE_SUDO = '''#!/usr/bin/env bash
# sudo -n: runs the command. A tee into the fake driver's unbind or bind file, or into a fake PCI
# device's rescan file, is logged (FAKE_DIR/driver.log: "<file> <address>", "rescan <port>") and handed
# to the fake kernel; a tee into any other sys tree is refused and logged, so no test can ever write the
# host's real /sys.
FAKE_DIR={fake}
[ "${{1:-}}" = -n ] && shift
if [ "${{1:-}}" = tee ]; then
  case ${{2:-}} in
    "$FAKE_DIR"/sys/bus/pci/drivers/tenstorrent/unbind|"$FAKE_DIR"/sys/bus/pci/drivers/tenstorrent/bind)
      value=$(cat)
      echo "${{2##*/}} $value" >> "$FAKE_DIR/driver.log"
      . "$FAKE_DIR/kernel.sh" "${{2##*/}}" "$value"
      exit $? ;;
    "$FAKE_DIR"/sys/bus/pci/devices/*/rescan)
      value=$(cat)
      port=${{2%/rescan}}
      port=${{port##*/}}
      if [ "$value" = 1 ]; then echo "rescan $port" >> "$FAKE_DIR/driver.log"; else echo "rescan $port <- $value" >> "$FAKE_DIR/driver.log"; exit 1; fi
      . "$FAKE_DIR/kernel.sh" rescan "$port"
      exit $? ;;
    */sys/*|/sys*) echo "REAL SYS WRITE $*" >> "$FAKE_DIR/driver.log"; exit 97 ;;
  esac
fi
exec "$@"
'''

FAKE_KERNEL = '''# The fake tenstorrent driver, sourced by the fake sudo for a write to its unbind or bind file: $1 the
# file, $2 the address written. Unbinding a bound board removes its node; binding brings the node back
# on the same number, and udev makes the by-id link two polls later once the board has been bound
# FAKE_DIR/heal-after times (never without that file: the ARC still does not answer). FAKE_DIR/bind-fails
# makes that many bind writes fail; FAKE_DIR/unbind-refused makes every unbind fail, writing nothing (sudo
# refused).
# A write to an upstream port's rescan file ($1 rescan, $2 the port) is additive, as the kernel's: a board
# whose PCI device is present is left alone. One whose device is absent re-enumerates from the
# FAKE_DIR/rescan-after-th rescan of its port on (never without that file: its link does not train), one
# poll later, bound, on the lowest free node; udev makes its by-id link two polls after that if
# FAKE_DIR/rescan-link exists (otherwise it comes back without it, the telemetry race).
op=$1
pci=$2
tt={tt}
drv=$FAKE_DIR/sys/bus/pci/drivers/tenstorrent
if [ "$op" = rescan ]; then
  case $2 in
    {port_m}) pci={pci_m}; card={card_m} ;;
    {port_a}) pci={pci_a}; card={card_a} ;;
    {port_b}) pci={pci_b}; card={card_b} ;;
    *) echo "tee: $2: No such file or directory" >&2; return 1 ;;
  esac
  [ -d "$FAKE_DIR/sys/bus/pci/devices/$2" ] || {{ echo "tee: $2: No such file or directory" >&2; return 1; }}
  rescans=$(( $(cat "$FAKE_DIR/rescans-$card" 2>/dev/null || echo 0) + 1 ))
  echo "$rescans" > "$FAKE_DIR/rescans-$card"
  after=$(cat "$FAKE_DIR/rescan-after" 2>/dev/null || echo 0)
  if [ -e "$FAKE_DIR/sys/bus/pci/devices/$pci" ] || [ "$after" = 0 ] || [ "$rescans" -lt "$after" ]; then return 0; fi
  n=0
  while [ -e "$tt/$n" ]; do n=$((n + 1)); done
  {{
    echo "mkdir -p '$FAKE_DIR/sys/bus/pci/devices/$pci' '$drv/$pci'"
    echo ": > '$tt/$n'"
    echo "echo $pci > '$FAKE_DIR/pci/$n'"
    if [ -f "$FAKE_DIR/rescan-link" ]; then
      echo "echo 2 > '$FAKE_DIR/pending'"
      echo "echo \\"printf '%s' '$tt/$n' > '$tt/by-id/$card'\\" > '$FAKE_DIR/comeback.sh'"
    fi
  }} > "$FAKE_DIR/comeback.sh"
  echo 1 > "$FAKE_DIR/pending"
  return 0
fi
case $pci in
  {pci_m}) card={card_m} ;;
  {pci_a}) card={card_a} ;;
  {pci_b}) card={card_b} ;;
  *) echo "tee: write error: No such device" >&2; return 1 ;;
esac
if [ "$op" = unbind ]; then
  if [ -f "$FAKE_DIR/unbind-refused" ]; then echo 'sudo: a password is required' >&2; return 1; fi
  [ -e "$drv/$pci" ] || {{ echo "tee: write error: No such device" >&2; return 1; }}
  rm -rf "$drv/$pci"
  for f in "$FAKE_DIR"/pci/*; do
    if [ "$(cat "$f")" = "$pci" ]; then
      rm -f "$tt/${{f##*/}}" "$f"
      echo "${{f##*/}}" > "$FAKE_DIR/unbound-$card"
    fi
  done
  return 0
fi
fails=$(cat "$FAKE_DIR/bind-fails" 2>/dev/null || echo 0)
if [ "$fails" -gt 0 ]; then
  echo $((fails - 1)) > "$FAKE_DIR/bind-fails"
  echo "tee: write error: No such device" >&2
  return 1
fi
[ -e "$FAKE_DIR/sys/bus/pci/devices/$pci" ] || {{ echo "tee: write error: No such device" >&2; return 1; }}
[ ! -e "$drv/$pci" ] || {{ echo "tee: write error: Device or resource busy" >&2; return 1; }}
mkdir -p "$drv/$pci"
n=$(cat "$FAKE_DIR/unbound-$card")
: > "$tt/$n"
echo "$pci" > "$FAKE_DIR/pci/$n"
binds=$(( $(cat "$FAKE_DIR/binds-$card" 2>/dev/null || echo 0) + 1 ))
echo "$binds" > "$FAKE_DIR/binds-$card"
after=$(cat "$FAKE_DIR/heal-after" 2>/dev/null || echo 0)
if [ "$after" -gt 0 ] && [ "$binds" -ge "$after" ]; then
  echo 2 > "$FAKE_DIR/pending"
  echo "printf '%s' '$tt/$n' > '$tt/by-id/$card'" > "$FAKE_DIR/comeback.sh"
fi
return 0
'''


@unittest.skipUnless(BASH, 'bash not found')
class GateStepExecutionTests(unittest.TestCase):
    """The gate's two steps, run as the runner runs them (bash, in the checkout), against a FakeRig:
    scripts/ci/serving_pair.sh is a shim that sources the real one and then the rig's stubs, and
    sudo, fuser and tt-smi are fakes first on PATH. The reset step's self-heal reads a fake /sys and
    writes the fake driver's unbind and bind files, and the upstream ports' rescan files, through the
    fake sudo, which hands them to a fake kernel (FAKE_KERNEL)."""

    @classmethod
    def setUpClass(cls):
        GateWorkflowTests.setUpClass()
        cls.holder = GateWorkflowTests.holder
        cls.reset = GateWorkflowTests.reset

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def rig(self, **options):
        return FakeRig(tempfile.mkdtemp(dir=self.tmp.name), **options)

    def run_step(self, rig, text, scenario='ok', held=(), after_reset=(), heal_after=0, bind_fails=0, clock=None,
                 unbind_refused=False, rescan_after=0, rescan_link=True):
        """after_reset: shell lines the fake tt-smi runs once a reset succeeds. heal_after: the bind (of a
        board's address) from which udev makes its by-id link again; 0, never. bind_fails: bind writes
        that fail before one succeeds. clock: the step's $SECONDS when it sources serving_pair.sh (the fake
        sleep never advances it: a slow reset). unbind_refused: every unbind write fails. rescan_after: the
        rescan (of a board's upstream port) from which its absent PCI device re-enumerates; 0, never.
        rescan_link: whether udev then makes its by-id link (False: the telemetry race follows)."""
        work = rig.dir / 'm3native'
        (work / 'scripts' / 'ci').mkdir(parents=True)
        # id: the shim runs as the unprivileged runner does, so every privileged write goes through sudo.
        extra = ['id() { echo 1000; }'] + (['SECONDS=%d' % clock] if clock is not None else [])
        shim = NL.join(['. %s' % q(SERVING_PAIR)] + rig.stubs() + extra) + NL
        (work / 'scripts' / 'ci' / 'serving_pair.sh').write_bytes(shim.encode('utf-8'))
        (work / 'step.sh').write_bytes(text.encode('utf-8'))
        fakes = rig.dir / 'bin'
        fakes.mkdir()
        values = dict(fake=q(rig.dir), tt=q(rig.tt), card_m=CARD_M, card_a=CARD_A, card_b=CARD_B, pci_m=PCI_M,
                      pci_a=PCI_A, pci_b=PCI_B, port_m=PORT_M, port_a=PORT_A, port_b=PORT_B)
        for name, body in (('tt-smi', FAKE_TT_SMI.format(**values)), ('fuser', FAKE_FUSER.format(**values)),
                           ('sudo', FAKE_SUDO.format(**values))):
            (fakes / name).write_bytes(body.encode('utf-8'))
            os.chmod(fakes / name, 0o755)
        (rig.dir / 'kernel.sh').write_bytes(FAKE_KERNEL.format(**values).encode('utf-8'))
        (rig.dir / 'scenario').write_text(scenario + NL)
        (rig.dir / 'held').write_text(' '.join(held) + NL)
        (rig.dir / 'heal-after').write_text('%d' % heal_after + NL)
        (rig.dir / 'bind-fails').write_text('%d' % bind_fails + NL)
        if unbind_refused:
            (rig.dir / 'unbind-refused').write_text('1' + NL)
        (rig.dir / 'rescan-after').write_text('%d' % rescan_after + NL)
        if rescan_link:
            (rig.dir / 'rescan-link').write_text('1' + NL)
        if after_reset:
            (rig.dir / 'after-reset.sh').write_bytes((NL.join(after_reset) + NL).encode('utf-8'))
        wrapper = rig.dir / 'runner.sh'
        wrapper.write_bytes(NL.join(rig.sysfs() + [
            'fakes=%s' % q(fakes),
            'export PATH="$(cygpath -u "$fakes" 2>/dev/null || echo "$fakes"):$PATH"',
            'cd %s' % q(work),
            'exec bash step.sh',
        ]).encode('utf-8') + NL.encode('utf-8'))
        return run([wrapper.as_posix()], timeout=180, RUNNER_NAME=RUNNER)

    def smi_calls(self, rig):
        log = rig.dir / 'smi.log'
        return log.read_text().splitlines() if log.is_file() else []

    def driver_calls(self, rig):
        log = rig.dir / 'driver.log'
        return log.read_text().splitlines() if log.is_file() else []

    def results(self, rig, name):
        return (rig.dir / 'm3native' / 'experiment-results' / name).read_text()

    def race(self, rig, *cards):
        """after_reset lines: the cards come back from the reset with their node and PCI device, bound,
        but without their by-id link (the ARC firmware was not ready when udev looked)."""
        return ['rm -f %s' % q(rig.byid(card)) for card in cards]

    def untrained(self, rig, card):
        """after_reset lines: the card's switch port cannot train its link (run v217): its PCI device is
        gone, with its driver binding, node and by-id link."""
        return ['rm -f %s %s %s' % (q(rig.byid(card)), q(rig.node(card)), q(rig.dir / 'pci' / rig.nodes[card])),
                'rm -rf %s %s' % (self.sys_path(rig, 'devices', rig.pci[card]),
                                  self.sys_path(rig, 'drivers', 'tenstorrent', rig.pci[card]))]

    def rescan(self, n):
        return 'PCI device absent after reset (link did not train); rescan of its upstream port %s %d/2' % (PORT_A, n)

    def sys_path(self, rig, *parts):
        return q(rig.dir.joinpath('sys', 'bus', 'pci', *parts))

    def bound(self, rig, address):
        """Whether the fake driver has the address bound (checked by bash: the name holds ':')."""
        result = run(['-c', 'test -e %s' % self.sys_path(rig, 'drivers', 'tenstorrent', address)])
        return result.returncode == 0

    def reprobe(self, n):
        return 'by-id missing after reset (telemetry race); driver re-probe %d/2' % n

    def assert_never_card_b(self, rig, calls):
        for call in calls:
            self.assertNotIn(rig.node(CARD_B), call.split())
            self.assertNotIn('f4:00', call)
            self.assertNotIn(PORT_B, call)
            self.assertNotIn('BARE RESET', call)
            if call.startswith('-r'):
                self.assertTrue(call.split()[1:], 'tt-smi -r with no board resets every board')

    def test_the_holder_step_checks_the_pair_only_and_a_card_b_session_does_not_fail_it(self):
        rig = self.rig()
        result = self.run_step(rig, self.holder, held=(rig.node(CARD_B),))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(set(rig.fuser_calls()), {'-v %s %s' % (rig.node(CARD_M), rig.node(CARD_A))})

    def test_the_holder_step_fails_on_a_holder_of_a_serving_card(self):
        rig = self.rig()
        result = self.run_step(rig, self.holder, held=(rig.node(CARD_A),))
        self.assertEqual(result.returncode, 1)
        self.assertIn('Device ownership of the serving pair not clear', result.stdout)

    def test_the_reset_is_one_call_naming_card_m_and_card_a_by_node(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.smi_calls(rig)
        self.assertEqual(calls, ['-r %s %s' % (rig.node(CARD_M), rig.node(CARD_A)), '-ls'])
        self.assert_never_card_b(rig, calls)
        self.assertIn('reset_exit=0', (rig.dir / 'm3native' / 'experiment-results' / 'reset.status').read_text())

    def test_a_failed_pair_reset_resolves_each_card_again_before_its_own_reset(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset, scenario='renumber')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        tt = rig.tt.as_posix()
        calls = self.smi_calls(rig)
        # The pair by its nodes then (2, 1); the fallback by the nodes they came back as (3, then 4 once
        # card A had reappeared), never the stale ones and never card B's node 0.
        self.assertEqual(calls, ['-r %s/2 %s/1' % (tt, tt), '-r %s/3' % tt, '-r %s/4' % tt, '-ls'])
        self.assert_never_card_b(rig, calls)
        log = (rig.dir / 'm3native' / 'experiment-results' / 'reset.log').read_text()
        # The fallback's heal, before card M is resolved, waited for card A (and re-probed nothing).
        self.assertIn("[reset] both serving cards' by-id links present 6 s after the reset; no re-probe", log)
        self.assertEqual(self.driver_calls(rig), [])

    def test_a_pair_that_does_not_come_back_resets_nothing_more_and_fails(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset, scenario='gone')
        self.assertEqual(result.returncode, 1)
        calls = self.smi_calls(rig)
        tt = rig.tt.as_posix()
        self.assertEqual(calls, ['-r %s/2 %s/1' % (tt, tt), '-ls'])
        self.assert_never_card_b(rig, calls)
        self.assertIn('reset_exit=1', (rig.dir / 'm3native' / 'experiment-results' / 'reset.status').read_text())
        # The fallback's heal refused card A (bound, but no node at its address), before any resolve.
        self.assertIn('0 device nodes are at PCI %s, not one' % PCI_A, self.results(rig, 'reset.log'))
        self.assertEqual(self.driver_calls(rig), [])

    def test_a_changed_mapping_resets_nothing(self):
        rig = self.rig(pci={CARD_M: PCI_B, CARD_A: PCI_A, CARD_B: PCI_M})
        result = self.run_step(rig, self.reset)
        self.assertEqual(result.returncode, 1)
        self.assertIn("the pair's mapping changed", result.stderr)
        self.assertEqual(self.smi_calls(rig), [])

    # The self-heal after the pair reset (run v190: card A came back without its by-id link).

    def test_a_pair_that_comes_back_whole_is_not_re_probed(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.driver_calls(rig), [])
        self.assertIn("[reset] both serving cards' by-id links present 0 s after the reset; no re-probe",
                      self.results(rig, 'reset.log'))
        self.assertIn('heal_exit=0', self.results(rig, 'heal.status'))

    def test_the_telemetry_race_is_healed_by_one_driver_re_probe_of_that_card(self):
        rig = self.rig()
        node_a = rig.node(CARD_A)
        result = self.run_step(rig, self.reset, after_reset=self.race(rig, CARD_A), heal_after=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # The reset itself is unchanged: one call, both link ends. The heal is a driver re-probe of card
        # A's own address - unbind, then bind - and no further reset.
        calls = self.smi_calls(rig)
        self.assertEqual(calls, ['-r %s %s' % (rig.node(CARD_M), node_a), '-ls'])
        self.assert_never_card_b(rig, calls)
        self.assertEqual(self.driver_calls(rig), ['unbind ' + PCI_A, 'bind ' + PCI_A])
        self.assertIn('-v ' + node_a, rig.fuser_calls())                           # its node re-checked first
        log = self.results(rig, 'reset.log')
        name = 'card A (%s, PCI %s)' % (CARD_A, PCI_A)
        self.assertIn('[reset] %s by-id link still missing 60 s after the reset' % name, log)
        self.assertIn('[reset] %s %s' % (name, self.reprobe(1)), log)
        self.assertIn('[reset] %s by-id link back 4 s after driver re-probe 1/2 -> %s' % (name, node_a), log)
        self.assertNotIn(self.reprobe(2), log)
        self.assertNotIn('card M (', log.replace('card M and card A', ''))        # card M was never touched
        self.assertIn('heal_exit=0', self.results(rig, 'heal.status'))
        self.assertIn('reset_exit=0', self.results(rig, 'reset.status'))
        self.assertIn('%s (card A, half of the serving pair) -> %s, PCI %s' % (CARD_A, node_a, PCI_A),
                      self.results(rig, 'serving-pair-after-reset.txt'))
        self.assertTrue(self.bound(rig, PCI_A))

    def test_the_heal_never_re_probes_card_b(self):
        # Card B lost its by-id link too (it shares the switch): only card A's address is re-probed.
        rig = self.rig()
        result = self.run_step(rig, self.reset, after_reset=self.race(rig, CARD_A, CARD_B), heal_after=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.driver_calls(rig), ['unbind ' + PCI_A, 'bind ' + PCI_A])
        self.assertNotIn(rig.node(CARD_B), ' '.join(rig.fuser_calls()))
        self.assertNotIn(CARD_B, self.results(rig, 'reset.log'))
        self.assertTrue(self.bound(rig, PCI_B))
        # Card B alone without its link: nothing is re-probed at all, and the step passes.
        alone = self.rig()
        result = self.run_step(alone, self.reset, after_reset=self.race(alone, CARD_B), heal_after=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.driver_calls(alone), [])
        self.assertIn('no re-probe', self.results(alone, 'reset.log'))
        # Card A without its link, and card A's address showing card B's node (card B's by-id target):
        # refused, nothing written.
        crossed = self.rig()
        lines = self.race(crossed, CARD_A) + ['echo %s > %s' % (PCI_A, q(crossed.dir / 'pci' / '0')),
                                              'rm -f %s' % q(crossed.dir / 'pci' / '1')]
        result = self.run_step(crossed, self.reset, after_reset=lines, heal_after=1)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.driver_calls(crossed), [])
        self.assertIn('is the by-id target of %s (card B, the qualification card)' % CARD_B, result.stdout)

    def test_a_holder_of_the_card_refuses_the_re_probe(self):
        rig = self.rig()
        node_a = rig.node(CARD_A)
        result = self.run_step(rig, self.reset, after_reset=self.race(rig, CARD_A), held=(node_a,), heal_after=1)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.driver_calls(rig), [])                                # nothing unbound
        self.assertEqual(rig.fuser_calls(), ['-v ' + node_a] * 5)                   # five tries, then refused
        log = self.results(rig, 'reset.log')
        self.assertIn('refusing: card A (%s, PCI %s) by-id missing after reset, and %s is held' % (CARD_A, PCI_A, node_a),
                      log)
        self.assertIn('thatch 4242', log)
        self.assertNotIn(self.reprobe(1), log)
        self.assertIn('heal_exit=1', self.results(rig, 'heal.status'))
        self.assertIn('reset_exit=0', self.results(rig, 'reset.status'))
        self.assertTrue(self.bound(rig, PCI_A))

    def test_the_heal_gives_up_after_two_re_probes(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset, after_reset=self.race(rig, CARD_A), heal_after=0)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.driver_calls(rig), ['unbind ' + PCI_A, 'bind ' + PCI_A] * 2)
        log = self.results(rig, 'reset.log')
        for n in (1, 2):
            self.assertIn(self.reprobe(n), log)
            self.assertIn('by-id link still missing 60 s after driver re-probe %d/2' % n, log)
        self.assertNotIn('re-probe 3/', log)
        self.assertIn('by-id link still missing after 2 driver re-probes; not guessing', log)
        self.assertIn('heal_exit=1', self.results(rig, 'heal.status'))
        self.assertEqual(self.smi_calls(rig), ['-r %s %s' % (rig.node(CARD_M), rig.node(CARD_A)), '-ls'])
        self.assertTrue(self.bound(rig, PCI_A))                                     # left bound, as it was

    def test_a_missing_pci_device_refuses_without_a_re_probe(self):
        for case, parts, why in (
                ('absent', (('devices', PCI_A), ('drivers', 'tenstorrent', PCI_A)),
                 'PCI device %s still absent after 2 rescans of its upstream port %s' % (PCI_A, PORT_A)),
                ('unbound', (('drivers', 'tenstorrent', PCI_A),),
                 'PCI device %s is not bound to the tenstorrent driver' % PCI_A)):
            with self.subTest(case=case):
                rig = self.rig()
                # Card A comes back from the reset without its by-id link and node, and without its PCI
                # device (the board did not re-enumerate, and the rescans below do not bring it back) or
                # without its driver.
                gone = self.race(rig, CARD_A) + ['rm -f %s %s' % (q(rig.node(CARD_A)), q(rig.dir / 'pci' / '1'))]
                gone += ['rm -rf %s' % self.sys_path(rig, *part) for part in parts]
                result = self.run_step(rig, self.reset, after_reset=gone, heal_after=1)
                self.assertEqual(result.returncode, 1)
                # No unbind or bind: an absent device is only rescanned at its port, an unbound one not even that.
                self.assertEqual(self.driver_calls(rig), ['rescan ' + PORT_A] * 2 if case == 'absent' else [])
                log = self.results(rig, 'reset.log')
                self.assertIn(why, log)
                self.assertIn('no driver re-probe', log)
                self.assertNotIn(self.reprobe(1), log)
                self.assertIn('heal_exit=1', self.results(rig, 'heal.status'))

    def test_a_bind_that_fails_twice_refuses_and_says_the_card_is_unbound(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset, after_reset=self.race(rig, CARD_A), heal_after=1, bind_fails=2)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.driver_calls(rig), ['unbind ' + PCI_A, 'bind ' + PCI_A, 'bind ' + PCI_A])
        log = self.results(rig, 'reset.log')
        self.assertIn('refusing: %s is UNBOUND after driver re-probe 1/2' % PCI_A, log)
        self.assertIn('echo %s | sudo -n tee /sys/bus/pci/drivers/tenstorrent/bind' % PCI_A, log)
        # One failed bind is retried once.
        once = self.rig()
        result = self.run_step(once, self.reset, after_reset=self.race(once, CARD_A), heal_after=1, bind_fails=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.driver_calls(once), ['unbind ' + PCI_A, 'bind ' + PCI_A, 'bind ' + PCI_A])

    def test_a_refused_unbind_refuses_and_says_the_card_is_still_bound(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset, after_reset=self.race(rig, CARD_A), heal_after=1, unbind_refused=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.driver_calls(rig), ['unbind ' + PCI_A])                   # no bind, no second try
        log = self.results(rig, 'reset.log')
        self.assertIn('refusing: unbinding %s failed; card A (%s, PCI %s) is still bound, as it was' % (PCI_A, CARD_A, PCI_A),
                      log)
        self.assertNotIn(self.reprobe(2), log)
        self.assertIn('heal_exit=1', self.results(rig, 'heal.status'))
        self.assertTrue(self.bound(rig, PCI_A))

    def test_no_unbind_starts_in_the_steps_last_two_minutes(self):
        # A slow reset: the step is past 720 s when the heal would re-probe card A. Refused, nothing written,
        # so the runner's 14-minute timeout can never land between an unbind and its bind.
        rig = self.rig()
        result = self.run_step(rig, self.reset, after_reset=self.race(rig, CARD_A), heal_after=1, clock=721)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.driver_calls(rig), [])
        log = self.results(rig, 'reset.log')
        self.assertIn('no unbind may start after 720 s', log)
        self.assertNotIn(self.reprobe(1), log)
        self.assertIn('heal_exit=1', self.results(rig, 'heal.status'))
        self.assertTrue(self.bound(rig, PCI_A))
        # Inside the limit the same race is healed.
        early = self.rig()
        result = self.run_step(early, self.reset, after_reset=self.race(early, CARD_A), heal_after=1, clock=690)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.driver_calls(early), ['unbind ' + PCI_A, 'bind ' + PCI_A])

    # A link that did not train (run v217: card A's PCI device was gone after the reset).

    def test_an_absent_card_is_brought_back_by_one_rescan_of_its_own_upstream_port(self):
        rig = self.rig()
        node_a = rig.node(CARD_A)
        result = self.run_step(rig, self.reset, after_reset=self.untrained(rig, CARD_A), rescan_after=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # One rescan of card A's own switch port; no driver re-probe, no further reset.
        self.assertEqual(self.driver_calls(rig), ['rescan ' + PORT_A])
        calls = self.smi_calls(rig)
        self.assertEqual(calls, ['-r %s %s' % (rig.node(CARD_M), node_a), '-ls'])
        self.assert_never_card_b(rig, calls)
        log = self.results(rig, 'reset.log')
        name = 'card A (%s, PCI %s)' % (CARD_A, PCI_A)
        self.assertIn('[reset] %s by-id link still missing 60 s after the reset' % name, log)
        self.assertIn('[reset] %s %s' % (name, self.rescan(1)), log)
        self.assertIn('[reset]   %s is back 2 s after rescan 1/2, bound to tenstorrent' % PCI_A, log)
        self.assertIn('[reset] %s by-id link back 4 s after rescan 1/2 -> %s' % (name, node_a), log)
        self.assertIn("[reset] both serving cards' by-id links present after the self-heal", log)
        self.assertNotIn(self.rescan(2), log)
        self.assertNotIn('driver re-probe', log)
        self.assertNotIn('card M (', log.replace('card M and card A', ''))        # card M was never touched
        self.assertIn('heal_exit=0', self.results(rig, 'heal.status'))
        self.assertIn('reset_exit=0', self.results(rig, 'reset.status'))
        self.assertIn('%s (card A, half of the serving pair) -> %s, PCI %s' % (CARD_A, node_a, PCI_A),
                      self.results(rig, 'serving-pair-after-reset.txt'))
        self.assertTrue(self.bound(rig, PCI_A))
        self.assertTrue(self.bound(rig, PCI_B))
        # Card M absent instead: its own port, the root port, is the one rescanned.
        card_m = self.rig()
        result = self.run_step(card_m, self.reset, after_reset=self.untrained(card_m, CARD_M), rescan_after=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.driver_calls(card_m), ['rescan ' + PORT_M])
        self.assertIn('[reset] card M (%s, PCI %s) PCI device absent after reset (link did not train); rescan of its '
                      'upstream port %s 1/2' % (CARD_M, PCI_M, PORT_M), self.results(card_m, 'reset.log'))
        # Card A back only after the second rescan: two rescans, still no re-probe.
        late = self.rig()
        result = self.run_step(late, self.reset, after_reset=self.untrained(late, CARD_A), rescan_after=2)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.driver_calls(late), ['rescan ' + PORT_A] * 2)
        log = self.results(late, 'reset.log')
        self.assertIn('[reset]   %s still absent 30 s after rescan 1/2' % PCI_A, log)
        self.assertIn('[reset] %s by-id link back 4 s after rescan 2/2 -> %s' % (name, late.node(CARD_A)), log)

    def test_a_card_rescanned_back_without_its_link_goes_on_to_one_driver_re_probe(self):
        rig = self.rig()
        node_a = rig.node(CARD_A)
        result = self.run_step(rig, self.reset, after_reset=self.untrained(rig, CARD_A), rescan_after=1,
                               rescan_link=False, heal_after=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # The rescan brings the device back, bound, but udev makes no link (the telemetry race after it):
        # one driver re-probe, with every check it makes (its node re-checked for holders first).
        self.assertEqual(self.driver_calls(rig), ['rescan ' + PORT_A, 'unbind ' + PCI_A, 'bind ' + PCI_A])
        self.assertIn('-v ' + node_a, rig.fuser_calls())
        log = self.results(rig, 'reset.log')
        name = 'card A (%s, PCI %s)' % (CARD_A, PCI_A)
        self.assertIn('[reset] %s %s' % (name, self.rescan(1)), log)
        self.assertIn('[reset] %s by-id link still missing 60 s after rescan 1/2; on to the driver re-probe' % name, log)
        self.assertIn('[reset] %s %s' % (name, self.reprobe(1)), log)
        self.assertIn('[reset] %s by-id link back 4 s after driver re-probe 1/2 -> %s' % (name, node_a), log)
        self.assertLess(log.index(self.rescan(1)), log.index(self.reprobe(1)))
        self.assertNotIn(self.rescan(2), log)
        self.assertNotIn(self.reprobe(2), log)
        self.assertIn('heal_exit=0', self.results(rig, 'heal.status'))
        self.assertTrue(self.bound(rig, PCI_A))

    def test_a_card_no_rescan_brings_back_is_refused_after_two_and_fails_the_step(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset, after_reset=self.untrained(rig, CARD_A), rescan_after=0, heal_after=1)
        self.assertEqual(result.returncode, 1)
        # Exactly SERVING_PAIR_RESCANS rescans of card A's port, nothing else written: no re-probe, and
        # never card B's port.
        self.assertEqual(self.driver_calls(rig), ['rescan ' + PORT_A] * 2)
        self.assertFalse((rig.dir / ('rescans-' + CARD_B)).exists())
        log = self.results(rig, 'reset.log')
        name = 'card A (%s, PCI %s)' % (CARD_A, PCI_A)
        for n in (1, 2):
            self.assertIn('[reset] %s %s' % (name, self.rescan(n)), log)
            self.assertIn('[reset]   %s still absent 30 s after rescan %d/2' % (PCI_A, n), log)
        self.assertNotIn('rescan of its upstream port %s 3/' % PORT_A, log)
        self.assertIn('refusing: %s PCI device %s still absent after 2 rescans of its upstream port %s; no driver re-probe.'
                      % (name, PCI_A, PORT_A), log)
        self.assertIn('a power-cycle of the PCIe switch or a host reboot', log)
        self.assertNotIn(self.reprobe(1), log)
        self.assertNotIn(PORT_B, log)
        self.assertIn('heal_exit=1', self.results(rig, 'heal.status'))
        self.assertIn('reset_exit=0', self.results(rig, 'reset.status'))
        calls = self.smi_calls(rig)
        self.assertEqual(calls, ['-r %s %s' % (rig.node(CARD_M), rig.node(CARD_A)), '-ls'])     # no further reset
        self.assert_never_card_b(rig, calls)
        self.assertTrue(self.bound(rig, PCI_B))

    def test_no_rescan_starts_in_the_steps_last_two_minutes(self):
        rig = self.rig()
        result = self.run_step(rig, self.reset, after_reset=self.untrained(rig, CARD_A), rescan_after=1, clock=721)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.driver_calls(rig), [])
        log = self.results(rig, 'reset.log')
        self.assertIn('no rescan may start after 720 s', log)
        self.assertNotIn(self.rescan(1), log)
        self.assertIn('heal_exit=1', self.results(rig, 'heal.status'))
        # Inside the limit the same card is rescanned back.
        early = self.rig()
        result = self.run_step(early, self.reset, after_reset=self.untrained(early, CARD_A), rescan_after=1, clock=690)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.driver_calls(early), ['rescan ' + PORT_A])

    def test_a_failed_pair_reset_that_lost_card_a_heals_it_before_each_card_is_reset_again(self):
        # Run v217: the pair reset exits 1 and card A's switch port cannot train its link, so card A's PCI
        # device is gone. The fallback's heal, before card M is resolved, rescans card A's port; then card
        # M and card A are each resolved and reset alone, and the step passes.
        rig = self.rig()
        tt = rig.tt.as_posix()
        result = self.run_step(rig, self.reset, scenario='untrained', rescan_after=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.smi_calls(rig)
        # The pair by its nodes (2, 1); then card M alone, then card A alone, back on node 1.
        self.assertEqual(calls, ['-r %s/2 %s/1' % (tt, tt), '-r %s/2' % tt, '-r %s/1' % tt, '-ls'])
        self.assert_never_card_b(rig, calls)
        self.assertEqual(self.driver_calls(rig), ['rescan ' + PORT_A])
        lines = self.results(rig, 'reset.log').splitlines()
        self.assertIn('pair reset exited 1; resetting card M, then card A, each resolved again by board id', lines)
        rescan = lines.index('[reset] card A (%s, PCI %s) %s' % (CARD_A, PCI_A, self.rescan(1)))
        self.assertLess(rescan, lines.index('reset %s/2' % tt))                     # healed before card M's reset
        self.assertLess(lines.index('reset %s/2' % tt), lines.index('reset %s/1' % tt))
        self.assertIn('heal_exit=0', self.results(rig, 'heal.status'))
        self.assertIn('reset_exit=0', self.results(rig, 'reset.status'))
        self.assertIn('%s (card A, half of the serving pair) -> %s/1, PCI %s' % (CARD_A, tt, PCI_A),
                      self.results(rig, 'serving-pair-after-reset.txt'))
        self.assertTrue(self.bound(rig, PCI_A))
        # The same failure with card A never coming back: no resolve, no further reset, the step fails.
        gone = self.rig()
        result = self.run_step(gone, self.reset, scenario='untrained', rescan_after=0)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.smi_calls(gone), ['-r %s %s' % (gone.node(CARD_M), gone.node(CARD_A)), '-ls'])
        self.assertEqual(self.driver_calls(gone), ['rescan ' + PORT_A] * 2)
        self.assertIn('still absent after 2 rescans of its upstream port %s' % PORT_A, self.results(gone, 'reset.log'))
        self.assertIn('reset_exit=1', self.results(gone, 'reset.status'))


@unittest.skipUnless(BASH, 'bash not found')
class HarnessTests(unittest.TestCase):
    """The harnesses themselves: the default target is card B, the serving pair is refused before
    anything touches docker or a device, and the dry runs launch on card B's board id."""

    ARGS = {
        OPS / 'sdpa_decode_qwen' / 'run_card_m.sh': ['reference'],
        OPS / 'sdpa_prefill_chain' / 'run_card_m_pf.sh': ['reference'],
        HERE / 'verify-t1-g0-rig.sh': ['{out}', 'sha256:' + '0' * 64],
        HERE / 'matmul64_sweep_rig.sh': ['{out}', 'sha256:' + '0' * 64],
        HERE / 'gdn-user-batch-rig.sh': ['{out}', 'sha256:' + '0' * 64],
        HERE / 'sdpa_bench_rig.sh': ['{out}', 'sha256:' + '0' * 64],
    }
    # Scripts whose node resolution follows the selection directly (no build or docker step between).
    RESOLVE_FIRST = [OPS / 'gdn_prefill_conv' / 'run_card_m.sh', OPS / 'sdpa_decode_qwen' / 'run_card_m.sh',
                     OPS / 'verify_t2' / 'run_card_m.sh', OPS / 'c1e_gateup' / 'run_card_b.sh',
                     OPS / 'draft_slide_inplace' / 'run_card_b.sh',
                     OPS / 'sdpa_decode_qwen' / 'run_probe_k1.sh', OPS / 'sdpa_decode_slice' / 'run_card_b.sh',
                     OPS / 'pair_row_probe' / 'run_card_b.sh'] + SOURCING

    def invoke(self, path, directory, **env):
        args = [arg.replace('{out}', Path(directory, 'out').as_posix()) for arg in self.ARGS.get(path, [])]
        env.setdefault('HOME', Path(directory).as_posix())
        env.setdefault('PF_SRC', (OPS / 'sdpa_prefill_chain').as_posix())
        env.setdefault('M1_SRC', (OPS / 'sdpa_prefill_bench').as_posix())
        env.setdefault('RESULTS', Path(directory, 'results').as_posix())
        return run([path.as_posix()] + args, **env)

    def test_every_harness_refuses_the_serving_pair_before_anything_else(self):
        for path in EMBEDDING + SOURCING:
            for card in (CARD_M, CARD_A):
                with self.subTest(path=path.relative_to(ROOT).as_posix(), card=card), \
                        tempfile.TemporaryDirectory() as directory:
                    result = self.invoke(path, directory, QUAL_CARD=card)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn('refusing: QUAL_CARD=%s is card ' % card, result.stderr)
                    self.assertNotIn('docker', result.stderr.lower().replace('docker run', ''))

    def test_the_default_target_is_card_b_everywhere(self):
        if Path('/dev/tenstorrent/by-id', CARD_B).exists():
            self.skipTest('card B is present on this host: the harnesses would run')
        for path in self.RESOLVE_FIRST:
            with self.subTest(path=path.relative_to(ROOT).as_posix()), tempfile.TemporaryDirectory() as directory:
                result = self.invoke(path, directory)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn('refusing: %s (card B, the qualification card) has no device node here' % CARD_B,
                              result.stderr)

    def dry_argv(self, result):
        lines = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        self.assertEqual(len(lines), 1, result.stdout + result.stderr)
        return shlex.split(lines[0][len('### argv: '):])

    def test_the_dry_runs_launch_on_card_b_by_board_id(self):
        runs = (
            (OPS / 'sdpa_prefill_bench' / 'run_m1.sh', dict(M1_DRY_RUN='1', M1_ARGS='--arms baseline'), 'qwen-sdpa-m1-card-b'),
            (OPS / 'sdpa_prefill_chain' / 'run_card_m_pf.sh', dict(PF_DRY_RUN='1'), 'qwen-sdpa-pf-card-b'),
            (OPS / 'sdpa_decode_qwen' / 'run_probe_k1.sh', dict(PROBE_DRY_RUN='1'), 'qwen-k1probe-card-b'),
            (OPS / 'sdpa_decode_slice' / 'run_card_b.sh', dict(K64I_DRY_RUN='1'), 'qwen-k64i-card-b'),
            (OPS / 'pair_row_probe' / 'run_card_b.sh', dict(PAIR_ROW_DRY_RUN='1'), 'qwen-pairrow-card-b'),
        )
        for path, env, name in runs:
            with self.subTest(path=path.name), tempfile.TemporaryDirectory() as directory:
                result = self.invoke(path, directory, **env)
                self.assertEqual(result.returncode, 0, result.stderr)
                argv = self.dry_argv(result)
                self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_B)
                self.assertEqual(argv[argv.index('--name') + 1], name)
                self.assertIn('card=%s (card-b)' % CARD_B, result.stdout)
                self.assertEqual(result.stderr, '')
                override = self.invoke(path, directory, QUAL_CARD=CARD_M, ALLOW_SERVING_CARD='1', **env)
                self.assertEqual(override.returncode, 0, override.stderr)
                argv = self.dry_argv(override)
                self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_M)
                self.assertIn('WARNING: ALLOW_SERVING_CARD=1', override.stderr)

    def test_the_pf_results_default_to_one_directory_per_board(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke(OPS / 'sdpa_prefill_chain' / 'run_card_m_pf.sh', directory, PF_DRY_RUN='1', RESULTS='')
            self.assertEqual(result.returncode, 0, result.stderr)
            mounts = [word for word in self.dry_argv(result) if word.endswith(',dst=/results')]
            self.assertEqual(mounts, ['type=bind,src=%s/card-b,dst=/results' % (OPS / 'sdpa_prefill_chain').as_posix()])

    def test_retired_card_overrides_are_refused(self):
        for path, variable in ((OPS / 'kernels-batch64' / 'attn_prep' / 'build-and-test-b64.sh', 'CARD'),
                               (OPS / 'kernels-batch64' / 'nlp_concat_heads_decode' / 'build-and-test-b64.sh', 'CARD'),
                               (HERE / 'gdn-user-batch-rig.sh', 'GDN_USER_BATCH_DEVICE')):
            with self.subTest(path=path.name), tempfile.TemporaryDirectory() as directory:
                result = self.invoke(path, directory, **{variable: '/dev/tenstorrent/2'})
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn('is retired', result.stderr)


if __name__ == '__main__':
    if '--sync' in sys.argv:
        sync()
        sys.exit(0)
    unittest.main()
