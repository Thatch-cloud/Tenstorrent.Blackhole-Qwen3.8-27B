"""scripts/ci/card_set.sh: the four-card set is every Blackhole board present, resolved by board id at the moment of
use, never named; the heal writes only to the pre-reset snapshot's own PCI addresses and upstream ports.

Runs the library in bash (Git Bash on Windows; skipped without it) against test_qual_card's FakeRig: a stand-in
/dev/tenstorrent whose by-id entries are files naming their node, stubs for readlink, device numbers and sysfs, and
here a fake /sys built by the shell. The board ids are made up: the library must work for any four."""

import os
import tempfile
import unittest
from pathlib import Path

from test_qual_card import BASH, NL, FakeRig, q, run

HERE = Path(__file__).resolve().parent
CARD_SET = HERE / 'card_set.sh'
IDS = ('blackhole-AAAA000000000001', 'blackhole-BBBB000000000002', 'blackhole-CCCC000000000003',
       'blackhole-DDDD000000000004')
PCI = dict(zip(IDS, ('0000:11:00.0', '0000:22:00.0', '0000:33:00.0', '0000:44:00.0')))
PORTS = dict(zip(IDS, ('0000:10:01.1', '0000:21:00.0', '0000:32:00.0', '0000:21:01.0')))
# Renumbered on purpose: node order is not id order, as after any reset.
NODES = dict(zip(IDS, ('3', '0', '2', '1')))


def rig(directory, ids=IDS):
    return FakeRig(directory, nodes={card: NODES[card] for card in ids}, pci={card: PCI[card] for card in ids})


def sysfs(fake, present=IDS):
    """Shell lines for a fake /sys: each present board's PCI device bound to the driver, and every upstream port."""
    pci = Path(fake.dir) / 'sys' / 'bus' / 'pci'
    lines = ['mkdir -p %s' % q(pci / 'drivers' / 'tenstorrent'),
             ': > %s' % q(pci / 'drivers' / 'tenstorrent' / 'bind'),
             ': > %s' % q(pci / 'drivers' / 'tenstorrent' / 'unbind')]
    for card in present:
        lines.append('mkdir -p %s %s' % (q(pci / 'devices' / PCI[card]), q(pci / 'drivers' / 'tenstorrent' / PCI[card])))
    for port in sorted(set(PORTS.values())):
        lines.append('mkdir -p %s' % q(pci / 'devices' / port))
        lines.append(': > %s' % q(pci / 'devices' / port / 'rescan'))
    return lines


def upstream_stub():
    cases = ' '.join('%s) echo %s ;;' % (PCI[card], PORTS[card]) for card in IDS)
    return 'card_set_upstream_of() { case $1 in %s esac; }' % cases


# sudo runs the command and logs it; a write to the fake kernel's files then acts through $FAKE_DIR/kernel.sh.
SUDO = ('sudo() { [ "$1" = -n ] && shift; local input=""; if [ "$1" = tee ]; then input=$(cat); '
        'echo "sudo tee ${2##*/sys/} <- $input" >> "$FAKE_DIR/sudo.log"; printf "%s\\n" "$input" > "$2"; '
        '[ -f "$FAKE_DIR/kernel.sh" ] && . "$FAKE_DIR/kernel.sh" "${2##*/}" "$input"; return 0; fi; "$@"; }')


@unittest.skipUnless(BASH, 'bash not found')
class CardSetTests(unittest.TestCase):
    SHOW = ('card_set_resolve; echo "ids=${CARD_SET_IDS[*]}"; echo "nodes=${CARD_SET_NODES[*]}"; '
            'echo "pci=${CARD_SET_PCI[*]}"; echo "ports=${CARD_SET_PORTS[*]}"')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def fake(self, ids=IDS):
        return rig(tempfile.mkdtemp(dir=self.tmp.name), ids)

    def run_set(self, fake, body, present=IDS):
        return fake.run(NL.join(sysfs(fake, present) + [upstream_stub(), SUDO, body]), library=CARD_SET)

    def test_the_set_is_every_board_present_by_id_with_its_own_upstream_port(self):
        fake = self.fake()
        result = self.run_set(fake, self.SHOW)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('ids=%s' % ' '.join(IDS), result.stdout, 'sorted by board id, not node')
        self.assertIn('nodes=%s' % ' '.join(fake.node(card) for card in IDS), result.stdout)
        self.assertIn('pci=%s' % ' '.join(PCI[card] for card in IDS), result.stdout)
        self.assertIn('ports=%s' % ' '.join(PORTS[card] for card in IDS), result.stdout)

    def test_a_board_short_or_a_board_too_many_is_refused(self):
        short = self.fake(IDS[:3])
        result = self.run_set(short, 'card_set_resolve')
        self.assertEqual(result.returncode, 1)
        self.assertIn('refusing: 3 Blackhole boards are present, not 4', result.stderr)
        five = self.fake()
        extra = 'blackhole-EEEE000000000005'
        (five.tt / '9').write_text('')
        (five.tt / 'by-id' / extra).write_text((five.tt / '9').as_posix())
        result = self.run_set(five, 'card_set_nodes 30')
        self.assertEqual(result.returncode, 1)
        self.assertIn('5 Blackhole boards are present', result.stderr)
        self.assertNotIn('all 4 boards resolved after', result.stdout, 'more boards never waits')

    def test_two_ids_on_one_node_is_refused(self):
        fake = self.fake()
        (fake.tt / 'by-id' / IDS[3]).write_text(fake.node(IDS[0]))
        result = self.run_set(fake, 'card_set_nodes')
        self.assertEqual(result.returncode, 1)
        self.assertIn('both resolve to %s' % fake.node(IDS[0]), result.stderr)

    def test_a_board_coming_back_late_is_waited_for(self):
        fake = self.fake()
        fake.pending(IDS[2], '7', polls=3)
        result = self.run_set(fake, 'card_set_nodes 10; echo "nodes=${CARD_SET_NODES[*]}"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('all 4 boards resolved after 6 s', result.stdout)
        self.assertIn((fake.tt / '7').as_posix(), result.stdout)

    def test_the_heal_writes_only_to_the_snapshot(self):
        fake = self.fake()
        body = NL.join([
            'card_set_resolve > /dev/null',
            'for pci in 0000:55:00.0 "" "*" %s:x; do st=0; card_set_driver_write "$pci" unbind 2>> "$FAKE_DIR/err.log" || st=$?; '
            'echo "[$pci]=$st"; done' % PCI[IDS[0]],
            'for port in 0000:54:00.0 %s ""; do st=0; card_set_rescan_write "$port" 2>> "$FAKE_DIR/err.log" || st=$?; '
            'echo "<$port>=$st"; done' % PCI[IDS[1]],
            'st=0; card_set_driver_write %s remove 2>> "$FAKE_DIR/err.log" || st=$?; echo "remove=$st"' % PCI[IDS[1]],
            'card_set_driver_write %s unbind' % PCI[IDS[1]],
            'card_set_rescan_write %s' % PORTS[IDS[3]],
        ])
        result = self.run_set(fake, body)
        self.assertEqual(result.returncode, 0, result.stderr)
        for pci in ('0000:55:00.0', '', '*', PCI[IDS[0]] + ':x'):
            self.assertIn('[%s]=2' % pci, result.stdout)
        for port in ('0000:54:00.0', PCI[IDS[1]], ''):
            self.assertIn('<%s>=2' % port, result.stdout, 'a board address is not an upstream port')
        self.assertIn('remove=2', result.stdout)
        self.assertEqual((fake.dir / 'sudo.log').read_text().splitlines(),
                         ['sudo tee bus/pci/drivers/tenstorrent/unbind <- %s' % PCI[IDS[1]],
                          'sudo tee bus/pci/devices/%s/rescan <- 1' % PORTS[IDS[3]]])

    def test_the_heal_needs_the_pre_reset_snapshot(self):
        fake = self.fake()
        result = self.run_set(fake, 'card_set_nodes; card_set_heal 2')
        self.assertEqual(result.returncode, 1)
        self.assertIn('no pre-reset snapshot', result.stderr)
        result = self.run_set(fake, 'card_set_resolve; card_set_heal abc')
        self.assertEqual(result.returncode, 1)
        self.assertIn('is not a number of seconds', result.stderr)

    def test_all_links_back_needs_no_heal(self):
        fake = self.fake()
        result = self.run_set(fake, 'card_set_resolve > /dev/null; card_set_heal 4')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('all 4 boards\' by-id links present 0 s after the reset; no heal', result.stdout)
        self.assertFalse((fake.dir / 'sudo.log').exists())

    def test_the_telemetry_race_is_healed_by_a_driver_reprobe(self):
        fake = self.fake()
        card = IDS[1]
        node = fake.node(card)
        # After the reset the board's node is back but udev made no by-id link; the bind brings it back.
        kernel = ('if [ "$1" = bind ] && [ "$2" = %s ]; then printf "%%s" %s > %s; fi'
                  % (PCI[card], q(node), q(fake.tt / 'by-id' / card)))
        (fake.dir / 'kernel.sh').write_text(kernel + NL)
        body = NL.join(['card_set_resolve > /dev/null', 'rm -f %s' % q(fake.tt / 'by-id' / card),
                        'fuser() { return 1; }', 'CARD_SET_REPROBE_WAIT=2', 'card_set_heal 2'])
        result = self.run_set(fake, body)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('driver re-probe 1/2 of %s' % PCI[card], result.stdout)
        self.assertEqual((fake.dir / 'sudo.log').read_text().splitlines(),
                         ['sudo tee bus/pci/drivers/tenstorrent/unbind <- %s' % PCI[card],
                          'sudo tee bus/pci/drivers/tenstorrent/bind <- %s' % PCI[card]])

    def test_an_absent_board_is_rescanned_at_its_own_upstream_port(self):
        fake = self.fake()
        card = IDS[2]
        pci_dir = fake.dir / 'sys' / 'bus' / 'pci'
        kernel = ('if [ "$1" = rescan ]; then mkdir -p %s %s; printf "%%s" %s > %s; fi'
                  % (q(pci_dir / 'devices' / PCI[card]), q(pci_dir / 'drivers' / 'tenstorrent' / PCI[card]),
                     q(fake.node(card)), q(fake.tt / 'by-id' / card)))
        (fake.dir / 'kernel.sh').write_text(kernel + NL)
        body = NL.join(['card_set_resolve > /dev/null',
                        'rm -rf %s %s %s' % (q(pci_dir / 'devices' / PCI[card]),
                                             q(pci_dir / 'drivers' / 'tenstorrent' / PCI[card]),
                                             q(fake.tt / 'by-id' / card)),
                        'CARD_SET_RESCAN_WAIT=2', 'CARD_SET_REPROBE_WAIT=2', 'card_set_heal 2'])
        result = self.run_set(fake, body)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('rescan of its upstream port %s 1/2' % PORTS[card], result.stdout)
        self.assertEqual((fake.dir / 'sudo.log').read_text().splitlines(),
                         ['sudo tee bus/pci/devices/%s/rescan <- 1' % PORTS[card]])

    def test_a_board_that_never_returns_is_refused_after_two_rescans(self):
        fake = self.fake()
        card = IDS[0]
        pci_dir = fake.dir / 'sys' / 'bus' / 'pci'
        body = NL.join(['card_set_resolve > /dev/null',
                        'rm -rf %s %s' % (q(pci_dir / 'devices' / PCI[card]), q(fake.tt / 'by-id' / card)),
                        'CARD_SET_RESCAN_WAIT=2', 'card_set_heal 2'])
        result = self.run_set(fake, body)
        self.assertEqual(result.returncode, 1)
        self.assertIn('still absent after 2 rescans of its upstream port %s' % PORTS[card], result.stderr)
        self.assertEqual(len((fake.dir / 'sudo.log').read_text().splitlines()), 2)

    def test_the_holder_check_passes_tt_smi_and_refuses_anything_else(self):
        fake = self.fake()
        telemetry = NL.join(['card_set_nodes > /dev/null',
                             'fuser() { echo "$1: root 99 F.... tt-smi" >&2; return 0; }',
                             'card_set_unheld "$FAKE_DIR/holders.txt" && echo CLEAR'])
        result = self.run_set(fake, telemetry)
        self.assertIn('CLEAR', result.stdout)
        held = NL.join(['card_set_nodes > /dev/null', 'sleep() { :; }',
                        'fuser() { echo "$2: thatch 4242 F.... python3" >&2; return 0; }',
                        'card_set_unheld "$FAKE_DIR/holders.txt" || echo HELD'])
        result = self.run_set(fake, held)
        self.assertIn('HELD', result.stdout)
        unresolved = self.run_set(fake, 'card_set_unheld "$FAKE_DIR/holders.txt" || echo REFUSED')
        self.assertIn('REFUSED', unresolved.stdout)
        self.assertIn('not resolved', unresolved.stderr)

    def test_the_heal_waits_ten_seconds_for_the_links_by_default(self):
        result = self.run_set(self.fake(), 'echo "wait=$CARD_SET_HEAL_WAIT"')
        self.assertIn('wait=10', result.stdout, result.stderr)

    def test_the_library_names_no_board_and_no_address(self):
        text = CARD_SET.read_text(encoding='utf-8')
        import re

        self.assertIsNone(re.search(r'blackhole-[0-9A-F]{16}', text), 'no board id')
        self.assertIsNone(re.search(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', text), 'no PCI address')
        self.assertNotIn(chr(13), text)


if __name__ == '__main__':
    unittest.main()
