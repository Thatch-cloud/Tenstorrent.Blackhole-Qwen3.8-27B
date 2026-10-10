"""The WP5 card templates (scripts/ci/references/fusion-jobs-wp5), generated (stdlib only).

    python3 optimisation/ttnn-op/ccl_sweep/make_pack.py --write   regenerate the templates and ORDER.txt
    python3 optimisation/ttnn-op/ccl_sweep/make_pack.py --check   exit 1 if the checked-in pack is not what this generates

What make_fusion_jobs.py does not make for the CCL options (F-C1): the isolated fabric sweeps (P0-P4: one fabric config and one payload per job, the fabric action's
ccl-sweep probe, tp4_ccl_sweep_probe.py), the plumbing control's audited attach (CCLS), and the hang qualification (CCLH1-CCLH5: five consecutive completions of the hang
shapes on the lever's profile). The audited attach (CCLA) and the timed ABAB (CCLC1 CCLL1 CCLC2 CCLL2) of the lever itself are make_fusion_jobs.py's, on the same image.

The lever's profile is `<base>-fx-ccl` (base = the production profile without its -traffic suffix) (make_fusion_profiles.py, from the lever `ccl` of scripts/ci/fusion-wp/WP5.json); the sweep decides its value, so the hang runs
and the ABAB wait for it. Nothing here names a rig, a card, a host or a registry; nothing stops, starts or hands back the node agent.
"""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
# A sibling of references/fusion-jobs, not a subfolder: test_tp4_fusion_jobs reads every entry of that folder as a file.
FOLDER = REPO / 'scripts' / 'ci' / 'references' / 'fusion-jobs-wp5'

IMAGE = 'tp4-fusion-1'
PRODUCTION = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'
BASE = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er'             # the integrator's twin namespace is BASE + '-fx-' (make_fusion_profiles.NAMESPACE)
LEVER = BASE + '-fx-ccl'
PLUMBING_AUDIT = BASE + '-fx-ccl-served-audit'
HANG_TESTS = 'warmup,concurrent4_steady,concurrent8_steady,steady_resend,replay_concurrent4,replay_concurrent8,concurrent8_code_equal,concurrent5_split,concurrent8_drain'
AUDIT_TESTS = 'warmup,concurrent8_steady'

HEADER = """# A committed TEMPLATE of .github/c2-serving-job.env for the op-fusion programme's four-card window, WP5 (CCL options; scripts/ci/c2_serving_job.py parses it): copy it over
# .github/c2-serving-job.env on a throwaway commit of tp4/fusion-1 and push a tag experiment/c2-serving-vN. Nothing here names a rig, a card, a host or a registry: the cards
# are resolved at run time. The jobs are templates only: no tag is pushed from the branch by its author. ORDER.txt has the order, the stop rules, the minutes and the read rules.
# ONE IMAGE FOR THE WHOLE WINDOW: C2_IMAGE_TAG=%(image)s is a PLACEHOLDER for the tag B0 builds from the pushed branch head (sed -i 's/%(image)s/<tag>/' *.env). The fabric sweeps
# (P0-P4) run the probe from the checkout and need only an image with the models tree; the lever jobs need the image B0 built. The cards are under development."""

PROBES = (
    ('P0-ccl-sweep-quick', 'stop', 15, '15', 'FABRIC_1D', 'ccl-sweep-quick',
     "P0 (STOP): the harness smoke: the short grid (one value of each option, no barrier-removal arms) under FABRIC_1D, the served fabric. It proves the probe opens the mesh, takes the model's TT_CCL and "
     "tt_all_reduce, bit-compares the five scenarios against the sequential engine's call and times them in a trace, before 40 minutes are spent on the full grid. READ: 'CCL_SWEEP verdict=DONE', "
     "every served row EXACT in all five scenarios, a per-call time near 18 us for rs and gathers near 18 us (M676: 18.2 and 18.4 us median). BASE-INEXACT (exit 1) is a finding about the stack, not the "
     "probe: stop and read the rows. A watchdog exit (3) names the config that hung: reset, then run P1 again as ccl-sweep-safe or ccl-sweep-core (the later stages dropped). Duration (estimate): 12 minutes."),
    ('P1-ccl-sweep-fabric1d', 'soft', 42, '42', 'FABRIC_1D', 'ccl-sweep',
     "P1: THE SWEEP under FABRIC_1D (what the TT plugin sets), the runtime's default payload: every offered value of links, workers, chunks per sync and buffers per channel for the unit-major "
     "reduce-scatter (input in DRAM and in L1) and the norm gather (DRAM and L1 input, the norm's sharded output and DRAM), one at a time around the served values, then the best values together; "
     "then the EXOTIC configs (three workers per direction, which cut a channel at tiles the chunk boundaries do not meet; the gather's line route and via-broadcast program), last the PROBE-ONLY barrier-removal arms (barrier_semaphore=None, persistent buffers) that no stack flag can name. READ: 'CCL_SWEEP verdict=DONE', the promote string of each scenario "
     "(promote=...), the A/A noise of each, every FAIL-BYTES row (an option that changed a bit: it can never be a lever; report it). The promote string with the most gain in the scenario the "
     "stack runs becomes the lever's value. Duration (estimate): 30 minutes."),
    ('P2-ccl-sweep-fabric1d-p8192', 'soft', 32, '42', 'FABRIC_1D', 'ccl-sweep-p8192',
     "P2: the same sweep with the fabric router asked for an 8192-byte packet payload (four bfloat16 tile pages; the runtime's default is 4352, two). The run measures nothing unless the runtime's "
     "own readback says 8192 (exit 2, lever_moved false). The payload is a PROCESS option: adopting it needs the serving worker's fabric call patched (scripts/ci/fabric_packet_adopt_patch.py, "
     "authorised 2026-09-19 and never wired into the image), not a per-call flag. Exact by construction: tile_granularity stays 8 for every payload from 4096. READ: the payload line, the served per-call times "
     "against P1's, and 'compare_reports' (the served fingerprints equal P1's). Duration (estimate): 30 minutes."),
    ('P3-ccl-sweep-ring', 'soft', 32, '42', 'FABRIC_1D_RING', 'ccl-sweep-core',
     "P3: the core sweep under FABRIC_1D_RING (the alternative fabric config; a profile choice, additional_config fabric_config), without the exotic stage and the barrier-removal arms. READ: as P1; the served fingerprints must "
     "equal P1's (the fabric config changes routes, not the add order). Duration (estimate): 30 minutes."),
    ('P4-ccl-sweep-ring-p8192', 'soft', 32, '42', 'FABRIC_1D_RING', 'ccl-sweep-p8192',
     "P4: FABRIC_1D_RING with the 8192-byte payload: the two process options together. READ: as P2. Duration (estimate): 30 minutes."),
)


def wrap(text, width=170):
    lines, current = [], ''
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + ' ' + word) if current else word
    if current:
        lines.append(current)
    return lines


def env_text(why, settings):
    lines = [HEADER % dict(image=IMAGE)] + ['# ' + piece for piece in wrap(why)] + settings
    return '\n'.join(lines) + '\n'


def jobs():
    """[dict(name, mode, minutes, text, needs)] in the order they run."""
    rows = []
    for name, mode, minutes, box, fabric, probe, why in PROBES:
        rows.append(dict(name=name, mode=mode, minutes=minutes,
                         needs=('P1 P2 P3 P4 <- P0',) if name.startswith('P0') else (),
                         text=env_text(why, ['C2_CARDS=quad', 'C2_ACTIONS=reset fabric', 'C2_IMAGE_TAG=%s' % IMAGE, 'C2_PROFILE=general-tp4', 'C2_FABRIC=%s' % fabric,
                                             'C2_FABRIC_PROBE=%s' % probe, 'C2_BOX_MINUTES=%s' % box])))
    rows.append(dict(name='CCLS-plumbing-audit-attach', mode='soft', minutes=40, needs=('CCLS <- B0 X0',),
                     text=env_text(
                         "CCLS: the PLUMBING CONTROL audited: %s (QWEN_FAST_CCL_OPTIONS=served plus the audit) over warmup and concurrent8_steady. The unit-major reduce-scatter hook and the DistributedNorm "
                         "gather shim are engaged with an EMPTY set, so every call carries the values the model passes today and the audit compares each audited call with itself. READ: the engaged line "
                         "(rs=128 or more, ag=128 or 129 a forward, fallbacks=0), the audit lines for both ops from a replay (round>=1, chips=4, exact=True) for each block, hashes equal to the production "
                         "profile's. A fall-back line here is the census not matching the stack's own calls (the reason is in the line): fix the census before any set is read. Duration (estimate): 40 minutes."
                         % PLUMBING_AUDIT,
                         ['C2_CARDS=quad', 'C2_ACTIONS=reset smoke', 'C2_IMAGE_TAG=%s' % IMAGE, 'C2_PROFILE=%s' % PLUMBING_AUDIT, 'C2_SMOKE_TESTS=%s' % AUDIT_TESTS])))
    for index in range(1, 6):
        rows.append(dict(name='CCLH%d-ccl-hang-shapes' % index, mode='stop', minutes=90, needs=('CCLH1 CCLH2 CCLH3 CCLH4 CCLH5 <- CCLA',) if index == 1 else (),
                         text=env_text(
                             "CCLH%d (run %d of 5): the HANG SHAPES at eight seats on %s (the production stack plus the CCL options lever, audits off). The lever changes how the 128 reduce-scatters and 129 gathers "
                             "of a pass are issued (worker cores, mux cores, semaphore cadence, buffer depth) and the hang class of this stack was collective sequencing, so it is gated like every new collective "
                             "shape: FIVE consecutive completions (CCLH1 to CCLH5), each after its own all-four reset. The shapes: concurrent4_steady, concurrent8_steady, steady_resend (an engine build after many "
                             "packed rounds), replay_concurrent4 and replay_concurrent8 (a rows=2 sequential replay), the exactness reference concurrent8_code_equal, then the split-block and drain shapes. READ: every user at "
                             "budget, c2_smoke_check clean (the engaged line, no fell-back line), no watchdog or replay-deadline exit, the concurrent8_code_equal hashes equal in every run and equal to the production "
                             "profile's. A stall waits for the job timeout and records nothing, which is itself the finding. Duration (estimate): 90 minutes. Stop rule: one stall stops the window." % (index, index, LEVER),
                             ['C2_CARDS=quad', 'C2_ACTIONS=reset smoke', 'C2_IMAGE_TAG=%s' % IMAGE, 'C2_PROFILE=%s' % LEVER, 'C2_SMOKE_TESTS=%s' % HANG_TESTS])))
    return rows


def render_order(rows):
    total = sum(row['minutes'] for row in rows)
    lines = [
        "# The WP5 (CCL options, F-C1) part of the op-fusion window (docs/tp4-fusion.md): the isolated fabric sweeps, the plumbing control and the hang qualification. The lever's audited attach (CCLA) and its timed ABAB (CCLC1 CCLL1",
        "# CCLC2 CCLL2) are make_fusion_jobs.py's, in scripts/ci/references/fusion-jobs, on the same image. One template per line: <template name without .env> <stop|soft> <image tag> <estimated minutes>.",
        "# THE ORDER: P0 (the harness smoke) first; P1 (the sweep under the served fabric) and, as the window allows, P2-P4 (the process options: the 8192 payload, FABRIC_1D_RING); then the owner reads the promote strings and the lever's",
        "# value in scripts/ci/fusion-wp/WP5.json is set to the winner (one line; make_fusion_profiles.py --write regenerates the twins; the placeholder value is the candidate before any sweep); then B0 builds the image,",
        "# X0 (the generated pack), CCLS (the plumbing control), CCLA (the lever audited), CCLH1-CCLH5 (the hang shapes, five consecutive completions) and the generated ABAB. P0-P4 need only an image with the models tree (the probe",
        "# runs from the checkout): they may run before B0 on the window's previous image, and each begins with its own all-four reset.",
        "# THE PROBE LAUNCH needs scripts/ci/fusion-wp/WP5-fabric-probe.patch applied (FABRIC_PROBES in c2_serving_job.py and the case in qwen-c2-serving.yml): until it is, c2_serving_job.py refuses P0-P4 by name.",
        "# CI PAUSED for the timed jobs only (P-jobs time a collective in a trace: the host runs no build or push beside them). The cards stay under development: no agentstart, no agentstop, no hand-back, no deploy.",
        "# GRID: the four cards expose a 13x10 compute grid since the firmware unlock; the sweep reads nothing from it (the collectives choose their worker cores from the device), and v678 measured the 13x10 cards at AG 0.65 us and RS 0.5 us per call",
        "# below v676 (collectives 4.61 to 4.63 ms a verify pass): P1 is the control of its own run, never read against an 11x10 number.",
    ]
    for row in rows:
        lines.extend('# NEEDS ' + need for need in row['needs'])
    lines += [
        "# READ RULES (each also needs no stall; the probe jobs end with 'CCL_SWEEP verdict=...' and a JSON report, the lever jobs with c2_smoke_check clean):",
        "#   P0     'CCL_SWEEP verdict=DONE' with every served row EXACT. Anything else stops the sweeps.",
        "#   P1-P4  'CCL_SWEEP verdict=DONE' (INCOMPLETE and BASE-INEXACT are findings, a watchdog exit 3 names the hung config: reset and rerun as ccl-sweep-safe or ccl-sweep-core). A promote string needs the config EXACT on every seed, in-trace, and a paired",
        "#          gain of at least max(0.3 us, twice the A/A noise) in at least 4 of 5 rounds; the probe-only arms are never a promote. P2-P4 are read against P1's served per-call times and fingerprints.",
        "#   CCLS   the engaged line with fallbacks=0, replay audit lines for both ops (exact=True), hashes equal to the production profile's.",
        "#   CCLH   five consecutive completions, every user at budget, the concurrent8_code_equal hashes equal in every run and to the control's.",
        "# NO-GO for adopting a set: any FAIL-BYTES row for the set's options, any audit mismatch, any fall-back line, any stall in CCLH1-CCLH5, a paired ABAB that is not negative beyond the control-to-control floor.",
        "# Time (estimate, minutes): P0 15, P1 42, P2-P4 32 each, CCLS 40, CCLH 90 each: %d minutes (%d h %d min) in all." % (total, total // 60, total % 60),
    ]
    lines += ['%s %s %s %d' % (row['name'], row['mode'], IMAGE, row['minutes']) for row in rows]
    return '\n'.join(lines) + '\n'


def generate():
    rows = jobs()
    out = dict(('%s.env' % row['name'], row['text']) for row in rows)
    out['ORDER.txt'] = render_order(rows)
    return out


def stale(folder, wanted):
    found = dict((path.name, path.read_text(encoding='utf-8')) for path in Path(folder).glob('*') if path.is_file())
    return [name for name in sorted(set(found) | set(wanted)) if found.get(name) != wanted.get(name)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--folder', default=str(FOLDER))
    arguments = parser.parse_args(argv)
    if not (arguments.write or arguments.check):
        parser.error('--write or --check')
    wanted = generate()
    folder = Path(arguments.folder)
    differ = stale(folder, wanted) if folder.is_dir() else sorted(wanted)
    if arguments.check:
        for name in differ:
            sys.stderr.write('%s is not what make_pack.py generates: run it with --write\n' % (folder / name))
        return 1 if differ else 0
    folder.mkdir(parents=True, exist_ok=True)
    for name in differ:
        path = folder / name
        if name in wanted:
            with open(str(path), 'w', encoding='utf-8', newline='\n') as handle:
                handle.write(wanted[name])
        else:
            path.unlink()
    return 0


if __name__ == '__main__':
    sys.exit(main())
