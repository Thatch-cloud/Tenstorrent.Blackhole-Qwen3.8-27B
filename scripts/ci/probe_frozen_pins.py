"""Which sources does the frozen 32K combined runtime pin, and which image files break the pin?

Run 35495227738 (image v42) refused the first admission at
frozen_combined_runtime.qualify:102-103, 'Combined runtime component source
differs: attention_replay.py'. That check hashes every source named in the
frozen evidence reports (draft-numerical.json and target-replay.json,
'sources') against /experiment-scripts/ci, and stops at the first mismatch, so
one name says nothing about the others. Image v42 overrides eight files the
earlier images did not; several may be pinned.

Reads, from the IMAGE by path: the evidence reports under
/experiment-scripts/ci/frozen-evidence, builds the same expected map
qualify() builds (same exclusions), hashes each named file in
/experiment-scripts/ci, and prints every mismatch with both digests. Also
prints, for each overridden file the fast-serving image copies (the Dockerfile
list), whether it is pinned at all.

CPU only: no device, no weights, no imports from the image.
"""

import hashlib
import io
import json
import sys
from pathlib import Path

CI = Path('/experiment-scripts/ci')
EVIDENCE = CI / 'frozen-evidence'
EXCLUDED = {'frozen_probe_evidence.py', 'dspark_attention_8k_gate.py', 'dspark_attention_value_diagnostics.py'}
OVERRIDDEN = ('attention_replay.py', 'gdn_snapshot.py', 'dflash_prefill_window.py', 'packed_verifier.py',
              'serving_packed_step.py', 'gdn_device_loop_state.py', 'gdn_records.py', 'model_batch.py',
              'verifier_engine.py', 'dflash_device.py', 'draft_kv_history.py', 'serving_buffer_pool.py',
              'serving_vllm_contract.py', 'draft_attention_branch.py', 'draft_mlp_branch.py',
              'draft_convolution.py', 'draft_convolution_fused.py', 'draft_convolution_fused_io.cpp',
              'dflash_t16_native_attention.py', 'verifier_pack.py')


def show(label, value):
    print('%-44s %s' % (label, value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    show('frozen-evidence present', EVIDENCE.is_dir())
    reports = {}
    for path in sorted(EVIDENCE.rglob('*.json')) if EVIDENCE.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except BaseException:
            continue
        if isinstance(data, dict) and isinstance(data.get('sources'), dict):
            reports[path.relative_to(EVIDENCE).as_posix()] = data['sources']
    show('reports with a sources map', len(reports))
    for name in sorted(reports):
        show('  ' + name, '%d sources' % len(reports[name]))
    draft = next((s for n, s in reports.items() if n.endswith('draft-numerical.json')), {})
    target = next((s for n, s in reports.items() if n.endswith('target-replay.json')), {})
    expected = {name: checksum for name, checksum in draft.items()
                if not name.endswith('-probe.py') and not name.startswith('../') and name not in EXCLUDED}
    for name, checksum in target.items():
        if name.endswith('-probe.py'):
            continue
        expected[name] = checksum
    show('pinned sources (as qualify builds them)', len(expected))
    mismatched = []
    for name in sorted(expected):
        path = CI / name
        live = digest(path) if path.is_file() else 'MISSING'
        if live != expected[name]:
            mismatched.append(name)
            show('  MISMATCH ' + name, 'expected %s live %s' % (expected[name][:12], live[:12]))
    show('mismatched pinned sources', mismatched)
    print('----- overridden files: pinned? -----')
    for name in OVERRIDDEN:
        show('  ' + name, 'PINNED' if name in expected else 'not pinned')
    print()
    print('VERDICT')
    if not expected:
        print('  No frozen evidence sources found in the image; the pin cannot be read here.')
        return 0
    if mismatched:
        print('  The frozen 32K combined runtime pins %d sources; these image files break it: %s.'
              % (len(expected), ', '.join(mismatched)))
        print('  Each must return to the pinned bytes (its change moved to an unpinned module)')
        print('  or the frozen evidence regenerated for it.')
    else:
        print('  Every pinned source matches; the refusal must have another cause.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
