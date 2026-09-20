"""Which scripts/ci modules differ between this repo and the image's bundle?

Only a thin adapter layer is COPYed over the bundle (inventory 2026-09-20). The
verify/commit core - verifier_engine, model_batch, full_dflash_request,
dflash_request_runtime, gdn_device_loop_state, serving_runner_bridge and their
imports - runs as the bundle's frozen copy. Overriding one of them is only safe
when this repo's copy is byte-identical or a deliberate change, so this hashes
every repo module against /experiment-scripts/ci and lists the drift.

The probe lane mounts the repo at /probe, so /probe/<name> is the repo copy and
/experiment-scripts/ci/<name> is what the server runs.

CPU only: no device, no weights.
"""

import hashlib
import os
import sys

CORE = ('verifier_engine.py', 'model_batch.py', 'full_dflash_request.py',
        'dflash_request_runtime.py', 'gdn_device_loop_state.py', 'serving_runner_bridge.py',
        'serving_vllm_contract.py', 'serving_cache_owner.py', 'serving_page_binding.py',
        'dflash_combined_request.py', 'dflash_prefill_window.py', 'dflash_proposal_trace.py',
        'gdn_snapshot.py', 'gdn_multitoken_conv.py', 'gdn_prefix.py', 'gdn_batched_conv.py',
        'feature_collective.py', 'feature_projection.py', 'draft_shared_head.py',
        'draft_selector.py', 'prepared_target_features.py', 'target_features.py',
        'attention_replay.py', 'attention_batch.py', 'dflash_t16_native_attention.py')


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read().replace(b'\r\n', b'\n')).hexdigest()


def main():
    repo, image = '/probe', '/experiment-scripts/ci'
    names = sorted(name for name in os.listdir(repo)
                   if (name.endswith('.py') or name.endswith('.cpp')) and not name.startswith('test_'))
    same, differ, missing_image = [], [], []
    for name in names:
        here = os.path.join(repo, name)
        there = os.path.join(image, name)
        if not os.path.exists(there):
            missing_image.append(name)
        elif digest(here) == digest(there):
            same.append(name)
        else:
            differ.append(name)
    print('repo modules: %d  identical: %d  differ: %d  not in image: %d'
          % (len(names), len(same), len(differ), len(missing_image)))
    print('DIFFER: %s' % ' '.join(differ))
    print('NOT IN IMAGE: %s' % ' '.join(missing_image))
    core_differ = [name for name in CORE if name in differ]
    core_same = [name for name in CORE if name in same]
    core_absent = [name for name in CORE if name in missing_image or name not in names]
    print()
    print('VERDICT')
    print('  core modules identical to the bundle (safe to override): %s' % ' '.join(core_same))
    print('  core modules that DIFFER (override would change behaviour): %s' % ' '.join(core_differ))
    print('  core modules absent: %s' % ' '.join(core_absent))
    return 0


if __name__ == '__main__':
    sys.exit(main())
