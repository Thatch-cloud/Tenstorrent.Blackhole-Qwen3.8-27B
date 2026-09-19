"""Experimental local-copy scheduling; no arithmetic or buffer changes."""

import hashlib
from unittest.mock import patch

from gdn_multitoken import replace_once


BUILD_RECORDS = []
ORIGINAL = '''            for (uint32_t word = 0; word < 16; ++word) {
                target[word] = source[word];
                target[256 + word] = source[256 + word];
            }
'''


def grouped_copy():
    lines = ['            for (uint32_t word = 0; word < 16; word += 4) {']
    for face, offset in (('low', 0), ('high', 256)):
        for index in range(4):
            lines.append(f'                const uint32_t {face}_{index} = source[word + {offset + index}];')
    for face, offset in (('low', 0), ('high', 256)):
        for index in range(4):
            lines.append(f'                target[word + {offset + index}] = {face}_{index};')
    return '\n'.join(lines + ['            }', ''])


def reader(source):
    return replace_once(source, ORIGINAL, grouped_copy())


def build(operations, mesh, tensors, *, root):
    import gdn_shared_qk_pipeline as pipeline

    original = pipeline.load_kernels

    def kernels(requested_root):
        result = {stage: dict(parts) for stage, parts in original(requested_root).items()}
        before = result['recurrence']['reader']
        after = reader(before)
        result['recurrence']['reader'] = after
        BUILD_RECORDS.append(dict(control_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(after.encode()).hexdigest(),
            extra_cb_bytes=0, local_copy_group_words=4, math_changed=False))
        return result

    with patch.object(pipeline, 'load_kernels', kernels):
        return pipeline.build(operations, mesh, tensors, root=root)
