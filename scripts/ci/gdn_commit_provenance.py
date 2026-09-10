"""Source identity for the commit-only simulator adapter and its publication kernels."""

import hashlib
from pathlib import Path


SOURCES = (
    '../../optimisation/sim/gdn-multitoken.py', 'gdn_commit_provenance.py',
    'gdn_multitoken.py', 'gdn_multitoken_conv.py', 'gdn_batched_conv.py', 'gdn_device_loop_state.py',
    'gdn_snapshot.py', 'gdn_prefix.py', 'gdn_records.py', 'gdn_state_copy.py', 'gdn_state_copy.cpp',
    'gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'gdn_conv_prefix_copy.py', 'gdn_conv_prefix_copy.cpp',
    'gdn_commit_dma.py', 'gdn_commit_dma.cpp',
)


def source_hashes(directory):
    return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() for name in SOURCES}
