"""Audit the pinned binary-tree scratch bound before any native runtime experiment."""

import hashlib
from pathlib import Path


ROOT = Path('ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device')
HASHES = {
    'sdpa_decode_program_factory.cpp': '05708e6d9ddeddfdf13303d8f8fa391941d73b742ea3a380beeb0883ce8d4792',
    'kernels/dataflow/writer_decode_all.cpp': '734c90c01c7a7174497133fae9df80110ead55275955faeb566d345bdccb60b8',
    'kernels/compute/sdpa_flash_decode.cpp': 'd24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f',
}


def audit(root):
    found = {name: hashlib.sha256((Path(root) / ROOT / name).read_bytes()).hexdigest() for name in HASHES}
    if found != HASHES:
        raise ValueError('SDPA tree allocation or reduction source differs from the audited baseline')
    return found


def scratch_slots(workers):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('Pinned tree supports one to 64 workers per head')
    return (workers - 1).bit_length()
