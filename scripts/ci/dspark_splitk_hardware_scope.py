"""Disposable hardware scope for the simulator-qualified decode transformation."""

from contextlib import contextmanager
import hashlib
from pathlib import Path

from dspark_splitk_unfused_correction import HEADER, guard_candidate, transform


SOURCE_SHA256 = 'd24769bdcbb8635f83f5f91a301fe0d89298d38263d4493a39c6d2decb57867f'


@contextmanager
def kernel_scope(root):
    source = Path(root) / HEADER
    original = source.read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_SHA256:
        raise ValueError('Exact pinned decode kernel required')
    candidate = guard_candidate(original.decode(), transform(original.decode())).encode()
    lock = source.with_suffix('.splitk-hardware.lock')
    with lock.open('x'):
        pass
    try:
        source.write_bytes(candidate)
        yield dict(source_before=SOURCE_SHA256,
            source_active=hashlib.sha256(candidate).hexdigest())
    finally:
        try:
            source.write_bytes(original)
        finally:
            lock.unlink()
