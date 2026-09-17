"""Bounded, hash-verified DSpark tensor loading without safetensors dependencies or checkpoint code."""

import hashlib
import json
import os
from pathlib import Path
import struct

from dspark_checkpoint import CHECKPOINT_SHA256, CHUNK_BYTES
from dspark_intake import CHECKPOINT_BYTES, HEADER_BYTES, HEADER_SHA256, validate_header
from dspark_markov_fixture import TENSORS


class VerifiedWeights:
    def __init__(self, path):
        self.path = Path(path)
        self.source = None
        if (self.path.stat().st_size != CHECKPOINT_BYTES
                or self.path.with_suffix(self.path.suffix + '.partial').exists()):
            raise ValueError('Complete pinned DSpark checkpoint required')
        self.source = self.path.open('rb', buffering=0)
        try:
            self.identity = self._identity(os.fstat(self.source.fileno()))
            prefix = self.source.read(8)
            if len(prefix) != 8 or struct.unpack('<Q', prefix)[0] != HEADER_BYTES:
                raise ValueError('Pinned safetensors header length required')
            encoded_header = self.source.read(HEADER_BYTES)
            if hashlib.sha256(encoded_header).hexdigest() != HEADER_SHA256:
                raise ValueError('Pinned safetensors header bytes required')
            self._header = json.loads(encoded_header)
            validate_header(self._header, HEADER_BYTES, CHECKPOINT_BYTES)
            overall = hashlib.sha256(prefix + encoded_header)
            self._tensor_hashes = {}
            entries = [(name, entry) for name, entry in self._header.items() if name != '__metadata__']
            for name, entry in sorted(entries, key=lambda item: item[1]['data_offsets'][0]):
                start, end = entry['data_offsets']
                if self.source.tell() != 8 + HEADER_BYTES + start:
                    raise ValueError('Checkpoint extent order differs from validated header')
                digest = hashlib.sha256()
                for offset in range(0, end - start, CHUNK_BYTES):
                    count = min(CHUNK_BYTES, end - start - offset)
                    data = self.source.read(count)
                    if len(data) != count:
                        raise ValueError('Truncated tensor payload')
                    overall.update(data)
                    digest.update(data)
                self._tensor_hashes[name] = digest.hexdigest()
            if self.source.tell() != CHECKPOINT_BYTES or overall.hexdigest() != CHECKPOINT_SHA256:
                raise ValueError('Checkpoint differs from its pinned whole-object hash')
            if any(self._tensor_hashes[name] != entry['sha256'] for name, entry in TENSORS.items()):
                raise ValueError('Markov ranges disagree with previously pinned matrices')
            self._assert_unchanged()
        except BaseException:
            self.source.close()
            self.source = None
            raise

    @staticmethod
    def _identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def _assert_unchanged(self):
        if (self.source is None or self._identity(os.fstat(self.source.fileno())) != self.identity
                or self._identity(self.path.stat()) != self.identity):
            raise ValueError('Verified checkpoint is closed, replaced or modified')

    def tensor(self, name):
        import torch

        self._assert_unchanged()
        if not isinstance(name, str) or name not in self._tensor_hashes:
            raise ValueError('A declared learned tensor name is required')
        entry = self._header[name]
        start, end = entry['data_offsets']
        self.source.seek(8 + HEADER_BYTES + start)
        storage = bytearray(end - start)
        if self.source.readinto(storage) != len(storage) or hashlib.sha256(storage).hexdigest() != self._tensor_hashes[name]:
            raise ValueError('Tensor bytes changed after whole-object verification')
        self._assert_unchanged()
        result = torch.frombuffer(storage, dtype=torch.bfloat16).reshape(entry['shape'])
        if not torch.isfinite(result).all():
            raise ValueError('Nonfinite learned tensor')
        return result

    def fingerprints(self):
        self._assert_unchanged()
        return self._tensor_hashes.copy()

    def __enter__(self):
        self._assert_unchanged()
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                self._assert_unchanged()
                self.source.seek(0)
                digest = hashlib.sha256()
                for data in iter(lambda: self.source.read(CHUNK_BYTES), b''):
                    digest.update(data)
                if self.source.tell() != CHECKPOINT_BYTES or digest.hexdigest() != CHECKPOINT_SHA256:
                    raise ValueError('Checkpoint content changed during learned execution')
                self._assert_unchanged()
        finally:
            if self.source is not None:
                self.source.close()
                self.source = None
