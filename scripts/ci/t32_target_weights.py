"""Read only complete target embedding/head tensors for the combined simulator proposal."""

import hashlib
import json
import os
from pathlib import Path

from dspark_inputs import VOCABULARY


HIDDEN_WIDTH = 5120


def identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class TargetWeights:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.paths = tuple(self.directory / name for name in ('config.json', 'model.safetensors.index.json'))
        self.identities = tuple(identity(path) for path in self.paths)
        contents = tuple(path.read_bytes() for path in self.paths)
        config, index = (json.loads(value) for value in contents)
        text = config.get('text_config', config)
        if text.get('hidden_size') != HIDDEN_WIDTH or text.get('vocab_size') != VOCABULARY:
            raise ValueError('Complete Qwen target vocabulary and hidden width required')
        mapping = index['weight_map']
        embeddings = [name for name in mapping if name.endswith('embed_tokens.weight')]
        heads = [name for name in mapping if name.endswith('lm_head.weight')]
        if len(embeddings) != 1 or len(heads) > 1:
            raise ValueError('Unambiguous target embedding and head required')
        if not heads and text.get('tie_word_embeddings', config.get('tie_word_embeddings')) is not True:
            raise ValueError('Missing untied target head')
        self.names = dict(embedding=embeddings[0], head=heads[0] if heads else embeddings[0])
        self.files = {}
        for name in set(self.names.values()):
            filename = mapping[name]
            if (not isinstance(filename, str) or not filename.endswith('.safetensors')
                    or '/' in filename or '\\' in filename or ':' in filename or Path(filename).name != filename):
                raise ValueError('Local safetensors shard basename required')
            self.files[name] = self.directory / filename
        self.shard_identities = {name: identity(path) for name, path in self.files.items()}
        self.manifest = dict(metadata={path.name: hashlib.sha256(data).hexdigest()
            for path, data in zip(self.paths, contents, strict=True)}, tensors={})
        self.check_unchanged()

    def check_unchanged(self):
        if (tuple(identity(path) for path in self.paths) != self.identities
                or any(identity(path) != self.shard_identities[name] for name, path in self.files.items())):
            raise ValueError('Target checkpoint changed during proposal loading')

    def tensor(self, role):
        import torch

        if role not in self.names:
            raise ValueError('Only target embedding and head may be loaded')
        self.check_unchanged()
        name = self.names[role]
        with self.files[name].open('rb') as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != self.shard_identities[name]:
                raise ValueError('Target shard replaced before opening')
            prefix = stream.read(8)
            header_size = int.from_bytes(prefix, 'little')
            if len(prefix) != 8 or not 2 <= header_size <= min(16 * 1024 * 1024, opened.st_size - 8):
                raise ValueError('Bounded complete safetensors header required')
            header = json.loads(stream.read(header_size))
            entry = header[name]
            start, end = entry['data_offsets']
            size = VOCABULARY * HIDDEN_WIDTH * 2
            if (entry['dtype'] != 'BF16' or entry['shape'] != [VOCABULARY, HIDDEN_WIDTH]
                    or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= opened.st_size - 8 - header_size or end - start != size):
                raise ValueError('Complete BF16 target matrix extent required')
            stream.seek(8 + header_size + start)
            storage = bytearray(size)
            if stream.readinto(storage) != size:
                raise ValueError('Truncated target matrix')
        self.check_unchanged()
        value = torch.frombuffer(storage, dtype=torch.bfloat16).reshape(VOCABULARY, HIDDEN_WIDTH)
        if not torch.isfinite(value).all():
            raise ValueError('Finite target matrix required')
        self.manifest['tensors'][role] = dict(name=name, shard=self.files[name].name,
            shape=list(value.shape), sha256=hashlib.sha256(storage).hexdigest())
        return value
