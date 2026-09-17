"""Keep draft-only SDPA helpers out of native target decode translation units."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
from unittest.mock import patch

import native_draft_sdpa


PREFIX = '#if defined(QWEN_SCORE_SCRATCH_CB) && defined(QWEN_SUM_SCRATCH_CB)\n'
SEPARATOR = '\n#else\n#define QWEN_NATIVE_ATTENTION_HEADER_BRANCH 1\n'
SUFFIX = '\n#endif\n'


def original_header(source):
    if source.startswith(PREFIX):
        if source.count(SEPARATOR) != 1 or not source.endswith(SUFFIX):
            raise ValueError('Complete unique draft/native header boundary required')
        source = source.split(SEPARATOR)[1][:-len(SUFFIX)]
    if hashlib.sha256(source.encode()).hexdigest() != native_draft_sdpa.SOURCE_HASHES['compute_common.hpp']:
        raise ValueError('Exact pinned native attention header required')
    return source


@contextmanager
def boundary_scope(root):
    path = Path(root) / native_draft_sdpa.KERNEL_DIRECTORY / 'compute_common.hpp'
    original = original_header(path.read_bytes().decode())
    previous = native_draft_sdpa.patched_sources

    def patched_sources(sources):
        result = previous(sources)
        result['compute_common.hpp'] = (PREFIX.encode() + result['compute_common.hpp']
            + SEPARATOR.encode() + original.encode() + SUFFIX.encode())
        return result

    def audit_active_kernel(runtime_root):
        directory = Path(runtime_root) / native_draft_sdpa.KERNEL_DIRECTORY
        observed = {name: (directory / name).read_bytes() for name in native_draft_sdpa.SOURCE_HASHES}
        header = observed['compute_common.hpp'].decode()
        if not header.startswith(PREFIX) or original_header(header) != original:
            raise ValueError('Active draft/native header boundary required')
        recovered = dict(observed)
        recovered['compute_common.hpp'] = header[len(PREFIX):].split(SEPARATOR)[0].encode()
        for name, substitutions in native_draft_sdpa.replacements().items():
            for before, after in reversed(substitutions):
                if recovered[name].count(after.encode()) != 1:
                    raise ValueError('Unique reversible active draft transformation required')
                recovered[name] = recovered[name].replace(after.encode(), before.encode())
        if patched_sources(recovered) != observed or not (directory / '.qwen-precise-draft.lock').is_file():
            raise ValueError('Exact owned draft/native attention header required')
        return dict(original=native_draft_sdpa.SOURCE_HASHES,
            patched={name: hashlib.sha256(source).hexdigest() for name, source in observed.items()},
            signature=native_draft_sdpa.SIGNATURE, scope=__doc__)

    with patch.object(native_draft_sdpa, 'patched_sources', patched_sources), \
            patch.object(native_draft_sdpa, 'audit_active_kernel', audit_active_kernel):
        yield
