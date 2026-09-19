"""Explicit simulator-only overlay on the admitted serial bulk-reader experiment."""

import argparse
import hashlib
import json
from pathlib import Path

from mlp_block_stream_pipeline import transform


OVERLAY = '''

from mlp_block_stream_pipeline import transform as pipeline_reader

serial_reader_source = reader_source


def reader_source(original):
    return pipeline_reader(serial_reader_source(original))
'''


def stage(checkout, manifest):
    scripts, manifest = Path(checkout) / 'scripts/ci', Path(manifest)
    source = scripts / 'mlp_block_stream.py'
    original = source.read_text()
    if manifest.exists() or 'pipeline_reader' in original:
        raise ValueError('Fresh serial block-stream staging required')
    namespace = {'__name__': 'serial_block_stream'}
    exec(compile(original, str(source), 'exec'), namespace)
    serial = namespace['reader_source']((scripts / 'fused_1d_weights.cpp').read_text())
    candidate = transform(serial)
    helper = Path(__file__).with_name('mlp_block_stream_pipeline.py').read_bytes()
    (scripts / 'mlp_block_stream_pipeline.py').write_bytes(helper)
    source.write_bytes((original + OVERLAY).encode())
    manifest.write_text(json.dumps(dict(simulator_qualified=False, hardware_qualified=False,
        performance_qualified=False, arithmetic_changed=False, extra_buffer_bytes=0,
        before=hashlib.sha256(original.encode()).hexdigest(),
        after=hashlib.sha256(source.read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(helper).hexdigest(),
        serial_reader_sha256=hashlib.sha256(serial.encode()).hexdigest(),
        candidate_reader_sha256=hashlib.sha256(candidate.encode()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    options = parser.parse_args()
    stage(options.checkout, options.manifest)
