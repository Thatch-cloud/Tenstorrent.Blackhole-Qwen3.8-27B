"""Reuse the exact combined hardware binary in a device-free simulator container."""

import hashlib
import json
from pathlib import Path
import shutil


CACHE_KEY = 'b501e1084c8fdb949b203b8ede33bf0049f45686a80944ca8ceab532f3a02c90'
BINARY_SHA256 = '4b7299c1c9233b25aad310bc9a9d751a0631c6af0602cb1a151f934b4bfa07ea'
BINARIES = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
FACTORY = 'ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp'
ORIGINAL_FACTORY = 'a263559fe23cdf6fa8194604b238a939d299a356592eae1c7b2df11868383ebc'
COMBINED_FACTORY = 'fd8c067661a6ed5438bcbd31ee782fab2653fb7e8c00456a0fb43883a6a89783'
ANCHOR = '''    tt::DataFormat im_df =
        tt::DataFormat::Float16_b;  // Keep most intermediates in bf16 to save L1; opt-in fp32 per-CB below.
    tt::DataFormat stats_df = im_df;'''
REPLACEMENT = '''    const bool qwen_draft_fp32_intermediates =
        B == 1 && NQH == 16 && NKH == 4 && DHt == 4 && vDHt == 4 &&
        (Skt == 144 || Skt == 272 || Skt == 528 || Skt == 1040 || Skt == 2064 || Skt == 4112 || Skt == 8208) && Sq_chunk_t == 1 && Sk_chunk_t == 8 &&
        !is_causal && compute_use_provided_mask && !is_chunked &&
        !use_attention_sink && !is_windowed && !use_streaming_compute &&
        fp32_dest_acc_en && !exp_approx_mode;
    tt::DataFormat im_df = tt::DataFormat::Float16_b;
    tt::DataFormat stats_df = qwen_draft_fp32_intermediates ? tt::DataFormat::Float32 : im_df;'''


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def factory_bytes(source):
    if hashlib.sha256(source).hexdigest() != ORIGINAL_FACTORY or source.count(ANCHOR.encode()) != 1:
        raise ValueError('Exact original SDPA factory required')
    result = source.replace(ANCHOR.encode(), REPLACEMENT.encode())
    if hashlib.sha256(result).hexdigest() != COMBINED_FACTORY:
        raise ValueError('Reconstructed factory does not match the combined hardware runtime')
    return result


def binary_hashes(root):
    result = {name: digest(Path(root) / name) for name in BINARIES}
    if any(value != BINARY_SHA256 for value in result.values()):
        raise ValueError('Exact combined hardware runtime binary required')
    return result


def install(root, cache):
    root, cache = Path(root), Path(cache)
    manifest = json.loads((cache / 'manifest.json').read_text())
    key = hashlib.sha256(json.dumps(manifest['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (key != CACHE_KEY or manifest.get('binary_sha256') != BINARY_SHA256
            or digest(cache / '_ttnncpp.so') != BINARY_SHA256):
        raise ValueError('Pinned combined runtime cache entry required; no rebuild or fallback')
    changed = factory_bytes((root / FACTORY).read_bytes())
    (root / FACTORY).write_bytes(changed)
    for name in BINARIES:
        shutil.copy2(cache / '_ttnncpp.so', root / name)
    return dict(cache_key=key, factory_sha256=digest(root / FACTORY), binaries=binary_hashes(root))


if __name__ == '__main__':
    import os

    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or os.environ.get('QWEN_CARDS_ALLOCATED')
            or os.environ.get('QWEN_HARDWARE_TESTS')):
        raise ValueError('Device-free simulator container required')
    print(json.dumps(install('/opt/tt-metal', Path('/combined-native-cache/dspark-native-v1') / CACHE_KEY)))
