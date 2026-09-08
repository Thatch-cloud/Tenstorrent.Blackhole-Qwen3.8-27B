import json
from pathlib import Path
import re
import subprocess


IMAGE_ID = 'sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465'
REPOSITORY = 'zot.thatch.local:5000/tt-vllm'
TAG = 'qwen38-k'


def verified_reference(document):
    if not isinstance(document, dict):
        raise ValueError('Expected one image manifest, not a multi-platform index')
    manifest = document.get('SchemaV2Manifest', {})
    descriptor = document.get('Descriptor', {})
    if manifest.get('config', {}).get('digest') != IMAGE_ID:
        raise ValueError('Registry config digest does not match the pinned runtime')
    digest = descriptor.get('digest', '')
    if not isinstance(digest, str) or not re.fullmatch(r'sha256:[a-f0-9]{64}', digest):
        raise ValueError('Missing or invalid immutable manifest digest')
    return f'{REPOSITORY}@{digest}'


def main():
    output = Path('experiment-results')
    output.mkdir(exist_ok=True)
    existing = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE_ID],
        capture_output=True, text=True, timeout=30)
    if existing.returncode == 0 and existing.stdout.strip() == IMAGE_ID:
        reference = IMAGE_ID
        restored = False
    else:
        result = subprocess.run(['docker', 'manifest', 'inspect', '--insecure', '--verbose',
            f'{REPOSITORY}:{TAG}'], capture_output=True, text=True, timeout=120)
        if result.returncode:
            print(result.stderr, flush=True)
            result.check_returncode()
        document = json.loads(result.stdout)
        reference = verified_reference(document)
        (output / 'runtime-registry-manifest.json').write_text(
            json.dumps(document, indent=2) + '\n', encoding='utf-8')
        subprocess.run(['docker', 'pull', reference], timeout=1800, check=True)
        actual = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', reference],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        if actual != IMAGE_ID:
            raise RuntimeError('Downloaded image ID mismatch; refusing runtime promotion')
        restored = True
    report = {'passed': True, 'image_id': IMAGE_ID, 'reference': reference,
        'restored': restored, 'cards_opened': False, 'serving_changed': False}
    (output / 'runtime-recovery.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
