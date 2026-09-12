import json
from pathlib import Path
import subprocess


IMAGE_ID = 'sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465'
REPOSITORY = 'zot.thatch.local:5000/tt-vllm'
REFERENCE = f'{REPOSITORY}@{IMAGE_ID}'


def main():
    output = Path('experiment-results')
    output.mkdir(exist_ok=True)
    existing = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE_ID],
        capture_output=True, text=True, timeout=30)
    if existing.returncode == 0 and existing.stdout.strip() == IMAGE_ID:
        reference = IMAGE_ID
        restored = False
    else:
        reference = REFERENCE
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
