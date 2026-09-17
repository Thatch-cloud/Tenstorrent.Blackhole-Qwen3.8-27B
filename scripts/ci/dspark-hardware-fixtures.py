"""Cache the pinned public checkpoint and config for an offline hardware container."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from dspark_checkpoint import fetch, verify
from dspark_intake import FILES, MODEL, REVISION, validate_config


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    checkpoint = directory/'model.safetensors'
    report = verify(checkpoint) if checkpoint.exists() else fetch(checkpoint)
    path = directory/'config.json'
    size,sha = FILES['config.json']
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/config.json',timeout=60) as response:
            data = response.read(size+1)
        if len(data)!=size or hashlib.sha256(data).hexdigest()!=sha:
            raise ValueError('Pinned DSpark config required')
        with path.open('xb') as stream:
            stream.write(data)
    if len(data)!=size or hashlib.sha256(data).hexdigest()!=sha:
        raise ValueError('Cached DSpark config changed')
    validate_config(json.loads(data))
    return dict(checkpoint=report,config_sha256=sha,remote_code_executed=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(parser.parse_args().output),indent=2))
