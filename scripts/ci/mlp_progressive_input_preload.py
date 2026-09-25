"""Qualify progressive activation delivery before loading target weights."""

import json
import os
from pathlib import Path

from mlp_progressive_input_gate import qualify
from mlp_register_epilogue_gate import qualify as qualify_register


def preload(directory, runtime):
    directory = Path(directory)
    register = qualify_register(directory, directory / 'register-epilogue-evidence', runtime_root=runtime)
    return qualify(directory, directory / 'block-stream-evidence', directory / 'progressive-input-evidence', register)


if __name__ == '__main__':
    print(json.dumps(preload(Path(__file__).parent, os.environ['TT_METAL_HOME']), indent=2))
