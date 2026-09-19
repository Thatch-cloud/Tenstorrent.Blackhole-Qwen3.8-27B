"""Check physical runtime/source admission before expensive model loading."""

import json
import os
from pathlib import Path

from mlp_block_stream_gate import qualify
from mlp_block_stream_runtime import require_hardware
from mlp_register_epilogue_gate import qualify as qualify_register


def main():
    require_hardware(os.environ)
    directory = Path(__file__).parent
    register = qualify_register(directory, directory / 'register-epilogue-evidence',
        runtime_root=os.environ['TT_METAL_HOME'])
    print(json.dumps(qualify(directory, directory / 'block-stream-evidence', register), indent=2))


if __name__ == '__main__':
    main()
